from __future__ import annotations

import dataclasses
import json
import struct
import sys
from pathlib import Path
from typing import Mapping

import pytest

from ruleforge.grpo import merging as merge_module
from ruleforge.judges import weights as weights_module
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate.aggregate import POLICY_FIELDS, analyze_campaign, policy_families, served_weights
from ruleforge.evaluate.campaign import build_plan, condition_spec_for, load_conditions, preflight, run_plan
from ruleforge.evaluate.identity import ConditionSpec
from ruleforge.judges.weights import (
    ModelWeightsIdentity,
    model_weights_identity,
    recorded_model_weights,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.identity import (
    DRAW_PROVENANCE_FNAME,
    RUNS_FNAME,
)
from ruleforge.evaluate.runner import (
    RESOLVED_ENGINE_KEY,
    STATE_COMPLETE,
    CampaignRunError,
    DrawRunner,
    served_model_weights,
)
from tests.support.hydra_runs import run_program
from tests.test_sampling_campaign import backend_for, compose_campaign, set_recorded_runtime
from tests.test_sampling_runner import RULES, FakeBackend, forced, make_spec, read_jsonl, write_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]

QWEN3_GENERATION_CONFIG = (
    '{\n    "bos_token_id": 151643,\n    "do_sample": true,\n    "eos_token_id": [\n        151645,\n'
    '        151643\n    ],\n    "pad_token_id": 151643,\n    "temperature": 0.6,\n    "top_k": 20,\n'
    '    "top_p": 0.95,\n    "transformers_version": "4.51.0"\n}'
)
#: The sampling truncation every existing Qwen3-8B campaign records (the decoder).
QWEN3_TRUNCATION = {"min_p": 0.0, "repetition_penalty": 1.0, "top_k": 20, "top_p": 0.95}
FP32_ENGINE = {"engine_dtype": "float32", "kv_cache_dtype": "float32", "attention_backend": "TRITON_ATTN"}


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(campaign_module, "code_state", lambda: {})
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)


@pytest.fixture
def inputs(tmp_path):
    path = write_inputs(tmp_path)
    return ScenarioManifest.build(path), [json.loads(line) for line in path.read_text().splitlines()]


def write_safetensors(path: Path, tensors: Mapping[str, tuple[str, list[int], bytes]]) -> Path:
    """A safetensors file: an 8-byte length, the JSON header, then each tensor's bytes."""
    header, data = {}, b""
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(payload)]}
        data += payload
    blob = json.dumps({"__metadata__": {"format": "pt"}, **header}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)
    return path


def tiny_tensors(seed: float = 0.0) -> dict[str, tuple[str, list[int], bytes]]:
    """Three fp32 tensors of a stand-in model; ``seed`` changes one value in two of them."""
    return {
        "lm_head.weight": ("F32", [2, 2], struct.pack("<4f", seed, 1.0, 2.0, 3.0)),
        "model.embed_tokens.weight": ("F32", [2, 2], struct.pack("<4f", 4.0, 5.0, 6.0, seed + 0.5)),
        "model.norm.weight": ("F32", [2], struct.pack("<2f", 1.0, 1.0)),
    }


def write_weights(directory: Path, *, seed: float = 0.0, shards: int = 1) -> Path:
    tensors = tiny_tensors(seed)
    names = sorted(tensors)
    for index in range(shards):
        part = {name: tensors[name] for name in names[index::shards]}
        write_safetensors(directory / f"model-{index + 1:05d}-of-{shards:05d}.safetensors", part)
    (directory / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}', encoding="utf-8")
    (directory / "generation_config.json").write_text(QWEN3_GENERATION_CONFIG, encoding="utf-8")
    return directory


def write_base(models_root: Path, name: str = "Qwen3-8B") -> Path:
    """The upstream directory ``name`` (Qwen3-8B's) under a model root: weights, no derivation manifest."""
    return write_weights(Path(models_root) / name, seed=-1.0)


def write_merged(root: Path, name: str = "checkpoint-125", *, seed: float = 0.0, shards: int = 1,
                 saved_dtype: str = "float32", manifest: Mapping | None = None) -> Path:
    """A merged checkpoint as ``ruleforge.grpo.merging`` leaves it."""
    directory = write_weights(Path(root) / name, seed=seed, shards=shards)
    payload = {
        "schema_version": 1, "kind": "lora_merge", "weights_digest": merge_module.WEIGHTS_DIGEST_VERSION,
        "merge": {"saved_dtype": saved_dtype},
        "result": {
            "weights_sha256": merge_module.weights_digest(directory),
            "config_sha256": sha256_file(directory / "config.json"),
            "copied_files": {"generation_config.json": sha256_file(directory / "generation_config.json")},
        },
        **(manifest or {}),
    }
    (directory / merge_module.MANIFEST_FNAME).write_text(json.dumps(payload), encoding="utf-8")
    return directory


def tamper(directory: Path) -> None:
    """Change one stored value of the first shard, header untouched: the same layout, other weights."""
    shard = sorted(directory.glob("*.safetensors"))[0]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0x01
    shard.write_bytes(bytes(data))


class TestModelWeightsIdentity:
    def test_upstream_weights_have_no_content_identity(self, tmp_path):
        assert model_weights_identity(write_base(tmp_path / "models")) is None
        assert model_weights_identity(tmp_path / "absent") is None  # the engine would fail, not the identity

    def test_a_merged_checkpoint_is_named_by_the_digest_its_manifest_records(self, tmp_path):
        merged = write_merged(tmp_path / "run")
        manifest = json.loads((merged / merge_module.MANIFEST_FNAME).read_text())
        identity = model_weights_identity(merged)
        assert identity == ModelWeightsIdentity(
            path=str(merged.resolve()),
            sha256=manifest["result"]["weights_sha256"],
            digest_version="smrc-weights-sha256-v1",
            manifest_sha256=sha256_file(merged / merge_module.MANIFEST_FNAME),
            kind="lora_merge",
            saved_dtype="float32",
        )
        assert identity.sha256 == merge_module.weights_digest(merged)
        (tmp_path / "alias").symlink_to(merged)
        assert model_weights_identity(tmp_path / "alias") == identity

    def test_the_digest_ignores_the_shard_layout(self, tmp_path):
        one = model_weights_identity(write_merged(tmp_path / "a", shards=1))
        three = model_weights_identity(write_merged(tmp_path / "b", shards=3))
        assert one.sha256 == three.sha256 and one.path != three.path

    def test_two_merges_under_one_name_with_other_weights_are_told_apart(self, tmp_path):
        first = model_weights_identity(write_merged(tmp_path / "run_a", seed=0.0))
        second = model_weights_identity(write_merged(tmp_path / "run_b", seed=1.0))
        assert Path(first.path).name == Path(second.path).name == "checkpoint-125"
        assert first.sha256 != second.sha256

    def test_a_tampered_shard_is_refused(self, tmp_path):
        merged = write_merged(tmp_path / "run", shards=2)
        tamper(merged)
        with pytest.raises(SamplingDesignError, match=r"have digest [0-9a-f]{64}, not the [0-9a-f]{64} its "
                                                      r"merge_manifest.json records: a shard was changed"):
            model_weights_identity(merged)

    def test_a_shard_cut_short_or_missing_is_refused(self, tmp_path):
        merged = write_merged(tmp_path / "run")
        shard = next(merged.glob("*.safetensors"))
        shard.write_bytes(shard.read_bytes()[:-3])
        with pytest.raises(SamplingDesignError, match="cannot digest the weights .* ends inside tensor"):
            model_weights_identity(merged)
        shard.write_bytes(b"abc")
        with pytest.raises(SamplingDesignError, match="cannot digest the weights .* not a safetensors file"):
            model_weights_identity(merged)
        shard.unlink()
        with pytest.raises(SamplingDesignError, match="cannot digest the weights .* no .safetensors file"):
            model_weights_identity(merged)

    def test_a_manifest_from_another_merge_is_refused(self, tmp_path):
        merged = write_merged(tmp_path / "run", manifest={"result": {"weights_sha256": "0" * 64}})
        with pytest.raises(SamplingDesignError, match=f"not the {'0' * 64} its merge_manifest.json records"):
            model_weights_identity(merged)

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("{not json", "cannot read"),
            ("[]", "not a derivation manifest"),
            ('{"result": []}', "not a derivation manifest"),
            ('{"weights_digest": "smrc-weights-sha256-v0", "result": {"weights_sha256": "%s"}}' % ("a" * 64),
             "records no smrc-weights-sha256-v1 weights digest"),
            ('{"weights_digest": "smrc-weights-sha256-v1", "result": {"weights_sha256": "XYZ"}}',
             "records no smrc-weights-sha256-v1 weights digest"),
            ('{"weights_digest": "smrc-weights-sha256-v1", "result": {}}', "result.weights_sha256 None"),
        ],
    )
    def test_a_manifest_this_build_cannot_read_is_refused(self, tmp_path, text, message):
        merged = write_merged(tmp_path / "run")
        (merged / merge_module.MANIFEST_FNAME).write_text(text, encoding="utf-8")
        with pytest.raises(SamplingDesignError, match=message):
            model_weights_identity(merged)

    @pytest.mark.parametrize("name", ["config.json", "generation_config.json"])
    def test_a_config_or_copied_file_other_than_the_recorded_one_is_refused(self, tmp_path, name):
        merged = write_merged(tmp_path / "run")
        (merged / name).write_text((merged / name).read_text() + "\n", encoding="utf-8")
        with pytest.raises(SamplingDesignError, match=rf"\['{name}'\] are not the files its merge_manifest.json "
                                                      "records"):
            model_weights_identity(merged)
        (merged / name).unlink()
        with pytest.raises(SamplingDesignError, match=rf"\['{name}'\] are not the files"):
            model_weights_identity(merged)

    def test_the_copied_files_must_be_a_mapping(self, tmp_path):
        merged = write_merged(tmp_path / "run")
        path = merged / merge_module.MANIFEST_FNAME
        payload = json.loads(path.read_text())
        payload["result"]["copied_files"] = ["generation_config.json"]
        path.write_text(json.dumps(payload))
        with pytest.raises(SamplingDesignError, match="result.copied_files must map file names to SHA-256"):
            model_weights_identity(merged)

    def test_a_derivation_manifest_need_not_say_how_it_was_made(self, tmp_path):
        merged = write_merged(tmp_path / "run")
        path = merged / merge_module.MANIFEST_FNAME
        payload = json.loads(path.read_text())
        del payload["kind"], payload["merge"], payload["result"]["config_sha256"], payload["result"]["copied_files"]
        path.write_text(json.dumps(payload))
        (merged / "config.json").unlink()  # nothing recorded, nothing checked: the weights digest is the identity
        identity = model_weights_identity(merged)
        assert (identity.kind, identity.saved_dtype) == (None, None)

    @pytest.mark.parametrize("added", ["chat_template.jinja", "tokenizer_config.json", "additional_chat_templates"])
    def test_a_serving_file_added_after_the_merge_is_refused(self, tmp_path, added):
        merged = write_merged(tmp_path / "run")
        identity = model_weights_identity(merged)
        if added == "additional_chat_templates":
            (merged / added).mkdir()
            (merged / added / "x.jinja").write_text("{{ messages }}", encoding="utf-8")
        else:
            (merged / added).write_text("{}", encoding="utf-8")
        with pytest.raises(SamplingDesignError, match=rf"\['{added}'\] are not among the files its "
                                                      "merge_manifest.json records \\(added after the merge\\)"):
            model_weights_identity(merged)
        path = merged / merge_module.MANIFEST_FNAME
        payload = json.loads(path.read_text())
        if added != "additional_chat_templates":
            payload["result"]["copied_files"][added] = sha256_file(merged / added)
            path.write_text(json.dumps(payload))
            assert model_weights_identity(merged).sha256 == identity.sha256

    def test_the_recorded_weights_are_read_without_the_digest(self, tmp_path):
        merged = write_merged(tmp_path / "run")
        checked = model_weights_identity(merged)
        tamper(merged)
        assert recorded_model_weights(merged) == checked
        assert recorded_model_weights(write_base(tmp_path / "models")) is None
        (merged / merge_module.MANIFEST_FNAME).write_text("[]", encoding="utf-8")
        with pytest.raises(SamplingDesignError, match="not a derivation manifest"):
            recorded_model_weights(merged)


def qwen_spec(**changes) -> ConditionSpec:
    """tests/test_sampling_training.py's Qwen3-8B family base condition, field for field."""
    fields = dict(
        condition_id="Qwen3-8B/figure5/base",
        model_name="Qwen3-8B", model_type="local", prompt_style="forced-grpo",
        individual_rule_context="isolated", judge_type="No_Interpretation", evaluation_mode="both",
        rule_set_name="openai_rules_all.jsonl", rule_set_sha256="a" * 64, temperature=1.0,
        prompt_shape_version=2, forced_json_requirement="label_only", label_rule="lenient",
        max_new_tokens=32768, seed_capability="local_deterministic", model_path="Qwen3-8B",
        enable_thinking=True,
        sampling_truncation={"top_p": 0.95, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
        extra=(("figure", "figure5"), ("policy", "base")),
    )
    fields.update(changes)
    return ConditionSpec(**fields)


#: qwen_spec()'s fingerprint without adapter digests, weights digests or serving dtypes.
PRE_IDENTITY_FINGERPRINT = "c91c84a9b0692cdbdc314e4b732f01cbe5ee021b273226b3f68480a45ba3a793"


class TestConditionIdentity:
    def test_a_condition_with_neither_field_keeps_its_fingerprint(self):
        spec = qwen_spec()
        assert spec.fingerprint == PRE_IDENTITY_FINGERPRINT
        payload = spec.to_dict()
        assert "model_weights_sha256" not in payload and "serving_dtype" not in payload
        assert ConditionSpec.from_dict(payload) == spec

    def test_the_weights_digest_is_identity(self):
        merged = qwen_spec(model_path="checkpoint-125", model_weights_sha256="1" * 64)
        payload = merged.to_dict()
        assert payload["model_weights_sha256"] == "1" * 64 and ConditionSpec.from_dict(payload) == merged
        other = dataclasses.replace(merged, model_weights_sha256="2" * 64)
        assert len({qwen_spec().fingerprint, merged.fingerprint, other.fingerprint}) == 3

    def test_the_serving_dtype_is_identity_and_only_it_sets_fp32_apart(self):
        bf16, fp32 = qwen_spec(), qwen_spec(serving_dtype="float32")
        declared_bf16 = qwen_spec(serving_dtype="bfloat16")
        assert len({bf16.fingerprint, fp32.fingerprint, declared_bf16.fingerprint}) == 3
        assert {key: value for key, value in fp32.to_dict().items() if bf16.to_dict().get(key) != value} == {
            "serving_dtype": "float32"
        }
        assert ConditionSpec.from_dict(fp32.to_dict()) == fp32

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"model_weights_sha256": "XYZ"}, "model_weights_sha256 must be a SHA-256 hex digest"),
            ({"model_weights_sha256": 7}, "model_weights_sha256 must be a SHA-256 hex digest"),
            ({"model_weights_sha256": "1" * 64, "model_path": None}, "weights digest but no model_path"),
            ({"serving_dtype": "float"}, r"serving_dtype must be one of \('float32', 'bfloat16', 'float16'\)"),
            ({"serving_dtype": "auto"}, "serving_dtype must be one of"),
        ],
    )
    def test_malformed_values_are_refused(self, changes, message):
        with pytest.raises(SamplingDesignError, match=message):
            qwen_spec(**changes)


def weights_campaign(tmp_path, model_path: Path | None = None, *, dtype: str | None = None,
                     approved: str | None = None, **overrides):
    """A campaign config on the default local judge (TRITON_ATTN), serving ``model_path`` in ``dtype``."""
    settings = dict(overrides)
    if model_path is not None:
        settings["judge_model.model_path"] = str(model_path)
    if dtype is not None:
        settings["+judge_model.dtype"] = dtype
    if approved is not None:
        settings["approved_model_weights_sha256"] = approved
    return compose_campaign(tmp_path, **settings)


def spec_of(cfg, modelspec, backend=None) -> ConditionSpec:
    return condition_spec_for(
        condition_id="qwen/forced-grpo", backend=backend or backend_for("forced-grpo"),
        task_spec=modelspec(reasoning_prompt_style="forced-grpo", evaluation_temperature=1.0),
        config=preflight(cfg)[0], judge_type="No_Interpretation", rule_set_path=Path(str(cfg.rule_set_path)),
        model_cfg=cfg.judge_model,
    )


class TestConditionSpecFor:
    def test_merged_weights_enter_by_content_and_the_serving_dtype_as_declared(self, tmp_path, modelspec):
        merged = write_merged(tmp_path / "run")
        spec = spec_of(weights_campaign(tmp_path, merged, dtype="FLOAT"), modelspec)
        assert (spec.model_path, spec.model_weights_sha256) == ("checkpoint-125", merge_module.weights_digest(merged))
        assert spec.serving_dtype == "float32"  # one spelling of vLLM's float32
        assert spec.to_dict()["sampling_truncation"] == QWEN3_TRUNCATION  # the base's, copied verbatim

    def test_upstream_weights_keep_the_basename_identity(self, tmp_path, modelspec):
        base = write_base(tmp_path / "models")
        spec = spec_of(weights_campaign(tmp_path, base), modelspec)
        assert (spec.model_path, spec.model_weights_sha256, spec.serving_dtype) == ("Qwen3-8B", None, None)
        assert not {"model_weights_sha256", "serving_dtype"} & set(spec.to_dict())

    def test_fp32_and_bf16_conditions_of_one_model_differ_in_the_serving_dtype_alone(self, tmp_path, modelspec):
        base = write_base(tmp_path / "models")
        bf16 = spec_of(weights_campaign(tmp_path, base), modelspec).to_dict()
        fp32 = spec_of(weights_campaign(tmp_path, base, dtype="float32"), modelspec).to_dict()
        assert {key: fp32[key] for key in set(fp32) ^ set(bf16)} == {"serving_dtype": "float32"}
        assert {key for key in set(fp32) & set(bf16) if fp32[key] != bf16[key]} == set()

    def test_two_merges_under_one_directory_name_are_two_conditions(self, tmp_path, modelspec):
        specs = [
            spec_of(weights_campaign(tmp_path, write_merged(tmp_path / run, seed=seed), dtype="float32"), modelspec)
            for run, seed in (("run_a", 0.0), ("run_b", 1.0))
        ]
        assert specs[0].model_path == specs[1].model_path == "checkpoint-125"
        assert specs[0].fingerprint != specs[1].fingerprint

    def test_a_provider_judge_records_neither(self, tmp_path, modelspec):
        cfg = compose_campaign(tmp_path, **{"model@judge_model": "gpt-5.6-luna-batch"})
        backend = backend_for("forced-grpo", model_type="openai_batch", seed_capability="unsupported")
        spec = spec_of(cfg, modelspec, backend)
        assert (spec.model_weights_sha256, spec.serving_dtype) == (None, None)


class TestPreflight:
    def test_an_approved_merged_checkpoint_passes(self, tmp_path):
        merged = write_merged(tmp_path / "run")
        preflight(weights_campaign(tmp_path, merged, dtype="float32", approved=merge_module.weights_digest(merged)))
        preflight(weights_campaign(tmp_path, merged, dtype="float32"))

    def test_another_digest_than_the_approved_one_is_refused(self, tmp_path):
        merged = write_merged(tmp_path / "run")
        with pytest.raises(CampaignRunError, match=r"model weights SHA-256 [0-9a-f]{64} \(.*checkpoint-125\) does "
                                                   "not match approved 0000"):
            preflight(weights_campaign(tmp_path, merged, dtype="float32", approved="0" * 64))

    def test_a_tampered_checkpoint_is_refused_whatever_the_approval(self, tmp_path):
        merged = write_merged(tmp_path / "run")
        approved = merge_module.weights_digest(merged)
        tamper(merged)
        for cfg in (weights_campaign(tmp_path, merged, dtype="float32", approved=approved),
                    weights_campaign(tmp_path, merged, dtype="float32")):
            with pytest.raises(SamplingDesignError, match="a shard was changed or replaced"):
                preflight(cfg)

    def test_an_approval_for_upstream_weights_is_refused(self, tmp_path):
        base = write_base(tmp_path / "models")
        with pytest.raises(CampaignRunError, match="carries no derivation manifest: its weights are named by "
                                                   "directory name"):
            preflight(weights_campaign(tmp_path, base, approved="1" * 64))

    def test_an_fp32_merge_needs_a_declared_serving_dtype(self, tmp_path):
        fp32 = write_merged(tmp_path / "run")
        with pytest.raises(CampaignRunError, match="holds float32 weights and judge_model .* declares no dtype"):
            preflight(weights_campaign(tmp_path, fp32))
        # the bf16 fallback's merge is served in bf16 under auto, as the base is
        preflight(weights_campaign(tmp_path, write_merged(tmp_path / "fallback", saved_dtype="bfloat16")))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"+judge_model.dtype": "auto"}, r"dtype 'auto' is not one of .*leave it unset for vLLM's auto"),
            ({"+judge_model.dtype": "fp8"}, "dtype 'fp8' is not one of"),
            ({"+judge_model.dtype": "float32", "judge_model.attention_backend": "FLASH_ATTN"},
             "dtype float32 with attention_backend FLASH_ATTN: that backend serves only fp16 and bf16"),
            ({"+judge_model.dtype": "float32", "judge_model.attention_backend": "FLASHINFER"},
             "dtype float32 with attention_backend FLASHINFER"),
        ],
    )
    def test_a_serving_dtype_the_engine_could_not_serve_is_refused(self, tmp_path, overrides, message):
        with pytest.raises(CampaignRunError, match=r"judge_model gemma-4-31B-it: " + message):
            preflight(compose_campaign(tmp_path, **overrides))

    @pytest.mark.parametrize("key", ["+judge_model.dtype=float32", "approved_model_weights_sha256=" + "1" * 64])
    def test_a_provider_judge_takes_neither(self, tmp_path, key):
        name, value = key.split("=", 1)
        cfg = compose_campaign(tmp_path, **{"model@judge_model": "gpt-5.6-luna-batch", name: value})
        with pytest.raises(CampaignRunError, match="only a local judge serves weights this repo names or a dtype"):
            preflight(cfg)

    def test_the_preflight_script_reports_the_weights_and_the_serving_dtype(self, tmp_path, capsys, monkeypatch):
        merged = write_merged(tmp_path / "run")
        digests = []
        real_digest = weights_module.weights_digest
        # counted where model_weights_identity reads it, beside it in ruleforge.judges.weights
        monkeypatch.setattr(weights_module, "weights_digest", lambda path: digests.append(path) or real_digest(path))
        scenarios = write_inputs(tmp_path)
        rules = tmp_path / "rules.jsonl"
        rules.write_text("".join(json.dumps(r) + "\n" for r in RULES), encoding="utf-8")
        script = "ruleforge.launch.programs.preflight_campaign"
        common = [str(script), "campaign_id=camp", f"campaign_root={tmp_path / 'root'}",
                  f"scenario_set_path={scenarios}", "approved_scenario_sha256=null", f"rule_set_path={rules}",
                  "sampling.campaign_mode=pilot", f"hydra.run.dir={tmp_path / 'hydra'}"]
        for extra, weights, dtype in (
            ([f"judge_model.model_path={merged}", "+judge_model.dtype=float32"], {
                "path": str(merged.resolve()), "sha256": merge_module.weights_digest(merged), "kind": "lora_merge",
                "saved_dtype": "float32"}, "float32"),
            ([], None, None),
            (["model@judge_model=gpt-5.6-luna-batch"], None, None),
        ):
            monkeypatch.setattr(sys, "argv", [*common, *extra])
            digests.clear()
            run_program(script)
            report = json.loads(capsys.readouterr().out)
            assert (report["model_weights"], report["serving_dtype"]) == (weights, dtype)
            assert len(digests) == (weights is not None)


def local_backend(*, model_path: Path | None = None, dtype: str | None = None, resolved=None, **kwargs) -> FakeBackend:
    """A local judge as ModelBackend leaves it after engine start."""
    backend = FakeBackend(**kwargs)
    backend.model_path = None if model_path is None else str(model_path)
    backend.engine_args = {"model": backend.model_path, **({"dtype": dtype} if dtype is not None else {})}
    backend.engine_resolved = resolved
    return backend


def weights_runner(tmp_path, inputs, modelspec, backend, *, weights_sha256=None, serving_dtype=None, draw=0,
                   model_path="checkpoint-125") -> DrawRunner:
    manifest, rows = inputs
    spec = make_spec(manifest, draw=draw)
    condition = dataclasses.replace(
        spec.condition, model_path=model_path, model_weights_sha256=weights_sha256, serving_dtype=serving_dtype,
    )
    return DrawRunner(
        backend=backend, task_spec=modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0),
        spec=dataclasses.replace(spec, condition=condition), draw_dir=tmp_path / "draws" / f"draw-{draw:03d}",
        scenario_rows=rows, rules=RULES,
    )


class TestRunner:
    def test_served_merged_weights_are_checked_once_per_engine_and_recorded(self, tmp_path, inputs, modelspec):
        merged = write_merged(tmp_path / "run")
        digest = merge_module.weights_digest(merged)
        backend = local_backend(model_path=merged, dtype="float32", resolved=FP32_ENGINE)
        runner = weights_runner(tmp_path, inputs, modelspec, backend, weights_sha256=digest, serving_dtype="float32")
        assert runner.run()["state"] == STATE_COMPLETE
        provenance = json.loads((runner.draw_dir / DRAW_PROVENANCE_FNAME).read_text())
        assert provenance["model_weights"] == {
            "path": str(merged.resolve()), "sha256": digest, "digest_version": "smrc-weights-sha256-v1",
            "manifest_sha256": sha256_file(merged / merge_module.MANIFEST_FNAME), "kind": "lora_merge",
            "saved_dtype": "float32",
        }
        runtime = read_jsonl(runner.draw_dir / RUNS_FNAME)[0]["runtime"]
        assert runtime[RESOLVED_ENGINE_KEY] == FP32_ENGINE and runtime["engine_args"]["dtype"] == "float32"
        tamper(merged)
        assert served_model_weights(backend).sha256 == digest
        weights_runner(tmp_path, inputs, modelspec, backend, weights_sha256=digest, serving_dtype="float32", draw=1)
        # ... and the next engine start digests the files again
        restarted = local_backend(model_path=merged, dtype="float32", resolved=FP32_ENGINE)
        with pytest.raises(SamplingDesignError, match="a shard was changed or replaced"):
            weights_runner(tmp_path, inputs, modelspec, restarted, weights_sha256=digest, serving_dtype="float32")

    def test_other_weights_or_weights_on_one_side_only_are_refused_before_generation(self, tmp_path, inputs, modelspec):
        named = merge_module.weights_digest(write_merged(tmp_path / "run_a", seed=0.0))
        other = local_backend(model_path=write_merged(tmp_path / "run_b", seed=1.0))
        with pytest.raises(CampaignRunError, match=rf"names model weights 'checkpoint-125' \(weights SHA-256 {named}\) "
                                                   r"but the backend serves .*run_b/checkpoint-125 \(weights SHA-256"):
            weights_runner(tmp_path, inputs, modelspec, other, weights_sha256=named)
        # a condition naming derived weights the engine does not serve
        upstream = local_backend(model_path=write_base(tmp_path / "models"))
        with pytest.raises(CampaignRunError, match="but the backend serves no derived weights"):
            weights_runner(tmp_path, inputs, modelspec, upstream, weights_sha256=named)
        with pytest.raises(CampaignRunError, match="but the backend serves no derived weights"):
            weights_runner(tmp_path, inputs, modelspec, FakeBackend(), weights_sha256=named)
        # an engine serving derived weights the condition does not name by content
        with pytest.raises(CampaignRunError, match=r"names model weights 'checkpoint-125' \(weights SHA-256 None\)"):
            weights_runner(tmp_path, inputs, modelspec, other)
        assert other.blocks == [] and upstream.blocks == []
        with pytest.raises(CampaignRunError, match=r"names model weights 'checkpoint-125' \(weights SHA-256 None\) but "
                                                   r"the backend serves the upstream weights directory .*/Qwen3-8B;"):
            weights_runner(tmp_path, inputs, modelspec, upstream)
        runner = weights_runner(tmp_path, inputs, modelspec, upstream, model_path="Qwen3-8B")
        assert runner.model_weights_provenance() is None and "model_weights" not in runner.provenance()

    @pytest.mark.parametrize(
        ("resolved", "found"),
        [
            # the auto downcast: an fp32 checkpoint served in bf16, with no error from vLLM
            ({"engine_dtype": "bfloat16", "kv_cache_dtype": "bfloat16"}, "bfloat16 for the model and bfloat16"),
            ({"engine_dtype": "float32", "kv_cache_dtype": "fp8_e4m3"}, "float32 for the model and fp8_e4m3"),
            (None, "None for the model and None"),
        ],
    )
    def test_an_engine_that_resolved_another_dtype_than_declared_is_refused(self, tmp_path, inputs, modelspec,
                                                                             resolved, found):
        backend = local_backend(dtype="float32", resolved=resolved)
        with pytest.raises(CampaignRunError, match=f"declares serving dtype float32, but the engine resolved {found} "
                                                   "for its KV cache"):
            weights_runner(tmp_path, inputs, modelspec, backend, serving_dtype="float32", model_path=None)
        assert backend.blocks == []

    @pytest.mark.parametrize(
        ("declared", "requested", "message"),
        [
            ("float32", None, "declares serving dtype float32 but the backend's engine was built for auto"),
            (None, "float32", "declares serving dtype auto but the backend's engine was built for float32"),
            ("float32", "bfloat16", "declares serving dtype float32 but the backend's engine was built for bfloat16"),
        ],
    )
    def test_the_engine_must_be_built_for_the_declared_dtype(self, tmp_path, inputs, modelspec, declared, requested,
                                                              message):
        backend = local_backend(dtype=requested, resolved=FP32_ENGINE)
        with pytest.raises(CampaignRunError, match=message):
            weights_runner(tmp_path, inputs, modelspec, backend, serving_dtype=declared, model_path=None)

    def test_a_declared_dtype_resolved_as_declared_runs(self, tmp_path, inputs, modelspec):
        backend = local_backend(dtype="torch.float32", resolved=FP32_ENGINE)  # vLLM's spellings are one dtype
        runner = weights_runner(tmp_path, inputs, modelspec, backend, serving_dtype="float32", model_path=None)
        assert runner.run()["state"] == STATE_COMPLETE

    def test_a_provider_or_an_engine_without_a_weights_directory_serves_no_derived_weights(self, tmp_path):
        merged = write_merged(tmp_path / "run")
        provider = FakeBackend(model_type="openai_batch", seed_capability="unsupported")
        provider.model_path = str(merged)
        assert served_model_weights(provider) is None
        assert served_model_weights(FakeBackend()) is None
        assert served_model_weights(local_backend(model_path=merged)).sha256 == merge_module.weights_digest(merged)


POLICIES = ("base", "ckpt125", "ckpt200")
FAMILY = "+".join(POLICIES)


def merged_policy_spec(policy: str, digest: str | None, **changes) -> ConditionSpec:
    extra = {"figure": "figure5", "policy": policy, "policy_family": FAMILY, "policy_reference": "base"}
    weights = {} if digest is None else {"model_path": f"{policy}-merged", "model_weights_sha256": digest}
    return qwen_spec(condition_id=f"Qwen3-8B/figure5/{policy}", extra=tuple(extra.items()),
                     **{"serving_dtype": "float32", **weights, **changes})


class TestPolicyFamilies:
    def test_merged_checkpoints_and_the_base_form_one_family(self):
        specs = [merged_policy_spec("base", None), merged_policy_spec("ckpt125", "1" * 64),
                 merged_policy_spec("ckpt200", "2" * 64)]
        assert {"model_path", "model_weights_sha256"} <= set(POLICY_FIELDS)
        (family,) = policy_families({spec.condition_id: spec for spec in specs}).values()
        ids = [spec.condition_id for spec in specs]
        assert family.pairs == [(ids[0], ids[1]), (ids[0], ids[2])]
        assert [served_weights(spec) for spec in specs] == [("Qwen3-8B", None), ("1" * 64, None), ("2" * 64, None)]

    def test_a_member_in_another_serving_dtype_is_not_a_policy_of_the_family(self):
        specs = [merged_policy_spec("base", None, serving_dtype=None), merged_policy_spec("ckpt125", "1" * 64)]
        assert sorted(len(f.members) for f in policy_families({s.condition_id: s for s in specs}).values()) == [1, 1]

    @pytest.mark.parametrize("names", [("same", "same"), ("ckpt125-merged", "renamed")], ids=["one-dir", "a-copy"])
    def test_two_members_serving_one_merged_checkpoint_are_refused(self, names):
        first = merged_policy_spec("ckpt125", "1" * 64, model_path=names[0])
        second = merged_policy_spec("ckpt200", "1" * 64, model_path=names[1])
        with pytest.raises(CampaignRunError, match="serve the same weights"):
            policy_families({spec.condition_id: spec for spec in (first, second)})


def run_merged_policies(tmp_path, modelspec, policies=POLICIES) -> Path:
    inconsistent = {"base": (0, 1), "ckpt125": (), "ckpt200": (0,)}
    base = write_base(tmp_path / "models")
    for index, policy in enumerate(policies):
        weights = base if policy == "base" else write_merged(tmp_path / "grpo" / policy, seed=float(index))
        extra = {"figure": "figure5", "policy": policy, "policy_family": FAMILY, "policy_reference": "base"}
        cfg = weights_campaign(
            tmp_path, weights, dtype="float32", **{
                "condition_id": f"qwen/figure5/{policy}",
                "condition_extra": "{" + ", ".join(f"{k}: {v}" for k, v in extra.items()) + "}",
                "eval_cfg.reasoning_prompt_style": "forced-grpo",
                "sampling.forced_json_requirement": "label_only",
            },
        )
        backend = local_backend(model_path=weights, dtype="float32", resolved=FP32_ENGINE, style="forced-grpo",
                                script={(i, "Rule 2"): [forced("yes")] for i in inconsistent[policy]})
        backend.max_new_tokens = 1024
        backend.engine_args.update({"attention_config": {"backend": "TRITON_ATTN"}, "max_model_len": None,
                                    "enable_lora": False})
        task_spec = modelspec(reasoning_prompt_style="forced-grpo", evaluation_temperature=1.0)
        run_plan(build_plan(cfg, backend=backend, task_spec=task_spec), backend=backend, task_spec=task_spec)
    return tmp_path / "root"


def test_base_against_merged_checkpoints_is_contrasted_on_one_fp32_engine(tmp_path, modelspec):
    root = run_merged_policies(tmp_path, modelspec)
    ids = [f"qwen/figure5/{policy}" for policy in POLICIES]
    conditions = load_conditions(root)
    assert {cid: (spec.model_weights_sha256 is None, spec.serving_dtype) for cid, spec in conditions.items()} == {
        ids[0]: (True, "float32"), ids[1]: (False, "float32"), ids[2]: (False, "float32"),
    }
    payload = analyze_campaign(root, n_permutations=100, n_resamples=100, rule_set_path=tmp_path / "rules.jsonl")
    family = payload["policy_families"]["+".join(ids)]
    # the weights directory differs on purpose (placement): not a stack mismatch
    assert not {"allow_stack_mismatch", "stack_mismatches"} & set(payload)
    assert [(row["policy_a"], row["policy_b"]) for row in family["contrasts"]] == [("base", "ckpt125"), ("base", "ckpt200")]
    assert [row["estimate"] for row in family["contrasts"]] == pytest.approx([2 / 3, 1 / 3])
    assert family["model_weights_sha256"] == {
        "base": None, **{policy: conditions[cid].model_weights_sha256 for policy, cid in zip(POLICIES[1:], ids[1:])},
    }
    assert family["adapter_sha256"] == {policy: None for policy in POLICIES}
    # the LoRA settings a served adapter would bring do not refuse the contrast either
    set_recorded_runtime(root / ids[2], engine_args={
        "model": "/x/ckpt200", "dtype": "float32", "attention_config": {"backend": "TRITON_ATTN"},
        "enable_lora": True, "max_lora_rank": 32,
    })
    assert "stack_mismatches" not in analyze_campaign(root, n_permutations=100, n_resamples=100, rule_set_path=tmp_path / "rules.jsonl")


@pytest.mark.parametrize(
    ("changed", "shown"),
    [
        ({"dtype": "bfloat16"}, 'engine_args.dtype="bfloat16"'),
        ({"dtype": None}, "engine_args.dtype=null"),
        ({"attention_config": {"backend": "FLEX_ATTENTION"}}, 'engine_args.attention_config={"backend":"FLEX_ATTENTION"}'),
        ({"max_model_len": 8192}, "engine_args.max_model_len=8192"),
    ],
    ids=["serving-dtype", "auto", "attention-backend", "context-window"],
)
def test_a_contrast_across_serving_dtypes_attention_backends_or_context_windows_is_refused(tmp_path, modelspec,
                                                                                           changed, shown):
    root = run_merged_policies(tmp_path, modelspec)
    policy = root / "qwen" / "figure5" / "ckpt200"
    recorded = json.loads((policy / "draw-000" / DRAW_PROVENANCE_FNAME).read_text())["runtime"]["engine_args"]
    set_recorded_runtime(policy, engine_args={**recorded, **changed})
    with pytest.raises(CampaignRunError, match=r"policy_contrasts: 1 contrast\(s\) would compare judgments from "
                                               r"different inference stacks or serving settings \(qwen/figure5/base vs "
                                               r"qwen/figure5/ckpt200"):
        analyze_campaign(root, n_permutations=100, n_resamples=100)
    allowed = analyze_campaign(root, n_permutations=100, n_resamples=100, allow_stack_mismatch=True, rule_set_path=tmp_path / "rules.jsonl")
    assert [mismatch["scope"] for mismatch in allowed["stack_mismatches"]] == [
        "policy_contrasts", "discrimination_contrasts",
    ]
    for mismatch in allowed["stack_mismatches"]:
        assert mismatch["conditions"] == ["qwen/figure5/base", "qwen/figure5/ckpt200"] and shown in mismatch["reason"]
