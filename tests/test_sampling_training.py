from __future__ import annotations

import dataclasses
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Callable

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ruleforge.grpo import merging as merge_module
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate.aggregate import (
    POLICY_ORIENTATION,
    PolicyFamily,
    analyze_campaign,
    first_attempt_acceptance_share,
    policy_families,
    ruleset_families,
    style_families,
)
from ruleforge.core.paths import walk_strictly
from ruleforge.core.record_names import identity_key, is_retired, public_style
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    build_plan,
    campaign_manifest,
    condition_spec_for,
    execution_settings,
    load_conditions,
    preflight,
    registers_retired,
    run_plan,
    validate_campaign_cfg,
)
from ruleforge.judges.weights import (
    AdapterIdentity,
    adapter_identity,
)
from ruleforge.evaluate.identity import (
    ConditionSpec,
    differing_manifest_fields,
    recorded_fingerprint,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.digests import (
    canonical_digest,
    sha256_file,
)
from ruleforge.evaluate.identity import DRAW_PROVENANCE_FNAME
from ruleforge.evaluate.runner import (
    STATE_COMPLETE,
    CampaignRunError,
    DrawRunner,
    DrawSpec,
)
from tests.fake_scandir import failing_scandir
from tests.identity.published import CONDITIONS, PUBLIC_STYLE_ROOTS, ROOTS, VALIDATION_ROOTS, read_fixture, rows_of
from tests.support.hydra_runs import run_program
from tests.test_sampling_campaign import backend_for, compose_campaign, run_two_conditions, set_recorded_vllm
from tests.test_sampling_runner import RULES, FakeBackend, forced, make_spec, write_inputs
from tests.test_sampling_weights import QWEN3_TRUNCATION, write_base, write_merged

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Read at import, before the autouse fixture repoints the variable.
REAL_ARTIFACT_ROOT = os.environ.get("RULEFORGE_ARTIFACT_ROOT")
#: Likewise; the launchers' default when it is unset.
REAL_MODEL_ROOT = os.environ.get("RULEFORGE_MODEL_ROOT") or str(Path.home() / "models")
POLICIES = ("base", "ckpt125", "ckpt200")
FAMILY = "+".join(POLICIES)


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(campaign_module, "code_state", lambda: {})
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)


def write_adapter(root: Path, checkpoint: str, *, content: bytes | None = None, config=None) -> Path:
    path = root / "run_12162" / checkpoint / "adapter_model.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(checkpoint.encode() if content is None else content)
    payload = {"r": 32, "lora_alpha": 64, "base_model_name_or_path": "/models/Qwen3-8B"} if config is None else config
    (path.parent / "adapter_config.json").write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )
    return path


class TestAdapterIdentity:
    def test_names_the_adapter_by_content_with_a_readable_label(self, tmp_path):
        path = write_adapter(tmp_path, "checkpoint-125")
        identity = adapter_identity(path)
        assert identity == AdapterIdentity(
            path=str(path.resolve()),
            label="checkpoint-125/adapter_model.safetensors",
            sha256=sha256_file(path),
            config_sha256=sha256_file(path.parent / "adapter_config.json"),
            rank=32,
            base_model="/models/Qwen3-8B",
        )
        # Two checkpoints share a file name but never a digest.
        other = adapter_identity(write_adapter(tmp_path, "checkpoint-200"))
        assert other.label != identity.label and other.sha256 != identity.sha256

    def test_a_config_without_a_base_model_is_fine(self, tmp_path):
        assert adapter_identity(write_adapter(tmp_path, "c", config={"r": 8})).base_model is None

    def test_a_relative_spelling_names_the_same_checkpoint(self, tmp_path, monkeypatch):
        path = write_adapter(tmp_path, "checkpoint-125")
        monkeypatch.chdir(path.parent)
        assert adapter_identity("adapter_model.safetensors") == adapter_identity(path)

    def test_refuses_a_directory_or_a_missing_file(self, tmp_path):
        path = write_adapter(tmp_path, "checkpoint-125")
        with pytest.raises(SamplingDesignError, match="must be an existing adapter weights FILE"):
            adapter_identity(path.parent)
        with pytest.raises(SamplingDesignError, match="must be an existing adapter weights FILE"):
            adapter_identity(path.parent / "absent.safetensors")

    @pytest.mark.parametrize(
        ("named", "beside", "served"),
        [
            ("weights.safetensors", (), "None"),
            ("weights.safetensors", ("adapter_model.safetensors",), "adapter_model.safetensors"),
            ("adapter_model.bin", ("adapter_model.safetensors",), "adapter_model.safetensors"),
            ("adapter_model.pt", ("adapter_model.bin",), "adapter_model.bin"),
        ],
    )
    def test_refuses_a_file_vllm_would_not_serve(self, tmp_path, named, beside, served):
        # vLLM is handed the directory and loads the first adapter_model.* it finds there.
        checkpoint = tmp_path / "checkpoint-125"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
        for name in (named, *beside):
            (checkpoint / name).write_bytes(name.encode())
        with pytest.raises(SamplingDesignError, match=rf"not the weights file vLLM serves .*\({served}\)"):
            adapter_identity(checkpoint / named)

    @pytest.mark.parametrize(
        ("named", "beside"), [("adapter_model.bin", ("adapter_model.pt", "weights.safetensors")), ("adapter_model.pt", ())]
    )
    def test_a_lower_priority_file_is_served_when_nothing_outranks_it(self, tmp_path, named, beside):
        checkpoint = tmp_path / "checkpoint-125"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
        for name in (named, *beside):
            (checkpoint / name).write_bytes(name.encode())
        identity = adapter_identity(checkpoint / named)
        assert (identity.label, identity.sha256) == (f"checkpoint-125/{named}", sha256_file(checkpoint / named))

    def test_a_symlink_is_judged_where_it_resolves(self, tmp_path):
        store = tmp_path / "store"
        store.mkdir()
        (store / "weights.safetensors").write_bytes(b"weights")
        (store / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
        link = tmp_path / "checkpoint-125" / "adapter_model.safetensors"
        link.parent.mkdir()
        link.symlink_to(store / "weights.safetensors")
        (link.parent / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
        with pytest.raises(SamplingDesignError, match="not the weights file vLLM serves from " + re.escape(str(store.resolve()))):
            adapter_identity(link)
        real = write_adapter(tmp_path, "checkpoint-200")
        (tmp_path / "alias").symlink_to(real.parent)
        assert adapter_identity(tmp_path / "alias" / real.name) == adapter_identity(real)

    @pytest.mark.parametrize(
        ("config", "message"),
        [
            (None, "cannot read"),
            ("{not json", "cannot read"),
            ("[32]", "no positive integer rank"),
            ({"lora_alpha": 64}, "no positive integer rank"),
            ({"r": 0}, "no positive integer rank"),
            ({"r": True}, "no positive integer rank"),
            ({"r": "32"}, "no positive integer rank"),
        ],
    )
    def test_refuses_a_config_whose_rank_does_not_parse(self, tmp_path, config, message):
        path = write_adapter(tmp_path, "c", config=config if config is not None else {"r": 1})
        if config is None:
            (path.parent / "adapter_config.json").unlink()
        with pytest.raises(SamplingDesignError, match=message):
            adapter_identity(path)


def qwen_spec(**changes) -> ConditionSpec:
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


class TestConditionIdentity:
    def test_an_adapter_free_condition_keeps_its_pre_adapter_fingerprint(self):
        spec = qwen_spec()
        # Computed from this exact spec without adapter digests.
        assert spec.fingerprint == "c91c84a9b0692cdbdc314e4b732f01cbe5ee021b273226b3f68480a45ba3a793"
        payload = spec.to_dict()
        assert "adapter_sha256" not in payload and "adapter_config_sha256" not in payload
        assert payload["adapter_path"] is None
        assert ConditionSpec.from_dict(payload) == spec

    def test_the_adapter_content_is_identity(self):
        tuned = qwen_spec(
            condition_id="Qwen3-8B/figure5/ckpt125",
            adapter_path="checkpoint-125/adapter_model.safetensors",
            adapter_sha256="1" * 64, adapter_config_sha256="2" * 64,
        )
        payload = tuned.to_dict()
        assert (payload["adapter_sha256"], payload["adapter_config_sha256"]) == ("1" * 64, "2" * 64)
        assert ConditionSpec.from_dict(payload) == tuned
        assert tuned.fingerprint != qwen_spec().fingerprint
        assert dataclasses.replace(tuned, adapter_sha256="3" * 64).fingerprint != tuned.fingerprint
        assert dataclasses.replace(tuned, adapter_config_sha256="3" * 64).fingerprint != tuned.fingerprint

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"adapter_sha256": "1" * 64, "adapter_config_sha256": "2" * 64}, "digests but no adapter_path"),
            ({"adapter_path": "c/a", "adapter_sha256": "1" * 64}, "together or not at all"),
            ({"adapter_path": "c/a", "adapter_sha256": "XYZ", "adapter_config_sha256": "2" * 64}, "hex digests"),
            ({"adapter_path": "c/a", "adapter_sha256": "1" * 64, "adapter_config_sha256": 7}, "hex digests"),
        ],
    )
    def test_adapter_digests_are_a_pair_beside_a_path(self, changes, message):
        with pytest.raises(SamplingDesignError, match=message):
            qwen_spec(**changes)

    def test_a_name_only_adapter_stays_constructible(self):
        spec = qwen_spec(adapter_path="final_adapter")
        assert spec.adapter_sha256 is None and "adapter_sha256" not in spec.to_dict()


def _training_like_campaign(tmp_path, policy, adapter=None, **overrides):
    extra = {"figure": "figure5", "policy": policy, "policy_family": FAMILY, "policy_reference": "base"}
    base = {
        "condition_id": f"qwen/figure5/{policy}",
        "condition_extra": "{" + ", ".join(f"{k}: {v}" for k, v in extra.items()) + "}",
        "eval_cfg.reasoning_prompt_style": "forced-grpo",
        "sampling.forced_json_requirement": "label_only",
    }
    if adapter is not None:
        base["+judge_model.adapter_path"] = str(adapter)
    base.update(overrides)
    return compose_campaign(tmp_path, **base)


class TestConditionSpecFor:
    def test_a_local_adapter_enters_by_label_and_digests(self, tmp_path, modelspec):
        adapter = write_adapter(tmp_path / "adapters", "checkpoint-125")
        cfg = _training_like_campaign(tmp_path, "ckpt125", adapter)
        task_spec = modelspec(reasoning_prompt_style="forced-grpo", evaluation_temperature=1.0)
        spec = condition_spec_for(
            condition_id="qwen/figure5/ckpt125", backend=backend_for("forced-grpo"), task_spec=task_spec,
            config=preflight(cfg)[0], judge_type="No_Interpretation", rule_set_path=Path(str(cfg.rule_set_path)),
            model_cfg=cfg.judge_model,
        )
        assert spec.adapter_path == "checkpoint-125/adapter_model.safetensors"
        assert spec.adapter_sha256 == sha256_file(adapter)
        assert spec.adapter_config_sha256 == sha256_file(adapter.parent / "adapter_config.json")

    def test_a_provider_judge_records_no_adapter(self, tmp_path, modelspec):
        cfg = compose_campaign(tmp_path, **{"model@judge_model": "gpt-5.6-luna-batch", "+judge_model.adapter_path": "/x/y.safetensors"})
        backend = backend_for("naive", model_type="openai_batch", seed_capability="unsupported")
        spec = condition_spec_for(
            condition_id="luna/naive", backend=backend,
            task_spec=modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0),
            config=preflight(compose_campaign(tmp_path))[0], judge_type="No_Interpretation",
            rule_set_path=Path(str(cfg.rule_set_path)), model_cfg=cfg.judge_model,
        )
        assert (spec.adapter_path, spec.adapter_sha256, spec.adapter_config_sha256) == (None, None, None)

    def test_an_empty_adapter_path_means_none(self, tmp_path, modelspec):
        cfg = compose_campaign(tmp_path, **{"+judge_model.adapter_path": "''"})
        assert campaign_module.judge_adapter(cfg.judge_model) is None
        preflight(cfg)


class TestPreflight:
    def test_an_approved_adapter_passes(self, tmp_path):
        adapter = write_adapter(tmp_path / "adapters", "checkpoint-125")
        cfg = _training_like_campaign(tmp_path, "ckpt125", adapter, approved_adapter_sha256=sha256_file(adapter))
        config, manifest = preflight(cfg)
        assert manifest.record_count == 3 and config.forced_json_requirement == "label_only"
        assert not (tmp_path / "root").exists()

    def test_refuses_an_adapter_directory(self, tmp_path):
        adapter = write_adapter(tmp_path / "adapters", "checkpoint-125")
        with pytest.raises(SamplingDesignError, match="must be an existing adapter weights FILE"):
            preflight(_training_like_campaign(tmp_path, "ckpt125", adapter.parent))

    def test_refuses_a_file_vllm_would_not_serve(self, tmp_path):
        adapter = write_adapter(tmp_path / "adapters", "checkpoint-125")
        shadowed = adapter.with_name("adapter_model.bin")
        shadowed.write_bytes(b"older weights")
        with pytest.raises(SamplingDesignError, match=r"not the weights file vLLM serves .*\(adapter_model.safetensors\)"):
            preflight(_training_like_campaign(tmp_path, "ckpt125", shadowed, approved_adapter_sha256=sha256_file(shadowed)))

    def test_refuses_an_adapter_without_its_config(self, tmp_path):
        adapter = write_adapter(tmp_path / "adapters", "checkpoint-125")
        (adapter.parent / "adapter_config.json").unlink()
        with pytest.raises(SamplingDesignError, match="cannot read .*adapter_config.json"):
            preflight(_training_like_campaign(tmp_path, "ckpt125", adapter))

    def test_refuses_an_unparseable_rank(self, tmp_path):
        adapter = write_adapter(tmp_path / "adapters", "checkpoint-125", config={"r": "wide"})
        with pytest.raises(SamplingDesignError, match="rank"):
            preflight(_training_like_campaign(tmp_path, "ckpt125", adapter))

    def test_refuses_a_digest_other_than_the_approved_one(self, tmp_path):
        adapter = write_adapter(tmp_path / "adapters", "checkpoint-125")
        with pytest.raises(CampaignRunError, match="adapter SHA-256 .* does not match approved 0000"):
            preflight(_training_like_campaign(tmp_path, "ckpt125", adapter, approved_adapter_sha256="0" * 64))

    def test_refuses_an_approval_without_an_adapter(self, tmp_path):
        with pytest.raises(CampaignRunError, match="judge with the base model"):
            preflight(_training_like_campaign(tmp_path, "ckpt125", approved_adapter_sha256="0" * 64))

    def test_the_preflight_script_reports_the_adapter(self, tmp_path, capsys, monkeypatch):
        adapter = write_adapter(tmp_path / "adapters", "checkpoint-125")
        scenarios = write_inputs(tmp_path)
        rules = tmp_path / "rules.jsonl"
        rules.write_text("".join(json.dumps(r) + "\n" for r in RULES), encoding="utf-8")
        script = "ruleforge.launch.programs.preflight_campaign"
        monkeypatch.setattr(sys, "argv", [
            str(script), "campaign_id=camp", f"campaign_root={tmp_path / 'root'}",
            f"scenario_set_path={scenarios}", "approved_scenario_sha256=null", f"rule_set_path={rules}",
            "sampling.campaign_mode=pilot", f"+judge_model.adapter_path={adapter}",
            f"approved_adapter_sha256={sha256_file(adapter)}", f"hydra.run.dir={tmp_path / 'hydra'}",
        ])
        run_program(script)
        report = json.loads(capsys.readouterr().out)
        assert report["adapter"] == {
            "label": "checkpoint-125/adapter_model.safetensors", "sha256": sha256_file(adapter),
            "rank": 32, "base_model": "/models/Qwen3-8B",
        }

    def test_refuses_an_adapter_on_a_provider_judge(self, tmp_path):
        adapter = write_adapter(tmp_path / "adapters", "checkpoint-125")
        cfg = compose_campaign(
            tmp_path, **{"model@judge_model": "gpt-5.6-luna-batch", "+judge_model.adapter_path": str(adapter)}
        )
        with pytest.raises(CampaignRunError, match="only a local judge serves a LoRA adapter"):
            preflight(cfg)


def adapter_draw_spec(inputs, adapter: Path, *, draw=0):
    manifest, _ = inputs
    spec = make_spec(manifest, draw=draw)
    identity = adapter_identity(adapter)
    condition = dataclasses.replace(
        spec.condition, adapter_path=identity.label, adapter_sha256=identity.sha256,
        adapter_config_sha256=identity.config_sha256,
    )
    return dataclasses.replace(spec, condition=condition)


def adapter_runner(tmp_path, inputs, modelspec, spec: DrawSpec, backend) -> DrawRunner:
    return DrawRunner(
        backend=backend, task_spec=modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0),
        spec=spec, draw_dir=tmp_path / "draw", scenario_rows=inputs[1], rules=RULES,
    )


@pytest.fixture
def inputs(tmp_path):
    path = write_inputs(tmp_path)
    return ScenarioManifest.build(path), [json.loads(line) for line in path.read_text().splitlines()]


class TestRunnerAdapter:
    def test_a_served_adapter_is_verified_and_recorded(self, tmp_path, inputs, modelspec):
        adapter = write_adapter(tmp_path / "adapters", "checkpoint-125")
        backend = FakeBackend()
        backend.adapter_path = adapter
        backend.adapter_metadata = {"name": "checkpoint-125", "vllm_int_id": 7}
        spec = adapter_draw_spec(inputs, adapter)
        assert adapter_runner(tmp_path, inputs, modelspec, spec, backend).run()["state"] == STATE_COMPLETE
        provenance = json.loads((tmp_path / "draw" / DRAW_PROVENANCE_FNAME).read_text())
        assert provenance["adapter"] == {
            "path": str(adapter.resolve()),
            "sha256": sha256_file(adapter),
            "config_sha256": sha256_file(adapter.parent / "adapter_config.json"),
            "rank": 32,
            "base_model": "/models/Qwen3-8B",
            "backend_metadata": {"name": "checkpoint-125", "vllm_int_id": 7},
        }

        moved = tmp_path / "moved" / "checkpoint-125" / "adapter_model.safetensors"
        moved.parent.mkdir(parents=True)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            (moved.parent / name).write_bytes((adapter.parent / name).read_bytes())
        backend.adapter_path = moved
        assert adapter_runner(tmp_path, inputs, modelspec, spec, backend).run()["state"] == STATE_COMPLETE

    def test_other_weights_under_the_same_name_are_refused_before_generation(self, tmp_path, inputs, modelspec):
        adapter = write_adapter(tmp_path / "adapters", "checkpoint-125")
        spec = adapter_draw_spec(inputs, adapter)
        backend = FakeBackend()
        backend.adapter_path = write_adapter(tmp_path / "elsewhere", "checkpoint-125", content=b"other weights")
        runner = adapter_runner(tmp_path, inputs, modelspec, spec, backend)
        with pytest.raises(CampaignRunError, match="refusing to add judgments"):
            runner.run()
        assert backend.blocks == [] and not (tmp_path / "draw" / DRAW_PROVENANCE_FNAME).exists()

    def test_an_adapter_on_one_side_only_is_refused(self, tmp_path, inputs, modelspec):
        adapter = write_adapter(tmp_path / "adapters", "checkpoint-125")
        served = FakeBackend()
        served.adapter_path = adapter
        with pytest.raises(CampaignRunError, match="but the backend serves .*adapter_model.safetensors"):
            adapter_runner(tmp_path, inputs, modelspec, make_spec(inputs[0]), served)
        with pytest.raises(CampaignRunError, match="but the backend serves None"):
            adapter_runner(tmp_path, inputs, modelspec, adapter_draw_spec(inputs, adapter), FakeBackend())
        spec = make_spec(inputs[0])
        name_only = dataclasses.replace(spec, condition=dataclasses.replace(spec.condition, adapter_path="checkpoint-125"))
        with pytest.raises(CampaignRunError, match="named by content"):
            adapter_runner(tmp_path, inputs, modelspec, name_only, served)
        with pytest.raises(CampaignRunError, match="but the backend serves None"):
            adapter_runner(tmp_path, inputs, modelspec, name_only, FakeBackend())

    def test_an_adapter_free_draw_has_no_adapter_provenance(self, tmp_path, inputs, modelspec):
        backend = FakeBackend()
        adapter_runner(tmp_path, inputs, modelspec, make_spec(inputs[0]), backend).run()
        provenance = json.loads((tmp_path / "draw" / DRAW_PROVENANCE_FNAME).read_text())
        assert "adapter" not in provenance

    def test_a_provider_backend_never_serves_its_recorded_adapter(self, tmp_path, inputs, modelspec):
        backend = FakeBackend(model_type="openai_batch", seed_capability="unsupported")
        backend.adapter_path = Path("/x/y.safetensors")
        runner = adapter_runner(tmp_path, inputs, modelspec, make_spec(inputs[0], seed_capability="unsupported"), backend)
        assert runner.adapter_provenance() is None


def policy_spec(policy: str, digest: str | None, **extra_changes) -> ConditionSpec:
    extra = {"figure": "figure5", "policy": policy, "policy_family": FAMILY, "policy_reference": "base"}
    extra.update(extra_changes)
    extra = {key: value for key, value in extra.items() if value is not None}
    adapter = {} if digest is None else {
        "adapter_path": f"{policy}/adapter_model.safetensors", "adapter_sha256": digest, "adapter_config_sha256": "c" * 64,
    }
    return qwen_spec(condition_id=f"Qwen3-8B/figure5/{policy}", extra=tuple(extra.items()), **adapter)


def family_of(*specs: ConditionSpec) -> dict[str, ConditionSpec]:
    return {spec.condition_id: spec for spec in specs}


class TestPolicyFamilies:
    def test_the_declared_family_contrasts_the_reference_with_every_member(self):
        conditions = family_of(policy_spec("ckpt200", "2" * 64), policy_spec("base", None), policy_spec("ckpt125", "1" * 64))
        ids = [f"Qwen3-8B/figure5/{p}" for p in POLICIES]
        assert policy_families(conditions) == {
            "+".join(ids): PolicyFamily(
                members=tuple(ids), policies=dict(zip(POLICIES, ids)), declared=POLICIES, reference="base",
            )
        }
        (family,) = policy_families(conditions).values()
        assert family.missing == []
        assert family.pairs == [(ids[0], ids[1]), (ids[0], ids[2])]
        assert ruleset_families(conditions) == {}
        assert sorted(style_families(conditions).values()) == [[cid] for cid in sorted(ids)]

    def test_a_partial_family_reports_what_is_missing(self):
        family = next(iter(policy_families(family_of(policy_spec("ckpt125", "1" * 64))).values()))
        assert family.missing == ["base", "ckpt200"] and family.pairs == []

    def test_conditions_without_a_policy_form_no_family(self):
        assert policy_families(family_of(qwen_spec(extra=()), qwen_spec(condition_id="x/y", extra=(("figure", "f"),)))) == {}

    def test_another_design_axis_keeps_policies_apart(self):
        conditions = family_of(policy_spec("base", None), dataclasses.replace(policy_spec("ckpt125", "1" * 64), label_rule="bare"))
        assert sorted(len(f.members) for f in policy_families(conditions).values()) == [1, 1]

    @pytest.mark.parametrize(
        ("specs", "message"),
        [
            ((dict(policy="base", digest=None, policy_family=None),), "not policy_family and policy_reference"),
            ((dict(policy="base", digest=None, policy_reference=None),), "not policy_family and policy_reference"),
            ((dict(policy="base", digest=None, policy_reference="ckpt9"),), "including the reference"),
            ((dict(policy="base", digest=None, policy_family="base+base"),), "distinct policies"),
            ((dict(policy="ckpt9", digest="1" * 64),), "distinct member of the declared family"),
        ],
    )
    def test_a_malformed_declaration_is_refused(self, specs, message):
        conditions = family_of(*[policy_spec(s.pop("policy"), s.pop("digest"), **s) for s in specs])
        with pytest.raises(CampaignRunError, match=message):
            policy_families(conditions)

    def test_two_members_under_one_label_are_refused(self):
        twin = dataclasses.replace(policy_spec("ckpt125", "1" * 64), condition_id="Qwen3-8B/figure5/twin", adapter_sha256="3" * 64)
        with pytest.raises(CampaignRunError, match="distinct member"):
            policy_families(family_of(policy_spec("ckpt125", "1" * 64), twin))

    @pytest.mark.parametrize("digests", [("1" * 64, "1" * 64), (None, None)], ids=["same-adapter", "two-base"])
    def test_two_members_serving_the_same_adapter_are_refused(self, digests):
        first = policy_spec("ckpt125", digests[0])
        second = dataclasses.replace(policy_spec("ckpt200", digests[1]), adapter_path=first.adapter_path)
        with pytest.raises(CampaignRunError, match="serve the same weights"):
            policy_families(family_of(first, second))


def run_policy_conditions(tmp_path, modelspec, policies=POLICIES, *, same_adapter=False, retried=None):
    """Forced (label_only) conditions of one judge under three policies."""
    inconsistent = {"base": (0, 1), "ckpt125": (), "ckpt200": (0,)}
    for policy in policies:
        adapter = None
        if policy != "base":
            adapter = write_adapter(tmp_path / "adapters", "shared" if same_adapter else policy)
        cfg = _training_like_campaign(tmp_path, policy, adapter)
        script = {(i, "Rule 2"): [forced("yes")] for i in inconsistent[policy]}
        script.update({(i, "full_ruleset"): ["not json", forced("no")] for i in (retried or {}).get(policy, ())})
        backend = backend_for("forced-grpo", script=script)
        backend.adapter_path = adapter
        task_spec = modelspec(reasoning_prompt_style="forced-grpo", evaluation_temperature=1.0)
        run_plan(build_plan(cfg, backend=backend, task_spec=task_spec), backend=backend, task_spec=task_spec)
    return tmp_path / "root"


def test_base_against_each_checkpoint_is_one_holm_family_of_two(tmp_path, modelspec):
    root = run_policy_conditions(tmp_path, modelspec, retried={"ckpt125": (2,)})
    ids = [f"qwen/figure5/{policy}" for policy in POLICIES]
    conditions = load_conditions(root)
    assert len({spec.fingerprint for spec in conditions.values()}) == 3
    payload = analyze_campaign(root, n_permutations=200, n_resamples=150, rule_set_path=tmp_path / "rules.jsonl")

    assert [payload["conditions"][cid]["estimate"] for cid in ids] == pytest.approx([2 / 3, 0.0, 1 / 3])
    family = payload["policy_families"]["+".join(ids)]
    assert family["members"] == ids and "missing_policies" not in family
    assert family["orientation"] == POLICY_ORIENTATION == "rate(reference) - rate(member)"
    assert family["reference_policy"] == "base" and family["declared_policies"] == list(POLICIES)
    assert family["policies"] == dict(zip(POLICIES, ids))
    assert family["adapter_sha256"]["base"] is None
    assert family["adapter_sha256"]["ckpt125"] == conditions[ids[1]].adapter_sha256
    assert family["attempt_statistics"] == {
        policy: payload["conditions"][cid]["attempt_statistics"] for policy, cid in zip(POLICIES, ids)
    }
    assert family["attempt_statistics"]["ckpt125"]["attempts_histogram"] == {"1": 5, "2": 1}
    assert family["attempt_statistics"]["ckpt125"]["rejected_attempts"] == 1
    assert family["first_attempt_acceptance_share"] == {"base": 1.0, "ckpt125": pytest.approx(5 / 6), "ckpt200": 1.0}
    rows = family["contrasts"]
    assert [(row["a"], row["b"]) for row in rows] == [(ids[0], ids[1]), (ids[0], ids[2])]
    assert [(row["policy_a"], row["policy_b"]) for row in rows] == [("base", "ckpt125"), ("base", "ckpt200")]
    # rate(base) - rate(member): positive where the checkpoint is less inconsistent.
    assert [row["estimate"] for row in rows] == pytest.approx([2 / 3, 1 / 3])
    for row in rows:
        assert row["family"] == "policy_contrasts" and row["family_size"] == 2
        assert row["inner_resample"] == "independent" and row["p_holm"] >= row["p_value"]
        assert row["test_method"] == "scenario_clustered_paired_randomization_v1"
    # Holm over two: the smaller p doubles, the larger is at least it.
    smaller, larger = sorted(rows, key=lambda row: row["p_value"])
    assert smaller["p_holm"] == pytest.approx(min(1.0, 2 * smaller["p_value"]))
    assert larger["p_holm"] == pytest.approx(max(smaller["p_holm"], larger["p_value"]))
    assert payload["forced_conditions"] == {} and payload["ruleset_families"] == {}
    assert all(group["contrasts"] == [] for group in payload["style_families"].values())


def test_a_policy_judged_on_another_inference_stack_is_not_contrasted(tmp_path, modelspec):
    root = run_policy_conditions(tmp_path, modelspec)
    for policy, version in (("base", "0.28.0"), ("ckpt125", "0.28.0"), ("ckpt200", "0.19.0")):
        set_recorded_vllm(root / "qwen" / "figure5" / policy, version)
    with pytest.raises(CampaignRunError, match=r"policy_contrasts: 1 contrast.*qwen/figure5/base vs qwen/figure5/ckpt200"):
        analyze_campaign(root, n_permutations=100, n_resamples=100)
    payload = analyze_campaign(root, n_permutations=100, n_resamples=100, allow_stack_mismatch=True, rule_set_path=tmp_path / "rules.jsonl")
    (family,) = payload["policy_families"].values()
    assert ["stack_mismatch" in row for row in family["contrasts"]] == [False, True]


def test_a_family_short_of_a_declared_policy_reports_no_contrast(tmp_path, modelspec):
    root = run_policy_conditions(tmp_path, modelspec, POLICIES[:2])
    with pytest.raises(CampaignRunError, match="missing declared policies .*ckpt200"):
        analyze_campaign(root, n_permutations=100, n_resamples=100)
    payload = analyze_campaign(root, n_permutations=100, n_resamples=100, allow_incomplete_conditions=True, rule_set_path=tmp_path / "rules.jsonl")
    ids = [f"qwen/figure5/{policy}" for policy in POLICIES[:2]]
    family = payload["policy_families"]["+".join(ids)]
    assert family["members"] == ids and family["contrasts"] == []
    assert family["missing_policies"] == ["ckpt200"]
    assert set(family["attempt_statistics"]) == set(family["first_attempt_acceptance_share"]) == {"base", "ckpt125"}


def exhaust(root: Path, condition_id: str, draw: int = 1) -> None:
    gate = root / condition_id / f"draw-{draw:03d}" / "completion.json"
    record = json.loads(gate.read_text())
    record["state"] = "incomplete_exhausted"
    gate.write_text(json.dumps(record))


@pytest.mark.parametrize("registered", [POLICIES, ("base", "ckpt200")], ids=["all-registered", "one-unregistered"])
def test_an_exhausted_policy_is_excluded_not_missing(tmp_path, modelspec, registered):
    root = run_policy_conditions(tmp_path, modelspec, registered)
    exhaust(root, "qwen/figure5/ckpt200")
    with pytest.raises(CampaignRunError, match="not complete"):
        analyze_campaign(root, n_permutations=100, n_resamples=100)
    payload = analyze_campaign(root, n_permutations=100, n_resamples=100, allow_incomplete_conditions=True, rule_set_path=tmp_path / "rules.jsonl")
    ids = [f"qwen/figure5/{policy}" for policy in POLICIES if policy in registered]
    family = payload["policy_families"]["+".join(ids)]
    assert family["members"] == ids and family["contrasts"] == []
    assert family["excluded_policies"] == {"ckpt200": "qwen/figure5/ckpt200"}
    unregistered = [policy for policy in POLICIES if policy not in registered]
    assert family.get("missing_policies", []) == unregistered
    excluded = payload["excluded_conditions"]["qwen/figure5/ckpt200"]
    assert family["attempt_statistics"]["ckpt200"] == excluded["attempt_statistics"]
    assert family["attempt_statistics"]["base"] == payload["conditions"]["qwen/figure5/base"]["attempt_statistics"]
    assert family["first_attempt_acceptance_share"]["ckpt200"] == 1.0
    assert "qwen/figure5/ckpt200" not in payload["conditions"]


def test_a_first_attempt_share_needs_an_accepted_bundle():
    assert first_attempt_acceptance_share({"attempts_histogram": {"1": 3, "4": 1}}) == 0.75
    assert first_attempt_acceptance_share({"attempts_histogram": {"2": 2}}) == 0.0
    assert first_attempt_acceptance_share({"attempts_histogram": {}}) is None


def test_a_declared_base_only_root_is_refused_until_its_checkpoints_land(tmp_path, modelspec):
    root = run_policy_conditions(tmp_path, modelspec, ("base",))
    with pytest.raises(CampaignRunError, match=r"missing declared policies .*\['ckpt125', 'ckpt200'\]"):
        analyze_campaign(root, n_permutations=100, n_resamples=100)


def test_two_checkpoints_with_the_same_weights_are_refused(tmp_path, modelspec):
    root = run_policy_conditions(tmp_path, modelspec, same_adapter=True)
    with pytest.raises(CampaignRunError, match="serve the same weights"):
        analyze_campaign(root, n_permutations=100, n_resamples=100)


def test_a_root_without_policies_gains_an_empty_block(tmp_path, modelspec):
    root, _ = run_two_conditions(tmp_path, modelspec)
    payload = analyze_campaign(root, n_permutations=100, n_resamples=100)
    assert payload["policy_families"] == {}
    assert json.loads((root / "aggregate" / "estimates.json").read_text())["policy_families"] == {}


JOB_DIR = REPO_ROOT  # the jobs' working directory: the configs' relative paths are the checkout's
PAPER_POPULATION_SHA256 = "c16f66f4ad3fc93fad46bf31efaa8325f30fb750104d1e13a9326b7cd46b10d4"
PAPER_POPULATION_MANIFEST_SHA256 = "a22af097639cfdaa15b4508c66631af1cb6d45780d8ada93a9fb1d5a29390462"


def declare(merged: dict[str, Path], family: str | None = None, **digests: str) -> list[str]:
    """Hydra overrides declaring ``merged`` checkpoints, each approved by a weights digest, and the family."""
    overrides = []
    for label, path in merged.items():
        digest = digests.get(label) or merge_module.weights_digest(path)
        overrides.append(f"+training.policies.{label}={{model_path: {path}, weights_sha256: {digest}}}")
    overrides.append(f"training.family={family or '+'.join(('base', *merged))}")
    return overrides


def compose_training(policy: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(
            config_name="sampling_training", overrides=["campaign_id=training", f"training.policy={policy}", *overrides]
        )


@pytest.fixture
def in_job_dir(monkeypatch):
    monkeypatch.chdir(JOB_DIR)


def test_the_template_is_the_decided_design_on_the_base_model(tmp_path):
    cfg = compose_training("base", "training.family=base")
    validate_campaign_cfg(cfg)
    assert cfg.condition_id == "Qwen3-8B/figure5/base"
    assert cfg.judge_model.model_name == "Qwen3-8B" and cfg.agent_model.model_name == "Qwen3-8B"
    assert (cfg.judge_model.dtype, cfg.judge_model.attention_backend) == ("float32", "TRITON_ATTN")
    assert campaign_module.judge_serving_dtype(cfg.judge_model) == "float32"
    assert cfg.judge_model.get("adapter_path") is None and cfg.approved_adapter_sha256 is None
    assert dict(cfg.condition_extra) == {
        "figure": "figure5", "policy": "base", "policy_family": "base", "policy_reference": "base",
    }
    # Only the base model is declared: the upstream directory, named by its name.
    assert {label: dict(entry) for label, entry in cfg.training.policies.items()} == {
        "base": {"model_path": str(tmp_path / "models" / "Qwen3-8B"), "weights_sha256": None},
    }
    assert cfg.judge_model.model_path == str(tmp_path / "models" / "Qwen3-8B")
    assert cfg.approved_model_weights_sha256 is None
    assert cfg.scenario_set_path == str(tmp_path / "artifacts" / "populations" / "frozen_500_r3.jsonl")
    assert cfg.approved_scenario_sha256 == PAPER_POPULATION_SHA256
    assert cfg.approved_rule_set_sha256 == sha256_file(JOB_DIR / str(cfg.rule_set_path))
    assert cfg.eval_cfg.reasoning_prompt_style == "forced-grpo"
    assert cfg.eval_cfg.individual_rule_context == "isolated"
    assert cfg.interpretation == "No_Interpretation" and cfg.evaluation_mode == "both"
    assert dict(cfg.sampling) | {"bootstrap": None} == {
        "enabled": True, "schema_version": 2, "campaign_mode": "final", "temperature": 1.0,
        "num_decoder_draws": 5, "max_semantic_attempts": 20, "retry_scope": "component",
        "require_complete": True, "generation_seed_base": 16, "scenarios_per_request_block": 50,
        "forced_json_requirement": "label_only", "truncation": "model_default",
        # A closed thinking block is part of validity: component retry
        "strict_think": True,
        "bootstrap": None,
    }


def test_every_committed_policy_but_the_reference_pins_its_weights(tmp_path):
    cfg = compose_training("base", "training.family=base")
    unpinned = sorted(
        label for label, entry in cfg.training.policies.items()
        if label != cfg.training.reference and entry.weights_sha256 is None
    )
    assert unpinned == [], f"training.policies {unpinned} name no approved weights_sha256"
    assert cfg.training.policies[cfg.training.reference].weights_sha256 is None  # upstream: named by directory


def test_the_template_refuses_every_member_until_its_family_is_declared(tmp_path):
    cfg = compose_training("base")
    validate_campaign_cfg(cfg)
    with pytest.raises(CampaignRunError, match=r"condition_extra does not resolve .*training\.family"):
        preflight(cfg)
    with pytest.raises(CampaignRunError, match="condition_extra does not resolve"):
        campaign_module.condition_extra(cfg)


def test_a_declared_policy_composes_into_its_own_condition(tmp_path):
    merged = write_merged(tmp_path / "grpo" / "run", "checkpoint-40")
    cfg = compose_training("grpo", *declare({"grpo": merged}))
    validate_campaign_cfg(cfg)
    assert cfg.condition_id == "Qwen3-8B/figure5/grpo"
    assert cfg.judge_model.model_path == str(merged)
    assert cfg.approved_model_weights_sha256 == merge_module.weights_digest(merged)
    assert campaign_module.condition_extra(cfg) == {
        "figure": "figure5", "policy": "grpo", "policy_family": "base+grpo", "policy_reference": "base",
    }


def _training_specs(tmp_path, modelspec, merged: dict[str, Path]) -> dict[str, ConditionSpec]:
    scenarios = write_inputs(tmp_path)
    specs = {}
    for policy in ("base", *merged):
        cfg = compose_training(
            policy, f"scenario_set_path={scenarios}", "approved_scenario_sha256=null",
            "sampling.campaign_mode=pilot", *declare(merged),
        )
        config, _ = preflight(cfg)
        specs[policy] = condition_spec_for(
            condition_id=str(cfg.condition_id), backend=backend_for("forced-grpo"),
            task_spec=modelspec(reasoning_prompt_style="forced-grpo", evaluation_temperature=1.0),
            config=config, judge_type=str(cfg.interpretation),
            rule_set_path=Path(str(cfg.rule_set_path)),
            model_cfg=cfg.judge_model, extra=campaign_module.condition_extra(cfg),
        )
    return specs


def test_the_declared_policies_are_conditions_with_distinct_fingerprints(tmp_path, modelspec, in_job_dir):
    write_base(tmp_path / "models")
    merged = {"grpo_a": write_merged(tmp_path / "grpo" / "run_a", "checkpoint-40", seed=1.0),
              "grpo_b": write_merged(tmp_path / "grpo" / "run_b", "checkpoint-40", seed=2.0)}
    specs = _training_specs(tmp_path, modelspec, merged)
    assert len({spec.condition_id for spec in specs.values()}) == 3
    assert len({spec.fingerprint for spec in specs.values()}) == 3
    assert (specs["base"].model_path, specs["base"].model_weights_sha256) == ("Qwen3-8B", None)
    assert specs["grpo_a"].model_path == specs["grpo_b"].model_path == "checkpoint-40"
    assert specs["grpo_a"].model_weights_sha256 == merge_module.weights_digest(merged["grpo_a"])
    assert all(spec.serving_dtype == "float32" and spec.adapter_path is None for spec in specs.values())
    assert all((spec.strict_think, spec.retry_scope) == (True, "component") for spec in specs.values())
    assert all(spec.to_dict()["sampling_truncation"] == QWEN3_TRUNCATION for spec in specs.values())
    (family,) = policy_families({s.condition_id: s for s in specs.values()}).values()
    assert family.pairs == [("Qwen3-8B/figure5/base", "Qwen3-8B/figure5/grpo_a"),
                            ("Qwen3-8B/figure5/base", "Qwen3-8B/figure5/grpo_b")]


def test_packing_a_policys_draws_is_scheduling_beside_its_weights_approval(tmp_path, in_job_dir):
    scenarios = write_inputs(tmp_path)
    merged = write_merged(tmp_path / "grpo" / "run", "checkpoint-40")
    overrides = [f"scenario_set_path={scenarios}", "approved_scenario_sha256=null", "sampling.campaign_mode=pilot",
                 *declare({"grpo": merged})]
    manifests = []
    for pack in (False, True):
        cfg = compose_training("grpo", *overrides, f"execution.pack_draws={str(pack).lower()}")
        config, manifest = preflight(cfg)
        assert execution_settings(cfg).pack_draws is pack
        manifests.append(campaign_manifest(str(cfg.campaign_id), config, manifest))
    assert differing_manifest_fields(*manifests) == []


@pytest.mark.parametrize(
    ("setup", "error", "message"),
    [
        ("retrained", CampaignRunError, "does not match approved"),
        ("tampered", SamplingDesignError, "a shard was changed or replaced"),
        ("no_manifest", CampaignRunError, "carries no derivation manifest"),
        ("bf16_engine", CampaignRunError, "holds float32 weights and judge_model Qwen3-8B declares no dtype"),
    ],
)
def test_the_template_refuses_a_checkpoint_that_is_not_the_approved_one(tmp_path, setup, error, message, in_job_dir):
    scenarios = write_inputs(tmp_path)
    approved = write_merged(tmp_path / "grpo" / "run", "checkpoint-40")
    digest = merge_module.weights_digest(approved)
    extra = []
    if setup == "retrained":  # another merge under the same name, with its own consistent manifest
        write_merged(tmp_path / "grpo" / "retrain", "checkpoint-40", seed=5.0)
        approved = tmp_path / "grpo" / "retrain" / "checkpoint-40"
    elif setup == "tampered":
        shard = next(approved.glob("*.safetensors"))
        shard.write_bytes(shard.read_bytes()[:-1] + b"\x00")
    elif setup == "no_manifest":
        (approved / merge_module.MANIFEST_FNAME).unlink()
    else:  # the bf16 engine's config would serve the fp32 merge in bf16 under auto
        extra = ["model@judge_model=qwen3-8b"]
    cfg = compose_training("grpo", f"scenario_set_path={scenarios}", "approved_scenario_sha256=null",
                          "sampling.campaign_mode=pilot", *extra, *declare({"grpo": approved}, grpo=digest))
    with pytest.raises(error, match=message):
        preflight(cfg)


@pytest.fixture
def real_artifact_root() -> Path:
    """The artifact root the test run started with, before the autouse fixture redirects it."""
    if not REAL_ARTIFACT_ROOT or not Path(REAL_ARTIFACT_ROOT).is_dir():
        pytest.skip("RULEFORGE_ARTIFACT_ROOT is not set")
    return Path(REAL_ARTIFACT_ROOT)


def validation_root_dirs(artifact_root: Path) -> list[Path]:
    """The validation set's campaign roots under ``artifact_root``, refused unless every shipped root is there."""
    sampling = artifact_root / "sampling"
    missing = [name for name in ROOTS if not (sampling / name).is_dir()]
    assert not missing, f"shipped roots of the release-validation set missing under {sampling}: {missing}"
    return [sampling / name for name in VALIDATION_ROOTS if (sampling / name).is_dir()]


def validation_condition_files(artifact_root: Path, walk: Callable[[Path], list[Path]]) -> list[Path]:
    """``walk``'s ``condition.json`` files under each validation root, refused when a root yields none."""
    paths = []
    for root in validation_root_dirs(artifact_root):
        found = walk(root)
        assert found, f"no registered condition under {root}"
        paths.extend(found)
    return paths


def check_registered_condition(payload: dict, *, public_style_root: bool, where: object) -> None:
    payload = dict(payload)
    payload.pop("schema_version", None)
    recorded = payload.pop("fingerprint")
    spec = ConditionSpec.from_dict(payload)
    key = identity_key(spec.prompt_style, spec.task)
    # What the writer hashed: the file's own payload without its label, its style under the recorded key.
    assert canonical_digest({**{k: v for k, v in payload.items() if k != "condition_id"}, "prompt_style": key}) == (
        recorded), where
    assert "adapter_sha256" not in spec.to_dict() and payload.get("adapter_path") is None, where
    assert payload["prompt_style"] == (spec.prompt_style if public_style_root else key), where
    assert spec.prompt_style == public_style(payload["prompt_style"], spec.task), where
    assert spec.to_dict() == {**payload, "prompt_style": spec.prompt_style}, where
    assert spec.fingerprint == recorded, where
    assert spec.identity_id == payload["condition_id"], where


def test_a_condition_keeps_its_fingerprint_in_the_spelling_its_root_stores():
    rows = rows_of(read_fixture(CONDITIONS), "condition")
    earlier = next(row for row in rows if row["root"] not in PUBLIC_STYLE_ROOTS
                   and row["payload"]["prompt_style"] != public_style(row["payload"]["prompt_style"],
                                                                      row["payload"].get("task")))
    written = next(row for row in rows if row["root"] in PUBLIC_STYLE_ROOTS)
    assert (earlier["payload"]["prompt_style"], written["payload"]["prompt_style"]) == ("legacy", "naive-rules-above")
    for row, public in ((earlier, False), (written, True)):
        check_registered_condition(row["payload"], public_style_root=public, where=row["path"])
        with pytest.raises(AssertionError):
            check_registered_condition(row["payload"], public_style_root=not public, where=row["path"])


@pytest.mark.cluster
def test_every_registered_condition_keeps_its_fingerprint(real_artifact_root):
    sampling = real_artifact_root / "sampling"
    paths = validation_condition_files(real_artifact_root,
                                       lambda root: condition_files(root, lambda name: name == "hydra"))
    for path in paths:
        payload, root = json.loads(path.read_text(encoding="utf-8")), sampling / path.relative_to(sampling).parts[0]
        # A retired style's condition is held to its recorded fingerprint and directory alone.
        if not registers_retired(payload, path, root):
            check_registered_condition(payload, public_style_root=root.name in PUBLIC_STYLE_ROOTS, where=path)


def condition_files(sampling_root: Path, skip: Callable[[str], bool]) -> list[Path]:
    """Every ``condition.json`` under ``sampling_root``, sorted, descending no directory whose name ``skip`` holds."""
    return sorted(
        path for path, info in walk_strictly(sampling_root, descend=lambda directory: not skip(directory.name))
        if path.name == CONDITION_FNAME and not stat.S_ISDIR(info.st_mode)
    )


def registered_condition_paths(sampling_root: Path) -> list[Path]:
    """Every ``condition.json`` under the campaign roots, draw, snapshot and Hydra directories left out."""
    return condition_files(
        sampling_root, lambda name: name in ("hydra", "code_snapshots") or name.startswith("draw-"))


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-0 directory")
def test_the_condition_walkers_pass_over_nothing_they_cannot_read(tmp_path, monkeypatch):
    for relative in ("a/m/base", "a/m/base/draw-000", "a/hydra/run", "a/code_snapshots"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
        (tmp_path / relative / CONDITION_FNAME).write_text("{}")

    def found(paths: list[Path]) -> list[str]:
        return [path.relative_to(tmp_path).as_posix() for path in paths]

    assert found(registered_condition_paths(tmp_path)) == ["a/m/base/condition.json"]
    assert found(condition_files(tmp_path, lambda name: name == "hydra")) == [
        "a/code_snapshots/condition.json", "a/m/base/condition.json", "a/m/base/draw-000/condition.json"]
    (tmp_path / "a" / "hydra").chmod(0)
    try:
        assert found(registered_condition_paths(tmp_path)) == ["a/m/base/condition.json"]
        (tmp_path / "a" / "m").chmod(0)
        with pytest.raises(PermissionError):
            registered_condition_paths(tmp_path)
    finally:
        for name in ("hydra", "m"):
            (tmp_path / "a" / name).chmod(0o700)
    monkeypatch.setattr(os, "scandir", failing_scandir(is_dir_fails={"m", "base"}))
    assert found(registered_condition_paths(tmp_path)) == ["a/m/base/condition.json"]


def truncation_configs(payload: dict, by_directory: dict[str, list], by_name: dict[str, list]) -> list:
    """The committed local model configs a registered condition's recorded truncation resolves through."""
    if payload.get("model_weights_sha256") is not None:
        return by_name[payload["model_name"]]
    return by_directory[payload["model_path"]]


def test_a_merged_checkpoint_resolves_its_truncation_through_its_base_s_configs():
    base, fp32, qwen = ("gemma4-31b-tp-2.yaml", "cfg"), ("gemma4-31b-fp32.yaml", "cfg"), ("qwen3-8b.yaml", "cfg")
    by_directory = {"gemma-4-31B-it": [base, fp32], "Qwen3-8B": [qwen]}
    by_name = {"gemma-4-31B-it": [base, fp32], "Qwen3-8B": [qwen]}
    condition = {"model_path": "gemma-4-31B-it", "model_name": "gemma-4-31B-it", "model_weights_sha256": None}
    assert truncation_configs(condition, by_directory, by_name) == [base, fp32]
    merged = {"model_path": "checkpoint-600", "model_name": "gemma-4-31B-it", "model_weights_sha256": "03a8" * 16}
    assert truncation_configs(merged, by_directory, by_name) == [base, fp32]
    with pytest.raises(KeyError, match="checkpoint-600"):
        truncation_configs({**merged, "model_weights_sha256": None}, by_directory, by_name)
    with pytest.raises(KeyError, match="Olmo-3-32B-Think"):
        truncation_configs({**merged, "model_name": "Olmo-3-32B-Think"}, by_directory, by_name)


@pytest.mark.cluster
def test_every_registered_local_condition_resolves_its_truncation_as_recorded(real_artifact_root, monkeypatch):
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", REAL_MODEL_ROOT)
    configs: dict[str, list] = {}
    by_name: dict[str, list] = {}
    for path in sorted((REPO_ROOT / "configs" / "model").glob("*.yaml")):
        cfg = OmegaConf.load(path)
        if cfg.get("model_type") == "local":
            configs.setdefault(Path(str(cfg.model_path)).name, []).append((path.name, cfg))
            by_name.setdefault(str(cfg.model_name), []).append((path.name, cfg))
    paths = validation_condition_files(real_artifact_root, registered_condition_paths)
    resolved: dict[tuple[str, str, str], dict] = {}
    checked, fingerprinted = set(), 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded = payload.get("sampling_truncation")
        if not isinstance(recorded, dict):
            continue
        root = path.parents[len(Path(payload["condition_id"]).parts)]
        policy = json.loads((root / "sampling_manifest.json").read_text(encoding="utf-8"))["truncation"]
        fields = {k: v for k, v in payload.items() if k not in ("schema_version", "fingerprint")}
        for name, cfg in truncation_configs(payload, configs, by_name):
            key = (name, payload["model_path"], policy)
            if key not in resolved:
                resolved[key] = campaign_module.resolve_sampling_truncation(
                    SimpleNamespace(truncation=policy), "local", cfg
                )
            assert json.dumps(resolved[key], sort_keys=True) == json.dumps(recorded, sort_keys=True), (path, name)
            checked.add(name)
            if is_retired(fields["prompt_style"], fields.get("task")):  # a condition read from its recorded fields
                assert recorded_fingerprint({**fields, "sampling_truncation": resolved[key]}) == payload["fingerprint"]
                fingerprinted += 1
                continue
            spec = ConditionSpec.from_dict({**fields, "sampling_truncation": resolved[key]})
            assert spec.to_dict() == {**fields, "prompt_style": public_style(fields["prompt_style"], spec.task)}, (
                path, name)
            assert spec.fingerprint == payload["fingerprint"], (path, name)
            fingerprinted += 1
    assert {"gemma4-31b-tp-2.yaml", "gpt-oss-tp-2.yaml", "olmo3-32b-tp-2.yaml", "qwen3-8b.yaml",
            "qwen3-8b-fp32.yaml"} <= checked
    assert fingerprinted


@pytest.mark.cluster
def test_the_template_passes_preflight_on_the_real_paper_population(real_artifact_root, tmp_path, monkeypatch,
                                                                     in_job_dir):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(real_artifact_root))
    cfg = compose_training("base", "training.family=base", f"campaign_root={tmp_path / 'root'}")
    config, manifest = preflight(cfg)
    assert (manifest.source_sha256, manifest.manifest_sha256) == (PAPER_POPULATION_SHA256,
                                                                    PAPER_POPULATION_MANIFEST_SHA256)
    assert manifest.record_count == 500 and config.forced_json_requirement == "label_only"
    assert campaign_module.judge_adapter(cfg.judge_model) is None
    assert not (tmp_path / "root").exists()
