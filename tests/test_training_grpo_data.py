from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import types
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from ruleforge.experiments.training.pins import PINNED_PROTECTED
from ruleforge.grpo import grpo_data as gd
from ruleforge.grpo.anchor_set import (
    ANCHORS_FNAME,
    ANCHORS_MANIFEST_FNAME,
    ANCHORS_SCHEMA_VERSION_STAGED,
    MIXED_SELECTION_KIND,
    REPLAY_INDEX_FNAME,
    SIDE_DEV,
    SOURCE_BANK,
    SOURCE_TOP_UP,
    STAGE_CALIBRATION,
    STAGE_CONFIRM,
    STAGE_YES_SIDE,
    Ruleset,
    read_anchor_set,
)
from ruleforge.grpo.leakage import LeakageError, disjoint_marker_path
from ruleforge.grpo.rewards import RowColumns
from ruleforge.judges.completion_format import CompletionFormat
from ruleforge.evaluate.campaign import load_conditions, register_condition
from ruleforge.evaluate.identity import FULL_RULESET_COMPONENT
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.runner import DrawRunner
from ruleforge.core.artifacts import prompt_sha256
from ruleforge.core.artifacts import compute_scenario_id, iter_jsonl_records, write_jsonl_records
from tests.test_training_anchors import (
    BANK_SCRIPT,
    FULL_SCRIPT,
    TOP_UP_SCRIPT,
    build,
    run_blocks,
    run_condition,
    run_top_up,
)
from tests.test_sampling_runner import RULES, FakeBackend, make_spec, restamp_stacks

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_POPULATION_SHA256 = "c16f66f4ad3fc93fad46bf31efaa8325f30fb750104d1e13a9326b7cd46b10d4"
RULES_57 = REPO_ROOT / "data" / "rules" / "modelspec" / "openai_rules_all.jsonl"
#: Qwen3-8B's own directory, resolved before any test points RULEFORGE_MODEL_ROOT elsewhere.
QWEN3_8B = Path(os.environ.get("RULEFORGE_MODEL_ROOT") or Path.home() / "models") / "Qwen3-8B"
#: The artifact root, likewise (the cluster test reads the frozen 500 from it, read only).
FROZEN_500 = Path(os.environ.get("RULEFORGE_ARTIFACT_ROOT") or "/nonexistent") / "populations" / "frozen_500.jsonl"
needs_qwen3 = pytest.mark.skipif(not (QWEN3_8B / "tokenizer.json").is_file(), reason="Qwen3-8B's tokenizer is absent")
GEMMA_31B = QWEN3_8B.parent / "gemma-4-31B-it"
needs_gemma = pytest.mark.skipif(not (GEMMA_31B / "tokenizer.json").is_file(),
                                 reason="gemma-4-31B-it's tokenizer is absent")
#: The directory the fake campaigns' judge names (the model config's), where the stand-in tokenizer goes.
JUDGE_DIR = "gemma-4-31B-it"
N = 4

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
THINK_TOKENS = ["<think>", "</think>"]
#: Qwen3's template in miniature: system and user turns, the generation prompt, the thinking switch.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false -%}{{ '<think>\\n\\n</think>\\n\\n' }}{%- endif -%}"
    "{%- endif -%}"
)


#: The stand-in's ids: 256 bytes, then SPECIALS (256-258), then THINK_TOKENS (259-260).
EOS, ENDOFTEXT, THINK_OPEN_ID = 258, 256, 259


#: The files a build writes beside its manifest, each with its disjointness marker.
OUTPUT_FNAMES = (gd.SCENARIOS_FNAME, gd.GRPO_ROWS_FNAME, gd.REPLAY_ROWS_FNAME)


def read_grpo_data(directory: Path) -> dict:
    """A build's manifest, checked to record every output and marker as it is on disk."""
    manifest = json.loads((directory / gd.MANIFEST_FNAME).read_text(encoding="utf-8"))
    for name in OUTPUT_FNAMES:
        assert sha256_file(directory / name) == manifest["outputs"][name]["sha256"], name
        assert sha256_file(disjoint_marker_path(directory / name)) == manifest["markers"][name], name
    return manifest


def save_byte_tokenizer(directory: Path) -> Path:
    """Every byte a token, Qwen3's special and thinking tokens added: exact ids for any text."""
    import transformers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab={char: index for index, char in enumerate(alphabet)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIALS)
    tok.add_tokens(THINK_TOKENS)
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>", chat_template=CHAT_TEMPLATE,
    ).save_pretrained(str(directory))
    (directory / gd.GENERATION_CONFIG_FNAME).write_text(json.dumps({"eos_token_id": [EOS, ENDOFTEXT]}))
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    return directory


def task_spec_factory(**eval_overrides):
    """A ModelSpecTaskSpec as the conftest's ``modelspec`` fixture builds one (usable at module scope)."""
    from omegaconf import OmegaConf

    from ruleforge.tasks.spec import ModelSpecTaskSpec

    return ModelSpecTaskSpec(OmegaConf.create({
        "task_spec": "modelspec",
        "rule_amendment": {"initial_rule_set_path": "unused.jsonl", "intent_weight": 0.0},
        "scenario_generation": {"good_seed_scenario_path": "unused.jsonl", "bad_seed_scenario_path": "unused.jsonl"},
        "agent_model": {},
        "judge_model": {},
        "eval_cfg": {"judge_types": ["No_Interpretation"], "evaluation_temperature": 0.1, **eval_overrides},
    }))


def population_rows(n: int = N) -> list[dict]:
    """Prompts of growing length (the fake judge reads each scenario's index as its prompt's last word)."""
    return [{"prompt": f"prompt {'pad ' * (40 * i)}{i}", "response": f"response {i}", "source": ("wildchat", "stress")[i % 2]}
            for i in range(n)]


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class MarkersDouble:
    """Stands in for the prompt-marker functions: a marker must name the dataset's current bytes."""

    def __init__(self, *, leak: bool = False):
        self.leak = leak
        self.attested: list[tuple[Path, list[Path]]] = []
        self.checked: list[tuple[str, list[str], list[str]]] = []

    def check(self, dataset, required_protected=(), required_carved=()):
        self.checked.append((Path(dataset).name, list(required_protected), list(required_carved)))
        path = disjoint_marker_path(Path(dataset))
        if not path.is_file():
            raise LeakageError(f"{dataset} has no prompt-disjointness marker")
        marker = json.loads(path.read_text())
        if marker.get("kind") != "prompt_disjoint" or marker["dataset"]["sha256"] != sha256_file(dataset):
            raise LeakageError(f"{path} does not attest {dataset}'s bytes")
        uncovered = sorted(set(required_protected) - {entry["sha256"] for entry in marker["protected"]})
        if uncovered:
            raise LeakageError(f"{path} was not checked against the required protected file(s) {uncovered}")
        read = {entry["sha256"] for key in ("protected", "carved") for entry in marker.get(key, ())}
        if unread := sorted(set(required_carved) - read):  # as check_prompt_marker(required_carved=...)
            raise LeakageError(f"{path} was not checked against the required file(s) {unread}, not even by the second pass")
        return marker

    def checker(self, dataset, marker_path, required_protected=(), required_carved=()):
        return self.check(dataset, required_protected, required_carved)

    def attest(self, scenarios, source, extra_protected):
        if self.leak:
            raise LeakageError("the second pass found a planted leak")
        self.attested.append((Path(scenarios), list(extra_protected)))
        rows = read_jsonl(scenarios)
        extra = [{"path": str(path), "sha256": sha256_file(path), "rows": len(read_jsonl(path))} for path in extra_protected]
        return {
            "kind": "prompt_disjoint", "schema_version": 1, "identity": "normalized_prompt", "disjoint": True,
            "policy": source["policy"],
            "dataset": {"name": Path(scenarios).name, "sha256": sha256_file(scenarios), "rows": len(rows),
                        "prompt_keys": len(rows), "prompt_keys_sha256": "0" * 64},
            "protected": list(source["protected"]), "carved": [*source.get("carved", ()), *extra],
        }

    def write(self, dataset, marker):
        path = disjoint_marker_path(Path(dataset))
        path.write_text(json.dumps(marker, indent=2))
        return path


def mark(path: Path, *, protected: Path, also=()) -> Path:
    """Give a scenario file the marker the population build would, against ``protected`` and ``also``."""
    return MarkersDouble().write(path, {
        "kind": "prompt_disjoint", "policy": {"shingle_n": 5},
        "dataset": {"name": path.name, "sha256": sha256_file(path), "rows": len(read_jsonl(path))},
        "protected": [{"path": str(file), "sha256": sha256_file(file), "rows": len(read_jsonl(file))}
                      for file in (protected, *also)],
    })


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The anchored campaigns, built once: a K = 2 bank, a full-set condition, a top-up of the two routed rows."""
    base = tmp_path_factory.mktemp("grpo_data")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RULEFORGE_ARTIFACT_ROOT", str(base / "artifacts"))
        patch.setenv("RULEFORGE_MODEL_ROOT", str(base / "models"))
        population = write_jsonl(base / "population.jsonl", population_rows())
        root = base / "anchors-train"
        common = dict(root=root, campaign_id="anch", population=population)
        run_condition(base, task_spec_factory, mode="individual_bank", script=BANK_SCRIPT, **common)
        run_condition(base, task_spec_factory, mode="full", script=FULL_SCRIPT, **common)
        bank = load_conditions(root)["gemma/individual_bank"]
        reference = register_condition(
            base / "training",
            dataclasses.replace(bank, condition_id="gemma/figure5/base", evaluation_mode="both",
                                extra=(("figure", "figure5"), ("policy", "base"))),
        )
        campaign = {"root": root, "reference": reference, "rules": base / "rules.jsonl", "tmp": base}
        top_up = run_top_up(campaign, task_spec_factory)
        build(campaign, top_up=top_up)
    protected = write_jsonl(base / "frozen.jsonl", [{"prompt": "a protected prompt", "response": "r"}])
    mark(population, protected=protected)
    tokenizer = save_byte_tokenizer(base / "models" / JUDGE_DIR)
    return types.SimpleNamespace(
        base=base, population=population, rules=base / "rules.jsonl", anchors=base / "anchor-set", root=root,
        top_up=top_up, reference=reference, model=tokenizer, protected=protected,
    )


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path, world):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(world.base / "models"))


def pinned_stand_in(world) -> dict[str, str]:
    """The pinned protected files, as the fake world stands them in: its protected file for frozen_500."""
    return {"frozen_500": sha256_file(world.protected)}


def build_into(world, out, **overrides):
    """A build over the whole population, which holds nothing out (declared); the held-out tests override it."""
    kwargs = dict(anchors=world.anchors, scenarios=world.population, rules=world.rules, model=world.model, out=out,
                  prompt_draws=4, class_a_share_cap=0.75, markers=MarkersDouble(), allow_no_held_out=True,
                  pinned_protected=pinned_stand_in(world))
    kwargs.update(overrides)
    return gd.build_grpo_data(**kwargs)


@pytest.fixture(scope="module")
def built(world, tmp_path_factory):
    """One build over the whole population (every prompt under the cap)."""
    out = tmp_path_factory.mktemp("built") / "grpo-data"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RULEFORGE_MODEL_ROOT", str(world.base / "models"))
        manifest = build_into(world, out)
    return types.SimpleNamespace(out=out, manifest=manifest, grpo=read_jsonl(out / gd.GRPO_ROWS_FNAME),
                                 replay=read_jsonl(out / gd.REPLAY_ROWS_FNAME))


def tokenizer_of(world):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(world.model))


def draw_runner(tmp_path, rows, rules, *, context="isolated") -> DrawRunner:
    """The campaign's own prompt builder over ``rows``: the training forced style, full set and every rule."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = write_jsonl(tmp_path / f"runner-{context}.jsonl", rows)
    manifest = ScenarioManifest.build(path)
    spec = make_spec(manifest, style="forced-grpo")
    spec = dataclasses.replace(spec, condition=dataclasses.replace(spec.condition, individual_rule_context=context))
    return DrawRunner(
        backend=FakeBackend(style="forced-grpo"),
        task_spec=task_spec_factory(reasoning_prompt_style="forced-grpo", evaluation_temperature=1.0,
                                    individual_rule_context=context),
        spec=spec,
        draw_dir=tmp_path / "camp" / context / "draw-000",
        scenario_rows=rows,
        rules=rules,
        code_revision="rev",
    )


def bank_condition(world):
    return load_conditions(world.root)["gemma/individual_bank"]


@pytest.mark.parametrize("context", ["isolated", "implicated"])
def test_the_prompts_are_the_draw_runners_text_for_text(tmp_path, context):
    rows = population_rows(3)
    runner = draw_runner(tmp_path, rows, RULES, context=context)
    condition = runner.spec.condition
    prompts = gd.CampaignPrompts.build(task_spec=gd.condition_task_spec(condition), rules=RULES, judge_type="No_Interpretation")
    for row in rows:
        bundle = runner.bundle_prompts(compute_scenario_id(row))
        assert [component for component, _ in bundle] == [FULL_RULESET_COMPONENT, "Rule 1", "Rule 2"]
        assert bundle == [(component, prompts.entry(row, component)) for component, _ in bundle]
    assert "Meta-Rule 1: meta body" in prompts.rule_set_string and set(prompts.variants) == {"Rule 1", "Rule 2"}


def _check_rendering(tokenizer, runner, *, enable_thinking=True, trl_tokenizer=None):
    kwargs = gd.chat_template_kwargs(enable_thinking)
    trl_tokenizer = tokenizer if trl_tokenizer is None else trl_tokenizer
    checked = 0
    for sid in runner.scenarios:
        entries = [entry for _, entry in runner.bundle_prompts(sid)]
        campaign = gd.campaign_prompt_ids(tokenizer, entries, enable_thinking=enable_thinking)
        assert gd.trl_prompt_ids(trl_tokenizer, [gd.conversation(entry) for entry in entries], kwargs) == campaign
        checked += len(entries)
    return checked


def test_trl_renders_the_grpo_and_replay_prompts_as_the_campaign_sends_them(tmp_path, world):
    tokenizer = tokenizer_of(world)
    runner = draw_runner(tmp_path, population_rows(3), RULES)
    assert _check_rendering(tokenizer, runner) == 9
    # the conversation is [system: P, user: ""], rendered up to the generation prompt, which leaves <think> to the model
    entry = runner.bundle_prompts(next(iter(runner.scenarios)))[0][1]
    ids = gd.campaign_prompt_ids(tokenizer, [entry], enable_thinking=True)[0]
    assert tokenizer.decode(ids).endswith("<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n")
    assert gd.conversation(entry) == [{"role": "system", "content": entry["system_prompt"]}, {"role": "user", "content": ""}]
    assert gd.chat_template_kwargs(False) == {"enable_thinking": False}
    assert _check_rendering(tokenizer, runner, enable_thinking=False) == 9
    off = gd.campaign_prompt_ids(tokenizer, [entry], enable_thinking=False)[0]
    assert tokenizer.decode(off).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert gd.campaign_prompt_ids(tokenizer, [], enable_thinking=True) == gd.trl_prompt_ids(tokenizer, [], {}) == []


TRICKY = [
    {"prompt": "Plain English prompt?", "response": "A plain answer."},
    {"prompt": "  leading and trailing spaces  \n\n", "response": "\ttabbed\r\nwindows lines\n"},
    {"prompt": "สวัสดีครับ नमस्ते مَرْحَبًا 你好 👨‍👩‍👧 café naïve", "response": "❤️ 👍🏽 — “quotes” … ends in a space "},
    {"prompt": "Literal markup: <|im_end|> <think> </think> {\"full_set_judgment\": \"yes\"}", "response": "```json\n{}\n```"},
]


@needs_qwen3
def test_qwen3_8b_renders_the_grpo_and_replay_prompts_as_the_campaign_sends_them(tmp_path):
    tokenizer = gd.load_tokenizer(QWEN3_8B, model_path="Qwen3-8B")
    rules = [json.loads(line) for line in RULES_57.read_text().splitlines()]
    runner = draw_runner(tmp_path, TRICKY, rules)
    assert _check_rendering(tokenizer, runner) == len(TRICKY) * 58
    entry = runner.bundle_prompts(next(iter(runner.scenarios)))[1][1]
    prompt = gd.campaign_prompt_ids(tokenizer, [entry], enable_thinking=True)[0]
    assert prompt[-3:] == tokenizer.encode("<|im_start|>assistant\n")
    stops = gd.stop_token_ids(QWEN3_8B)
    assert tokenizer.eos_token_id == 151645 and stops == [151645, 151643]
    encoder = gd.completion_encoder(tokenizer, stops)
    completion = ('<think>\nThe user wrote <|im_end|> and <|endoftext|>; <|im_start|> may be emitted.\n</think>\n\n'
                  '{"full_set_judgment": "no"}')
    ids = gd.completion_ids(encoder, [completion])[0]
    assert ids[0] == 151667 and 151668 in ids and 151644 in ids and tokenizer.decode(ids) == completion
    assert not {151645, 151643} & set(ids)  # the quoted stop tokens stay text
    plain = TRICKY[2]["prompt"] + TRICKY[3]["response"]
    assert gd.completion_ids(encoder, [plain]) == [tokenizer(plain, add_special_tokens=False)["input_ids"]]
    stop_texts, others = gd.special_texts(tokenizer, stops)
    assert stop_texts == ["<|endoftext|>", "<|im_end|>"] and "<|im_start|>" in others and "<think>" not in others


@needs_gemma
def test_gemma_4_31b_renders_the_grpo_and_replay_prompts_as_the_campaign_sends_them(tmp_path, monkeypatch):
    from ruleforge.judges.completion_format import load_training_format

    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(GEMMA_31B.parent))
    tokenizer = gd.load_tokenizer(GEMMA_31B, model_path="gemma-4-31B-it")
    training, completion = load_training_format(GEMMA_31B)
    rules = [json.loads(line) for line in RULES_57.read_text().splitlines()]
    runner = draw_runner(tmp_path, TRICKY, rules)
    assert _check_rendering(tokenizer, runner, trl_tokenizer=training) == len(TRICKY) * 58
    entry = runner.bundle_prompts(next(iter(runner.scenarios)))[1][1]
    prompt = gd.campaign_prompt_ids(tokenizer, [entry], enable_thinking=True)[0]
    assert tokenizer.decode(prompt).endswith("<|turn>model\n") and completion.think_end_token_id not in prompt
    assert (tokenizer.eos_token_id, training.eos_token_id, completion.finish_token_ids[0]) == (1, 106, 106)
    stops = gd.stop_token_ids(GEMMA_31B)
    assert stops == [1, 106, 50]
    encoder = gd.completion_encoder(tokenizer, stops)
    text = '<|channel>thought\nThe user quoted <turn|> and <eos>.<channel|>{"full_set_judgment": "no"}'
    ids = gd.completion_ids(encoder, [text])[0]
    assert ids[0] == tokenizer.convert_tokens_to_ids("<|channel>") and completion.think_end_token_id in ids
    assert not set(stops) & set(ids) and tokenizer.decode(ids) == text


@pytest.mark.cluster
@needs_qwen3
def test_qwen3_8b_renders_the_frozen_500_as_the_campaign_sends_them(tmp_path):
    """Real scenarios: all 500 full-set prompts and every rule of the first 20 (read only)."""
    rows = read_jsonl(FROZEN_500)
    tokenizer = gd.load_tokenizer(QWEN3_8B, model_path="Qwen3-8B")
    rules = [json.loads(line) for line in RULES_57.read_text().splitlines()]
    runner = draw_runner(tmp_path, rows, rules)
    entries = [runner.bundle_prompts(sid)[0][1] for sid in runner.scenarios]
    campaign = gd.campaign_prompt_ids(tokenizer, entries, enable_thinking=True)
    trl = gd.trl_prompt_ids(tokenizer, [gd.conversation(entry) for entry in entries], {"enable_thinking": True})
    assert len(campaign) == 500 and trl == campaign and max(map(len, campaign)) <= gd.PROMPT_TOKEN_CAP
    subset = draw_runner(tmp_path / "subset", rows[:20], rules)
    assert _check_rendering(tokenizer, subset) == 20 * 58


def test_the_tokenizer_must_be_the_judges_and_end_turns(tmp_path, world, monkeypatch):
    assert gd.load_tokenizer(world.model, model_path=f"/elsewhere/{JUDGE_DIR}").eos_token_id == EOS
    with pytest.raises(gd.GrpoDataError, match="the anchors were drawn on 'Qwen3-8B'"):
        gd.load_tokenizer(world.model, model_path="Qwen3-8B")
    import transformers

    other = tmp_path / "other" / JUDGE_DIR
    shutil.copytree(world.model, other)
    assert gd.load_tokenizer(other, model_path=JUDGE_DIR).eos_token_id == EOS
    drifted = transformers.AutoTokenizer.from_pretrained(str(other))
    drifted.chat_template = CHAT_TEMPLATE.replace("assistant\\n", "assistant\\n\\n")
    drifted.save_pretrained(str(other))
    with pytest.raises(gd.GrpoDataError, match="the 'gemma-4-31B-it' a campaign's model config resolves, differ in"):
        gd.load_tokenizer(other, model_path=JUDGE_DIR)
    monkeypatch.delenv("RULEFORGE_MODEL_ROOT")
    assert gd.campaign_model_dir(JUDGE_DIR) is None and gd.load_tokenizer(other, model_path=JUDGE_DIR).eos_token_id == EOS

    no_eos = tmp_path / JUDGE_DIR
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(world.model))
    tokenizer.eos_token = None
    tokenizer.save_pretrained(str(no_eos))
    with pytest.raises(gd.GrpoDataError, match="no eos token"):
        gd.load_tokenizer(no_eos, model_path=JUDGE_DIR)
    stops = tmp_path / "stops" / JUDGE_DIR
    shutil.copytree(world.model, stops)
    config = stops / gd.GENERATION_CONFIG_FNAME
    for payload, accepted in (({"eos_token_id": EOS}, True), ({"eos_token_id": ENDOFTEXT}, False), ({}, False)):
        config.write_text(json.dumps(payload))
        if accepted:
            assert gd.load_tokenizer(stops, model_path=JUDGE_DIR).eos_token_id == EOS
        else:
            with pytest.raises(gd.GrpoDataError, match="is not among the ids its generation_config.json stops on"):
                gd.load_tokenizer(stops, model_path=JUDGE_DIR)
    config.unlink()
    with pytest.raises(gd.GrpoDataError, match="has no generation_config.json"):
        gd.load_tokenizer(stops, model_path=JUDGE_DIR)

    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(world.base / "models"))
    templated = tokenizer_of(world)
    record = gd.tokenizer_record(world.model, templated)
    assert record["files"]["tokenizer.json"] == sha256_file(world.model / "tokenizer.json")
    assert record["chat_template_sha256"] and record["eos_token_id"] == EOS and record["stop_token_ids"] == [EOS, ENDOFTEXT]
    assert record["campaign_model_dir"] == str(world.model)
    templated.chat_template = None
    assert gd.tokenizer_record(world.model, templated)["chat_template_sha256"] is None


def test_a_stop_tokens_text_in_a_completion_stays_text(world):
    tokenizer = tokenizer_of(world)
    encoder = gd.completion_encoder(tokenizer, [EOS, ENDOFTEXT])
    texts = ["<think>\nThe user wrote <|im_end|> and <|endoftext|>; <|im_start|> may be emitted.\n</think>\n\n{}", "plain"]
    ids = gd.completion_ids(encoder, texts)
    assert [tokenizer.decode(row) for row in ids] == texts
    assert ids[0][0] == THINK_OPEN_ID and 257 in ids[0] and not {EOS, ENDOFTEXT} & set(ids[0])
    assert EOS in tokenizer(texts[0], add_special_tokens=False)["input_ids"]
    assert ids[1] == tokenizer(texts[1], add_special_tokens=False)["input_ids"] and gd.completion_ids(encoder, []) == []
    assert gd.special_texts(tokenizer, [EOS, ENDOFTEXT]) == (["<|endoftext|>", "<|im_end|>"], ["<|im_start|>"])
    assert EOS in tokenizer(texts[0], add_special_tokens=False)["input_ids"]


def _classes(a, b, c):
    return {**{f"a{i}": "a" for i in range(a)}, **{f"b{i}": "b" for i in range(b)}, **{f"c{i}": "c" for i in range(c)}}


def _drawn(weights, classes, name):
    return sum(copies for sid, copies in weights.copies.items() if classes[sid] == name)


def _prefix_gap(weights, classes):
    rows = len(weights.order)
    draws = Counter(classes[sid] for sid in weights.order)
    placed, worst = Counter(), Fraction(0)
    for position, sid in enumerate(weights.order, start=1):
        placed[classes[sid]] += 1
        worst = max(worst, *(abs(placed[name] - Fraction(position * draws[name], rows)) for name in draws))
    return worst


def test_the_final_train_anchor_set_gets_the_authors_rule():
    """a 35, b 2,789, c 176 over D = 2,400: (a) 8 x 35 = 280 draws (8 each), (b) and (c) the other 2,120 by count."""
    classes = _classes(35, 2789, 176)
    weights = gd.sampling_weights(classes)
    declaration = weights.declaration
    # 2,120 x 2,789 / 2,965 = 1,994.16 and 2,120 x 176 / 2,965 = 125.84: (c) has the larger remainder
    assert declaration["draws"] == {"a": 280, "b": 1994, "c": 126} == {name: _drawn(weights, classes, name) for name in "abc"}
    assert declaration["target_draws"] == {"a": 280.0, "b": 2120 * 2789 / 2965, "c": 2120 * 176 / 2965}
    assert declaration["target_shares"]["a"] == 280 / 2400 == declaration["realized_shares"]["a"]
    assert declaration["realized_shares"] == {"a": 280 / 2400, "b": 1994 / 2400, "c": 126 / 2400}
    assert {weights.copies[f"a{i}"] for i in range(35)} == {8}
    assert declaration["copies"] == {
        "a": {"min": 8, "max": 8, "scenarios_by_copies": {"8": 35}},
        "b": {"min": 0, "max": 1, "scenarios_by_copies": {"0": 795, "1": 1994}},
        "c": {"min": 0, "max": 1, "scenarios_by_copies": {"0": 50, "1": 126}},
    }
    assert declaration["scenarios_not_drawn"] == {"a": 0, "b": 795, "c": 50}
    assert declaration["scenarios_not_drawn_b_c"] == 845 and declaration["scenarios_drawn"] == {"a": 35, "b": 1994, "c": 126}
    assert (declaration["counts"], declaration["seed"], declaration["prompt_draws"]) == ({"a": 35, "b": 2789, "c": 176}, 0, 2400)
    # the file is exactly the D draws: each scenario as many rows as its copies, every prefix within a row of its share
    assert len(weights.order) == declaration["rows"] == 2400 and Counter(weights.order) == +Counter(weights.copies)
    assert _prefix_gap(weights, classes) < 1 and declaration["max_draws_per_scenario"] == 8
    assert declaration["consumption"].startswith("the rows are the run's D draws")


def test_the_always_yes_remedy_is_a_no_op_under_the_repeat_cap_of_8():
    classes = _classes(35, 2789, 176)
    first = gd.sampling_weights(classes)
    remedy = gd.sampling_weights(classes, class_a_share_cap=0.25)
    assert remedy.order == first.order and remedy.copies == first.copies
    assert remedy.declaration["draws"] == {"a": 280, "b": 1994, "c": 126}
    assert remedy.declaration["realized_shares"]["a"] == 280 / 2400


def test_the_always_yes_remedy_at_25_percent_needs_a_repeat_cap_of_18():
    classes = _classes(35, 2789, 176)
    weights = gd.sampling_weights(classes, class_a_share_cap=0.25, repeat_cap=18)
    declaration = weights.declaration
    assert declaration["draws"] == {"a": 600, "b": 1693, "c": 107}
    assert declaration["realized_shares"]["a"] == 0.25
    assert declaration["copies"]["a"] == {"min": 17, "max": 18, "scenarios_by_copies": {"17": 30, "18": 5}}
    assert declaration["scenarios_not_drawn"] == {"a": 0, "b": 1096, "c": 69}
    assert declaration["max_draws_per_scenario"] == 18 and declaration["repeat_cap"] == 18
    assert len(weights.order) == 2400 and Counter(weights.order) == +Counter(weights.copies)
    assert _prefix_gap(weights, classes) < 1
    # one copy short of 18 falls under the share: 17 x 35 = 595 draws
    short = gd.sampling_weights(classes, class_a_share_cap=0.25, repeat_cap=17)
    assert short.declaration["draws"]["a"] == 595


def test_the_draws_are_deterministic_and_the_seed_picks_the_subset_not_the_counts():
    classes = _classes(35, 2789, 176)
    first, again = gd.sampling_weights(classes), gd.sampling_weights(classes)
    assert (first.order, first.copies, first.declaration) == (again.order, again.copies, again.declaration)
    other = gd.sampling_weights(classes, seed=1)
    assert other.declaration["draws"] == first.declaration["draws"] and other.declaration["seed"] == 1
    assert {sid for sid, n in other.copies.items() if n} != {sid for sid, n in first.copies.items() if n}
    assert other.order != first.order


def test_class_a_at_the_share_cap_spreads_its_draws_evenly_under_the_repeat_cap():
    """50 class (a): 8 x 50 = 400 > 15% of 2,400 = 360, so 7 or 8 draws each."""
    classes = _classes(50, 2000, 950)
    weights = gd.sampling_weights(classes)
    assert weights.declaration["draws"]["a"] == 360 and weights.declaration["target_shares"]["a"] == 0.15
    assert weights.declaration["copies"]["a"] == {"min": 7, "max": 8, "scenarios_by_copies": {"7": 40, "8": 10}}
    diluted = gd.sampling_weights(_classes(1000, 1500, 600))  # (b), (c): 2,040 x 15/21 = 1,457.1, x 6/21 = 582.9
    assert diluted.declaration["copies"]["a"]["scenarios_by_copies"] == {"0": 640, "1": 360}
    assert diluted.declaration["scenarios_not_drawn"]["a"] == 640
    assert diluted.declaration["draws"] == {"a": 360, "b": 1457, "c": 583}


def test_the_extra_copy_goes_to_the_first_of_the_seeded_shuffle_and_the_rows_are_passes_over_it():
    classes = {"a2": "a", "a0": "a", "a1": "a", **_classes(0, 0, 197)}
    weights = gd.sampling_weights(classes, prompt_draws=150)  # (a): min(0.15 x 150, 8 x 3) = 22 draws: 8, 7, 7
    rng, shuffled = random.Random(0), ["a2", "a0", "a1"]  # the draws' own sequence: the tie-break, then (a)
    rng.shuffle(["b", "c"])
    rng.shuffle(shuffled)
    assert {sid: n for sid, n in weights.copies.items() if sid.startswith("a")} == dict(zip(shuffled, (8, 7, 7)))
    assert [sid for sid in weights.order if sid.startswith("a")] == shuffled * 7 + shuffled[:1]
    assert sorted(n for sid, n in weights.copies.items() if sid.startswith("c")) == [0] * 69 + [1] * 128
    assert weights.declaration["draws"] == {"a": 22, "b": 0, "c": 128}
    assert "b" not in weights.declaration["copies"] and weights.declaration["counts"]["b"] == 0


def test_an_equal_remainder_goes_by_the_seeded_tie_break():
    """One (b) and one (c) for one draw: 1/2 each, and the seeded order decides; each seed always the same one."""
    classes = _classes(0, 1, 1)
    picked = {}
    for seed in range(8):
        weights = gd.sampling_weights(classes, prompt_draws=1, seed=seed)
        assert weights.declaration["draws"][weights.declaration["tie_break"][0]] == 1
        assert gd.sampling_weights(classes, prompt_draws=1, seed=seed).order == weights.order
        picked[seed] = weights.order
    assert {tuple(order) for order in picked.values()} == {("b0",), ("c0",)}
    assert gd.class_draws({"b": 1, "c": 1}, prompt_draws=1, repeat_cap=8, class_a_share_cap=0.15,
                          tie_break=("c", "b"))[0] == {"a": 0, "b": 0, "c": 1}


def test_the_rule_holds_over_many_mixes():
    rng = random.Random(98)
    feasible = 0
    for _ in range(300):
        counts = {"a": rng.randint(0, 60), "b": rng.randint(0, 300), "c": rng.randint(0, 300) if rng.random() > 0.2 else 0}
        draws = rng.choice([1, 50, 120, 400])
        classes = _classes(counts["a"], counts["b"], counts["c"])
        if not classes:
            continue
        try:
            weights = gd.sampling_weights(classes, prompt_draws=draws, seed=rng.randint(0, 9))
        except gd.GrpoDataError as error:
            assert "each scenario at most once" in str(error)
            continue
        feasible += 1
        got = {name: _drawn(weights, classes, name) for name in "abc"}
        share_a = min(Fraction(3, 20), Fraction(8 * counts["a"], draws))
        assert got["a"] == math.floor(draws * share_a) and sum(got.values()) == draws == len(weights.order)
        rest, pool = draws - got["a"], counts["b"] + counts["c"]
        for name in "bc":
            if counts[name]:
                assert abs(got[name] - Fraction(rest * counts[name], pool)) < 1
        assert all(weights.copies[sid] <= (8 if cls == "a" else 1) for sid, cls in classes.items())
        values = [weights.copies[sid] for sid, cls in classes.items() if cls == "a"]
        assert not values or max(values) - min(values) <= 1
        assert Counter(weights.order) == +Counter(weights.copies) and _prefix_gap(weights, classes) < 1
    assert feasible > 150


def test_the_interleave_places_the_class_furthest_behind():
    assert gd.interleave({"a": ["a0", "a0"], "b": ["b0", "b1", "b2", "b3"], "c": []}) == [
        "b0", "a0", "b1", "b2", "a0", "b3"]
    assert gd.interleave({"a": [], "b": []}) == []


def test_impossible_draws_are_refused():
    # (a) 1 of 8 draws at 15%: (b) and (c) would need 7 draws from 2 scenarios, each at most once
    with pytest.raises(gd.GrpoDataError, match=r"class \(a\) \(1 scenario\(s\)\) takes 1 of the 8 prompt draws; the "
                                               r"other 7 .* each scenario at most once, but they hold 2"):
        gd.sampling_weights(_classes(1, 1, 1), prompt_draws=8)
    with pytest.raises(gd.GrpoDataError, match="the other 9 .* but they hold 0"):
        gd.sampling_weights(_classes(1, 0, 0), prompt_draws=10)  # (a) alone: 1 draw of 10 at 15%
    # class (a) alone fills every draw at a share cap of 1 and 8 copies each
    alone = gd.sampling_weights(_classes(3, 0, 0), prompt_draws=24, class_a_share_cap=1.0)
    assert alone.declaration["draws"] == {"a": 24, "b": 0, "c": 0} and set(alone.copies.values()) == {8}
    with pytest.raises(gd.GrpoDataError, match="no training scenario"):
        gd.sampling_weights({})
    with pytest.raises(gd.GrpoDataError, match="unknown anchor classes"):
        gd.sampling_weights({"s": "d"})
    for bad in (dict(prompt_draws=0), dict(repeat_cap=0), dict(class_a_share_cap=0.0), dict(class_a_share_cap=1.5)):
        with pytest.raises(gd.GrpoDataError, match="must be positive"):
            gd.sampling_weights(_classes(1, 1, 1), **bad)


@pytest.mark.torch
def test_trls_sampler_walks_one_permutation_of_the_d_rows_in_the_runs_600_steps():
    pytest.importorskip("trl")
    from trl.trainer.utils import RepeatSampler

    classes = _classes(35, 2789, 176)
    weights = gd.sampling_weights(classes)
    ranks, per_device, accumulation, generations = 2, 2, 8, 8
    steps_per_generation = accumulation  # base.yaml: one generation per optimizer step
    prompts_per_step = ranks * per_device * steps_per_generation // generations
    # GRPOTrainer._get_train_sampler, as TRL 1.13 builds it (num_iterations 1)
    sampler = RepeatSampler(data_source=weights.order, mini_repeat_count=generations, batch_size=prompts_per_step,
                            repeat_count=steps_per_generation, shuffle=True, seed=0)
    indices = list(sampler)
    steps = len(indices) // (per_device * steps_per_generation) // ranks // accumulation
    assert prompts_per_step == 4 and steps == 600 == gd.PROMPT_DRAWS // prompts_per_step
    assert set(Counter(indices).values()) == {generations * steps_per_generation}
    permutation = list(dict.fromkeys(indices))
    assert sorted(permutation) == list(range(2400)) and permutation != list(range(2400))
    assert Counter(classes[weights.order[index]] for index in permutation) == Counter(weights.declaration["draws"])


def _verdict_row(verdict: str, margin: float = 1.0, *, anchor_class: str = "a") -> dict:
    """An anchor row whose q-hat favours ``verdict`` by ``margin`` (no against yes; unsure alone)."""
    if verdict == "unsure":
        q = {"q_yes": 0.0, "q_unsure": 1.0, "q_no": 0.0}
    else:
        high, low = (1 + margin) / 2, (1 - margin) / 2
        q = {"q_yes": high if verdict == "yes" else low, "q_unsure": 0.0, "q_no": high if verdict == "no" else low}
    return {**q, "mode": verdict, "anchor_class": anchor_class}


def _mix(no=(), *, under=0, under_margin=0.0, unsure=0, yes=0) -> dict[str, dict]:
    """Rows favouring no by each margin of ``no``, ``under`` more by ``under_margin``, then unsure and yes rows."""
    return {
        **{f"n{i}": _verdict_row("no", margin) for i, margin in enumerate(no)},
        **{f"u{i}": _verdict_row("no", under_margin) for i in range(under)},
        **{f"q{i}": _verdict_row("unsure") for i in range(unsure)},
        **{f"y{i}": _verdict_row("yes", anchor_class="c" if i % 2 else "b") for i in range(yes)},
    }


def _stratum(weights, rows, name):
    return sum(copies for sid, copies in weights.copies.items() if _stratum_of(weights, rows, sid) == name)


def _stratum_of(weights, rows, sid):
    verdict, margin = gd.favoured_verdict(rows[sid])
    if verdict != "no":
        return verdict
    return "no" if margin >= weights.declaration["no_margin"] else "no_under_margin"


def test_the_reward_favours_its_largest_q_hat_ties_to_no_by_a_margin_of_zero():
    assert gd.favoured_verdict({"q_yes": 0.5, "q_unsure": 0.0, "q_no": 0.5, "mode": "no"}) == ("no", 0.0)
    # 0.65 - 0.35 is 0.30000000000000004 in binary: rounded as the reward's margin buckets round it
    assert gd.favoured_verdict({"q_yes": 0.35, "q_unsure": 0.0, "q_no": 0.65, "mode": "no"}) == ("no", 0.3)
    assert gd.favoured_verdict({"q_yes": 0.2, "q_unsure": 0.5, "q_no": 0.3, "mode": "unsure"}) == ("unsure", 0.2)
    assert gd.favoured_verdict(_verdict_row("yes", 0.4)) == ("yes", 0.4)
    with pytest.raises(gd.GrpoDataError, match="its q-hat columns favour 'no' but its mode is 'yes'"):
        gd.favoured_verdict({"scenario_id": "s", "q_yes": 0.5, "q_unsure": 0.0, "q_no": 0.5, "mode": "yes"})


def test_the_verdict_mix_takes_its_share_from_many_prompts_under_the_cap():
    """200 rows favour no by at least 0.3 (150 by 0.5, 50 by 0.3)."""
    rows = _mix([0.5] * 150 + [0.3] * 50, under=50, under_margin=0.2, unsure=10, yes=2740)
    weights = gd.verdict_weights(rows, no_share=0.35, no_margin=0.3, repeat_cap=5)
    declaration = weights.declaration
    assert declaration["counts"] == {"no": 200, "no_under_margin": 50, "unsure": 10, "yes": 2740}
    assert declaration["draws"]["no"] == 840 == _stratum(weights, rows, "no")
    assert declaration["realized_shares"]["no"] == 0.35 == declaration["target_shares"]["no"]
    assert declaration["copies"]["no"] == {"min": 4, "max": 5, "scenarios_by_copies": {"4": 160, "5": 40}}
    assert declaration["scenarios_drawn"]["no"] == 200 and declaration["max_draws_per_scenario"] == 5
    first = next(name for name in declaration["tie_break"] if name in ("unsure", "yes"))
    expected = {"no": 840, "no_under_margin": 28, "unsure": 5, "yes": 1526}
    expected[first] += 1
    assert declaration["draws"] == expected == {name: _stratum(weights, rows, name) for name in expected}
    assert declaration["target_draws"]["yes"] == 1560 * 2740 / 2800
    assert all(copies <= 1 for sid, copies in weights.copies.items() if not sid.startswith("n"))
    assert declaration["scenarios_not_drawn"]["yes"] == 2740 - expected["yes"]
    assert len(weights.order) == declaration["rows"] == 2400 and Counter(weights.order) == +Counter(weights.copies)
    strata = {sid: _stratum_of(weights, rows, sid) for sid in rows}
    assert _prefix_gap(weights, strata) < 1
    assert declaration["draws_by_favoured_verdict"] == {"no": 868, "unsure": expected["unsure"],
                                                        "yes": expected["yes"]}
    assert sum(declaration["draws_by_class"].values()) == 2400
    assert declaration["draws_by_class"]["a"] == 868 + expected["unsure"]
    assert declaration["margins"]["drawn_no"] == {"min": 0.3, "median": 0.5, "max": 0.5}
    assert declaration["margins"]["favouring_no"] == {"min": 0.2, "median": 0.5, "max": 0.5}
    assert declaration["margins"]["favouring_no_by_a_tie"] == 0
    assert declaration["margins"]["favouring_no_at_least"] == {
        "0.1": 250, "0.2": 250, "0.3": 200, "0.4": 150, "0.5": 150, "0.6": 0, "0.7": 0, "0.8": 0, "0.9": 0, "1.0": 0}
    assert (declaration["stratification"], declaration["no_share"], declaration["no_margin"]) == (
        "favoured_verdict", 0.35, 0.3)


def test_the_verdict_mix_is_seeded_and_its_share_is_a_floor():
    rows = _mix([1.0] * 30, unsure=5, yes=165)
    first, again = (gd.verdict_weights(rows, no_share=0.3, no_margin=0.5, prompt_draws=150) for _ in range(2))
    assert (first.order, first.copies, first.declaration) == (again.order, again.copies, again.declaration)
    other = gd.verdict_weights(rows, no_share=0.3, no_margin=0.5, prompt_draws=150, seed=1)
    assert other.declaration["draws"] == first.declaration["draws"] and other.order != first.order
    # ceil(10 x 0.25) = 3 draws: at least the declared share, never under it
    small = gd.verdict_weights(rows, no_share=0.25, no_margin=0.5, prompt_draws=10)
    assert small.declaration["draws"]["no"] == 3 and small.declaration["realized_shares"]["no"] == 0.3
    assert small.declaration["target_draws"]["no"] == 2.5 and small.declaration["scenarios_not_drawn"]["no"] == 27
    # the whole run on the stratum: 30 rows x 2 draws, nothing left for the rest
    whole = gd.verdict_weights(rows, no_share=1.0, no_margin=0.5, prompt_draws=60, repeat_cap=2)
    assert whole.declaration["draws"] == {"no": 60, "no_under_margin": 0, "unsure": 0, "yes": 0}
    assert set(whole.copies[f"n{i}"] for i in range(30)) == {2} and "no_under_margin" not in whole.declaration["copies"]
    # a tied row favours no by 0: never in the 'no' stratum, whatever the (positive) margin
    tied = gd.verdict_weights(_mix([1.0] * 2, under=3, yes=5), no_share=0.2, no_margin=0.01, prompt_draws=10)
    assert tied.declaration["counts"] == {"no": 2, "no_under_margin": 3, "unsure": 0, "yes": 5}
    assert tied.declaration["margins"]["favouring_no_by_a_tie"] == 3


def test_the_verdict_mix_refuses_a_share_its_cap_cannot_meet():
    rows = _mix([0.5] * 100, under=40, under_margin=0.2, yes=2860)
    with pytest.raises(gd.GrpoDataError, match=r"the declared no share 0\.35 of the 2400 prompt draws is 840 draws, "
                                               r"but 100 scenario\(s\) favour 'no' by at least 0\.3, and at most 5 "
                                               r"draws each give 500") as refused:
        gd.verdict_weights(rows, no_share=0.35, no_margin=0.3, repeat_cap=5)
    # the refusal counts the rows each margin gives, and their draws at the cap: what a lower margin would serve
    assert str(refused.value).endswith(
        "at a cap of 5: 0.1: 140 (700), 0.2: 140 (700), 0.3: 100 (500), 0.4: 100 (500), 0.5: 100 (500), 0.6: 0 (0), "
        "0.7: 0 (0), 0.8: 0 (0), 0.9: 0 (0), 1.0: 0 (0)")
    # a lower margin (140 x 5 = 700) is still short; a cap of 6 then serves it
    with pytest.raises(gd.GrpoDataError, match="140 scenario"):
        gd.verdict_weights(rows, no_share=0.35, no_margin=0.2, repeat_cap=5)
    assert gd.verdict_weights(rows, no_share=0.35, no_margin=0.2, repeat_cap=6).declaration["copies"]["no"]["max"] == 6
    with pytest.raises(gd.GrpoDataError, match="the other 10 are drawn from the rest, each scenario at most once, "
                                               "but it holds 5"):
        gd.verdict_weights(_mix([1.0] * 10, yes=5), no_share=0.5, no_margin=0.3, prompt_draws=20, repeat_cap=1)
    with pytest.raises(gd.GrpoDataError, match="no training scenario"):
        gd.verdict_weights({}, no_share=0.5, no_margin=0.3)
    for bad in (dict(no_share=0.0), dict(no_share=1.5), dict(no_margin=0.0), dict(no_margin=-0.1),
                dict(no_margin=1.1), dict(prompt_draws=0), dict(repeat_cap=0)):
        declared = {"no_share": 0.35, "no_margin": 0.3, **bad}
        with pytest.raises(gd.GrpoDataError, match="must be positive"):
            gd.verdict_weights(rows, **declared)
        with pytest.raises(gd.GrpoDataError, match="must be positive"):
            gd.draw_rule(**declared)


def test_the_draw_rule_is_the_class_rule_unless_the_mix_is_declared():
    rows = {**_mix([1.0] * 20, unsure=15), **{f"y{i}": _verdict_row("yes", anchor_class="c" if i % 3 == 0 else "b")
                                                  for i in range(2965)}}
    classes = {sid: row["anchor_class"] for sid, row in rows.items()}
    for declared, cap in ((dict(class_a_share_cap=0.25, repeat_cap=18), 0.25), (dict(), gd.CLASS_A_SHARE_CAP)):
        by_rule = gd.draw_rule(**declared)(rows)
        by_class = gd.sampling_weights(classes, repeat_cap=declared.get("repeat_cap", gd.REPEAT_CAP),
                                       class_a_share_cap=cap)
        assert (by_rule.order, by_rule.copies, by_rule.declaration) == (
            by_class.order, by_class.copies, by_class.declaration)
    mixed = gd.draw_rule(no_share=0.2, no_margin=0.5, repeat_cap=24)(rows)
    assert mixed.declaration == gd.verdict_weights(rows, no_share=0.2, no_margin=0.5, repeat_cap=24).declaration
    with pytest.raises(gd.GrpoDataError, match=r"no_margin \(0\.3\) belongs to the judgment mix"):
        gd.draw_rule(no_margin=0.3)
    with pytest.raises(gd.GrpoDataError, match="declare one rule"):
        gd.draw_rule(class_a_share_cap=0.25, no_share=0.35, no_margin=0.3)
    with pytest.raises(gd.GrpoDataError, match="needs its margin"):
        gd.draw_rule(no_share=0.35)


def test_the_build_writes_what_the_design_asks(world, built, tmp_path):
    manifest, grpo, replay = built.manifest, built.grpo, built.replay
    anchors = {row["scenario_id"]: row for row in read_anchor_set(world.anchors).rows}
    assert read_grpo_data(built.out) == manifest
    assert manifest["counts"] == {
        "anchor_rows": 4, "anchor_rows_not_training": 0, "held_out_scenarios": 0, "training_scenarios": 4,
        "dropped_by_prompt_cap": 0,
        "kept": 4, "kept_by_source": {"stress": 2, "wildchat": 2}, "prompt_tokens_max": max(r["prompt_tokens"] for r in grpo),
    }
    assert (built.out / gd.SCENARIOS_FNAME).read_text() == world.population.read_text()
    sids = list(anchors)
    copies = {sid: sum(1 for row in grpo if row["scenario_id"] == sid) for sid in sids}
    assert all(row["sampling_weight"] == copies[row["scenario_id"]] for row in grpo)
    assert sum(copies.values()) == len(grpo) == manifest["sampling_weights"]["rows"] == 4
    weights = manifest["sampling_weights"]
    assert weights["draws"] == {"a": 3, "b": 1, "c": 0} and Counter(r["anchor_class"] for r in grpo) == {"a": 3, "b": 1}
    assert weights["tie_break"] == ["b", "c"] and weights["scenarios_not_drawn"] == {"a": 0, "b": 0, "c": 1}
    assert weights["copies"]["a"] == {"min": 1, "max": 2, "scenarios_by_copies": {"1": 1, "2": 1}}
    assert (weights["seed"], weights["class_a_share_cap"], weights["scenarios_not_drawn_b_c"]) == (gd.DRAW_SEED, 0.75, 1)
    columns = RowColumns.read({name: [row[name] for row in grpo] for name in grpo[0]}, len(grpo))
    for row, q in zip(grpo, columns.q):
        anchor = anchors[row["scenario_id"]]
        assert q == {"yes": anchor["q_yes"], "unsure": anchor["q_unsure"], "no": anchor["q_no"]}
        assert (row["anchor_class"], row["topped_up"], row["base_yes_rules"], row["k"], row["mode"]) == (
            anchor["anchor_class"], anchor["topped_up"], anchor["base_yes_rules"], anchor["k"], anchor["mode"])
    assert [anchors[sid]["anchor_class"] for sid in sids] == ["c", "b", "a", "a"]
    # the GRPO prompt is the one the campaign's own runner builds, and TRL renders it to the campaign's ids
    tokenizer = tokenizer_of(world)
    runner = draw_runner(tmp_path, population_rows(), read_jsonl(world.rules))
    for row in grpo:
        entry = runner.bundle_prompts(row["scenario_id"])[0][1]
        assert row["prompt"] == gd.conversation(entry)
        ids = gd.trl_prompt_ids(tokenizer, [row["prompt"]], {"enable_thinking": True})[0]
        assert ids == gd.campaign_prompt_ids(tokenizer, [entry], enable_thinking=True)[0] and len(ids) == row["prompt_tokens"]
    # replay rows: the training per-rule prompt, then the base's completion and the end of turn
    index = read_jsonl(world.anchors / REPLAY_INDEX_FNAME)
    assert len(replay) == len(index) == manifest["replay"]["rows"] == 2 * 2 * 2 + 2 * 2 * 4
    raw = {}
    for source, root, k in (("bank", world.root, 2), ("top_up", world.top_up, 4)):
        for draw in range(k):
            for row in read_jsonl(root / "gemma/individual_bank" / f"draw-{draw:03d}" / "raw_outputs.jsonl"):
                raw[(source, row["scenario_id"], row["semantic_attempt_id"], row["request_block_id"], row["component_id"])] = row
    for row, entry in zip(replay, index):
        prompt = [token for token, label in zip(row["input_ids"], row["labels"]) if label == gd.IGNORE_INDEX]
        completion = row["input_ids"][len(prompt):]
        assert row["labels"] == [gd.IGNORE_INDEX] * len(prompt) + completion and "attention_mask" not in row
        per_rule = dict(runner.bundle_prompts(row["scenario_id"]))[row["rule_id"]]
        assert prompt == gd.campaign_prompt_ids(tokenizer, [per_rule], enable_thinking=True)[0]
        assert prompt_sha256(per_rule) == row["prompt_sha256"] == entry["prompt_sha256"]
        text = raw[(entry["source"], entry["scenario_id"], entry["semantic_attempt_id"], entry["request_block_id"],
                    entry["rule_id"])]["raw_response"]
        assert completion[-1] == tokenizer.eos_token_id and tokenizer.decode(completion[:-1]) == text
        assert completion[0] == tokenizer.convert_tokens_to_ids("<think>")
        assert (row["scenario_id"], row["rule_id"], row["decoder_draw_id"], row["label"], row["raw_output_sha256"]) == (
            entry["scenario_id"], entry["rule_id"], entry["decoder_draw_id"], entry["label"], entry["raw_output_sha256"])
    assert manifest["replay"]["completions_opening_think"] == len(replay)
    assert manifest["replay"]["completions_quoting_stop_tokens"] == 0
    assert manifest["replay"]["completions_with_other_special_tokens"] == 0


def test_each_scenario_rules_replayed_label_mix_is_its_p_hat_and_the_weights_average_one(world, built):
    anchors = {row["scenario_id"]: row for row in read_anchor_set(world.anchors).rows}
    replay = built.replay
    weights = [row["weight"] for row in replay]
    assert abs(sum(weights) / len(weights) - 1) < 1e-12 and built.manifest["replay"]["weight_mean"] == pytest.approx(1)
    pairs = {(row["scenario_id"], row["rule_id"]) for row in replay}
    assert len(pairs) == 4 * 2 == built.manifest["replay"]["pairs"]
    totals = set()
    for sid, rule in pairs:
        rows = [row for row in replay if (row["scenario_id"], row["rule_id"]) == (sid, rule)]
        total = sum(row["weight"] for row in rows)
        totals.add(round(total, 12))
        mix = {label: sum(row["weight"] for row in rows if row["label"] == label) / total for label in ("no", "unsure", "yes")}
        assert mix == pytest.approx(anchors[sid]["rule_marginals"][rule], abs=1e-12)
        assert len(rows) == anchors[sid]["k"]
    # every (scenario, rule) weighs the same in total, whatever its K
    assert len(totals) == 1
    # K = 2 and K = 4 rows: K-bar = 3, so 3/2 and 3/4
    assert {row["weight"] for row in replay} == {1.5, 0.75}


def test_every_output_carries_a_marker_for_its_own_bytes(world, built):
    markers = MarkersDouble()
    scenario_marker = markers.check(built.out / gd.SCENARIOS_FNAME)
    for name in (gd.GRPO_ROWS_FNAME, gd.REPLAY_ROWS_FNAME):
        marker = markers.check(built.out / name)
        assert marker["dataset"]["name"] == name and marker["dataset"]["rows"] == built.manifest["outputs"][name]["rows"]
        assert marker["derived_from"] == {
            "by": gd.ROWS_DERIVATION, "name": gd.SCENARIOS_FNAME, "sha256": sha256_file(built.out / gd.SCENARIOS_FNAME),
            "rows": 4, "marker_sha256": sha256_file(disjoint_marker_path(built.out / gd.SCENARIOS_FNAME)),
        }
        assert marker["protected"] == scenario_marker["protected"] and marker["policy"] == scenario_marker["policy"]
    assert scenario_marker["protected"][0]["sha256"] == sha256_file(world.protected)
    assert built.manifest["markers"][gd.GRPO_ROWS_FNAME] == sha256_file(disjoint_marker_path(built.out / gd.GRPO_ROWS_FNAME))
    assert built.manifest["scenarios"]["marker_sha256"] == sha256_file(disjoint_marker_path(world.population))


def test_the_manifest_records_every_input_and_output_digest(world, built):
    manifest = built.manifest
    assert manifest["anchors"]["manifest_sha256"] == sha256_file(world.anchors / ANCHORS_MANIFEST_FNAME)
    assert manifest["scenarios"]["sha256"] == sha256_file(world.population)
    assert manifest["rules"] == Ruleset.load(world.rules).record()
    for name, root, raw, draws in (("bank", world.root, True, 2), ("top_up", world.top_up, True, 4),
                                   ("full_set", world.root, False, 2)):
        condition = "gemma/full" if name == "full_set" else "gemma/individual_bank"
        record = manifest["campaigns"][name]
        assert record["condition_id"] == condition and len(record["draws"]) == draws
        for draw, digests in record["draws"].items():
            assert digests["attempts_sha256"] == sha256_file(root / condition / draw / "attempts.jsonl")
            assert ("raw_outputs_sha256" in digests) == raw
            if raw:
                assert digests["raw_outputs_sha256"] == sha256_file(root / condition / draw / "raw_outputs.jsonl")
    assert manifest["anchor_condition"] == str(world.root / "gemma/individual_bank/condition.json")
    assert manifest["tokenizer"]["files"]["tokenizer.json"] == sha256_file(world.model / "tokenizer.json")
    assert manifest["tokenizer"]["stop_token_ids"] == [EOS, ENDOFTEXT]
    assert manifest["tokenizer"]["campaign_model_dir"] == str(world.model)
    assert manifest["rendering"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert (manifest["allow_no_held_out"], manifest["protected_extra"], manifest["required_protected"]) == (True, [], [])
    assert manifest["pinned_protected"] == [{"name": "frozen_500", "sha256": sha256_file(world.protected)}]
    assert manifest["allow_no_pinned_protected"] is False
    assert (manifest["held_out"], manifest["counts"]["held_out_scenarios"]) == ([], 0)
    for name in OUTPUT_FNAMES:
        assert manifest["outputs"][name]["sha256"] == sha256_file(built.out / name)
    assert manifest["sampling_weights"]["prompt_draws"] == 4 and manifest["prompt_token_cap"] == gd.PROMPT_TOKEN_CAP


def test_the_rows_are_the_draws_in_their_order():
    anchor = {"q_yes": 1.0, "q_unsure": 0.0, "q_no": 0.0, "mode": "yes", "anchor_class": "c", "topped_up": False,
              "k": 2, "base_yes_rules": ["Rule 1"]}
    prompts = [gd.GrpoPrompt(gd.TrainingScenario(sid, {}, "", anchor), {"system_prompt": sid, "user_prompt": ""}, 7)
               for sid in ("s2", "s0", "s1")]
    weights = gd.SamplingWeights(copies={"s2": 1, "s0": 3, "s1": 0}, order=["s0", "s2", "s0", "s0"], declaration={})
    rows = gd.grpo_rows(prompts, weights)
    assert [(row["scenario_id"], row["sampling_weight"]) for row in rows] == [("s0", 3), ("s2", 1), ("s0", 3), ("s0", 3)]
    assert rows[1]["prompt"] == [{"role": "system", "content": "s2"}, {"role": "user", "content": ""}]


def test_the_build_is_deterministic(world, built, tmp_path):
    again = build_into(world, tmp_path / "again")
    for name in OUTPUT_FNAMES:
        assert (tmp_path / "again" / name).read_bytes() == (built.out / name).read_bytes()
        assert disjoint_marker_path(tmp_path / "again" / name).read_bytes() == disjoint_marker_path(built.out / name).read_bytes()
    assert {**again, "created_at": None} == {**built.manifest, "created_at": None}


GOLDEN_OUTPUTS = {
    gd.SCENARIOS_FNAME: {"sha256": "74a17e54e6d9655fdf22680ae1a4c6a5b069d8f712fa94412071aa591265c4a6", "rows": 4},
    gd.GRPO_ROWS_FNAME: {"sha256": "8471f82eda96fde746151e3016ed6fc3e02c510e64775e5077b8b2ce44165fc8", "rows": 4},
    gd.REPLAY_ROWS_FNAME: {"sha256": "0fd52339d52a254d5d19e0c7f6d092a2ed9aa5a0f394f9362e51fe4c74e2a25c", "rows": 24},
}
GOLDEN_DECLARATION_SHA256 = "c82b3acd6ce60f827a68b03a2dfc9bdd8ad053ef14a51cd24567cb02541f216d"


def _canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def test_the_class_rule_rebuilds_an_earlier_build_byte_for_byte(built):
    assert built.manifest["outputs"] == GOLDEN_OUTPUTS
    assert _canonical_sha256(built.manifest["sampling_weights"]) == GOLDEN_DECLARATION_SHA256
    assert "stratification" not in built.manifest["sampling_weights"]


BUILT_RUNS = ("train-final", "train-a25", "train-a85")
GRPO_DATA_ROOT = Path(os.environ.get("RULEFORGE_ARTIFACT_ROOT") or "/nonexistent") / "grpo_data"


@pytest.mark.cluster
@pytest.mark.parametrize("name", BUILT_RUNS)
def test_the_retrain_s_rows_rebuild_byte_for_byte_under_the_class_rule(name, tmp_path):
    directory = GRPO_DATA_ROOT / name
    if not (directory / gd.MANIFEST_FNAME).is_file():
        pytest.skip(f"{directory} is not here: set RULEFORGE_ARTIFACT_ROOT to the root holding grpo_data/; their byte "
                    "identity was not checked")
    manifest = json.loads((directory / gd.MANIFEST_FNAME).read_text(encoding="utf-8"))
    anchor_sets = manifest["anchors"] if isinstance(manifest["anchors"], list) else [manifest["anchors"]]
    anchors = {row["scenario_id"]: row for entry in anchor_sets
               for row in iter_jsonl_records(Path(entry["path"]) / ANCHORS_FNAME)}
    order = [compute_scenario_id(row) for row in iter_jsonl_records(directory / gd.SCENARIOS_FNAME)]
    declared = manifest["sampling_weights"]
    weights = gd.draw_rule(prompt_draws=declared["prompt_draws"], repeat_cap=declared["repeat_cap"],
                           seed=declared["seed"], class_a_share_cap=declared["class_a_share_cap"])(
        {sid: anchors[sid] for sid in order})
    assert json.loads(json.dumps(weights.declaration)) == declared
    rows = {row["scenario_id"]: row for row in iter_jsonl_records(directory / gd.GRPO_ROWS_FNAME)}
    assert set(rows) == {sid for sid, copies in weights.copies.items() if copies}
    rebuilt = tmp_path / gd.GRPO_ROWS_FNAME
    write_jsonl_records(rebuilt, [{**rows[sid], "sampling_weight": weights.copies[sid]} for sid in weights.order])
    assert sha256_file(rebuilt) == manifest["outputs"][gd.GRPO_ROWS_FNAME]["sha256"] == sha256_file(
        directory / gd.GRPO_ROWS_FNAME)


def test_a_verdict_mix_build_draws_its_share_and_changes_only_the_grpo_rows(world, built, tmp_path):
    """The world's rows: scenario 3 favours no by 1 (q-hat no 1), scenario 2 by 0 (a tie), 0 and 1 favour yes."""
    anchors = {row["scenario_id"]: row for row in read_jsonl(world.anchors / ANCHORS_FNAME)}
    by_q = {(row["q_no"], row["mode"]): sid for sid, row in anchors.items()}
    strong, tied = by_q[(1.0, "no")], by_q[(0.5, "no")]
    out = tmp_path / "mix"
    manifest = build_into(world, out, class_a_share_cap=None, no_share=0.5, no_margin=0.3, repeat_cap=2)
    rows = read_jsonl(out / gd.GRPO_ROWS_FNAME)
    drawn = Counter(row["scenario_id"] for row in rows)
    assert drawn[strong] == 2 and drawn[tied] == 1 and sum(drawn.values()) == 4 and len(drawn) == 3
    assert {row["scenario_id"]: row["sampling_weight"] for row in rows} == dict(drawn)
    declaration = manifest["sampling_weights"]
    assert declaration["draws"] == {"no": 2, "no_under_margin": 1, "unsure": 0, "yes": 1}
    assert declaration["realized_shares"]["no"] == 0.5 and declaration["scenarios_drawn"]["no"] == 1
    assert declaration["margins"]["drawn_no"] == {"min": 1.0, "median": 1.0, "max": 1.0}
    assert declaration["margins"]["favouring_no_by_a_tie"] == 1
    assert manifest["dropped_by_prompt_cap"] == []
    yes_drawn = next(sid for sid in drawn if anchors[sid]["mode"] == "yes")
    assert declaration["draws_by_class"] == {"a": 3, anchors[yes_drawn]["anchor_class"]: 1}
    columns = RowColumns.read({key: [row[key] for row in rows] for key in rows[0]}, len(rows))
    assert columns.classes == [anchors[row["scenario_id"]]["anchor_class"] for row in rows]
    for name in (gd.SCENARIOS_FNAME, gd.REPLAY_ROWS_FNAME):
        assert (out / name).read_bytes() == (built.out / name).read_bytes()
    assert read_grpo_data(out)["sampling_weights"] == declaration
    with pytest.raises(gd.GrpoDataError, match="the declared no share 0.75 of the 4 prompt draws is 3 draws"):
        build_into(world, tmp_path / "short", class_a_share_cap=None, no_share=0.75, no_margin=0.3, repeat_cap=2)
    assert not (tmp_path / "short").exists() and not (tmp_path / "short.partial").exists()
    with pytest.raises(gd.GrpoDataError, match="declare one rule"):
        build_into(world, tmp_path / "both", anchors=tmp_path / "absent", no_share=0.5, no_margin=0.3)


def test_a_verdict_mix_build_records_what_the_prompt_cap_took_from_each_stratum(world, tmp_path, monkeypatch):
    anchors = {row["scenario_id"]: row for row in read_jsonl(world.anchors / ANCHORS_FNAME)}
    tied = next(sid for sid, row in anchors.items() if (row["q_no"], row["mode"]) == (0.5, "no"))
    rendered = gd.grpo_prompts

    def longer(*args, **kwargs):
        return [dataclasses.replace(prompt, tokens=10**6) if prompt.scenario.scenario_id == tied else prompt
                for prompt in rendered(*args, **kwargs)]

    monkeypatch.setattr(gd, "grpo_prompts", longer)
    manifest = build_into(world, tmp_path / "mix", class_a_share_cap=None, no_share=0.5, no_margin=0.3,
                          repeat_cap=2, prompt_token_cap=10**6 - 1)
    assert manifest["dropped_by_prompt_cap"] == [{"scenario_id": tied, "prompt_tokens": 10**6, "anchor_class": "a",
                                                  "favoured_verdict": "no", "margin": 0.0}]
    assert manifest["sampling_weights"]["counts"]["no_under_margin"] == 0


def test_the_prompt_cap_drops_a_scenario_with_its_replay_rows(world, built, tmp_path):
    lengths = sorted({row["prompt_tokens"] for row in built.grpo})
    manifest = build_into(world, tmp_path / "capped", prompt_token_cap=lengths[-2])
    longest = next(row["scenario_id"] for row in built.grpo if row["prompt_tokens"] == lengths[-1])
    assert manifest["counts"]["dropped_by_prompt_cap"] == 1 and manifest["counts"]["kept"] == 3
    assert manifest["dropped_by_prompt_cap"] == [{"scenario_id": longest, "prompt_tokens": lengths[-1], "anchor_class": "a"}]
    for name in (gd.GRPO_ROWS_FNAME, gd.REPLAY_ROWS_FNAME):
        assert longest not in {row["scenario_id"] for row in read_jsonl(tmp_path / "capped" / name)}
    assert len(read_jsonl(tmp_path / "capped" / gd.SCENARIOS_FNAME)) == 3
    assert manifest["replay"]["rows"] == len(built.replay) - 2 * 4
    with pytest.raises(gd.GrpoDataError, match="every training prompt exceeds the 10-token cap"):
        build_into(world, tmp_path / "none", prompt_token_cap=10)
    with pytest.raises(gd.GrpoDataError, match="prompt_token_cap must be positive"):
        build_into(world, tmp_path / "zero", prompt_token_cap=0)


def held_out_split(world, tmp_path):
    """The pool less its last scenario (marked), and that scenario as ``dev_nonyes``: the shape."""
    lines = world.population.read_text().splitlines()
    training = tmp_path / "train.jsonl"
    training.write_text("\n".join(lines[:3]) + "\n")
    mark(training, protected=world.protected)
    return training, write_jsonl(tmp_path / "dev_nonyes.jsonl", [json.loads(lines[3])])


def test_a_held_out_scenario_is_left_out_and_an_extra_protected_file_is_attested(world, tmp_path):
    training, held = held_out_split(world, tmp_path)
    dev = write_jsonl(tmp_path / "dev.jsonl", [{"prompt": "a dev prompt", "response": "r"}])
    mark(training, protected=world.protected, also=[dev])
    markers = MarkersDouble()
    manifest = build_into(world, tmp_path / "out", scenarios=training, protected=[held], markers=markers,
                          required_protected=[dev], allow_no_held_out=False)
    assert manifest["counts"]["anchor_rows_not_training"] == 1 and manifest["counts"]["kept"] == 3
    assert manifest["counts"]["held_out_scenarios"] == 1
    assert manifest["held_out"] == [{"path": str(held), "sha256": sha256_file(held), "rows": 1}]
    assert markers.attested == [(tmp_path / "out.partial" / gd.SCENARIOS_FNAME, [held])]
    assert manifest["protected_extra"] == [{"path": str(held), "sha256": sha256_file(held)}]
    assert manifest["required_protected"] == [{"path": str(dev), "sha256": sha256_file(dev)}]
    assert manifest["pinned_protected"] == [{"name": "frozen_500", "sha256": sha256_file(world.protected)}]
    assert manifest["allow_no_held_out"] is False
    pinned, required = sha256_file(world.protected), sha256_file(dev)
    assert markers.checked == [("train.jsonl", [required], []), ("train.jsonl", [pinned], []),
                               (gd.SCENARIOS_FNAME, sorted({pinned, required}), [sha256_file(held)])]
    written = markers.check(tmp_path / "out" / gd.SCENARIOS_FNAME)
    assert written["carved"][-1]["sha256"] == sha256_file(held)
    assert sha256_file(held) not in {entry["sha256"] for entry in written["protected"]}


def test_d7s_pinned_protected_files_must_be_covered_unless_declared(world, tmp_path):
    training, held = held_out_split(world, tmp_path)
    mark(training, protected=held)
    common = dict(scenarios=training, protected=[held], allow_no_held_out=False)
    for number, (pins, match) in enumerate(((PINNED_PROTECTED, r"frozen_500 \(fa9bdc5eb6b9\), frozen_500_fig5_grpo_"
                                             r"disjoint \(d67712565648\), wildchat_stress_weird_200x3 \(77c6cf97a1c3\)"),
                                            (pinned_stand_in(world), r"frozen_500 \([0-9a-f]{12}\)\)"))):
        markers = MarkersDouble()
        with pytest.raises(LeakageError, match=f"not checked against the required .*{match}.*--allow-no-pinned-protected"):
            build_into(world, tmp_path / f"out{number}", pinned_protected=pins, markers=markers, **common)
        assert markers.checked == [("train.jsonl", [], []), ("train.jsonl", sorted(set(pins.values())), [])]
        assert not (tmp_path / f"out{number}").exists() and not (tmp_path / f"out{number}.partial").exists()
    markers = MarkersDouble()
    manifest = build_into(world, tmp_path / "declared", pinned_protected=PINNED_PROTECTED,
                          allow_no_pinned_protected=True, markers=markers, **common)
    assert (manifest["pinned_protected"], manifest["allow_no_pinned_protected"]) == ([], True)
    assert markers.checked[-1] == (gd.SCENARIOS_FNAME, [], [sha256_file(held)])


def test_the_pinned_protected_files_are_the_ones_the_rows_build_pins():
    """frozen_500 as the runs held dev apart from it; its derivation and reserve as recorded."""
    import yaml

    assert gd.NO_PINS == {} and gd.build_grpo_data.__kwdefaults__["pinned_protected"] is gd.NO_PINS
    frozen_500 = PINNED_PROTECTED["frozen_500"]
    assert frozen_500 == "fa9bdc5eb6b9bac8e7b42a27f4ab9f6b60bd5a0e4ac4ae3a63190eb00e487411"
    for experiment in ("ruleset_size", "ruleforge", "training"):
        config = yaml.safe_load((REPO_ROOT / "configs" / f"sampling_{experiment}.yaml").read_text())
        assert config["approved_scenario_sha256"] == PAPER_POPULATION_SHA256
        assert config["scenario_set_path"] == "${oc.env:RULEFORGE_ARTIFACT_ROOT}/populations/frozen_500_r3.jsonl"


def test_a_build_that_holds_nothing_out_is_refused_unless_declared(world, tmp_path):
    training, held = held_out_split(world, tmp_path)
    mixed = write_jsonl(tmp_path / "mixed.jsonl", [*read_jsonl(held), {"prompt": "not anchored", "response": "r"}])
    empty = write_jsonl(tmp_path / "empty.jsonl", [])
    refused = "no protected file carves anchor rows out of"
    cases = {
        "the pool, nothing named": ({}, f"{refused} .*population.jsonl \\(none named\\)"),
        "held out, nothing named": ({"scenarios": training}, f"{refused} .*train.jsonl \\(none named\\)"),
        "held out, an unrelated file named": (
            {"scenarios": training, "protected": [world.protected]}, "frozen.jsonl: 0 of its 1 rows anchored"),
        "held out, a part-anchored file named": (
            {"scenarios": training, "protected": [mixed]}, "mixed.jsonl: 1 of its 2 rows anchored"),
        "held out, an empty file named": (
            {"scenarios": training, "protected": [empty]}, "empty.jsonl: 0 of its 0 rows anchored"),
    }
    for number, (overrides, match) in enumerate(cases.values()):
        with pytest.raises(LeakageError, match=match):
            build_into(world, tmp_path / f"out{number}", allow_no_held_out=False, **overrides)
        assert not (tmp_path / f"out{number}").exists() and not (tmp_path / f"out{number}.partial").exists()
    manifest = build_into(world, tmp_path / "declared", scenarios=training, protected=[world.protected])
    assert (manifest["held_out"], manifest["counts"]["held_out_scenarios"], manifest["allow_no_held_out"]) == ([], 0, True)


def test_a_protected_file_naming_a_training_scenario_is_refused_whatever_is_declared(world, tmp_path):
    _, held = held_out_split(world, tmp_path)
    for number, declared in enumerate((False, True)):
        with pytest.raises(LeakageError, match="dev_nonyes.jsonl, a protected file, names 1 of the 4 training scenarios"):
            build_into(world, tmp_path / f"out{number}", protected=[world.protected, held], allow_no_held_out=declared)
        assert not (tmp_path / f"out{number}").exists() and not (tmp_path / f"out{number}.partial").exists()


@pytest.mark.parametrize("text, match", [("{\"prompt\": \"p\"\n", "line 1 is not JSON"),
                                         ("\n[\"p\", \"r\"]\n", "line 2 is not a JSON object")])
def test_a_protected_file_must_be_json_objects(world, tmp_path, text, match):
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text(text, encoding="utf-8")
    with pytest.raises(gd.GrpoDataError, match=match):
        build_into(world, tmp_path / "out", protected=[malformed])


def test_every_required_and_held_out_file_must_be_covered(world, tmp_path):
    training, held = held_out_split(world, tmp_path)
    common = dict(scenarios=training, protected=[held], allow_no_held_out=False)
    dev = write_jsonl(tmp_path / "dev.jsonl", [{"prompt": "a dev prompt", "response": "r"}])
    with pytest.raises(LeakageError, match="was not checked against the required protected file"):
        build_into(world, tmp_path / "o1", required_protected=[dev], **common)

    class DropsTheHeldOut(MarkersDouble):
        def attest(self, scenarios, source, extra_protected):
            return super().attest(scenarios, source, [])

    with pytest.raises(LeakageError, match="was not checked against the required file.* not even by the second pass"):
        build_into(world, tmp_path / "o2", markers=DropsTheHeldOut(), **common)
    assert not (tmp_path / "o2").exists() and not (tmp_path / "o2.partial").exists()


def test_the_marker_is_required_and_a_leak_refuses_before_anything_is_left(world, tmp_path):
    unmarked = tmp_path / "unmarked.jsonl"
    shutil.copy(world.population, unmarked)
    with pytest.raises(LeakageError, match="no prompt-disjointness marker"):
        build_into(world, tmp_path / "out", scenarios=unmarked)
    with pytest.raises(LeakageError, match="planted leak"):
        build_into(world, tmp_path / "out", markers=MarkersDouble(leak=True))
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists()


def test_block_2s_module_is_bound_by_name_and_required(monkeypatch, tmp_path):
    with monkeypatch.context() as absent:  # as on an install without the marker check's module
        absent.setitem(sys.modules, gd.EXCLUSION_MODULE, None)
        with pytest.raises(LeakageError, match="ruleforge.grpo.exclusion is not importable"):
            gd.Block2Markers()
    calls = []

    class Policy:
        @staticmethod
        def from_mapping(payload):
            calls.append(("policy", payload))
            return "policy"

    def prompt_disjoint_marker(dataset, protected, *, policy, allowed_family_pairs, carved, screen):
        calls.append(("second pass", dataset, protected, carved, policy, allowed_family_pairs, screen))
        return {"protected": [{"sha256": "p1"}, {"sha256": "p2"}], "carved": [{"sha256": "c1"}]}

    def check_prompt_marker(dataset, *, required_protected, required_carved, allow_unscreened):
        return {"checked": str(dataset), "required": required_protected, "read": required_carved,
                "unscreened_taken": allow_unscreened}

    module = types.SimpleNamespace(
        check_prompt_marker=check_prompt_marker,
        ExclusionPolicy=Policy,
        prompt_disjoint_marker=prompt_disjoint_marker,
        write_prompt_disjoint_marker=lambda dataset, marker: Path(f"{dataset}.written"),
        marker_screen=lambda marker: marker.get("screen", "full"),
    )
    monkeypatch.setitem(sys.modules, gd.EXCLUSION_MODULE, module)
    markers = gd.Block2Markers()
    assert markers.check(Path("s.jsonl")) == {"checked": "s.jsonl", "required": [], "read": [],
                                              "unscreened_taken": True}
    assert markers.check(Path("s.jsonl"), ("d1", "d2"), ("c1",)) == {"checked": "s.jsonl", "required": ["d1", "d2"],
                                                                     "read": ["c1"], "unscreened_taken": True}
    source = {"policy": {"shingle_n": 5}, "protected": [{"path": "artifact://frozen.jsonl", "sha256": "p1"}],
              "carved": [{"path": "artifact://held.jsonl", "sha256": "c1"}], "allowed_family_pairs": [["p", "c"]]}
    assert markers.attest(Path("s.jsonl"), source, [Path("dev_nonyes.jsonl")])["protected"][0] == {"sha256": "p1"}
    assert calls[-1] == ("second pass", Path("s.jsonl"), ["artifact://frozen.jsonl"],
                         ["artifact://held.jsonl", "dev_nonyes.jsonl"], "policy", [("p", "c")], "full")
    markers.attest(Path("s.jsonl"), {**source, "screen": "containment-only, unscreened"}, [])
    assert calls[-1][-1] == "containment-only, unscreened"
    assert markers.write(Path("g.jsonl"), {}) == Path("g.jsonl.written")
    with pytest.raises(LeakageError, match="have changed since"):
        markers.attest(Path("s.jsonl"), {**source, "protected": [{"path": "x", "sha256": "gone"}]}, [])


def test_a_file_the_second_pass_alone_read_is_never_relabelled_as_screened(tmp_path):
    from ruleforge.grpo import exclusion as ex

    def rows(path, *prompts):
        return write_jsonl(path, [{"prompt": prompt, "response": "r", "source": "wildchat"} for prompt in prompts])

    scenarios = rows(tmp_path / "scenarios.jsonl", "a fresh prompt about cooking pasta", "a prompt about the tides")
    frozen, dev, extra = (rows(tmp_path / f"{name}.jsonl", f"a {name} prompt about something else entirely")
                          for name in ("frozen", "dev", "extra"))
    source = ex.prompt_disjoint_marker(scenarios, [frozen], policy=ex.ExclusionPolicy(), carved=[dev],
                                       screen=ex.FULL_SCREEN)
    marker = gd.Block2Markers().attest(scenarios, source, [extra])
    assert [entry["sha256"] for entry in marker["protected"]] == [sha256_file(frozen)]
    assert [entry["sha256"] for entry in marker["carved"]] == [sha256_file(dev), sha256_file(extra)]
    assert marker["policy_sha256"] == source["policy_sha256"] == ex.ExclusionPolicy().digest()
    gd.Block2Markers().write(scenarios, marker)
    assert gd.Block2Markers().check(scenarios, [sha256_file(frozen)], [sha256_file(dev), sha256_file(extra)])
    with pytest.raises(LeakageError, match="was not checked against the required protected file"):
        gd.Block2Markers().check(scenarios, [sha256_file(extra)])
    dev.write_text(dev.read_text() + json.dumps({"prompt": "one more dev prompt", "response": "r"}) + "\n")
    with pytest.raises(LeakageError, match="have changed since"):
        gd.Block2Markers().attest(scenarios, source, [extra])


def test_a_derived_marker_is_the_scenarios_attestation_for_the_outputs_bytes():
    scenario_marker = {"kind": "prompt_disjoint", "dataset": {"name": "scenarios.jsonl", "sha256": "s", "rows": 3,
                                                               "prompt_keys": 3}, "protected": [{"sha256": "p"}]}
    marker = gd.derived_marker(scenario_marker, name="grpo_rows.jsonl", sha256="g", rows=9, scenarios_marker_sha256="m")
    assert marker["dataset"] == {"name": "grpo_rows.jsonl", "sha256": "g", "rows": 9, "prompt_keys": 3}
    assert marker["derived_from"] == {"by": gd.ROWS_DERIVATION, "name": "scenarios.jsonl", "sha256": "s", "rows": 3,
                                      "marker_sha256": "m"}
    assert marker["protected"] == scenario_marker["protected"] and scenario_marker["dataset"]["rows"] == 3


def copy_anchors(world, tmp_path, *, manifest=None, index=None, anchor_rows=None) -> Path:
    """A copy of the anchor set."""
    target = tmp_path / "anchors"
    shutil.copytree(world.anchors, target)
    recorded = json.loads((target / ANCHORS_MANIFEST_FNAME).read_text())
    for name, edit in ((REPLAY_INDEX_FNAME, index), (ANCHORS_FNAME, anchor_rows)):
        if edit is not None:
            write_jsonl(target / name, edit(read_jsonl(target / name)))
            recorded["outputs"][name]["sha256"] = sha256_file(target / name)
    if manifest is not None:
        recorded = manifest(recorded)
    (target / ANCHORS_MANIFEST_FNAME).write_text(json.dumps(recorded))
    return target


def test_the_inputs_must_be_the_anchor_sets(world, tmp_path):
    (tmp_path / "exists").mkdir()
    with pytest.raises(gd.GrpoDataError, match="exists; GRPO data is written once"):
        build_into(world, tmp_path / "exists")
    with pytest.raises(gd.GrpoDataError, match="'dev'-side anchor set"):
        build_into(world, tmp_path / "o1", anchors=copy_anchors(world, tmp_path / "1", manifest=lambda m: {**m, "side": SIDE_DEV}))
    with pytest.raises(gd.GrpoDataError, match="not a schema_version 1, 2 or 3 anchor set"):
        build_into(world, tmp_path / "o2", anchors=copy_anchors(world, tmp_path / "2", manifest=lambda m: {**m, "schema_version": 0}))
    tampered = copy_anchors(world, tmp_path / "3")
    (tampered / REPLAY_INDEX_FNAME).write_text((tampered / REPLAY_INDEX_FNAME).read_text() + "\n")
    with pytest.raises(gd.GrpoDataError, match="is not the file its manifest records"):
        build_into(world, tmp_path / "o3", anchors=tampered)
    other_rules = write_jsonl(tmp_path / "rules.jsonl", [*read_jsonl(world.rules), {"id": "Rule 3", "rule": "x"}])
    with pytest.raises(gd.GrpoDataError, match="is not the ruleset the anchors judged"):
        build_into(world, tmp_path / "o4", rules=other_rules)
    moved = copy_anchors(world, tmp_path / "5", manifest=lambda m: {**m, "bank": {**m["bank"], "generation_seed_base": 99}})
    with pytest.raises(gd.GrpoDataError, match="is not the campaign the anchors read: \\['generation_seed_base'\\]"):
        build_into(world, tmp_path / "o5", anchors=moved)
    with pytest.raises(gd.GrpoDataError, match="renders a local judge's prompts only"):
        gd._refuse_non_local(types.SimpleNamespace(model_type="openai_batch", reasoning_effort=None))
    with pytest.raises(gd.GrpoDataError, match="renders a local judge's prompts only"):
        gd._refuse_non_local(types.SimpleNamespace(model_type="local", reasoning_effort="high"))


def test_the_training_scenarios_must_be_anchored_once_each(world, tmp_path):
    lines = world.population.read_text().splitlines()
    cases = {
        "twice": ([lines[0], lines[1], lines[0]], "names scenario .* twice"),
        "stranger": ([lines[0], json.dumps({"prompt": "new 0", "response": "r"})], "has no anchor row"),
        "empty": ([], "holds no training scenario"),
    }
    for name, (rows, match) in cases.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(line + "\n" for line in rows))
        mark(path, protected=world.protected)
        with pytest.raises(gd.GrpoDataError, match=match):
            build_into(world, tmp_path / f"out-{name}", scenarios=path)


def _drop_rule_2(rows):
    return [row for row in rows if not (row["rule_id"] == "Rule 2" and row["decoder_draw_id"] == 1)]


def _other_weight(rows):
    return [{**row, "weight": 0.3} if i == 0 else row for i, row in enumerate(rows)]


def _other_source(rows):
    return [{**row, "source": "top_up"} if i == 0 else row for i, row in enumerate(rows)]


def _flip_label(rows):
    first = rows[0]
    flipped = {"no": "yes", "yes": "no", "unsure": "no"}[first["label"]]
    return [{**first, "label": flipped}, *rows[1:]]


def _other_prompt(rows):
    return [{**rows[0], "prompt_sha256": "0" * 64}, *rows[1:]]


def _other_raw(rows):
    return [{**rows[0], "raw_output_sha256": "0" * 64}, *rows[1:]]


def _scattered(rows):
    return [*rows, rows[0]]


def _no_scenario_0(rows):
    first = rows[0]["scenario_id"]
    return [row for row in rows if row["scenario_id"] != first]


def _no_root(rows):
    return [{key: value for key, value in rows[0].items() if key != "root"}, *rows[1:]]


def _other_root(rows):
    return [{**rows[0], "root": "/elsewhere/anchors-train"}, *rows[1:]]


@pytest.mark.parametrize(
    "edit,match",
    [
        (_drop_rule_2, "does not hold each of its 2 rules x 2 draws once"),
        (_other_weight, "are not the bank draws at weight 1/2"),
        (_other_source, "are not the bank draws at weight 1/2"),
        (_flip_label, "Rule 1: the replayed labels .* are not its p-hat"),
        (_other_prompt, "are not the ones the base answered"),
        (_other_raw, "1 replayed completion\\(s\\) have no raw output"),
        (_scattered, "does not keep scenario .* rows together"),
        (_no_scenario_0, "1 training scenario\\(s\\) have no replay rows"),
        # the root is the one of its input that judged the scenario, where its raw output is read
        (_no_root, "its replay rows name the root\\(s\\) \\['/.*/anchors-train', 'None'\\], but the bank root that "
                   "judged it is /.*/anchors-train;"),
        (_other_root, "its replay rows name the root\\(s\\) \\[.*'/elsewhere/anchors-train'.*\\], but the bank root that "
                      "judged it is /.*/anchors-train;"),
    ],
)
def test_a_replay_index_that_is_not_the_draws_behind_q_hat_is_refused(world, tmp_path, edit, match):
    anchors = copy_anchors(world, tmp_path, index=edit)
    with pytest.raises(gd.GrpoDataError, match=match):
        build_into(world, tmp_path / "out", anchors=anchors)
    assert not (tmp_path / "out.partial").exists()


@pytest.mark.parametrize("column,value", [("q_yes", 0.5), ("q_no", 0.25), ("mode", "no"), ("anchor_class", "b"),
                                          ("base_yes_rules", [])])
def test_an_anchor_row_that_is_not_its_replayed_labels_estimate_is_refused(world, tmp_path, column, value):
    anchors = copy_anchors(world, tmp_path, anchor_rows=lambda rows: [{**rows[0], column: value}, *rows[1:]])
    with pytest.raises(gd.GrpoDataError, match=f"its anchor row's \\['{column}'\\] are not the product form"):
        build_into(world, tmp_path / "out", anchors=anchors)
    assert not (tmp_path / "out.partial").exists()


def test_raw_outputs_changed_in_place_during_the_build_are_refused(world, tmp_path):
    root = tmp_path / "bank-copy"
    shutil.copytree(world.root, root)
    anchor_set = gd.AnchorSets.open(bank_moved_to(world, tmp_path, root))
    kept = {scenario.scenario_id: scenario for scenario in gd.read_training_scenarios(world.population, anchor_set)}
    sources = {name: anchor_set.sources(name, evaluation_mode=gd.EVALUATION_MODE_BANK) for name in ("bank", "top_up")}
    plan = gd.replay_plan(anchor_set, kept, Ruleset.load(world.rules).rule_ids, gd.judging_roots(sources))
    wanted = {key: digest for key, digest in plan.items() if key.source == "bank"}
    assert {key.root for key in wanted} == {str(root)}
    raw_file = root / "gemma/individual_bank/draw-000/raw_outputs.jsonl"
    with gd.RawTexts(sources, wanted) as raw:
        key = min((key for key in wanted if key.decoder_draw_id == 0), key=raw._offsets.__getitem__)
        assert raw.read(key)["raw_output_sha256"] == wanted[key]
        # the row's text edited in place, at the same length: every offset still lands on a row
        data = raw_file.read_bytes()
        at = data.index(b"<think>", raw._offsets[key])
        raw_file.write_bytes(data[:at] + b"<thonk>" + data[at + len(b"<think>"):])
        raw._handles[("bank", str(root), 0)].close()
        raw._handles[("bank", str(root), 0)] = raw_file.open("rb")
        with pytest.raises(gd.GrpoDataError, match=f"raw outputs \\(of {root}\\) changed during the build"):
            raw.read(key)


def test_the_manifest_counts_completions_holding_a_special_tokens_text(world, tmp_path, monkeypatch):
    # every fake completion opens with <think> and closes a JSON object
    monkeypatch.setattr(gd, "special_texts", lambda tokenizer, stop_ids: (["<think>"], ["}"]))
    replay = build_into(world, tmp_path / "out")["replay"]
    assert replay["completions_quoting_stop_tokens"] == replay["completions_with_other_special_tokens"] == 24


def bank_moved_to(world, tmp_path, root: Path) -> Path:
    """A copy of the anchor set whose bank root is ``root``: its manifest record and its bank replay rows name it."""
    return copy_anchors(
        world, tmp_path, manifest=lambda m: {**m, "bank": {**m["bank"], "root": str(root)}},
        index=lambda rows: [{**row, "root": str(root)} if row["source"] == "bank" else row for row in rows],
    )


def bank_copy(world, tmp_path, edit) -> Path:
    """An anchor set whose bank root is a copy with draw 0's raw outputs edited (``edit``: text to text)."""
    root = tmp_path / "bank-copy"
    shutil.copytree(world.root, root)
    raw = root / "gemma/individual_bank/draw-000/raw_outputs.jsonl"
    raw.write_text(edit(raw.read_text()))
    return bank_moved_to(world, tmp_path, root)


def test_a_replayed_trace_must_have_finished(world, tmp_path):
    anchors = bank_copy(world, tmp_path, lambda text: "".join(
        json.dumps({**json.loads(line), "finish_reason": "length"}) + "\n" for line in text.splitlines()))
    with pytest.raises(gd.GrpoDataError, match="finished by 'length', not by stopping"):
        build_into(world, tmp_path / "out", anchors=anchors)
    assert not (tmp_path / "out.partial").exists()


def test_the_raw_outputs_are_read_line_by_line_as_the_ledger_wrote_them(world, built, tmp_path):
    blank = bank_copy(world, tmp_path / "blank", lambda text: "\n" + text.replace("\n", "\n\n", 1))
    manifest = build_into(world, tmp_path / "blank" / "out", anchors=blank)
    assert (tmp_path / "blank" / "out" / gd.REPLAY_ROWS_FNAME).read_bytes() == (built.out / gd.REPLAY_ROWS_FNAME).read_bytes()
    assert manifest["replay"]["rows"] == 24
    torn = bank_copy(world, tmp_path / "torn", lambda text: text + '{"scenario_id": ')
    with pytest.raises(json.JSONDecodeError):
        build_into(world, tmp_path / "torn" / "out", anchors=torn)
    empty = write_jsonl(tmp_path / "empty.jsonl", [])
    assert list(gd.replay_groups(empty)) == []


def test_a_declared_untopped_train_set_replays_the_bank_alone(world, tmp_path):
    from ruleforge.grpo.anchor_set import build_anchor_set

    untopped = tmp_path / "untopped"
    build_anchor_set(bank=world.root, full_set=world.root, reference=world.reference, rules=world.rules,
                     side="train", out=untopped, allow_untopped=True)
    manifest = build_into(world, tmp_path / "out", anchors=untopped)
    replay = read_jsonl(tmp_path / "out" / gd.REPLAY_ROWS_FNAME)
    assert set(manifest["campaigns"]) == {"bank", "full_set"} and manifest["anchors"]["allow_untopped"] is True
    assert {row["source"] for row in replay} == {"bank"} and len(replay) == 4 * 2 * 2
    assert {row["weight"] for row in replay} == {1.0}  # every K is 2
    # every row keeps its bank estimate (the routed rows too), and the reward's column says so
    assert {row["topped_up"] for row in read_jsonl(tmp_path / "out" / gd.GRPO_ROWS_FNAME)} == {False}


def test_a_topped_up_row_with_no_top_up_root_is_refused(world, tmp_path):
    anchors = copy_anchors(world, tmp_path, manifest=lambda m: {**m, "top_up": None})
    with pytest.raises(gd.GrpoDataError, match="but the top_up root that judged it is none the anchor manifest records"):
        build_into(world, tmp_path / "out", anchors=anchors)
    assert not (tmp_path / "out.partial").exists()


def test_a_prompt_that_is_not_the_campaigns_is_refused(world, monkeypatch):
    scenarios = gd.read_training_scenarios(world.population, gd.AnchorInputs.open(world.anchors))
    condition = bank_condition(world)
    prompts = gd.CampaignPrompts.build(task_spec=gd.condition_task_spec(condition), rules=read_jsonl(world.rules),
                                       judge_type=condition.judge_type)
    tokenizer = tokenizer_of(world)
    call = functools.partial(gd.grpo_prompts, scenarios, prompts=prompts, tokenizer=tokenizer, enable_thinking=True,
                             trl_tokenizer=tokenizer)
    with pytest.raises(gd.GrpoDataError, match="not the one the base's full-set campaign sent"):
        call(full_set_digests={})
    digests = gd.full_set_prompt_digests(gd.open_source(read_anchor_set(world.anchors).manifest["full_set"],
                                                        evaluation_mode="full"))
    # every draw must have sent it: one draw's other digest refuses, though another draw's matches
    first = scenarios[0].scenario_id
    with pytest.raises(gd.GrpoDataError, match="not the one the base's full-set campaign sent"):
        call(full_set_digests={**digests, first: digests[first] | {"0" * 64}})
    tokens = [prompt.tokens for prompt in call(full_set_digests=digests)]
    assert tokens == sorted(set(tokens))
    monkeypatch.setattr(gd, "RENDER_BATCH", 3)
    assert [prompt.tokens for prompt in call(full_set_digests=digests)] == tokens
    monkeypatch.setattr(gd, "trl_prompt_ids", lambda tokenizer, conversations, kwargs: [[0] for _ in conversations])
    with pytest.raises(gd.GrpoDataError, match="TRL renders the conversation as 1 tokens, the campaign sends"):
        call(full_set_digests=digests)


def test_the_command_line_builds(world, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gd, "Block2Markers", MarkersDouble)
    out = tmp_path / "cli"
    assert gd.main(["--anchors", str(world.anchors), "--scenarios", str(world.population), "--rules", str(world.rules),
                    "--model", str(world.model), "--out", str(out), "--prompt-draws", "4", "--class-a-share-cap", "0.75",
                    "--draw-seed", "3", "--allow-no-held-out",
                    "--require-protected", str(world.protected), "--allow-no-pinned-protected"]) == 0
    printed = json.loads(capsys.readouterr().out)
    manifest = read_grpo_data(out)
    assert printed["grpo_rows"] == len(read_jsonl(out / gd.GRPO_ROWS_FNAME)) and printed["replay_rows"] == 24
    assert printed["prompt_draws"] == 4 and printed["replay_row_tokens_max"] == manifest["replay"]["row_tokens_max"]
    assert printed["draws"] == manifest["sampling_weights"]["draws"] and manifest["sampling_weights"]["seed"] == 3
    assert printed["scenarios_not_drawn"] == manifest["sampling_weights"]["scenarios_not_drawn"]
    assert printed["replay_row_tokens_p95"] == manifest["replay"]["row_tokens_p95"]
    assert manifest["allow_no_held_out"] is True and manifest["required_protected"][0]["path"] == str(world.protected)
    assert (manifest["pinned_protected"], manifest["allow_no_pinned_protected"]) == ([], True)
    assert printed["trainer"] == {
        "data.train": str(out / gd.GRPO_ROWS_FNAME), "data.replay": str(out / gd.REPLAY_ROWS_FNAME),
        "data.anchor_condition": str(world.root / "gemma/individual_bank/condition.json"),
        "data.prompt_token_cap": gd.PROMPT_TOKEN_CAP, "grpo.chat_template_kwargs": {"enable_thinking": True},
        "grpo.shuffle_dataset": True,
    }
    undeclared = ["--anchors", str(world.anchors), "--scenarios", str(world.population), "--rules", str(world.rules),
                  "--model", str(world.model), "--out", str(tmp_path / "undeclared")]
    with pytest.raises(LeakageError, match=r"frozen_500 \(fa9bdc5eb6b9\).*--allow-no-pinned-protected"):
        gd.main(undeclared, pins=PINNED_PROTECTED)
    with pytest.raises(LeakageError, match="--allow-no-held-out"):
        gd.main([*undeclared, "--allow-no-pinned-protected"], pins=PINNED_PROTECTED)
    with pytest.raises(LeakageError, match="--allow-no-held-out"):
        gd.main(undeclared)


def test_the_paper_s_build_is_the_command_line_under_the_paper_s_pins(world, tmp_path, monkeypatch, capsys):
    import runpy
    import sys

    from ruleforge.launch.programs import REPO_ROOT
    from ruleforge.launch.programs import build_grpo_data as program
    from tests.support.equivalence import tree

    monkeypatch.setattr(gd, "Block2Markers", MarkersDouble)
    monkeypatch.chdir(REPO_ROOT)  # the program runs from the checkout's top only
    out = tmp_path / "run"
    argv = ["--anchors", str(world.anchors), "--scenarios", str(world.population), "--rules", str(world.rules),
            "--model", str(world.model), "--out", str(out), "--prompt-draws", "4", "--class-a-share-cap", "0.75",
            "--draw-seed", "3", "--allow-no-held-out", "--require-protected", str(world.protected)]
    with pytest.raises(LeakageError, match=r"frozen_500 \(fa9bdc5eb6b9\).*--allow-no-pinned-protected"):
        program.main(argv)
    monkeypatch.setattr(sys, "argv", ["build_grpo_data", *argv, "--allow-no-pinned-protected"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("ruleforge.launch.programs.build_grpo_data", run_name="__main__")
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    ran = out.rename(tmp_path / "program")
    assert gd.main([*argv, "--allow-no-pinned-protected"], pins=PINNED_PROTECTED) == 0
    assert capsys.readouterr().out == text and tree(out) == tree(ran)


def test_the_command_line_declares_the_verdict_mix(world, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gd, "Block2Markers", MarkersDouble)
    common = ["--anchors", str(world.anchors), "--scenarios", str(world.population), "--rules", str(world.rules),
              "--model", str(world.model), "--prompt-draws", "4", "--repeat-cap", "2", "--allow-no-held-out",
              "--allow-no-pinned-protected"]
    out = tmp_path / "mix"
    assert gd.main([*common, "--out", str(out), "--no-share", "0.5", "--no-margin", "0.3"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["draws"] == {"no": 2, "no_under_margin": 1, "unsure": 0, "yes": 1}
    assert printed["realized_shares"]["no"] == 0.5 == read_grpo_data(out)["sampling_weights"]["no_share"]
    with pytest.raises(gd.GrpoDataError, match="declare one rule"):
        gd.main([*common, "--out", str(tmp_path / "both"), "--no-share", "0.5", "--no-margin", "0.3",
                 "--class-a-share-cap", "0.75"])
    with pytest.raises(gd.GrpoDataError, match="belongs to the judgment mix"):
        gd.main([*common, "--out", str(tmp_path / "half"), "--no-margin", "0.3"])
    assert gd._build_parser().parse_args([*common, "--out", "x"]).class_a_share_cap is None


@pytest.fixture(scope="module")
def blocked_world(world):
    """The world's campaigns again with every input in blocks, and the anchor set built from them."""
    from ruleforge.grpo.anchor_set import build_anchor_set

    base = world.base
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RULEFORGE_ARTIFACT_ROOT", str(base / "artifacts"))
        patch.setenv("RULEFORGE_MODEL_ROOT", str(base / "models"))
        blocks, banks = run_blocks(base, task_spec_factory, name="bank", population=world.population,
                                   mode="individual_bank", script=BANK_SCRIPT)
        _, fulls = run_blocks(base, task_spec_factory, name="full", population=world.population, mode="full",
                              script=FULL_SCRIPT)
        _, tops = run_blocks(base, task_spec_factory, name="topup", population=base / "topup.jsonl",
                             mode="individual_bank", script=TOP_UP_SCRIPT, block_size=1, campaign_id="anch-topup",
                             draws=4, seed_base=17)
        common = dict(bank=banks, full_set=fulls, reference=world.reference, rules=world.rules, side="train",
                      population=world.population, blocks=blocks)
        manifest = build_anchor_set(top_up=tops, out=base / "anchor-set-blocked", **common)
        untopped = build_anchor_set(out=base / "anchor-set-blocked-untopped", allow_untopped=True, **common)
    assert (manifest["schema_version"], untopped["schema_version"]) == (2, 2)
    return types.SimpleNamespace(anchors=base / "anchor-set-blocked", untopped=base / "anchor-set-blocked-untopped",
                                 banks=banks, fulls=fulls, tops=tops, blocks=blocks)


def assert_same_outputs(out, reference):
    """The rows, the scenarios and each output's marker, byte for byte."""
    for name in OUTPUT_FNAMES:
        assert (out / name).read_bytes() == (reference / name).read_bytes(), name
        assert disjoint_marker_path(out / name).read_bytes() == disjoint_marker_path(reference / name).read_bytes()


def test_a_blocked_anchor_set_gives_the_rows_of_one_root(world, built, blocked_world, tmp_path):
    manifest = build_into(world, tmp_path / "out", anchors=blocked_world.anchors)
    assert_same_outputs(tmp_path / "out", built.out)
    assert read_grpo_data(tmp_path / "out") == manifest
    for key in ("counts", "sampling_weights", "replay", "outputs", "markers", "rendering", "tokenizer"):
        assert manifest[key] == built.manifest[key], key
    # each input's roots listed in the anchor manifest's order, each with every draw's ledger and raw-output digests
    assert manifest["schema_version"] == gd.GRPO_DATA_SCHEMA_VERSION_ROOTS and built.manifest["schema_version"] == 1
    for name, roots, raw, draws in (("bank", blocked_world.banks, True, 2), ("top_up", blocked_world.tops, True, 4),
                                    ("full_set", blocked_world.fulls, False, 2)):
        condition = "gemma/full" if name == "full_set" else "gemma/individual_bank"
        records = manifest["campaigns"][name]
        assert [record["root"] for record in records] == [str(root) for root in roots]
        for root, record in zip(roots, records):
            assert record["condition_id"] == condition and len(record["draws"]) == draws
            for draw, digests in record["draws"].items():
                assert digests["attempts_sha256"] == sha256_file(root / condition / draw / "attempts.jsonl")
                assert digests.get("raw_outputs_sha256") == (
                    sha256_file(root / condition / draw / "raw_outputs.jsonl") if raw else None)
    assert manifest["anchor_condition"] == str(blocked_world.banks[0] / "gemma/individual_bank/condition.json")
    index = read_jsonl(blocked_world.anchors / REPLAY_INDEX_FNAME)
    judged = {(name, sid): str(root) for name, roots in (("bank", blocked_world.banks), ("top_up", blocked_world.tops))
              for root in roots for sid in ScenarioManifest.load(root / "scenario_manifest.json").scenario_ids}
    assert {row["root"] for row in index} == {str(blocked_world.banks[0]), *map(str, blocked_world.tops)}
    assert all(row["root"] == judged[(row["source"], row["scenario_id"])] for row in index)


def test_a_blocked_untopped_train_set_builds_and_gives_the_rows_of_one_root(world, blocked_world, tmp_path):
    from ruleforge.grpo.anchor_set import build_anchor_set

    one = tmp_path / "untopped"
    build_anchor_set(bank=world.root, full_set=world.root, reference=world.reference, rules=world.rules,
                     side="train", out=one, allow_untopped=True)
    reference = build_into(world, tmp_path / "one", anchors=one)
    manifest = build_into(world, tmp_path / "blocked", anchors=blocked_world.untopped)
    assert_same_outputs(tmp_path / "blocked", tmp_path / "one")
    assert set(manifest["campaigns"]) == {"bank", "full_set"} and manifest["anchors"]["allow_untopped"] is True
    assert manifest["counts"] == reference["counts"] and manifest["schema_version"] == 2
    assert {row["topped_up"] for row in read_jsonl(tmp_path / "blocked" / gd.GRPO_ROWS_FNAME)} == {False}


def test_a_replay_row_naming_another_block_root_is_refused(world, blocked_world, tmp_path):
    first = read_jsonl(blocked_world.anchors / REPLAY_INDEX_FNAME)[0]
    other = str(blocked_world.banks[1])
    assert first["source"] == "bank" and first["root"] == str(blocked_world.banks[0])
    target = copy_anchors(blocked_world, tmp_path, index=lambda rows: [{**rows[0], "root": other}, *rows[1:]])
    with pytest.raises(gd.GrpoDataError, match=f"name the root\\(s\\) .*'{other}'.*, but the bank root that judged it "
                                               f"is {blocked_world.banks[0]};"):
        build_into(world, tmp_path / "out", anchors=target)
    assert not (tmp_path / "out.partial").exists()


def test_a_blocked_bank_with_a_one_root_full_set_and_top_up_gives_the_rows_of_one_root(
    world, built, blocked_world, tmp_path
):
    from ruleforge.grpo.anchor_set import build_anchor_set

    anchors = tmp_path / "anchors"
    build_anchor_set(bank=blocked_world.banks, full_set=world.root, top_up=world.top_up, reference=world.reference,
                     rules=world.rules, side="train", out=anchors, population=world.population,
                     blocks=blocked_world.blocks)
    manifest = build_into(world, tmp_path / "out", anchors=anchors)
    assert_same_outputs(tmp_path / "out", built.out)
    for key in ("counts", "sampling_weights", "replay", "outputs", "markers", "rendering", "tokenizer"):
        assert manifest[key] == built.manifest[key], key
    campaigns = manifest["campaigns"]
    assert manifest["schema_version"] == gd.GRPO_DATA_SCHEMA_VERSION_ROOTS
    assert [record["root"] for record in campaigns["bank"]] == [str(root) for root in blocked_world.banks]
    assert campaigns["top_up"] == built.manifest["campaigns"]["top_up"]
    assert campaigns["full_set"] == built.manifest["campaigns"]["full_set"]


def test_bank_and_top_up_roots_that_are_not_one_condition_are_refused(world, tmp_path, monkeypatch):
    top_up = json.loads((world.anchors / ANCHORS_MANIFEST_FNAME).read_text())["top_up"]["root"]
    real = gd.open_source

    def open_source(record, *, evaluation_mode):
        source = real(record, evaluation_mode=evaluation_mode)
        if record["root"] != top_up:
            return source
        return dataclasses.replace(
            source, condition=dataclasses.replace(source.condition, extra=(*source.condition.extra, ("probe", "x"))))

    monkeypatch.setattr(gd, "open_source", open_source)
    with pytest.raises(gd.GrpoDataError,
                       match=r"replayed roots \(its bank and top-up, or its stages\) are not one condition"):
        build_into(world, tmp_path / "out")
    assert not (tmp_path / "out.partial").exists()


@pytest.fixture(scope="module")
def split_sets(world, blocked_world):
    """The world's draws as two anchor sets over disjoint halves, each built on its own."""
    from ruleforge.grpo.anchor_set import build_anchor_set

    base = world.base
    common = dict(reference=world.reference, rules=world.rules, side="train")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RULEFORGE_ARTIFACT_ROOT", str(base / "artifacts"))
        patch.setenv("RULEFORGE_MODEL_ROOT", str(base / "models"))
        first = build_anchor_set(bank=blocked_world.banks[0], full_set=blocked_world.fulls[0], out=base / "set-first",
                                 **common)
        second = build_anchor_set(bank=blocked_world.banks[1], full_set=blocked_world.fulls[1],
                                  top_up=blocked_world.tops, out=base / "set-second", **common)
    assert (first["counts"]["routed"], second["counts"]["topped_up"]) == (0, 2)
    return types.SimpleNamespace(first=base / "set-first", second=base / "set-second", blocked=blocked_world)


def test_several_anchor_sets_give_the_rows_of_one_set(world, built, split_sets, tmp_path):
    manifest = build_into(world, tmp_path / "out", anchors=[split_sets.first, split_sets.second])
    assert_same_outputs(tmp_path / "out", built.out)
    assert read_grpo_data(tmp_path / "out") == manifest
    for key in ("counts", "sampling_weights", "replay", "outputs", "markers", "rendering", "tokenizer"):
        assert manifest[key] == built.manifest[key], key
    assert manifest["schema_version"] == gd.GRPO_DATA_SCHEMA_VERSION_SETS
    assert [record["path"] for record in manifest["anchors"]] == [str(split_sets.first), str(split_sets.second)]
    assert [record["counts"]["scenarios"] for record in manifest["anchors"]] == [2, 2]
    blocked = split_sets.blocked
    for name, roots in (("bank", blocked.banks), ("top_up", blocked.tops), ("full_set", blocked.fulls)):
        assert [record["root"] for record in manifest["campaigns"][name]] == [str(root) for root in roots], name
    assert manifest["anchor_condition"] == str(blocked.banks[0] / "gemma/individual_bank/condition.json")
    assert [(entry["anchor_rows"], entry["kept"]) for entry in manifest["by_anchor_set"]] == [(2, 2), (2, 2)]
    assert sum(entry["draws"] for entry in manifest["by_anchor_set"]) == 4
    assert "strata" not in manifest["by_anchor_set"][0] and "q_hat_stages" not in manifest
    assert manifest["by_anchor_set"][1]["manifest_sha256"] == sha256_file(split_sets.second / ANCHORS_MANIFEST_FNAME)


def test_the_command_line_reads_several_anchor_sets(world, built, split_sets, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gd, "Block2Markers", MarkersDouble)
    out = tmp_path / "cli"
    assert gd.main(["--anchors", str(split_sets.first), "--anchors", str(split_sets.second),
                    "--scenarios", str(world.population), "--rules", str(world.rules), "--model", str(world.model),
                    "--out", str(out), "--prompt-draws", "4", "--class-a-share-cap", "0.75", "--allow-no-held-out",
                    "--allow-no-pinned-protected"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["kept"] == 4 and printed["anchor_rows"] == 4 and printed["replay_rows"] == 24
    assert printed["trainer"]["data.anchor_condition"] == str(
        split_sets.blocked.banks[0] / "gemma/individual_bank/condition.json")
    assert read_grpo_data(out)["schema_version"] == gd.GRPO_DATA_SCHEMA_VERSION_SETS
    for name in OUTPUT_FNAMES:
        assert (out / name).read_bytes() == (built.out / name).read_bytes(), name


def test_anchor_sets_must_be_named_once_disjoint_and_one_estimate(world, split_sets, tmp_path):
    with pytest.raises(gd.GrpoDataError, match="no anchor set given"):
        gd.AnchorSets.open([])
    with pytest.raises(gd.GrpoDataError, match="an anchor set is named twice"):
        gd.AnchorSets.open([split_sets.first, split_sets.first])
    with pytest.raises(gd.GrpoDataError, match="anchors 2 scenario\\(s\\) an earlier anchor set anchors"):
        gd.AnchorSets.open([world.anchors, split_sets.second])
    second = types.SimpleNamespace(anchors=split_sets.second)
    edits = {
        "estimator": lambda m: {**m, "estimator": "other"},
        "rules_sha256": lambda m: {**m, "rules": {**m["rules"], "sha256": "0" * 64}},
        "reference": lambda m: {**m, "reference": {**m["reference"], "fingerprint": "f" * 64}},
    }
    for key, edit in edits.items():
        other = copy_anchors(second, tmp_path / key, manifest=edit)
        with pytest.raises(gd.GrpoDataError, match=f"is not one estimate with .*: their \\['{key}'\\] differ"):
            gd.AnchorSets.open([split_sets.first, other])
    moved = copy_anchors(second, tmp_path / "path", manifest=lambda m: {**m, "rules": {**m["rules"], "path": "x"}})
    assert set(gd.AnchorSets.open([split_sets.first, moved]).rows) == set(gd.AnchorSets.open(world.anchors).rows)
    assert gd.AnchorSets.open([split_sets.first, moved]).directory == f"{split_sets.first}, {moved}"


def test_a_replay_index_row_of_another_sets_scenario_is_refused(world, split_sets, tmp_path):
    foreign = read_jsonl(split_sets.second / REPLAY_INDEX_FNAME)[:1]
    first = copy_anchors(types.SimpleNamespace(anchors=split_sets.first), tmp_path,
                         index=lambda rows: [*rows, *foreign])
    with pytest.raises(gd.GrpoDataError, match=f"holds replay rows of scenario {foreign[0]['scenario_id'][:12]}, which "
                                               "its anchor set does not anchor"):
        build_into(world, tmp_path / "out", anchors=[first, split_sets.second])
    assert not (tmp_path / "out.partial").exists()


def test_a_campaign_root_that_two_anchor_sets_read_is_refused(world, split_sets, tmp_path):
    full = json.loads((split_sets.first / ANCHORS_MANIFEST_FNAME).read_text())["full_set"]
    second = copy_anchors(types.SimpleNamespace(anchors=split_sets.second), tmp_path,
                          manifest=lambda m: {**m, "full_set": full})
    with pytest.raises(gd.GrpoDataError, match=f"full_set root {full['root']} is an input of two anchor sets"):
        build_into(world, tmp_path / "out", anchors=[split_sets.first, second])


def test_full_set_roots_that_are_not_one_condition_are_refused(world, split_sets, tmp_path, monkeypatch):
    second_full = json.loads((split_sets.second / ANCHORS_MANIFEST_FNAME).read_text())["full_set"]["root"]
    real = gd.open_source

    def open_source(record, *, evaluation_mode):
        source = real(record, evaluation_mode=evaluation_mode)
        if record["root"] != second_full:
            return source
        return dataclasses.replace(
            source, condition=dataclasses.replace(source.condition, extra=(*source.condition.extra, ("probe", "x"))))

    monkeypatch.setattr(gd, "open_source", open_source)
    with pytest.raises(gd.GrpoDataError, match="full-set roots are not one condition"):
        build_into(world, tmp_path / "out", anchors=[split_sets.first, split_sets.second])
    assert not (tmp_path / "out.partial").exists()


def restacked(root: Path, out: Path) -> Path:
    """A copy of campaign root ``root`` whose every draw records another inference stack (vLLM 0.19, torch 2.10)."""
    shutil.copytree(root, out)
    provenances = sorted(out.rglob("draw_provenance.json"))
    assert provenances
    for path in provenances:
        payload = json.loads(path.read_text())
        payload["runtime"].update(vllm_version="0.19.0+cu128", torch_version="2.10.0+cu128")
        path.write_text(json.dumps(payload))
        restamp_stacks(path.parent)
    return out


def test_anchor_sets_on_two_inference_stacks_are_refused(world, split_sets, tmp_path):
    from ruleforge.grpo.anchor_set import AnchorError, build_anchor_set

    blocked = split_sets.blocked
    bank = restacked(blocked.banks[1], tmp_path / "bank")
    full = restacked(blocked.fulls[1], tmp_path / "full")
    tops = [restacked(root, tmp_path / f"top-{i}") for i, root in enumerate(blocked.tops)]
    common = dict(reference=world.reference, rules=world.rules, side="train")
    with pytest.raises(AnchorError, match="different inference stacks"):
        build_anchor_set(bank=bank, full_set=blocked.fulls[1], top_up=tops, out=tmp_path / "mixed", **common)
    second = tmp_path / "second"
    build_anchor_set(bank=bank, full_set=full, top_up=tops, out=second, **common)
    assert set(gd.AnchorSets.open([split_sets.first, second]).rows) == set(gd.AnchorSets.open(world.anchors).rows)
    with pytest.raises(gd.GrpoDataError, match="the anchor sets' inputs are not on one inference stack: .*different "
                                               "inference stacks .*0\\.19\\.0\\+cu128"):
        build_into(world, tmp_path / "out", anchors=[split_sets.first, second])
    assert not (tmp_path / "out.partial").exists()


def save_tiny_qwen3(directory: Path) -> Path:
    """A random two-layer Qwen3 over the byte tokenizer, shipping no truncation (the fake campaigns' decoder)."""
    import torch
    import transformers

    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=264, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=8192, tie_word_embeddings=False,
        eos_token_id=258, pad_token_id=256, bos_token_id=256,
    )
    transformers.Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(str(directory))
    (directory / "generation_config.json").write_text(json.dumps({"do_sample": True, "eos_token_id": [258, 256],
                                                                  "pad_token_id": 256}))
    return directory


#: The reward the dry run trains under: its functions and weights.
REWARD_NAMES = ["flat_soft_reward", "internal_consistency_reward", "map_completeness_reward"]
REWARD_WEIGHTS = [1.0, 0.3, 0.1]


@pytest.mark.torch
def test_the_cpu_dry_run_trainer_reads_what_the_build_writes(world, built, tmp_path, monkeypatch):
    pytest.importorskip("torch", reason="needs the grpo group (CPU torch is enough)")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    import transformers

    from ruleforge.grpo import trainer as tn
    from ruleforge.grpo.kl_anchor import IGNORE_INDEX
    from ruleforge.grpo.rewards import TRUNCATED_ZERO, GrpoReward

    assert gd.IGNORE_INDEX == IGNORE_INDEX

    model = tmp_path / "models" / JUDGE_DIR
    shutil.copytree(world.model, model)
    save_tiny_qwen3(model)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    overrides = [
        "run_name=dry-run", f"model.path={model}", f"data.train={built.out / gd.GRPO_ROWS_FNAME}",
        f"data.replay={built.out / gd.REPLAY_ROWS_FNAME}", f"data.anchor_condition={built.manifest['anchor_condition']}",
        f"data.rules={world.rules}", f"data.rules_sha256={sha256_file(world.rules)}", "reward.truncated=zero", "sync_probe.max_rows=8", "grpo.num_generations=2",
        "grpo.per_device_train_batch_size=2", "grpo.gradient_accumulation_steps=1", "grpo.max_completion_length=8",
        "grpo.max_steps=1", "grpo.save_steps=1", "grpo.warmup_steps=0", "grpo.report_to=[]", "anchor.anchor_accum=2",
        "+grpo.use_cpu=true",
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name="training/grpo/base", overrides=overrides), resolve=True)
    # the reward reads the ids the model's tokenizer gives (ruleforge.judges.completion_format)
    plan = tn.plan_grpo(cfg, reward_factory=GrpoReward, marker_checker=MarkersDouble().checker)
    assert (plan.reward_set.finish_token_ids, plan.reward_set.think_end_token_id) == ((258, 256), 260)
    assert plan.train_rows == built.grpo and len(plan.replay) == len(built.replay)
    assert plan.replay.weight_scale == pytest.approx(1.0) and plan.replay.sha256 == built.manifest["outputs"][gd.REPLAY_ROWS_FNAME]["sha256"]
    assert plan.provenance["prompt_tokens"]["max"] == built.manifest["counts"]["prompt_tokens_max"]
    assert set(plan.provenance["identity"]["digests"]["markers"]) == {"train", "replay"}
    assert isinstance(plan.reward_set, GrpoReward) and plan.reward_set.truncated == TRUNCATED_ZERO
    trainer = tn.build_trainer(plan, grpo_overrides={"use_vllm": False, "use_cpu": True})
    logged: list[dict] = []

    class Logs(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logged.append(dict(logs or {}))

    trainer.add_callback(Logs())
    # what TRL feeds its engine for each row is the campaign's token sequence
    tokenizer = tokenizer_of(world)
    ids, _, _ = trainer._tokenize_prompts([row["prompt"] for row in built.grpo])
    entries = [{"system_prompt": row["prompt"][0]["content"], "user_prompt": ""} for row in built.grpo]
    assert [list(row) for row in ids] == gd.campaign_prompt_ids(tokenizer, entries, enable_thinking=True)
    trainer.train()
    seen = {key: value for entry in logged for key, value in entry.items()}
    for key in ("consistency/valid_rate", "consistency/margin_0.4-1.0/share", "rewards/flat_soft_reward/mean", "anchor/kl"):
        assert key in seen, key
    assert {key.split("/")[1] for key in seen if key.startswith("rewards/") and key.endswith("/mean")} == set(REWARD_NAMES)
    assert trainer.reward_weights.tolist() == pytest.approx(REWARD_WEIGHTS)
    record = json.loads((Path(trainer.args.output_dir) / "checkpoint-1" / tn.PROVENANCE_FNAME).read_text())
    assert record["grpo_config"]["reward_weights"] == REWARD_WEIGHTS
    assert seen["anchor/kl"] == 0.0
    assert trainer.anchor_manager.rows_served == 2 and trainer.state.global_step == 1


def test_a_staged_set_s_inputs_and_stages_are_read_from_its_rows_and_recorded_by_stage_and_stratum():
    assert gd.replay_inputs({"stages": ["confirm", "yes_side"]}) == ["confirm", "yes_side"]
    assert gd.row_source({"stage": "confirm", "topped_up": False}) == "confirm"
    assert gd.row_source({"topped_up": True}) == gd.SOURCE_TOP_UP
    kept = {sid: gd.TrainingScenario(sid, {}, "", {**_verdict_row(verdict, margin), "stage": stage, "k": k})
            for sid, verdict, margin, stage, k in (("a", "no", 0.5, "confirm", 2), ("b", "yes", 0.4, "confirm", 2),
                                                   ("c", "no", 0.1, "yes_side", 4), ("d", "unsure", 1.0, "yes_side", 4))}
    weights = gd.SamplingWeights(copies={"a": 3, "b": 1, "c": 2, "d": 1}, order=list(kept),
                                 declaration={"stratification": "favoured_verdict", "no_margin": 0.3})
    record = gd.q_hat_stages(kept, weights)
    assert record["by_scenario"] == {"a": "confirm", "b": "confirm", "c": "yes_side", "d": "yes_side"}
    assert record["rows"] == {"confirm": 2, "yes_side": 2} and record["k"] == {"confirm": [2], "yes_side": [4]}
    assert record["draws"] == {"confirm": 4, "yes_side": 3}
    assert record["strata"]["confirm"]["no"] == {"rows": 1, "draws": 3}
    assert record["strata"]["yes_side"]["no_under_margin"] == {"rows": 1, "draws": 2}
    assert record["strata"]["yes_side"]["unsure"] == {"rows": 1, "draws": 1}
    plain = gd.SamplingWeights(copies=weights.copies, order=weights.order, declaration={})
    assert "strata" not in gd.q_hat_stages(kept, plain)


def rows_copy(built, tmp_path, edit=None) -> Path:
    copy = tmp_path / "rows"
    shutil.copytree(built.out, copy)
    if edit is not None:
        manifest = json.loads((copy / gd.MANIFEST_FNAME).read_text())
        edit(manifest["tokenizer"])
        (copy / gd.MANIFEST_FNAME).write_text(json.dumps(manifest))
    return copy / gd.GRPO_ROWS_FNAME


def test_rows_built_on_other_tokenizer_files_than_the_checkpoints_are_refused(world, built, tmp_path):
    model = tmp_path / "models" / JUDGE_DIR
    shutil.copytree(world.model, model)
    rows = rows_copy(built, tmp_path)
    assert gd.check_rows_model(rows, model)["tokenizer_files"] == built.manifest["tokenizer"]["files"]
    config = model / "tokenizer_config.json"
    config.write_text(config.read_text() + "\n")
    with pytest.raises(gd.GrpoDataError, match=r"holds other \['tokenizer_config.json'\]"):
        gd.check_rows_model(rows, model)


def test_rows_record_the_base_weights_they_replay_and_rows_of_other_weights_are_refused(world, built, tmp_path):
    from ruleforge.judges.weights import weights_digest

    # the test base is a tokenizer alone: no weights to record
    assert gd.base_weights_sha256(world.model) is None and built.manifest["base_weights_sha256"] is None
    weighted = save_tiny_qwen3(tmp_path / "weighted")
    assert gd.base_weights_sha256(weighted) == weights_digest(weighted)
    rows = rows_copy(built, tmp_path)
    manifest = json.loads((rows.parent / gd.MANIFEST_FNAME).read_text())
    (rows.parent / gd.MANIFEST_FNAME).write_text(json.dumps({**manifest, "base_weights_sha256": "a" * 64}))
    assert gd.check_rows_model(rows, world.model, weights_sha256="a" * 64)["base_weights_sha256"] == "a" * 64
    with pytest.raises(gd.GrpoDataError, match=r"was built on base weights aaaaaaaaaaaa\.\.\., and .* holds "
                                                r"bbbbbbbbbbbb\.\.\.: train on rows built on the weights being trained"):
        gd.check_rows_model(rows, world.model, weights_sha256="b" * 64)
    # a recorded pin is never skipped for want of today's digest
    with pytest.raises(gd.GrpoDataError, match=r"was built on base weights aaaaaaaaaaaa\.\.\., and .* holds no "
                                                r"weights digested here"):
        gd.check_rows_model(rows, world.model)
    # rows recording no base digest (a tokenizer alone, or an earlier build) are compared without one
    (rows.parent / gd.MANIFEST_FNAME).write_text(json.dumps(manifest))
    assert gd.check_rows_model(rows, world.model, weights_sha256="b" * 64)["base_weights_sha256"] is None
    # weights in a format the digest does not read are refused, not passed over as no weights
    (weighted / "model.safetensors").rename(weighted / "pytorch_model.bin")
    with pytest.raises(gd.GrpoDataError, match=r"holds weights in a format .*\['pytorch_model.bin'\].* safetensors"):
        gd.base_weights_sha256(weighted)
    # nor hidden by a safetensors file beside them that the loader, reading the .bin, never opens
    shutil.copy(weighted / "pytorch_model.bin", weighted / "backup.safetensors")
    with pytest.raises(gd.GrpoDataError, match=r"holds weights in a format .*\['pytorch_model.bin'\]"):
        gd.base_weights_sha256(weighted)


def test_rows_recording_no_completion_format_are_taken_only_where_the_eos_ends_the_turn(
    world, built, tmp_path, monkeypatch,
):
    def drop(tokenizer):
        del tokenizer["completion_format"]

    rows = rows_copy(built, tmp_path, drop)
    expected = built.manifest["tokenizer"]["completion_format"]
    assert gd.check_rows_model(rows, world.model)["completion_format"] == expected
    other = CompletionFormat(family="gemma4", finish_token_ids=(1, 0), think_end_token_id=2, think_open="thought")
    monkeypatch.setattr(gd, "load_training_format", lambda model: (None, other))
    with pytest.raises(gd.GrpoDataError, match="records no completion format, and .* end of turn is not its"):
        gd.check_rows_model(rows, world.model)


def test_rows_built_with_another_completion_format_are_refused(world, built, tmp_path):
    recorded = built.manifest["tokenizer"]["completion_format"]

    def other(tokenizer):
        tokenizer["completion_format"] = {**recorded, "think_open": "<reasoning>"}

    with pytest.raises(gd.GrpoDataError, match="was built with the completion format .* training tokenizer gives"):
        gd.check_rows_model(rows_copy(built, tmp_path, other), world.model)


#: The world's two inputs read as a staged set's stages: its bank the confirm stage, its top-up the yes side.
STAGE_OF_SOURCE = {SOURCE_BANK: STAGE_CONFIRM, SOURCE_TOP_UP: STAGE_YES_SIDE}


def staged_copy(world, tmp_path, *, stage_of=None) -> Path:
    stage_of = stage_of or {}

    def manifest(recorded):
        stages = {STAGE_OF_SOURCE[name]: recorded.pop(name) for name in (SOURCE_BANK, SOURCE_TOP_UP)}
        return {**recorded, "schema_version": ANCHORS_SCHEMA_VERSION_STAGED, "stages": stages}

    def rows(anchor_rows):
        return [{**row, "stage": stage_of.get(row["scenario_id"], STAGE_OF_SOURCE[gd.row_source(row)])}
                for row in anchor_rows]

    def index(entries):
        return [{**entry, "source": STAGE_OF_SOURCE[entry["source"]]} for entry in entries]

    return copy_anchors(world, tmp_path, manifest=manifest, index=index, anchor_rows=rows)


def test_an_anchor_row_naming_no_stage_of_its_set_is_refused(world, tmp_path):
    first = read_jsonl(world.anchors / ANCHORS_FNAME)[0]["scenario_id"]
    staged = staged_copy(world, tmp_path, stage_of={first: STAGE_CALIBRATION})
    with pytest.raises(gd.GrpoDataError, match=rf"1 anchor row\(s\) name no stage of the set \(first {first[:12]}\)"):
        gd.AnchorInputs.open(staged)


def test_a_staged_build_records_its_schema_version_and_each_rows_stage(world, built, tmp_path):
    manifest = build_into(world, tmp_path / "out", anchors=staged_copy(world, tmp_path))
    assert manifest["schema_version"] == gd.GRPO_DATA_SCHEMA_VERSION_STAGED
    assert read_grpo_data(tmp_path / "out") == manifest
    assert set(manifest["campaigns"]) == {STAGE_CONFIRM, STAGE_YES_SIDE, "full_set"}
    stages = manifest["q_hat_stages"]
    assert stages["rows"] == {STAGE_CONFIRM: 2, STAGE_YES_SIDE: 2}
    assert stages["k"] == {STAGE_CONFIRM: [2], STAGE_YES_SIDE: [4]}
    assert sum(stages["draws"].values()) == 4 and "by_anchor_set" not in manifest
    for name in (gd.GRPO_ROWS_FNAME, gd.REPLAY_ROWS_FNAME):
        assert manifest["outputs"][name]["rows"] == built.manifest["outputs"][name]["rows"], name


def test_a_set_anchoring_no_training_scenario_beside_another_is_refused(world, split_sets):
    both = gd.AnchorSets.open([split_sets.first, split_sets.second])
    training = gd.read_training_scenarios(world.population, both)
    seconds = [scenario for scenario in training if scenario.scenario_id in both.sets[1].rows]
    assert seconds and len(seconds) < len(training)
    with pytest.raises(gd.GrpoDataError, match=rf"the anchor set\(s\) {re.escape(str(split_sets.first))} anchor no"):
        gd.check_training_by_set(both, seconds)


def test_a_mixed_selection_set_gives_its_yes_favoured_rows_alone(world, tmp_path):
    mixed = copy_anchors(world, tmp_path, manifest=lambda m: {**m, "selection": {"kind": MIXED_SELECTION_KIND}})
    anchor_set = gd.AnchorSets.open(mixed)
    training = gd.read_training_scenarios(world.population, anchor_set)
    favoured = Counter(gd.favoured_verdict(scenario.anchor)[0] for scenario in training)
    assert favoured["yes"] and favoured["yes"] < len(training)
    gd.check_training_by_set(anchor_set, [s for s in training if gd.favoured_verdict(s.anchor)[0] == "yes"])
    others = {verdict: count for verdict, count in sorted(favoured.items()) if verdict != "yes"}
    with pytest.raises(gd.GrpoDataError, match=(
        rf"{sum(others.values())} training scenario\(s\) anchored by .* a set built on a mixed selection, favour "
        rf"another judgment than yes \({re.escape(str(others))}\)"
    )):
        gd.check_training_by_set(anchor_set, training)


def test_several_sets_record_their_kept_rows_by_stratum_under_the_verdict_mix(world, split_sets):
    both = gd.AnchorSets.open([split_sets.first, split_sets.second])
    kept = {scenario.scenario_id: scenario for scenario in gd.read_training_scenarios(world.population, both)}
    copies = dict(zip(kept, (1, 2, 3, 4)))
    weights = gd.SamplingWeights(copies=copies, order=list(kept),
                                 declaration={"stratification": "favoured_verdict", "no_margin": 0.3})
    records = gd.by_anchor_set(both, kept, weights)
    assert [record["path"] for record in records] == [str(split_sets.first), str(split_sets.second)]
    for record, each in zip(records, both.sets):
        mine = [sid for sid in kept if sid in each.rows]
        expected = Counter(gd.verdict_stratum(*gd.favoured_verdict(kept[sid].anchor), 0.3) for sid in mine)
        assert (record["kept"], record["draws"]) == (len(mine), sum(copies[sid] for sid in mine))
        assert set(record["strata"]) == set(gd.VERDICT_STRATA)
        assert {name: entry["rows"] for name, entry in record["strata"].items() if entry["rows"]} == dict(expected)
        assert sum(entry["draws"] for entry in record["strata"].values()) == record["draws"]
