from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import types
from collections import Counter
from pathlib import Path

import openai
import pytest

from ruleforge.core import artifacts as artifacts_module
from ruleforge.evaluate import cost as cost_module
from ruleforge.evaluate import outputs as outputs_module
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate.cost import (
    DEFAULT_RATES,
    BatchCostError,
    Rates,
    Usage,
    attribute,
    group_conditions,
    load_rates,
    price,
)
from ruleforge.evaluate.campaign import build_plan, run_plan
from ruleforge.core.digests import request_namespace
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.record_names import identity_block_id
from ruleforge.evaluate.ledger import LEDGER_FNAME
from ruleforge.evaluate.identity import component_kind
from ruleforge.judges import anthropic_batch, batch_api, batch_collect
from ruleforge.judges.backend import ModelBackend
from ruleforge.judges.batch_api import BatchExecutionError, BatchUsage
from tests.test_anthropic_batch import FakeAnthropic
from tests.test_anthropic_batch import failed as anthropic_failed
from tests.test_anthropic_batch import succeeded as anthropic_succeeded
from tests.test_anthropic_batch import text as anthropic_text
from tests.test_anthropic_batch import thinking as anthropic_thinking
from tests.test_sampling_campaign import compose_campaign
from tests.test_sampling_runner import FakeBackend, write_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]
SOL, LUNA = "gpt-6.1-sol", "gpt-5.6-luna"
BATCH = "gpt-5.6-luna-batch"
#: Tokens per request of each fake judge's batches: every bucket distinct, cached and written within the input.
SOL_TOKENS = {"input_tokens": 1000, "cached_tokens": 100, "cache_write_tokens": 600, "output_tokens": 50,
              "reasoning_tokens": 40}
LUNA_TOKENS = {"input_tokens": 500, "cached_tokens": 0, "cache_write_tokens": 300, "output_tokens": 20,
               "reasoning_tokens": 16}
DESIGN_TOKENS = {"input_tokens": 2000, "cached_tokens": 400, "cache_write_tokens": 1000, "output_tokens": 80,
                 "reasoning_tokens": 70}
REAL_ARTIFACT_ROOT = os.environ.get("RULEFORGE_ARTIFACT_ROOT")
#: The usage of each arm of GPT-5.6 Luna's published prompt table; ``good`` is a retired prompt's arm, by its recorded key.
LUNA_PROMPT_ABLATION = {
    "naive-rules-above": Usage(requests=145_000, input_tokens=139_171_470, cached_tokens=168_035, cache_write_tokens=89_581_820,
                    output_tokens=14_303_525, reasoning_tokens=13_326_748),
    "good": Usage(requests=145_000, input_tokens=158_456_470, cached_tokens=164_570,
                  cache_write_tokens=109_252_001, output_tokens=12_075_142, reasoning_tokens=11_100_118),
    "forced-grpo": Usage(requests=145_116, input_tokens=175_142_608, cached_tokens=65_640,
                       cache_write_tokens=126_518_864, output_tokens=25_393_495, reasoning_tokens=17_476_684),
}
SOL_RATES = Rates(model=SOL, tier="batch", input=1.0, cached_input=0.05, cache_write=1.25, output=5.0,
                  as_of="2026-10-04", source="test")
HAIKU = "claude-haiku-5-5"
ANTHROPIC_USAGE = {"input_tokens": 1000, "cache_read_input_tokens": 150, "cache_creation_input_tokens": 50,
                   "output_tokens": 300, "output_tokens_details": {"thinking_tokens": 250}}
HAIKU_TOKENS = {"input_tokens": 1200, "cached_tokens": 150, "cache_write_tokens": 50, "output_tokens": 300,
                "reasoning_tokens": 250}


class BatchJudge(FakeBackend):
    """The runner test's fake judge, each request block answered as one paid batch."""

    def __init__(self, batch_root: Path, tokens: dict, *, model_name: str, style: str, script=None, weights=None,
                 kind_weights=None, per_character=False):
        super().__init__(style=style, script=script, seed_capability="unsupported", model_type="openai_batch")
        self.model_name, self.max_new_tokens = model_name, 65536
        self.batch_root, self.tokens, self.weights = batch_root, tokens, weights or {}
        self.kind_weights, self.per_character = kind_weights or {}, per_character

    def body(self, entry) -> dict:
        scenario, component = self.locate(entry)
        weight = self.weights.get(scenario, 1) * self.kind_weights.get(component_kind(component), 1)
        tokens = {name: count * weight for name, count in self.tokens.items()}
        if self.per_character:
            tokens.update({name: tokens[name] * len(entry["system_prompt"]) for name in cost_module.INPUT_FIELDS})
        return {"usage": {
            "input_tokens": tokens["input_tokens"], "output_tokens": tokens["output_tokens"],
            "input_tokens_details": {"cached_tokens": tokens["cached_tokens"],
                                     "cache_write_tokens": tokens["cache_write_tokens"]},
            "output_tokens_details": {"reasoning_tokens": tokens["reasoning_tokens"]}}}

    def generate_once(self, prompts, *, request_block_id="", **kwargs):
        outcomes = super().generate_once(prompts, request_block_id=request_block_id, **kwargs)
        directory = self.batch_root / hashlib.sha256(request_block_id.encode()).hexdigest()[:32]
        (directory / "outputs").mkdir(parents=True)
        usage, lines = BatchUsage(), []
        for index, entry in enumerate(prompts):
            body = self.body(entry)
            usage.add_body(body)
            lines.append(json.dumps({"custom_id": f"r{index}", "response": {"status_code": 200, "body": body}}))
        (directory / "outputs" / "part-0000.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (directory / "usage.json").write_text(json.dumps(usage.to_dict()), encoding="utf-8")
        return [dataclasses.replace(outcome, request_block_ref=str(directory), provider_request_ref=f"r{index}")
                for index, outcome in enumerate(outcomes)]


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batch"))
    monkeypatch.setattr(campaign_module, "code_state", lambda: {})
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)


def run(tmp_path, modelspec, campaign_id, *, style="naive", judge=None, config=None, model_name=None, execute=True,
        extra=None, backend_from_config=False):
    """One condition of ``campaign_id`` judged by ``judge`` under the ``config`` model config and ``extra``."""
    overrides = {"campaign_id": campaign_id, "campaign_root": str(tmp_path / campaign_id),
                 "eval_cfg.reasoning_prompt_style": style, **(extra or {})}
    if config is not None:
        overrides.update({"model@judge_model": config, "model@agent_model": config})
    if model_name is not None:
        overrides.update({"judge_model.model_name": model_name, "agent_model.model_name": model_name})
    cfg = compose_campaign(tmp_path, **overrides)
    judge = ModelBackend(cfg.judge_model) if backend_from_config else judge or FakeBackend(style=style)
    judge.max_new_tokens = getattr(judge, "max_new_tokens", 1024)
    task_spec = modelspec(reasoning_prompt_style=style, evaluation_temperature=1.0)
    plan = build_plan(cfg, backend=judge, task_spec=task_spec)
    if execute:
        run_plan(plan, backend=judge, task_spec=task_spec)
    return tmp_path / campaign_id


def anthropic_client(style: str, *, fail=None) -> FakeAnthropic:
    """The Anthropic transport test's fake client, each request answered as the fake judge answers under ``style``."""
    judge = FakeBackend(style=style)

    def respond(custom_id, params):
        kind = fail(custom_id) if fail is not None else None
        if kind is not None:
            return anthropic_failed(custom_id, kind)
        _, component = judge.locate({"system_prompt": params["system"]})
        return anthropic_succeeded(custom_id, anthropic_thinking(), anthropic_text(judge._default(component)),
                                   usage=ANTHROPIC_USAGE)

    return FakeAnthropic(responder=respond)


def run_anthropic(tmp_path, modelspec, monkeypatch, campaign_id, *, style, extra=None, client=None):
    """One condition of ``campaign_id`` judged through the real Anthropic Batch path."""
    client = client or anthropic_client(style)
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)
    monkeypatch.setattr(anthropic_batch, "default_client", lambda: client)
    return run(tmp_path, modelspec, campaign_id, style=style, config="claude-haiku-5-5-batch", extra=extra,
               backend_from_config=True)


@pytest.fixture
def roots(tmp_path, modelspec):
    """``sol``: Sol under the naive prompt (draw 0's scenario 0 rejected once: a retry batch) and the forced one."""
    batch = tmp_path / "batch"

    def judge(tokens, model, style, script=None):
        return BatchJudge(batch, tokens, model_name=model, style=style, script=script)

    retry = {(0, "full_ruleset"): ["not a label", "no"]}
    run(tmp_path, modelspec, "sol", judge=judge(SOL_TOKENS, SOL, "naive", retry), config=BATCH, model_name=SOL)
    run(tmp_path, modelspec, "sol", style="forced-grpo", judge=judge(SOL_TOKENS, SOL, "forced-grpo"),
        config=BATCH, model_name=SOL)
    run(tmp_path, modelspec, "luna", judge=judge(LUNA_TOKENS, LUNA, "naive"), config=BATCH)
    run(tmp_path, modelspec, "design", judge=judge(DESIGN_TOKENS, LUNA, "naive"), config=BATCH)
    run(tmp_path, modelspec, "local")
    run(tmp_path, modelspec, "local", style="encouraged", execute=False)
    (batch / "f").mkdir()  # a batch no root names, with no usage: prepared, in flight, or ended without one
    (batch / "e").mkdir()
    (batch / "e" / "usage.json").write_text(json.dumps({"requests": 2, **{k: v * 2 for k, v in LUNA_TOKENS.items()}}))
    (batch / "stray.txt").write_text("not a batch")
    return {name: tmp_path / name for name in ("sol", "luna", "design", "local")} | {"batch": batch}


def by_label(attribution):
    return {condition.label: condition for condition in attribution.conditions}


def main(argv):
    """``ruleforge campaign cost ARGV``, run in this process: its exit status."""
    from ruleforge.cli.app import main as ruleforge_main

    return ruleforge_main(["campaign", "cost", *[str(item) for item in argv]])


def test_each_condition_sums_the_batches_its_ledgers_name(roots):
    attribution = attribute([roots["sol"], roots["luna"], roots["local"]], roots["batch"])
    conditions = by_label(attribution)
    assert sorted(conditions) == ["local/gemma-4-31B-it/encouraged", "local/gemma-4-31B-it/naive",
                                  "luna/gpt-5.6-luna/naive",
                                  "sol/gpt-6.1-sol/forced-grpo", "sol/gpt-6.1-sol/naive"]
    naive = conditions["sol/gpt-6.1-sol/naive"]
    # Two draws of one 9-request block each, and draw 0's retry block of scenario 0 alone: three batches.
    assert (len(naive.batches), naive.draws, naive.bundles, naive.grid_requests) == (3, 2, 6, 18)
    assert naive.usage == Usage(requests=21, **{name: count * 21 for name, count in SOL_TOKENS.items()})
    assert naive.condition.model_name == SOL and naive.condition.prompt_style == "naive"
    forced = conditions["sol/gpt-6.1-sol/forced-grpo"]
    assert (len(forced.batches), forced.usage.requests, forced.bundles, forced.grid_requests) == (2, 18, 6, 18)
    # A local judge names no batch; a condition with no draw has nothing at all.
    local, unrun = conditions["local/gemma-4-31B-it/naive"], conditions["local/gemma-4-31B-it/encouraged"]
    assert (local.batches, local.usage, local.bundles, unrun.draws, unrun.grid_requests) == ((), Usage(), 6, 0, 0)
    # Every batch the given roots name is theirs; the design root's two, "e" (with usage) and "f" (without) are not.
    named = {key for condition in attribution.conditions for key in condition.batches}
    assert sorted(attribution.unattributed) == sorted(
        {path.name for path in roots["batch"].iterdir() if path.is_dir()} - named - {"f"})
    assert len(attribution.unattributed) == 3 and attribution.unread == ("f",) and attribution.missing == {}
    assert len(naive.scenarios) == 3 and naive.measured and not local.measured and not unrun.measured


def test_a_one_draw_screening_pilot_of_a_batch_judge_runs_and_is_read(tmp_path, modelspec):
    batch = tmp_path / "batch"
    for style in ("naive", "forced-grpo"):
        run(tmp_path, modelspec, "pilot", style=style, config=BATCH, model_name=SOL,
            judge=BatchJudge(batch, SOL_TOKENS, model_name=SOL, style=style),
            extra={"sampling.campaign_mode": "screening", "sampling.num_decoder_draws": 1})
    manifest = json.loads((tmp_path / "pilot" / "sampling_manifest.json").read_text())
    assert (manifest["campaign_mode"], manifest["expected_decoder_draws"]) == ("screening", 1)
    conditions = by_label(attribute([tmp_path / "pilot"], batch))
    assert {label: (len(c.batches), c.usage.requests, c.bundles) for label, c in conditions.items()} == {
        "pilot/gpt-6.1-sol/naive": (1, 9, 3), "pilot/gpt-6.1-sol/forced-grpo": (1, 9, 3)}


def test_a_root_given_twice_is_read_once_and_two_roots_of_one_campaign_are_refused(roots, tmp_path):
    once = attribute([roots["sol"], roots["sol"]], roots["batch"])
    assert len(once.conditions) == 2 and len(once.roots) == 1
    shutil.copytree(roots["sol"], tmp_path / "copy")
    with pytest.raises(BatchCostError, match="another given root is campaign 'sol' too"):
        attribute([roots["sol"], tmp_path / "copy"], roots["batch"])


def test_a_batch_two_conditions_name_is_refused(roots):
    sol_key = by_label(attribute([roots["sol"]], roots["batch"]))["sol/gpt-6.1-sol/naive"].batches[0]
    ledger = roots["luna"] / "gpt-5.6-luna" / "naive" / "draw-000" / LEDGER_FNAME
    [luna_key] = {Path(json.loads(line)["components"][0]["request_block_ref"]).name
                  for line in ledger.read_text().splitlines()}
    ledger.write_text(ledger.read_text().replace(luna_key, sol_key))
    with pytest.raises(BatchCostError, match=f"batch {sol_key} is named by sol/gpt-6.1-sol/naive and by luna/"):
        attribute([roots["sol"], roots["luna"]], roots["batch"])


def test_usage_refuses_a_published_copy_and_a_condition_whose_finished_draws_hold_no_ledger(roots, capsys):
    from ruleforge.core.constants import PUBLISHED_COPY_FNAME
    from ruleforge.evaluate.cost import campaign_usage
    from ruleforge.evaluate.identity import PublishedCopyError

    (roots["sol"] / "gpt-6.1-sol" / "naive" / "draw-001" / LEDGER_FNAME).unlink()  # its outcomes stay
    with pytest.raises(BatchCostError, match=r"1 complete draw\(s\) hold no attempts.jsonl, which their usage is read "
                                             r"from: sol/gpt-6.1-sol/naive draw-001; read"):
        campaign_usage([roots["sol"]], roots["batch"])
    assert main(["usage", "--root", roots["sol"]]) == 2
    assert "sol/gpt-6.1-sol/naive draw-001" in capsys.readouterr().err
    _, _, attribution = campaign_usage([roots["sol"]], roots["batch"], conditions=["gpt-6.1-sol/forced-grpo"])
    assert [condition.usage.requests for condition in attribution.conditions] == [18]
    (roots["sol"] / PUBLISHED_COPY_FNAME).write_text("{}")
    with pytest.raises(PublishedCopyError, match=r"sol is a published copy of a campaign root \(published_copy.json\): "
                                                 r".*not each draw's attempt ledger and request blocks in full, which "
                                                 "campaign cost usage reads"):
        campaign_usage([roots["sol"]], roots["batch"], conditions=["gpt-6.1-sol/forced-grpo"])


def test_a_named_batch_without_usage_is_missing_not_counted(roots):
    key = by_label(attribute([roots["luna"]], roots["batch"]))["luna/gpt-5.6-luna/naive"].batches[0]
    (roots["batch"] / key / "usage.json").unlink()
    attribution = attribute([roots["luna"]], roots["batch"])
    assert attribution.missing == {key: "luna/gpt-5.6-luna/naive"}
    assert by_label(attribution)["luna/gpt-5.6-luna/naive"].usage.requests == 9
    assert key not in attribution.unattributed and key not in attribution.unread


def test_a_batch_root_that_is_not_a_directory_is_refused(roots, tmp_path):
    with pytest.raises(BatchCostError, match="is not a directory"):
        attribute([roots["sol"]], tmp_path / "nowhere")


@pytest.mark.parametrize("payload, message", [
    ([1], "holds exactly"),
    ({"requests": 1}, "holds exactly"),
    ({**dict.fromkeys(cost_module.USAGE_FIELDS, 0), "requests": -1}, "requests must be a non-negative count"),
    ({**dict.fromkeys(cost_module.USAGE_FIELDS, 0), "output_tokens": 1.5}, "output_tokens must be"),
    ({**dict.fromkeys(cost_module.USAGE_FIELDS, 0), "input_tokens": True}, "input_tokens must be"),
])
def test_a_usage_record_is_exactly_the_batch_layer_s_counts(tmp_path, payload, message):
    path = tmp_path / "usage.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(BatchCostError, match=message):
        Usage.read(path)


def test_an_unparseable_usage_record_is_refused_where_named_and_listed_where_not(roots, tmp_path):
    (tmp_path / "usage.json").write_text("{")
    with pytest.raises(BatchCostError, match="not a readable usage record"):
        Usage.read(tmp_path / "usage.json")
    (roots["batch"] / "g").mkdir()
    (roots["batch"] / "g" / "usage.json").write_text("{")
    assert attribute([roots["luna"]], roots["batch"]).unread == ("f", "g")
    key = by_label(attribute([roots["luna"]], roots["batch"]))["luna/gpt-5.6-luna/naive"].batches[0]
    (roots["batch"] / key / "usage.json").write_bytes(b"\xff")
    with pytest.raises(BatchCostError, match="not a readable usage record"):
        attribute([roots["luna"]], roots["batch"])


def test_the_batch_layer_s_usage_record_reads_as_written(tmp_path):
    from ruleforge.judges.batch_api import BatchUsage

    usage = BatchUsage(input_tokens=7, cached_tokens=1, cache_write_tokens=2, output_tokens=3, reasoning_tokens=2,
                       requests=1)
    (tmp_path / "usage.json").write_text(json.dumps(usage.to_dict()))
    assert Usage.read(tmp_path / "usage.json").to_dict() == usage.to_dict()


@pytest.mark.parametrize("by, labels", [
    ("campaign", ["sol", "luna"]),
    ("prompt_style", ["sol:prompt_style=forced-grpo", "sol:prompt_style=naive", "luna:prompt_style=naive"]),
])
def test_conditions_group_by_campaign_or_a_condition_field(roots, by, labels):
    attribution = attribute([roots["sol"], roots["luna"]], roots["batch"])
    groups = group_conditions(attribution.conditions, by)
    assert [group.label for group in groups] == labels
    assert sum(group.usage.requests for group in groups) == 21 + 18 + 18
    with pytest.raises(BatchCostError, match="cannot group by 'colour'"):
        group_conditions(attribution.conditions, "colour")


def test_input_counts_the_cached_and_written_tokens_and_output_the_reasoning():
    usage = Usage(requests=1, input_tokens=1_000_000, cached_tokens=100_000, cache_write_tokens=600_000,
                  output_tokens=50_000, reasoning_tokens=40_000)
    assert usage.fresh_input_tokens == 300_000
    assert price(usage, SOL_RATES) == pytest.approx(
        {"input": 0.30, "cached_input": 0.005, "cache_write": 0.75, "output": 0.25, "total": 1.305})


def test_cached_and_written_tokens_beyond_the_input_are_refused_and_rounding_is_not():
    with pytest.raises(BatchCostError, match="exceed the input"):
        price(Usage(input_tokens=10, cached_tokens=6, cache_write_tokens=6), SOL_RATES)
    rounded = Usage(input_tokens=1.0, cached_tokens=0.5, cache_write_tokens=0.5 + 1e-12)
    assert price(rounded, SOL_RATES)["input"] == 0.0


def test_the_committed_rates_reproduce_luna_s_bill_and_price_its_usage_at_sol_s():
    rates = load_rates(DEFAULT_RATES)
    assert sorted(rates) == [HAIKU, LUNA, SOL]
    assert [getattr(rates[SOL], name) for name in cost_module.RATE_FIELDS] == [1.0, 0.05, 1.25, 5.0]
    assert [getattr(rates[LUNA], name) for name in cost_module.RATE_FIELDS] == [0.10, 0.01, 0.125, 0.60]
    assert (rates[SOL].as_of, rates[LUNA].as_of, rates[SOL].tier) == ("2026-10-04", "2026-09-03", "batch")
    luna = {style: price(usage, rates[LUNA])["total"] for style, usage in LUNA_PROMPT_ABLATION.items()}
    sol = {style: price(usage, rates[SOL])["total"] for style, usage in LUNA_PROMPT_ABLATION.items()}
    assert round(sum(luna.values()), 2) == 86.44
    assert {style: round(cost, 2) for style, cost in sol.items()} == {
        "naive-rules-above": 232.92, "good": 245.99, "forced-grpo": 333.68}
    pilot = sum(sol[style] * 12 * 58 / usage.requests for style, usage in LUNA_PROMPT_ABLATION.items())
    assert round(pilot, 2) == 3.90


@pytest.mark.parametrize("text, message", [
    ("[1]", "holds exactly schema_version and rates"),
    ("schema_version: 2\nrates: {}", "this build reads 1"),
    ("rates: [", "not a readable YAML file"),
    ("schema_version: 1\nrates: []", "rates must map"),
    ("schema_version: 1\nrates: {m: {input: 1}}", "'m' must give exactly"),
    ("schema_version: 1\nrates: {m: [1]}", "'m' must give exactly"),
    ("schema_version: 1\nrates: {m: {tier: batch, input: -1, cached_input: 0, cache_write: 0, output: 0, "
     "as_of: x, source: y}}", "'m' input must be a non-negative price"),
    ("schema_version: 1\nrates: {m: {tier: batch, input: true, cached_input: 0, cache_write: 0, output: 0, "
     "as_of: x, source: y}}", "'m' input must be"),
    ("schema_version: 1\nrates: {m: {tier: batch, input: 1, cached_input: 0, cache_write: 0, output: 0, "
     "as_of: 2026-10-04, source: y}}", "'m' as_of must be text"),
    ("schema_version: 1\nrates: {m: {tier: batch, input: 1, cached_input: 0, cache_write: 0, output: 0, "
     "as_of: x, source: ' '}}", "'m' source must be text"),
])
def test_a_rate_file_is_refused_unless_every_price_and_its_provenance_is_given(tmp_path, text, message):
    path = tmp_path / "rates.yaml"
    path.write_text(text)
    with pytest.raises(BatchCostError, match=message):
        load_rates(path)


def test_a_restriction_to_one_kind_counts_its_requests_bundles_and_shape(roots):
    naive = by_label(attribute([roots["sol"]], roots["batch"]))["sol/gpt-6.1-sol/naive"]
    assert naive.shapes == {(1, 2)}
    full = cost_module.restrict(naive, None, kind="full_set")
    rules = cost_module.restrict(naive, frozenset(naive.scenarios), kind="rule")
    assert (full.usage.requests, full.bundles, full.grid_requests, full.shapes) == (7, 6, 6, {(1, 0)})
    assert (rules.usage.requests, rules.bundles, rules.grid_requests, rules.shapes) == (14, 6, 12, {(0, 2)})
    assert full.usage + rules.usage == naive.usage and full.batches == rules.batches == naive.batches
    assert cost_module.restrict(naive, None) is naive and cost_module.restrict(naive, frozenset(naive.scenarios)) is naive
    cache: dict = {}
    first = cost_module.restrict(naive, frozenset(naive.scenarios), cache, kind="rule")
    assert cost_module.restrict(naive, None, cache, kind="rule") is first and len(cache) == 1
    with pytest.raises(BatchCostError, match="component_kind 'bundle' is not one of"):
        cost_module.restrict(naive, None, kind="bundle")
    with pytest.raises(BatchCostError, match="attribute: component_kind 'bundle'"):
        attribute([roots["sol"]], roots["batch"], component_kind="bundle")


def test_a_restriction_reads_a_one_shot_iterable_of_scenarios(roots):
    naive = by_label(attribute([roots["sol"]], roots["batch"]))["sol/gpt-6.1-sol/naive"]
    scenario = sorted(naive.scenarios)[0]
    once = cost_module.restrict(naive, (sid for sid in [scenario]))
    assert (once.scope, once.usage) == ({scenario}, cost_module.restrict(naive, [scenario]).usage)


def test_a_restriction_stays_within_the_condition_s_own(roots):
    naive = by_label(attribute([roots["sol"]], roots["batch"]))["sol/gpt-6.1-sol/naive"]
    scenario = sorted(naive.scenarios)[0]
    cut = cost_module.restrict(naive, {scenario})
    rules = cost_module.restrict(cut, None, kind="rule")
    assert (rules.scenarios, rules.scope, rules.kind) == ({scenario}, {scenario}, "rule")
    assert rules.usage == cost_module.restrict(naive, {scenario}, kind="rule").usage
    assert cost_module.restrict(cut, sorted(naive.scenarios)) is cut and cost_module.restrict(rules, None) is rules
    with pytest.raises(BatchCostError, match="is restricted to its rule components; it has no full_set components"):
        cost_module.restrict(rules, None, kind="full_set")


def test_usage_of_one_component_kind_reads_each_request_from_the_batch_outputs(roots, tmp_path, capsys):
    out = tmp_path / "usage.json"
    assert main(["usage", "--root", str(roots["sol"]), "--component-kind", "rule", "--price", SOL,
                 "--out", str(out)]) == 0
    assert capsys.readouterr().out.startswith("rule requests only")
    record = json.loads(out.read_text())
    groups = {group["label"]: group for group in record["groups"]}
    assert {label: group["usage"]["requests"] for label, group in groups.items()} == {
        "sol/gpt-6.1-sol/naive": 14, "sol/gpt-6.1-sol/forced-grpo": 12}
    assert groups["sol/gpt-6.1-sol/naive"]["shapes"] == [{"full_set": 0, "rule": 2}]
    assert record["component_kind"] == "rule" and record["conditions"][0]["shapes"] == [{"full_set": 0, "rule": 2}]
    with pytest.raises(SystemExit):
        main(["usage", "--root", str(roots["sol"]), "--component-kind", "bundle"])
    assert "invalid choice: 'bundle'" in capsys.readouterr().err


def test_usage_over_some_scenarios_reads_each_request_from_the_batch_outputs(tmp_path, modelspec, capsys):
    batch = tmp_path / "batch"
    cut = tmp_path / "cut.jsonl"
    cut.write_text(write_inputs(tmp_path).read_text().splitlines(keepends=True)[2])
    run(tmp_path, modelspec, "luna", config=BATCH,
        judge=BatchJudge(batch, LUNA_TOKENS, model_name=LUNA, style="naive", weights={2: 5}))
    run(tmp_path, modelspec, "local")
    out = tmp_path / "usage.json"
    assert main(["usage", "--root", str(tmp_path / "luna"), "--root", str(tmp_path / "local"), "--scenarios",
                 str(cut), "--price", SOL, "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.startswith(f"over the 1 scenarios of {cut} only")
    record = json.loads(out.read_text())
    groups = {group["label"]: group for group in record["groups"]}
    assert groups["luna/gpt-5.6-luna/naive"]["usage"] == Usage(
        requests=6, **{name: count * 30 for name, count in LUNA_TOKENS.items()}).to_dict()
    assert groups["luna/gpt-5.6-luna/naive"]["retry_overhead"] == 0.0
    assert groups["local/gemma-4-31B-it/naive"]["usage"]["requests"] == 0
    assert record["scenarios"]["count"] == 1 and record["scenarios"]["sha256"]
    assert main(["usage", "--root", str(tmp_path / "luna"), "--scenarios", str(tmp_path / "nowhere.jsonl")]) == 2
    assert "is not a file" in capsys.readouterr().err


def test_a_batch_s_outputs_are_read_per_request_only_when_they_sum_to_its_usage(roots, tmp_path):
    key = by_label(attribute([roots["luna"]], roots["batch"]))["luna/gpt-5.6-luna/naive"].batches[0]
    directory = roots["batch"] / key
    part = directory / "outputs" / "part-0000.jsonl"
    lines = part.read_text().splitlines()
    usages = cost_module.request_usage(directory, "openai_batch")
    assert sorted(usages) == [f"r{index}" for index in range(9)] and usages["r0"] == Usage(requests=1, **LUNA_TOKENS)
    failed = json.dumps({"custom_id": "r9", "response": {"status_code": 500, "body": {"usage": {"input_tokens": 9}}}})
    part.write_text("\n".join([*lines, failed, ""]) + "\n")
    (directory / "outputs" / "part-0000.errors.jsonl").write_text("not json\n")
    assert sorted(cost_module.request_usage(directory, "openai_batch")) == sorted(usages)
    for text, message in ((lines[0] + "\n" + lines[0], "a result names its own custom id, once"),
                          ("[1]", "a result names its own custom id, once"),
                          ("{", "is not a result record"),
                          ("\n".join(lines[1:]), "do not sum to its usage.json")):
        part.write_text(text + "\n")
        with pytest.raises(BatchCostError, match=message):
            cost_module.request_usage(directory, "openai_batch")
    shutil.rmtree(directory / "outputs")
    with pytest.raises(BatchCostError, match="holds no downloaded outputs"):
        cost_module.request_usage(directory, "openai_batch")


def test_a_restriction_counts_a_missing_batch_and_refuses_a_component_naming_no_request(roots):
    keys = set(by_label(attribute([roots["sol"]], roots["batch"]))["sol/gpt-6.1-sol/naive"].batches)
    for key in keys:
        (roots["batch"] / key / "usage.json").unlink()
    naive = by_label(attribute([roots["sol"]], roots["batch"]))["sol/gpt-6.1-sol/naive"]
    assert set(naive.missing) == keys and naive.batch_usages == {}
    scenario = sorted(naive.scenarios)[0]
    restricted = cost_module.restrict(naive, frozenset({scenario}))
    assert restricted.missing and set(restricted.missing) <= keys and restricted.usage == Usage()
    assert not restricted.measured and len(restricted.scenarios) == 1
    ledger = roots["sol"] / "gpt-6.1-sol" / "naive" / "draw-000" / LEDGER_FNAME
    ledger.write_text(re.sub(r'"provider_request_ref": "r\d+"', '"provider_request_ref": null', ledger.read_text()))
    with pytest.raises(BatchCostError, match="names no request"):
        cost_module.restrict(by_label(attribute([roots["sol"]], roots["batch"]))["sol/gpt-6.1-sol/naive"],
                            frozenset(naive.scenarios) - {sorted(naive.scenarios)[-1]})


def test_a_restriction_skips_a_failed_request_as_the_batch_bills_it(roots):
    luna = by_label(attribute([roots["luna"]], roots["batch"]))["luna/gpt-5.6-luna/naive"]
    directory = roots["batch"] / luna.batches[0]
    part = directory / "outputs" / "part-0000.jsonl"
    records = [json.loads(line) for line in part.read_text().splitlines()]
    records[0]["response"]["status_code"] = 500
    part.write_text("".join(json.dumps(record) + "\n" for record in records))
    usage = json.loads((directory / "usage.json").read_text())
    (directory / "usage.json").write_text(json.dumps({**usage, "requests": usage["requests"] - 1, **{
        name: usage[name] - count for name, count in LUNA_TOKENS.items()}}))
    restricted = cost_module.restrict(by_label(attribute([roots["luna"]], roots["batch"]))["luna/gpt-5.6-luna/naive"],
                                     None, kind="full_set")
    assert restricted.usage == Usage(requests=5, **{name: count * 5 for name, count in LUNA_TOKENS.items()})


LUNA_REVISED_ROOT = "prompt-ablation-v2-2026-10-luna"
REVISED_PILOT_STYLES = ("encouraged", "forced")


SOL_KINDS, LUNA_KINDS = {"full_set": 2}, {"full_set": 3}


class CampaignOpenAI:
    """The OpenAI SDK client a campaign's Sol Batch judge buys through, in memory and without network."""

    def __init__(self, style: str, *, ending=None, script=None, kind_weights=SOL_KINDS):
        self.judge = FakeBackend(style=style)
        self.ending, self.script, self.kind_weights = dict(ending or {}), dict(script or {}), kind_weights
        self.uploads: dict[str, list] = {}
        self.inputs: dict[str, str] = {}
        self.answers: dict[tuple, int] = {}
        self.calls: list[tuple[str, str]] = []
        self.files = types.SimpleNamespace(create=self._create_file, content=self._content)
        self.batches = types.SimpleNamespace(create=self._create_batch, retrieve=self._retrieve)

    def _create_file(self, *, file, purpose):
        file_id = f"file{len(self.uploads)}"
        self.uploads[file_id] = [json.loads(line) for line in file.read().decode("utf-8").splitlines()]
        self.calls.append(("upload", file_id))
        return types.SimpleNamespace(id=file_id)

    def with_options(self, **options):
        return self

    def _create_batch(self, *, input_file_id, **kwargs):
        batch_id = f"batch{len(self.inputs)}"
        self.inputs[batch_id] = input_file_id
        self.calls.append(("create", batch_id))
        return types.SimpleNamespace(id=batch_id, status="validating")

    def _end(self, batch_id):
        lines = self.uploads[self.inputs[batch_id]]
        namespace = lines[0]["custom_id"].rsplit("-p", 1)[0]
        status, returned = self.ending.get(namespace, ("completed", range(len(lines))))
        return lines, status, [index for index in range(len(lines)) if index in returned]

    def _retrieve(self, batch_id):
        self.calls.append(("retrieve", batch_id))
        lines, status, returned = self._end(batch_id)
        counts = {"total": len(lines), "completed": len(returned), "failed": len(lines) - len(returned)}
        return types.SimpleNamespace(
            id=batch_id, status=status, output_file_id=f"out-{batch_id}" if returned else None,
            error_file_id=f"err-{batch_id}" if len(returned) < len(lines) else None,
            request_counts=types.SimpleNamespace(model_dump=lambda: counts))

    def _answer(self, line):
        entry = batch_api.prompt_of_body(line["body"], batch_api.RESPONSES_ENDPOINT)
        key = self.judge.locate(entry)
        count = self.answers[key] = self.answers.get(key, -1) + 1
        scripted = self.script.get(key)
        text = scripted[min(count, len(scripted) - 1)] if scripted else self.judge._default(key[1])
        weight = self.kind_weights.get(component_kind(key[1]), 1)
        tokens = {name: value * weight for name, value in SOL_TOKENS.items()}
        return {"custom_id": line["custom_id"], "response": {"status_code": 200, "body": {
            "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": tokens["input_tokens"], "output_tokens": tokens["output_tokens"],
                      "input_tokens_details": {"cached_tokens": tokens["cached_tokens"],
                                               "cache_write_tokens": tokens["cache_write_tokens"]},
                      "output_tokens_details": {"reasoning_tokens": tokens["reasoning_tokens"]}}}}}

    def _content(self, file_id):
        self.calls.append(("download", file_id))
        kind, _, batch_id = file_id.partition("-")
        lines, status, returned = self._end(batch_id)
        records = ([self._answer(lines[index]) for index in returned] if kind == "out" else
                   [{"custom_id": line["custom_id"], "response": None,
                     "error": {"code": f"batch_{status}", "message": "unfinished"}}
                    for index, line in enumerate(lines) if index not in returned])
        payload = "".join(json.dumps(record) + "\n" for record in records).encode("utf-8")
        return types.SimpleNamespace(write_to_file=lambda path: Path(path).write_bytes(payload))


def block_namespace(campaign_id: str, style: str, attempt: int = 0, draw: int = 0) -> str:
    return sent_namespace(f"{campaign_id}/{SOL}/{style}/draw-{draw:03d}/attempt-{attempt}/block-0000")


def sent_namespace(block_id: str) -> str:
    """The request namespace a judge's batch carries for the recorded block ``block_id``."""
    campaign, model, style, _ = block_id.split("/", 3)
    return request_namespace(identity_block_id(block_id, campaign_id=campaign, condition_id=f"{model}/{style}",
                                               prompt_style=style))


def run_sol(base, modelspec, monkeypatch, campaign_id, client, *, style="naive", extra=None):
    """One Sol condition through the real OpenAI Batch path against ``client``, the spend confirmed."""
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(base / "batch"))
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)
    monkeypatch.setenv("OPENAI_API_KEY", "a test's")
    monkeypatch.setattr(openai, "OpenAI", lambda api_key: client)
    return run(base, modelspec, campaign_id, style=style, config=BATCH, model_name=SOL, backend_from_config=True,
               extra={"sampling.campaign_mode": "screening", "sampling.num_decoder_draws": 1, **(extra or {})})


@pytest.fixture
def expired(tmp_path, modelspec, monkeypatch):
    """Sol's naive block expired with two of its rule requests unfinished (indices 4 and 8)."""
    base = tmp_path / "expired"
    namespace = block_namespace("sol", "naive")
    client = CampaignOpenAI("naive", ending={namespace: ("expired", {0, 1, 2, 3, 5, 6, 7})})
    with pytest.raises(BatchExecutionError, match="non-completed terminal state"):
        run_sol(base, modelspec, monkeypatch, "sol", client)
    [directory] = [path for path in (base / "batch").iterdir()]
    marker = batch_collect.collect(directory, openai_client=lambda: client)
    run(base, modelspec, "luna", config=BATCH,
        judge=BatchJudge(base / "batch", LUNA_TOKENS, model_name=LUNA, style="naive", kind_weights=LUNA_KINDS))
    complete = run_sol(tmp_path / "complete", modelspec, monkeypatch, "sol", CampaignOpenAI("naive"))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(base / "batch"))
    return {"sol": base / "sol", "luna": base / "luna", "batch": base / "batch", "key": directory.name,
            "marker": marker, "complete": complete, "client": client}


def test_an_expired_block_is_billed_to_its_condition_from_its_collected_batch(expired):
    assert expired["marker"]["requests_returned"] == 7 and expired["marker"]["missing_reasons"] == {
        "batch_expired": 2}
    assert not (expired["sol"] / f"{SOL}/naive/draw-000" / LEDGER_FNAME).exists()
    attribution = attribute([expired["sol"], expired["luna"]], expired["batch"])
    naive = by_label(attribution)[f"sol/{SOL}/naive"]
    [batch] = naive.partial
    assert (batch.key, batch.returned, batch.expected, batch.states, batch.draw) == (
        expired["key"], 7, 9, ("expired",), 0)
    assert batch.block_id == f"sol/{SOL}/naive/draw-000/attempt-0/block-0000"
    assert (naive.batches, naive.draws, naive.bundles, naive.grid_requests, naive.shapes) == (
        (expired["key"],), 1, 3, 9, frozenset({(1, 2)}))
    assert naive.partial_requests == (7, 9) and naive.measured and attribution.missing == {}
    full, rule = ({name: value * weight for name, value in SOL_TOKENS.items()} for weight in (2, 1))
    assert naive.usage == Usage(requests=7, **{name: 3 * full[name] + 4 * rule[name] for name in SOL_TOKENS})
    sent = {(row.scenario_id, component.component_id, component.provider_request_ref, component.request_block_ref,
             component.prompt_sha256) for row in batch.rows for component in row.components}
    [ledger] = outputs_module.CampaignRoot.open(expired["complete"]).ledgers(f"{SOL}/naive")
    recorded = {(record.scenario_id, component.component_id, component.provider_request_ref,
                 Path(component.request_block_ref).name, component.prompt_sha256)
                for record in ledger.load().records for component in record.components}
    assert sent == recorded and len(sent) == 9
    assert expired["key"] not in attribution.unattributed and attribution.unattributed_partial == ()
    rules = cost_module.restrict(naive, None, kind="rule")
    assert (rules.usage.requests, rules.partial_requests, rules.shapes) == (4, (4, 6), frozenset({(0, 2)}))
    full_sets = cost_module.restrict(naive, None, kind="full_set")
    assert (full_sets.usage.requests, full_sets.partial_requests) == (3, (3, 3))
    assert cost_module.combine("", [full_sets]).complete and not cost_module.combine("", [rules]).complete
    assert not {call for call, _ in expired["client"].calls} - {"upload", "create", "retrieve", "download"}


def test_the_bill_marks_a_partial_condition_returned_over_expected(expired, modelspec, capsys):
    out = expired["batch"].parent / "usage.json"
    assert main(["usage", "--root", str(expired["sol"]), "--price", SOL, "--out", str(out)]) == 0
    text = capsys.readouterr().out
    header = next(line for line in text.splitlines() if line.startswith("group"))
    assert header.split()[-3:] == ["partial", "$", SOL] and "retry" in header
    [row] = [line for line in text.splitlines() if line.startswith(f"sol/{SOL}/naive")]
    assert row.split()[-3:-1] == ["-", "7/9"]
    assert "partial: requests returned / sent by the batches that ended without completing" in text
    record = json.loads(out.read_text())
    [group] = record["groups"]
    assert group["partial"]["returned"] == 7 and group["partial"]["expected"] == 9
    assert group["partial"]["batches"][0]["from_block_plan"] and group["retry_overhead"] is None
    assert record["conditions"][0]["partial"][0]["missing_reasons"] == {"batch_expired": 2}
    run(expired["batch"].parent, modelspec, "idle", execute=False)
    assert main(["usage", "--root", str(expired["sol"]), "--root", str(expired["batch"].parent / "idle"),
                 "--component-kind", "full_set"]) == 0
    text = capsys.readouterr().out
    assert "3/3" in text and "idle/" in text
    assert main(["usage", "--root", str(expired["luna"])]) == 0
    text = capsys.readouterr().out
    assert f"1 unattributed batch(es) ended without completing (partial.json): {expired['key']}" in text
    assert "partial" not in next(line for line in text.splitlines() if line.startswith("group"))


def test_a_component_scope_retry_that_expired_is_read_from_its_plan(tmp_path, modelspec, monkeypatch):
    script = {(0, "full_ruleset"): ["not a label", "no"]}
    extra = {"sampling.retry_scope": "component"}
    namespace = block_namespace("sol", "naive", attempt=1)
    client = CampaignOpenAI("naive", script=script, ending={namespace: ("expired", set())})
    with pytest.raises(BatchExecutionError, match="non-completed terminal state"):
        run_sol(tmp_path / "expired", modelspec, monkeypatch, "sol", client, extra=extra)
    batch = tmp_path / "expired" / "batch"
    [directory] = [path for path in batch.iterdir() if not (path / "usage.json").exists()]
    marker = batch_collect.collect(directory, openai_client=lambda: client)
    assert (marker["requests_expected"], marker["requests_returned"]) == (1, 0)
    complete = run_sol(tmp_path / "complete", modelspec, monkeypatch, "sol",
                       CampaignOpenAI("naive", script=script), extra=extra)
    naive = by_label(attribute([tmp_path / "expired" / "sol"], batch))[f"sol/{SOL}/naive"]
    [partial] = naive.partial
    [row] = partial.rows
    [ledger] = outputs_module.CampaignRoot.open(complete).ledgers(f"{SOL}/naive")
    [recorded] = [record for record in ledger.load().records if record.semantic_attempt_id == 1]
    assert (row.scenario_id, row.semantic_attempt_id, [c.component_id for c in row.components]) == (
        recorded.scenario_id, 1, ["full_ruleset"]) == (recorded.scenario_id, 1, [
            c.component_id for c in recorded.components])
    assert row.components[0].prompt_sha256 == recorded.components[0].prompt_sha256
    assert naive.partial_requests == (0, 1) and naive.usage.requests == 9  # attempt 0's nine; none of attempt 1's
    # read by kind: the batch that returned nothing has no output file to read, and bills no request
    assert not (directory / "outputs" / "part-0000.jsonl").exists()
    assert cost_module.request_usage(directory, "openai_batch") == {}
    assert cost_module.request_usage(directory, "openai_batch", naive.batch_usages[directory.name]) == {}
    with pytest.raises(BatchCostError, match="is not the usage its partial.json was written with"):
        cost_module.request_usage(directory, "openai_batch", Usage(requests=1))
    full_sets = cost_module.restrict(naive, None, kind="full_set")
    assert (full_sets.usage.requests, full_sets.partial_requests) == (3, (0, 1))


def test_a_collected_batch_is_refused_unless_its_record_and_its_block_agree(expired):
    directory, roots = expired["batch"] / expired["key"], [expired["sol"]]
    marker_path, usage_path = directory / "partial.json", directory / "usage.json"
    marker, usage = marker_path.read_text(), usage_path.read_text()
    usage_path.write_text(json.dumps({**json.loads(usage), "requests": 8}))
    with pytest.raises(BatchCostError, match="usage.json is not the usage its partial.json was written with"):
        attribute(roots, expired["batch"])
    usage_path.write_text(usage)
    for text, refusal in (("{", "not a readable partial marker"), ('{"kind": "x"}', "not a partial marker"),
                          (json.dumps({**json.loads(marker), "usage": {"requests": 7}}), "is not the usage")):
        marker_path.write_text(text)
        with pytest.raises(BatchCostError, match=refusal):
            attribute(roots, expired["batch"])
    marker_path.write_text(marker)
    twin = expired["batch"] / ("0" * 32)
    shutil.copytree(directory, twin)
    with pytest.raises(BatchCostError, match=r"partial batches \[.*\] both carry request block .*'s namespace"):
        attribute(roots, expired["batch"])
    shutil.rmtree(twin)
    blocks = expired["sol"] / f"{SOL}/naive/draw-000/request_blocks.jsonl"
    record = blocks.read_text()
    blocks.write_text(record + record)
    with pytest.raises(BatchCostError, match="matched by two request blocks"):
        attribute(roots, expired["batch"])
    other = json.loads(record)
    blocks.write_text(record + json.dumps({**other, "request_block_id": other["request_block_id"][:-1] + "9"}) + "\n")
    assert by_label(attribute(roots, expired["batch"]))[f"sol/{SOL}/naive"].partial_requests == (7, 9)
    blocks.write_text(record)
    provenance = expired["sol"] / f"{SOL}/naive/draw-000/draw_provenance.json"
    original = json.loads(provenance.read_text())
    provenance.write_text(json.dumps({**original, "component_ids": original["component_ids"][:2]}))
    with pytest.raises(BatchCostError, match="its 9 request.*are not request block .*'s plan of 6"):
        attribute(roots, expired["batch"])
    provenance.write_text(json.dumps(original))
    assert by_label(attribute(roots, expired["batch"]))[f"sol/{SOL}/naive"].partial_requests == (7, 9)


def test_a_batch_named_by_its_ledger_and_marked_partial_is_counted_whole_and_marked(roots):
    key = by_label(attribute([roots["luna"]], roots["batch"]))[f"luna/{LUNA}/naive"].batches[0]
    usage = json.loads((roots["batch"] / key / "usage.json").read_text())
    (roots["batch"] / key / "partial.json").write_text(json.dumps({
        "kind": "smrc-partial-batch", "schema_version": 1, "requests_expected": usage["requests"] + 1,
        "requests_returned": usage["requests"], "missing_reasons": {"batch_expired": 1}, "custom_id_prefix": None,
        "parts": {"part-0000.jsonl": {"state": "expired"}}, "usage": usage}))
    unnamed = json.loads((roots["batch"] / "e" / "usage.json").read_text())
    (roots["batch"] / "e" / "partial.json").write_text(json.dumps({
        "kind": "smrc-partial-batch", "schema_version": 1, "requests_expected": 3, "requests_returned": 2,
        "missing_reasons": {}, "custom_id_prefix": None, "parts": {}, "usage": unnamed}))
    attribution = attribute([roots["luna"]], roots["batch"])
    luna = by_label(attribution)[f"luna/{LUNA}/naive"]
    [batch] = luna.partial
    assert (batch.key, batch.rows, batch.draw, luna.partial_requests) == (key, (), None, (9, 10))
    assert batch.to_dict()["from_block_plan"] is False and attribution.unattributed_partial == ("e",)
    assert cost_module.restrict(luna, None, kind="full_set").partial_requests == (3, 3)


@pytest.mark.cluster
def test_luna_s_prompt_table_root_is_not_priced_without_its_retired_arm():
    """The root holds a retired style's arm: a bill of the rest would leave that arm's batches out."""
    root = Path(REAL_ARTIFACT_ROOT or "/nonexistent")
    if not (root / "sampling" / "final-2026-09-luna").is_dir() or not (root / "batch").is_dir():
        pytest.skip("needs RULEFORGE_ARTIFACT_ROOT with final-2026-09-luna and the batch state")
    with pytest.raises(SamplingDesignError, match=r"gpt-5\.6-luna/good/condition\.json registers a condition of a "
                                                  r"retired style \('good'\)"):
        attribute([root / "sampling" / "final-2026-09-luna"], root / "batch")


def test_the_committed_haiku_rates_are_anthropic_s_batch_prices():
    rates = load_rates(DEFAULT_RATES)[HAIKU]
    assert [getattr(rates, name) for name in cost_module.RATE_FIELDS] == [0.05, 0.005, 0.0625, 0.25]
    assert (rates.tier, rates.as_of) == ("batch", "2026-10-13") and "platform.claude.com" in rates.source
    assert {style: round(price(usage, rates)["total"], 2) for style, usage in LUNA_PROMPT_ABLATION.items()} == {
        "naive-rules-above": 11.65, "good": 12.30, "forced-grpo": 16.68}


def test_an_anthropic_message_s_usage_prices_to_its_hand_computed_bill():
    usage = Usage.of_anthropic_message({"usage": ANTHROPIC_USAGE})
    assert usage == Usage(requests=1, **HAIKU_TOKENS)
    by_hand = {"input": 1000 * 0.05 / 1e6, "cached_input": 150 * 0.005 / 1e6, "cache_write": 50 * 0.0625 / 1e6,
               "output": 300 * 0.25 / 1e6}
    assert price(usage, load_rates(DEFAULT_RATES)[HAIKU]) == pytest.approx({**by_hand, "total": 0.000128875})


def test_an_anthropic_batch_is_read_request_by_request_as_its_transport_sums_it(tmp_path, monkeypatch):
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)
    client = FakeAnthropic(responder=lambda cid, params: anthropic_failed(cid, "errored") if cid.endswith("02-c00")
                           else anthropic_succeeded(cid, anthropic_thinking(), anthropic_text("un\u2028sure"),
                                                    usage=ANTHROPIC_USAGE))
    prompts = [{"system_prompt": f"judge {index}", "user_prompt": ""} for index in range(3)]
    requests = anthropic_batch.build_requests(prompts, model=HAIKU, max_tokens=16384, num_responses=1, seed=None,
                                              namespace="ns0")
    runner = anthropic_batch.AnthropicBatchRunner(root=tmp_path / "batch", model=HAIKU, client_factory=lambda: client,
                                                  sleep=lambda seconds: None)
    outcomes, counts = runner.run(requests)
    directory = runner.workdir(requests)
    keys = [request.custom_id for request in requests]
    usages = cost_module.request_usage(directory, "anthropic_batch")
    assert usages == {keys[0]: Usage(requests=1, **HAIKU_TOKENS), keys[1]: Usage(requests=1, **HAIKU_TOKENS)}
    assert cost_module.total(usages.values()) == Usage.read(directory / "usage.json") == Usage(**counts.to_dict())
    assert outcomes[keys[2]].error == "errored:overloaded_error"
    with pytest.raises(BatchCostError, match="billed requests do not sum to its usage.json"):
        cost_module.request_usage(directory, "openai_batch")
    expected = [(request.custom_id, artifacts_module.prompt_sha256(entry)) for request, entry in zip(requests, prompts)]
    assert cost_module.sent_requests(directory, "anthropic_batch") == expected
    part = directory / "outputs" / "part-0000.jsonl"
    part.write_text(part.read_text().replace('"output_tokens":300', '"output_tokens":301', 1))
    with pytest.raises(BatchCostError, match="do not sum to its usage.json"):
        cost_module.request_usage(directory, "anthropic_batch")


def cut_short(directory: Path, model_type: str) -> Usage:
    """A batch output file without its last billed line, and a ``usage.json`` summed from what remains."""
    read = cost_module.RESULT_READERS[model_type]
    part = directory / "outputs" / "part-0000.jsonl"
    records = [json.loads(line) for line in part.read_text(encoding="utf-8").split("\n") if line.strip()]
    last = max(index for index, record in enumerate(records) if read(record)[1] is not None)
    kept = records[:last] + records[last + 1:]
    part.write_text("".join(json.dumps(record) + "\n" for record in kept), encoding="utf-8")
    usage = cost_module.total(usage for _, usage in map(read, kept) if usage is not None)
    (directory / "usage.json").write_text(json.dumps({name: int(count) for name, count in usage.to_dict().items()}))
    assert cost_module.total(cost_module.request_usage(directory, model_type).values()) == usage
    return usage


def test_an_attribution_reads_each_usage_json_once_and_prices_the_read_its_partial_marker_was_held_to(expired,
                                                                                                     monkeypatch):
    reads = Counter()
    read = Usage.read.__func__

    def counted(cls, path):
        reads[Path(path)] += 1
        return read(cls, path)

    usage_path = expired["batch"] / expired["key"] / "usage.json"
    attribute_partial = cost_module._attribute_partial

    def rewritten(*args):
        usage_path.write_text(json.dumps({**expired["marker"]["usage"], "requests": 9}))
        return attribute_partial(*args)

    monkeypatch.setattr(Usage, "read", classmethod(counted))
    monkeypatch.setattr(cost_module, "_attribute_partial", rewritten)  # where attribute() reads it
    attribution = attribute([expired["sol"], expired["luna"]], expired["batch"])
    naive = by_label(attribution)[f"sol/{SOL}/naive"]
    marked = Usage(**expired["marker"]["usage"])
    assert naive.batch_usages == {expired["key"]: marked} and naive.usage == marked and marked.requests == 7
    assert reads[usage_path] == 1 and set(reads.values()) == {1}
    assert set(reads) == {expired["batch"] / key / "usage.json" for key in os.listdir(expired["batch"])}


def test_a_restriction_holds_its_outputs_to_the_read_of_usage_json_its_condition_priced(roots):
    luna = by_label(attribute([roots["luna"]], roots["batch"]))["luna/gpt-5.6-luna/naive"]
    one = frozenset(sorted(luna.scenarios)[:1])
    restricted = cost_module.restrict(luna, one)
    assert restricted.batch_usages == {key: luna.batch_usages[key] for key in restricted.batches}
    assert restricted.usage.requests < luna.usage.requests
    cut = {key: cut_short(roots["batch"] / key, "openai_batch") for key in restricted.batches}
    with pytest.raises(BatchCostError, match="do not sum to its usage.json"):
        cost_module.restrict(luna, one)
    fresh = by_label(attribute([roots["luna"]], roots["batch"]))["luna/gpt-5.6-luna/naive"]
    assert cost_module.restrict(fresh, one).batch_usages == cut


def test_a_transport_whose_lines_are_not_read_request_by_request_is_refused(tmp_path):
    with pytest.raises(BatchCostError, match="a 'fake' judge's input lines are not read request by request"):
        cost_module.sent_requests(tmp_path, "fake")
    with pytest.raises(BatchCostError, match="a 'fake' judge's outputs are not read request by request"):
        cost_module.request_usage(tmp_path, "fake")
