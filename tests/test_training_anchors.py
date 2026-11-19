from __future__ import annotations

import dataclasses
import json
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pytest

from ruleforge.grpo import anchor_set as anchors
from ruleforge.grpo.anchor_set import (
    ANCHORS_FNAME,
    ANCHORS_MANIFEST_FNAME,
    CLASS_A,
    CLASS_B,
    CLASS_C,
    PROJECTION_EXCLUDED,
    REPLAY_INDEX_FNAME,
    FORCED_INERT,
    ROUTE_SIDECAR_SUFFIX,
    SIDE_DEV,
    SIDE_TRAIN,
    TRAIN_SIDE_EXCLUDED,
    AnchorError,
    Ruleset,
    anchor_class,
    build_anchor_set,
    check_condition_projection,
    check_reference,
    condition_key,
    estimate,
    input_records,
    majority_labels,
    mode_of,
    one_recorded_condition,
    open_condition,
    read_anchor_set,
    read_condition_file,
    recorded_condition_key,
    replay_source,
    route_top_up,
    routes_to_top_up,
)
from ruleforge.evaluate.campaign import build_plan, load_conditions, register_condition, run_plan
from ruleforge.evaluate.identity import (
    FORCED_JSON_COMPLETE_RULE_MAP,
    FORCED_JSON_LABEL_ONLY,
    LABEL_RULE,
    ConditionSpec,
    SamplingConfig,
)
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.ledger import DrawLedger, raw_output_sha256
from ruleforge.scenarios.subset import block_name, build_blocks, sidecar_path
from tests.test_sampling_campaign import compose_campaign
from tests.test_sampling_runner import FakeBackend, forced, restamp_stacks

REPO_ROOT = Path(__file__).resolve().parents[1]
RULE_IDS = ("Rule 1", "Rule 2")
N_SCENARIOS = 4


RECORDED_LABEL_RULE = "lenient"


@contextmanager
def recorded_before_the_one_reader(label_rule: str = RECORDED_LABEL_RULE) -> Iterator[None]:
    """Inside, every campaign a config builds records the label reader ``label_rule`` as a config key."""
    original = SamplingConfig.from_mapping.__func__

    def recorded(cls, mapping):
        return dataclasses.replace(original(cls, mapping), label_rule=label_rule)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SamplingConfig, "from_mapping", classmethod(recorded))
        yield


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))


def test_the_product_form_by_hand():
    draws = [{"Rule 1": "yes", "Rule 2": "no"}, {"Rule 1": "no", "Rule 2": "unsure"}]
    got = estimate(draws, RULE_IDS)
    half, quarter = Fraction(1, 2), Fraction(1, 4)
    assert got.k == 2
    assert got.marginals == {
        "Rule 1": {"no": half, "unsure": 0, "yes": half},
        "Rule 2": {"no": half, "unsure": half, "yes": 0},
    }
    # q_no = 1/2 * 1/2; q_yes = 1 - (1 - 1/2)(1 - 0); q_unsure the rest.
    assert got.q == {"no": quarter, "unsure": quarter, "yes": half}
    # The per-draw maxima are yes and unsure: the frequency estimator sees no "no".
    assert got.q_freq == {"no": 0, "unsure": half, "yes": half}


def test_an_estimate_refuses_a_56_label_draw():
    rule_ids = tuple(f"Rule {i}" for i in range(1, 58))
    full = {rule: "no" for rule in rule_ids}
    short = dict(full)
    del short["Rule 57"]
    assert estimate([full, full], rule_ids).q["no"] == 1
    with pytest.raises(AnchorError, match="draw 1 carries 56 rule label"):
        estimate([full, short], rule_ids)
    with pytest.raises(AnchorError, match="labels outside"):
        estimate([{**full, "Rule 3": "maybe"}], rule_ids)
    with pytest.raises(AnchorError, match="at least one draw"):
        estimate([], rule_ids)


def _simulate(rng, probs, k, reps):
    """``reps`` x ``k`` draws of per-rule labels, each rule independent with ``probs[r]`` over (no, unsure, yes)."""
    labels = np.array(["no", "unsure", "yes"])
    columns = [labels[rng.choice(3, size=(reps, k), p=p)] for p in probs]
    return [
        [{f"Rule {r + 1}": columns[r][rep, draw] for r in range(len(probs))} for draw in range(k)]
        for rep in range(reps)
    ]


def _truth(probs):
    no_yes = np.prod([1 - p[2] for p in probs])
    all_no = np.prod([p[0] for p in probs])
    return {"yes": 1 - no_yes, "unsure": no_yes - all_no, "no": all_no}


def test_the_product_form_is_unbiased_and_steadier_than_the_frequency_of_maxima():
    rng = np.random.default_rng(16)
    probs = [(0.9, 0.05, 0.05), (0.7, 0.2, 0.1), (0.95, 0.0, 0.05), (0.6, 0.3, 0.1), (1.0, 0.0, 0.0), (0.8, 0.1, 0.1)]
    rule_ids = [f"Rule {r + 1}" for r in range(len(probs))]
    truth = _truth(probs)
    reps = _simulate(rng, probs, k=2, reps=12_000)
    product = {verdict: [] for verdict in truth}
    frequency = {verdict: [] for verdict in truth}
    for draws in reps:
        got = estimate(draws, rule_ids)
        for verdict in truth:
            product[verdict].append(float(got.q[verdict]))
            frequency[verdict].append(float(got.q_freq[verdict]))
    for verdict, expected in truth.items():
        for values in (product[verdict], frequency[verdict]):
            values = np.asarray(values)
            standard_error = values.std(ddof=1) / np.sqrt(len(values))
            assert abs(values.mean() - expected) < 4.5 * standard_error
        assert np.var(product[verdict]) <= np.var(frequency[verdict])


def test_the_top_up_estimate_is_unbiased_given_the_routing_and_pooling_would_not_be():
    rng = np.random.default_rng(7)
    probs = [(0.5, 0.0, 0.5), (0.8, 0.2, 0.0)]
    rule_ids = ["Rule 1", "Rule 2"]
    truth = _truth(probs)
    routing = _simulate(rng, probs, k=2, reps=10_000)
    fresh = _simulate(rng, probs, k=4, reps=10_000)
    fresh_q, pooled_q = [], []
    for first, second in zip(routing, fresh):
        if not routes_to_top_up(estimate(first, rule_ids).q):
            continue
        fresh_q.append(float(estimate(second, rule_ids).q["yes"]))
        pooled_q.append(float(estimate(first + second, rule_ids).q["yes"]))
    fresh_q, pooled_q = np.asarray(fresh_q), np.asarray(pooled_q)
    assert 0.7 < len(fresh_q) / len(routing) < 0.8  # routed unless Rule 1 said yes twice
    fresh_se = fresh_q.std(ddof=1) / np.sqrt(len(fresh_q))
    assert abs(fresh_q.mean() - truth["yes"]) < 4.5 * fresh_se
    assert truth["yes"] - pooled_q.mean() > 20 * pooled_q.std(ddof=1) / np.sqrt(len(pooled_q))


def test_the_mode_breaks_ties_toward_no_and_the_classes_follow_it():
    assert mode_of({"yes": 0.5, "unsure": 0.0, "no": 0.5}) == "no"
    assert mode_of({"yes": 0.4, "unsure": 0.4, "no": 0.2}) == "unsure"
    assert mode_of({"yes": 0.6, "unsure": 0.3, "no": 0.1}) == "yes"
    assert mode_of({"yes": Fraction(1, 3), "unsure": Fraction(1, 3), "no": Fraction(1, 3)}) == "no"
    assert anchor_class({"yes": 0.5, "unsure": 0.0, "no": 0.5}, ["yes"]) == CLASS_A
    assert anchor_class({"yes": 0.45, "unsure": 0.3, "no": 0.25}, ["yes"]) == CLASS_A
    assert anchor_class({"yes": 0.9, "unsure": 0.05, "no": 0.05}, ["yes", "no"]) == CLASS_B
    assert anchor_class({"yes": 0.9, "unsure": 0.05, "no": 0.05}, ["yes", "yes"]) == CLASS_C
    assert majority_labels({"Rule 1": {"no": 0.5, "unsure": 0.0, "yes": 0.5}}) == {"Rule 1": "no"}
    assert routes_to_top_up({"yes": Fraction(1)}) is False
    assert routes_to_top_up({"yes": Fraction(3, 4)}) is True


def _spec(**overrides) -> ConditionSpec:
    fields = dict(
        condition_id="qwen/anchor-bank",
        model_name="Qwen3-8B",
        model_type="local",
        prompt_style="forced-grpo",
        individual_rule_context="isolated",
        judge_type="No_Interpretation",
        evaluation_mode="individual_bank",
        rule_set_name="openai_rules_all.jsonl",
        rule_set_sha256="a" * 64,
        temperature=1.0,
        prompt_shape_version=2,
        forced_json_requirement=FORCED_JSON_LABEL_ONLY,
        label_rule="lenient",  # the published anchors' recorded label rule: inert under the forced prompt
        max_new_tokens=32768,
        seed_capability="local_deterministic",
        model_path="Qwen3-8B",
        enable_thinking=True,
        sampling_truncation={"top_k": 20, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0},
    )
    fields.update(overrides)
    return ConditionSpec(**fields)


VARIANTS = {
    "model_name": {"model_name": "Qwen3-8B-other"},
    "model_type": {"model_type": "openai_batch"},
    "prompt_style": {"prompt_style": "naive"},
    "individual_rule_context": {"individual_rule_context": "implicated"},
    "judge_type": {"judge_type": "Narrow"},
    "rule_set_name": {"rule_set_name": "openai_rules_all_full.jsonl"},
    "rule_set_sha256": {"rule_set_sha256": "b" * 64},
    "prompt_shape_version": {"prompt_shape_version": 3},
    "forced_json_requirement": {"forced_json_requirement": FORCED_JSON_COMPLETE_RULE_MAP},
    "validation_schema_version": {"validation_schema_version": 99},
    "endpoint": {"endpoint": "/v1/responses"},
    "reasoning_effort": {"reasoning_effort": "high"},
    "max_new_tokens": {"max_new_tokens": 6144},
    "send_max_output_tokens": {"send_max_output_tokens": True},
    "seed_capability": {"seed_capability": "provider_best_effort"},
    "model_path": {"model_path": "checkpoint-125"},
    "adapter_path": {"adapter_path": "checkpoint-125/adapter_model.safetensors"},
    "adapter_sha256": {
        "adapter_path": "checkpoint-125/adapter_model.safetensors",
        "adapter_sha256": "c" * 64,
        "adapter_config_sha256": "d" * 64,
    },
    "adapter_config_sha256": {
        "adapter_path": "checkpoint-125/adapter_model.safetensors",
        "adapter_sha256": "c" * 64,
        "adapter_config_sha256": "d" * 64,
    },
    "enable_thinking": {"enable_thinking": False},
    "sampling_truncation": {"sampling_truncation": {"top_k": 0, "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0}},
    "model_weights_sha256": {"model_weights_sha256": "e" * 64},
    "serving_dtype": {"serving_dtype": "float32"},
    "strict_think": {"strict_think": True},
    "retry_scope": {"retry_scope": "component"},
    # The hierarchy condition's levels file, which only its rule context carries.
    "rule_levels_sha256": {"prompt_style": "naive-hierarchy", "individual_rule_context": "superiors",
                           "rule_levels_sha256": "f" * 64},
    "task": {"task": "legal", "prompt_style": "naive"},  # the legal task's one style
    "response_contract": {"task": "demo", "response_contract": "forced_json"},
    "judge_extension": {"model_type": "registered", "judge_extension": "{}"},
}
#: Compared, but no ConditionSpec can carry another value (a campaign runs at T = 1 only).
UNVARIABLE = {"temperature"}


def test_every_compared_field_is_checked():
    reference = _spec(condition_id="qwen/figure5/base", evaluation_mode="both", extra=(("policy", "base"),))
    names = {f.name for f in dataclasses.fields(ConditionSpec)}
    compared = names - set(PROJECTION_EXCLUDED) - set(FORCED_INERT)
    variants = {name: overrides for name, overrides in VARIANTS.items() if name in names}
    assert set(variants) | UNVARIABLE == compared
    check_condition_projection(_spec(), reference, side=SIDE_TRAIN)
    check_condition_projection(_spec(), reference, side=SIDE_DEV)
    for field_name, overrides in variants.items():
        for side in (SIDE_TRAIN, SIDE_DEV):
            if field_name in TRAIN_SIDE_EXCLUDED and side == SIDE_TRAIN:
                check_condition_projection(_spec(**overrides), reference, side=side)
                continue
            with pytest.raises(AnchorError, match=field_name):
                check_condition_projection(_spec(**overrides), reference, side=side)
    with pytest.raises(Exception, match="temperature"):
        _spec(temperature=0.6)


def test_a_forced_condition_s_label_rule_is_inert_and_a_label_condition_s_is_compared():
    reference = _spec(condition_id="qwen/figure5/base", evaluation_mode="both", extra=(("policy", "base"),))
    for side in (SIDE_TRAIN, SIDE_DEV):
        check_condition_projection(_spec(label_rule="lenient"), dataclasses.replace(reference, label_rule=LABEL_RULE),
                                   side=side)
        with pytest.raises(AnchorError, match="label_rule: 'lenient' vs 'lenient_last_line'"):
            check_condition_projection(_spec(prompt_style="naive", label_rule="lenient"),
                                       dataclasses.replace(reference, prompt_style="naive", label_rule=LABEL_RULE),
                                       side=side)


def test_the_condition_key_sets_a_forced_condition_s_label_reader_aside():
    recorded, fresh = _spec(label_rule=RECORDED_LABEL_RULE), _spec(label_rule=LABEL_RULE)
    assert recorded.fingerprint != fresh.fingerprint
    assert condition_key(recorded) == condition_key(fresh) == fresh.fingerprint
    assert condition_key(_spec(label_rule=RECORDED_LABEL_RULE, max_new_tokens=6144)) != fresh.fingerprint
    label = _spec(prompt_style="naive", label_rule=RECORDED_LABEL_RULE)
    assert condition_key(label) == label.fingerprint != condition_key(dataclasses.replace(label, label_rule=LABEL_RULE))


def _registered(directory: Path, spec: ConditionSpec) -> dict:
    register_condition(directory, spec)
    return {"root": str(directory), "condition_id": spec.condition_id, "fingerprint": spec.fingerprint}


def test_records_are_one_condition_by_the_conditions_their_roots_register(tmp_path):
    recorded = _registered(tmp_path / "recorded", _spec(label_rule=RECORDED_LABEL_RULE))
    fresh = _registered(tmp_path / "fresh", _spec(label_rule=LABEL_RULE))
    assert recorded_condition_key(recorded) == recorded_condition_key(fresh) == fresh["fingerprint"]
    assert one_recorded_condition([recorded, fresh, fresh])
    assert one_recorded_condition([{"fingerprint": "f"}, {"fingerprint": "f"}])
    label = [_registered(tmp_path / rule, _spec(prompt_style="naive", label_rule=rule))
             for rule in (RECORDED_LABEL_RULE, LABEL_RULE)]
    assert not one_recorded_condition(label)
    gone = {**fresh, "root": str(tmp_path / "gone")}
    assert recorded_condition_key(gone) is None and not one_recorded_condition([recorded, gone])
    assert recorded_condition_key({**recorded, "fingerprint": fresh["fingerprint"]}) is None
    assert recorded_condition_key({"fingerprint": fresh["fingerprint"]}) is None


def test_the_excluded_fields_may_differ_and_the_side_is_checked():
    reference = _spec()
    check_condition_projection(
        _spec(condition_id="qwen/other", evaluation_mode="full", extra=(("role", "fullset"),)),
        reference, side=SIDE_TRAIN,
    )
    with pytest.raises(AnchorError, match="side must be one of"):
        check_condition_projection(_spec(), reference, side="eval")


@dataclasses.dataclass(frozen=True)
class _EngineSpec(ConditionSpec):
    serving_dtype: Optional[str] = None
    model_weights_sha256: Optional[str] = None


def test_the_serving_dtype_is_compared_on_the_dev_side_only():
    values = {f.name: getattr(_spec(), f.name) for f in dataclasses.fields(ConditionSpec)}
    fp32 = _EngineSpec(**{**values, "serving_dtype": "float32"})
    bf16 = dataclasses.replace(fp32, serving_dtype=None)
    check_condition_projection(bf16, fp32, side=SIDE_TRAIN)
    check_condition_projection(fp32, fp32, side=SIDE_DEV)
    with pytest.raises(AnchorError, match="serving_dtype"):
        check_condition_projection(bf16, fp32, side=SIDE_DEV)
    check_reference(fp32)
    with pytest.raises(AnchorError, match="must be the base judge's arm"):
        check_reference(dataclasses.replace(fp32, model_weights_sha256="e" * 64))
    with pytest.raises(AnchorError, match="must be the base judge's arm"):
        check_reference(_spec(adapter_path="checkpoint-125/adapter_model.safetensors"))


def test_a_reference_condition_file_is_read_with_its_fingerprint_checked(tmp_path):
    path = register_condition(tmp_path / "training", _spec(condition_id="qwen/figure5/base", evaluation_mode="both"))
    assert read_condition_file(path) == _spec(condition_id="qwen/figure5/base", evaluation_mode="both")
    payload = json.loads(path.read_text())
    payload["max_new_tokens"] = 1
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(payload))
    with pytest.raises(AnchorError, match="does not match the condition"):
        read_condition_file(forged)
    del payload["fingerprint"]
    forged.write_text(json.dumps(payload))
    with pytest.raises(AnchorError, match="not a registered condition"):
        read_condition_file(forged)
    forged.write_text("{")
    with pytest.raises(AnchorError, match="cannot read"):
        read_condition_file(forged)


def write_population(tmp_path: Path, n: int = N_SCENARIOS) -> Path:
    path = tmp_path / "population.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"prompt": f"prompt {i}", "response": f"response {i}"}) + "\n" for i in range(n)),
        encoding="utf-8",
    )
    return path


def run_condition(tmp_path, modelspec, *, root, campaign_id, mode, script, draws=2, seed_base=16,
                  population=None, condition_id=None, extra=None, draw_ids=None):
    """One condition of a pilot campaign, judged by the fake judge in the forced style."""
    population = population or write_population(tmp_path)
    overrides = {
        "campaign_id": campaign_id,
        "campaign_root": str(root),
        "scenario_set_path": str(population),
        "evaluation_mode": mode,
        "eval_cfg.reasoning_prompt_style": "forced-grpo",
        "sampling.forced_json_requirement": FORCED_JSON_LABEL_ONLY,
        "sampling.num_decoder_draws": draws,
        "sampling.generation_seed_base": seed_base,
        "condition_id": condition_id or f"gemma/{mode}",
    }
    if extra is not None:
        overrides["condition_extra"] = extra
    if draw_ids is not None:
        overrides["decoder_draw_ids"] = draw_ids
    cfg = compose_campaign(tmp_path, **overrides)
    backend = FakeBackend(style="forced-grpo", script=script)
    backend.max_new_tokens = 1024
    task_spec = modelspec(reasoning_prompt_style="forced-grpo", evaluation_temperature=1.0)
    plan = build_plan(cfg, backend=backend, task_spec=task_spec)
    run_plan(plan, backend=backend, task_spec=task_spec)
    return Path(root)


YES, NO = forced("yes"), forced("no")
BANK_SCRIPT = {(0, "Rule 1"): [YES], (0, "Rule 2"): ["not json", NO], (1, "Rule 1"): [YES], (2, "Rule 2"): [YES, NO]}
FULL_SCRIPT = {(0, "full_ruleset"): [YES], (1, "full_ruleset"): [YES, NO], (2, "full_ruleset"): [YES]}
TOP_UP_SCRIPT = {(2, "Rule 2"): [YES, NO, YES, NO]}


def make_campaign(tmp_path, modelspec):
    """A bank (K = 2) and a full-set condition in one root, the reference, and the rules."""
    root = tmp_path / "anchors-train"
    run_condition(tmp_path, modelspec, root=root, campaign_id="anch", mode="individual_bank", script=BANK_SCRIPT)
    run_condition(tmp_path, modelspec, root=root, campaign_id="anch", mode="full", script=FULL_SCRIPT)
    bank = load_conditions(root)["gemma/individual_bank"]
    reference = register_condition(
        tmp_path / "training",
        dataclasses.replace(bank, condition_id="gemma/figure5/base", evaluation_mode="both",
                            extra=(("figure", "figure5"), ("policy", "base"))),
    )
    return {"root": root, "reference": reference, "rules": tmp_path / "rules.jsonl", "tmp": tmp_path}


@pytest.fixture
def campaign(tmp_path, modelspec):
    return make_campaign(tmp_path, modelspec)


def route(campaign, **overrides):
    kwargs = dict(
        bank=campaign["root"], reference=campaign["reference"], rules=campaign["rules"], side=SIDE_TRAIN,
        population=campaign["tmp"] / "population.jsonl", out=campaign["tmp"] / "topup.jsonl",
    )
    kwargs.update(overrides)
    return route_top_up(**kwargs)


def run_top_up(campaign, modelspec, *, script=TOP_UP_SCRIPT, seed_base=17, **kwargs):
    route(campaign)
    return run_condition(
        campaign["tmp"], modelspec, root=campaign["tmp"] / "anchors-train-topup", campaign_id="anch-topup",
        mode="individual_bank", script=script, draws=4, seed_base=seed_base,
        population=campaign["tmp"] / "topup.jsonl", **kwargs,
    )


def build(campaign, **overrides):
    """Build from the fixture; the side defaults by whether a top-up is given."""
    kwargs = dict(
        bank=campaign["root"], full_set=campaign["root"], reference=campaign["reference"], rules=campaign["rules"],
        side=SIDE_TRAIN if overrides.get("top_up") else SIDE_DEV, out=campaign["tmp"] / "anchor-set",
    )
    kwargs.update(overrides)
    return build_anchor_set(**kwargs)


def sid(campaign, index):
    return open_condition(campaign["root"], None, evaluation_mode="individual_bank").scenario_ids[index]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_routing_writes_the_rows_without_a_sure_yes_verbatim(campaign):
    summary = route(campaign)
    out = campaign["tmp"] / "topup.jsonl"
    assert summary["routed"] == 2 and summary["out"] == str(out)
    source_lines = (campaign["tmp"] / "population.jsonl").read_text().splitlines()
    assert out.read_text().splitlines() == source_lines[2:]
    sidecar = json.loads(Path(str(out) + ROUTE_SIDECAR_SUFFIX).read_text())
    assert sidecar["scenario_ids"] == [sid(campaign, 2), sid(campaign, 3)]
    assert sidecar["output_sha256"] == sha256_file(out)
    assert route(campaign)["routed"] == 2
    crlf = out.read_bytes().replace(b"\n", b"\r\n")
    out.write_bytes(crlf)  # the same text, other bytes: a campaign may have frozen them
    with pytest.raises(AnchorError, match="exists with different content"):
        route(campaign)
    assert out.read_bytes() == crlf
    out.write_text("edited\n")
    with pytest.raises(AnchorError, match="exists with different content"):
        route(campaign)
    other = campaign["tmp"] / "other.jsonl"
    other.write_text((campaign["tmp"] / "population.jsonl").read_text() + "\n\n")
    with pytest.raises(AnchorError, match="not the population the bank"):
        route(campaign, population=other)


def test_routing_nothing_writes_nothing(tmp_path, modelspec):
    root = tmp_path / "sure"
    script = {(i, "Rule 1"): [YES] for i in range(N_SCENARIOS)}
    run_condition(tmp_path, modelspec, root=root, campaign_id="sure", mode="individual_bank", script=script)
    bank = load_conditions(root)["gemma/individual_bank"]
    reference = register_condition(tmp_path / "training", dataclasses.replace(bank, condition_id="f/base"))
    summary = route_top_up(
        bank=root, reference=reference, rules=tmp_path / "rules.jsonl", side=SIDE_TRAIN,
        population=tmp_path / "population.jsonl", out=tmp_path / "topup.jsonl",
    )
    assert summary["routed"] == 0 and summary["out"] is None
    assert not (tmp_path / "topup.jsonl").exists()


def test_the_anchor_set_with_a_top_up(campaign, modelspec):
    top_root = run_top_up(campaign, modelspec)
    manifest = build(campaign, top_up=top_root)
    anchor_set = read_anchor_set(campaign["tmp"] / "anchor-set")
    assert anchor_set.manifest == manifest
    rows = {row["scenario_id"]: row for row in anchor_set.rows}
    ids = [sid(campaign, i) for i in range(N_SCENARIOS)]
    assert list(rows) == ids

    assert (rows[ids[0]]["k"], rows[ids[0]]["topped_up"], rows[ids[0]]["q_yes"]) == (2, False, 1.0)
    assert rows[ids[0]]["anchor_class"] == CLASS_C and rows[ids[0]]["mode"] == "yes"
    assert rows[ids[1]]["anchor_class"] == CLASS_B and rows[ids[1]]["full_set_labels"] == ["yes", "no"]
    tied = rows[ids[2]]
    assert (tied["k"], tied["topped_up"]) == (4, True)
    assert (tied["q_yes"], tied["q_unsure"], tied["q_no"]) == (0.5, 0.0, 0.5)
    assert tied["mode"] == "no" and tied["anchor_class"] == CLASS_A
    assert tied["q_freq"] == {"no": 0.5, "unsure": 0.0, "yes": 0.5}
    assert tied["rule_marginals"]["Rule 2"] == {"no": 0.5, "unsure": 0.0, "yes": 0.5}
    assert tied["base_rule_labels"] == {"Rule 1": "no", "Rule 2": "no"}
    assert tied["base_yes_rules"] == ["Rule 2"]
    assert (rows[ids[3]]["q_no"], rows[ids[3]]["anchor_class"], rows[ids[3]]["base_yes_rules"]) == (1.0, CLASS_A, [])
    assert manifest["counts"] == {
        "scenarios": 4, "routed": 2, "topped_up": 2, "classes": {"a": 2, "b": 1, "c": 1},
        "modes": {"no": 2, "unsure": 0, "yes": 2}, "replay_rows": 2 * 2 * 2 + 2 * 2 * 4,
    }
    assert manifest["top_up"]["decoder_draws"] == 4 and manifest["top_up"]["generation_seed_base"] == 17
    assert manifest["projection_excluded"] == ["condition_id", "evaluation_mode", "extra", "serving_dtype"]
    assert manifest["rules"]["n_rules"] == 2 and manifest["estimator"] == "product_form_v1"


def test_the_replay_index_covers_exactly_the_draws_behind_q_hat(campaign, modelspec):
    top_root = run_top_up(campaign, modelspec)
    build(campaign, top_up=top_root)
    anchor_set = read_anchor_set(campaign["tmp"] / "anchor-set")
    sources = {
        "bank": (campaign["root"] / "gemma/individual_bank", 2),
        "top_up": (top_root / "gemma/individual_bank", 4),
    }
    raw = {
        name: {
            (row["scenario_id"], row["semantic_attempt_id"], row["request_block_id"], row["component_id"]): row
            for draw in range(k)
            for row in read_jsonl(path / f"draw-{draw:03d}/raw_outputs.jsonl")
        }
        for name, (path, k) in sources.items()
    }
    prompts = {
        name: {
            (row["scenario_id"], row["semantic_attempt_id"], row["request_block_id"], c["component_id"]):
                c["prompt_sha256"]
            for draw in range(k)
            for row in read_jsonl(path / f"draw-{draw:03d}/attempts.jsonl")
            for c in row["components"]
        }
        for name, (path, k) in sources.items()
    }
    for row in anchor_set.rows:
        source = "top_up" if row["topped_up"] else "bank"
        k = sources[source][1]
        index = [entry for entry in anchor_set.replay if entry["scenario_id"] == row["scenario_id"]]
        assert sorted((e["rule_id"], e["decoder_draw_id"]) for e in index) == sorted(
            (rule, draw) for rule in RULE_IDS for draw in range(k)
        )
        assert {e["source"] for e in index} == {source} and {e["weight"] for e in index} == {1 / k}
        for rule in RULE_IDS:
            mix = {label: sum(e["weight"] for e in index if e["rule_id"] == rule and e["label"] == label)
                   for label in ("no", "unsure", "yes")}
            assert mix == row["rule_marginals"][rule]
        for entry in index:
            # The ledger key names the raw output: its text hashes to the digest, from the draw named.
            text = raw[source][
                (entry["scenario_id"], entry["semantic_attempt_id"], entry["request_block_id"], entry["rule_id"])
            ]
            assert raw_output_sha256(text["raw_response"]) == entry["raw_output_sha256"]
            assert text["decoder_draw_id"] == entry["decoder_draw_id"]
            assert entry["prompt_sha256"] == prompts[source][
                (entry["scenario_id"], entry["semantic_attempt_id"], entry["request_block_id"], entry["rule_id"])
            ]
            assert f"/draw-{entry['decoder_draw_id']:03d}/" in entry["request_block_id"]
    retried = [e for e in anchor_set.replay if e["scenario_id"] == sid(campaign, 0) and e["decoder_draw_id"] == 0]
    assert {e["semantic_attempt_id"] for e in retried} == {1}


def test_a_rule_retried_alone_is_keyed_to_the_attempt_that_generated_it(campaign, monkeypatch):
    scenario = sid(campaign, 0)
    real_load = DrawLedger.load

    def component_scope(ledger, **kwargs):
        state = real_load(ledger, **kwargs)
        if (ledger.condition_id, ledger.decoder_draw_id) != ("gemma/individual_bank", 0):
            return state
        first, accepted = [record for record in state.records if record.scenario_id == scenario]
        rule_1 = next(c for c in first.components if c.component_id == "Rule 1")
        rule_2 = next(c for c in accepted.components if c.component_id == "Rule 2")
        state.accepted[scenario] = dataclasses.replace(accepted, components=(rule_2,))
        settled = {"Rule 1": (first, rule_1), "Rule 2": (state.accepted[scenario], rule_2)}
        state.accepted_components = lambda key: settled if key == scenario else {
            c.component_id: (state.accepted[key], c) for c in state.accepted[key].components
        }
        return state

    monkeypatch.setattr(DrawLedger, "load", component_scope)
    build(campaign)  # every raw output, attempt 0's Rule 1 among them, read back against its digest
    replay = read_anchor_set(campaign["tmp"] / "anchor-set").replay
    keys = {e["rule_id"]: e["semantic_attempt_id"] for e in replay
            if e["scenario_id"] == scenario and e["decoder_draw_id"] == 0}
    assert keys == {"Rule 1": 0, "Rule 2": 1}


def test_the_dev_references_take_no_top_up(campaign):
    manifest = build(campaign, side=SIDE_DEV)
    rows = read_anchor_set(campaign["tmp"] / "anchor-set").rows
    assert manifest["top_up"] is None and manifest["counts"]["routed"] == 2 and manifest["counts"]["topped_up"] == 0
    assert all(row["k"] == 2 and not row["topped_up"] for row in rows)
    assert rows[2]["q_yes"] == 0.5 and rows[2]["mode"] == "no"
    assert manifest["projection_excluded"] == list(PROJECTION_EXCLUDED) and manifest["allow_untopped"] is False


def test_a_train_build_without_the_top_up_is_refused_unless_declared(campaign):
    with pytest.raises(AnchorError, match="2 train row\\(s\\) route to the top-up"):
        build(campaign, side=SIDE_TRAIN)
    assert not (campaign["tmp"] / "anchor-set.partial").exists()
    manifest = build(campaign, side=SIDE_TRAIN, allow_untopped=True)
    assert manifest["allow_untopped"] is True and manifest["side"] == SIDE_TRAIN
    assert (manifest["counts"]["routed"], manifest["counts"]["topped_up"]) == (2, 0)


def test_the_toolkit_routes_and_builds_through_the_same_builders_and_returns_their_outputs(campaign):
    out = campaign["tmp"] / "topup.jsonl"
    assert anchors.route(bank=campaign["root"], reference=campaign["reference"], rules=campaign["rules"],
                         side=SIDE_TRAIN, population=campaign["tmp"] / "population.jsonl", out=out) == out
    sidecar = Path(str(out) + ROUTE_SIDECAR_SUFFIX).read_bytes()
    assert route(campaign)["routed"] == 2 and Path(str(out) + ROUTE_SIDECAR_SUFFIX).read_bytes() == sidecar
    built = anchors.build(bank=campaign["root"], full_set=campaign["root"], reference=campaign["reference"],
                          rules=campaign["rules"], side=SIDE_TRAIN, out=campaign["tmp"] / "toolkit",
                          allow_untopped=True)
    assert built == campaign["tmp"] / "toolkit"
    reference = build(campaign, side=SIDE_TRAIN, out=campaign["tmp"] / "builder", allow_untopped=True)
    assert json.loads((built / ANCHORS_MANIFEST_FNAME).read_text())["counts"] == reference["counts"]
    assert (built / ANCHORS_FNAME).read_bytes() == (campaign["tmp"] / "builder" / ANCHORS_FNAME).read_bytes()


def test_the_command_line_routes_and_builds(campaign, modelspec, capsys):
    common = ["--bank", str(campaign["root"]), "--reference", str(campaign["reference"]),
              "--rules", str(campaign["rules"]), "--side", "train"]
    assert anchors.main(["route", *common, "--population", str(campaign["tmp"] / "population.jsonl"),
                         "--out", str(campaign["tmp"] / "topup.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["routed"] == 2
    top_root = run_top_up(campaign, modelspec)
    out = campaign["tmp"] / "cli-set"
    assert anchors.main(["build", *common, "--top-up", str(top_root), "--full-set", str(campaign["root"]),
                         "--full-set-condition", "gemma/full", "--out", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["out"] == str(out) and summary["topped_up"] == 2
    untopped = ["build", *common, "--no-top-up", "--bank-condition", "gemma/individual_bank",
                "--full-set", str(campaign["root"]), "--out", str(campaign["tmp"] / "untopped-set")]
    with pytest.raises(AnchorError, match="--allow-untopped"):
        anchors.main(untopped)
    assert anchors.main([*untopped, "--allow-untopped"]) == 0
    assert json.loads(capsys.readouterr().out)["topped_up"] == 0
    dev = [arg if arg != "train" else "dev" for arg in untopped]
    assert anchors.main([*dev[:-1], str(campaign["tmp"] / "dev-set")]) == 0
    assert read_anchor_set(campaign["tmp"] / "dev-set").manifest["allow_untopped"] is False
    with pytest.raises(SystemExit):
        anchors.main(["build", *common, "--full-set", str(campaign["root"]), "--out", str(out)])


def test_the_module_runs_its_command_line_as_the_retired_script_did(campaign, capsys, monkeypatch):
    import runpy
    import sys

    from tests.support.equivalence import tree

    out = campaign["tmp"] / "run"
    argv = ["build", "--bank", str(campaign["root"]), "--reference", str(campaign["reference"]), "--rules",
            str(campaign["rules"]), "--side", "train", "--no-top-up", "--bank-condition", "gemma/individual_bank",
            "--full-set", str(campaign["root"]), "--allow-untopped", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", ["build_anchors", *argv])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("ruleforge.grpo.anchor_set", run_name="__main__")
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    ran = out.rename(campaign["tmp"] / "program")
    assert anchors.main(argv) == 0
    assert capsys.readouterr().out == text and tree(out) == tree(ran)


def test_an_existing_output_or_a_torn_one_is_refused(campaign):
    (campaign["tmp"] / "anchor-set.partial").mkdir()
    with pytest.raises(AnchorError, match="partial exists"):
        build(campaign)
    (campaign["tmp"] / "anchor-set.partial").rmdir()
    build(campaign)
    with pytest.raises(AnchorError, match="anchor-set exists"):
        build(campaign)


def test_a_tampered_or_foreign_anchor_set_is_refused(campaign):
    build(campaign)
    directory = campaign["tmp"] / "anchor-set"
    replay = directory / REPLAY_INDEX_FNAME
    replay.write_text(replay.read_text().replace('"label": "no"', '"label": "yes"', 1))
    with pytest.raises(AnchorError, match="not the file its manifest records"):
        read_anchor_set(directory)
    manifest = json.loads((directory / ANCHORS_MANIFEST_FNAME).read_text())
    manifest["schema_version"] = 0
    (directory / ANCHORS_MANIFEST_FNAME).write_text(json.dumps(manifest))
    with pytest.raises(AnchorError, match="not a schema_version"):
        read_anchor_set(directory)


def test_an_incomplete_bank_is_refused(tmp_path, modelspec):
    root = tmp_path / "partial"
    run_condition(tmp_path, modelspec, root=root, campaign_id="part", mode="individual_bank", script=BANK_SCRIPT,
                  draw_ids="[0]")
    with pytest.raises(AnchorError, match="are not all complete"):
        open_condition(root, None, evaluation_mode="individual_bank")


def _edit_json(path: Path, **changes):
    payload = json.loads(path.read_text())
    payload.update(changes)
    path.write_text(json.dumps(payload))


def test_the_campaign_shape_is_checked(campaign):
    root = campaign["root"]
    with pytest.raises(AnchorError, match="holds 0 'both' condition"):
        open_condition(root, None, evaluation_mode="both")
    with pytest.raises(AnchorError, match="registers no condition"):
        open_condition(root, "gemma/nothing", evaluation_mode="individual_bank")
    with pytest.raises(AnchorError, match="is registered as 'full', not 'individual_bank'"):
        open_condition(root, "gemma/full", evaluation_mode="individual_bank")
    manifest = root / "sampling_manifest.json"
    original = manifest.read_text()
    _edit_json(manifest, scenario_manifest_sha256="0" * 64)
    with pytest.raises(AnchorError, match="not the one the campaign manifest froze"):
        open_condition(root, None, evaluation_mode="individual_bank")
    manifest.write_text(original)
    _edit_json(manifest, campaign_mode="final")
    with pytest.raises(AnchorError, match="'final' campaign"):
        open_condition(root, None, evaluation_mode="individual_bank")


def test_a_ruleset_without_rules_or_with_a_repeated_one_is_refused(tmp_path):
    path = tmp_path / "meta.jsonl"
    write_jsonl(path, [{"id": "Meta-Rule 1", "rule": "m"}])
    with pytest.raises(AnchorError, match="no non-meta rule"):
        Ruleset.load(path)
    write_jsonl(path, [{"id": "Rule 1", "rule": "a"}, {"id": "Rule 1", "rule": "b"}])
    with pytest.raises(AnchorError, match="names one twice"):
        Ruleset.load(path)


def test_another_ruleset_or_condition_is_refused(campaign):
    rules = campaign["tmp"] / "rules_other.jsonl"
    write_jsonl(rules, [{"id": "Rule 1", "rule": "x"}, {"id": "Rule 2", "rule": "y"}])
    with pytest.raises(AnchorError, match="judged ruleset"):
        build(campaign, rules=rules)
    reference = read_condition_file(campaign["reference"])
    other = register_condition(campaign["tmp"] / "training-other",
                               dataclasses.replace(reference, max_new_tokens=reference.max_new_tokens + 1))
    with pytest.raises(AnchorError, match="max_new_tokens"):
        build(campaign, reference=other)
    policy = register_condition(
        campaign["tmp"] / "training-policy",
        dataclasses.replace(reference, adapter_path="checkpoint-125/adapter_model.safetensors"),
    )
    with pytest.raises(AnchorError, match="must be the base judge's arm"):
        build(campaign, reference=policy)


def test_full_set_draws_over_another_population_are_refused(campaign, modelspec):
    tmp = campaign["tmp"]
    population = write_population(tmp / "other", n=3)
    root = run_condition(tmp, modelspec, root=tmp / "fullset-other", campaign_id="fs", mode="full",
                         script=FULL_SCRIPT, population=population)
    with pytest.raises(AnchorError, match="do not cover exactly the bank"):
        build(campaign, full_set=root)


def test_a_full_set_view_over_other_rules_is_refused(campaign):
    view = campaign["root"] / "gemma/full/draw-001/bundle_outcomes.jsonl"
    rows = read_jsonl(view)
    rows[0]["ruleset_rule_ids"] = ["Rule 1"]
    write_jsonl(view, rows)
    with pytest.raises(AnchorError, match="is not a full-set judgment over the ruleset"):
        build(campaign)


def test_a_top_up_that_is_not_the_banks_condition_or_seed_or_routing_is_refused(campaign, modelspec):
    with pytest.raises(AnchorError, match="another generation_seed_base"):
        build(campaign, top_up=run_top_up(campaign, modelspec, seed_base=16))
    tmp = campaign["tmp"]
    other = run_condition(tmp, modelspec, root=tmp / "topup-extra", campaign_id="tx", mode="individual_bank",
                          script=TOP_UP_SCRIPT, draws=4, seed_base=17, population=tmp / "topup.jsonl",
                          extra="{role: topup}")
    with pytest.raises(AnchorError, match="fingerprints differ"):
        build(campaign, top_up=other)
    everyone = run_condition(tmp, modelspec, root=tmp / "topup-all", campaign_id="ta", mode="individual_bank",
                             script=TOP_UP_SCRIPT, draws=4, seed_base=17)
    with pytest.raises(AnchorError, match="not the 2 the bank routes"):
        build(campaign, top_up=everyone)


def test_a_top_up_this_version_runs_joins_a_bank_recorded_before_the_one_reader(tmp_path, modelspec):
    with recorded_before_the_one_reader():
        recorded = make_campaign(tmp_path / "recorded", modelspec)
    fresh = make_campaign(tmp_path / "fresh", modelspec)
    manifest = build(recorded, top_up=run_top_up(recorded, modelspec))
    expected = build(fresh, top_up=run_top_up(fresh, modelspec))
    bank, top_up = (load_conditions(Path(manifest[name]["root"]))[manifest[name]["condition_id"]]
                    for name in ("bank", "top_up"))
    assert (bank.label_rule, top_up.label_rule) == (RECORDED_LABEL_RULE, LABEL_RULE)
    assert bank.fingerprint != top_up.fingerprint and condition_key(bank) == condition_key(top_up)
    assert (manifest["bank"]["fingerprint"], manifest["top_up"]["fingerprint"]) == (bank.fingerprint,
                                                                                   top_up.fingerprint)
    assert read_anchor_set(recorded["tmp"] / "anchor-set").rows == read_anchor_set(fresh["tmp"] / "anchor-set").rows
    assert manifest["counts"] == expected["counts"]


def test_judgments_from_another_stack_are_refused(campaign, modelspec):
    top_root = run_top_up(campaign, modelspec)
    for root, version in ((campaign["root"], "0.28.0+cu129"), (top_root, "0.19.0")):
        for provenance in (root / "gemma/individual_bank").glob("draw-*/draw_provenance.json"):
            payload = json.loads(provenance.read_text())
            payload["runtime"]["vllm_version"] = version
            provenance.write_text(json.dumps(payload))
            restamp_stacks(provenance.parent)
    with pytest.raises(AnchorError, match="different inference stacks"):
        build(campaign, top_up=top_root)


def _ledger(campaign, draw=0):
    return campaign["root"] / f"gemma/individual_bank/draw-{draw:03d}/attempts.jsonl"


def _tamper_accepted(campaign, edit):
    path = _ledger(campaign)
    rows = read_jsonl(path)
    edit(next(row for row in rows if row["status"] == "accepted"))
    write_jsonl(path, rows)


def test_a_bundle_short_of_a_rule_is_refused(campaign):
    _tamper_accepted(campaign, lambda row: row.update(components=row["components"][:1]))
    with pytest.raises(AnchorError, match="carries 1 rule label"):
        build(campaign)


def test_a_bundle_naming_a_rule_twice_is_refused(campaign):
    _tamper_accepted(campaign, lambda row: row.update(components=[row["components"][0]] * 2))
    with pytest.raises(AnchorError, match="names a rule twice"):
        build(campaign)


def test_a_judgment_without_a_raw_digest_is_refused(campaign):
    _tamper_accepted(campaign, lambda row: row["components"][1].update(raw_output_sha256=None))
    with pytest.raises(AnchorError, match="no raw-output or prompt digest"):
        build(campaign)


def test_a_ledger_missing_a_scenario_is_refused(campaign):
    path = _ledger(campaign, draw=1)
    write_jsonl(path, read_jsonl(path)[1:])
    with pytest.raises(AnchorError, match="not the manifest's scenarios"):
        build(campaign)


def test_a_view_that_disagrees_with_its_ledger_is_refused(campaign):
    view = campaign["root"] / "gemma/individual_bank/draw-000/bundle_outcomes.jsonl"
    rows = read_jsonl(view)
    rows[3]["rule_judgments"]["Rule 1"] = "yes"
    write_jsonl(view, rows)
    with pytest.raises(AnchorError, match="disagrees with the ledger"):
        build(campaign)


def test_a_raw_output_that_is_changed_or_missing_is_refused(campaign):
    path = campaign["root"] / "gemma/individual_bank/draw-001/raw_outputs.jsonl"
    rows = read_jsonl(path)
    edited = [dict(rows[0], raw_response=rows[0]["raw_response"] + " ")] + rows[1:]
    write_jsonl(path, edited)
    with pytest.raises(AnchorError, match="does not hash to the digest"):
        build(campaign)
    write_jsonl(path, rows[1:])
    with pytest.raises(AnchorError, match="no raw output on disk"):
        build(campaign)


def run_blocks(tmp, modelspec, *, name, population, mode, script, block_size=2, campaign_id="anch", **kwargs):
    """One root per block of ``population``, each judged by the fake judge under one campaign id and seed base."""
    blocks = tmp / f"{name}-blocks"
    payloads = build_blocks(population, blocks, block_size=block_size)
    return blocks, [
        run_condition(tmp, modelspec, root=tmp / name / f"b{index}", campaign_id=campaign_id, mode=mode,
                      script=script, population=blocks / block_name(population, index), **kwargs)
        for index in range(len(payloads))
    ]


@pytest.fixture
def blocked(campaign, modelspec):
    """The fixture's population cut into two blocks of two: a bank root and a full-set root per block."""
    tmp = campaign["tmp"]
    population = tmp / "population.jsonl"
    blocks, banks = run_blocks(tmp, modelspec, name="bank", population=population, mode="individual_bank",
                               script=BANK_SCRIPT)
    _, fulls = run_blocks(tmp, modelspec, name="full", population=population, mode="full", script=FULL_SCRIPT)
    return {**campaign, "banks": banks, "fulls": fulls, "blocks": blocks, "population": population}


def build_blocked(blocked, **overrides):
    kwargs = dict(bank=blocked["banks"], full_set=blocked["fulls"], population=blocked["population"],
                  blocks=blocked["blocks"], out=blocked["tmp"] / "blocked-set")
    kwargs.update(overrides)
    return build(blocked, **kwargs)


def without_root(rows):
    return [{key: value for key, value in row.items() if key != "root"} for row in rows]


def generation_seeds(*roots, mode="individual_bank"):
    """``{(scenario, draw, attempt, rule): generation_seed}`` over every ledger row of ``roots``, each key once."""
    seeds = {}
    for root in roots:
        for ledger in sorted((Path(root) / "gemma" / mode).glob("draw-*/attempts.jsonl")):
            for record in read_jsonl(ledger):
                for component in record["components"]:
                    key = (record["scenario_id"], record["decoder_draw_id"], record["semantic_attempt_id"],
                           component["component_id"])
                    assert key not in seeds
                    seeds[key] = component["generation_seed"]
    return seeds


def test_a_two_block_bank_gives_the_anchors_of_one_root(blocked):
    tmp = blocked["tmp"]
    one = build(blocked, out=tmp / "one-root")
    two = build_blocked(blocked)
    assert (tmp / "blocked-set" / ANCHORS_FNAME).read_bytes() == (tmp / "one-root" / ANCHORS_FNAME).read_bytes()
    one_set, two_set = read_anchor_set(tmp / "one-root"), read_anchor_set(tmp / "blocked-set")
    assert without_root(two_set.replay) == without_root(one_set.replay)
    assert two["counts"] == one["counts"]
    one_seeds, block_seeds = generation_seeds(blocked["root"]), generation_seeds(*blocked["banks"])
    assert block_seeds == one_seeds and len(set(one_seeds.values())) == len(one_seeds)
    assert (one["schema_version"], two["schema_version"]) == (1, 2)
    # One root keeps its manifest; several are listed, each with its digests, with the population they tile.
    assert "population" not in one and one["bank"]["root"] == str(blocked["root"])
    assert {row["root"] for row in one_set.replay} == {str(blocked["root"])}
    assert [record["root"] for record in two["bank"]] == [str(root) for root in blocked["banks"]]
    assert [record["root"] for record in two["full_set"]] == [str(root) for root in blocked["fulls"]]
    assert [record["n_scenarios"] for record in two["bank"]] == [2, 2]
    assert {record["fingerprint"] for record in two["bank"]} == {one["bank"]["fingerprint"]}
    assert two["population"]["name"] == "population.jsonl"
    assert two["population"]["sha256"] == sha256_file(blocked["population"])
    assert [block["row_range"] for block in two["population"]["blocks"]] == [[0, 2], [2, 4]]
    for row in two_set.replay:
        record = replay_source(two, row)
        state = DrawLedger(
            Path(record["root"]) / record["condition_id"] / f"draw-{row['decoder_draw_id']:03d}",
            campaign_id=record["campaign_id"], condition_id=record["condition_id"],
            decoder_draw_id=row["decoder_draw_id"],
        ).load()
        assert row["scenario_id"] in state.accepted


def test_a_blocked_bank_routes_and_builds_with_a_blocked_top_up(blocked, modelspec):
    tmp = blocked["tmp"]
    single = route(blocked)
    summary = route(blocked, bank=blocked["banks"], blocks=blocked["blocks"], out=tmp / "topup-blocked.jsonl")
    assert summary["routed"] == single["routed"] == 2
    assert (tmp / "topup-blocked.jsonl").read_bytes() == (tmp / "topup.jsonl").read_bytes()
    sidecar = json.loads(Path(str(tmp / "topup-blocked.jsonl") + ROUTE_SIDECAR_SUFFIX).read_text())
    assert [record["root"] for record in sidecar["bank"]] == [str(root) for root in blocked["banks"]]
    assert sidecar["scenario_ids"] == [sid(blocked, 2), sid(blocked, 3)]
    assert [block["name"] for block in sidecar["population"]["blocks"]] == [
        "population.block-000.jsonl", "population.block-001.jsonl",
    ]
    one_top = run_condition(tmp, modelspec, root=tmp / "anchors-train-topup", campaign_id="anch-topup",
                            mode="individual_bank", script=TOP_UP_SCRIPT, draws=4, seed_base=17,
                            population=tmp / "topup.jsonl")
    _, tops = run_blocks(tmp, modelspec, name="topup", population=tmp / "topup-blocked.jsonl",
                         mode="individual_bank", script=TOP_UP_SCRIPT, block_size=1, campaign_id="anch-topup",
                         draws=4, seed_base=17)
    one = build(blocked, top_up=one_top, out=tmp / "one-root")
    two = build_blocked(blocked, top_up=tops, side=SIDE_TRAIN)
    top_seeds = generation_seeds(*tops)
    assert top_seeds == generation_seeds(one_top) and not set(top_seeds.values()) & set(
        generation_seeds(*blocked["banks"]).values())
    assert (tmp / "blocked-set" / ANCHORS_FNAME).read_bytes() == (tmp / "one-root" / ANCHORS_FNAME).read_bytes()
    replay = read_anchor_set(tmp / "blocked-set").replay
    assert without_root(replay) == without_root(read_anchor_set(tmp / "one-root").replay)
    assert (one["counts"]["topped_up"], two["counts"]["topped_up"]) == (2, 2)
    assert [record["decoder_draws"] for record in input_records(two, "top_up")] == [4, 4]
    assert {row["root"] for row in replay if row["source"] == "top_up"} == {str(root) for root in tops}
    # Block 1 holds scenarios 2 and 3, both topped up: no replay row reads its bank draws.
    assert {replay_source(two, row)["root"] for row in replay} == {str(root) for root in [blocked["banks"][0], *tops]}


def test_the_planned_layout_a_blocked_bank_with_one_full_set_root(blocked):
    """The plan's own layout (``train_full`` is not blocked): block roots for the bank, one root for the full set."""
    tmp = blocked["tmp"]
    one = build(blocked, out=tmp / "one-root")
    mixed = build_blocked(blocked, full_set=blocked["root"])
    assert (tmp / "blocked-set" / ANCHORS_FNAME).read_bytes() == (tmp / "one-root" / ANCHORS_FNAME).read_bytes()
    assert mixed["counts"] == one["counts"] and mixed["schema_version"] == 2
    assert isinstance(mixed["bank"], list) and mixed["full_set"]["root"] == str(blocked["root"])
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp)
        relative = build(blocked, bank=blocked["root"].name, full_set=blocked["root"].name, out=tmp / "relative",
                         rules=blocked["rules"].name, reference=blocked["reference"].relative_to(tmp))
    assert relative["bank"]["root"] == str(blocked["root"])
    assert {row["root"] for row in read_anchor_set(tmp / "relative").replay} == {str(blocked["root"])}
    assert (relative["rules"]["path"], relative["reference"]["path"]) == (
        str(blocked["rules"]), str(blocked["reference"]))


def test_roots_whose_frozen_ids_are_not_the_populations_are_refused(blocked, monkeypatch):
    import ruleforge.scenarios.subset as subset

    original = anchors.compute_scenario_id
    monkeypatch.setattr(subset, "scenario_id", lambda record, where: "new-" + original(record))
    for index in range(2):
        path = sidecar_path(blocked["blocks"] / block_name(blocked["population"], index))
        payload = json.loads(path.read_text())
        path.write_text(json.dumps({**payload, "scenario_ids": ["new-" + sid for sid in payload["scenario_ids"]]}))
    with pytest.raises(AnchorError, match="froze 4 scenario id"):
        build_blocked(blocked)
    monkeypatch.setattr(anchors, "compute_scenario_id", lambda record: "new-" + original(record))
    with pytest.raises(AnchorError, match="holds 0 of the 2 routed scenario"):
        route(blocked)


def test_the_manifest_records_are_read_one_way_whatever_their_number(blocked):
    manifest = build(blocked)
    assert input_records(manifest, "bank") == [manifest["bank"]] and input_records(manifest, "top_up") == []
    row = {"source": "bank", "root": str(blocked["root"])}
    assert replay_source(manifest, row) == manifest["bank"]
    with pytest.raises(AnchorError, match="names no top_up root"):
        replay_source(manifest, {"source": "top_up", "root": str(blocked["root"])})
    with pytest.raises(AnchorError, match="names no bank root"):
        replay_source(manifest, {"source": "bank", "root": "/elsewhere"})


def test_blocks_that_overlap_are_missing_or_undeclared_are_refused(blocked):
    banks, fulls = blocked["banks"], blocked["fulls"]
    with pytest.raises(AnchorError, match="share 2 scenario"):
        build_blocked(blocked, bank=[banks[0], blocked["root"]])
    with pytest.raises(AnchorError, match=r"rows \[2, 4\) are in no block"):
        build_blocked(blocked, bank=banks[:1], full_set=fulls[:1])
    with pytest.raises(AnchorError, match=r"rows \[2, 4\) are in no block"):
        route(blocked, bank=banks[:1], blocks=blocked["blocks"])
    with pytest.raises(AnchorError, match="the full-set draws do not cover exactly"):
        build_blocked(blocked, full_set=fulls[:1])
    with pytest.raises(AnchorError, match="2 bank roots need the population they partition"):
        build_blocked(blocked, blocks=None, population=None)
    with pytest.raises(AnchorError, match="2 bank roots need the population they partition"):
        route(blocked, bank=banks)
    with pytest.raises(AnchorError, match="declare the population too"):
        build_blocked(blocked, population=None)
    with pytest.raises(AnchorError, match="is named twice"):
        build_blocked(blocked, bank=[banks[0], banks[0]])
    with pytest.raises(AnchorError, match="no campaign root given"):
        build_blocked(blocked, bank=[])
    with pytest.raises(AnchorError, match="cannot read the block sidecar"):
        build(blocked, population=blocked["population"], blocks=blocked["blocks"])
    assert build(blocked, population=blocked["population"])["population"]["name"] == "population.jsonl"
    with pytest.raises(AnchorError, match="is not the population the bank"):
        build(blocked, population=blocked["blocks"] / "population.block-000.jsonl", out=blocked["tmp"] / "x")


def test_a_block_sidecar_must_vouch_for_the_block_its_root_froze(blocked):
    path = sidecar_path(blocked["blocks"] / "population.block-001.jsonl")
    payload = json.loads(path.read_text())
    path.write_text(json.dumps({**payload, "output_sha256": "0" * 64}))
    with pytest.raises(AnchorError, match="not the block its sidecar"):
        build_blocked(blocked)
    path.write_text(json.dumps({**payload, "scenario_ids": payload["scenario_ids"][::-1]}))
    with pytest.raises(AnchorError, match="scenario ids are not rows"):
        build_blocked(blocked)


def test_block_roots_of_another_condition_or_design_are_refused(blocked, modelspec):
    tmp = blocked["tmp"]
    block = blocked["blocks"] / "population.block-001.jsonl"
    extra = run_condition(tmp, modelspec, root=tmp / "extra" / "b1", campaign_id="anch", mode="individual_bank",
                          script=BANK_SCRIPT, population=block, extra="{role: other}")
    with pytest.raises(AnchorError, match="the bank roots are not one condition"):
        build_blocked(blocked, bank=[blocked["banks"][0], extra])
    three = run_condition(tmp, modelspec, root=tmp / "three" / "b1", campaign_id="anch", mode="individual_bank",
                          script=BANK_SCRIPT, population=block, draws=3)
    with pytest.raises(AnchorError, match="drew different numbers of draws"):
        build_blocked(blocked, bank=[blocked["banks"][0], three])
    partial = run_condition(tmp, modelspec, root=tmp / "partial" / "b1", campaign_id="anch", mode="individual_bank",
                            script=BANK_SCRIPT, population=block, draw_ids="[0]")
    with pytest.raises(AnchorError, match="are not all complete"):
        build_blocked(blocked, bank=[blocked["banks"][0], partial])


def test_every_block_root_is_checked_against_the_reference_and_the_ruleset(blocked):
    tmp = blocked["tmp"]
    rules = tmp / "rules_other.jsonl"
    write_jsonl(rules, [{"id": "Rule 1", "rule": "x"}, {"id": "Rule 2", "rule": "y"}])
    with pytest.raises(AnchorError, match="judged ruleset"):
        build_blocked(blocked, rules=rules)
    reference = read_condition_file(blocked["reference"])
    other = register_condition(tmp / "training-other", dataclasses.replace(reference, max_new_tokens=1))
    with pytest.raises(AnchorError, match="max_new_tokens"):
        build_blocked(blocked, reference=other)
    with pytest.raises(AnchorError, match="max_new_tokens"):
        route(blocked, bank=blocked["banks"], blocks=blocked["blocks"], reference=other)


def set_runtime(root, mode="individual_bank", **fields):
    """Edit the runtime every draw of ``root``'s condition records (its judgments' stack)."""
    for provenance in (Path(root) / f"gemma/{mode}").glob("draw-*/draw_provenance.json"):
        payload = json.loads(provenance.read_text())
        payload["runtime"].update(fields)
        provenance.write_text(json.dumps(payload))
        restamp_stacks(provenance.parent)


def test_block_roots_share_one_stack_but_not_one_placement(blocked):
    first, second = blocked["banks"]
    engine = {"dtype": "bfloat16", "enable_prefix_caching": True}
    for root, placement in ((first, {"tensor_parallel_size": 1, "max_num_seqs": 256}),
                            (second, {"tensor_parallel_size": 2, "max_num_seqs": 512})):
        set_runtime(root, vllm_version="0.28.0+cu129", engine_args={**engine, **placement})
    for root in blocked["fulls"]:
        set_runtime(root, mode="full", vllm_version="0.28.0+cu129", engine_args={**engine, "tensor_parallel_size": 4})
    build_blocked(blocked)  # placement differs across the blocks; the GPU model is not in the stack signature
    set_runtime(second, engine_args={**engine, "enable_prefix_caching": False, "tensor_parallel_size": 2})
    with pytest.raises(AnchorError, match=f"bank roots: {first} on .* {second} on"):
        build_blocked(blocked, out=blocked["tmp"] / "set-2")
    with pytest.raises(AnchorError, match="different inference stacks"):
        route(blocked, bank=blocked["banks"], blocks=blocked["blocks"])
    set_runtime(second, vllm_version="0.19.0", engine_args={**engine, "tensor_parallel_size": 2})
    with pytest.raises(AnchorError, match="different inference stacks .*vllm"):
        build_blocked(blocked, out=blocked["tmp"] / "set-3")


def test_a_blocked_top_up_must_be_the_banks_condition_with_fresh_seeds(blocked, modelspec):
    tmp = blocked["tmp"]
    route(blocked, bank=blocked["banks"], blocks=blocked["blocks"])
    fresh = run_condition(tmp, modelspec, root=tmp / "tops" / "b0", campaign_id="anch-topup", mode="individual_bank",
                          script=TOP_UP_SCRIPT, draws=4, seed_base=17, population=tmp / "topup.jsonl")
    reused = run_condition(tmp, modelspec, root=tmp / "tops-16" / "b0", campaign_id="anch-topup",
                           mode="individual_bank", script=TOP_UP_SCRIPT, draws=4, seed_base=16,
                           population=tmp / "topup.jsonl")
    with pytest.raises(AnchorError, match="another generation_seed_base"):
        build_blocked(blocked, top_up=[reused], side=SIDE_TRAIN)
    with pytest.raises(AnchorError, match="the top_up roots .* share 2 scenario"):
        build_blocked(blocked, top_up=[fresh, reused], side=SIDE_TRAIN)
    assert build_blocked(blocked, top_up=[fresh], side=SIDE_TRAIN)["counts"]["topped_up"] == 2


def test_the_command_line_takes_the_roots_of_every_block(blocked, capsys):
    tmp = blocked["tmp"]
    common = ["--bank", *map(str, blocked["banks"]), "--blocks", str(blocked["blocks"]),
              "--reference", str(blocked["reference"]), "--rules", str(blocked["rules"])]
    assert anchors.main(["route", *common, "--side", "train", "--population", str(blocked["population"]),
                         "--out", str(tmp / "topup.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["routed"] == 2
    out = tmp / "cli-set"
    assert anchors.main(["build", *common, "--side", "dev", "--no-top-up", "--full-set", *map(str, blocked["fulls"]),
                         "--population", str(blocked["population"]), "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["scenarios"] == 4
    assert len(read_anchor_set(out).manifest["bank"]) == 2


def test_a_condition_is_named_by_its_registered_id_or_by_its_public_id():
    from ruleforge.core import record_names
    from ruleforge.core.errors import SamplingDesignError

    key = record_names.identity_key("forced-grpo")
    recorded = _spec(condition_id=f"qwen/{key}/anchor-bank")
    conditions = {recorded.condition_id: recorded}
    assert anchors.named_condition(conditions, recorded.condition_id) is recorded
    assert anchors.named_condition(conditions, "qwen/forced-grpo/anchor-bank") is recorded
    with pytest.raises(SamplingDesignError, match="no registered condition has the public id 'qwen/nothing'"):
        anchors.named_condition(conditions, "qwen/nothing")
    twin = dataclasses.replace(recorded, condition_id="qwen/forced-grpo/anchor-bank")
    both = {**conditions, twin.condition_id: twin}
    assert anchors.named_condition(both, recorded.condition_id) is recorded
    with pytest.raises(SamplingDesignError, match="2 registered conditions have the public id"):
        anchors.named_condition(both, twin.condition_id)


def test_a_recorded_bank_opens_by_its_public_id_and_reads_as_its_public_spelling(tmp_path, modelspec):
    from ruleforge.core import record_names
    from ruleforge.grpo import grpo_data as gd
    from tests.test_training_family_population import as_recorded

    roots = {}
    for recorded in (False, True):
        base = tmp_path / ("recorded" if recorded else "public")
        base.mkdir()
        root = base / "anchors-train"
        for mode, script in (("individual_bank", BANK_SCRIPT), ("full", FULL_SCRIPT)):
            run_condition(base, modelspec, root=root, campaign_id="anch", mode=mode, script=script,
                          condition_id=f"gemma/forced-grpo/{mode}")
        if recorded:
            as_recorded(root, "gemma/forced-grpo", "forced-grpo")
        roots[recorded] = root
    key = record_names.identity_key("forced-grpo")
    bank = open_condition(roots[True], "gemma/forced-grpo/individual_bank", evaluation_mode="individual_bank")
    public = open_condition(roots[False], "gemma/forced-grpo/individual_bank", evaluation_mode="individual_bank")
    assert bank.condition_id == f"gemma/{key}/individual_bank" != public.condition_id
    assert (bank.condition.prompt_style, bank.condition.fingerprint) == ("forced-grpo", public.condition.fingerprint)
    assert bank.condition.identity_id == public.condition.identity_id == bank.condition_id
    assert bank.outcome_cells() == public.outcome_cells()
    assert bank.record()["condition_id"] == bank.condition_id
    assert gd.open_source(bank.record(), evaluation_mode="individual_bank").condition == bank.condition
    full = anchors.open_input("full_set", roots[True], "gemma/forced-grpo/full", evaluation_mode="full")
    assert full.condition.condition_id == f"gemma/{key}/full"
