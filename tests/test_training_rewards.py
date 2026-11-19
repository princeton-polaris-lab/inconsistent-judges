from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ruleforge.grpo import rewards as rw
from ruleforge.grpo.rewards import (
    LAMBDA_COMP,
    LAMBDA_INT,
    QWEN3_FINISH_TOKEN_IDS,
    QWEN3_THINK_END_TOKEN_ID,
    TRUNCATED_NONE,
    TRUNCATED_ZERO,
    UNCLOSED_THINK,
    GrpoReward,
    ParsedCompletion,
    RewardTerms,
    margin_bucket,
    parse_completion,
    score,
    zero_variance_group_share,
)
from ruleforge.evaluate.validation import (
    DUPLICATE_RULE_IDS,
    EMPTY_RESPONSE,
    INVALID_LABEL,
    TRUNCATED,
)
from ruleforge.core.artifacts import load_jsonl_records

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES = load_jsonl_records(REPO_ROOT / "data" / "rules" / "modelspec" / "openai_rules_all.jsonl")
RULE_IDS = [rule["id"] for rule in RULES]
#: Qwen3-8B's ``<think>`` and ``\n\n`` (the cluster tests check both against the checkpoint).
THINK_OPEN_ID, NEWLINES_ID = 151667, 271
FINISHED_IDS = [THINK_OPEN_ID, 11, 12, QWEN3_THINK_END_TOKEN_ID, 13, QWEN3_FINISH_TOKEN_IDS[0]]
#: Cut off at the cap after the reasoning closed, so the ids hold ``</think>`` as the text does.
CLIPPED_IDS = [THINK_OPEN_ID, 11, 12, QWEN3_THINK_END_TOKEN_ID, 13]
#: Qwen3-8B's canonical empty block ``<think>\n\n</think>\n\n``, then an answer.
EMPTY_THINK_IDS = [THINK_OPEN_ID, NEWLINES_ID, QWEN3_THINK_END_TOKEN_ID, NEWLINES_ID, 13, QWEN3_FINISH_TOKEN_IDS[0]]

#: A hard row: q̂(yes) beats every other verdict by more than λ_i + λ_c.
HARD = {"yes": 0.95, "unsure": 0.03, "no": 0.02}
SOFT = {"yes": 0.45, "unsure": 0.15, "no": 0.40}


def rule_map(label="no", **labels):
    mapping = {rule: label for rule in RULE_IDS}
    mapping.update({rule.replace("_", " "): value for rule, value in labels.items()})
    return mapping


def answer(verdict, rules=None, **extra):
    payload = {"rule_judgments": {} if rules is None else rules, "preempted_rules": [], "full_set_judgment": verdict}
    payload.update(extra)
    return json.dumps(payload)


def completion(answer_text, reasoning="Weighing each rule against the response."):
    return f"<think>\n{reasoning}\n</think>\n\n{answer_text}"


def columns(n, q=HARD, cls="c", topped=False, yes_rules=("Rule 3",)):
    return {
        "q_yes": [q["yes"]] * n, "q_unsure": [q["unsure"]] * n, "q_no": [q["no"]] * n,
        "anchor_class": [cls] * n, "topped_up": [topped] * n, "base_yes_rules": [list(yes_rules)] * n,
    }


@pytest.fixture
def reward():
    return GrpoReward(RULES, truncated=TRUNCATED_ZERO, num_generations=1)


def terms_of(reward, text, ids=FINISHED_IDS, q=HARD):
    parsed = parse_completion(text, ids, schema=reward.schema)
    return parsed, score(parsed, q, truncated=reward.truncated)


def test_a_complete_consistent_map_earns_every_term(reward):
    parsed, terms = terms_of(reward, completion(answer("yes", rule_map(Rule_3="yes"))))
    assert parsed.valid and parsed.complete_map and parsed.verdict == "yes"
    assert list(parsed.rule_map) == RULE_IDS and parsed.think_tokens == 3
    assert terms == RewardTerms(HARD["yes"], 1.0, 1.0)


def test_an_empty_map_justifies_nothing(reward):
    # Ungated, "no" with an empty map would be internally consistent for free.
    parsed, terms = terms_of(reward, completion(answer("no", {})))
    assert parsed.valid and not parsed.complete_map
    assert terms == RewardTerms(HARD["no"], 0.0, 0.0)


def test_a_partial_map_with_a_yes_justifies_nothing(reward):
    # {"Rule k": "yes"} would justify a yes verdict by the max over a map of one.
    parsed, terms = terms_of(reward, completion(answer("yes", {"Rule 3": "yes"})))
    assert parsed.verdict == "yes" and not parsed.complete_map
    assert terms == RewardTerms(HARD["yes"], 0.0, 0.0)


def test_only_the_answer_after_the_reasoning_is_read(reward):
    reasoning = 'A first draft: {"full_set_judgment": "yes"}. On reflection Rule 3 does not apply.'
    parsed, terms = terms_of(reward, completion(answer("no", rule_map()), reasoning=reasoning))
    assert parsed.verdict == "no" and terms == RewardTerms(HARD["no"], 1.0, 1.0)
    parsed, terms = terms_of(reward, completion("   ", reasoning=reasoning))
    assert (parsed.valid, parsed.failure, terms) == (False, EMPTY_RESPONSE, RewardTerms(0.0, 0.0, 0.0))
    assert (parsed.answer, parsed.empty_think) == ("", False)


def test_the_parse_keeps_the_answer_the_preempted_rules_and_an_empty_reasoning(reward):
    text = answer("no", {}, preempted_rules=["Rule 2"])
    parsed = parse_completion(completion(text, reasoning=" \n "), FINISHED_IDS, schema=reward.schema)
    assert (parsed.answer, parsed.preempted_rules, parsed.empty_think) == (text, ("Rule 2",), True)
    maybe = '{"full_set_judgment": "maybe"}'
    parsed = parse_completion(completion(maybe), FINISHED_IDS, schema=reward.schema)
    assert (parsed.failure, parsed.answer, parsed.empty_think) == (INVALID_LABEL, maybe, False)
    parsed = parse_completion(completion(text), CLIPPED_IDS, schema=reward.schema)
    assert (parsed.answer, parsed.preempted_rules, parsed.empty_think) == (None, (), False)


def test_a_finished_completion_without_a_closing_think_tag_is_invalid(reward):
    text = '<think>\nRule 3 applies, so {"full_set_judgment": "yes", "rule_judgments": {}}'
    parsed, terms = terms_of(reward, text, ids=[THINK_OPEN_ID, 11, 12, QWEN3_FINISH_TOKEN_IDS[1]])
    assert parsed.finished and not parsed.closed_think and parsed.think_tokens is None
    assert (parsed.failure, terms) == (UNCLOSED_THINK, RewardTerms(0.0, 0.0, 0.0))


def test_the_canonical_empty_think_block_is_empty_thinking(reward):
    parsed = parse_completion('<think>\n\n</think>\n\n{"full_set_judgment": "no"}', EMPTY_THINK_IDS,
                              schema=reward.schema)
    assert (parsed.valid, parsed.verdict, parsed.empty_think, parsed.think_tokens) == (True, "no", True, 2)
    parsed = parse_completion('\n\n</think>\n\n{"full_set_judgment": "no"}', EMPTY_THINK_IDS[1:],
                              schema=reward.schema)
    assert (parsed.empty_think, parsed.think_tokens) == (True, 1)
    for reasoning in ("Rule 3 does not apply.", "<think>"):  # only the one opening tag is dropped
        parsed = parse_completion(completion(answer("no"), reasoning=reasoning), FINISHED_IDS, schema=reward.schema)
        assert not parsed.empty_think


def test_a_meta_rule_key_makes_the_map_incomplete(reward):
    parsed, terms = terms_of(reward, completion(answer("no", {**rule_map(), "Meta-Rule 1": "no"})))
    assert parsed.valid and not parsed.complete_map
    assert terms == RewardTerms(HARD["no"], 0.0, 0.0)


def test_a_rule_named_twice_makes_the_map_incomplete(reward):
    text = completion('{"rule_judgments": {' + ", ".join(f'"{rule}": "no"' for rule in RULE_IDS)
                      + ', "7": "yes"}, "preempted_rules": [], "full_set_judgment": "no"}')
    parsed, terms = terms_of(reward, text)
    assert parsed.valid and not parsed.complete_map
    assert reward.schema.classify_forced(text.rsplit("</think>", 1)[1], require_map=True).failure_reason == (
        DUPLICATE_RULE_IDS
    )
    assert terms == RewardTerms(HARD["no"], 0.0, 0.0)


def test_a_nested_value_makes_the_map_incomplete(reward):
    parsed, terms = terms_of(reward, completion(answer("no", {**rule_map(), "Rule 1": {"label": "no"}})))
    assert parsed.valid and not parsed.complete_map


def test_a_truncated_completion_scores_zero_or_none(reward):
    text = completion(answer("yes", rule_map(Rule_3="yes")))
    parsed, terms = terms_of(reward, text, ids=CLIPPED_IDS)
    assert (parsed.finished, parsed.failure) == (False, TRUNCATED)
    assert terms == RewardTerms(0.0, 0.0, 0.0) and terms.total(LAMBDA_INT, LAMBDA_COMP) == 0.0
    none = score(parsed, HARD, truncated=TRUNCATED_NONE)
    assert none == RewardTerms(None, None, None) and none.total(LAMBDA_INT, LAMBDA_COMP) is None
    assert parse_completion(text, [], schema=reward.schema).failure == TRUNCATED


def test_an_all_yes_map_is_consistent_with_yes(reward):
    parsed, terms = terms_of(reward, completion(answer("yes", rule_map("yes"))))
    assert terms == RewardTerms(HARD["yes"], 1.0, 1.0)
    parsed, terms = terms_of(reward, completion(answer("no", rule_map("yes"))))
    assert terms == RewardTerms(HARD["no"], 0.0, 1.0)


def test_an_unparseable_verdict_is_invalid(reward):
    parsed, terms = terms_of(reward, completion('{"full_set_judgment": "maybe"}'))
    assert (parsed.failure, terms) == (INVALID_LABEL, RewardTerms(0.0, 0.0, 0.0))


def total(reward, text, q):
    values = [func([completion(text)], [FINISHED_IDS], **columns(1, q=q))[0] for func in reward.reward_funcs]
    return sum(weight * value for weight, value in zip(reward.reward_weights, values))


def test_on_hard_rows_the_anchor_verdict_beats_a_wrong_verdict_with_a_perfect_map(reward):
    assert reward.reward_weights == [1.0, 0.3, 0.1]
    bare = answer("yes")  # the anchor verdict, no map at all
    for margin in (0.41, 0.6, 0.98):
        q_yes = (1 + margin) / 2
        q = {"yes": q_yes, "unsure": 0.0, "no": 1 - q_yes}
        wrong = answer("no", rule_map())  # complete, consistent, all no
        assert total(reward, bare, q) > total(reward, wrong, q)
    assert total(reward, bare, HARD) == pytest.approx(0.95)
    assert total(reward, answer("no", rule_map()), HARD) == pytest.approx(0.02 + 0.3 + 0.1)


@pytest.mark.parametrize("q", [SOFT, {"yes": 0.55, "unsure": 0.0, "no": 0.45}])
def test_on_soft_rows_a_complete_no_can_out_earn_an_incomplete_yes(reward, q):
    complete_no = total(reward, answer("no", rule_map()), q)
    incomplete_yes = total(reward, answer("yes", {"Rule 3": "yes"}), q)
    complete_yes = total(reward, answer("yes", rule_map(Rule_3="yes")), q)
    assert complete_no > incomplete_yes
    assert complete_yes > complete_no  # a complete yes map still wins when q̂(yes) > q̂(no)


def test_the_three_functions_share_one_parse_per_completion(monkeypatch):
    calls = []
    real = rw.parse_completion

    def counting(text, ids, **kwargs):
        calls.append(text)
        return real(text, ids, **kwargs)

    monkeypatch.setattr(rw, "parse_completion", counting)
    reward = GrpoReward(RULES, truncated=TRUNCATED_ZERO, num_generations=2)
    batch = [[{"role": "assistant", "content": completion(answer(v, rule_map()))}] for v in ("no", "yes")]
    ids = [FINISHED_IDS, FINISHED_IDS]
    kwargs = dict(prompts=[[{"role": "system", "content": "P"}]] * 2, trainer_state=None, **columns(2))
    results = [func(completions=batch, completion_ids=ids, **kwargs) for func in reward.reward_funcs]
    assert len(calls) == 2
    assert results == [[HARD["no"], HARD["yes"]], [1.0, 0.0], [1.0, 1.0]]
    reward.flat_soft_reward(completions=batch[:1] * 2, completion_ids=ids, **columns(2))
    assert len(calls) == 2 and len(reward._cache) == 1
    assert [f.__name__ for f in reward.reward_funcs] == [
        "flat_soft_reward", "internal_consistency_reward", "map_completeness_reward",
    ]


def test_the_reward_refuses_misconfiguration_and_misaligned_columns(reward):
    with pytest.raises(ValueError, match="truncated must be one of"):
        GrpoReward(RULES, truncated="drop", num_generations=8)
    with pytest.raises(ValueError, match="num_generations must be positive"):
        GrpoReward(RULES, truncated=TRUNCATED_NONE, num_generations=0)
    text = [completion(answer("no"))]
    with pytest.raises(ValueError, match="1 completions but 2 completion_ids"):
        reward.flat_soft_reward(text, [FINISHED_IDS] * 2, **columns(1))
    missing = columns(1)
    del missing["base_yes_rules"]
    with pytest.raises(ValueError, match="lacks the reward columns"):
        reward.map_completeness_reward(text, [FINISHED_IDS], **missing)
    with pytest.raises(ValueError, match="do not align"):
        reward.internal_consistency_reward(text, [FINISHED_IDS], **columns(2))
    with pytest.raises(ValueError, match="not a distribution"):
        reward.flat_soft_reward(text, [FINISHED_IDS], **columns(1, q={"yes": 0.5, "unsure": 0.5, "no": 0.5}))
    with pytest.raises(ValueError, match="not a distribution"):
        reward.flat_soft_reward(text, [FINISHED_IDS], **columns(1, q={"yes": 1.5, "unsure": -0.5, "no": 0.0}))
    with pytest.raises(ValueError, match="unknown anchor classes"):
        reward.flat_soft_reward(text, [FINISHED_IDS], **columns(1, cls="d"))
    # a bank given as a string would read as its characters, every flag off it
    with pytest.raises(ValueError, match=r"row 0: base_yes_rules is a string \(\"\['Rule 3'\]\"\), not a list"):
        reward.flat_soft_reward(text, [FINISHED_IDS], **{**columns(1), "base_yes_rules": ["['Rule 3']"]})
    grouped = GrpoReward(RULES, truncated=TRUNCATED_ZERO, num_generations=2)
    with pytest.raises(ValueError, match="not whole groups of 2"):
        grouped.flat_soft_reward(text, [FINISHED_IDS], **columns(1))


def test_margin_buckets_and_zero_variance_groups():
    assert margin_bucket(0.5, 0.35) == "0.0-0.2"
    assert margin_bucket(0.6, 0.4) == "0.2-0.4"  # 0.19999999999999998 in binary, 0.2 as declared
    assert margin_bucket(0.35, 0.55) == "0.2-0.4"
    assert margin_bucket(0.7, 0.3) == "0.4-1.0"
    assert margin_bucket(1.0, 0.0) == "0.4-1.0"
    assert zero_variance_group_share([0.1, 0.1, 0.5, 0.2], 2) == 0.5
    assert zero_variance_group_share([None, 0.3, None, None, 0.4, 0.4], 3) == 1.0
    assert zero_variance_group_share([0.1, None, 0.2], 3) == 0.0
    with pytest.raises(ValueError, match="not whole groups"):
        zero_variance_group_share([], 2)


def logged(reward, texts, ids, cols):
    """What the first function logs for one batch: ``(metrics, extras)``, each name once."""
    metrics, extras = {}, {}
    reward.flat_soft_reward(
        texts, ids, log_metric=lambda name, value: metrics.setdefault(name, value),
        log_extra=lambda name, values: extras.setdefault(name, values), **cols,
    )
    return metrics, extras


def pooled(metrics, name):
    den = metrics[f"consistency/{name}/den"]
    return metrics[f"consistency/{name}/num"] / den if den else None


def test_the_telemetry_is_logged_once_per_batch_by_the_first_function():
    reward = GrpoReward(RULES, truncated=TRUNCATED_NONE, num_generations=2)
    texts = [
        completion(answer("yes", rule_map(Rule_3="yes", Rule_9="yes"))),  # group 1: flags 1 of 2 known
        completion(answer("no", rule_map())),
        completion(answer("yes", {"Rule 3": "yes"}), reasoning=""),  # group 2: one empty think, one clipped
        completion(answer("yes", rule_map())),
    ]
    ids = [FINISHED_IDS, FINISHED_IDS, EMPTY_THINK_IDS, CLIPPED_IDS]
    cols = columns(4, q=SOFT, cls="a", topped=True, yes_rules=("Rule 3",))
    metrics, extras = logged(reward, texts, ids, cols)
    assert reward.flat_soft_reward(texts, ids, **cols) == [SOFT["yes"], SOFT["no"], SOFT["yes"], None]
    assert metrics["consistency/valid_rate"] == 0.75 and metrics["consistency/clipped_rate"] == 0.25
    assert metrics["consistency/unclosed_think_rate"] == 0.0 and metrics["consistency/complete_map_rate"] == 0.5
    assert metrics["consistency/empty_think_rate"] == 0.25 and metrics["consistency/think_marker_mismatch_rate"] == 0.0
    assert metrics["consistency/zero_variance_group_share"] == 0.5  # group 2 has one scored completion
    assert metrics["consistency/zero_variance_group_share_finished"] == 0.5
    assert (metrics["consistency/yes_rate/num"], metrics["consistency/yes_rate/den"]) == (2.0, 3.0)
    assert pooled(metrics, "yes_rate_class_a") == pytest.approx(2 / 3)
    assert pooled(metrics, "mode_rate_nonyes_rows") is None  # SOFT's mode is yes
    assert pooled(metrics, "unsure_rate_majority_yes_rows") is None  # every row is class a
    assert pooled(metrics, "flag_precision") == 0.5 and pooled(metrics, "flag_overlap") == 1.0
    assert pooled(metrics, "map_yes_labels_mean") == 1.0 and pooled(metrics, "map_majority_yes_share") == 0.0
    assert pooled(metrics, "map_all_yes_share") == 0.0 and pooled(metrics, "map_all_no_share") == 0.5
    assert pooled(metrics, "map_all_no_share_class_a") == 0.5
    assert pooled(metrics, "map_all_no_share_class_b") is None and pooled(metrics, "map_all_no_share_class_c") is None
    assert (metrics["consistency/think_tokens_mean/num"], metrics["consistency/think_tokens_mean/den"]) == (11.0, 4.0)
    assert pooled(metrics, "think_short_share") == 1.0
    assert metrics["consistency/margin_0.0-0.2/share"] == 1.0 and metrics["consistency/margin_0.4-1.0/share"] == 0.0
    assert pooled(metrics, "margin_0.0-0.2/r_flat") == pytest.approx((0.45 + 0.40 + 0.45) / 3)
    assert pooled(metrics, "margin_0.0-0.2/r_int") == pytest.approx(2 / 3)
    assert pooled(metrics, "margin_0.0-0.2/yes_rate") == pytest.approx(2 / 3)
    assert pooled(metrics, "margin_0.4-1.0/r_flat") is None
    assert extras["anchor_class"] == ["a"] * 4 and extras["topped_up"] == [True] * 4
    assert extras["margin_bucket"] == ["0.0-0.2"] * 4
    assert extras["map_yes_labels"] == [2, 0, None, None]
    assert extras["think_tokens"] == [3, 3, 2, 3]
    assert extras["parse_failure"] == [None, None, None, TRUNCATED]
    # The other two functions log nothing, and nothing is logged without TRL's hooks.
    calls = []
    reward.internal_consistency_reward(texts, ids, log_metric=lambda *a: calls.append(a), **cols)
    reward.flat_soft_reward(texts, ids, **cols)
    assert calls == []


def test_the_mode_rate_on_non_yes_rows_and_the_unsure_rate_on_majority_yes_rows(reward):
    q = {"yes": 0.1, "unsure": 0.1, "no": 0.8}
    metrics, _ = logged(reward, [completion(answer("no")), completion(answer("yes"))], [FINISHED_IDS] * 2,
                        columns(2, q=q, cls="a"))
    assert pooled(metrics, "mode_rate_nonyes_rows") == 0.5
    assert metrics["consistency/zero_variance_group_share"] == 1.0  # groups of one never vary
    assert pooled(metrics, "flag_precision") is None and pooled(metrics, "map_yes_labels_mean") is None
    metrics, _ = logged(reward, [completion(answer("unsure")), completion(answer("yes"))], [FINISHED_IDS] * 2,
                        columns(2, q=HARD, cls="b"))
    assert pooled(metrics, "unsure_rate_majority_yes_rows") == 0.5 and pooled(metrics, "yes_rate_class_a") is None


def test_the_favoured_no_yes_rate_is_logged_only_when_the_run_asks(reward):
    leaning_no, tied = {"yes": 0.1, "unsure": 0.1, "no": 0.8}, {"yes": 0.45, "unsure": 0.1, "no": 0.45}
    texts = [completion(answer("yes")), completion(answer("no")), completion(answer("yes")), completion(answer("yes"))]
    cols = {name: columns(2, q=leaning_no)[name] + columns(1, q=tied)[name] + columns(1)[name] for name in columns(1)}
    switched = GrpoReward(RULES, truncated=TRUNCATED_ZERO, num_generations=1, favoured_no_rate=True)
    metrics, _ = logged(switched, texts, [FINISHED_IDS] * 4, cols)
    assert (metrics["consistency/yes_rate_favoured_no/num"], metrics["consistency/yes_rate_favoured_no/den"]) == (2.0, 3.0)
    plain, _ = logged(reward, texts, [FINISHED_IDS] * 4, cols)
    assert sorted(set(metrics) - set(plain)) == ["consistency/yes_rate_favoured_no/den", "consistency/yes_rate_favoured_no/num"]
    names = [sorted(logged(switched, *batch)[0]) for batch in rank_batches()]
    assert names[0] == names[1] and "consistency/yes_rate_favoured_no/den" in names[0]


def test_each_completion_is_counted_against_its_own_rows_bank(reward):
    texts, ids = [completion(answer("yes", rule_map(Rule_3="yes")))] * 2, [FINISHED_IDS] * 2
    for banks in ((("Rule 3",), ("Rule 9",)), (("Rule 9",), ("Rule 3",))):
        cols = {name: columns(1, yes_rules=banks[0])[name] + columns(1, yes_rules=banks[1])[name]
                for name in columns(1)}
        metrics, _ = logged(reward, texts, ids, cols)
        for name in ("flag_precision", "flag_overlap"):
            assert pooled(metrics, name) == 0.5, name


def test_a_truncation_under_zero_varies_a_group_only_through_its_zero():
    reward = GrpoReward(RULES, truncated=TRUNCATED_ZERO, num_generations=2)
    texts = [completion(answer("yes"))] * 2
    metrics, _ = logged(reward, texts, [FINISHED_IDS, CLIPPED_IDS], columns(2))
    assert metrics["consistency/zero_variance_group_share"] == 0.0  # 0.95 against the truncation's 0
    assert metrics["consistency/zero_variance_group_share_finished"] == 1.0  # one finished completion: no signal


def rank_batches():
    """Two ranks' batches of 2 groups of 8 that share no part-of-batch population."""
    return [
        ([completion(answer("no", rule_map()))] * 16, [FINISHED_IDS] * 16, columns(16, q=SOFT, cls="a")),
        ([completion(answer("yes"))] * 16, [FINISHED_IDS] * 15 + [CLIPPED_IDS], columns(16, q=HARD, cls="c")),
    ]


def test_every_batch_logs_the_same_metric_names():
    reward = GrpoReward(RULES, truncated=TRUNCATED_NONE, num_generations=8)
    names = [sorted(logged(reward, *batch)[0]) for batch in rank_batches()]
    lone = GrpoReward(RULES, truncated=TRUNCATED_ZERO, num_generations=1)
    names.append(sorted(logged(lone, [completion(answer("maybe"))], [CLIPPED_IDS], columns(1))[0]))
    assert names[0] == names[1] == names[2]
    ratios = {name.rsplit("/", 1)[0] for name in names[0] if name.endswith(("/num", "/den"))}
    assert {f"{ratio}/{part}" for ratio in ratios for part in ("num", "den")} <= set(names[0])


def test_the_mean_over_ranks_of_num_and_den_is_the_pooled_rate():
    reward = GrpoReward(RULES, truncated=TRUNCATED_NONE, num_generations=8)
    batches = rank_batches()
    ranks = [logged(reward, *batch)[0] for batch in batches]
    # Both ranks' completions as one batch, as a single process would see them.
    texts, ids = (sum((batch[i] for batch in batches), []) for i in (0, 1))
    cols = {name: sum((batch[2][name] for batch in batches), []) for name in batches[0][2]}
    whole = logged(reward, texts, ids, cols)[0]
    for name, value in whole.items():
        expected = value / len(ranks) if name.endswith(("/num", "/den")) else value
        assert sum(rank[name] for rank in ranks) / len(ranks) == pytest.approx(expected)
    assert pooled(whole, "yes_rate") == pytest.approx(15 / 31)  # 31 valid, the 16th yes clipped
    assert pooled(whole, "map_all_no_share") == 1.0 and pooled(whole, "flag_overlap") is None


def test_a_parsed_completion_reports_validity_and_completeness():
    parsed = ParsedCompletion(True, True, 3, "no", None, None)
    assert parsed.valid and not parsed.complete_map


@pytest.mark.cluster
def test_the_qwen3_token_ids_are_the_checkpoints():
    root = Path(os.environ.get("RULEFORGE_MODEL_ROOT") or Path.home() / "models") / "Qwen3-8B"
    config = json.loads((root / "tokenizer_config.json").read_text())
    ids = {entry["content"]: int(token_id) for token_id, entry in config["added_tokens_decoder"].items()}
    # TRL's truncation mask reads the eos and pad tokens; the reward's finish set must be those two.
    assert (ids[config["eos_token"]], ids[config["pad_token"]]) == QWEN3_FINISH_TOKEN_IDS
    assert (ids["<think>"], ids["</think>"]) == (THINK_OPEN_ID, QWEN3_THINK_END_TOKEN_ID)


@pytest.mark.cluster
def test_the_empty_think_signal_reads_qwen3_as_trl_decodes_it():
    from transformers import AutoTokenizer

    root = Path(os.environ.get("RULEFORGE_MODEL_ROOT") or Path.home() / "models") / "Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(str(root))
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": "P"}, {"role": "user", "content": ""}], tokenize=False,
        add_generation_prompt=True,
    )
    # The prompt leaves <think> to the model, and TRL's decode keeps it.
    assert prompt.endswith("<|im_start|>assistant\n")
    ids = [*tokenizer.encode('<think>\n\n</think>\n\n{"full_set_judgment": "no"}'), QWEN3_FINISH_TOKEN_IDS[0]]
    assert ids[:4] == [THINK_OPEN_ID, NEWLINES_ID, QWEN3_THINK_END_TOKEN_ID, NEWLINES_ID]
    text = tokenizer.batch_decode([ids], skip_special_tokens=True)[0]
    assert text.startswith("<think>")
    reward = GrpoReward(RULES, truncated=TRUNCATED_ZERO, num_generations=1)
    parsed = parse_completion(text, ids, schema=reward.schema)
    assert parsed.valid and parsed.empty_think and parsed.think_tokens == 2


@pytest.fixture(scope="module")
def gemma(tmp_path_factory):
    """The tiny Gemma checkpoint's training tokenizer and completion format (``tests/tiny_gemma.py``)."""
    pytest.importorskip("transformers")
    from ruleforge.judges.completion_format import load_training_format
    from tests.tiny_gemma import save_tiny_gemma

    return load_training_format(save_tiny_gemma(tmp_path_factory.mktemp("models") / "gemma-4-31B-it", weights=False))


def gemma_completion(tokenizer, answer_text, *, reasoning="Weighing each rule.", end="<turn|>"):
    """A completion as Gemma 4 writes it and TRL hands it over."""
    raw = f"<|channel>thought\n{reasoning}<channel|>{answer_text}" if reasoning is not None else answer_text
    ids = tokenizer.encode(raw, add_special_tokens=False) + (
        [] if end is None else [tokenizer.convert_tokens_to_ids(end)])
    return tokenizer.decode(ids, skip_special_tokens=True), ids


def test_a_gemma_completion_is_split_at_its_last_channel_close_by_id(gemma):
    tokenizer, fmt = gemma
    reward = GrpoReward(RULES, truncated=TRUNCATED_ZERO, num_generations=1, completion_format=fmt)
    body = answer("yes", rule_map(Rule_3="yes"))
    text, ids = gemma_completion(tokenizer, body)
    assert "<channel|>" not in text  # TRL's decode dropped the markers: the text alone cannot be split
    [parsed] = reward.parse([text], [ids])
    assert parsed.valid and parsed.complete_map and parsed.verdict == "yes" and parsed.answer == body
    assert parsed.think_tokens == ids.index(fmt.think_end_token_id) and not parsed.empty_think
    # the reasoning's own draft verdict is never read; the canonical empty channel is empty thinking
    text, ids = gemma_completion(tokenizer, answer("no"), reasoning='draft {"full_set_judgment": "yes"}')
    assert reward.parse([text], [ids])[0].verdict == "no"
    text, ids = gemma_completion(tokenizer, answer("no"), reasoning="")
    assert reward.parse([text], [ids])[0].empty_think
    # never closed, ended on another stop id (truncated for TRL too), cut at the cap
    text, ids = gemma_completion(tokenizer, answer("no"), reasoning=None)
    assert reward.parse([text], [ids])[0].failure == UNCLOSED_THINK
    for end in ("<eos>", None):
        text, ids = gemma_completion(tokenizer, answer("no"), end=end)
        assert reward.parse([text], [ids])[0].failure == TRUNCATED


def test_a_format_with_other_stop_ids_logs_how_often_a_batch_ended_on_one(gemma):
    tokenizer, fmt = gemma
    reward = GrpoReward(RULES, truncated=TRUNCATED_ZERO, num_generations=2, completion_format=fmt)
    ends = ("<turn|>", "<eos>", None, "<turn|>")  # finished, another stop id, cut at the cap, finished
    pairs = [gemma_completion(tokenizer, answer("no", rule_map()), end=end) for end in ends]
    texts, ids = [text for text, _ in pairs], [ids for _, ids in pairs]
    metrics, _ = logged(reward, texts, ids, columns(4))
    assert metrics["consistency/clipped_rate"] == 0.5  # both kinds of truncation
    assert metrics["consistency/other_stop_rate"] == 0.25 and metrics["consistency/cap_clipped_rate"] == 0.25
    # Qwen3-8B's format has none, and logs no such names
    qwen, _ = logged(GrpoReward(RULES, truncated=TRUNCATED_ZERO, num_generations=1),
                     [completion(answer("no"))], [FINISHED_IDS], columns(1))
    assert "consistency/other_stop_rate" not in qwen and "consistency/cap_clipped_rate" not in qwen
    with pytest.raises(ValueError, match="or a completion_format, not both"):
        GrpoReward(RULES, truncated=TRUNCATED_ZERO, num_generations=1, completion_format=fmt,
                     think_end_token_id=fmt.think_end_token_id)


BASE_REWARD_DIGEST = "c3fd0f562dad2decf8dcf74eeaf2f90b69e36c68aa2141c74189e749a399a713"
GEMMA_REWARD_DIGEST = "d3091de9df393ec719dc977662f30c70f260ecfffdb85e21c01d4df89af7810b"


@pytest.mark.torch
def test_the_trainers_reward_names_and_config_digests_are_unchanged():
    pytest.importorskip("torch", reason="needs the grpo group (CPU torch is enough)")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    from omegaconf import OmegaConf

    from ruleforge.grpo import trainer as tn
    from ruleforge.core.digests import canonical_digest

    assert tn.REWARD_FUNC_NAMES == ("flat_soft_reward", "internal_consistency_reward", "map_completeness_reward")
    grpo = REPO_ROOT / "configs" / "training" / "grpo"
    base = OmegaConf.load(grpo / "base.yaml").reward
    block = OmegaConf.to_container(base)
    assert block == {"truncated": "???", "lambda_int": 0.3, "lambda_comp": 0.1}
    gemma = OmegaConf.to_container(OmegaConf.merge(base, OmegaConf.load(grpo / "gemma4_31b.yaml").reward))
    assert gemma == {**block, "favoured_no_rate": True}
    for given, digest in ((block, BASE_REWARD_DIGEST), (gemma, GEMMA_REWARD_DIGEST)):
        current = {**given, "truncated": "zero"}
        assert canonical_digest(tn.resume_free({"reward": current})) == digest
        recorded = {**current, "lambda_prec": 0.0, "lambda_just": 0.0}
        assert tn.resume_free({"reward": recorded}) == {"reward": current}
        assert canonical_digest(tn.resume_free({"reward": recorded})) == digest
