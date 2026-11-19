from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from omegaconf import OmegaConf

from ruleforge.core.constants import INTENT_TRACE_FNAME, PRIOR_TRACE_FNAME
from ruleforge.judges.backend import STRIP_FAILED_KEY, ModelBackend
from ruleforge.judges.base import JudgeOptions
from ruleforge.judges.batch_api import REASONING_KEY, REASONING_SUMMARIES_KEY
from ruleforge.judges.models import _extract_text_and_strip_status, call_local_model_with_details
from ruleforge.optimize.agents.psrl_rule_amendment import PSRLRuleAmendment
from ruleforge.optimize.agents.psrl_scenario_generation import PSRLScenarioGeneration
from ruleforge.optimize.agents.traces import unfinished_reason
from ruleforge.optimize.judging import ValidatedJudge
from ruleforge.optimize.metrics.intent import IntentScoringError, _checked_scores, run_intent_rubric
from ruleforge.tasks import RewardWeights
from tests.support.scripted import ScriptedFailure, ScriptedJudge

RUBRIC = {"0": "fine", "1": "drift"}
INTENT_ANSWER = '{"rules": [{"id": "Rule 1", "score": 1, "reasoning": "minor"}, {"id": "Rule 2", "score": 0}]}'


class _ReasoningJudge(ScriptedJudge):
    def generate_once(self, prompts, **kwargs):
        return [replace(outcome, reasoning_summaries=("weighed",), reasoning=f"thought: {outcome.text}")
                if outcome.error is None else outcome
                for outcome in super().generate_once(prompts, **kwargs)]


def _write(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _validated(judge) -> ValidatedJudge:
    return ValidatedJudge(judge, num_tries=3, max_retry_seconds=None)


def _answers(*texts):
    """Answers in order, one per call; an Exception instance raises (a transport failure)."""
    queue = list(texts)

    def answer(system_prompt, user_prompt):
        value = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(value, Exception):
            raise value
        return value

    return answer


PROMPT = [{"system_prompt": "judge", "user_prompt": ""}]


def test_a_judge_s_traced_calls_return_its_untraced_results_with_its_reasoning():
    script = ("", ScriptedFailure("down"), "yes", "no")
    plain, traced = _ReasoningJudge(_answers(*script)), _ReasoningJudge(_answers(*script))
    expected = _validated(plain).generate_with_validation(PROMPT, ("yes", "no"))
    judgments, traces = _validated(traced).generate_with_validation_and_traces(PROMPT, ("yes", "no"))
    assert judgments == expected == ["yes"]
    assert traced.requests == plain.requests
    blank, failed, kept = (attempt["candidates"][0] for attempt in traces[0]["attempts"])
    assert (blank["completion"], blank["accepted"]) == ("", False)
    assert (failed["raw_response"], failed["completion"], REASONING_KEY in failed) == ("", "", False)
    assert (kept["raw_response"], kept["completion"], kept["accepted"]) == ("yes", "yes", True)
    assert (kept[REASONING_SUMMARIES_KEY], kept[REASONING_KEY]) == (["weighed"], "thought: yes")

    plain, traced = _ReasoningJudge(_answers("a", "b")), _ReasoningJudge(_answers("a", "b"))
    results, traces = _validated(traced).generate_with_traces(PROMPT, num_responses=2)
    assert results == _validated(plain).generate(PROMPT, num_responses=2) == [["a", "b"]]
    assert [c["result_index"] for c in traces[0]["attempts"][0]["candidates"]] == [0, 1]

    plain, traced = _ReasoningJudge(_answers('{"x": 1}')), _ReasoningJudge(_answers('{"x": 1}'))
    results, traces = _validated(traced).generate_with_json_validation_and_traces(PROMPT, temperature=0)
    assert results == _validated(plain).generate_with_json_validation(PROMPT, temperature=0) == [{"x": 1}]
    assert traces[0]["result"] == {"x": 1}


def test_a_model_backend_s_traced_calls_are_its_own(stub_backend):
    backend = stub_backend(batch_generate_fn=lambda *a: [["yes"]], num_tries=1)
    backend._batch_generate_with_details = lambda *a: [[{"raw_response": "<think>x</think>yes", "completion": "yes",
                                                          "finish_reason": "stop"}]]
    judge = ValidatedJudge(backend, num_tries=1, max_retry_seconds=None)
    _, traces = judge.generate_with_validation_and_traces(PROMPT, ("yes",))
    assert traces[0]["attempts"][0]["candidates"][0]["raw_response"] == "<think>x</think>yes"
    _, traces = judge.generate_with_traces(PROMPT)
    assert traces[0]["result"] == ["yes"]
    _, traces = judge.generate_with_json_validation_and_traces(PROMPT)
    assert traces[0]["result"] is None


@pytest.fixture
def rulesets(tmp_path):
    rules = _write(tmp_path / "rules.jsonl", [{"id": "Rule 1", "rule": "be helpful"},
                                               {"id": "Rule 2", "rule": "be honest"}])
    revised = _write(tmp_path / "revised.jsonl", [{"id": "Rule 1", "rule": "be helpful to the user"},
                                                   {"id": "Rule 2", "rule": "be honest"}])
    return rules, revised


def test_the_rubric_keeps_its_call_beside_its_result(tmp_path, rulesets):
    rules, revised = rulesets
    judge = _validated(_ReasoningJudge(_answers(INTENT_ANSWER)))
    result = run_intent_rubric(rules, revised, RUBRIC, judge, tmp_path / "out")
    trace = json.loads((tmp_path / "out" / INTENT_TRACE_FNAME).read_text())
    assert result["penalty"] == 1.0 and trace["result"] == json.loads(INTENT_ANSWER)
    assert "# Revised Ruleset" in trace["prompt"]["system_prompt"]
    assert trace["attempts"][0]["candidates"][0][REASONING_KEY] == f"thought: {INTENT_ANSWER}"


def test_a_deletion_makes_no_call_and_an_unparsed_answer_is_kept_before_the_failure(tmp_path, rulesets):
    rules, revised = rulesets
    shortened = _write(tmp_path / "short.jsonl", [{"id": "Rule 1", "rule": "be helpful"}])
    run_intent_rubric(rules, shortened, RUBRIC, _validated(ScriptedJudge(_answers("unused"))), tmp_path / "deleted")
    assert not (tmp_path / "deleted" / INTENT_TRACE_FNAME).exists()
    with pytest.raises(IntentScoringError, match="did not score every revised rule"):
        run_intent_rubric(rules, revised, RUBRIC, _validated(ScriptedJudge(_answers("no JSON"))), tmp_path / "unparsed")
    trace = json.loads((tmp_path / "unparsed" / INTENT_TRACE_FNAME).read_text())
    assert trace["result"] is None and len(trace["attempts"]) == 2
    assert not (tmp_path / "unparsed" / "intent_penalty.json").exists()
    assert run_intent_rubric(rules, revised, RUBRIC, _validated(ScriptedJudge(_answers(INTENT_ANSWER))),
                             None)["penalty"] == 1.0


def test_a_greedy_rubric_stops_after_two_attempts_and_a_sampled_one_uses_all(tmp_path, rulesets):
    rules, revised = rulesets
    for temperature, sent in ((0.0, 2), (0, 2), (0.5, 3)):
        judge = ScriptedJudge(_answers("no JSON"))
        with pytest.raises(IntentScoringError):
            run_intent_rubric(rules, revised, RUBRIC, _validated(judge), tmp_path / str(temperature),
                              temperature=temperature)
        assert len(judge.requests) == sent
    judge = ScriptedJudge(_answers("no JSON"))
    with pytest.raises(IntentScoringError):
        run_intent_rubric(rules, revised, RUBRIC, ValidatedJudge(judge, num_tries=1, max_retry_seconds=None),
                          tmp_path / "one")
    assert len(judge.requests) == 1


def test_an_incomplete_rubric_reply_is_retried_and_never_scored_as_no_penalty(tmp_path, rulesets):
    rules, revised = rulesets
    judge = ScriptedJudge(_answers('{"rules": []}', '{"rules": [{"id": "Rule 1", "score": 0}]}', INTENT_ANSWER))
    result = run_intent_rubric(rules, revised, RUBRIC, _validated(judge), tmp_path / "out", temperature=0.5)
    assert result["penalty"] == 1.0 and len(judge.requests) == 3
    trace = json.loads((tmp_path / "out" / INTENT_TRACE_FNAME).read_text())
    assert [attempt["candidates"][0]["accepted"] for attempt in trace["attempts"]] == [False, False, True]
    assert trace["result"] == json.loads(INTENT_ANSWER)


@pytest.mark.parametrize("reply", ["{}", '{"rules": []}', '{"rules": [{"id": "Rule 1", "score": "invalid"}]}'])
def test_a_rubric_reply_that_stays_incomplete_is_an_error_with_no_result(tmp_path, rulesets, reply):
    rules, revised = rulesets
    with pytest.raises(IntentScoringError, match="did not score every revised rule"):
        run_intent_rubric(rules, revised, RUBRIC, _validated(ScriptedJudge(_answers(reply))), tmp_path / "out")
    assert not (tmp_path / "out" / "intent_penalty.json").exists()
    assert json.loads((tmp_path / "out" / INTENT_TRACE_FNAME).read_text())["result"] is None


@pytest.mark.parametrize("reply_id", ["Rules 1 and 2", "Rule 1.2", "Rule 1, Rule 2", "1 2"])
def test_a_rubric_id_naming_two_numbers_does_not_score_the_rule_its_digits_spell(reply_id):
    assert _checked_scores({"rules": [{"id": reply_id, "score": 0}]}, ["Rule 12"], 4) is None
    assert _checked_scores({"rules": [{"id": "Rule 1", "score": 0}, {"id": "Rule 2", "score": 0},
                                      {"id": reply_id, "score": 0}]}, ["Rule 1", "Rule 2", "Rule 12"], 4) is None


@pytest.mark.parametrize(("reply_id", "rule_id"), [
    ("Rule 12", "Rule 12"), ("rule 012", "Rule 12"), ("12", "Rule 12"), ("Meta-Rule 3", "Meta-Rule 3"),
    (" Rule 12 ", "Rule 12"), ("Section 1.2", "Section 1.2"), ("Scope", "Scope"),
])
def test_a_rubric_id_naming_one_rule_is_read(reply_id, rule_id):
    assert _checked_scores({"rules": [{"id": reply_id, "score": 1}]}, [rule_id], 4) == [
        {"id": reply_id, "score": 1, "reasoning": ""}]


RUNAWAY = "<|channel>thought\nlet me think la l' la l'"
SCENARIOS = "# Scenario 1\n<PROMPT>\nHi\n</PROMPT>\n<RESPONSE>\nHello\n</RESPONSE>"


class _CutOffJudge(ScriptedJudge):
    def generate_once(self, prompts, **kwargs):
        return [replace(outcome, finish_reason="length") for outcome in super().generate_once(prompts, **kwargs)]


UNCLOSED = "holds an unclosed thought"


@pytest.mark.parametrize(("completion", "finish_reason", "family", "reason"), [
    ("a prior", "stop", "gemma4", None),
    ("a prior", None, "gemma4", None),
    (RUNAWAY, "stop", "gemma4", UNCLOSED),
    (f"  {RUNAWAY}", "stop", "gemma4", UNCLOSED),
    ("a prior, then <|channel>thought and more", "stop", "gemma4", UNCLOSED),  # a special token: it is a thought anywhere
    ("<|channel>thought draft<channel|>", "stop", "gemma4", UNCLOSED),  # a closed thought with no answer
    ("<|channel>thought closed<channel|> a prior", "stop", "gemma4", None),
    ("<|channel>thought closed<channel|> a prior <|channel>thought open", "stop", "gemma4", UNCLOSED),
    ("<think>never ends", "stop", "qwen3", UNCLOSED),
    ("<think>never ends", "stop", "olmo3", UNCLOSED),
    ("<think>closed</think>", "stop", "qwen3", UNCLOSED),
    ("<think>closed</think> a prior", "stop", "qwen3", None),
    ("<think>closed</think> <think>open", "stop", "qwen3", UNCLOSED),
    # a literal tag in finished text is text
    ("Explain the literal tag <think> in this text.", "stop", "qwen3", None),
    ("<think>closed</think> and the tag <think> in it", "stop", "qwen3", None),
    # another family's markers are text too
    ("Explain the literal tag <think> in this text.", "stop", "gemma4", None),
    ("<think>never ends", "stop", "gemma4", None),
    ("a tag <|channel>thought in text", "stop", "qwen3", None),
    # families whose strippers report it themselves, and judges that show reasoning apart
    ("<think>never ends", "stop", "openai_gptoss", None),
    (RUNAWAY, "stop", "muse_glimmer", None),
    (RUNAWAY, "stop", None, None),
    ("a prior cut off", "length", "gemma4", "finished on length"),
    ("a prior cut off", "max_tokens", None, "finished on length"),
])
def test_what_counts_as_an_unfinished_answer(completion, finish_reason, family, reason):
    assert unfinished_reason(completion, finish_reason, strip_thinking_model=family) == reason


def test_a_judge_reads_its_family_from_its_model_config():
    config = {"model_type": "fake", "model_name": "fake-16", "seed": 16, "strip_thinking_model": "gemma4"}
    judge = ValidatedJudge(ModelBackend(OmegaConf.create(config)), num_tries=1, max_retry_seconds=None)
    assert judge.strip_thinking_model == "gemma4"
    assert _validated(ScriptedJudge(_answers("a"))).strip_thinking_model is None


def test_a_literal_tag_in_a_finished_batch_is_read_as_its_scenarios(modelspec):
    batch = f"{SCENARIOS}\nThe tag <think> is shown here."
    agent = _generator(modelspec, ScriptedJudge(_answers(batch), strip_thinking_model="qwen3"))
    assert agent.posterior_sampling(current_rule_set="Rule 1: Be kind", already_generated_scenarios=[]) == batch


def _generator(modelspec, judge):
    cfg = OmegaConf.create({"posterior_update_temperature": 0.7, "posterior_sampling_temperature": 0.9})
    return PSRLScenarioGeneration(modelspec(), cfg, _validated(judge))


def _amender(modelspec, judge):
    cfg = OmegaConf.create({"posterior_update_temperature": 0.7, "posterior_sampling_temperature": 0.9,
                            "best_of_n": 2})
    return PSRLRuleAmendment(modelspec(), cfg, _validated(judge), weights=RewardWeights(intent=1.0))


@pytest.mark.parametrize("build", [_amender, _generator], ids=["amender", "generator"])
@pytest.mark.parametrize(("judge", "reason"), [
    (ScriptedJudge(_answers(RUNAWAY), strip_thinking_model="gemma4"), "holds an unclosed thought"),
    (_CutOffJudge(_answers("a prior that stopped at the ceil")), "finished on length"),
], ids=["unclosed", "length"])
def test_an_unfinished_update_keeps_the_old_prior_with_a_warning(tmp_path, modelspec, caplog, build, judge, reason):
    agent = build(modelspec, judge)
    prior = agent.prior
    assert agent.posterior_update(trace_path=tmp_path / PRIOR_TRACE_FNAME) == prior == agent.prior
    assert any(reason in record.getMessage() and "keeping current prior" in record.getMessage()
               for record in caplog.records if record.levelname == "WARNING")
    [row] = [json.loads(line) for line in (tmp_path / PRIOR_TRACE_FNAME).read_text().splitlines()]
    assert row["unfinished"] == {"0": reason} and row["result"]
    assert [attempt["attempt"] for attempt in row["attempts"]] == [1, 2, 3]
    assert row["retries"] == [{"attempt": 1, "reason": reason}, {"attempt": 2, "reason": reason}]
    assert sum(f"Retrying the posterior_update after attempt {n} of 3: its answer {reason}" in record.getMessage()
               for record in caplog.records for n in (1, 2)) == 2


@pytest.mark.parametrize("build", [_amender, _generator], ids=["amender", "generator"])
def test_a_retried_update_takes_the_first_finished_answer(tmp_path, modelspec, caplog, build):
    judge = ScriptedJudge(_answers(RUNAWAY, "a finished prior"), strip_thinking_model="gemma4", seed=16)
    agent = build(modelspec, judge)
    assert agent.posterior_update(trace_path=tmp_path / PRIOR_TRACE_FNAME) == "a finished prior" == agent.prior
    assert not any("keeping current prior" in record.getMessage() for record in caplog.records)
    [row] = [json.loads(line) for line in (tmp_path / PRIOR_TRACE_FNAME).read_text().splitlines()]
    assert (row["result"], row["selected_attempt"], row["unfinished"]) == (["a finished prior"], 2, {})
    assert row["retries"] == [{"attempt": 1, "reason": "holds an unclosed thought"}]
    assert [attempt["attempt"] for attempt in row["attempts"]] == [1, 2]
    assert len(judge.requests) == 2 and judge.requests[0]["seed"] != judge.requests[1]["seed"]


def test_a_retried_scenario_batch_takes_the_first_finished_answer(tmp_path, modelspec):
    judge = ScriptedJudge(_answers(RUNAWAY, RUNAWAY, SCENARIOS), strip_thinking_model="gemma4")
    agent = _generator(modelspec, judge)
    trace = tmp_path / "trace.jsonl"
    assert agent.posterior_sampling(current_rule_set="Rule 1: Be kind", already_generated_scenarios=[],
                                    trace_path=trace) == SCENARIOS
    [row] = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [retry["attempt"] for retry in row["retries"]] == [1, 2] and row["selected_attempt"] == 3
    assert len({request["seed"] for request in judge.requests}) == 3


def test_each_scenario_batch_of_one_prompt_has_a_seed_of_its_own(modelspec):
    def batch_seeds(agent_seed):
        judge = ScriptedJudge(_answers(SCENARIOS, SCENARIOS, SCENARIOS), strip_thinking_model="gemma4", seed=agent_seed)
        agent = _generator(modelspec, judge)
        for _ in range(3):
            agent.posterior_sampling(current_rule_set="Rule 1: Be kind", already_generated_scenarios=[])
        return [request["seed"] for request in judge.requests]

    seeds = batch_seeds(16)
    assert len(set(seeds)) == 3
    assert batch_seeds(16) == seeds
    assert batch_seeds(17) != seeds


def test_a_scenario_batch_counts_as_empty_only_after_its_attempts(modelspec):
    judge = ScriptedJudge(_answers(RUNAWAY), strip_thinking_model="gemma4")
    agent = _generator(modelspec, judge)
    assert agent.posterior_sampling(current_rule_set="Rule 1: Be kind", already_generated_scenarios=[]) == ""
    assert len(judge.requests) == 3


def test_an_empty_attempt_counts_against_the_same_attempts(tmp_path, modelspec):
    judge = ScriptedJudge(_answers(RUNAWAY, ""), strip_thinking_model="gemma4")
    agent = _generator(modelspec, judge)
    prior = agent.prior
    assert agent.posterior_update(trace_path=tmp_path / PRIOR_TRACE_FNAME) == prior
    assert len(judge.requests) == 3
    [row] = [json.loads(line) for line in (tmp_path / PRIOR_TRACE_FNAME).read_text().splitlines()]
    assert (row["result"], row["unfinished"]) == ([RUNAWAY], {"0": "holds an unclosed thought"})
    assert [attempt["attempt"] for attempt in row["attempts"]] == [1, 2, 3]
    assert row["attempts"][1]["seed"] != row["attempts"][0]["seed"]


def test_an_amender_s_unfinished_candidate_is_not_asked_again(tmp_path, modelspec):
    judge = ScriptedJudge(_answers(RUNAWAY, "fine"), strip_thinking_model="gemma4")
    agent = _amender(modelspec, judge)
    rules = _write(tmp_path / "rules.jsonl", [{"id": "Rule 1", "rule": "be helpful"}])
    assert agent.posterior_sampling(rules) == [RUNAWAY, "fine"]
    assert agent.unfinished_samples == {0: "holds an unclosed thought"} and len(judge.requests) == 2


def test_each_retry_reaches_a_seeded_engine_with_its_own_seed(modelspec, monkeypatch, stub_backend):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    sent = []

    class Engine:
        @staticmethod
        def generate(prompts, sampling_params):
            sent.append(sampling_params)
            return [SimpleNamespace(outputs=[SimpleNamespace(text=RUNAWAY, token_ids=[1], finish_reason="stop")])]

    class Tokenizer:
        @staticmethod
        def apply_chat_template(chat, **kwargs):
            return chat[0]["content"]

        @staticmethod
        def encode(text, add_special_tokens):
            return [1]

    def details(prompts, temperature, max_new_tokens, num_responses, seed):
        return call_local_model_with_details(Engine, Tokenizer, prompts, temperature=temperature,
                                             max_new_tokens=max_new_tokens, num_responses=num_responses, seed=seed,
                                             strip_thinking_model="gemma4")

    backend = stub_backend(lambda *args: [[]], seed=7, num_tries=5, _batch_generate_with_details=details,
                           _options=JudgeOptions({"strip_thinking_model": "gemma4"}))
    agent = _on_engine(modelspec, ValidatedJudge(backend, num_tries=5, max_retry_seconds=None))
    prior = agent.prior
    assert agent.posterior_update() == prior
    assert [params["temperature"] for params in sent] == [0.2] * 5
    assert len({params["seed"] for params in sent}) == 5


def _engine_answering(stub_backend, text, strip_thinking_model="gemma4", finish_reason="stop"):
    """A judge whose engine answers with `text`, read by the family's real stripper; `seeds` holds each attempt's seed."""
    seeds = []

    def details(prompts, temperature, max_new_tokens, num_responses, seed):
        seeds.append(seed)
        answers = text if isinstance(text, list) else [text]
        answer = answers[min(len(seeds), len(answers)) - 1]
        completion, strip_failed = _extract_text_and_strip_status({"text": answer}, strip_thinking_model)
        return [[{"raw_response": answer, "completion": completion, "finish_reason": finish_reason,
                  **({STRIP_FAILED_KEY: True} if strip_failed else {})}]]

    backend = stub_backend(lambda *args: [[]], num_tries=1, _batch_generate_with_details=details,
                           _options=JudgeOptions({"strip_thinking_model": strip_thinking_model}))
    judge = ValidatedJudge(backend, num_tries=1, max_retry_seconds=None)
    judge.engine_seeds = seeds
    return judge


def _on_engine(modelspec, judge):
    cfg = OmegaConf.create({"posterior_update_temperature": 0.2, "posterior_sampling_temperature": 1.0})
    return PSRLScenarioGeneration(modelspec(), cfg, judge)


def test_an_answer_that_is_only_a_closed_thought_is_never_text(modelspec, stub_backend, caplog):
    agent = _on_engine(modelspec, _engine_answering(stub_backend, "<|channel>thought I am still thinking...<channel|>"))
    prior = agent.prior
    assert agent.posterior_update() == prior
    assert any("Empty posterior update response" in record.getMessage() for record in caplog.records)
    with pytest.raises(ValueError, match="No output from posterior_sampling"):
        agent.posterior_sampling(current_rule_set="Rule 1: Be kind", already_generated_scenarios=[])


def test_a_finished_update_is_taken(modelspec):
    agent = _generator(modelspec, ScriptedJudge(_answers("a prior")))
    assert agent.posterior_update() == "a prior" == agent.prior


def test_an_amender_s_unfinished_samples_are_named_by_their_result_index(tmp_path, modelspec):
    agent = _amender(modelspec, ScriptedJudge(_answers("fine", RUNAWAY), strip_thinking_model="gemma4"))
    rules = _write(tmp_path / "rules.jsonl", [{"id": "Rule 1", "rule": "be helpful"}])
    assert agent.posterior_sampling(rules) == ["fine", RUNAWAY]
    assert agent.unfinished_samples == {1: "holds an unclosed thought"}
    agent = _amender(modelspec, _CutOffJudge(_answers("one", "two")))
    agent.posterior_sampling(rules)
    assert agent.unfinished_samples == {0: "finished on length", 1: "finished on length"}


@pytest.mark.parametrize(("judge", "reason"), [
    (ScriptedJudge(_answers(f"<think>{SCENARIOS}"), strip_thinking_model="qwen3"), "holds an unclosed thought"),
    (_CutOffJudge(_answers(SCENARIOS)), "finished on length"),
], ids=["unclosed", "length"])
def test_a_generator_s_unfinished_sample_holds_no_scenarios(modelspec, caplog, judge, reason):
    agent = _generator(modelspec, judge)
    assert agent.posterior_sampling(current_rule_set="Rule 1: Be kind", already_generated_scenarios=[]) == ""
    assert any(reason in record.getMessage() for record in caplog.records if record.levelname == "WARNING")


def _sized(judge, count, limit):
    validated = _validated(judge)
    validated._backend.prompt_token_count = count
    validated._backend.prompt_token_limit = limit
    return validated


@pytest.mark.parametrize("build", [PSRLRuleAmendment, PSRLScenarioGeneration], ids=["amender", "generator"])
def test_a_prompt_over_the_limit_fails_the_call_without_generating(tmp_path, modelspec, caplog, build):
    judge = ScriptedJudge(_answers("a prior"))
    cfg = OmegaConf.create({"posterior_update_temperature": 0.7, "posterior_sampling_temperature": 0.9,
                            "best_of_n": 1})
    options = {"weights": RewardWeights(intent=1.0)} if build is PSRLRuleAmendment else {}
    agent = build(modelspec(), cfg, _sized(judge, lambda prompt: 100, 100), **options)
    prior = agent.prior
    assert agent.posterior_update(trace_path=tmp_path / PRIOR_TRACE_FNAME) == prior
    assert judge.requests == []
    assert any("100 tokens" in record.getMessage() for record in caplog.records if record.levelname == "WARNING")
    [row] = [json.loads(line) for line in (tmp_path / PRIOR_TRACE_FNAME).read_text().splitlines()]
    assert (row["prompt_tokens"], row["prompt_token_limit"], row["result"], row["attempts"]) == (100, 100, None, [])
    assert "the model takes fewer than 100" in row["error"]


def test_a_sample_over_the_limit_ends_the_run_as_a_failed_call_does(tmp_path, modelspec):
    agent = _amender(modelspec, ScriptedJudge(_answers("unused")))
    agent.backend = _sized(ScriptedJudge(_answers("unused")), lambda prompt: 100, 100)
    rules = _write(tmp_path / "rules.jsonl", [{"id": "Rule 1", "rule": "be helpful"}])
    with pytest.raises(ValueError, match="No output from posterior_sampling generation in RA agent"):
        agent.posterior_sampling(rules)


def test_each_agent_call_records_its_prompt_s_token_count(tmp_path, modelspec, caplog):
    caplog.set_level("INFO")
    judge = ScriptedJudge(_answers("a prior"))
    seen = []
    agent = _agent(modelspec, _answers("unused"))
    agent.backend = _sized(judge, lambda prompt: seen.append(prompt) or 99, 100)
    assert agent.posterior_update(trace_path=tmp_path / PRIOR_TRACE_FNAME) == "a prior"
    [row] = [json.loads(line) for line in (tmp_path / PRIOR_TRACE_FNAME).read_text().splitlines()]
    assert (row["prompt_tokens"], row["prompt_token_limit"]) == (99, 100) and len(seen) == 1
    assert seen[0] == row["prompt"] and len(judge.requests) == 1
    assert any("posterior_update prompt: 99 tokens of the model's 100" in record.getMessage()
               for record in caplog.records)


def test_a_judge_that_cannot_count_cheaply_records_no_count(tmp_path, modelspec):
    agent = _agent(modelspec, _answers("a prior"))
    assert agent.backend.prompt_size(PROMPT[0]) == (None, None)
    agent.posterior_update(trace_path=tmp_path / PRIOR_TRACE_FNAME)
    [row] = [json.loads(line) for line in (tmp_path / PRIOR_TRACE_FNAME).read_text().splitlines()]
    assert (row["prompt_tokens"], row["prompt_token_limit"]) == (None, None)


def test_a_count_with_no_limit_is_recorded_and_not_refused(tmp_path, modelspec):
    agent = _agent(modelspec, _answers("unused"))
    agent.backend = _sized(ScriptedJudge(_answers("a prior")), lambda prompt: 10 ** 9, None)
    assert agent.posterior_update() == "a prior"


def _agent(modelspec, answer):
    cfg = OmegaConf.create({"posterior_update_temperature": 0.7, "posterior_sampling_temperature": 0.9,
                            "best_of_n": 2})
    return PSRLRuleAmendment(modelspec(), cfg, _validated(_ReasoningJudge(answer)),
                             weights=RewardWeights(intent=1.0))


def test_an_agent_s_update_keeps_its_trace_only_where_it_is_asked_to(tmp_path, modelspec, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _agent(modelspec, _answers("a prior")).posterior_update() == "a prior"
    assert list(tmp_path.iterdir()) == []
    assert _agent(modelspec, _answers("a prior")).posterior_update(trace_path=tmp_path / PRIOR_TRACE_FNAME) == (
        "a prior")
    [row] = [json.loads(line) for line in (tmp_path / PRIOR_TRACE_FNAME).read_text().splitlines()]
    assert (row["call"], row["result"]) == ("posterior_update", ["a prior"])
    assert row["attempts"][0]["candidates"][0][REASONING_KEY] == "thought: a prior"
