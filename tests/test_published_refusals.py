from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from ruleforge import hub
from ruleforge.core.errors import RuleForgeError
from ruleforge.evaluate.ledger import AttemptRecord, ComponentRecord, raw_output_sha256
from ruleforge.evaluate.stack import stack_digest
from ruleforge.hub import fetch, pack

# Every refusal the paper's rebuild readers make on a file kind the package publishes survives publication: each test
# plants the violation in a recorded root, publishes the root through the packer under the profile of the roots that
# reader reads (a real pack and fetch), and holds the published copy to the same refusal; or the packer refuses to
# publish it.
SCENARIOS = ("s1", "s2")
CONDITION = "m/forced"
MANIFEST_SHA, FINGERPRINT = "a" * 64, "f" * 64
RUNTIME = {"vllm_version": "0.19.0", "torch_version": "2.10.0", "transformers_version": "5.5.0",
           "tokenizers_version": "0.22.2", "triton_version": "3.6.0",
           "precision_env": {"VLLM_FLOAT32_MATMUL_PRECISION": "highest"},
           "engine_args": {"dtype": "bfloat16", "max_model_len": 8192, "model": "judge"}, "hostname": "host", "pid": 7}
NEWER = {**RUNTIME, "vllm_version": "0.28.0"}
LOCAL = ("prompt-ablation-local", "hierarchy-local", "training-gemma", "training-qwen")
OUTCOMES = tuple(hub.PROFILES)


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _json(path: Path, record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def _attempt(sid: str, draw: int, *, judged_on: str, judgment: str = "yes") -> dict:
    full = ComponentRecord("full_ruleset", True, None, judgment, "stop", 5, True, raw_output_sha256(judgment),
                           rule_judgments={"Rule 1": "yes"})
    return AttemptRecord("c", CONDITION, sid, draw, 0, f"b{draw}", "accepted", "local_deterministic", (full,),
                         "2026-10-01T00:00:00Z", "2026-10-01T00:00:01Z", judged_on=judged_on).to_dict()


def recorded(tmp_path: Path, draws: int = 2) -> Path:
    """A complete recorded campaign root of a local judge: two scenarios, every kind the package publishes."""
    root = tmp_path / "recorded" / "sampling" / "r1"
    _json(root / "sampling_manifest.json", {"campaign_id": "c"})
    for draw in range(draws):
        place = root / CONDITION / f"draw-{draw:03d}"
        _jsonl(place / "attempts.jsonl", [_attempt(sid, draw, judged_on=stack_digest(RUNTIME)) for sid in SCENARIOS])
        _jsonl(place / "raw_outputs.jsonl", [{"scenario_id": sid, "decoder_draw_id": draw, "semantic_attempt_id": 0,
                                               "request_block_id": f"b{draw}", "component_id": "full_ruleset",
                                               "raw_response": "yes", "completion": "yes"} for sid in SCENARIOS])
        _jsonl(place / "bundle_outcomes.jsonl", [
            {"scenario_id": sid, "decoder_draw_id": draw, "semantic_attempt_id": 0, "full_set_judgment": "yes",
             "rule_judgments": {"Rule 1": "yes"}, "in_context_rule_judgments": {"Rule 1": "yes"},
             "contradiction": 0.0, "full_set_phi": 1.0, "max_rule_phi": 1.0, "penalty": 0.0} for sid in SCENARIOS])
        _json(place / "completion.json", {"state": "complete", "decoder_draw_id": draw,
                                          "scenario_manifest_sha256": MANIFEST_SHA, "condition_fingerprint": FINGERPRINT,
                                          "run_log_rows": 1, "materialized": 2, "accepted": 2, "campaign_id": "c",
                                          "condition_id": CONDITION, "pending": 0})
        _json(place / "draw_provenance.json", {"model_type": "local", "runtime": RUNTIME, "tp_size": 1,
                                               "attention_backend": "FLASH", "model_weights": {"sha256": "w" * 64}})
        _jsonl(place / "runs.jsonl", [{"runtime": RUNTIME, "code_revision": "r", "code_dirty": False,
                                       "code_state_sha256": "s" * 64}])
        _jsonl(place / "request_blocks.jsonl", [{"request_block_id": f"b{draw}", "scenario_ids": list(SCENARIOS),
                                                 "recorded_at": "2026-10-01T00:00:00Z"}])
        if draw == 0:
            _jsonl(place / "evaluations.jsonl", [{"scenario_id": sid, "scenario": {"source": "wildchat", "prompt": "p"}}
                                                 for sid in SCENARIOS])
    return root


def publish(root: Path, profile: str) -> Path:
    """``root`` as a fetched published copy of it holds it, under ``profile``."""
    artifacts, out = root.parent.parent, root.parent.parent.parent / f"repo-{profile}"
    inventory = {"format": 1, "experiments": {"x": {"profile": profile, "roots": {root.name: None},
                                                    "all_attempts": [root.name], "raw_outputs": [root.name],
                                                    "draw0_evaluations": True}}}
    pack(artifacts, inventory, out, level=1, echo=lambda _: None)
    fetched = out.parent / f"fetched-{profile}"
    fetched.mkdir()
    fetch(out, fetched, echo=lambda _: None)
    return fetched / "sampling" / root.name


def same_refusal(read, root: Path, profile: str) -> str:
    """The refusal ``read`` gives on ``root``, recorded, and on its published copy under ``profile``: the same, the root's
    place aside."""
    with pytest.raises((RuleForgeError, ValueError)) as before:
        read(root)
    published = publish(root, profile)
    with pytest.raises((RuleForgeError, ValueError)) as after:
        read(published)
    shown = str(before.value).replace(str(root), "<root>")
    assert str(after.value).replace(str(published), "<root>") == shown
    return shown


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _edit_rows(path: Path, edit) -> None:
    rows = _rows(path)
    edit(rows)
    _jsonl(path, rows)


def _edit(path: Path, edit) -> None:
    record = json.loads(path.read_text())
    edit(record)
    _json(path, record)


def _outcomes(root: Path):
    from ruleforge.evaluate.inference import load_condition_outcomes

    return load_condition_outcomes(root / CONDITION, expected_scenario_ids=SCENARIOS, expected_draws=2,
                                   scenario_manifest_sha256=MANIFEST_SHA)


DRAW0 = f"{CONDITION}/draw-000"
# The outcome reader (inference.read_draw_outcomes, load_condition_outcomes), every root's: each planted violation.
OUTCOME_VIOLATIONS = {
    "an incomplete draw": (lambda root: _edit(root / DRAW0 / "completion.json",
                                              lambda r: r.update(state="running")), "completion state is 'running'"),
    "another draw's gate": (lambda root: _edit(root / DRAW0 / "completion.json",
                                               lambda r: r.update(decoder_draw_id=1)), "completion record is for draw 1"),
    "another population": (lambda root: _edit(root / DRAW0 / "completion.json",
                                              lambda r: r.update(scenario_manifest_sha256="b" * 64)),
                           "is not the campaign's"),
    "a scenario twice": (lambda root: _edit_rows(root / DRAW0 / "bundle_outcomes.jsonl",
                                                 lambda rows: rows.append(rows[0])), "appears twice"),
    "another draw's row": (lambda root: _edit_rows(root / DRAW0 / "bundle_outcomes.jsonl",
                                                   lambda rows: rows[0].update(decoder_draw_id=1)),
                           "an outcome row claims draw 1"),
    "a missing scenario": (lambda root: _edit_rows(root / DRAW0 / "bundle_outcomes.jsonl", lambda rows: rows.pop()),
                           "outcomes cover 1 scenarios, the manifest 2"),
    "two fingerprints": (lambda root: _edit(root / DRAW0 / "completion.json",
                                            lambda r: r.update(condition_fingerprint="e" * 64)),
                         "draws disagree on the condition fingerprint"),
}


@pytest.mark.parametrize("profile", OUTCOMES)
@pytest.mark.parametrize("case", sorted(OUTCOME_VIOLATIONS))
def test_the_outcome_reader_refuses_a_published_copy_as_the_recorded_root(tmp_path, profile, case):
    plant, message = OUTCOME_VIOLATIONS[case]
    root = recorded(tmp_path)
    plant(root)
    assert message in same_refusal(_outcomes, root, profile)


@pytest.mark.parametrize("profile", [name for name in OUTCOMES if name != "ruleset-size"])
def test_a_grid_of_a_quantity_a_row_lacks_is_refused_on_a_published_copy_too(tmp_path, profile):
    from ruleforge.evaluate.inference import outcome_grid

    root = recorded(tmp_path)
    _edit_rows(root / DRAW0 / "bundle_outcomes.jsonl", lambda rows: rows[0].update(contradiction=None))

    def read(place):
        cells, _ = _outcomes(place)
        return outcome_grid(CONDITION, cells, scenario_ids=SCENARIOS, n_draws=2)

    assert "has no 'contradiction'" in same_refusal(read, root, profile)


def _arm_stack(root: Path):
    from ruleforge.experiments.campaign_roots import arm_stack

    return arm_stack(SimpleNamespace(path=root), SimpleNamespace(condition_id=CONDITION, model_type="local"), "arm")


STACK_VIOLATIONS = {
    # a run cut from the log its gate counted
    "a cut run log": (lambda root: _edit(root / DRAW0 / "completion.json", lambda r: r.update(run_log_rows=2)),
                      "rows were cut from the run log since"),
    # a row judged on a stack neither the provenance nor any run records
    "an unlogged stack": (lambda root: _edit_rows(root / DRAW0 / "attempts.jsonl",
                                                  lambda rows: rows[0].update(judged_on="0" * 64)),
                          "neither its draw_provenance.json nor any complete row of its runs.jsonl records"),
    # a local draw's rows judged on a run's other stack with no stack change logged: model_type decides this one
    "an unlogged stack change": (lambda root: (
        _edit_rows(root / DRAW0 / "runs.jsonl", lambda rows: rows.append({"runtime": NEWER})),
        _edit(root / DRAW0 / "completion.json", lambda r: r.update(run_log_rows=2)),
        _edit_rows(root / DRAW0 / "attempts.jsonl", lambda rows: rows[0].update(judged_on=stack_digest(NEWER)))),
        "were judged on another stack than its provenance's"),
    # the same change logged by the run (stack_change): its rows read as judged on the run's stack, so the arm holds
    # two stacks; a copy without the logged change would refuse it as unlogged instead
    "a logged stack change": (lambda root: (
        _edit_rows(root / DRAW0 / "runs.jsonl", lambda rows: rows.extend([
            {"runtime": NEWER}, {"runtime": NEWER, "stack_change": {"vllm_version": ["0.19.0", "0.28.0"]}}])),
        _edit(root / DRAW0 / "completion.json", lambda r: r.update(run_log_rows=3)),
        _edit_rows(root / DRAW0 / "attempts.jsonl", lambda rows: rows[0].update(judged_on=stack_digest(NEWER)))),
        "own judgments come from several stacks"),
    "no stack recorded": (lambda root: [_edit(root / f"{CONDITION}/draw-00{draw}/draw_provenance.json",
                                              lambda r: r.update(runtime={})) for draw in (0, 1)],
                          "record no inference stack"),
    "two stacks": (lambda root: (
        _edit(root / f"{CONDITION}/draw-001/draw_provenance.json", lambda r: r.update(runtime=NEWER)),
        _edit_rows(root / f"{CONDITION}/draw-001/attempts.jsonl",
                   lambda rows: [row.update(judged_on=stack_digest(NEWER)) for row in rows]),
        _edit_rows(root / f"{CONDITION}/draw-001/runs.jsonl", lambda rows: rows[0].update(runtime=NEWER))),
        "own judgments come from several stacks"),
}


@pytest.mark.parametrize("profile", LOCAL)
@pytest.mark.parametrize("case", sorted(STACK_VIOLATIONS))
def test_the_stack_checks_refuse_a_published_copy_as_the_recorded_root(tmp_path, profile, case):
    plant, message = STACK_VIOLATIONS[case]
    root = recorded(tmp_path)
    plant(root)
    assert message in same_refusal(_arm_stack, root, profile)


def _leniency(root: Path):
    from ruleforge.experiments.leniency.analysis import read_condition, require_materialized

    _, draws = read_condition(root, CONDITION, campaign_id="c", scenario_ids=SCENARIOS, n_draws=2,
                              manifest_sha256=MANIFEST_SHA, rules=[SimpleNamespace(rule_id="Rule 1")],
                              reads_preemption=False)
    require_materialized(draws, fingerprint=FINGERPRINT, campaign_id="c", condition_id=CONDITION, where="arm")


LENIENCY_VIOLATIONS = {
    "an outcome no ledger row accepted": (
        lambda root: _edit_rows(root / DRAW0 / "attempts.jsonl", lambda rows: rows.pop()), "no accepted ledger row"),
    "another accepted judgment": (
        lambda root: _edit_rows(root / DRAW0 / "bundle_outcomes.jsonl",
                                lambda rows: rows[0].update(full_set_judgment="no")),
        "the accepted full-set response is not the one the bundle outcome was built from"),
    "another accepted attempt": (
        lambda root: _edit_rows(root / DRAW0 / "bundle_outcomes.jsonl",
                                lambda rows: rows[0].update(semantic_attempt_id=1)),
        "the accepted full-set response is not the one the bundle outcome was built from"),
    "another rule map": (
        lambda root: _edit_rows(root / DRAW0 / "bundle_outcomes.jsonl",
                                lambda rows: rows[0].update(in_context_rule_judgments={"Rule 1": "no"})),
        "the accepted full-set response is not the one the bundle outcome was built from"),
    "other rules": (lambda root: _edit_rows(root / DRAW0 / "bundle_outcomes.jsonl",
                                            lambda rows: rows[0].update(rule_judgments={"Rule 2": "yes"})),
                    "rule_judgments do not cover exactly the rules file's ids"),
    "an unmaterialized draw": (lambda root: _edit(root / DRAW0 / "completion.json",
                                                  lambda r: r.update(materialized=0)),
                               "a draw is not materialized under this campaign and condition"),
    "another campaign's draw": (lambda root: _edit(root / DRAW0 / "completion.json",
                                                   lambda r: r.update(campaign_id="d")),
                                "a draw is not materialized under this campaign and condition"),
}


@pytest.mark.parametrize("profile", ("prompt-ablation-local", "prompt-ablation-api"))
@pytest.mark.parametrize("case", sorted(LENIENCY_VIOLATIONS))
def test_the_leniency_reader_refuses_a_published_copy_as_the_recorded_root(tmp_path, profile, case):
    plant, message = LENIENCY_VIOLATIONS[case]
    root = recorded(tmp_path)
    plant(root)
    assert message in same_refusal(_leniency, root, profile)


def _ruleforge(root: Path):
    from ruleforge.experiments.ruleforge.comparison import require_materialized

    _, provenance = _outcomes(root)
    require_materialized(provenance, CONDITION, fingerprint=FINGERPRINT, campaign_id="c", accepted=2)


@pytest.mark.parametrize("field, value", [("accepted", 1), ("materialized", 0), ("condition_id", "m/other")])
def test_the_ruleforge_reader_refuses_a_published_copy_as_the_recorded_root(tmp_path, field, value):
    root = recorded(tmp_path)
    _edit(root / DRAW0 / "completion.json", lambda r: r.update({field: value}))
    assert "incomplete materialization or wrong draw identity" in same_refusal(_ruleforge, root, "ruleforge")


def _strata(root: Path):
    from ruleforge.evaluate.discrimination import read_strata

    return read_strata([root / CONDITION], SCENARIOS)


@pytest.mark.parametrize("profile", ("training-gemma", "training-qwen"))
def test_the_training_readers_refuse_a_published_copy_as_the_recorded_root(tmp_path, profile):
    from ruleforge.experiments.training.campaign_readout import _engine_disclosure
    from ruleforge.experiments.training.family_readout import _member_merge, check_engine, draw_provenances

    cases = [
        (lambda root: _edit_rows(root / DRAW0 / "evaluations.jsonl", lambda rows: rows.pop()), _strata,
         "its scenario rows are not the campaign's 2 scenarios"),
        (lambda root: _edit_rows(root / DRAW0 / "evaluations.jsonl",
                                 lambda rows: rows[0]["scenario"].pop("source")), _strata,
         "1 of 2 scenarios carry no 'source'"),
        (lambda root: _edit(root / f"{CONDITION}/draw-001/draw_provenance.json",
                            lambda r: r.update(attention_backend="OTHER")),
         lambda root: _engine_disclosure([root / CONDITION], 2, "arm"), "ran on different engine settings"),
        (lambda root: _edit(root / f"{CONDITION}/draw-001/draw_provenance.json",
                            lambda r: r["runtime"].update(vllm_version="0.28.0")),
         lambda root: _engine_disclosure([root / CONDITION], 2, "arm"), "ran on different engine settings"),
        (lambda root: _edit(root / f"{CONDITION}/draw-001/draw_provenance.json",
                            lambda r: r["model_weights"].update(format="other")),
         lambda root: _member_merge("trained", SimpleNamespace(model_weights_sha256="w" * 64, adapter_sha256=None,
                                                               condition_id=CONDITION),
                                    draw_provenances(root / CONDITION, 2), "family"),
         "not one merged checkpoint of digest"),
        (lambda root: _edit(root / f"{CONDITION}/draw-001/draw_provenance.json", lambda r: r.update(tp_size=2)),
         lambda root: check_engine({"base": {"signatures": [], "description": "", "short": ""}},
                                   {"base": draw_provenances(root / CONDITION, 2)}, "family"),
         "the arms ran at tensor-parallel sizes"),
    ]
    for number, (plant, read, message) in enumerate(cases):
        root = recorded(tmp_path / f"case{number}")
        plant(root)
        assert message in same_refusal(read, root, profile), number


def test_a_ledger_the_strict_replay_refuses_is_never_published(tmp_path):
    root = recorded(tmp_path)
    _edit_rows(root / DRAW0 / "attempts.jsonl", lambda rows: rows[0].update(semantic_attempt_id=1))
    for profile in hub.PROFILES:
        with pytest.raises(RuleForgeError, match=re.escape("has semantic_attempt_id 1 for scenario s1; the ledger holds "
                                                           "0 consumed attempt(s), so 0 was expected")):
            publish(root, profile)
