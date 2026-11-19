from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.identity import generate_fixtures as gen
from tests.identity import published as pub

ROOT = "demo-root"
CONDITION = {
    "condition_id": "judge/bank", "prompt_style": "forced-grpo", "individual_rule_context": "isolated",
    "judge_type": "No_Interpretation", "rule_set_name": "rules.jsonl", "rule_set_sha256": "a" * 64,
    "forced_json_requirement": "complete_rule_map", "label_rule": "lenient",
    "evaluation_mode": "both", "strict_think": None, "retry_scope": None, "model_type": "local",
}
SCENARIO = {"prompt": "May I?", "response": "Yes."}


def _component(component_id: str, seed: int, **fields) -> dict:
    return {"component_id": component_id, "valid": True, "failure_reason": None, "judgment": "no",
            "finish_reason": "stop", "generation_seed": seed, "seed_sent": True,
            "raw_output_sha256": f"{component_id}-raw", "provider_request_ref": None, "request_block_ref": None,
            "rule_judgments": None, "preempted_rules": [], "prompt_sha256": f"{component_id}-prompt", **fields}


def _attempt(attempt_id: int, components: list[dict]) -> dict:
    return {"campaign_id": "demo", "condition_id": "judge/bank", "scenario_id": "s1", "decoder_draw_id": 0,
            "semantic_attempt_id": attempt_id, "request_block_id": f"demo/judge/bank/block-000{attempt_id}",
            "status": "accepted", "components": components, "failure_reason": None}


def _raw(attempt: dict, component: dict) -> dict:
    return {"scenario_id": attempt["scenario_id"], "semantic_attempt_id": attempt["semantic_attempt_id"],
            "request_block_id": attempt["request_block_id"], "component_id": component["component_id"],
            "raw_output_sha256": component["raw_output_sha256"], "raw_response": "<think>x</think>no",
            "completion": "no", "finish_reason": "stop", "error": None}


def _write_lines(path: Path, rows: list[dict]) -> list[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [(json.dumps(row) + "\n").encode("utf-8") for row in rows]
    path.write_bytes(b"".join(lines))
    return lines


@pytest.fixture
def published(tmp_path):
    """A one-condition artifact root and the table rows the generator's builders write from it."""
    base = tmp_path / "sampling" / ROOT
    base.mkdir(parents=True)
    (base / "sampling_manifest.json").write_text(json.dumps({"generation_seed_base": 16}))
    (base / "judge" / "bank").mkdir(parents=True)
    (base / "judge" / "bank" / "condition.json").write_text(json.dumps(CONDITION))
    first = _attempt(0, [_component("full_ruleset", 123), _component("rule1", 7)])
    retry = _attempt(1, [_component("rule1", 9, provider_request_ref="ns1234-0")])
    attempts = base / "judge" / "bank" / "draw-000" / "attempts.jsonl"
    attempt_lines = _write_lines(attempts, [first, retry])
    raw_lines = _write_lines(attempts.parent / "raw_outputs.jsonl",
                             [_raw(first, component) for component in first["components"]]
                             + [_raw(retry, component) for component in retry["components"]])

    def source(number):
        return gen._source(tmp_path, attempts, number, attempt_lines[number])

    raws = [{"row": gen.stored_raw(json.loads(line), strict_think=False, truncated=False), "line": number,
             "sha256": pub.sha256_bytes(line)} for number, line in enumerate(raw_lines)]
    tables = {
        pub.SEEDS: [gen._seed_row(first, first["components"][0], 16, source(0)), gen._block_row(retry, source(1))],
        pub.PROMPTS: [{"kind": "scenario", "scenario_id": "s1", "row": SCENARIO},
                      gen._prompt_row(ROOT, CONDITION, first, first["components"][1], source(0))],
        pub.SETTLEMENT: [gen._attempt_row(ROOT, CONDITION, first, 2, source(0), raws[:2]),
                         gen._attempt_row(ROOT, CONDITION, retry, 2, source(1), raws[2:])],
    }
    return tmp_path, json.loads(json.dumps(tables))


def _problems(published, population=None):
    root, tables = published
    return gen.check_tables(root, tables, {"s1": SCENARIO} if population is None else population)


def test_rows_built_from_their_sources_pass(published):
    assert _problems(published) == []
    assert published[1][pub.SETTLEMENT][1]["category"][-1] is True  # the retry holds fewer than the bundle's two


@pytest.mark.parametrize(
    ("table", "index", "change", "reported"),
    [
        (pub.SEEDS, 0, {"generation_seed": 456}, "generation_seed not what the line"),
        (pub.SEEDS, 0, {"generation_seed_base": 17}, "generation_seed_base not what the line"),
        (pub.SEEDS, 0, {"scenario_id": "s2"}, "scenario_id not what the line"),
        (pub.SEEDS, 1, {"provider_request_ref": "ns9999-0"}, "provider_request_ref not what the line"),
        (pub.SEEDS, 1, {"request_block_id": "demo/judge/bank/block-0007"}, "request_block_id not what the line"),
        (pub.PROMPTS, 1, {"prompt_sha256": "f" * 64}, "prompt_sha256 not what the line"),
        (pub.PROMPTS, 1, {"rule_set_sha256": "b" * 64}, "rule_set_sha256 not what the line"),
        (pub.PROMPTS, 1, {"individual_rule_context": "implicated", "root": "other"},
         "individual_rule_context, root not what the line"),
        (pub.PROMPTS, 1, {"component_id": "rule9"}, "holds no component 'rule9'"),
        (pub.SETTLEMENT, 0, {"condition": {**CONDITION, "label_rule": "bare"}}, "condition not what the line"),
        (pub.SETTLEMENT, 0, {"condition": {}}, "condition not what the line"),
        (pub.SETTLEMENT, 1, {"category": ["forced-grpo"]}, "category not what the line"),
        (pub.SETTLEMENT, 0, {"raw_path": "elsewhere/raw_outputs.jsonl"}, "raw_path not what the line"),
        (pub.SEEDS, 0, {"kind": "mystery"}, "a row of kind 'mystery', which no check reads"),
        (pub.PROMPTS, 0, {"row": {**SCENARIO, "response": "No."}}, "scenario s1 is not the population's row"),
    ],
)
def test_a_value_that_is_not_its_source_s_is_reported(published, table, index, change, reported):
    published[1][table][index].update(change)
    problems = _problems(published)
    assert len(problems) == 1 and reported in problems[0], problems


def test_a_changed_line_and_a_line_past_the_end_are_reported(published):
    tables = published[1]
    tables[pub.SEEDS][0]["source"]["sha256"] = "0" * 64
    tables[pub.SEEDS][1]["source"]["line"] = 5
    tables[pub.SETTLEMENT][0]["raw"][1]["sha256"] = "0" * 64
    tables[pub.SETTLEMENT][0]["raw"][0]["row"]["completion"] = "yes"
    problems = _problems(published)
    assert [problem.split(": ", 1)[1] for problem in problems] == [
        f"sampling/{ROOT}/judge/bank/draw-000/attempts.jsonl:0 is not the recorded line",
        f"sampling/{ROOT}/judge/bank/draw-000/attempts.jsonl:5: the ledger has no such line",
        f"sampling/{ROOT}/judge/bank/draw-000/raw_outputs.jsonl:0 is not the stored raw row",
        f"sampling/{ROOT}/judge/bank/draw-000/raw_outputs.jsonl:1 is not the recorded line",
    ]


def test_a_raw_row_of_another_component_or_attempt_is_reported(published):
    raws = published[1][pub.SETTLEMENT][0]["raw"]
    raws[1]["row"]["semantic_attempt_id"] = 1
    raws.append(raws[0])
    problems = _problems(published)
    assert sum("is not one of its components' outputs" in problem for problem in problems) == 2
    assert any("raw_outputs.jsonl:1 is not the stored raw row" in problem for problem in problems)


def test_an_attempt_missing_the_raw_row_of_a_component_it_recorded_is_reported(published):
    raws = published[1][pub.SETTLEMENT][0]["raw"]
    dropped = raws.pop()["row"]["component_id"]
    problems = _problems(published)
    assert len(problems) == 1 and f"no raw row for the recorded output of {dropped}" in problems[0], problems


def test_a_source_outside_a_draw_s_ledger_is_reported(published):
    row = published[1][pub.SEEDS][0]
    row["source"]["path"] = f"sampling/{ROOT}/judge/bank/attempts.jsonl"
    assert _problems(published) == [f"sampling/{ROOT}/judge/bank/attempts.jsonl is not a draw's attempt ledger "
                                    "under sampling/<root>/"]


def test_a_ledger_without_its_condition_or_a_raw_ledger_that_is_gone_is_reported(published):
    root, _ = published
    (root / "sampling" / ROOT / "judge" / "bank" / "draw-000" / "raw_outputs.jsonl").unlink()
    assert {problem.split(": ", 2)[-1] for problem in _problems(published)} == {"the raw ledger has no such line"}
    (root / "sampling" / ROOT / "judge" / "bank" / "condition.json").unlink()
    assert {problem.split(": ", 2)[-1] for problem in _problems(published)} == {
        "no condition.json beside its draw, or no sampling_manifest.json in its root",
        "the raw ledger has no such line"}


@pytest.fixture
def batch(tmp_path):
    directory = tmp_path / "batch" / "work"
    (directory / "inputs").mkdir(parents=True)
    lines = ['{"custom_id": "ns1-0"}\n', '{"custom_id": "ns1-1"}\n']
    (directory / "inputs" / "part-0000.jsonl").write_text("".join(lines))
    (directory / "state.json").write_text(json.dumps({"endpoint": "/v1/responses"}))
    row = {"kind": "batch", "provider": "openai_batch", "work_dir": "work", "endpoint": "/v1/responses",
           "parts": [{"name": "part-0000.jsonl", "sha256": pub.sha256_bytes("".join(lines).encode()),
                      "lines": lines}]}
    return tmp_path, row


def test_a_batch_row_is_its_work_directory_s_inputs_line_for_line(batch):
    root, row = batch
    assert gen.check_batch(root, [row]) == []
    row["parts"][0]["lines"] = ['{"custom_id": "ns1-1"}\n', '{"custom_id": "ns1-0"}\n']
    row["endpoint"] = "/v1/chat/completions"
    assert gen.check_batch(root, [row]) == [
        f"{pub.BATCH}: work: endpoint '/v1/chat/completions', but the work directory's is '/v1/responses'",
        f"{pub.BATCH}: work: part-0000.jsonl's lines are not the recorded input's",
    ]
    (root / "batch" / "work" / "inputs" / "part-0001.jsonl").write_text("{}\n")
    row["parts"].append({"name": "part-0002.jsonl", "sha256": "0" * 64, "lines": []})
    assert gen.check_batch(root, [row]) == [
        f"{pub.BATCH}: work: its parts are not the work directory's inputs ['part-0000.jsonl', 'part-0001.jsonl']",
        f"{pub.BATCH}: work: endpoint '/v1/chat/completions', but the work directory's is '/v1/responses'",
        f"{pub.BATCH}: work: part-0000.jsonl's lines are not the recorded input's",
        f"{pub.BATCH}: work: part-0002.jsonl is not the recorded input",
    ]
