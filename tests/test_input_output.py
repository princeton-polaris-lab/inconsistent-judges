from __future__ import annotations

import json
import logging
import re

import pytest

from ruleforge.core.artifacts import ArtifactSchemaError
from ruleforge.core.files import process_jsonl_file
from ruleforge.optimize.input_output import (
    read_string_from_file,
    save_data_to_jsonl,
    save_string_to_file,
)
from ruleforge.tasks.history import (
    build_scenario_set_string,
    sample_history_scenarios,
)

KEYS = ("prompt", "response")


def test_the_reader_refuses_a_line_that_does_not_parse_naming_it_and_passes_blank_lines(tmp_path, caplog):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert list(process_jsonl_file(path)) == [{"a": 1}, {"a": 2}]
    assert caplog.text == ""
    path.write_text('{"a": 1}\n{"a": 2}\nnot json\n{"a": 3}\n', encoding="utf-8")
    with pytest.raises(ArtifactSchemaError, match=f"^{re.escape(str(path))}:3: invalid JSON"):
        list(process_jsonl_file(path))


def test_the_optimizer_refuses_an_initial_ruleset_with_a_torn_line_naming_the_line(tmp_path):
    from omegaconf import OmegaConf

    from ruleforge.optimize.trainer import check_run_inputs

    rules = tmp_path / "rules.jsonl"
    rules.write_text('{"id": "Rule 1", "rule": "Be kind."}\n{"id": "Rule 2", "rule": "Be\n{"id": "Rule 3", "rule": "x"}\n',
                     encoding="utf-8")
    windows = OmegaConf.create({"eval_cfg": {}, **{section: {"termination_episodes": 3}
                                                 for section in ("rule_amendment", "scenario_generation")}})
    with pytest.raises(ArtifactSchemaError, match=f"^{re.escape(str(rules))}:2: invalid JSON"):
        check_run_inputs(windows, rules)


def test_a_scenario_set_string_is_built_from_a_file_or_from_rows_alike(tmp_path):
    rows = [{"prompt": "p1", "response": "r1", "source": "x"}, {"prompt": "p2", "response": "r2"}]
    path = tmp_path / "scenarios.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    expected = "## Scenario 1\nPrompt: p1\n\nResponse: r1\n## Scenario 2\nPrompt: p2\n\nResponse: r2\n"
    assert build_scenario_set_string(path, KEYS) == build_scenario_set_string(rows, KEYS) == expected
    with pytest.raises(ValueError, match="must be a path to a JSONL file or a list of dictionaries"):
        build_scenario_set_string(str(path), KEYS)


def test_history_examples_need_a_history():
    with pytest.raises(ValueError, match="No history"):
        sample_history_scenarios([])


def test_a_string_file_is_written_in_its_new_directory_and_replaced_by_a_later_write(tmp_path):
    path = tmp_path / "out" / "text.txt"
    save_string_to_file("first", path)
    save_string_to_file("second", path)
    assert read_string_from_file(path) == "second"


def test_a_list_is_saved_one_row_a_line_and_anything_else_as_one_line(tmp_path):
    path = tmp_path / "out" / "data.jsonl"
    save_data_to_jsonl([{"a": 1}, {"a": 2.5}], path)
    assert path.read_text(encoding="utf-8") == '{"a": 1}\n{"a": 2.5}\n'
    save_data_to_jsonl({"a": "é"}, path)
    assert path.read_text(encoding="utf-8") == '{"a": "é"}\n'
