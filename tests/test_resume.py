from __future__ import annotations

import json
import logging

import pytest

from ruleforge.core.resume import count_complete_jsonl_records, prepare_resume_skip


def _write(path, *lines):
    path.write_bytes("".join(lines).encode("utf-8"))
    return path


def _record(index):
    return json.dumps({"i": index}) + "\n"


def test_missing_file_counts_zero(tmp_path):
    assert count_complete_jsonl_records(tmp_path / "absent.jsonl") == 0


def test_empty_file_counts_zero(tmp_path):
    assert count_complete_jsonl_records(_write(tmp_path / "a.jsonl")) == 0


def test_complete_file_counts_every_record(tmp_path):
    path = _write(tmp_path / "a.jsonl", _record(0), _record(1), _record(2))
    assert count_complete_jsonl_records(path) == 3


def test_unterminated_final_line_is_not_counted(tmp_path):
    # The exact shape of a process killed mid-write.
    path = _write(tmp_path / "a.jsonl", _record(0), '{"i": 1}')
    assert count_complete_jsonl_records(path) == 1


def test_malformed_line_stops_the_count(tmp_path):
    path = _write(tmp_path / "a.jsonl", _record(0), "not json\n", _record(2))
    assert count_complete_jsonl_records(path) == 1


def test_merged_records_with_no_separator_stop_the_count(tmp_path):
    path = _write(tmp_path / "a.jsonl", _record(0), '{"i": 1}{"i": 2}\n')
    assert count_complete_jsonl_records(path) == 1


def test_blank_line_stops_the_count(tmp_path):
    path = _write(tmp_path / "a.jsonl", _record(0), "\n", _record(2))
    assert count_complete_jsonl_records(path) == 1


def test_undecodable_bytes_stop_the_count(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_bytes(_record(0).encode("utf-8") + b"\xff\xfe\n" + _record(2).encode())
    assert count_complete_jsonl_records(path) == 1


def test_counting_without_repair_leaves_the_file_untouched(tmp_path):
    path = _write(tmp_path / "a.jsonl", _record(0), '{"i": 1}')
    before = path.read_bytes()
    assert count_complete_jsonl_records(path, repair=False) == 1
    assert path.read_bytes() == before


def test_repair_truncates_the_torn_tail(tmp_path):
    path = _write(tmp_path / "a.jsonl", _record(0), _record(1), '{"i": 2')
    assert count_complete_jsonl_records(path, repair=True) == 2
    assert path.read_bytes() == (_record(0) + _record(1)).encode("utf-8")


@pytest.mark.parametrize("leading", [0, 1])
def test_repair_keeps_a_whole_last_record_that_lost_only_its_newline(tmp_path, leading):
    whole = "".join(_record(i) for i in range(leading + 1))
    path = _write(tmp_path / "a.jsonl", whole.removesuffix("\n"))
    assert count_complete_jsonl_records(path) == leading
    assert count_complete_jsonl_records(path, repair=True) == leading + 1
    assert path.read_bytes() == whole.encode("utf-8")


def test_repair_is_a_no_op_on_a_clean_file(tmp_path):
    path = _write(tmp_path / "a.jsonl", _record(0), _record(1))
    before = path.read_bytes()
    assert count_complete_jsonl_records(path, repair=True) == 2
    assert path.read_bytes() == before


def test_repair_then_append_reproduces_the_uninterrupted_file(tmp_path):
    clean = (_record(0) + _record(1) + _record(2)).encode("utf-8")

    interrupted = _write(tmp_path / "a.jsonl", _record(0), _record(1), '{"i": 2')
    skip = count_complete_jsonl_records(interrupted, repair=True)
    with interrupted.open("a", encoding="utf-8") as handle:
        for index in range(skip, 3):
            handle.write(_record(index))

    assert interrupted.read_bytes() == clean


def test_repair_logs_what_it_dropped(tmp_path, caplog):
    path = _write(tmp_path / "a.jsonl", _record(0), "torn")
    with caplog.at_level(logging.WARNING, logger="ruleforge.core.resume"):
        count_complete_jsonl_records(path, repair=True)
    assert "dropped 4 trailing byte(s)" in caplog.text


def test_repair_refuses_to_empty_a_file_of_foreign_content(tmp_path):
    path = _write(tmp_path / "a.jsonl", "garbage\n")
    with pytest.raises(ValueError, match="refusing to truncate"):
        count_complete_jsonl_records(path, repair=True)
    assert path.read_bytes() == b"garbage\n"


@pytest.mark.parametrize("damaged", [0, 1])
def test_repair_refuses_a_damaged_line_that_more_lines_follow(tmp_path, damaged):
    lines = [_record(0), _record(1), _record(2)]
    lines[damaged] = '{"i": "torn\n'
    path = _write(tmp_path / "a.jsonl", *lines)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=rf"line {damaged + 1} of '.*' is not a JSON record.*refusing to truncate"):
        count_complete_jsonl_records(path, repair=True)
    assert path.read_bytes() == before
    assert count_complete_jsonl_records(path) == damaged


def test_repair_truncates_a_damaged_last_line_with_its_newline(tmp_path):
    path = _write(tmp_path / "a.jsonl", _record(0), "torn\n")
    assert count_complete_jsonl_records(path, repair=True) == 1
    assert path.read_bytes() == _record(0).encode("utf-8")


def test_a_run_killed_inside_its_first_record_still_resumes(tmp_path):
    path = tmp_path / "evals.jsonl"
    path.write_bytes(b'{"i": 0, "judgm')
    assert prepare_resume_skip(path, resume=True) == 0
    assert path.read_bytes() == b""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(_record(0))
    assert path.read_bytes() == _record(0).encode("utf-8")


def test_leading_whitespace_does_not_look_foreign(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_bytes(b'  {"i": 0')
    assert count_complete_jsonl_records(path, repair=True) == 0
    assert path.read_bytes() == b""


def test_repair_still_accepts_an_empty_file(tmp_path):
    path = _write(tmp_path / "a.jsonl")
    assert count_complete_jsonl_records(path, repair=True) == 0


def test_resume_surfaces_the_refusal(tmp_path):
    path = _write(tmp_path / "a.jsonl", "garbage\n")
    with pytest.raises(ValueError, match="refusing to truncate"):
        prepare_resume_skip(path, resume=True)


def test_absent_output_skips_nothing(tmp_path):
    assert prepare_resume_skip(tmp_path / "absent.jsonl", resume=False) == 0


def test_existing_output_without_resume_is_refused(tmp_path):
    path = _write(tmp_path / "a.jsonl", _record(0))
    with pytest.raises(ValueError, match="already exists"):
        prepare_resume_skip(path, resume=False)


def test_existing_output_is_not_modified_when_refused(tmp_path):
    path = _write(tmp_path / "a.jsonl", _record(0), "torn")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        prepare_resume_skip(path, resume=False)
    assert path.read_bytes() == before


def test_resume_returns_the_complete_record_count(tmp_path):
    path = _write(tmp_path / "a.jsonl", _record(0), _record(1))
    assert prepare_resume_skip(path, resume=True) == 2


def test_resume_repairs_a_torn_tail(tmp_path):
    path = _write(tmp_path / "a.jsonl", _record(0), '{"i": 1')
    assert prepare_resume_skip(path, resume=True) == 1
    assert path.read_bytes() == _record(0).encode("utf-8")


def test_resume_logs_the_skip_with_its_label(tmp_path, caplog):
    path = _write(tmp_path / "a.jsonl", _record(0))
    with caplog.at_level(logging.INFO, logger="ruleforge.core.resume"):
        prepare_resume_skip(path, resume=True, label="evaluations")
    assert "evaluations" in caplog.text
    assert "1 complete record(s)" in caplog.text


def test_resume_on_an_empty_file_logs_nothing(tmp_path, caplog):
    path = _write(tmp_path / "a.jsonl")
    with caplog.at_level(logging.INFO, logger="ruleforge.core.resume"):
        assert prepare_resume_skip(path, resume=True) == 0
    assert caplog.text == ""
