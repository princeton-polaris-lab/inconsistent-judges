from __future__ import annotations

import pytest

from ruleforge.experiments.checks import require
from ruleforge.experiments.goldens import format_interval, read_json_golden, write_csv


class Refused(ValueError):
    pass


def test_require_names_every_failed_check_once():
    require([(True, "fine")], "here", error=Refused)
    with pytest.raises(Refused, match="^here: first; second$"):
        require([(False, "first"), (True, "fine"), (False, "second")], "here", error=Refused)


def test_a_golden_that_is_not_json_is_refused_naming_the_file(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    assert read_json_golden(path, "here", error=Refused) == {"a": 1}
    path.write_text("{", encoding="utf-8")
    with pytest.raises(Refused, match=r"^here: .*golden\.json cannot be read as JSON \(JSONDecodeError: "):
        read_json_golden(path, "here", error=Refused)


def test_a_csv_spells_booleans_in_lower_case_with_lf_line_ends(tmp_path):
    path = tmp_path / "rows.csv"
    write_csv([{"a": True, "b": 1.5}, {"a": False, "b": "x"}], ("a", "b"), path)
    assert path.read_bytes() == b"a,b\ntrue,1.5\nfalse,x\n"


def test_an_interval_prints_four_places_and_an_absent_one_prints_undefined():
    entry = {"estimate": 0.5, "ci_low": 0.25, "ci_high": 0.75}
    assert format_interval(entry) == "0.5000 [0.2500, 0.7500]"
    assert format_interval(entry, signed=True) == "+0.5000 [+0.2500, +0.7500]"
    assert format_interval(None) == format_interval({"estimate": None}) == "undefined"
