from __future__ import annotations

import json
import re

import pytest

from tests.conftest import REPO_ROOT

TABLE = REPO_ROOT / "src/ruleforge/core/record_names.json"
#: The documents that carry the note, and those that point to it.
NOTED = ("data/README.md",)
POINTING = ("README.md", "data/hub_cards/dataset.md", "docs/development.md", "docs/extending.md", "docs/measure.md",
            "docs/reproduce.md", "docs/rerun.md", "docs/ruleforge.md", "docs/training.md", "docs/training_report.md")
NUMBERS = ("no", "one", "two", "three", "four", "five", "six")
REFILED_CLAIMS = (re.compile(r"every condition of the published[^.]*records"), re.compile(r"filed under the one label"))
OPEN, CLOSE = "<!-- names-note -->", "<!-- /names-note -->"
HEADER = "| Recorded key | Public name |"
RECORDED_ONLY = " (recorded only)"
ROW = re.compile(r"^\| `(?P<recorded>[^`]+)` \| `(?P<public>[^`]+)`(?P<only> \(recorded only\))? \|$")
CODE = re.compile(r"`([^`\n]+)`")


def _table() -> dict:
    return json.loads(TABLE.read_text(encoding="utf-8"))


def _split(text: str) -> tuple[str, str]:
    assert text.count(OPEN) == 1 and text.count(CLOSE) == 1, "the note must be there once, between its markers"
    head, rest = text.split(OPEN)
    note, tail = rest.split(CLOSE)
    return note, head + tail


def _rows(note: str) -> dict[str, tuple[str, bool]]:
    """The note's table: each recorded key -> (its public name, whether it is recorded only)."""
    lines = note.splitlines()
    start = lines.index(HEADER)
    assert lines[start + 1] == "|---|---|"
    rows = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        match = ROW.match(line)
        assert match, f"not a row of the note's table: {line!r}"
        assert match["recorded"] not in rows, f"{match['recorded']} is in the table twice"
        rows[match["recorded"]] = (match["public"], bool(match["only"]))
    return rows


def _keys(table: dict) -> tuple[set[str], set[str]]:
    keys = {row["recorded"] for row in table["styles"]} | set(table["retired"])
    return {key for key in keys if "_" in key}, {key for key in keys if "_" not in key}


def _old_names(text: str, table: dict) -> list[str]:
    words, single = _keys(table)
    hits = [key for key in sorted(words) if re.search(rf"(?<![A-Za-z0-9]){re.escape(key)}", text)]
    for code in CODE.findall(text):
        hits += [segment for segment in re.split(r"[/\s,=]", code) if segment in single]
    return hits


@pytest.mark.parametrize("path", NOTED)
def test_the_note_s_table_is_the_recorded_names_table(path):
    note, _ = _split((REPO_ROOT / path).read_text(encoding="utf-8"))
    table = _table()
    expected = {row["recorded"]: (row["public"], row["record_only"]) for row in table["styles"]}
    assert _rows(note) == expected
    assert "`src/ruleforge/core/record_names.json`" in note
    assert all(f"`{key}`" in note for key in table["retired"]), "the note names the retired keys"


@pytest.mark.parametrize("path", NOTED + POINTING)
def test_no_document_spells_a_recorded_key_outside_the_note(path):
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    rest = _split(text)[1] if path in NOTED else text
    assert OPEN not in rest and _old_names(rest, _table()) == []


def test_the_data_readme_names_the_directories_the_published_roots_leave_out():
    from ruleforge.hub import load_inventory

    card = " ".join((REPO_ROOT / "data/hub_cards/dataset.md").read_text(encoding="utf-8").split())
    assert OPEN not in card and "`data/README.md` in the code repository maps them" in card
    assert "leave out condition directories no result reads" in card
    assert "`data/README.md` in the code repository lists them" in card
    text = (REPO_ROOT / "data/README.md").read_text(encoding="utf-8")
    inventory = load_inventory(REPO_ROOT / "data" / "hub_inventory.yaml")
    omitted = {root: globs for experiment in inventory["experiments"].values()
               for root, globs in experiment.get("omit", {}).items()}
    assert omitted and all(f"`{root}`" in text for root, globs in omitted.items()
                           if any(glob.endswith("/**") for glob in globs))
    assert all(f"`{glob.removesuffix('/**')}`" in text for globs in omitted.values() for glob in globs)
    # the card names no condition directory a root leaves out; data/README.md names them, and every file glob
    assert not [glob for globs in omitted.values() for glob in globs
                if glob.endswith("/**") and glob.removesuffix("/**") in card]


def test_the_documents_say_which_published_roots_record_an_earlier_label_reading():
    from ruleforge.evaluate.campaign import condition_of
    from ruleforge.evaluate.identity import LABEL_RULE
    from ruleforge.hub import load_inventory
    from ruleforge.tasks.base import FORCED_JSON
    from tests.identity import published

    inventory = load_inventory(REPO_ROOT / "data" / "hub_inventory.yaml")
    shipped = {root for experiment in inventory["experiments"].values() for root in experiment.get("roots", {})}
    conditions = [row for row in published.rows_of(published.read_fixture("conditions.jsonl.gz"), "condition")
                  if row["root"] in shipped]
    assert {row["root"] for row in conditions} == shipped
    earlier = {row["root"] for row in conditions if row["payload"]["label_rule"] != LABEL_RULE}
    grpo = set(inventory["experiments"]["training"]["roots"])
    hierarchy = earlier - grpo
    assert grpo <= earlier and len(hierarchy) == 1 and hierarchy <= set(inventory["experiments"]["hierarchy"]["roots"])
    assert {condition_of(row["payload"], row["path"]).contract for row in conditions if row["root"] in grpo} == {
        FORCED_JSON}
    [root] = hierarchy
    [recorded] = {row["payload"]["label_rule"] for row in conditions if row["root"] == root}
    exception = f"The {NUMBERS[len(grpo)]} training roots and the two training anchor conditions are the exception"
    named = f"`{root}`, is published as recorded too, under `{recorded}`"
    for path in NOTED:
        note = " ".join(_split((REPO_ROOT / path).read_text(encoding="utf-8"))[0].split())
        assert exception in note, f"{path}'s note does not name the roots under an earlier reading"
        assert named in note, f"{path}'s note does not name {root} under its recorded reader"
    for path in NOTED + POINTING:
        text = " ".join((REPO_ROOT / path).read_text(encoding="utf-8").split())
        assert [claim.pattern for claim in REFILED_CLAIMS if claim.search(text)] == [], path
        assert [root for root in sorted(earlier) if f"{root}-rescored" in text] == [], path


def test_the_old_names_check_finds_what_it_is_for():
    table = _table()
    words, single = _keys(table)
    word, one = sorted(words)[0], sorted(single)[0]
    assert _old_names(f"a `{word}*` directory", table) == [word]
    assert _old_names(f"`judge/{one}` and `judge/{one}/draw-000`", table) == [one, one]
    assert _old_names(f"`--seeds-{one}` and a {one} prompt", table) == []
