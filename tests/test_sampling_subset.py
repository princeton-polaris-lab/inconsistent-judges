from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ruleforge.cli import app
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.runner import CampaignRunError
from ruleforge.scenarios.subset import (
    SELECTION_CONCATENATION,
    SELECTION_DIFFERENCE,
    SELECTION_SLICE,
    SUBSET_SCHEMA_VERSION,
    SubsetError,
    block_name,
    build_blocks,
    build_concatenation,
    build_difference,
    check_slices,
    read_slice_sidecar,
    sidecar_path,
)
from ruleforge.core.artifacts import compute_scenario_id, load_jsonl_records
from ruleforge.scenarios.manifest import ScenarioManifest
from tests.support.hydra_runs import run_program
from tests.test_sampling_runner import N_SCENARIOS, RULES, write_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]
def write_population(tmp_path: Path, n: int = 20) -> Path:
    """``n`` rows."""
    path = tmp_path / "population.jsonl"
    rows = [json.dumps({"prompt": f"p{i}", "response": f"r{i}"}) for i in range(n)]
    path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")
    return path


def write_sourced_population(tmp_path: Path, sizes: dict) -> Path:
    """Rows carrying a ``source`` value, interleaved so no stratum is contiguous."""
    path = tmp_path / "sourced.jsonl"
    labels = [label for label, size in sizes.items() for _ in range(size)]
    labels = labels[::2] + labels[1::2]
    rows = [json.dumps({"prompt": f"p{i}", "response": f"r{i}", "source": label}) for i, label in enumerate(labels)]
    path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")
    return path


def test_the_preflight_script_reports_the_condition(tmp_path, capsys, monkeypatch):
    scenarios = write_inputs(tmp_path)
    rules = tmp_path / "rules.jsonl"
    rules.write_text("".join(json.dumps(r) + "\n" for r in RULES), encoding="utf-8")
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    script = "ruleforge.launch.programs.preflight_campaign"
    overrides = [
        "campaign_id=camp", f"campaign_root={tmp_path / 'root'}",
        f"scenario_set_path={scenarios}", f"rule_set_path={rules}",
        "sampling.campaign_mode=pilot", "sampling.num_decoder_draws=2",
        "eval_cfg.reasoning_prompt_style=forced-grpo",
        f"hydra.run.dir={tmp_path / 'hydra'}",
    ]
    approved = [*overrides, f"approved_scenario_sha256={sha256_file(scenarios)}"]
    bad = [*overrides, "approved_scenario_sha256=" + "0" * 64]

    monkeypatch.setattr(sys, "argv", [str(script), *approved])
    run_program(script)
    report = json.loads(capsys.readouterr().out)
    non_meta = sum(1 for rule in RULES if not str(rule["id"]).startswith("Meta"))
    assert report["scenarios"] == N_SCENARIOS
    assert report["requests_per_draw"] == N_SCENARIOS * (1 + non_meta)
    assert report["non_meta_rules"] == non_meta
    assert report["rules"] == len(RULES)
    assert report["prompt_style"] == "forced-grpo"
    assert report["condition_id"].endswith("/forced-grpo")
    assert report["num_decoder_draws"] == 2
    assert report["campaign_mode"] == "pilot"
    assert report["scenario_source_sha256"] == sha256_file(scenarios)
    assert report["forced_json_requirement"] == "complete_rule_map"
    assert not (tmp_path / "root").exists()

    monkeypatch.setattr(sys, "argv", [str(script), *bad])
    with pytest.raises((SystemExit, CampaignRunError)) as exit_info:
        run_program(script)
    if isinstance(exit_info.value, SystemExit):
        assert exit_info.value.code == 1


def test_blocks_tile_the_population_byte_for_byte_with_their_sidecars(tmp_path):
    source = write_sourced_population(tmp_path, {"stress": 5, "wildchat": 5})
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    ids = [compute_scenario_id(record) for record in load_jsonl_records(source)]
    payloads = build_blocks(source, tmp_path / "blocks", block_size=4, expected_source_sha256=sha256_file(source))
    assert [payload["row_range"] for payload in payloads] == [[0, 4], [4, 8], [8, 10]]
    names = [block_name(source, index) for index in range(3)]
    assert names == ["sourced.block-000.jsonl", "sourced.block-001.jsonl", "sourced.block-002.jsonl"]
    assert "".join((tmp_path / "blocks" / name).read_text(encoding="utf-8") for name in names) == "".join(lines)
    last = json.loads(sidecar_path(tmp_path / "blocks" / names[2]).read_text(encoding="utf-8"))
    assert last == payloads[2] == {
        "schema_version": SUBSET_SCHEMA_VERSION, "selection": SELECTION_SLICE, "source_name": "sourced.jsonl",
        "source_sha256": sha256_file(source), "population_count": 10, "count": 2, "row_range": [8, 10],
        "block": {"index": 2, "of": 3, "size": 4}, "scenario_ids": ids[8:], "output_name": names[2],
        "output_sha256": sha256_file(tmp_path / "blocks" / names[2]),
    }
    assert read_slice_sidecar(sidecar_path(tmp_path / "blocks" / names[0])) == payloads[0]
    assert check_slices(source, list(reversed(payloads))) == ids
    assert build_blocks(source, tmp_path / "blocks", block_size=4) == payloads
    with pytest.raises(SubsetError, match="exists with different content"):
        build_blocks(source, tmp_path / "blocks", block_size=5)
    assert (tmp_path / "blocks" / names[0]).read_text(encoding="utf-8") == "".join(lines[:4])


def test_blocks_refuse_a_bad_request(tmp_path):
    source = write_population(tmp_path, 3)
    with pytest.raises(SubsetError, match="does not exist"):
        build_blocks(tmp_path / "missing.jsonl", tmp_path / "blocks", block_size=2)
    for size in (0, True, 2.0):
        with pytest.raises(SubsetError, match="block_size must be a positive integer"):
            build_blocks(source, tmp_path / "blocks", block_size=size)
    with pytest.raises(SubsetError, match="expected 0000"):
        build_blocks(source, tmp_path / "blocks", block_size=2, expected_source_sha256="0000")
    (tmp_path / "empty.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(SubsetError, match="empty.jsonl holds no scenarios"):
        build_blocks(tmp_path / "empty.jsonl", tmp_path / "blocks", block_size=2)
    assert not (tmp_path / "blocks").exists()


def test_a_population_that_repeats_a_scenario_is_refused_before_it_is_cut(tmp_path):
    source = write_population(tmp_path, 4)
    payloads = build_blocks(source, tmp_path / "blocks", block_size=2)
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    # Row 0 again at row 2: each block alone is duplicate-free, the population is not.
    source.write_text("".join(lines[:2] + lines[:1] + lines[3:]), encoding="utf-8")
    with pytest.raises(SubsetError, match="holds 1 duplicate scenario id"):
        build_blocks(source, tmp_path / "other", block_size=2)
    assert not (tmp_path / "other").exists()
    rebuilt = [{**payload, "source_sha256": sha256_file(source)} for payload in payloads]
    with pytest.raises(SubsetError, match="holds 1 duplicate scenario id"):
        check_slices(source, rebuilt)


def test_an_existing_output_is_compared_byte_for_byte(tmp_path):
    source = write_population(tmp_path, 4)
    build_blocks(source, tmp_path / "blocks", block_size=2)
    block = tmp_path / "blocks" / block_name(source, 0)
    # The same rows with CRLF endings read back as the same text, but they are other bytes a campaign may have frozen.
    block.write_bytes(block.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(SubsetError, match="refusing to overwrite a block"):
        build_blocks(source, tmp_path / "blocks", block_size=2)


def test_slices_that_do_not_tile_the_population_are_refused(tmp_path):
    source = write_population(tmp_path, 6)
    payloads = build_blocks(source, tmp_path / "blocks", block_size=2)
    first, second, third = payloads
    with pytest.raises(SubsetError, match=r"rows \[2, 4\) are in no block"):
        check_slices(source, [first, third])
    with pytest.raises(SubsetError, match=r"rows \[4, 6\) are in no block"):
        check_slices(source, [first, second])
    with pytest.raises(SubsetError, match="overlaps rows"):
        check_slices(source, [first, second, {**second, "output_name": "copy"}, third])
    with pytest.raises(SubsetError, match="names another source digest"):
        check_slices(source, [{**first, "source_sha256": "0" * 64}, second, third])
    with pytest.raises(SubsetError, match="are not rows of"):
        check_slices(source, [first, second, {**third, "row_range": [4, 7]}])
    with pytest.raises(SubsetError, match="scenario ids are not rows"):
        check_slices(source, [first, second, {**third, "scenario_ids": second["scenario_ids"]}])
    with pytest.raises(SubsetError, match="digest is not that of rows"):
        check_slices(source, [first, second, {**third, "output_sha256": second["output_sha256"]}])


def test_a_sidecar_that_does_not_describe_a_block_is_refused(tmp_path):
    joined = build_concatenation(_sources(tmp_path), tmp_path / "joined.jsonl")
    with pytest.raises(SubsetError, match="is not the sidecar of a block"):
        read_slice_sidecar(sidecar_path(tmp_path / "joined.jsonl"))
    assert joined["selection"] != SELECTION_SLICE
    with pytest.raises(SubsetError, match="cannot read the block sidecar"):
        read_slice_sidecar(tmp_path / "missing.manifest.json")
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    with pytest.raises(SubsetError, match="is not the sidecar of a block"):
        read_slice_sidecar(tmp_path / "list.json")


def test_the_partition_command_prints_each_block(tmp_path, capsys):
    source = write_population(tmp_path, 5)
    assert app.main(["scenarios", "partition", "--source", str(source), "--block-size", "2", "--out-dir",
                     str(tmp_path / "blocks"), "--expect-source-sha256", sha256_file(source)]) == 0
    out = capsys.readouterr().out
    assert f"source: {source} (5 rows, sha256 {sha256_file(source)})" in out
    last = tmp_path / "blocks" / block_name(source, 2)
    assert f"{last}: rows [4, 5) sha256 {sha256_file(last)}" in out


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "a").mkdir()
    first = write_population(tmp_path / "a", 4)
    second = tmp_path / "b" / "second.jsonl"
    second.parent.mkdir()
    second.write_text("".join(json.dumps({"prompt": f"q{i}", "response": f"s{i}"}) + "\n" for i in range(3))
                      + "\n", encoding="utf-8")
    return first, second


def test_a_concatenation_keeps_each_sources_rows_and_blocks_byte_for_byte(tmp_path):
    first, second = _sources(tmp_path)
    out = tmp_path / "pool.jsonl"
    payload = build_concatenation([first, second], out,
                                  expected_source_sha256s=[sha256_file(first), sha256_file(second)])
    assert out.read_text(encoding="utf-8") == first.read_text(encoding="utf-8") + second.read_text(
        encoding="utf-8").rstrip("\n") + "\n"
    assert payload == json.loads(sidecar_path(out).read_text(encoding="utf-8")) == {
        "schema_version": SUBSET_SCHEMA_VERSION, "selection": SELECTION_CONCATENATION,
        "sources": [{"name": "population.jsonl", "sha256": sha256_file(first), "count": 4, "row_range": [0, 4]},
                    {"name": "second.jsonl", "sha256": sha256_file(second), "count": 3, "row_range": [4, 7]}],
        "population_count": 7, "output_name": "pool.jsonl", "output_sha256": sha256_file(out),
    }
    build_blocks(first, tmp_path / "first-blocks", block_size=2)
    build_blocks(out, tmp_path / "pool-blocks", block_size=2)
    for index in range(2):
        assert sha256_file(tmp_path / "pool-blocks" / block_name(out, index)) == sha256_file(
            tmp_path / "first-blocks" / block_name(first, index))
    assert build_concatenation([first, second], out) == payload
    with pytest.raises(SubsetError, match="refusing to overwrite a concatenated population"):
        build_concatenation([second, first], out)


def test_a_concatenation_refuses_what_it_cannot_write_exactly(tmp_path):
    first, second = _sources(tmp_path)
    out = tmp_path / "pool.jsonl"
    with pytest.raises(SubsetError, match="at least two sources, got 1"):
        build_concatenation([first], out)
    with pytest.raises(SubsetError, match="a source is named twice"):
        build_concatenation([first, tmp_path / "a" / ".." / "a" / "population.jsonl"], out)
    with pytest.raises(SubsetError, match="1 expected digest"):
        build_concatenation([first, second], out, expected_source_sha256s=[sha256_file(first)])
    with pytest.raises(SubsetError, match="expected 0000"):
        build_concatenation([first, second], out, expected_source_sha256s=[None, "0000"])
    overlap = tmp_path / "overlap.jsonl"
    overlap.write_text(first.read_text(encoding="utf-8").splitlines(keepends=True)[1], encoding="utf-8")
    with pytest.raises(SubsetError, match="holds 1 duplicate scenario id"):
        build_concatenation([first, second, overlap], out)
    assert not out.exists()


def test_the_concatenation_command_prints_each_sources_rows(tmp_path, capsys):
    first, second = _sources(tmp_path)
    out = tmp_path / "pool.jsonl"
    assert app.main(["scenarios", "concat", "--source", str(first), "--expect-source-sha256", sha256_file(first),
                     "--source", str(second), "--expect-source-sha256", sha256_file(second), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert f"second.jsonl (sha256 {sha256_file(second)}): rows [4, 7)" in printed
    assert f"{out}: 7 rows, sha256 {sha256_file(out)}" in printed


def _carved(tmp_path: Path, source: Path, positions, name: str) -> Path:
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    path = tmp_path / name
    path.write_text("".join(lines[position] for position in positions), encoding="utf-8")
    return path


def test_a_difference_keeps_the_rows_no_carved_population_holds_verbatim_in_order(tmp_path):
    source = write_population(tmp_path, 8)
    first = _carved(tmp_path, source, [6, 1, 3], "anchored.jsonl")
    second = _carved(tmp_path, source, [3, 7], "screened.jsonl")
    out = tmp_path / "unanchored.jsonl"
    payload = build_difference(source, out, exclude=[first, second], expected_source_sha256=sha256_file(source))
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    assert out.read_text(encoding="utf-8") == "".join(lines[i] for i in (0, 2, 4, 5))
    ids = [compute_scenario_id(json.loads(line)) for line in lines]
    assert payload == json.loads(sidecar_path(out).read_text(encoding="utf-8")) == {
        "schema_version": SUBSET_SCHEMA_VERSION, "selection": SELECTION_DIFFERENCE, "source_name": "population.jsonl",
        "source_sha256": sha256_file(source), "population_count": 8,
        # a row two carved populations hold (row 3) counts for the first
        "excluded": [{"name": "anchored.jsonl", "sha256": sha256_file(first), "count": 3, "removed": 3},
                     {"name": "screened.jsonl", "sha256": sha256_file(second), "count": 2, "removed": 1}],
        "count": 4, "indices": [0, 2, 4, 5], "scenario_ids": [ids[i] for i in (0, 2, 4, 5)],
        "output_name": "unanchored.jsonl", "output_sha256": sha256_file(out)}
    assert build_difference(source, out, exclude=[first, second]) == payload
    with pytest.raises(SubsetError, match="refusing to overwrite a population difference"):
        build_difference(source, out, exclude=[first])
    with pytest.raises(SubsetError, match="is not the sidecar of a block"):
        read_slice_sidecar(sidecar_path(out))


@pytest.mark.parametrize("field", ["prompt", "response"])
def test_a_scenario_holding_the_id_separator_is_refused(tmp_path, field):
    """Two pairs split differently around U+001E would share one scenario id, and a difference would drop a stranger."""
    a, b = {"prompt": "a\x1eb", "response": "c"}, {"prompt": "a", "response": "b\x1ec"}
    assert compute_scenario_id(a) == compute_scenario_id(b)
    row = {"prompt": "keep", "response": "this", field: "x\x1ey"}
    source, excluded = tmp_path / "source.jsonl", tmp_path / "excluded.jsonl"
    source.write_text(json.dumps(a) + "\n" + json.dumps({"prompt": "keep", "response": "this"}) + "\n")
    excluded.write_text(json.dumps(b) + "\n")
    with pytest.raises(ValueError, match=r"source.jsonl row 1: its prompt holds U\+001E"):
        build_difference(source, tmp_path / "out.jsonl", exclude=[excluded])
    assert not (tmp_path / "out.jsonl").exists()
    population = tmp_path / "population.jsonl"
    population.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=rf"population.jsonl row 2: its {field} holds U\+001E"):
        ScenarioManifest.build(population)


def test_a_difference_refuses_what_is_not_carved_from_its_source(tmp_path):
    source = write_population(tmp_path, 4)
    out = tmp_path / "out.jsonl"
    carved = _carved(tmp_path, source, [0], "carved.jsonl")
    with pytest.raises(SubsetError, match="at least one population to exclude"):
        build_difference(source, out, exclude=[])
    (tmp_path / "sub").mkdir()
    with pytest.raises(SubsetError, match="named twice, or is the source"):
        build_difference(source, out, exclude=[carved, tmp_path / "sub" / ".." / "carved.jsonl"])
    with pytest.raises(SubsetError, match="named twice, or is the source"):
        build_difference(source, out, exclude=[source])
    with pytest.raises(SubsetError, match="does not exist"):
        build_difference(source, out, exclude=[tmp_path / "missing.jsonl"])
    # an empty excluded population is carved from nothing: subtracting it would silently keep every row
    for name, text in (("empty.jsonl", ""), ("blank.jsonl", "\n   \n\n")):
        (tmp_path / name).write_text(text, encoding="utf-8")
        with pytest.raises(SubsetError, match=f"{name} holds no rows: nothing carved from population.jsonl"):
            build_difference(source, out, exclude=[tmp_path / name])
    with pytest.raises(SubsetError, match="repeated.jsonl holds 1 duplicate scenario id"):
        build_difference(source, out, exclude=[_carved(tmp_path, source, [1, 1, 2], "repeated.jsonl")])
    stranger = tmp_path / "stranger.jsonl"
    stranger.write_text(json.dumps({"prompt": "elsewhere", "response": "r"}) + "\n", encoding="utf-8")
    with pytest.raises(SubsetError, match="holds 1 row\\(s\\) population.jsonl does not"):
        build_difference(source, out, exclude=[stranger])
    with pytest.raises(SubsetError, match="hold every row of population.jsonl: nothing is left"):
        build_difference(source, out, exclude=[_carved(tmp_path, source, [0, 1, 2, 3], "all.jsonl")])
    with pytest.raises(SubsetError, match="expected 0000"):
        build_difference(source, out, exclude=[carved], expected_source_sha256="0000")
    assert not out.exists()


def test_the_subtraction_command_prints_what_each_population_removed(tmp_path, capsys):
    source = write_population(tmp_path, 5)
    carved = _carved(tmp_path, source, [1, 2], "carved.jsonl")
    out = tmp_path / "rest.jsonl"
    assert app.main(["scenarios", "subtract", "--source", str(source), "--expect-source-sha256", sha256_file(source),
                     "--exclude", str(carved), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert f"source: {source} (5 rows, sha256 {sha256_file(source)})" in printed
    assert f"less carved.jsonl (sha256 {sha256_file(carved)}): 2 of its 2 rows" in printed
    assert f"{out}: 3 rows, sha256 {sha256_file(out)}" in printed
