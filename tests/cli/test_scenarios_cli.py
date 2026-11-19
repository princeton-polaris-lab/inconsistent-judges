from __future__ import annotations

import pytest

from ruleforge.cli import records
from ruleforge.core.digests import sha256_file
from ruleforge.scenarios.subset import block_name, build_blocks, build_concatenation, build_difference
from tests.support.trees import tree_files
from tests.test_sampling_subset import _carved, _sources, write_population


def files(directory):
    """A directory's files, but the command's run records beside its outputs."""
    return {str(path.relative_to(directory)): path.read_bytes() for path in tree_files(directory)
            if records.RUNS_SUFFIX + "/" not in str(path.relative_to(directory))}


def test_concat_writes_the_library_s_concatenation_and_prints_where_each_source_landed(cli, tmp_path):
    first, second = _sources(tmp_path)
    digests = [sha256_file(first), sha256_file(second)]
    (tmp_path / "s").mkdir()
    (tmp_path / "c").mkdir()
    build_concatenation([first, second], tmp_path / "s" / "pool.jsonl", expected_source_sha256s=digests)
    out = tmp_path / "c" / "pool.jsonl"
    status, printed, _ = cli("scenarios", "concat", "--source", first, "--expect-source-sha256", digests[0],
                             "--source", second, "--expect-source-sha256", digests[1], "--out", out)
    assert status == 0 and printed.splitlines() == [f"population.jsonl (sha256 {digests[0]}): rows [0, 4)",
                                                    f"second.jsonl (sha256 {digests[1]}): rows [4, 7)",
                                                    f"{out}: 7 rows, sha256 {sha256_file(out)}"]
    assert files(tmp_path / "s") == files(tmp_path / "c")
    assert next(records.runs_dir(out).iterdir()).name.endswith("-scenarios-concat")


def test_subtract_writes_the_library_s_difference_and_prints_what_each_population_removed(cli, tmp_path):
    source = write_population(tmp_path, 5)
    carved = _carved(tmp_path, source, [1, 2], "carved.jsonl")
    (tmp_path / "s").mkdir()
    (tmp_path / "c").mkdir()
    build_difference(source, tmp_path / "s" / "rest.jsonl", exclude=[carved], expected_source_sha256=sha256_file(source))
    out = tmp_path / "c" / "rest.jsonl"
    status, printed, _ = cli("scenarios", "subtract", "--source", source, "--expect-source-sha256", sha256_file(source),
                             "--exclude", carved, "--out", out)
    assert status == 0 and printed.splitlines() == [f"source: {source} (5 rows, sha256 {sha256_file(source)})",
                                                    f"less carved.jsonl (sha256 {sha256_file(carved)}): 2 of its 2 rows",
                                                    f"{out}: 3 rows, sha256 {sha256_file(out)}"]
    assert files(tmp_path / "s") == files(tmp_path / "c")


def test_partition_writes_the_library_s_blocks_and_prints_each(cli, tmp_path):
    source = write_population(tmp_path, 7)
    build_blocks(source, tmp_path / "s", block_size=3, expected_source_sha256=sha256_file(source))
    out = tmp_path / "c"
    status, printed, _ = cli("scenarios", "partition", "--source", source, "--block-size", "3",
                             "--expect-source-sha256", sha256_file(source), "--out-dir", out)
    blocks = [out / block_name(source, index) for index in range(3)]
    assert status == 0 and printed.splitlines() == [
        f"source: {source} (7 rows, sha256 {sha256_file(source)})",
        *(f"{block}: rows [{start}, {stop}) sha256 {sha256_file(block)}"
          for block, (start, stop) in zip(blocks, [(0, 3), (3, 6), (6, 7)]))]
    assert files(tmp_path / "s") == files(tmp_path / "c")


@pytest.mark.parametrize("command, arguments, message", [
    ("concat", lambda tmp_path, source: ["--source", str(source), "--out", str(tmp_path / "o.jsonl")],
     "at least two sources, got 1"),
    ("subtract", lambda tmp_path, source: ["--source", str(source), "--exclude", str(source), "--out",
                                           str(tmp_path / "o.jsonl")], "named twice, or is the source"),
    ("partition", lambda tmp_path, source: ["--source", str(source), "--block-size", "3", "--out-dir",
                                            str(tmp_path / "blocks"), "--expect-source-sha256", "0000"],
     "expected 0000"),
])
def test_a_refusal_after_the_run_record_exits_1_with_its_reason_and_writes_no_output(cli, tmp_path, command,
                                                                                     arguments, message):
    source = write_population(tmp_path, 4)
    status, _, error = cli("scenarios", command, *arguments(tmp_path, source))
    assert status == 1 and message in error
    assert not (tmp_path / "o.jsonl").exists() and not (tmp_path / "blocks").exists()


@pytest.mark.parametrize("command", ["concat", "subtract"])
def test_a_link_where_the_sidecar_goes_is_refused_before_anything_is_written(cli, tmp_path, command):
    """The ``.manifest.json`` beside the output is a place the command writes too."""
    first, second = _sources(tmp_path)
    arguments = {"concat": ["--source", str(first), "--source", str(second)],
                 "subtract": ["--source", str(first), "--exclude", str(_carved(tmp_path, first, [1], "carved.jsonl"))]}
    protected = tmp_path / "protected.json"
    protected.write_text("{}", encoding="utf-8")
    (tmp_path / "o.jsonl.manifest.json").symlink_to(protected)
    status, _, error = cli("scenarios", command, *arguments[command], "--out", tmp_path / "o.jsonl")
    assert status == 2 and "o.jsonl.manifest.json is a symlink" in error
    assert not (tmp_path / "o.jsonl").exists() and not records.runs_dir(tmp_path / "o.jsonl").exists()
    assert protected.read_text(encoding="utf-8") == "{}"
