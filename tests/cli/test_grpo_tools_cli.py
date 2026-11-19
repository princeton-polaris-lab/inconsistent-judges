from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ruleforge.cli import app, grpo, mounted, records
from ruleforge.grpo.leakage import LeakageError
from tests.test_training_report_tables import _population

TOOLS = {"parity": grpo.PARITY, "report": grpo.REPORT, "tables": grpo.TABLES}
#: The names the command line's own namespace holds: no mounted step may set one.
OWN = {"command", "verb", "step", "stage", "verbose", "handler", "takes_overrides", "overrides"}


#: ``grpo report saves``'s arguments but its ``--out``.
SAVES = ["grpo", "report", "saves", "--settings", "p", "--base", "b", "--saves", "c", "--rules", "r"]
#: What only running one of these commands imports (``lora_parity`` is read for ``grpo merge``'s flags already).
LAZY = ("ruleforge.grpo.report", "ruleforge.grpo.report_tables", "ruleforge.grpo.exclusion",
        "ruleforge.judges.batch_collect", "ruleforge.evaluate.cost", "ruleforge.evaluate.bank",
        "ruleforge.optimize.select")


def test_building_the_command_line_imports_none_of_what_its_commands_run():
    probe = ("import json, sys; from ruleforge.cli import app; app.build_parser(); "
             "print(json.dumps(sorted(sys.modules)))")
    loaded = set(json.loads(subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                                           check=True).stdout))
    assert not set(LAZY) & loaded


#: A path flag's dest naming an output: every one of a step's is among what the step lists as written.
OUTPUT = re.compile(r"(^|_)(out|report)(_|$)")


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_every_step_of_a_tool_is_mounted_with_its_help_and_every_flag_it_writes(name):
    tool = TOOLS[name]
    steps = mounted.subcommands(getattr(importlib.import_module(tool.module), tool.parser)())
    assert sorted(steps) == sorted(tool.steps)
    helps = {action.dest: action.help for action in getattr(importlib.import_module(tool.module), tool.parser)()
             ._subparsers._group_actions[0]._choices_actions}
    for step, spec in tool.steps.items():
        dests = {action.dest for action in steps[step]._actions}
        assert spec.help == helps[step] and set(spec.writes) | set(spec.creates) <= dests and not dests & OWN
        assert not set(steps[step]._defaults) & OWN
        outputs = {action.dest for action in steps[step]._actions if action.type is Path and OUTPUT.search(action.dest)}
        assert outputs <= set(spec.writes), (step, outputs - set(spec.writes))


def test_a_step_s_help_lists_the_tool_s_flags_under_the_command_s_name(capsys):
    with pytest.raises(SystemExit) as exited:
        app.main(["grpo", "report", "saves", "--help"])
    out = capsys.readouterr().out
    assert exited.value.code == 0 and out.startswith("usage: ruleforge grpo report saves")
    assert "--settings SETTINGS" in out and "the report's settings" in out


def test_a_step_is_parsed_by_the_command_line_s_rules(capsys):
    """Each option spelled whole (the tool's own parser would take an abbreviation), a missing flag named."""
    with pytest.raises(SystemExit) as exited:
        app.main(["grpo", "report", "saves", "--sett", "p"])
    assert exited.value.code == 2 and "usage: ruleforge grpo report saves" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        app.main(["grpo", "parity", "compare", "--sequences", "s"])
    assert "the following arguments are required" in capsys.readouterr().err


def recorder(monkeypatch, tool, status=0):
    seen = []

    def main(argv, **keywords):
        seen.append((list(argv), keywords))
        return status

    monkeypatch.setattr(tool.load(), tool.main, main)
    return seen


@pytest.mark.parametrize("argv, tool, step, target", [
    (["grpo", "parity", "compare", "--sequences", "s", "--hf", "h", "--vllm", "v", "--report", "r.json"],
     "parity", "compare", "r.json"),
    (["grpo", "parity", "inspect", "--adapter", "a"], "parity", "inspect", None),
    ([*SAVES, "--out", "report"], "report", "saves", "report"),
    (["grpo", "report", "populations", "--settings", "s", "--dev", "d", "--out-dir", "pops"], "tables", "populations", "pops"),
    # A greedy-only run writes its greedy rows alone, and is recorded beside them.
    (["grpo", "parity", "hf", "--model-config", "m.yaml", "--greedy-prompts", "p.jsonl", "--greedy-out", "g.jsonl"],
     "parity", "hf", "g.jsonl"),
    # The sequences a vllm step samples when they are not there are written too.
    (["grpo", "parity", "vllm", "--model-config", "m.yaml", "--arm", "vllm_base", "--sequences", "seq.jsonl",
      "--out", "v.json"], "parity", "vllm", "v.json"),
])
def test_a_step_runs_the_tool_s_main_on_its_own_arguments_and_records_beside_what_it_writes(
        cli, monkeypatch, argv, tool, step, target):
    seen = recorder(monkeypatch, TOOLS[tool], status=3)
    status, _, _ = cli("-v", *argv)
    words = len(argv) - argv[::-1].index(step)
    assert status == 3 and seen == [([step, *argv[words:]], {})]
    if target is None:
        assert not list(Path.cwd().glob("*.runs"))
    else:
        [record] = list(records.runs_dir(Path.cwd() / target).iterdir())
        assert record.name.endswith("-" + "-".join(argv[:words]))
        assert json.loads((record / records.ARGV_FNAME).read_text())["argv"] == ["-v", *argv]


def campaign_root(directory: Path) -> Path:
    """A directory holding a campaign root's sampling manifest: a root nothing but its campaign writes."""
    directory.mkdir(parents=True)
    (directory / "sampling_manifest.json").write_text("{}", encoding="utf-8")
    return directory


@pytest.mark.parametrize("argv", [
    # Each output a step writes is checked, not only the first: the greedy rows, the decode report.
    ["grpo", "parity", "hf", "--model-config", "m.yaml", "--greedy-prompts", "p.jsonl", "--greedy-out",
     "ROOT/g.jsonl"],
    ["grpo", "parity", "hf", "--model-config", "m.yaml", "--sequences", "s.jsonl", "--out", "h.json",
     "--decode-sequences", "d.jsonl", "--decode-out", "ROOT/d.json"],
    # A file a step writes when it is not there: the sequences a vllm step would sample.
    ["grpo", "parity", "vllm", "--model-config", "m.yaml", "--arm", "vllm_base", "--sequences", "ROOT/seq.jsonl",
     "--out", "v.json"],
])
def test_a_step_writing_into_a_campaign_root_is_refused_before_the_tool_runs(cli, monkeypatch, argv):
    root = campaign_root(Path.cwd() / "root")
    seen = recorder(monkeypatch, grpo.PARITY if "parity" in argv else grpo.SCREEN)
    status, _, error = cli(*[item.replace("ROOT", str(root)) for item in argv])
    assert status == 2 and "inside the campaign root" in error and seen == []
    assert not list(Path.cwd().glob("*.runs")) and sorted(path.name for path in root.iterdir()) == [
        "sampling_manifest.json"]




@pytest.mark.parametrize("argv", [
    [*SAVES, "--out", "alias/../aggregate/report"],
    [*SAVES, "--out=alias/../aggregate/report"],
    ["grpo", "parity", "vllm", "--model-config", "m.yaml", "--arm", "vllm_base", "--sequences",
     "alias/../aggregate/seq.jsonl", "--out", "v.json"],
])
def test_a_step_s_output_spelled_with_a_dot_dot_after_a_symlink_is_refused_before_the_tool_runs(cli, monkeypatch,
                                                                                                argv):
    root = campaign_root(Path.cwd() / "protected" / "campaign")
    (root / "child").mkdir()
    (Path.cwd() / "alias").symlink_to(root / "child", target_is_directory=True)
    seen = recorder(monkeypatch, grpo.PARITY if "parity" in argv else grpo.REPORT)
    status, _, error = cli(*argv)
    assert status == 2 and "names two places" in error and str(root / "aggregate") in error and seen == [], error
    assert sorted(path.name for path in root.iterdir()) == ["child", "sampling_manifest.json"]
    assert sorted(path.name for path in Path.cwd().iterdir()) == ["alias", "protected"]


def test_a_step_s_output_spelled_with_a_dot_dot_naming_one_place_runs(cli, monkeypatch):
    (Path.cwd() / "real").mkdir()
    seen = recorder(monkeypatch, grpo.REPORT)
    status, _, error = cli(*SAVES, "--out", "real/../report")
    assert status == 0 and seen == [(["saves", *SAVES[3:], "--out", "real/../report"], {})], error
    assert records.runs_dir(Path.cwd() / "report").is_dir()


@pytest.mark.parametrize("argv, sidecar", [
    # A response table's identity, written in place beside it: a link there would carry the write wherever it points.
    (["grpo", "report", "read", "--scenarios", "s", "--root", "r", "--policy", "c", "--kind", "stage1",
      "--settings", "c", "--rules", "x", "--out", "stage1.jsonl"], "stage1.identity.json"),
])
def test_a_file_a_step_writes_beside_its_output_is_checked_before_the_tool_runs(cli, monkeypatch, argv, sidecar):
    protected = Path.cwd() / "protected.json"
    protected.write_text("{}", encoding="utf-8")
    (Path.cwd() / sidecar).symlink_to(protected)
    seen = recorder(monkeypatch, grpo.TABLES)
    status, _, error = cli(*argv)
    assert status == 2 and f"{sidecar} is a symlink" in error and seen == []
    assert protected.read_text(encoding="utf-8") == "{}" and not list(Path.cwd().glob("*.runs"))


def test_a_file_a_step_reads_when_it_is_there_is_not_checked(cli, monkeypatch):
    root = campaign_root(Path.cwd() / "root")
    (root / "seq.jsonl").write_text("", encoding="utf-8")
    seen = recorder(monkeypatch, grpo.PARITY)
    status, _, _ = cli("grpo", "parity", "vllm", "--model-config", "m.yaml", "--arm", "vllm_base", "--sequences",
                       root / "seq.jsonl", "--out", "v.json")
    assert status == 0 and len(seen) == 1 and records.runs_dir(Path.cwd() / "v.json").is_dir()


def test_the_words_after_a_command_are_its_arguments():
    assert mounted.arguments_after(["-v", "grpo", "parity", "hf", "--out", "parity"], ("grpo", "parity", "hf")) == [
        "--out", "parity"]
    with pytest.raises(ValueError, match="does not name the command grpo parity hf"):
        mounted.arguments_after(["grpo", "parity"], ("grpo", "parity", "hf"))


def test_a_lazy_parser_without_a_source_is_an_ordinary_one():
    parser = mounted.LazyParser(prog="x")
    parser.add_argument("--a")
    assert parser.parse_args(["--a", "1"]).a == "1" and "usage: x" in parser.format_usage()


def test_a_refused_input_ends_the_command_with_its_reason(cli, tmp_path):
    (tmp_path / "settings.yaml").write_text("{}")
    status, _, error = cli(*["grpo", "report", "saves", "--settings", tmp_path / "settings.yaml", "--base", "b", "--saves",
                             "c", "--rules", "r", "--out", tmp_path / "out"])
    assert status == 1 and "the report's settings have no" in error


def test_populations_writes_what_the_module_writes(cli, tmp_path, capsys):
    from ruleforge.grpo import report_tables as rt
    from tests.test_training_report_tables import _settings

    dev = _population(tmp_path / "dev.jsonl", range(9), sources="abc")
    nonyes = _population(tmp_path / "nonyes.jsonl", range(9, 11), sources="abc")
    (tmp_path / "settings.yaml").write_text(json.dumps(_settings(dev_scenarios=6)))
    assert rt.main(["populations", "--settings", str(tmp_path / "settings.yaml"), "--dev", str(dev), "--dev-nonyes",
                    str(nonyes), "--out-dir", str(tmp_path / "script")]) == 0
    capsys.readouterr()
    status, out, _ = cli("grpo", "report", "populations", "--settings", tmp_path / "settings.yaml", "--dev", dev,
                         "--dev-nonyes", nonyes, "--out-dir", tmp_path / "pops")
    assert status == 0 and out["stage1"]["count"] == 8
    assert (tmp_path / "pops" / "scenarios.jsonl").read_bytes() == (tmp_path / "script" / "scenarios.jsonl").read_bytes()


@pytest.fixture
def dataset(tmp_path):
    """A training file of six rows and a protected file of two others, none sharing a prompt."""
    def rows(prefix, count):
        return "".join(json.dumps({"prompt": f"{prefix} prompt {i} about topic {i}", "response": "r",
                                   "source": "wildchat"}) + "\n" for i in range(count))

    train, held = tmp_path / "train.jsonl", tmp_path / "held.jsonl"
    train.write_text(rows("train", 6), encoding="utf-8")
    held.write_text(rows("held", 2), encoding="utf-8")
    return train, held


def test_attest_writes_the_marker_the_toolkit_s_attestation_writes(cli, dataset, tmp_path):
    from ruleforge.grpo.exclusion import check_prompt_marker

    train, held = dataset
    status, out, _ = cli("grpo", "exclusion", "attest", "--dataset", train, "--protected", held)
    marker = tmp_path / "train.jsonl.disjoint.json"
    assert status == 0 and out.startswith("disjoint: 6 rows (6 prompts) against 2 protected rows")
    assert "screen: containment-only, unscreened (the second pass alone" in out
    assert out.rstrip().endswith(f"marker: {marker}") and marker.is_file()
    with pytest.raises(LeakageError, match="second pass alone"):
        check_prompt_marker(train)
    assert check_prompt_marker(train, allow_unscreened=True)["protected"][0]["rows"] == 2
    assert next(records.runs_dir(marker).iterdir()).name.endswith("-grpo-exclusion-attest")


def test_attest_refuses_a_leak(cli, dataset, tmp_path):
    train, held = dataset
    held.write_text(train.read_text(encoding="utf-8").splitlines(keepends=True)[0], encoding="utf-8")
    status, _, error = cli("grpo", "exclusion", "attest", "--dataset", train, "--protected", held)
    assert status == 1 and "is not prompt-disjoint from the protected set" in error
    assert not (tmp_path / "train.jsonl.disjoint.json").exists()


def test_a_parser_namespace_is_held_apart_from_the_command_line_s():
    parser = argparse.ArgumentParser()
    group = mounted.lazy_group(parser, "step")
    mounted.mount(group, "tables", grpo.TABLES, "x", path=("grpo", "report"))
    args = parser.parse_args(["tables", "populations", "--settings", "s", "--dev", "p", "--out-dir", "o"])
    assert args.step == "populations" and args.takes_overrides is False and args.dev == Path("p")


@pytest.mark.parametrize("joined", [True, False], ids=["--out=~/x", "--out '~/x'"])
def test_a_step_s_home_spelled_output_is_handed_on_expanded_and_checked_there(cli, monkeypatch, tmp_path, joined):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    seen = recorder(monkeypatch, TOOLS["report"])
    out = tmp_path / "home" / "report"
    status, _, error = cli(*SAVES, *(["--out=~/report"] if joined else ["--out", "~/report"]))
    handed = [f"--out={out}"] if joined else ["--out", str(out)]
    assert status == 0 and seen == [(["saves", *SAVES[3:], *handed], {})], error
    assert records.runs_dir(out).is_dir() and not (Path.cwd() / "~").exists()


def test_the_places_checked_are_read_from_the_tool_s_own_parse_of_what_it_is_handed(tmp_path, monkeypatch):
    from ruleforge.cli.common import CliError

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    saves = ["saves", *SAVES[3:]]
    with pytest.raises(CliError, match=r"~/x.json \(out\): the tool would take this '~' as given"):
        mounted.handed(mounted.absolute_outputs([*saves, "--ou", "~/x.json"], grpo.REPORT, "saves"),
                       grpo.REPORT, "saves")
    given = mounted.handed(mounted.absolute_outputs([*saves, "--out", "~/x.json"], grpo.REPORT, "saves"),
                           grpo.REPORT, "saves")
    assert given.out == home / "x.json" and not list(home.iterdir())


@pytest.mark.parametrize("decode, message", [("h.json", "are one place"), ("h.json.runs/d.json", "inside")])
def test_a_step_s_outputs_are_held_apart_before_its_tool_runs(cli, monkeypatch, decode, message):
    seen = recorder(monkeypatch, TOOLS["parity"])
    status, _, error = cli("grpo", "parity", "hf", "--model-config", "m.yaml", "--sequences", "s.jsonl", "--out",
                           "h.json", "--decode-sequences", "d.jsonl", "--decode-out", decode)
    assert status == 2 and message in error and seen == [], error
    assert not list(Path.cwd().iterdir())
