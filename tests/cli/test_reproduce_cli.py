from __future__ import annotations

import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import ruleforge.evaluate.api  # noqa: F401 -- imported before any test patches paths.repository_root (below)
from ruleforge.cli import reproduce
from ruleforge.core import paths
from ruleforge.core.errors import ArtifactRootError
from ruleforge.experiments import table
from ruleforge.experiments.table import Experiment, Golden, Step
from tests.conftest import REPO_ROOT

ENV_PROBE = ("python -c \"import os, sys; open(sys.argv[1], 'w').write(os.environ['SOURCE_DATE_EPOCH'] + ' ' + "
             "os.environ['OMP_NUM_THREADS'])\"")
#: Writes the environment's PYTHONPATH to its one argument.
PATH_PROBE = "python -c \"import os, sys; open(sys.argv[1], 'w').write(os.environ['PYTHONPATH'])\""
#: Writes the path a producer reads an artifact:// input from to its one argument.
ROOT_PROBE = ("python -c \"import sys; from ruleforge.core import paths; "
              "open(sys.argv[1], 'w').write(str(paths.resolve('artifact://roots/estimates.json')))\"")


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A toy checkout (its table set by the test) and the artifact root the CLI tests' fixture set."""
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    (repo / "results" / "golden.json").write_text(json.dumps({"rate": 0.5}))
    (repo / "results" / "broken.json").write_text("{")
    (tmp_path / "artifacts" / "roots").mkdir(parents=True)
    (tmp_path / "artifacts" / "roots" / "estimates.json").write_text(json.dumps({"draws": 5}))
    monkeypatch.setattr(paths, "repository_root", lambda: repo)

    def write(renders=(), builds=(), paper=None):
        monkeypatch.setattr(table, "TABLE", tuple(
            Experiment(name, (paper or {}).get(name, ()), builds=tuple(step for owner, step in builds if owner == name),
                       plots=tuple(step for owner, step in renders if owner == name))
            for name in reproduce.EXPERIMENTS))
        return repo

    return write


def _render(id_, argv, experiment="ruleset-size"):
    return experiment, Step(id_, argv)


def _build(id_, argv, experiment="ruleset-size", title="", **extra):
    return experiment, Step(id_, argv, title or id_, **extra)


def test_a_render_draws_each_entry_into_out_and_records_the_run(checkout, cli, tmp_path, monkeypatch):
    repo = checkout(renders=[
        _render("table", "python -m json.tool repo://results/golden.json {out}/table.json"),
        _render("other", "python -m json.tool repo://results/golden.json {out}/nested/x.json", experiment="training"),
        _render("env", f"{ENV_PROBE} {{out}}/env.txt"),
        _render("path", f"{PATH_PROBE} {{out}}/path.txt"),
    ])
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "elsewhere" / "src"))
    out = tmp_path / "figures"
    status, printed, error = cli("reproduce", "ruleset-size", "render", "--out", out)
    assert status == 0 and f"wrote {out}" in printed
    assert f"the checkout {repo}" in error
    assert json.loads((out / "table.json").read_text()) == {"rate": 0.5}
    assert not (out / "nested").exists()
    assert (out / "env.txt").read_text() == "0 4"
    assert (out / "path.txt").read_text() == os.pathsep.join([str(repo / "src"), str(tmp_path / "elsewhere" / "src")])
    runs = tmp_path / "figures.runs"
    (run,) = [path for path in runs.iterdir() if path.is_dir()]
    assert sorted(path.name for path in runs.iterdir()) == sorted([run.name, reproduce.LOCK_FNAME])
    assert run.name.endswith("-reproduce-render") and json.loads((run / "argv.json").read_text())["argv"][:2] == [
        "reproduce", "ruleset-size"]
    assert "table: " in (run / "log.txt").read_text()


@pytest.mark.parametrize("level", ["render", "rebuild"])
def test_a_render_and_a_rebuild_first_name_each_experiment_s_figures_and_tables_in_the_paper(checkout, cli, tmp_path,
                                                                                            level):
    argv = "python -m json.tool repo://results/golden.json {out}/x.json"
    checkout(renders=[_render("x", argv)], builds=[_build("x", argv)],
             paper={"ruleset-size": ("Figure 3", "Figure 10"), "training": ("Figure 5",)})
    status, printed, _ = cli("reproduce", "ruleset-size", level, "--out", tmp_path / "out")
    assert status == 0 and printed.splitlines()[:2] == ["ruleset-size: Figure 3, Figure 10", "== x"]
    assert reproduce.paper_lines(reproduce.EXPERIMENTS) == ["ruleset-size: Figure 3, Figure 10", "training: Figure 5"]


def test_a_render_keeps_a_source_date_epoch_it_is_given_and_all_runs_every_experiment(checkout, cli, tmp_path,
                                                                                    monkeypatch):
    checkout(renders=[_render("env", f"{ENV_PROBE} {{out}}/env.txt"),
                      _render("other", "python -m json.tool repo://results/golden.json {out}/grpo.json",
                              experiment="training")])
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    out = tmp_path / "all"
    assert cli("reproduce", "all", "render", "--out", out)[0] == 0
    assert (out / "env.txt").read_text() == "1700000000 4" and (out / "grpo.json").is_file()


def test_a_render_without_matplotlib_is_refused_in_one_line_naming_what_installs_it(checkout, cli, tmp_path,
                                                                                   monkeypatch):
    checkout(renders=[_render("x", "python -m json.tool repo://results/golden.json {out}/x.json")])
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    status, _, error = cli("reproduce", "ruleset-size", "render", "--out", tmp_path / "out")
    assert status == 2 and error.splitlines()[-1] == (
        "ruleforge reproduce: MissingDependency: drawing the figures needs matplotlib, which this environment lacks: "
        "add --group viz to uv sync, or pip install -e '.[viz]' from the checkout")
    assert "Traceback" not in error and not (tmp_path / "out").exists()


def test_a_render_that_fails_stops_there_with_exit_1(checkout, cli, tmp_path):
    checkout(renders=[_render("bad", "python -m json.tool repo://results/broken.json {out}/bad.json"),
                      _render("never", "python -m json.tool repo://results/golden.json {out}/never.json")])
    status, _, error = cli("reproduce", "ruleset-size", "render", "--out", tmp_path / "out")
    assert status == 1 and "bad failed" in error and not (tmp_path / "out" / "never.json").exists()


def test_a_rebuild_writes_goldens_logs_and_copies_and_names_a_verify_entry(checkout, cli, tmp_path, monkeypatch):
    checkout(builds=[
        _build("readout", "python -m json.tool artifact://roots/estimates.json {out}/readout/family.json",
               title="the readout", copied_from="{out}/readout/family.json", golden=Golden("x", "results/x/golden.json", 0.0)),
        _build("subsets", "python -m ruleforge.experiments.ruleset_size.verify_subsets", title="subsets", check_only=True),
        _build("env", f"{ENV_PROBE} {{out}}/results/env.txt", title="env"),
    ])
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    out = tmp_path / "rebuilt"
    status, printed, _ = cli("reproduce", "ruleset-size", "rebuild", "--out", out)
    assert status == 0 and printed.splitlines()[0] == "== the readout"
    assert json.loads((out / "results" / "x" / "golden.json").read_text()) == {"draws": 5}
    assert (out / "readout" / "family.json").is_file() and (out / "logs" / "the_readout.log").is_file()
    assert ("== subsets: not rebuilt and not checked (a rebuild regenerates nothing it verifies); once regenerated, "
            "check with python -m ruleforge.experiments.ruleset_size.verify_subsets\n") in printed
    assert (out / "results" / "env.txt").read_text() == "0 4"
    assert f"rebuilt into {out}" in printed


def test_a_rebuild_that_fails_prints_its_log_s_tail_and_exits_1(checkout, cli, tmp_path):
    checkout(builds=[_build("bad", "python -m json.tool repo://results/broken.json {out}/results/bad.json",
                            title="bad build")])
    status, printed, error = cli("reproduce", "ruleset-size", "rebuild", "--out", tmp_path / "out")
    assert status == 1 and "== bad build" in printed
    assert "Expecting property name" in error and "bad build failed (its log: " in error


def _hold(path: Path) -> int:
    """Hold ``path``'s ``flock`` through a descriptor of its own, as another process would."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o666)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def test_a_second_run_into_one_directory_is_refused_while_the_first_holds_its_lock(checkout, cli, tmp_path):
    checkout(builds=[_build("one", "python -m json.tool repo://results/golden.json {out}/results/golden.json",
                            title="one")],
             renders=[_render("x", "python -m json.tool repo://results/golden.json {out}/x.json")])
    out, lock = tmp_path / "out", tmp_path / "out.runs" / reproduce.LOCK_FNAME
    descriptor = _hold(lock)
    try:
        for level in ("rebuild", "render"):
            status, _, error = cli("reproduce", "ruleset-size", level, "--out", out)
            assert status == 2 and f"another process holds {lock}: a ruleforge reproduce is writing {out}" in error
        assert not out.exists() and [path.name for path in lock.parent.iterdir()] == [lock.name]
    finally:
        os.close(descriptor)
    assert cli("reproduce", "ruleset-size", "rebuild", "--out", out)[0] == 0
    assert (out / "results" / "golden.json").is_file() and lock.is_file()


#: A producer that marks itself started (its pid, written whole) and draws once released; only its first run waits.
SLOW_PRODUCER = """\
import os, pathlib, sys, time
started, release, figure = map(pathlib.Path, sys.argv[1:4])
first = not started.exists()
started.with_name("started.tmp").write_text(str(os.getpid()))
os.replace(started.with_name("started.tmp"), started)
deadline = time.monotonic() + 300
while first and not release.exists() and time.monotonic() < deadline:
    time.sleep(0.05)
figure.write_text("drawn")
"""
#: The command run by a process of its own, reading the toy checkout ``argv[1]`` and the plot ``argv[2]`` as the
#: checkout fixture does.
DRIVER = ("import sys; from pathlib import Path; from ruleforge.core import paths; "
          "from ruleforge.experiments import table; paths.repository_root = lambda: Path(sys.argv[1]); "
          "table.TABLE = (table.Experiment('ruleset-size', plots=(table.Step('slow', sys.argv[2]),)),); "
          "from ruleforge.cli import main; sys.exit(main(sys.argv[3:]))")


def _until(condition, what: str, seconds: float = 120) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, f"no {what} after {seconds} s"
        time.sleep(0.05)


def _free(lock: Path) -> bool:
    descriptor = os.open(lock, os.O_WRONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(descriptor)


def test_a_producer_left_running_by_its_killed_run_keeps_the_directory_locked(checkout, cli, tmp_path):
    """The run's lock is held by every producer it starts, not by the command's process alone."""
    script, started, release = tmp_path / "producer.py", tmp_path / "started", tmp_path / "release"
    script.write_text(SLOW_PRODUCER)
    plot = shlex.join(["python", str(script), str(started), str(release), "{out}/figure.txt"])
    repo = checkout(renders=[_render("slow", plot)])
    out, lock, log = tmp_path / "out", tmp_path / "out.runs" / reproduce.LOCK_FNAME, tmp_path / "driver.log"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"),
                                                        *filter(None, [os.environ.get("PYTHONPATH")])])}
    with log.open("w") as handle:  # a file, not a pipe: the orphaned producer keeps what it inherited open
        driver = subprocess.Popen([sys.executable, "-c", DRIVER, str(repo), plot, "reproduce", "ruleset-size", "render",
                                   "--out", str(out)], env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=handle)
    try:
        _until(lambda: started.exists() or driver.poll() is not None, "producer started")
        assert started.exists(), log.read_text()
        driver.send_signal(signal.SIGTERM)
        assert driver.wait(timeout=60) == -signal.SIGTERM
        os.kill(int(started.read_text()), 0)  # its producer outlives it, drawing still
        status, _, error = cli("reproduce", "ruleset-size", "render", "--out", out)
        assert status == 2 and f"another process holds {lock}: a ruleforge reproduce is writing {out}" in error
        assert "(or a producer it started still is)" in error and not (out / "figure.txt").exists()
    finally:
        release.touch()
        if driver.poll() is None:
            driver.kill()
            driver.wait()
    _until(lambda: _free(lock), "lock released once the producer ended")
    assert (out / "figure.txt").read_text() == "drawn"
    assert cli("reproduce", "ruleset-size", "render", "--out", out)[0] == 0


def test_a_lock_that_is_not_a_file_of_its_own_is_refused(checkout, cli, tmp_path):
    checkout(renders=[_render("x", "python -m json.tool repo://results/golden.json {out}/x.json")])
    (tmp_path / "out.runs").mkdir()
    (tmp_path / "out.runs" / reproduce.LOCK_FNAME).symlink_to(tmp_path / "elsewhere")
    status, _, error = cli("reproduce", "ruleset-size", "render", "--out", tmp_path / "out")
    assert status == 2 and "cannot be opened as a file of its own" in error
    assert not (tmp_path / "out").exists() and not (tmp_path / "elsewhere").exists()


def test_a_log_written_during_the_rebuild_stops_it_unrun(checkout, cli, tmp_path):
    checkout(builds=[_build("first", "python -m json.tool repo://results/golden.json {out}/logs/second.log",
                            title="first"),
                     _build("second", "python -m json.tool repo://results/golden.json {out}/results/second.json",
                            title="second")])
    out = tmp_path / "out"
    status, printed, error = cli("reproduce", "ruleset-size", "rebuild", "--out", out)
    assert status == 1 and "== second" in printed and "second.log was written during this run" in error
    assert "second was not run" in error and not (out / "results" / "second.json").exists()


@pytest.mark.parametrize(("level", "place"), [("render", "results"), ("render", "results/x"), ("rebuild", "results"),
                                              ("rebuild", "results/x")])
def test_neither_level_writes_into_the_checkout_s_committed_files(checkout, cli, level, place):
    repo = checkout(renders=[_render("x", "python -m json.tool repo://results/golden.json {out}/x.json")],
                    builds=[_build("x", "python -m json.tool repo://results/golden.json {out}/results/x.json")])
    status, _, error = cli("reproduce", "ruleset-size", level, "--out", repo / place)
    assert status == 2 and f"lies in the checkout's {place.split('/')[0]}/" in error
    assert not (repo / place / "x.json").exists() and not (repo / place / "results").exists()


@pytest.mark.parametrize("spelled", ["relative", "absolute"])
def test_an_out_whose_dotdot_follows_a_symlink_is_refused_not_taken_lexically(checkout, cli, tmp_path, monkeypatch,
                                                                             spelled):
    """``--out <safe>/link/../results``, the link into the checkout's ``figures/``."""
    repo = checkout(builds=[_build("x", "python -m json.tool repo://results/golden.json {out}/results/x.json")])
    (repo / "figures").mkdir()
    safe = tmp_path / "safe"
    safe.mkdir()
    (safe / "link").symlink_to(repo / "figures", target_is_directory=True)
    monkeypatch.chdir(safe)
    given = "link/../results" if spelled == "relative" else f"{safe}/link/../results"
    before = sorted(path.name for path in (repo / "results").iterdir())
    status, _, error = cli("reproduce", "ruleset-size", "rebuild", "--out", given)
    reached = Path(os.path.realpath(repo)) / "results"
    assert status == 2 and (f"--out {given} names two places: {safe / 'results'} as written out, and {reached} where "
                            "the kernel takes it") in error
    assert sorted(path.name for path in (repo / "results").iterdir()) == before
    assert sorted(path.name for path in safe.iterdir()) == ["link"]


@pytest.mark.parametrize(("level", "written"), [("render", "x.json"), ("rebuild", "results/x.json")])
def test_either_level_writes_into_a_figures_directory_of_the_checkout(checkout, cli, level, written):
    """The checkout commits no figures/: it is a directory like any other."""
    repo = checkout(renders=[_render("x", "python -m json.tool repo://results/golden.json {out}/x.json")],
                    builds=[_build("x", "python -m json.tool repo://results/golden.json {out}/results/x.json")])
    assert cli("reproduce", "ruleset-size", level, "--out", repo / "figures")[0] == 0
    assert (repo / "figures" / written).is_file()


def test_a_rebuild_refuses_a_directory_holding_one_of_its_outputs(checkout, cli, tmp_path):
    checkout(builds=[_build("one", "python -m json.tool repo://results/golden.json {out}/results/golden.json",
                            title="one")])
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "one.log").write_text("an earlier run's\n")
    status, _, error = cli("reproduce", "ruleset-size", "rebuild", "--out", out)
    assert status == 2 and "one.log exist(s) already" in error
    assert not (out / "results").exists() and not (tmp_path / "out.runs").exists()


@pytest.mark.parametrize(("word", "message"), [
    ("artifact://roots/absent.json", "inputs not found"),
    ("repo://results/absent.json", "inputs not found"),
    ("artifact://../escape.json", "escapes the artifact root"),
])
def test_an_input_that_is_not_there_is_refused_before_anything_runs(checkout, cli, tmp_path, word, message):
    checkout(renders=[_render("first", "python -m json.tool repo://results/golden.json {out}/first.json"),
                      _render("second", f"python -m json.tool {word} {{out}}/second.json")])
    status, _, error = cli("reproduce", "ruleset-size", "render", "--out", tmp_path / "out")
    assert status == 2 and message in error and "second: " in error and not (tmp_path / "out").exists()


def test_a_declared_input_is_checked_with_the_command_line_s(checkout, cli, tmp_path):
    checkout(builds=[_build("first", "python -m json.tool repo://results/golden.json {out}/results/first.json"),
                     _build("family", "python -m json.tool repo://results/golden.json {out}/results/family.json",
                            inputs=("artifact://roots/estimates.json", "artifact://roots/absent"))])
    status, _, error = cli("reproduce", "ruleset-size", "rebuild", "--out", tmp_path / "out")
    assert status == 2 and "inputs not found" in error and "family: artifact://roots/absent" in error
    assert "estimates.json" not in error and not (tmp_path / "out").exists()


def test_an_artifact_input_without_an_artifact_root_is_refused(checkout, cli, tmp_path, monkeypatch):
    checkout(builds=[_build("one", "python -m json.tool artifact://roots/estimates.json {out}/results/x.json")])
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT")
    status, _, error = cli("reproduce", "ruleset-size", "rebuild", "--out", tmp_path / "out")
    assert status == 2 and "one: artifact://roots/estimates.json" in error and "ARTIFACT_ROOT is unset" in error


def test_render_and_rebuild_need_an_out(checkout, cli):
    checkout(renders=[_render("x", "python -m json.tool repo://results/golden.json {out}/x.json")])
    for level in ("render", "rebuild"):
        status, _, error = cli("reproduce", "ruleset-size", level)
        assert status == 2 and f"{level} writes into --out DIR" in error


def test_out_is_taken_as_every_command_takes_it(checkout, cli, tmp_path, monkeypatch):
    checkout(renders=[_render("x", "python -m json.tool repo://results/golden.json {out}/x.json")])
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    status, printed, _ = cli("reproduce", "ruleset-size", "render", "--out", "~/figures")
    assert status == 0 and f"wrote {tmp_path / 'home' / 'figures'}" in printed
    assert json.loads((tmp_path / "home" / "figures" / "x.json").read_text()) == {"rate": 0.5}
    assert not (Path.cwd() / "~").exists()


def test_an_out_spelled_with_an_interpolation_is_one_place_for_the_checks_and_the_producers(checkout, cli, tmp_path,
                                                                                          monkeypatch):
    repo = checkout(renders=[_render("x", "python -m json.tool repo://results/golden.json {out}/copy.json")])
    (repo / "${X}").mkdir()
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    monkeypatch.chdir(work)
    status, printed, _ = cli("reproduce", "ruleset-size", "render", "--out", "${X}/../results")
    assert status == 0 and f"wrote {work / 'results'}\n" in printed
    assert json.loads((work / "results" / "copy.json").read_text()) == {"rate": 0.5}
    assert not (repo / "results" / "copy.json").exists()
    monkeypatch.chdir(repo)
    status, _, error = cli("reproduce", "ruleset-size", "render", "--out", "${X}/../results")
    assert status == 2 and f"{repo / 'results'} lies in the checkout's results/" in error
    assert not (repo / "results" / "copy.json").exists()


def test_the_command_needs_a_source_checkout(cli, monkeypatch):
    def installed():
        raise ArtifactRootError("cannot resolve repo:// input outside a ruleforge source checkout")

    monkeypatch.setattr(paths, "repository_root", installed)
    status, _, error = cli("reproduce", "all", "render")
    assert status == 2 and "reads a source checkout of ruleforge" in error and "clone the repository" in error


def test_the_paper_s_entries_resolve_to_this_interpreter_and_the_checkout(tmp_path):
    import sys

    out = tmp_path / "out"
    for level in ("render", "rebuild"):
        for entry in reproduce.entries(level, reproduce.EXPERIMENTS):
            argv = reproduce.command(entry, out)
            assert argv[0] == sys.executable and argv[1] == "-m" and argv[2].startswith("ruleforge.experiments.")
            assert all(not word.startswith(("{out}", "repo://", "artifact://")) for word in argv)
            assert all(path.is_relative_to(out) for path in reproduce.outputs_of(entry, out))
            assert shlex.split(shlex.join(argv)) == argv


@pytest.mark.parametrize(("name", "expected"), [("all", reproduce.EXPERIMENTS), ("prompt_ablation", ("prompt-ablation",)),
                                                ("training", ("training",))])
def test_an_experiment_name_stands_for_its_experiments(name, expected):
    assert reproduce.experiments_of(name) == expected


@pytest.mark.parametrize("variables", [("RULEFORGE_ARTIFACT_ROOT",)])
def test_a_relative_artifact_root_is_one_place_for_the_checks_and_the_producers(checkout, cli, tmp_path, monkeypatch,
                                                                                variables):
    repo = checkout(builds=[_build("probe", f"{ROOT_PROBE} {{out}}/read.txt",
                                   inputs=("artifact://roots/estimates.json",))])
    decoy = repo / "artifacts" / "roots" / "estimates.json"
    decoy.parent.mkdir(parents=True)
    decoy.write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT")
    for variable in variables:
        monkeypatch.setenv(variable, "artifacts")
    status, _, error = cli("reproduce", "ruleset-size", "rebuild", "--out", str(tmp_path / "out"))
    assert status == 0, error
    assert (tmp_path / "out" / "read.txt").read_text() == str(tmp_path / "artifacts" / "roots" / "estimates.json")


def test_a_relative_artifact_root_naming_two_places_is_refused(checkout, cli, tmp_path, monkeypatch):
    checkout(builds=[_build("probe", f"{ROOT_PROBE} {{out}}/read.txt")])
    (tmp_path / "elsewhere" / "deep").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "elsewhere" / "deep")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", "link/../artifacts")
    status, _, error = cli("reproduce", "ruleset-size", "rebuild", "--out", str(tmp_path / "out"))
    assert status == 2 and "RULEFORGE_ARTIFACT_ROOT: link/../artifacts names two places" in error
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("words", [("all", "rerun"), ("all", "render", "--submit")])
def test_the_rerun_verb_and_submit_are_gone(cli, words):
    with pytest.raises(SystemExit) as stopped:
        cli("reproduce", *words)
    assert stopped.value.code == 2


def test_the_help_names_every_spelling_and_no_figure_number():
    from ruleforge.cli import app

    (reproducing,) = [action.choices["reproduce"] for action in app.build_parser()._actions
                      if isinstance(getattr(action, "choices", None), dict) and "reproduce" in action.choices]
    text = " ".join(reproducing.format_help().split())
    assert all(word in text for word in (*reproduce.EXPERIMENTS, *reproduce.ALIASES, "the ruleset-size rule subsets"))
    assert "Figure" not in text
