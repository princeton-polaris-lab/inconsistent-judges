from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

import ruleforge.optimize.api as optimize_api
from ruleforge.cli import optimize as optimize_cli
from ruleforge.cli import records
from ruleforge.cli.common import FAKE_NOTE
from ruleforge.optimize import OptimizeConfig, OptimizeRun
from ruleforge.judges import batch_api


@pytest.fixture
def calls(monkeypatch, tmp_path):
    """``optimize`` replaced by a recorder that writes the run's ruleset where the optimizer would."""
    seen = []

    def optimize(rules, **kwargs):
        seen.append({"rules": rules, **kwargs})
        out = Path(kwargs["out"])
        (out / "episode_0").mkdir(parents=True, exist_ok=True)
        if kwargs.get("logger") is not None:
            kwargs["logger"].log({"reward": 1.0}, step=0)
        return OptimizeRun(out=out, last_episode=0, stopped_by=("max_episodes",), ruleset_path=Path(rules),
                           reward={"effective_reward": 1.0})

    monkeypatch.setattr(optimize_api, "optimize", optimize)
    return seen


def test_an_unconfirmed_judge_that_bills_each_call_leaves_no_run_and_no_record(cli, monkeypatch):
    from ruleforge.judges import Capabilities, JudgeSpec
    from tests.support.scripted import RuleForgeScript, scripted_backend

    requests: list = []
    billed = Capabilities(seed="local_deterministic", per_request_errors=True, spends_money=True)
    free = Capabilities(seed="local_deterministic", per_request_errors=True)

    def capabilities(self):
        return billed if self.options.get("max_new_tokens") == 200 else free

    def start(self):
        backend = scripted_backend(self.options.config(), RuleForgeScript(), requests=requests)
        backend.capabilities = capabilities(self)
        return backend

    monkeypatch.setattr(JudgeSpec, "capabilities", capabilities)
    monkeypatch.setattr(JudgeSpec, "start", start)
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "optimize"
    given = ("optimize", "--rules", fixtures / "rules.jsonl", "--judge", "fake", "--agent", "fake:2", "--out", "runs",
             "--seeds-good", fixtures / "good_seeds.jsonl", "--seeds-bad", fixtures / "bad_seeds.jsonl",
             "--episodes", "0", "--sweep", "agent.max_new_tokens=100,200")
    status, _, error = cli(*given)
    assert status == 2 and "pass --yes-spend" in error and requests == [] and not Path("runs").exists()
    status, _, error = cli(*given, "--yes-spend")
    assert status == 0 and requests and sorted(path.name for path in Path("runs").iterdir()) == [
        "agent.max_new_tokens.100", "agent.max_new_tokens.100.runs", "agent.max_new_tokens.200",
        "agent.max_new_tokens.200.runs"], error


def test_optimize_runs_the_api_on_the_packaged_defaults(cli, inputs, calls):
    rules = inputs[1]
    status, out, error = cli("optimize", "--rules", rules, "--judge", "fake", "--seeds-good", "g.jsonl", "--seeds-bad",
                             "b.jsonl", "--out", "runs/opt")
    assert status == 0 and out["stopped_by"] == ["max_episodes"] and out["reward"] == {"effective_reward": 1.0}
    assert error.splitlines()[-1] == FAKE_NOTE
    call = calls[0]
    config = call["config"]
    assert config == OptimizeConfig(num_tries=config.num_tries, max_retry_seconds=config.max_retry_seconds)
    assert call["rules"] == str(rules) and call["task"] == "modelspec" and call["agent"] is None
    assert call["seeds"] == {"good": Path.cwd() / "g.jsonl", "bad": Path.cwd() / "b.jsonl"}
    assert call["out"] == Path.cwd() / "runs" / "opt" and call["logger"] is None
    assert call["judge"].model_name == "fake-1" and config.num_tries == 5
    assert next(records.runs_dir(call["out"]).iterdir()).name.endswith("-optimize")


def test_its_settings_and_its_agent_come_from_flags_and_overrides(cli, inputs, calls):
    status, _, _ = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--agent", "fake:2", "--task", "legal",
                       "--seeds", "seed.jsonl", "--out", "o", "--episodes", "3", "--", "rule_amendment.best_of_n=2",
                       "scenario_generation.trigger_threshold=0.25", "eval_cfg.judge_types=[Narrow]")
    config = calls[0]["config"]
    assert (config.max_episodes, config.rule_amendment.best_of_n, config.scenario_generation.trigger_threshold,
            config.judge_types) == (3, 2, 0.25, ("Narrow",))
    assert calls[0]["agent"].model_name == "fake-2" and calls[0]["task"] == "legal"
    assert calls[0]["seeds"] == {"single": Path.cwd() / "seed.jsonl"}


def test_a_sweep_runs_one_optimizer_per_member_under_out(cli, inputs, calls):
    status, _, _ = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--agent", "fake:2", "--out", "runs",
                       "--sweep", "agent.max_new_tokens=100,200")
    assert status == 0
    assert [call["out"] for call in calls] == [Path.cwd() / "runs" / "agent.max_new_tokens.100",
                                               Path.cwd() / "runs" / "agent.max_new_tokens.200"]
    assert [call["agent"].options["max_new_tokens"] for call in calls] == [100, 200]
    status, _, _ = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", "t/{judge.max_new_tokens}",
                       "--sweep", "judge.max_new_tokens=3")
    assert status == 0 and calls[-1]["out"] == Path.cwd() / "t" / "3"


def test_a_sweep_of_the_agents_needs_agents_of_their_own(cli, inputs, calls):
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", "o", "--sweep", "agent.seed=1")
    assert status == 2 and "sweep judge.<key>" in error and not calls


def test_a_sweep_of_agents_a_config_file_selects_runs(cli, inputs, calls, tmp_path):
    """Whether the run has agents of their own is its composed config's to say: a --config file may select them."""
    conf = tmp_path / "conf"
    (conf / "model").mkdir(parents=True)
    (conf / "model" / "agents.yaml").write_text("model_type: fake\nmodel_name: fake-2\nseed: 2\n")
    (conf / "opt.yaml").write_text("defaults:\n  - override /model@agent_model: agents\n")
    status, _, error = cli("optimize", "--config", conf / "opt.yaml", "--rules", inputs[1], "--judge", "fake",
                           "--out", "runs", "--sweep", "agent.max_new_tokens=100,200")
    assert status == 0, error
    assert [(call["agent"].model_name, call["agent"].options["max_new_tokens"]) for call in calls] == [
        ("fake-2", 100), ("fake-2", 200)]


def test_two_members_of_one_directory_are_refused(cli, inputs, calls):
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", "same", "--sweep",
                           "judge.seed=1,2", "--", "out=same")
    assert status == 2


def _aliases(tmp_path: Path, kind: str) -> str:
    """An ``--out`` template whose two members are spelled apart but meet as a write reaches them."""
    if kind == "dot-dot":
        for value in ("3", "4"):
            (tmp_path / "x" / value).mkdir(parents=True)
        return f"{tmp_path}/x/{{judge.max_new_tokens}}/../out"
    (tmp_path / "a").mkdir()
    if kind == "linked parents":
        (tmp_path / "d").mkdir()
        for value in ("3", "4"):
            (tmp_path / "a" / value).symlink_to(tmp_path / "d", target_is_directory=True)
        return f"{tmp_path}/a/{{judge.max_new_tokens}}/out"
    (tmp_path / "a" / "4").mkdir()
    (tmp_path / "a" / "3").symlink_to(tmp_path / "a" / "4" / "out", target_is_directory=True)
    return f"{tmp_path}/a/{{judge.max_new_tokens}}/out"


@pytest.mark.parametrize("kind, relation", [("dot-dot", "are one place"), ("linked parents", "are one place"),
                                            ("linked inside", "lie one inside the other")])
@pytest.mark.parametrize("submitted", [False, True])
def test_members_whose_directories_meet_as_a_write_reaches_them_are_refused_before_any_runs(
        cli, inputs, calls, tmp_path, kind, relation, submitted):
    slurm = ["--slurm", "example", "--dry-run", "--json"] if submitted else []
    before = sorted(tmp_path.rglob("*"))
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", _aliases(tmp_path, kind),
                           "--sweep", "judge.max_new_tokens=3,4", *slurm)
    assert status == 2 and "sweep members judge.max_new_tokens=3 and judge.max_new_tokens=4" in error, error
    assert relation in error and "as a write reaches them" in error and not calls
    created = set(sorted(tmp_path.rglob("*"))) - set(before)
    assert not [path for path in created if path.name in ("out", "out" + records.RUNS_SUFFIX)]


def test_a_member_s_out_with_a_dotdot_after_a_directory_not_there_yet_is_refused_before_any_runs(cli, inputs, calls,
                                                                                                  tmp_path):
    before = sorted(tmp_path.rglob("*"))
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out",
                           f"{tmp_path}/x/{{judge.max_new_tokens}}/../out", "--sweep", "judge.max_new_tokens=3,4")
    assert status == 2 and not calls, error
    made = os.path.realpath(tmp_path / "x" / "3")
    assert f"names two places: {tmp_path}/x/out as written out, and {made}, which a write" in error, error
    assert sorted(tmp_path.rglob("*")) == before


def test_a_run_s_out_is_never_its_own_w_b_directory(cli, inputs, calls):
    """W&B writes in ``<out's parent>/wandb``: an ``--out`` naming that directory would be written into by W&B."""
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", "runs/wandb", "--wandb")
    assert status == 2 and "the run: its --out" in error and "and its W&B directory" in error and not calls
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out",
                           "runs/{judge.max_new_tokens}/wandb", "--wandb", "--sweep", "judge.max_new_tokens=3,4")
    assert status == 2 and "sweep member judge.max_new_tokens=3: its --out" in error and not calls
    assert not (Path.cwd() / "runs").exists()


def test_a_member_s_out_in_another_s_run_records_is_refused():
    """``x`` and ``x.runs`` are spelled apart, and the second is where the first's run records go."""
    from ruleforge.cli.common import CliError
    from ruleforge.cli.sweep import Axis, Member, refuse_overlapping

    def places(values, *, wandb):
        axis = Axis("label", "judge_model.label", "condition", values)
        cfg = types.SimpleNamespace(wandb=types.SimpleNamespace(enabled=wandb))
        return optimize_cli.places([(Member(index, ((axis, value),)), cfg, {"out": Path("/runs") / value,
                                                                             "judge": None, "agent": None})
                                    for index, value in enumerate(values)], launching=False)

    with pytest.raises(CliError, match=r"run records' directory /runs/x.runs and --out /runs/x.runs are one place"):
        refuse_overlapping(places(("x", "x.runs"), wandb=False), way_out="")
    refuse_overlapping(places(("x", "y"), wandb=True), way_out="")


def test_wandb_logs_each_episode_to_a_run_opened_for_it(cli, inputs, calls, monkeypatch):
    opened = []

    class Run:
        def __init__(self, **kwargs):
            opened.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            opened.append("finished")

    logged = []
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(
        init=lambda **kwargs: Run(**kwargs), log=lambda metrics, step: logged.append((metrics, step))))
    status, _, _ = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", "runs/w", "--wandb")
    assert status == 0 and opened[0]["project"] == "ruleforge" and opened[0]["name"] == "w" and opened[-1] == "finished"
    assert logged == [({"reward": 1.0}, 0)]


def test_a_linked_wandb_directory_is_refused_before_the_run(cli, inputs, calls, tmp_path):
    """W&B writes the run's files in ``<out's parent>/wandb``, beside the target."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (Path.cwd() / "runs").mkdir()
    (Path.cwd() / "runs" / "wandb").symlink_to(elsewhere)
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", "runs/w", "--wandb")
    assert status == 2 and "the run's W&B directory" in error and "is a symlink" in error and not calls
    assert not list(elsewhere.iterdir()) and not records.runs_dir(Path.cwd() / "runs" / "w").exists()


def test_a_slurm_dry_run_is_one_job_per_member_sized_by_its_engines(cli, inputs, calls, tmp_path):
    status, out, _ = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--agent", "fake:2", "--out", "runs",
                         "--sweep", "agent.max_new_tokens=100,200", "--slurm", "example", "--dry-run", "--json")
    assert status == 0 and not calls
    argvs = [job["argv"] for job in out["jobs"]]
    assert argvs[0][:2] == ["ruleforge", "optimize"] and "--slurm" not in argvs[0] and "--sweep" not in argvs[0]
    assert argvs[1][argvs[1].index("--out") + 1] == str(Path.cwd() / "runs" / "agent.max_new_tokens.200")
    assert argvs[1][-2:] == ["--", "++agent_model.max_new_tokens=200"]
    assert [job["resources"]["gpus"] for job in out["jobs"]] == [0, 0]


def test_a_local_judge_s_engine_sizes_its_job(inputs):
    from ruleforge.judges import JudgeSpec

    local = JudgeSpec.local("/weights/Qwen3-8B", tp_size=2)
    assert optimize_cli._gpus(local, None) == 2 and optimize_cli._gpus(local, local) == 2
    assert optimize_cli._gpus(local, JudgeSpec.local("/weights/other")) == 3
    assert optimize_cli._gpus(JudgeSpec.fake(1), None) == 0


def test_the_run_record_keeps_the_resolved_config(cli, inputs, calls):
    cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", "runs/r")
    record = next(records.runs_dir(Path.cwd() / "runs" / "r").iterdir())
    assert "num_tries: 5" in (record / records.CONFIG_FNAME).read_text()
    assert json.loads((record / records.ARGV_FNAME).read_text())["subcommand"] == "optimize"


def published_root(path: Path) -> Path:
    """A campaign root nothing but its own campaign writes (its sampling manifest marks it)."""
    path.mkdir(parents=True)
    (path / "sampling_manifest.json").write_text("{}")
    return path


@pytest.mark.parametrize("role", ["--judge", "--agent"])
@pytest.mark.parametrize("submitted", [False, True])
def test_a_batch_root_inside_a_campaign_root_is_refused_before_anything_is_written(cli, inputs, calls, tmp_path,
                                                                                   monkeypatch, role, submitted):
    published = published_root(tmp_path / "published")
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(published / "batch"))
    judges = {"--judge": "fake", "--agent": "fake:2", role: "claude-haiku-5-5-batch"}
    slurm = ["--slurm", "example"] if submitted else []
    status, _, error = cli("optimize", "--rules", inputs[1], *(item for pair in judges.items() for item in pair),
                           "--out", "runs/r", *slurm)
    assert status == 2 and "batch_root" in error and "is inside the campaign root" in error and not calls
    assert sorted(published.rglob("*")) == [published / "sampling_manifest.json"]
    assert not records.runs_dir(Path.cwd() / "runs" / "r").exists()
    assert not (tmp_path / "artifacts" / ".ruleforge-launch").exists()


def test_a_batch_root_spelled_with_a_dot_dot_after_a_symlink_is_refused_before_anything_is_written(cli, inputs, calls,
                                                                                                   tmp_path,
                                                                                                   monkeypatch):
    """``RULEFORGE_BATCH_ROOT`` reaches the transport as given."""
    published = published_root(tmp_path / "published")
    (published / "cache").mkdir()
    (tmp_path / "safe").mkdir()
    (tmp_path / "safe" / "link").symlink_to(published / "cache", target_is_directory=True)
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", f"{tmp_path / 'safe'}/link/../batch")
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "claude-haiku-5-5-batch", "--out", "runs/r")
    assert status == 2 and "batch_root" in error and "names two places" in error and not calls
    assert sorted(path.name for path in published.iterdir()) == ["cache", "sampling_manifest.json"]
    assert not records.runs_dir(Path.cwd() / "runs" / "r").exists()


@pytest.mark.parametrize("place", ["W&B directory", "run records' directory"])
def test_a_batch_root_meeting_a_shared_place_of_another_kind_is_refused_before_anything_is_written(
        cli, inputs, calls, place, monkeypatch):
    batch = Path.cwd() / "runs" / ("wandb" if place == "W&B directory" else "3.runs")
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(batch))
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "claude-haiku-5-5-batch", "--out",
                           "runs/{judge.max_new_tokens}", "--wandb", "--sweep", "judge.max_new_tokens=3,4")
    assert status == 2 and f"{place} {batch}" in error and "are one place" in error, error
    assert "batch_root" in error and not calls and not (Path.cwd() / "runs").exists()


@pytest.mark.parametrize("given", ["model config", "environment"])
def test_a_batch_root_and_an_out_spelled_with_a_tilde_are_expanded_before_anything_is_checked(cli, inputs, calls,
                                                                                             tmp_path, monkeypatch,
                                                                                             given):
    from ruleforge.env import package_path

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    judge = "claude-haiku-5-5-batch"
    if given == "model config":
        packaged = package_path(f"configs/model/{judge}.yaml")
        judge = tmp_path / "judge.yaml"
        judge.write_text(packaged.read_text().replace("${rf.env:BATCH_ROOT,null}", "~/batches"))
    else:
        monkeypatch.setenv("RULEFORGE_BATCH_ROOT", "~/batches")
    status, _, _ = cli("optimize", "--rules", inputs[1], "--judge", judge, "--out", "~/runs/r")
    assert status == 0
    options = calls[0]["judge"].options
    assert options.batch_root == options.config().batch_root == str(home / "batches")
    assert calls[0]["out"] == home / "runs" / "r" and records.runs_dir(home / "runs" / "r").is_dir()
    assert not (Path.cwd() / "~").exists()


def test_a_work_directory_its_transport_derives_is_checked_before_the_transport_touches_it(cli, inputs, tmp_path,
                                                                                         monkeypatch):
    batch, elsewhere = tmp_path / "batch", tmp_path / "elsewhere"
    batch.mkdir()
    elsewhere.mkdir()
    (batch / "requests").symlink_to(elsewhere, target_is_directory=True)
    derived = []

    def optimize(rules, **kwargs):
        for name in ("fine", "requests"):
            derived.append(batch_api.work_directory(batch, name))

    monkeypatch.setattr(optimize_api, "optimize", optimize)
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", "runs/r")
    assert status == 1 and "the batch work directory" in error and "is a symlink" in error
    assert derived == [batch / "fine"] and not list(elsewhere.iterdir())
    assert batch_api.work_directory(batch, "requests") == batch / "requests"


@pytest.fixture
def no_submission(monkeypatch):
    """``--slurm`` with nothing submitted: a submission reaching ``slurm.submit`` fails the test."""
    from ruleforge.cli import slurm

    real = slurm.submit
    monkeypatch.setattr(slurm, "submit", lambda *args, dry_run, **kwargs: real(*args, dry_run=True, **kwargs)
                        if dry_run else pytest.fail("a job was submitted"))


@pytest.mark.parametrize("submitted", [False, True])
def test_a_batch_root_in_another_member_s_out_is_refused_before_any_member_runs(cli, inputs, calls, tmp_path,
                                                                                monkeypatch, no_submission, submitted):
    runs = tmp_path / "p"
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(runs / "judge.max_new_tokens.4" / "batch"))
    slurm = ["--slurm", "example"] if submitted else []
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "claude-haiku-5-5-batch", "--out", runs,
                           "--sweep", "judge.max_new_tokens=3,4", *slurm)
    assert status == 2 and "sweep members judge.max_new_tokens=3 and judge.max_new_tokens=4" in error, error
    assert "batch_root" in error and "lie one inside the other" in error and not calls
    assert not runs.exists() and not (tmp_path / "artifacts" / ".ruleforge-launch").exists()


@pytest.mark.parametrize("batch", ["own out", "above the members", "apart"])
def test_a_batch_root_in_its_own_out_or_shared_beside_the_members_runs(cli, inputs, calls, tmp_path, monkeypatch,
                                                                     batch):
    runs = tmp_path / "p"
    places = {"own out": runs / "batch", "above the members": tmp_path, "apart": tmp_path / "batches"}
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(places[batch]))
    sweep = () if batch == "own out" else ("--sweep", "judge.max_new_tokens=3,4")
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "claude-haiku-5-5-batch", "--out", runs,
                           *sweep)
    assert status == 0, error
    assert {call["judge"].options.batch_root for call in calls} == {str(places[batch])}


@pytest.mark.parametrize("out, relation", [
    (".ruleforge-launch", "are one place"), (".ruleforge-launch.lock", "are one place"), ("..", "lie one inside the other"),
    (".ruleforge-launch/inner", "lie one inside the other")])
def test_an_out_meeting_the_launch_records_is_refused_before_anything_is_submitted(cli, inputs, calls, tmp_path,
                                                                                   monkeypatch, no_submission, out,
                                                                                   relation):
    artifacts = tmp_path / "outer" / "artifacts"
    artifacts.mkdir(parents=True)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(artifacts))
    target = Path(os.path.normpath(artifacts / out))
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", target, "--slurm",
                           "example")
    assert status == 2 and "the run: its --out" in error and "the launch's records" in error, error
    assert relation in error and not calls and sorted(artifacts.iterdir()) == []
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", target, "--slurm",
                           "example", "--dry-run")
    assert status == 0, error


@pytest.mark.parametrize(("flags", "confirmed"), [((), False), (("--yes-spend",), True)], ids=["unconfirmed", "flag"])
def test_a_run_confirms_its_spend_by_its_flag(cli, inputs, monkeypatch, flags, confirmed):
    seen = []

    def optimize(rules, **kwargs):
        seen.append(batch_api.spend_is_confirmed())
        batch_api.require_spend_confirmation()
        return OptimizeRun(out=Path(kwargs["out"]), last_episode=0, stopped_by=("max_episodes",),
                           ruleset_path=Path(rules), reward={"effective_reward": 1.0})

    monkeypatch.setattr(optimize_api, "optimize", optimize)
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--seeds-good", "g.jsonl", "--seeds-bad",
                           "b.jsonl", "--out", "runs/opt", *flags)
    assert seen == [confirmed] and status == (0 if confirmed else 1)
    assert confirmed or ("BatchSpendNotConfirmed" in error and "--yes-spend" in error)


def test_a_slurm_job_carries_the_spend_flag(cli, inputs, calls):
    status, out, _ = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", "runs", "--yes-spend", "--slurm",
                         "example", "--dry-run", "--json")
    assert status == 0 and not calls and "--yes-spend" in out["jobs"][0]["argv"]
