from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

from ruleforge.cli import evaluate as evaluate_cli
from ruleforge.cli.app import build_parser
from ruleforge.cli import records
from ruleforge.cli.common import FAKE_NOTE, note_fake
from ruleforge.evaluate import api
from ruleforge.evaluate.aggregate import AGGREGATE_DIRNAME, ESTIMATES_FNAME
from ruleforge.evaluate.campaign import campaign_completion, load_conditions
from ruleforge.judges import JudgeIdentity, JudgeSpec, Provider, extension, providers, register_judge
from ruleforge.judges.backend import SEED_LOCAL_DETERMINISTIC
from ruleforge.judges.base import Capabilities
from ruleforge.rulesets import Ruleset
from ruleforge.scenarios import ScenarioSet
from tests.conftest import requires_git_worktree
from tests.evaluate.helpers import cluster_bootstrap, recorded_cells, tree
from tests.support.scripted import scripted_backend
from tests.test_core_errors import RUNTIME_REFUSALS, TRACEBACK_ERRORS, UNFINISHED_ERRORS, _error

BILLED = Capabilities(seed="local_deterministic", batch=False, per_request_errors=True, spends_money=True)
RECORDED = Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "sweep_slurm_dry_run.json"


def evaluate(cli, inputs, *extra):
    """``ruleforge evaluate`` of the inputs under the fake judge, two draws unless a sweep gives them."""
    scenarios, rules = inputs
    draws = () if any("draws=" in str(item) for item in extra) else ("--draws", "2")
    return cli("evaluate", "--rules", rules, "--scenarios", scenarios, "--judge", "fake", *draws, *extra)


def test_the_quickstart_s_estimate_is_the_one_computed_outside_the_library(cli, inputs):
    status, out, _ = evaluate(cli, inputs, "--out", "runs/quick", "--campaign-id", "quick")
    root = Path.cwd() / "runs" / "quick"
    assert status == 0 and out["root"] == str(root) and out["campaign_id"] == "quick"
    assert out["condition_id"].startswith("fake-1/naive-") and out["draws"] == {"0": "complete", "1": "complete"}
    cells = recorded_cells(root / out["condition_id"], 2, fake_seed=1)
    values = np.array([[cells[(sid, draw)] for draw in range(2)] for sid in ScenarioSet.load(inputs[0]).ids])
    rate = out["estimates"]["rate"]
    assert rate["value"] == pytest.approx(values.mean())
    assert rate["ci"] == pytest.approx(cluster_bootstrap(values, n_resamples=10_000, ci_level=0.95, seed=16))
    assert out["estimates"]["file"] == str(root / AGGREGATE_DIRNAME / ESTIMATES_FNAME)
    version = metadata.version("ruleforge")
    assert json.loads((root / "sampling_manifest.json").read_text())["package_version"] == version
    provenance = json.loads((root / out["condition_id"] / "draw-000" / "draw_provenance.json").read_text())
    assert provenance["package_version"] == version


def test_a_rate_the_fake_judge_makes_is_marked_so_on_stderr_alone(cli, inputs):
    """Its stdout is the same JSON as any judge's, for a single condition, a sweep and an analysis."""
    status, out, error = evaluate(cli, inputs, "--out", "runs/quick", "--campaign-id", "quick")
    assert status == 0 and out["estimates"]["rate"]["value"] is not None
    assert error.splitlines()[-1] == FAKE_NOTE
    status, out, error = evaluate(cli, inputs, "--out", "runs/s", "--campaign-id", "s", "--sweep",
                                  "prompt=naive,encouraged")
    assert status == 0 and len(out["conditions"]) == 2 and error.splitlines()[-1] == FAKE_NOTE
    status, out, error = cli("analyze", "runs/s", "--replace")
    assert status == 0 and len(out["conditions"]) == 2 and error.splitlines()[-1] == FAKE_NOTE


class ScriptedProvider(Provider):
    """A free judge of a model type of its own, answering "no" to every prompt."""

    def identity(self, spec, *, verify_weights=True):
        return JudgeIdentity(model_name=spec.model_name, model_type=spec.model_type,
                             seed_capability=SEED_LOCAL_DETERMINISTIC, max_new_tokens=None,
                             extension=extension("scripted", "1.0", {}))

    def capabilities(self, spec):
        return Capabilities(seed=SEED_LOCAL_DETERMINISTIC, per_request_errors=True)

    def start(self, spec):
        backend = scripted_backend(spec.options.config(), lambda system, user: "no")
        backend.model_type, backend.max_new_tokens, backend._options = spec.model_type, None, spec.options
        return backend


def test_the_fake_note_follows_the_conditions_printed_not_the_judges_asked_for(cli, inputs, monkeypatch, tmp_path):
    """A root analysed over the fake and another judge, read again through the other judge alone."""
    monkeypatch.setattr(providers, "_REGISTERED", {})
    register_judge("scripted", ScriptedProvider)
    judge = tmp_path / "scripted.yaml"
    judge.write_text("model_type: scripted\nmodel_name: scripted\n")
    given = ("evaluate", "--rules", inputs[1], "--scenarios", inputs[0], "--draws", "2", "--campaign-id", "m",
             "--condition-id", "c-{judge}")
    status, both, error = cli(*given, "--out", "runs/m", "--sweep", f"judge=fake,{judge}")
    assert status == 0 and sorted(both["conditions"]) == ["c-fake-1", "c-scripted"], error
    status, again, error = cli(*given, "--out", "runs/m", "--sweep", f"judge={judge}", "--slurm", "example",
                               "--dry-run")
    assert status == 0 and again == both and error.splitlines()[-1] == FAKE_NOTE
    status, alone, error = cli(*given, "--out", "runs/o", "--sweep", f"judge={judge}")
    assert status == 0 and list(alone["conditions"]) == ["c-scripted"] and FAKE_NOTE not in error


def test_a_rate_no_fake_judge_makes_carries_no_note(capsys):
    note_fake(["openai", "local"])
    assert capsys.readouterr() == ("", "")


def test_a_relative_path_whose_dotdot_follows_a_symlink_is_refused_not_read_where_it_is_written(cli, inputs, tmp_path):
    """``link/../s.jsonl``: the kernel reads the link target's sibling, a lexical rebase the working directory's."""
    scenarios, rules = inputs
    rows = scenarios.read_text(encoding="utf-8").splitlines(keepends=True)
    (tmp_path / "actual" / "inner").mkdir(parents=True)
    (tmp_path / "actual" / "s.jsonl").write_text("".join(rows), encoding="utf-8")
    Path("s.jsonl").write_text("".join(rows[:2]), encoding="utf-8")
    Path("link").symlink_to(tmp_path / "actual" / "inner", target_is_directory=True)
    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "link").symlink_to(tmp_path / "actual" / "inner", target_is_directory=True)
    (conf / "run.yaml").write_text("scenario_set_path: link/../s.jsonl\n", encoding="utf-8")
    for given in (["--scenarios", "link/../s.jsonl", "--out", "runs/r"],
                  ["--scenarios", scenarios, "--out", "link/../run"],
                  ["--out", "runs/r", "--", "scenario_set_path=link/../s.jsonl"],
                  ["--config", conf / "run.yaml", "--out", "runs/r"]):
        status, out, error = cli("evaluate", "--rules", rules, "--judge", "fake", "--draws", "2", *given)
        assert status == 2 and out == "" and "link/../" in error and "names two places" in error, error
    assert not Path("runs").exists() and not (tmp_path / "actual" / "run").exists()
    assert sorted(path.name for path in Path.cwd().iterdir()) == ["link", "s.jsonl"]


def test_a_local_judge_without_vllm_ends_in_one_line_naming_what_installs_it(cli, inputs, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "vllm", None)
    scenarios, rules = inputs
    status, out, error = cli("evaluate", "--rules", rules, "--scenarios", scenarios, "--judge", f"local:{tmp_path}",
                             "--draws", "2", "--out", tmp_path / "out")
    assert status == 1 and out == "" and error == (
        "ruleforge evaluate: MissingDependency: serving a local judge needs vllm, which this environment lacks: add "
        "--group vllm to uv sync, or pip install -e '.[vllm]' from the checkout\n")


@requires_git_worktree  # the run's record names its code digest, taken over git's file list
def test_a_run_is_recorded_beside_its_root_never_inside_it(cli, inputs):
    status, out, _ = evaluate(cli, inputs, "--out", "runs/quick", "--campaign-id", "quick")
    root = Path(out["root"])
    [record] = list(records.runs_dir(root).iterdir())
    assert record.name.endswith("-evaluate") and not list(root.rglob("*.runs"))
    argv = json.loads((record / records.ARGV_FNAME).read_text())
    assert argv["argv"][:2] == ["evaluate", "--rules"] and argv["cwd"] == str(Path.cwd())
    assert argv["subcommand"] == "evaluate" and argv["code"]["code_state_sha256"]
    config = (record / records.CONFIG_FNAME).read_text()
    assert "num_decoder_draws: 2" in config and "model_type: fake" in config
    assert "Draw 1 of fake-1/naive-" in (record / records.LOG_FNAME).read_text()


def test_evaluate_again_reads_the_complete_root_and_records_the_read(cli, inputs, monkeypatch):
    status, first, _ = evaluate(cli, inputs, "--out", "runs/quick", "--campaign-id", "quick")
    before = tree(Path(first["root"]))
    monkeypatch.setattr(api, "run_plan", lambda *args, **kwargs: pytest.fail("run_plan was called"))
    status, again, _ = evaluate(cli, inputs, "--out", "runs/quick", "--campaign-id", "quick")
    assert status == 0 and again == first and tree(Path(first["root"])) == before
    assert len(list(records.runs_dir(Path(first["root"])).iterdir())) == 2


def test_the_default_root_is_the_artifact_root_s_then_the_working_directory_s(cli, inputs, monkeypatch, tmp_path):
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT")  # the suite's root
    status, out, _ = evaluate(cli, inputs)
    assert status == 0 and out["root"] == str(Path.cwd() / "ruleforge-runs" / out["campaign_id"])
    assert out["campaign_id"].startswith("eval-")
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    status, out, _ = evaluate(cli, inputs, "--draws", "3", "--attempts", "4")
    assert status == 0 and out["root"] == str(tmp_path / "artifacts" / "sampling" / out["campaign_id"])
    [verbs] = [action for action in build_parser()._actions if isinstance(action, argparse._SubParsersAction)]
    [help_text] = [action.help for action in verbs.choices["evaluate"]._actions if action.dest == "out"]
    assert "$RULEFORGE_ARTIFACT_ROOT; if unset, under ./ruleforge-runs" in help_text


def test_a_two_member_sweep_registers_both_then_analyses_once_after_the_second(cli, inputs, monkeypatch):
    registered_at_runs, analyses = [], []
    real_run, real_analyze = api.Campaign.run, api.analyze_campaign

    def run(self, condition_id=None, **kwargs):
        registered_at_runs.append((condition_id, sorted(load_conditions(self.root)),
                                   (self.root / AGGREGATE_DIRNAME / ESTIMATES_FNAME).exists()))
        return real_run(self, condition_id, **kwargs)

    def analyze(root, **kwargs):
        gate = campaign_completion(root, write=False)
        analyses.append(sorted(cid for cid, entry in gate["conditions"].items()
                               if all(draw["state"] == "complete" for draw in entry["draws"].values())))
        return real_analyze(root, **kwargs)

    monkeypatch.setattr(api.Campaign, "run", run)
    monkeypatch.setattr(api, "analyze_campaign", analyze)
    status, out, _ = evaluate(cli, inputs, "--out", "runs/sweep", "--campaign-id", "sweep", "--sweep",
                              "prompt=naive,encouraged")
    assert status == 0
    ids = ["fake-1/encouraged/prompt.encouraged-", "fake-1/naive/prompt.naive-"]
    assert [condition for condition, _, _ in registered_at_runs] == sorted(out["conditions"], reverse=True)
    for _, registered, analysed in registered_at_runs:
        assert [cid[:-8] for cid in registered] == ids and not analysed
    assert analyses == [sorted(out["conditions"])]
    estimates = json.loads((Path.cwd() / "runs" / "sweep" / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_text())
    assert sorted(estimates["conditions"]) == sorted(out["conditions"]) and out["estimates"]
    assert all(entry["estimates"]["rate"]["value"] is not None for entry in out["conditions"].values())


def test_a_sweep_read_again_once_complete_runs_nothing(cli, inputs, monkeypatch):
    sweep = ("--out", "runs/sweep", "--campaign-id", "sweep", "--sweep", "prompt=naive,encouraged")
    status, first, _ = evaluate(cli, inputs, *sweep)
    monkeypatch.setattr(api, "run_plan", lambda *args, **kwargs: pytest.fail("run_plan was called"))
    status, again, _ = evaluate(cli, inputs, *sweep)
    assert status == 0 and again == first


def test_a_collection_only_sweep_writes_no_analysis_and_returns_both_members(cli, inputs):
    status, out, _ = evaluate(cli, inputs, "--out", "runs/full", "--campaign-id", "full", "--mode", "full",
                              "--sweep", "prompt=naive,encouraged")
    assert status == 0 and out["estimates"] is None
    assert sorted(cid[:-8] for cid in out["conditions"]) == ["fake-1/encouraged/mode.full/prompt.encouraged-",
                                                             "fake-1/naive/mode.full/prompt.naive-"]
    assert all(entry["estimates"] is None and entry["mode"] == "full" for entry in out["conditions"].values())
    assert not (Path.cwd() / "runs" / "full" / AGGREGATE_DIRNAME).exists()


def test_a_campaign_level_sweep_gives_each_member_its_own_root_and_campaign(cli, inputs, tmp_path):
    status, out, _ = evaluate(cli, inputs, "--sweep", "draws=2,3", "--sweep", "seed_base=16")
    assert status == 0
    status, two, _ = evaluate(cli, inputs, "--out", "runs/c-{draws}", "--campaign-id", "c-{draws}", "--sweep",
                              "draws=2,3")
    assert status == 0
    for draws in (2, 3):
        root = Path.cwd() / "runs" / f"c-{draws}"
        manifest = json.loads((root / "sampling_manifest.json").read_text())
        assert manifest["campaign_id"] == f"c-{draws}" and manifest["expected_decoder_draws"] == draws
        assert (root / AGGREGATE_DIRNAME / ESTIMATES_FNAME).is_file()
    defaults = sorted(path.name for path in (tmp_path / "artifacts" / "sampling").iterdir()
                      if not path.name.startswith(".") and path.suffix != records.RUNS_SUFFIX)
    # Each default root is eval-<digest>-<member>: sorted by member, not by the digest of its settings.
    assert len(defaults) == 2 and sorted(name.rsplit("-", 2)[1:] for name in defaults) == [
        ["draws.2", "seed_base.16"], ["draws.3", "seed_base.16"]]


@pytest.mark.parametrize("extra, message", [
    (("--sweep", "draws=2,3", "--out", "runs/x"), "runs/x' under a sweep is a template naming {key}"),
    (("--sweep", "draws=2,3", "--campaign-id", "x"), "--campaign-id 'x' under a sweep is a template"),
    (("--sweep", "prompt=naive,forced-grpo", "--condition-id", "c"), "--condition-id 'c' under a sweep"),
    (("--sweep", "colour=red",), "--sweep colour: not a key this command sweeps"),
    (("--sweep", "draws=2", "--sweep", "draws=3"), "sweeps draws twice"),
    (("--sweep", "draws=2,2",), "gives 2 more than once"),
    (("--sweep", "draws=3..2",), "'3..2' is not a value"),
    (("--sweep", "draws=2,",), "'' is not a value"),
    (("--sweep", "draws",), "expected KEY=VALUES"),
    (("--sweep", "extra.label=a b",), "cannot name a member"),
])
def test_a_sweep_that_cannot_name_its_members_is_refused_before_anything_is_written(cli, inputs, extra, message):
    status, _, error = evaluate(cli, inputs, *extra)
    assert status == 2 and message in error
    assert not list(Path.cwd().iterdir())


def test_two_members_of_one_destination_are_refused_before_any_runs(cli, inputs, tmp_path):
    scenarios, rules = inputs
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        shutil.copy(rules, tmp_path / name / "x.jsonl")
    sweep = ("--sweep", f"rules={tmp_path / 'a' / 'x.jsonl'},{tmp_path / 'b' / 'x.jsonl'}")
    for extra in ((), ("--condition-id", "c/{rules}")):
        status, _, error = cli("evaluate", "--scenarios", scenarios, "--judge", "fake", "--out", "runs/r", *sweep,
                               *extra)
        assert status == 2 and "would both write" in error and "name them apart with a template" in error
    assert not list(Path.cwd().iterdir())


def test_two_roots_one_inside_the_other_are_refused_before_any_runs(cli, inputs, tmp_path):
    (tmp_path / "r" / "3").mkdir(parents=True)
    (tmp_path / "r" / "2").symlink_to(tmp_path / "r" / "3" / "inner", target_is_directory=True)
    status, _, error = evaluate(cli, inputs, "--out", f"{tmp_path}/r/{{draws}}", "--campaign-id", "c{draws}",
                                "--sweep", "draws=2,3")
    assert status == 2 and "sweep members draws=2 and draws=3" in error and "lie one inside the other" in error
    assert sorted(path.name for path in (tmp_path / "r").rglob("*")) == ["2", "3"]
    assert not list(Path.cwd().iterdir())


def test_members_sharing_a_root_must_share_its_campaign(cli, inputs):
    status, _, error = evaluate(cli, inputs, "--out", "runs/x{draws}{block_size}", "--campaign-id",
                                "c{draws}{block_size}", "--sweep", "draws=2,22", "--sweep", "block_size=22,2")
    assert status == 2 and "share the root" in error and not list(Path.cwd().iterdir())


def test_a_swept_judge_is_named_by_its_config_its_file_or_its_model(cli, inputs, tmp_path):
    (tmp_path / "judge.yaml").write_text("model_type: fake\nmodel_name: fake-7\nseed: 7\n")
    status, out, error = cli("evaluate", "--rules", inputs[1], "--scenarios", inputs[0], "--draws", "2", "--out",
                             "runs/j", "--sweep", f"judge=fake:3,{tmp_path / 'judge.yaml'}")
    assert status == 0, error
    assert sorted(cid[:-8] for cid in out["conditions"]) == ["fake-3/naive/judge.fake-3-",
                                                             "fake-7/naive/judge.judge-"]


def _normalised(payload, tmp_path):
    """The jobs as recorded: the test's directory as ``{tmp}``, the environment's ``bin`` as ``{bin}``."""
    jobs = json.loads(json.dumps(payload["jobs"], sort_keys=True).replace(str(tmp_path), "{tmp}"))
    for job in jobs:
        job["request"]["bin"] = "{bin}"
    return jobs


def test_a_slurm_dry_run_renders_a_job_per_member_and_an_analysis_after_all(cli, inputs, tmp_path):
    status, out, _ = evaluate(cli, inputs, "--out", "runs/sweep", "--campaign-id", "sweep", "--sweep",
                              "prompt=naive,encouraged", "--slurm", "example", "--dry-run", "--json")
    assert status == 0 and out["dry_run"] is True
    jobs = _normalised(out, tmp_path)
    assert [job["job_id"] for job in jobs] == ["evaluate-0-0", "evaluate-0-1", "analyze-0"]
    assert jobs[2]["depends_on"] == ["evaluate-0-0", "evaluate-0-1"]
    assert jobs[2]["request"]["submitit"]["slurm_additional_parameters"]["dependency"] == \
        "afterok:{evaluate-0-0}:{evaluate-0-1}"
    assert jobs == json.loads(RECORDED.read_text())
    assert not (Path.cwd() / "runs").exists()


def test_a_collection_only_dry_run_has_no_analysis_job(cli, inputs, tmp_path):
    status, out, _ = evaluate(cli, inputs, "--out", "runs/full", "--mode", "full", "--sweep",
                              "prompt=naive,encouraged", "--slurm", "example", "--dry-run", "--json")
    assert status == 0 and [job["job_id"] for job in out["jobs"]] == ["evaluate-0-0", "evaluate-0-1"]


def test_a_campaign_level_dry_run_is_one_evaluate_job_per_root(cli, inputs, tmp_path):
    status, out, _ = evaluate(cli, inputs, "--sweep", "draws=2,3", "--prompt", "naive", "--slurm", "example",
                              "--dry-run", "--json", "--", "sampling.truncation=none")
    assert status == 0
    argvs = [job["argv"] for job in out["jobs"]]
    assert [argv[:2] for argv in argvs] == [["ruleforge", "evaluate"]] * 2
    for argv, draws in zip(argvs, (2, 3)):
        assert argv[argv.index("--draws", argv.index("--campaign-id")) + 1] == str(draws)
        assert argv[-2:] == ["--", "sampling.truncation=none"] and "--slurm" not in argv and "--sweep" not in argv


def test_a_slurm_submission_registers_the_members_and_writes_their_judges(cli, inputs, monkeypatch, tmp_path):
    from ruleforge.cli import slurm

    submitted = []
    monkeypatch.setattr(slurm, "submit", lambda specs, profile, **kwargs: submitted.append(specs) or 0)
    status, _, _ = evaluate(cli, inputs, "--out", "runs/s", "--campaign-id", "s", "--sweep", "prompt=naive,encouraged",
                            "--slurm", "example", "--yes-spend", "--", "execution.pack_max_prompts=7")
    assert status == 0
    root = Path.cwd() / "runs" / "s"
    assert len(load_conditions(root)) == 2 and not list(root.glob("*/draw-*"))
    member = submitted[0][0].argv
    judge = Path(member[member.index("--judge") + 1])
    assert judge.is_file() and "model_type: fake" in judge.read_text()
    assert list(member[-3:]) == ["--yes-spend", "--", "execution.pack_max_prompts=7"]


def test_a_completed_sweep_submitted_again_is_read_as_a_local_sweep_reads_it(cli, inputs, monkeypatch):
    from ruleforge.cli import slurm

    sweep = ("--out", "runs/s", "--campaign-id", "s", "--sweep", "prompt=naive,encouraged")
    status, local, _ = evaluate(cli, inputs, *sweep)
    root = Path.cwd() / "runs" / "s"
    before = tree(root)
    monkeypatch.setattr(slurm, "submit", lambda *args, **kwargs: pytest.fail("a job was submitted"))
    for submission in (("--slurm", "example"), ("--slurm", "example", "--dry-run", "--json")):
        status, again, error = evaluate(cli, inputs, *sweep, *submission)
        assert status == 0 and again == local and tree(root) == before
        assert error.splitlines()[-1] == FAKE_NOTE
    status, _, _ = cli("analyze", root, "--replace", "--ci-level", "0.5", "--n-resamples", "100")
    status, _, error = evaluate(cli, inputs, *sweep, "--slurm", "example", "--dry-run")
    assert status == 2 and "analysed under other settings" in error and "--replace" in error
    status, _, error = evaluate(cli, inputs, "--out", "runs/s", "--campaign-id", "s", "--mode", "full", "--sweep",
                                "prompt=naive,encouraged", "--slurm", "example", "--dry-run")
    assert status == 2 and "not registered there: a condition joins an analysed root explicitly" in error


def test_an_analysed_root_of_a_sweep_submits_no_job_beside_the_roots_still_to_run(cli, inputs, tmp_path):
    status, _, _ = evaluate(cli, inputs, "--sweep", "draws=2", "--sweep", "prompt=naive,encouraged")
    assert status == 0
    status, out, error = evaluate(cli, inputs, "--sweep", "draws=2,3", "--sweep", "prompt=naive,encouraged",
                                  "--slurm", "example", "--dry-run", "--json")
    assert status == 0 and [job["job_id"] for job in out["jobs"]] == ["evaluate-1-0", "evaluate-1-1", "analyze-1"]
    assert "draws.2 holds an accepted analysis of its members: nothing of it to run" in error


def test_a_condition_added_beside_an_analysis_is_run_then_the_root_analysed_again(cli, inputs, tmp_path):
    scenarios, rules = inputs
    status, first, _ = evaluate(cli, inputs, "--out", "runs/r", "--campaign-id", "r")
    root = Path(first["root"])
    status, added, _ = cli("campaign", "add", root, "--scenarios", scenarios, "--rules", rules, "--judge", "fake:2",
                           "--extra", "label=second")
    assert status == 0 and added["condition_id"].startswith("fake-2/naive-")
    status, ran, _ = cli("campaign", "run", root, added["condition_id"], "--judge", "fake:2", "--rules", rules,
                         "--scenarios", scenarios)
    assert status == 0 and ran["draws"] == {"0": "complete", "1": "complete"}
    status, _, error = cli("analyze", root)
    assert status == 1 and "--replace" in error
    status, analysed, _ = cli("campaign", "analyze", root, "--replace", "--ci-level", "0.5", "--n-resamples", "100")
    assert status == 0 and sorted(analysed["conditions"]) == sorted([first["condition_id"], added["condition_id"]])
    assert (analysed["bootstrap"]["ci_level"], analysed["bootstrap"]["n_resamples"]) == (0.5, 100)
    status, gate, _ = cli("campaign", "status", root)
    assert status == 0 and gate["state"] == "complete" and gate["n_conditions"] == 2


def test_campaign_run_refuses_a_condition_the_root_does_not_register(cli, inputs):
    status, first, _ = evaluate(cli, inputs, "--out", "runs/r", "--campaign-id", "r")
    status, _, error = cli("campaign", "run", first["root"], "nope", "--judge", "fake", "--rules", inputs[1],
                           "--scenarios", inputs[0], "--", "execution.pack_max_prompts=3")
    assert status == 2 and "registers no condition 'nope'" in error
    status, _, error = cli("campaign", "run", first["root"], first["condition_id"], "--judge", "fake", "--rules",
                           inputs[1], "--scenarios", inputs[0], "--", "execution.nonsense=3")
    assert status == 2 and "execution.<setting>" in error


def test_a_root_this_library_did_not_create_is_refused_before_any_record(cli, inputs, tmp_path):
    foreign = Path.cwd() / "published"
    foreign.mkdir()
    (foreign / "sampling_manifest.json").write_text("{}")
    status, _, error = evaluate(cli, inputs, "--out", "published", "--campaign-id", "x")
    assert status == 2 and "did not create" in error
    status, _, error = cli("analyze", foreign)
    assert status == 2 and "did not create" in error
    assert not records.runs_dir(foreign).exists() and sorted(path.name for path in foreign.iterdir()) == [
        "sampling_manifest.json"]


def test_a_refusal_s_traceback_shows_under_verbose(cli, inputs):
    with pytest.raises(Exception, match="did not create"):
        foreign = Path.cwd() / "published"
        foreign.mkdir()
        (foreign / "x").write_text("")
        cli("-v", "analyze", foreign)


def test_a_swept_judge_s_label_and_a_member_s_job_argv(cli, inputs):
    assert evaluate_cli._judge_label("judge", "qwen3-8b") == "qwen3-8b"
    assert evaluate_cli._judge_label("prompt", "naive") is None
    status, out, _ = evaluate(cli, inputs, "--sweep", "draws=2..3", "--sweep", "block_size=1", "--slurm", "example",
                              "--dry-run", "--json")
    assert status == 0
    argv = out["jobs"][1]["argv"]
    assert argv[-2:] == ["--", "sampling.scenarios_per_request_block=1"] and "--draws" in argv


def test_a_campaign_run_job_names_every_input_by_path():
    import types

    from ruleforge.evaluate.packing import ExecutionSettings

    call = types.SimpleNamespace(rules=types.SimpleNamespace(path=Path("/r.jsonl")),
                                 scenarios=types.SimpleNamespace(path=Path("/s.jsonl")),
                                 condition=types.SimpleNamespace(rule_levels=Path("/levels.json")),
                                 confirm_spend=False, relocate_batch_root=True,
                                 execution=ExecutionSettings(), sampling=api.SamplingSettings())
    item = types.SimpleNamespace(call=call, judge_file=Path("/j.yaml"))
    condition = types.SimpleNamespace(condition_id="c/1", task_locator="pkg.tasks:Mine")
    assert evaluate_cli._run_argv(Path("/root"), item, condition) == [
        "campaign", "run", "/root", "c/1", "--judge", "/j.yaml", "--rules", "/r.jsonl", "--scenarios", "/s.jsonl",
        "--rule-levels", "/levels.json", "--task", "pkg.tasks:Mine", "--relocate-batch-root"]
    call.sampling, call.execution = api.SamplingSettings(max_transport_retries=9), ExecutionSettings(pack_draws=True)
    assert evaluate_cli._run_argv(Path("/root"), item, condition)[-3:] == [
        "--", "execution.pack_draws=true", "max_transport_retries=9"]


def test_a_sweep_names_a_key_by_its_config_key_too(cli, inputs):
    status, out, _ = evaluate(cli, inputs, "--out", "runs/c{draws}", "--campaign-id", "c{draws}", "--sweep",
                              "sampling.num_decoder_draws=2")
    assert status == 0 and out["root"] == str(Path.cwd() / "runs" / "c2") and out["campaign_id"] == "c2"


def test_campaign_operations_refuse_what_they_cannot_read(cli, inputs, tmp_path):
    scenarios, rules = inputs
    status, first, _ = evaluate(cli, inputs, "--out", "runs/r", "--campaign-id", "r")
    status, _, error = cli("campaign", "add", first["root"], "--scenarios", scenarios, "--rules", rules, "--judge",
                           "fake", "--extra", "nonsense")
    assert status == 2 and "--extra 'nonsense': expected KEY=VALUE" in error
    status, _, error = cli("campaign", "run", first["root"], first["condition_id"], "--judge", "fake", "--rules",
                           rules, "--scenarios", scenarios, "--", "other.x=1")
    assert status == 2 and "['other'] are no execution settings" in error
    (tmp_path / "list.yaml").write_text("- a\n")
    status, _, error = cli("judges", "show", tmp_path / "list.yaml")
    assert status == 2 and "not a model config (a mapping)" in error
    (tmp_path / "typeless.yaml").write_text("model_name: x\n")
    status, _, error = cli("judges", "show", tmp_path / "typeless.yaml")
    assert status == 2 and "the judge's model config is refused" in error


def test_a_swept_extra_value_is_recorded_as_typed_as_campaign_add_records_it(cli, inputs):
    scenarios, rules = inputs
    status, out, error = evaluate(cli, inputs, "--out", "runs/x", "--campaign-id", "x", "--mode", "full", "--sweep",
                                  "extra.version=1.10,1.1")
    assert status == 0, error
    root = Path(out["conditions"][next(iter(out["conditions"]))]["root"])
    registered = load_conditions(root)
    assert sorted(dict(spec.extra)["version"] for spec in registered.values()) == ["1.1", "1.10"]
    assert len({spec.fingerprint for spec in registered.values()}) == 2
    swept = next(spec for spec in registered.values() if dict(spec.extra)["version"] == "1.10")
    status, added, _ = cli("campaign", "add", root, "--scenarios", scenarios, "--rules", rules, "--judge", "fake",
                           "--mode", "full", "--extra", "version=1.10", "--condition-id", "again")
    assert status == 0 and added["fingerprint"] == swept.fingerprint


@pytest.mark.parametrize("given, message", [
    (("--judge", "fake", "--rules", "{rules}", "--out", "runs/r{{seed_base}}", "--campaign-id", "r{{seed_base}}",
      "--sweep", "seed_base=16,1_6"),
     "--sweep seed_base: 16 and 1_6 compose to one value (16)"),
    (("--rules", "{rules}", "--sweep", "judge=fake,fake:1"), "--sweep judge: fake and fake:1 compose to one value"),
    (("--judge", "fake", "--sweep", "rules={rules},../work/{relative}"), "compose to one value"),
    # Two judge configs that compose apart, one of them with a setting no condition records: one condition.
    (("--rules", "{rules}", "--sweep", "judge=fake,{retrying}"),
     "sweep members judge=fake and judge={retrying} are one condition (fingerprint"),
])
def test_two_spellings_of_one_value_are_refused_before_anything_is_written(cli, inputs, given, message):
    scenarios, rules = inputs
    relative = Path("..") / rules.relative_to(Path.cwd().parent)
    retrying = rules.parent / "retrying.yaml"
    retrying.write_text("model_type: fake\nmodel_name: fake-1\nseed: 1\nnum_tries: 5\n", encoding="utf-8")
    status, _, error = cli("evaluate", "--scenarios", scenarios, "--draws", "2",
                           *(item.format(rules=rules, relative=relative, retrying=retrying) for item in given))
    message = message.format(retrying=retrying)
    assert status == 2 and message in error, error
    assert not list(Path.cwd().iterdir())


def test_what_evaluate_refuses_before_it_writes_leaves_no_root(cli, inputs, tmp_path):
    status, _, error = evaluate(cli, inputs, "--mode", "bogus", "--out", "runs/b")
    assert status == 2 and "mode 'bogus' is not one of" in error
    bare = tmp_path / "bare.jsonl"
    bare.write_text('{"prompt": "p"}\n')
    status, _, error = cli("evaluate", "--rules", inputs[1], "--scenarios", bare, "--judge", "fake", "--out", "runs/c")
    assert status == 2 and "'response'" in error
    assert not list(Path.cwd().iterdir())


def test_a_command_s_record_is_refused_inside_a_campaign_root_before_it_writes(cli, inputs, monkeypatch):
    import ruleforge.grpo as grpo_package

    import functools

    published = Path.cwd() / "published"
    published.mkdir()
    (published / "sampling_manifest.json").write_text("{}")
    real = grpo_package.rows
    monkeypatch.setattr(grpo_package, "rows", functools.wraps(real)(lambda **kwargs: pytest.fail("rows ran")))
    status, _, error = cli("grpo", "rows", "--anchors", "a", "--scenarios", "s", "--rules", "r", "--model", "m",
                           "--out", published / "rows")
    assert status == 2 and "is inside the campaign root" in error
    status, _, error = cli("optimize", "--rules", inputs[1], "--judge", "fake", "--out", published / "existing")
    assert status == 2 and "is inside the campaign root" in error
    assert sorted(path.name for path in published.iterdir()) == ["sampling_manifest.json"]


def test_a_linked_run_records_directory_is_refused_before_the_analysis(cli, inputs, tmp_path):
    status, out, _ = evaluate(cli, inputs, "--out", "runs/r", "--campaign-id", "r")
    root = Path(out["root"])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.rmtree(records.runs_dir(root))
    records.runs_dir(root).symlink_to(elsewhere)
    before = tree(root)
    status, _, error = cli("analyze", root, "--replace")
    assert status == 2 and "is a symlink" in error and tree(root) == before and not list(elsewhere.iterdir())


def test_a_run_leaving_a_draw_incomplete_exits_1(cli, inputs, monkeypatch):
    from ruleforge.evaluate.runner import TransportExhausted

    monkeypatch.setattr(api, "run_plan", lambda *args, **kwargs: {})
    status, out, _ = evaluate(cli, inputs, "--out", "runs/f", "--mode", "full")
    assert status == 1 and "complete" not in out["draws"].values()
    status, out, _ = evaluate(cli, inputs, "--out", "runs/g", "--mode", "full", "--sweep", "prompt=naive,encouraged")
    assert status == 1 and out["estimates"] is None

    def exhausted(*args, **kwargs):
        raise TransportExhausted("the judge kept failing")

    monkeypatch.setattr(api, "run_plan", exhausted)
    status, _, error = evaluate(cli, inputs, "--out", "runs/h")
    assert status == 1 and "TransportExhausted: the judge kept failing" in error


def _raised_by_the_run(monkeypatch, error):
    def run_plan(*args, **kwargs):
        raise error

    monkeypatch.setattr(api, "run_plan", run_plan)


def test_a_full_disk_exits_2_before_the_run_record_and_1_after_it(cli, inputs, monkeypatch):
    full = OSError(28, "No space left on device")

    def no_space(*args, **kwargs):
        raise full

    with monkeypatch.context() as patched:
        patched.setattr(api.Campaign, "create", no_space)
        status, _, error = evaluate(cli, inputs, "--out", "runs/before")
    assert status == 2 and "No space left on device" in error and not Path("runs/before.runs").exists()
    _raised_by_the_run(monkeypatch, full)
    status, _, error = evaluate(cli, inputs, "--out", "runs/after")
    [record] = records.runs_dir(Path.cwd() / "runs" / "after").iterdir()
    assert status == 1 and "No space left on device" in error
    assert (record / records.LOG_FNAME).read_text().splitlines()[-1].endswith(
        "the run ended on OSError: [Errno 28] No space left on device")


@pytest.mark.parametrize("name", TRACEBACK_ERRORS)
def test_a_traceback_error_ends_the_command_with_its_traceback(cli, inputs, monkeypatch, name):
    error = _error(name)("the run broke")
    _raised_by_the_run(monkeypatch, error)
    with pytest.raises(type(error)):
        evaluate(cli, inputs, "--out", "runs/t")


@pytest.mark.parametrize("name", [*RUNTIME_REFUSALS, "ruleforge.judges.base.JudgeError", *UNFINISHED_ERRORS,
                                  "ruleforge.judges.batch_api.BatchAlreadyClaimed"])
def test_a_refusal_or_an_unfinished_run_ends_the_command_in_one_line_and_its_traceback_only_when_verbose(
        cli, inputs, monkeypatch, name):
    error = _error(name)("the run stopped here")
    _raised_by_the_run(monkeypatch, error)
    ended, _, err = evaluate(cli, inputs, "--out", "runs/r")
    assert ended == 1 and "Traceback" not in err
    assert f"ruleforge evaluate: {type(error).__name__}: the run stopped here" in err.splitlines()
    with pytest.raises(type(error)):
        cli("--verbose", "evaluate", "--rules", inputs[1], "--scenarios", inputs[0], "--judge", "fake", "--draws", "2",
            "--out", "runs/v")


def test_an_abbreviated_option_is_no_option():
    from ruleforge.cli.app import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["evaluate", "--slu", "example"])


def test_each_member_s_job_is_sized_for_its_own_judge(cli, inputs, monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    status, out, error = cli("evaluate", "--rules", inputs[1], "--scenarios", inputs[0], "--draws", "2", "--out",
                             "runs/j", "--sweep", "judge=fake,gemma4-31b-tp-2", "--slurm", "example", "--dry-run",
                             "--json")
    assert status == 0, error
    assert [(job["job_id"], job["resources"]["gpus"]) for job in out["jobs"]] == [
        ("evaluate-0-0", 0), ("evaluate-0-1", 2), ("analyze-0", 0)]


def test_a_job_is_given_its_destination_once(cli, inputs):
    status, out, _ = evaluate(cli, inputs, "--slurm", "example", "--dry-run", "--json", "--", "campaign_root=runs/x",
                              "campaign_id=c", "sampling.truncation=none")
    [job] = out["jobs"]
    assert status == 0 and job["argv"][-2:] == ["--", "sampling.truncation=none"]
    assert job["argv"][job["argv"].index("--out") + 1] == str(Path.cwd() / "runs" / "x")
    assert job["argv"][job["argv"].index("--campaign-id") + 1] == "c" and job["kind"] == "evaluate"
    status, out, _ = cli("-v", "optimize", "--rules", inputs[1], "--judge", "fake", "--slurm", "example", "--dry-run",
                         "--json", "--", "out=runs/o")
    [job] = out["jobs"]
    assert status == 0 and job["kind"] == "optimize" and not any(item.startswith("out=") for item in job["argv"])
    assert job["argv"][job["argv"].index("--out") + 1] == str(Path.cwd() / "runs" / "o")


def test_a_member_s_judge_is_written_as_registered_and_its_retries_carried(cli, inputs, monkeypatch, tmp_path):
    from ruleforge.cli import slurm

    (tmp_path / "judge.yaml").write_text("model_type: fake\nmodel_name: fake-16\nseed: ${sampling.generation_seed_base}\n")
    submitted = []
    monkeypatch.setattr(slurm, "submit", lambda specs, profile, **kwargs: submitted.append(specs) or 0)
    status, _, error = evaluate(cli, inputs, "--judge", tmp_path / "judge.yaml", "--out", "runs/s", "--campaign-id",
                                "s", "--sweep", "prompt=naive,encouraged", "--slurm", "example", "--",
                                "max_transport_retries=9")
    assert status == 0, error
    member = submitted[0][0].argv
    judge = Path(member[member.index("--judge") + 1])
    written = judge.read_text()
    assert "seed: 16" in written and "${" not in written
    recorded = (judge.parents[1] / records.CONFIG_FNAME).read_text()
    assert not recorded.startswith("# unresolved") and "prompt=encouraged:" in recorded
    assert list(member[-2:]) == ["--", "max_transport_retries=9"]
    root = Path.cwd() / "runs" / "s"
    status, ran, error = cli(*member[1:])
    assert status == 0 and ran["draws"] == {"0": "complete", "1": "complete"}, error
    status, _, error = cli(*member[1:-1], "max_transport_retries=x")
    assert status == 2 and "max_transport_retries='x' is no whole number" in error
    assert root.is_dir()


def test_a_final_campaign_approving_another_count_is_rebound_with_it(inputs):
    scenarios = ScenarioSet.load(inputs[0])
    root = Path.cwd() / "runs" / "final"
    api.Campaign.create(root, campaign_id="final", scenarios=scenarios,
                        sampling=api.SamplingSettings(campaign_mode="final", approved_scenario_count=3, draws=2))
    rebound = api.Campaign.create(**evaluate_cli._recorded_claim(root, inputs[0]))
    assert rebound.sampling.approved_scenario_count == 3 and rebound.config.campaign_mode == "final"
    pilot = Path.cwd() / "runs" / "pilot"
    api.Campaign.create(pilot, campaign_id="pilot", scenarios=scenarios, sampling=api.SamplingSettings(draws=2))
    again = api.Campaign.create(**evaluate_cli._recorded_claim(pilot, inputs[0], max_transport_retries=7))
    assert again.sampling.approved_scenario_count is None and again.sampling.max_transport_retries == 7


def test_a_value_given_twice_to_a_campaign_operation_is_refused(cli, inputs):
    scenarios, rules = inputs
    status, first, _ = evaluate(cli, inputs, "--out", "runs/r", "--campaign-id", "r")
    status, _, error = cli("campaign", "add", first["root"], "--scenarios", scenarios, "--rules", rules, "--judge",
                           "fake", "--extra", "v=1", "--extra", "v=2")
    assert status == 2 and "--extra v=1 and --extra v=2 both set v" in error
    status, _, error = cli("campaign", "run", first["root"], first["condition_id"], "--judge", "fake", "--rules",
                           rules, "--scenarios", scenarios, "--", "execution.pack_draws=true",
                           "execution.pack_draws=false")
    assert status == 2 and "-- execution.pack_draws given more than once" in error


def test_every_root_s_registrations_are_checked_before_any_root_is_registered(cli, inputs, monkeypatch):
    from ruleforge.cli import slurm

    scenarios, rules = inputs
    status, _, _ = evaluate(cli, inputs, "--draws", "3", "--out", "runs/c3", "--campaign-id", "c3",
                            "--condition-id", "c/naive", "--mode", "full")
    assert status == 0
    monkeypatch.setattr(slurm, "submit", lambda *args, **kwargs: pytest.fail("submitted"))
    status, _, error = cli("evaluate", "--rules", rules, "--scenarios", scenarios, "--judge", "fake", "--out",
                           "runs/c{draws}", "--campaign-id", "c{draws}", "--condition-id", "c/{prompt}", "--sweep",
                           "draws=2,3", "--sweep", "prompt=naive,encouraged", "--slurm", "example")
    assert status == 2 and "different identity" in error
    assert not (Path.cwd() / "runs" / "c2").exists()


def test_a_member_s_config_that_does_not_resolve_here_is_recorded_as_written(monkeypatch):
    import types

    from omegaconf import OmegaConf

    monkeypatch.delenv("SURELY_UNSET_VARIABLE_X", raising=False)
    members = [types.SimpleNamespace(member=types.SimpleNamespace(describe=lambda value=value: f"k={value}"),
                                     call=types.SimpleNamespace(cfg=OmegaConf.create(cfg)))
               for value, cfg in ((1, {"a": 1, "b": "${a}"}), (2, {"a": "${oc.env:SURELY_UNSET_VARIABLE_X}"}))]
    text = records.config_text(evaluate_cli.configs_text(members))
    assert text.startswith("# unresolved") and "SURELY_UNSET_VARIABLE_X" in text
    assert OmegaConf.to_container(evaluate_cli.configs_text(members[:1] * 2)) == {"k=1": {"a": 1, "b": 1}}


@pytest.fixture
def no_submission(monkeypatch):
    """``--slurm`` with nothing submitted: a submission reaching ``slurm.submit`` fails the test."""
    from ruleforge.cli import slurm

    monkeypatch.setattr(slurm, "submit", lambda *args, **kwargs: pytest.fail("a job was submitted"))


BATCH = "claude-haiku-5-5-batch"


@pytest.mark.parametrize("mode", [("--mode", "full"), ()], ids=["collections", "analyses"])
def test_a_completed_sweep_over_campaign_roots_is_read_without_the_spend_confirmed(cli, inputs, tmp_path,
                                                                                  monkeypatch, mode):
    sweep = ("--out", f"{tmp_path}/c-{{draws}}", "--campaign-id", "c-{draws}", "--sweep", "draws=2,3", *mode)
    status, first, error = evaluate(cli, inputs, *sweep)
    assert status == 0, error
    monkeypatch.setattr(JudgeSpec, "capabilities", lambda self: BILLED)
    monkeypatch.setattr(api, "run_plan", lambda *args, **kwargs: pytest.fail("run_plan was called"))
    status, again, error = evaluate(cli, inputs, *sweep)
    assert status == 0 and again == first, error


@pytest.mark.parametrize("ran", [False, True], ids=["pending", "complete"])
def test_an_unconfirmed_paid_judge_asking_more_draws_than_its_root_froze_meets_the_root_s_refusal(cli, inputs, ran,
                                                                                               monkeypatch):
    scenarios, rules = inputs
    flags = ("--out", "runs/r", "--campaign-id", "r", "--condition-id", "c", "--mode", "full")
    if ran:
        assert evaluate(cli, inputs, *flags)[0] == 0
    else:
        api.Campaign.create("runs/r", campaign_id="r", scenarios=ScenarioSet.load(scenarios),
                            sampling=api.SamplingSettings(draws=2)).condition(
            "c", rules=Ruleset.load(rules), judge=JudgeSpec.fake(1), settings=api.ConditionSettings(mode="full"))
    before = tree(Path.cwd() / "runs")
    monkeypatch.setattr(JudgeSpec, "capabilities", lambda self: BILLED)
    for draws in ("3", "4"):
        status, _, error = cli("evaluate", "--rules", rules, "--scenarios", scenarios, "--judge", "fake", *flags,
                               "--draws", draws)
        assert status == 2 and "expected_decoder_draws" in error and "Traceback" not in error, error
    assert tree(Path.cwd() / "runs") == before


@pytest.mark.parametrize("submission", [(), ("--sweep", "prompt=naive,encouraged"), ("--slurm", "example"),
                                        ("--slurm", "example", "--dry-run")])
def test_an_unconfirmed_paid_judge_is_refused_before_any_root_or_record_is_made(cli, inputs, tmp_path, monkeypatch,
                                                                              no_submission, submission):
    monkeypatch.setattr(JudgeSpec, "capabilities", lambda self: BILLED)
    status, _, error = evaluate(cli, inputs, "--out", tmp_path / "r", "--campaign-id", "c", *submission)
    assert status == 2 and "spends money on every call; pass confirm_spend=True (--yes-spend)" in error, error
    assert sorted(path.name for path in tmp_path.iterdir()) == ["artifacts", "inputs", "work"]
    assert not list((tmp_path / "artifacts").iterdir())


@pytest.mark.parametrize("submission", [(), ("--sweep", "prompt=naive,encouraged", "--slurm", "example")])
def test_a_batch_root_in_another_member_s_root_is_refused_before_any_root_is_made(cli, inputs, tmp_path, monkeypatch,
                                                                                 no_submission, submission):
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "r" / "3" / "batch"))
    status, _, error = cli("evaluate", "--rules", inputs[1], "--scenarios", inputs[0], "--judge", BATCH, "--out",
                           f"{tmp_path}/r/{{draws}}", "--campaign-id", "c{draws}", "--sweep", "draws=2,3",
                           "--yes-spend", *submission)
    assert status == 2 and "sweep members draws=2" in error and "batch_root" in error, error
    assert "lie one inside the other" in error and not (tmp_path / "r").exists()
    assert not (tmp_path / "artifacts" / ".ruleforge-launch").exists()


@pytest.mark.parametrize("batch", ["own root", "above the root"])
def test_a_batch_root_in_its_own_root_or_shared_above_the_roots_is_accepted(cli, inputs, tmp_path, monkeypatch,
                                                                           batch):
    root = tmp_path / "r"
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(root / "batch" if batch == "own root" else tmp_path))
    status, out, error = cli("evaluate", "--rules", inputs[1], "--scenarios", inputs[0], "--judge", BATCH, "--out",
                             root, "--campaign-id", "c", "--sweep", "prompt=naive,encouraged", "--yes-spend",
                             "--slurm", "example", "--dry-run", "--json")
    assert status == 0, error
    assert [job["job_id"] for job in out["jobs"]] == ["evaluate-0-0", "evaluate-0-1", "analyze-0"]


@pytest.mark.parametrize("spelling", ["a '..' after a link", "a '~'"])
def test_a_batch_root_is_held_apart_where_its_campaign_writes_it(cli, inputs, tmp_path, monkeypatch, no_submission,
                                                                 spelling):
    home, safe = tmp_path / "home", tmp_path / "safe"
    (tmp_path / "elsewhere" / "inner").mkdir(parents=True)
    safe.mkdir()
    home.mkdir()
    (safe / "link").symlink_to(tmp_path / "elsewhere" / "inner", target_is_directory=True)
    monkeypatch.setenv("HOME", str(home))
    base = safe if spelling == "a '..' after a link" else home
    batch = f"{safe}/link/../r/3/batch" if spelling == "a '..' after a link" else "~/r/3/batch"
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", batch)
    for submission in ((), ("--slurm", "example")):
        status, _, error = cli("evaluate", "--rules", inputs[1], "--scenarios", inputs[0], "--judge", BATCH, "--out",
                               f"{base}/r/{{draws}}", "--campaign-id", "c{draws}", "--sweep", "draws=2,3",
                               "--yes-spend", *submission)
        expected = ("names two places" if spelling == "a '..' after a link" else
                    f"batch_root {home / 'r' / '3' / 'batch'} and root {home / 'r' / '3'} lie one inside the other")
        assert status == 2 and expected in error, error
        assert not (base / "r").exists() and not (tmp_path / "elsewhere" / "r").exists()
        assert not (tmp_path / "artifacts" / ".ruleforge-launch").exists()


@pytest.mark.parametrize("sweep", [(), ("--sweep", "draws=2,3")])
def test_a_batch_root_at_a_campaign_lock_is_refused_before_any_root_is_made(cli, inputs, tmp_path, monkeypatch,
                                                                            sweep):
    """An analysis takes ``.<root name>.lock`` beside its root: a batch root there."""
    out, lock = (f"{tmp_path}/r/{{draws}}", tmp_path / "r" / ".3.lock") if sweep else (tmp_path / "r" / "x",
                                                                                     tmp_path / "r" / ".x.lock")
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(lock))
    status, _, error = cli("evaluate", "--rules", inputs[1], "--scenarios", inputs[0], "--judge", BATCH, "--out",
                           out, "--campaign-id", "c{draws}" if sweep else "c", *sweep, "--yes-spend")
    assert status == 2 and "campaign lock" in error and f"are one place, {lock}" in error, error
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize("record", ["completion.json", ".ruleforge-campaign.lock", "aggregate/estimates.json",
                                    "c1/naive/condition.json", "c1/naive/.ruleforge-run.lock"])
def test_a_batch_root_on_a_record_of_its_campaign_root_is_refused_by_the_preflight(cli, inputs, tmp_path,
                                                                                    monkeypatch, record):
    root = tmp_path / "r"
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(root / record))
    status, _, error = cli("evaluate", "--rules", inputs[1], "--scenarios", inputs[0], "--judge", BATCH, "--out",
                           root, "--campaign-id", "c", "--condition-id", "c1/naive", "--yes-spend")
    assert status == 2 and "a record of the campaign's own" in error, error
    assert not root.exists()


@pytest.mark.parametrize("place", ["records lock", "records directory"])
def test_a_batch_root_meeting_the_launch_records_is_refused_before_any_root_is_registered(cli, inputs, tmp_path,
                                                                                         monkeypatch, no_submission,
                                                                                         place):
    launch = tmp_path / "artifacts" / ".ruleforge-launch"
    batch = launch.with_name(".ruleforge-launch.lock") if place == "records lock" else launch / "batch"
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(batch))
    status, _, error = cli("evaluate", "--rules", inputs[1], "--scenarios", inputs[0], "--judge", BATCH, "--out",
                           tmp_path / "r", "--campaign-id", "c", "--sweep", "prompt=naive,encouraged", "--yes-spend",
                           "--slurm", "example")
    relation = "are one place" if place == "records lock" else "lie one inside the other"
    assert status == 2 and f"the launch's {place}" in error and relation in error, error
    assert not (tmp_path / "r").exists() and not list((tmp_path / "artifacts").iterdir())


def test_a_root_at_another_root_s_campaign_lock_is_refused_before_any_root_is_made(cli, inputs, tmp_path):
    """An analysis of a root takes ``.<root name>.lock`` beside it."""
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "2").symlink_to(tmp_path / "r" / ".3.lock", target_is_directory=True)
    status, _, error = evaluate(cli, inputs, "--out", f"{tmp_path}/r/{{draws}}", "--campaign-id", "c{draws}",
                                "--sweep", "draws=2,3")
    assert status == 2 and "sweep members draws=2 and draws=3" in error and "campaign lock" in error, error
    assert "are one place" in error and sorted(path.name for path in (tmp_path / "r").iterdir()) == ["2"]


@pytest.mark.parametrize("mode", [(), ("--mode", "full")], ids=["analysed", "collection"])
def test_a_linked_campaign_lock_is_refused_before_the_root_is_made(cli, inputs, tmp_path, mode):
    """The campaign lock is taken by the root's creation and by an analysis."""
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / ".r.lock").symlink_to(tmp_path / "elsewhere" / "lock")
    status, _, error = evaluate(cli, inputs, "--out", tmp_path / "r", "--campaign-id", "c", *mode)
    assert status == 2 and "the campaign lock beside the root" in error and "is a symlink" in error, error
    assert not (tmp_path / "r").exists() and not list((tmp_path / "elsewhere").iterdir())


def test_a_root_in_another_root_s_run_records_is_refused_before_any_root_is_made(cli, inputs, tmp_path):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "2").symlink_to(tmp_path / "r" / "3.runs" / "inner", target_is_directory=True)
    status, _, error = evaluate(cli, inputs, "--out", f"{tmp_path}/r/{{draws}}", "--campaign-id", "c{draws}",
                                "--sweep", "draws=2,3")
    assert status == 2 and "run records' directory" in error and "lie one inside the other" in error, error
    assert sorted(path.name for path in (tmp_path / "r").iterdir()) == ["2"]


@pytest.mark.parametrize("out", [".ruleforge-launch", ".."])
def test_a_root_meeting_the_launch_records_is_refused_before_any_root_is_registered(cli, inputs, tmp_path,
                                                                                   monkeypatch, no_submission, out):
    """Under ``--slurm`` the launch writes its record under the artifact root's launch records directory."""
    artifacts = tmp_path / "outer" / "artifacts"
    artifacts.mkdir(parents=True)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(artifacts))
    root = Path(os.path.normpath(artifacts / out))
    for sweep in ((), ("--sweep", "prompt=naive,encouraged")):
        status, _, error = evaluate(cli, inputs, "--out", root, "--campaign-id", "c", *sweep, "--slurm", "example")
        assert status == 2 and "the launch's records" in error, error
        assert sorted(path.relative_to(tmp_path) for path in (tmp_path / "outer").rglob("*")) == [
            Path("outer") / "artifacts"]


def test_a_submission_without_an_artifact_root_is_refused_before_any_root_is_registered(cli, inputs, monkeypatch,
                                                                                       no_submission):
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT")
    status, _, error = evaluate(cli, inputs, "--out", "runs/s", "--campaign-id", "s", "--sweep",
                                "prompt=naive,encouraged", "--slurm", "example")
    assert status == 2 and "set RULEFORGE_ARTIFACT_ROOT, or pass --dry-run" in error
    assert not list(Path.cwd().iterdir())
