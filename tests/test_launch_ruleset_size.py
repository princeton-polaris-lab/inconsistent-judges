from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from ruleforge.cli.app import build_parser
from ruleforge.cli.evaluate import bank_config
from ruleforge.launch import cli as launch_cli
from ruleforge.launch.environment import launcher_setup
from ruleforge.launch.jobs import TIMEOUT_REQUEUES
from tests.conftest import requires_git_worktree

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = yaml.safe_load((REPO / "configs" / "hydra" / "launcher" / "submitit_slurm_gpu.yaml").read_text())
OPEN_MODELS = ("gemma4-31b-tp-2", "gpt-oss-tp-2", "olmo3-32b-tp-2")
#: Every job's command up to the bank driver's arguments.
BANK_RUN = ["python", "-m", "ruleforge", "bank", "run", "--config", "configs/sampling_ruleset_size.yaml", "--"]
#: The open models' seed shares (the launcher's JOBS_PER_MODEL=2, round robin).
SHARES = ("[0,2,4,6,8]", "[1,3,5,7,9]")


def _render(spec: str, variables: dict[str, str], extra: list[str], capsys, options: tuple[str, ...] = ()
            ) -> list[dict]:
    argv = ["job", "--spec", spec, *options, *(f"--var={name}={value}" for name, value in variables.items()),
            "--dry-run", "--json", "--allow-dirty", *(["--", *extra] if extra else [])]
    assert launch_cli.main(argv) == 0
    return json.loads(capsys.readouterr().out)["jobs"]


def _overrides(job: dict) -> dict[str, str]:
    assert job["argv"][:len(BANK_RUN)] == BANK_RUN and job["request"]["cwd"] == str(REPO), job["argv"]
    return dict(item.split("=", 1) for item in job["argv"][len(BANK_RUN):])


@pytest.fixture
def environ(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("RULEFORGE_BATCH_ROOT", raising=False)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(root))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("RULEFORGE_SLURM_PARTITION", "gpu")
    monkeypatch.setenv("RULEFORGE_SLURM_ACCOUNT", "acct")
    return root


@requires_git_worktree
@pytest.mark.parametrize("variables, options, extra, expected", [
    ({"campaign_id": "fig2-2026-09"}, (), [],
     {"gpus": 1, "qos": "normal", "timeout_min": 10000, "campaign_id": "fig2-2026-09", "driver": {}}),
    ({"gpus": "2", "campaign_id": "ruleset-size-test"}, ("--qos", "compact", "--timeout-min", "600"),
     ["campaign_root=ROOT", "execution.pack_draws=true", "execution.pack_max_prompts=64",
      "sampling.max_semantic_attempts=30"],
     {"gpus": 2, "qos": "compact", "timeout_min": 600, "campaign_id": "ruleset-size-test",
      "driver": {"campaign_root": "ROOT", "execution.pack_draws": "true", "execution.pack_max_prompts": "64",
                 "sampling.max_semantic_attempts": "30"}}),
], ids=["defaults", "knobs"])
def test_the_open_models_spec_renders_the_jobs_its_launcher_submitted(variables, options, extra, expected, environ,
                                                                      capsys):
    root = str(environ / "sampling" / "ruleset-size-test-root")
    extra = [item.replace("ROOT", root) for item in extra]
    driver = {key: value.replace("ROOT", root) for key, value in expected["driver"].items()}
    gpus = expected["gpus"]
    rendered, launched = {}, {}
    for share in SHARES:
        for job in _render("ruleset_size_open_models", {**variables, "seeds": share}, extra, capsys, options):
            request, resources = job["request"], job["resources"]
            assert request["submitit"]["slurm_setup"] == launcher_setup(REPO)
            assert request["submitit"]["slurm_partition"] == "gpu" and request["max_num_timeout"] == TIMEOUT_REQUEUES
            assert (resources["gpus"], resources["qos"], resources["timeout_min"], resources["resumable"],
                    resources["cpus"], resources["mem_gb"]) == (
                gpus, expected["qos"], expected["timeout_min"], True, LAUNCHER["cpus_per_task"],
                int(LAUNCHER["mem_per_gpu"].rstrip("G")) * gpus)
            overrides = _overrides(job)
            rendered[(overrides["model@judge_model"], overrides["ruleset_size.jobs_seeds"])] = overrides
    for model in OPEN_MODELS:
        for share in SHARES:
            launched[(model, share)] = {"campaign_id": expected["campaign_id"], "judge_model.tp_size": str(gpus),
                                        "ruleset_size.jobs_seeds": share, "model@judge_model": model,
                                        "model@agent_model": model, **driver}
    assert rendered == launched


@requires_git_worktree
def test_the_open_models_preflight_is_the_spec_s_job_with_its_preflight_flag(environ, capsys):
    for share in SHARES:
        [job] = _render("ruleset_size_open_models", {"campaign_id": "fig2-2026-09", "models": "gpt-oss-tp-2",
                                                     "seeds": share},
                        ["ruleset_size.preflight_only=true"], capsys)
        assert _overrides(job) == {"campaign_id": "fig2-2026-09", "judge_model.tp_size": "1",
                                   "ruleset_size.jobs_seeds": share, "ruleset_size.preflight_only": "true",
                                   "model@judge_model": "gpt-oss-tp-2", "model@agent_model": "gpt-oss-tp-2"}


@requires_git_worktree
@pytest.mark.parametrize("variables, extra, driver", [
    ({"campaign_id": "fig2-2026-09-luna"}, [], {}),
    ({"campaign_id": "ruleset-size-haiku", "model": "claude-haiku-5-5-batch"},
     ["sampling.retry_scope=component", "sampling.max_semantic_attempts=25", "decoder_draw_ids=[0,1,2]"],
     {"sampling.retry_scope": "component", "sampling.max_semantic_attempts": "25", "decoder_draw_ids": "[0,1,2]"}),
], ids=["defaults", "knobs"])
def test_the_batch_judge_spec_renders_the_pollers_its_launcher_started(variables, extra, driver, environ, capsys):
    model = variables.get("model", "gpt-5.6-luna-batch")
    campaign_id = variables["campaign_id"]
    rendered = {}
    for shares, bank in (("[0],[3]", "false"), ("[]", "true")):
        for job in _render("ruleset_size_batch_judge", {**variables, "shares": shares, "run_bank": bank}, extra, capsys):
            assert job["resources"]["gpus"] == 0
            overrides = _overrides(job)
            rendered[overrides.pop("ruleset_size.jobs_seeds")] = overrides
    assert rendered == {share: {"campaign_id": campaign_id, "model@judge_model": model, "model@agent_model": model,
                                "ruleset_size.run_bank": bank, **driver}
                        for share, bank in (("[0]", "false"), ("[3]", "false"), ("[]", "true"))}


def _driver_config(overrides: list[str]) -> dict:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name="sampling_ruleset_size", overrides=overrides), resolve=True)
    cfg["rule_set_path"] = str((REPO / cfg["rule_set_path"]).resolve())
    for key in ("subset_root", "subset_manifest_dir"):
        cfg["ruleset_size"][key] = str((REPO / cfg["ruleset_size"][key]).resolve())
    return cfg


@requires_git_worktree
@pytest.mark.parametrize("spec, variables", [
    ("ruleset_size_open_models", {"campaign_id": "fig2-2026-09", "seeds": "[1,3,5,7,9]"}),
    ("ruleset_size_batch_judge", {"campaign_id": "fig2-2026-09-luna", "shares": "[]", "run_bank": "true"})])
def test_each_job_s_command_composes_the_config_its_driver_composed(spec, variables, environ, capsys, monkeypatch):
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(environ / "batch"))
    for job in _render(spec, variables, [], capsys):
        args = build_parser().parse_args(job["argv"][3:len(BANK_RUN) - 1])  # main() cuts at --
        args.overrides = job["argv"][len(BANK_RUN):]
        composed = OmegaConf.to_container(bank_config(args, Path(job["request"]["cwd"])), resolve=True)
        assert composed == _driver_config(args.overrides)


@pytest.mark.parametrize("spec, variables", [("ruleset_size_open_models", {"seeds": "[0]"}),
                                             ("ruleset_size_batch_judge", {"shares": "[0]"})])
def test_a_ruleset_size_launch_names_its_campaign_id(spec, variables, environ, capsys):
    argv = ["job", "--spec", spec, *(f"--var={name}={value}" for name, value in variables.items()), "--dry-run",
            "--allow-dirty"]
    assert launch_cli.main(argv) == 2
    assert "needs --var for campaign_id" in capsys.readouterr().err
