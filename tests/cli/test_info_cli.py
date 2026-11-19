from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

import ruleforge.judges.base as judges_base
from ruleforge.cli import app
from ruleforge.cli.common import CliError
from ruleforge.cli.judging import judge_node
from ruleforge.env import package_path
from ruleforge.judges import SHORTHANDS, JudgeError, JudgeSpec
from ruleforge.tasks import register_task
from ruleforge.tasks import registry as task_registry
from tests.conftest import requires_git_worktree

CHECKOUT_MODELS = Path(__file__).resolve().parents[2] / "configs" / "model"
PACKAGED_MODELS = package_path("configs") / "model"


def _written(node) -> dict:
    return OmegaConf.to_container(node, resolve=False)


def _listed(out: str) -> list[str]:
    configs = out.partition("Shorthands")[0].splitlines()
    assert configs[0] == "Model configs (--judge NAME):"
    return [line.strip() for line in configs[1:]]


@pytest.fixture
def checkout_models(tmp_path, monkeypatch):
    """A checkout's own ``configs/model``, ahead of the package's, with an added config and a packaged name's copy."""
    place = tmp_path / "checkout" / "configs" / "model"
    place.mkdir(parents=True)
    luna = (PACKAGED_MODELS / "gpt-5.6-luna.yaml").read_text(encoding="utf-8")
    (place / "mine.yaml").write_text(luna.replace('model_name: "gpt-5.6-luna"', 'model_name: "mine"'),
                                     encoding="utf-8")
    (place / "qwen3-8b.yaml").write_text(luna.replace('model_name: "gpt-5.6-luna"', 'model_name: "own-qwen"'),
                                         encoding="utf-8")
    monkeypatch.setattr(judges_base, "MODEL_CONFIG_DIRS", (place, PACKAGED_MODELS))
    return place


def test_judges_list_names_every_model_config_a_bare_name_resolves(cli):
    status, out, _ = cli("judges", "list")
    names = _listed(out)
    expected = sorted({path.stem for place in (CHECKOUT_MODELS, PACKAGED_MODELS) for path in place.glob("*.yaml")})
    assert status == 0 and names == expected
    assert "qwen3-8b" in names and "gemma4-31b-tp-2" in names


def test_judges_list_shows_every_shorthand_with_what_it_names(cli):
    _, out, _ = cli("judges", "list")
    shorthands = out.partition("Shorthands (--judge FORM):\n")[2].splitlines()
    assert [line.split()[0] for line in shorthands[:-1]] == list(SHORTHANDS)
    assert all(meaning in out for meaning in SHORTHANDS.values())
    assert "openai_compatible:MODEL@URL" in out and shorthands[-1].endswith("--judge path/to/judge.yaml")


def test_a_bare_judge_name_is_the_checkout_s_model_config_first_then_the_package_s():
    for name in ("gemma4-31b-tp-2", "claude-haiku-5-5-batch"):
        packaged = _written(OmegaConf.load(PACKAGED_MODELS / f"{name}.yaml"))
        assert _written(judge_node(name, cwd=Path.cwd())) == packaged, name
        assert JudgeSpec.from_config(name) == JudgeSpec.from_config(PACKAGED_MODELS / f"{name}.yaml"), name


def test_a_model_config_added_to_the_checkout_is_a_judge_name_everywhere(cli, checkout_models):
    status, out, _ = cli("judges", "list")
    assert status == 0 and "mine" in _listed(out) and "gpt-oss-tp-2" in _listed(out)
    status, shown, _ = cli("judges", "show", "mine")
    assert status == 0 and shown["model_name"] == "mine"
    assert JudgeSpec.from_config("mine").model_name == "mine"
    assert judge_node("qwen3-8b", cwd=Path.cwd()).model_name == "own-qwen"
    assert JudgeSpec.from_config("qwen3-8b").model_name == "own-qwen"
    packaged = _written(OmegaConf.load(PACKAGED_MODELS / "gpt-oss-tp-2.yaml"))
    assert _written(judge_node("gpt-oss-tp-2", cwd=Path.cwd())) == packaged
    assert JudgeSpec.from_config("gpt-oss-tp-2") == JudgeSpec.from_config(PACKAGED_MODELS / "gpt-oss-tp-2.yaml")


def test_an_installed_package_resolves_a_bare_name_from_its_own_configs(cli, monkeypatch):
    monkeypatch.setattr(judges_base, "MODEL_CONFIG_DIRS", (PACKAGED_MODELS,))
    model_path = _written(judge_node("gemma4-31b-tp-2", cwd=Path.cwd()))["model_path"]
    assert model_path == "${rf.env:MODEL_ROOT}/gemma-4-31B-it"
    assert JudgeSpec.from_config("gemma4-31b-tp-2") == JudgeSpec.from_config(PACKAGED_MODELS / "gemma4-31b-tp-2.yaml")
    with pytest.raises(CliError, match="no model config named 'only-in-a-checkout'"):
        judge_node("only-in-a-checkout", cwd=Path.cwd())
    with pytest.raises(JudgeError, match="no model config named 'only-in-a-checkout'"):
        JudgeSpec.from_config("only-in-a-checkout")
    status, out, _ = cli("judges", "list")
    assert status == 0 and _listed(out) == sorted(path.stem for path in PACKAGED_MODELS.glob("*.yaml"))


ROOT_NAMES = ("RULEFORGE_MODEL_ROOT", "RULEFORGE_BATCH_ROOT")


@pytest.mark.parametrize("names", [{"RULEFORGE_MODEL_ROOT": "/new", "RULEFORGE_BATCH_ROOT": "/b-new"}, {}],
                         ids=["set", "neither"])
def test_a_bare_name_reads_its_roots_by_the_library_s_rule_wherever_its_file_lies(cli, monkeypatch, tmp_path, names):
    assert judges_base.MODEL_CONFIG_DIRS == (CHECKOUT_MODELS, PACKAGED_MODELS)
    for name in ROOT_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in names.items():
        monkeypatch.setenv(name, value)
    model_root = names.get("RULEFORGE_MODEL_ROOT")
    batch_root = names.get("RULEFORGE_BATCH_ROOT")
    added = tmp_path / "checkout" / "configs" / "model"
    added.mkdir(parents=True)
    (added / "mine.yaml").write_text('model_type: local\nmodel_name: mine\n'
                                     'model_path: "${oc.env:RULEFORGE_MODEL_ROOT}/Mine"\n', encoding="utf-8")
    for dirs in ((CHECKOUT_MODELS, PACKAGED_MODELS), (PACKAGED_MODELS,), (added, CHECKOUT_MODELS, PACKAGED_MODELS)):
        monkeypatch.setattr(judges_base, "MODEL_CONFIG_DIRS", dirs)
        for name, weights in (("qwen3-8b", "Qwen3-8B"), ("mine", "Mine")):
            if name == "mine" and added not in dirs:
                continue
            expected = None if model_root is None else f"{model_root}/{weights}"
            status, shown, error = cli("judges", "show", name)
            assert status == 0, error
            if expected is None:
                assert shown["options"]["model_path"] == f"${{rf.env:MODEL_ROOT}}/{weights}"
                with pytest.raises(OmegaConfBaseException, match="MODEL_ROOT"):
                    JudgeSpec.from_config(name).options.model_path
            else:
                assert shown["options"]["model_path"] == expected, (dirs, name)
                assert JudgeSpec.from_config(name).options.model_path == expected
        assert JudgeSpec.from_config("gpt-5.6-luna-batch").options.batch_root == batch_root
        assert judge_node("gpt-5.6-luna-batch", cwd=tmp_path).batch_root == batch_root
        status, shown, error = cli("judges", "show", "gpt-5.6-luna-batch")
        assert status == 0 and shown["options"]["batch_root"] == batch_root, error


def test_a_model_config_given_by_path_reads_its_roots_as_the_named_one_does(monkeypatch):
    assert _written(judge_node(str(CHECKOUT_MODELS / "qwen3-8b.yaml"), cwd=Path.cwd()))["model_path"] == (
        "${rf.env:MODEL_ROOT}/Qwen3-8B")
    assert JudgeSpec.from_config(CHECKOUT_MODELS / "gpt-5.6-luna-batch.yaml").options.written() == (
        JudgeSpec.from_config("gpt-5.6-luna-batch").options.written())


def test_judges_show_and_check_describe_a_judge(cli, monkeypatch):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    status, shown, _ = cli("judges", "show", "qwen3-8b")
    assert status == 0 and shown["options"]["model_path"] == "/models/Qwen3-8B"
    assert shown["capabilities"]["local_engine"] is True
    status, checked, _ = cli("judges", "check", "fake:4")
    assert status == 0 and checked["ok"] is True and checked["identity"]["model_name"] == "fake-4"
    assert checked["probe"]["text"] in ("yes", "no", "unsure")
    status, unprobed, _ = cli("judges", "check", "fake", "--no-probe")
    assert status == 0 and unprobed["probe"] is None


def test_a_judge_that_does_not_answer_fails_its_check(cli, monkeypatch):
    import dataclasses

    import ruleforge.judges as judges

    real = judges.check

    def failing(judge, probe=None):
        report = real(judge, probe=probe)
        down = judges.GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="down")
        return dataclasses.replace(report, probe=down)

    monkeypatch.setattr(judges, "check", failing)
    status, checked, _ = cli("judges", "check", "fake")
    assert status == 1 and checked["ok"] is False and checked["probe"]["error"] == "down"


def test_a_batch_judge_s_probe_is_refused_where_its_state_would_land_in_a_campaign_root(cli, monkeypatch, tmp_path):
    published = tmp_path / "published"
    published.mkdir()
    (published / "sampling_manifest.json").write_text("{}")
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(published / "batch"))
    status, _, error = cli("judges", "check", "claude-haiku-5-5-batch", "--probe")
    assert status == 2 and "batch_root" in error and "is inside the campaign root" in error
    assert not (published / "batch").exists()
    status, checked, _ = cli("judges", "check", "claude-haiku-5-5-batch")
    assert status == 0 and checked["probe"] is None


def test_a_batch_judge_s_batch_root_is_made_absolute_where_it_resolves(monkeypatch, tmp_path):
    from ruleforge.cli.judging import absolute_batch_root
    from ruleforge.judges import JudgeSpec

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    fake = JudgeSpec.fake()
    assert absolute_batch_root(fake) is fake
    for kept in (None, "${rf.env:NO_SUCH_BATCH_ROOT}", str(tmp_path / "placed")):
        spec = JudgeSpec.from_config("claude-haiku-5-5-batch", batch_root=kept)
        assert absolute_batch_root(spec) is spec
    for given, place in (("~/b", str(tmp_path / "home" / "b")), ("rel/../b", f"{tmp_path}/rel/../b")):
        spec = absolute_batch_root(JudgeSpec.from_config("claude-haiku-5-5-batch", batch_root=given))
        assert spec.options.batch_root == spec.options.config().batch_root == place


@pytest.mark.parametrize("given", ["model config", "environment"])
def test_a_batch_root_spelled_with_a_tilde_is_one_place_to_the_checks_and_the_transport(cli, monkeypatch, tmp_path,
                                                                                       given):
    import ruleforge.judges.anthropic_batch as transport
    from ruleforge.judges import batch_api

    published = tmp_path / "published"
    published.mkdir()
    (published / "sampling_manifest.json").write_text("{}")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(published)
    packaged = package_path("configs/model/claude-haiku-5-5-batch.yaml")
    judge = "claude-haiku-5-5-batch"
    if given == "model config":
        judge = tmp_path / "judge.yaml"
        judge.write_text(packaged.read_text().replace("${rf.env:BATCH_ROOT,null}", "~/batches"))
    else:
        monkeypatch.setenv("RULEFORGE_BATCH_ROOT", "~/batches")
    derived = []

    def block_generate(*, root, **kwargs):
        def generate(prompts, *args):
            derived.append(batch_api.work_directory(root, "probe"))
            return [{"raw_response": "yes", "completion": "yes", "finish_reason": "stop"} for _ in prompts]

        return generate

    monkeypatch.setattr(transport, "make_batch_generate", lambda **kwargs: (None, None))
    monkeypatch.setattr(transport, "make_block_generate", block_generate)
    status, checked, _ = cli("judges", "check", judge, "--probe", "--yes-spend")
    assert status == 0 and checked["ok"] is True and derived == [home / "batches" / "probe"]
    monkeypatch.setenv("HOME", str(published / "home"))
    status, _, error = cli("judges", "check", judge, "--probe")
    assert status == 2 and "batch_root" in error and "is inside the campaign root" in error
    assert len(derived) == 1 and sorted(path.name for path in published.iterdir()) == ["sampling_manifest.json"]


def test_an_unknown_judge_is_refused_naming_the_packaged_ones(cli):
    status, _, error = cli("judges", "show", "no-such-model")
    assert status == 2 and "qwen3-8b" in error and "openai_compatible:MODEL@URL" in error
    status, _, error = cli("judges", "show", "nowhere/judge.yaml")
    assert status == 2 and "not a model config" in error
    status, _, error = cli("judges", "show", "openai:")
    assert status == 2


def test_tasks_list_lists_every_task_and_fails_on_a_broken_one(cli, monkeypatch):
    status, found, _ = cli("tasks", "list")
    assert status == 0 and [task["name"] for task in found][:2] == ["modelspec", "legal"]
    assert found[0]["styles"]["naive"] == "label"
    monkeypatch.setattr(task_registry, "_REGISTERED", {})
    from tests.test_tasks import DemoTask

    class Broken(DemoTask):
        name = "broken"

        def __init__(self):
            raise RuntimeError("needs a config")

    register_task("broken", Broken)
    status, found, _ = cli("tasks", "list")
    assert status == 1 and found[-1]["name"] == "broken" and found[-1]["error"]


@requires_git_worktree  # the code digest is taken over git's file list
def test_version_names_the_code(cli):
    status, out, _ = cli("version")
    assert status == 0 and out["code"]["code_state_sha256"] and "version" in out


def test_overrides_are_refused_where_a_command_takes_none(capsys):
    with pytest.raises(SystemExit):
        app.main(["version", "--", "x=1"])
    assert "version takes no `-- key=value` overrides" in capsys.readouterr().err


@pytest.mark.parametrize("module", ["ruleforge", "ruleforge.cli"])
def test_the_command_runs_as_a_module(module, tmp_path):
    result = subprocess.run([sys.executable, "-m", module, "judges", "list"], capture_output=True, text=True,
                            cwd=tmp_path, check=False)
    assert result.returncode == 0 and "qwen3-8b" in result.stdout.split()


def test_every_command_has_help(capsys):
    parser = app.build_parser()
    for argv in (["evaluate"], ["campaign", "run"], ["analyze"], ["optimize"], ["grpo", "rows"], ["grpo", "train"],
                 ["grpo", "anchors", "build"], ["judges", "check"], ["tasks", "list"]):
        with pytest.raises(SystemExit):
            parser.parse_args([*argv, "--help"])
        assert "usage: ruleforge" in capsys.readouterr().out


def test_a_batch_judge_s_probe_asks_for_the_flag_and_runs_with_it(cli, monkeypatch, tmp_path):
    import ruleforge.judges.anthropic_batch as transport
    from ruleforge.judges import batch_api

    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batches"))
    confirmed = []

    def block_generate(*, root, **kwargs):
        def generate(prompts, *args):
            confirmed.append(batch_api.spend_is_confirmed())
            return [{"raw_response": "yes", "completion": "yes", "finish_reason": "stop"} for _ in prompts]

        return generate

    monkeypatch.setattr(transport, "make_batch_generate", lambda **kwargs: (None, None))
    monkeypatch.setattr(transport, "make_block_generate", block_generate)
    status, out, error = cli("judges", "check", "claude-haiku-5-5-batch", "--probe")
    assert status == 2 and out == "" and "BatchSpendNotConfirmed" in error and "--yes-spend" in error and not confirmed
    status, checked, _ = cli("judges", "check", "claude-haiku-5-5-batch", "--probe", "--yes-spend")
    assert status == 0 and checked["ok"] is True and confirmed == [True]


def test_a_paid_judge_without_a_batch_refuses_its_probe_the_same_way(cli, monkeypatch):
    monkeypatch.setattr(JudgeSpec, "start", lambda self: pytest.fail("started before the spend was confirmed"))
    status, _, error = cli("judges", "check", "gpt-5.6-luna", "--probe")
    assert status == 2 and "BatchSpendNotConfirmed" in error and "--yes-spend" in error
