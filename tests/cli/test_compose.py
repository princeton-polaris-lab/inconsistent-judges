from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import ruleforge.grpo.anchor_set as anchors_module
import ruleforge.grpo as grpo_package
import ruleforge.judges.base as judges_base
from ruleforge.cli import compose as compose_module
from ruleforge.cli import evaluate as evaluate_cli
from ruleforge.cli import optimize as optimize_cli
from ruleforge.cli.common import Assignment, CliError, overlap, quoted, setting
from ruleforge.cli.compose import Rebasing, compose, parse_overrides, refuse_conflicts
from ruleforge.cli.schema import PATH, Operation
from ruleforge.env import package_path

CHECKOUT_MODELS = Path(__file__).resolve().parents[2] / "configs" / "model"


class Stop(Exception):
    """Ends a command once what it is about to call has been seen."""


def capture(monkeypatch, module, name):
    """Replace ``module.name`` by a recorder of its keyword arguments that ends the command."""
    real = getattr(module, name)
    seen = []

    @functools.wraps(real)
    def recorder(*args, **kwargs):
        seen.append(kwargs)
        raise Stop

    monkeypatch.setattr(module, name, recorder)
    return seen


def captured_config(monkeypatch):
    seen = []
    real = evaluate_cli.convert

    def recorder(cfg):
        seen.append(cfg)
        raise Stop

    monkeypatch.setattr(evaluate_cli, "convert", recorder)
    assert real is not recorder
    return seen


@pytest.fixture
def layout(tmp_path):
    """A config directory (``conf``), a data directory (``data``) and the working directory (``work``), apart."""
    conf, data = tmp_path / "conf", tmp_path / "data"
    (conf / "model").mkdir(parents=True)
    data.mkdir()
    return conf, data, Path.cwd()


def test_evaluate_config_composes_a_model_group_file_by_relative_path_from_a_third_directory(layout, cli, monkeypatch):
    conf, data, work = layout
    (conf / "run.yaml").write_text("defaults:\n  - override /model@judge_model: mine\nrule_set_path: rules.jsonl\n"
                                   "sampling:\n  num_decoder_draws: 3\n")
    (conf / "model" / "mine.yaml").write_text("model_type: local\nmodel_name: mine\nmodel_path: weights/mine\n"
                                              "adapter_path: ../adapters/a\nbatch_root: null\n")
    seen = captured_config(monkeypatch)
    with pytest.raises(Stop):
        cli("evaluate", "--config", "../conf/run.yaml", "--scenarios", "../data/scen.jsonl")
    cfg = seen[0]
    assert cfg.rule_set_path == str(conf / "rules.jsonl")
    assert cfg.scenario_set_path == str(data / "scen.jsonl")
    assert cfg.judge_model.model_path == str(conf / "model" / "weights" / "mine")
    assert cfg.judge_model.adapter_path == str(conf / "adapters" / "a")
    assert cfg.sampling.num_decoder_draws == 3 and cfg.campaign_root is None


#: A bare name's two trees: this checkout's lookup (its ``configs/model`` first), and an installed package's.
TREES = pytest.mark.parametrize("installed", [False, True], ids=["checkout", "installed"])


def _tree(monkeypatch, installed: bool) -> None:
    if installed:  # no checkout's configs/model ahead of the package's
        monkeypatch.setattr(judges_base, "MODEL_CONFIG_DIRS", (package_path("configs") / "model",))
    else:
        assert judges_base.MODEL_CONFIG_DIRS == (CHECKOUT_MODELS, package_path("configs") / "model")


@TREES
@pytest.mark.parametrize("variable", ["RULEFORGE_MODEL_ROOT"])
def test_a_packaged_model_config_reads_its_root_under_either_name(layout, cli, monkeypatch, variable, installed):
    _tree(monkeypatch, installed)
    monkeypatch.setenv(variable, "/models")
    seen = captured_config(monkeypatch)
    with pytest.raises(Stop):
        cli("evaluate", "--judge", "qwen3-8b", "--rules", "r.jsonl", "--scenarios", "s.jsonl")
    assert seen[0].judge_model.model_path == "/models/Qwen3-8B"
    assert seen[0].rule_set_path == str(layout[2] / "r.jsonl")


def _recorded(monkeypatch, command):
    """``command``'s composed configs, each recorded as it is converted, the command ended there."""
    module = evaluate_cli if command == "evaluate" else optimize_cli
    seen = []

    def recorder(cfg):
        seen.append(cfg)
        raise Stop

    monkeypatch.setattr(module, "convert", recorder)
    if command == "evaluate":
        return seen, ("--rules", "r.jsonl", "--scenarios", "s.jsonl")
    return seen, ("--rules", "r.jsonl", "--seeds-good", "g.jsonl", "--seeds-bad", "b.jsonl", "--out", "o")


@pytest.mark.parametrize("command", ["evaluate", "optimize"])
def test_a_composed_judge_name_is_the_checkout_s_model_config_first_then_the_package_s(layout, cli, monkeypatch,
                                                                                         tmp_path, command):
    checkout = tmp_path / "checkout" / "configs" / "model"
    checkout.mkdir(parents=True)
    (checkout / "mine.yaml").write_text("model_type: local\nmodel_name: mine\nmodel_path: weights/mine\n")
    (checkout / "qwen3-8b.yaml").write_text("model_type: local\nmodel_name: own-qwen\nmodel_path: /w/q\n")
    monkeypatch.setattr(judges_base, "MODEL_CONFIG_DIRS", (checkout, package_path("configs") / "model"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    seen, argv = _recorded(monkeypatch, command)
    for name, model_name, model_path in (("mine", "mine", str(checkout / "weights" / "mine")),
                                         ("qwen3-8b", "own-qwen", "/w/q"),
                                         ("gpt-oss-tp-2", "gpt-oss-120b", "/models/gpt-oss-120b")):
        with pytest.raises(Stop):
            cli(command, "--judge", name, *argv)
        judge = seen.pop().judge_model
        assert (judge.model_name, judge.model_path) == (model_name, model_path), name
    conf = layout[0]
    (conf / "model" / "qwen3-8b.yaml").write_text("model_type: local\nmodel_name: config-qwen\n"
                                                  "model_path: ${oc.env:RULEFORGE_MODEL_ROOT}/c\n")
    (conf / "run.yaml").write_text("{}\n")
    with pytest.raises(Stop):
        cli(command, "--config", "../conf/run.yaml", "--judge", "qwen3-8b", *argv)
    judge = seen.pop().judge_model
    assert judge.model_name == "config-qwen"
    assert OmegaConf.to_container(judge, resolve=False)["model_path"] == "${rf.env:MODEL_ROOT}/c"


@pytest.mark.parametrize("command", ["evaluate", "optimize"])
def test_the_checkout_s_own_model_config_composes_by_name(cli, monkeypatch, command):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    seen, argv = _recorded(monkeypatch, command)
    with pytest.raises(Stop):
        cli(command, "--judge", "gemma4-31b-tp-2", *argv)
    written = judges_base.as_packaged(OmegaConf.load(CHECKOUT_MODELS / "gemma4-31b-tp-2.yaml"))
    assert OmegaConf.to_container(seen[0].judge_model, resolve=False) == OmegaConf.to_container(written,
                                                                                                resolve=False)
    assert seen[0].judge_model.model_path == "/models/gemma-4-31B-it"


@TREES
@pytest.mark.parametrize("command", ["evaluate", "optimize"])
def test_a_composed_bare_name_reads_its_roots_by_the_library_s_rule(cli, monkeypatch, command, installed):
    _tree(monkeypatch, installed)
    seen, argv = _recorded(monkeypatch, command)
    for name in ("RULEFORGE_MODEL_ROOT", "RULEFORGE_BATCH_ROOT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(Stop):
        cli(command, "--judge", "claude-haiku-5-5-batch", *argv)
    assert seen.pop().judge_model.batch_root is None
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/new")
    with pytest.raises(Stop):
        cli(command, "--judge", "qwen3-8b", *argv)
    assert seen.pop().judge_model.model_path == "/new/Qwen3-8B"


@pytest.mark.parametrize("command", ["evaluate", "optimize"])
def test_a_model_config_reads_its_roots_alike_by_name_by_path_and_beside_a_config(layout, cli, monkeypatch, command):
    """A byte copy of a checkout model config, given by its path or in model/ beside --config, is the named one."""
    from ruleforge.judges import JudgeSpec

    conf = layout[0]
    seen, argv = _recorded(monkeypatch, command)
    for name in ("RULEFORGE_MODEL_ROOT", "RULEFORGE_BATCH_ROOT"):
        monkeypatch.delenv(name, raising=False)
    for name in ("claude-haiku-5-5-batch", "qwen3-8b"):
        (conf / "model" / f"my-{name}.yaml").write_bytes((CHECKOUT_MODELS / f"{name}.yaml").read_bytes())
    (conf / "run.yaml").write_text("defaults:\n  - override /model@judge_model: my-claude-haiku-5-5-batch\n")
    for given in (["--judge", "claude-haiku-5-5-batch"], ["--judge", conf / "model" / "my-claude-haiku-5-5-batch.yaml"],
                  ["--config", conf / "run.yaml"]):
        with pytest.raises(Stop):
            cli(command, *given, *argv)
        assert seen.pop().judge_model.batch_root is None, given
    assert JudgeSpec.from_config(conf / "model" / "my-claude-haiku-5-5-batch.yaml").options.batch_root is None
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    with pytest.raises(Stop):
        cli(command, "--judge", conf / "model" / "my-qwen3-8b.yaml", *argv)
    assert seen.pop().judge_model.model_path == "/models/Qwen3-8B"
    assert JudgeSpec.from_config(conf / "model" / "my-qwen3-8b.yaml").options.model_path == "/models/Qwen3-8B"


def test_a_judge_file_given_by_flag_reads_its_paths_from_its_own_directory(layout, cli, monkeypatch):
    conf, _, _ = layout
    (conf / "model" / "local.yaml").write_text("model_type: local\nmodel_path: w\n")
    seen = captured_config(monkeypatch)
    with pytest.raises(Stop):
        cli("evaluate", "--judge", "../conf/model/local.yaml", "--rules", "r", "--scenarios", "s")
    assert seen[0].judge_model.model_path == str(conf / "model" / "w")


@TREES
@pytest.mark.parametrize("variable", ["RULEFORGE_BATCH_ROOT"])
def test_optimize_config_reads_its_ruleset_and_seed_files_from_its_directory(layout, cli, monkeypatch, variable,
                                                                             installed):
    conf, data, work = layout
    _tree(monkeypatch, installed)
    monkeypatch.setenv(variable, str(data / "batch"))
    (conf / "opt.yaml").write_text(
        "rule_amendment:\n  initial_rule_set_path: rules.jsonl\nscenario_generation:\n"
        "  good_seed_scenario_path: seeds/good.jsonl\n  bad_seed_scenario_path: seeds/bad.jsonl\nout: runs/one\n")
    import ruleforge.optimize.api as optimize_api

    seen = capture(monkeypatch, optimize_api, "optimize")
    with pytest.raises(Stop):
        cli("optimize", "--config", "../conf/opt.yaml", "--judge", "claude-haiku-5-5-batch")
    call = seen[0]
    assert call["out"] == conf / "runs" / "one"
    assert call["seeds"] == {"good": conf / "seeds" / "good.jsonl", "bad": conf / "seeds" / "bad.jsonl"}
    assert call["judge"].options.batch_root == str(data / "batch") and call["agent"] is None
    assert seen and (conf / "runs" / "one.runs").is_dir()


def test_grpo_rows_takes_relative_protected_files_from_the_working_directory_and_the_config(layout, cli,
                                                                                            monkeypatch):
    conf, _, work = layout
    (conf / "rows.yaml").write_text("protected: [dev.jsonl]\nrules: rules.jsonl\n")
    seen = capture(monkeypatch, grpo_package, "rows")
    with pytest.raises(Stop):
        cli("grpo", "rows", "--config", "../conf/rows.yaml", "--anchors", "a1", "a2", "--scenarios", "train.jsonl",
            "--model", "/weights/Qwen3", "--out", "rows", "--require-protected", "test.jsonl", "eval/test2.jsonl")
    call = seen[0]
    assert call["anchors"] == [str(work / "a1"), str(work / "a2")]
    assert call["protected"] == [str(conf / "dev.jsonl")] and call["rules"] == str(conf / "rules.jsonl")
    assert call["require_protected"] == [str(work / "test.jsonl"), str(work / "eval" / "test2.jsonl")]
    assert call["model"] == "/weights/Qwen3" and call["out"] == str(work / "rows") and call["pins"] == {}
    assert call["prompt_draws"] == 2400 and call["allow_no_held_out"] is False


def test_grpo_rows_overrides_are_rebased_on_the_working_directory_too(layout, cli, monkeypatch):
    seen = capture(monkeypatch, grpo_package, "rows")
    with pytest.raises(Stop):
        cli("grpo", "rows", "--anchors", "a", "--scenarios", "t", "--rules", "r", "--model", "m", "--out", "o",
            "--", "protected=[d1,sub/d2]", "+pins.test=abc", "prompt_draws=3")
    work = layout[2]
    assert seen[0]["protected"] == [str(work / "d1"), str(work / "sub" / "d2")]
    assert seen[0]["pins"] == {"test": "abc"} and seen[0]["prompt_draws"] == 3


@pytest.mark.parametrize("stage, function, extra", [
    ("build", "build_anchor_set", ["--full-set", "full", "--top-up", "top", "--side", "train"]),
    ("route", "route_top_up", ["--side", "train"]),
])
def test_grpo_anchors_takes_its_roots_relative(layout, cli, monkeypatch, stage, function, extra):
    conf, _, work = layout
    (conf / "anchors.yaml").write_text("reference: base/condition.json\npopulation: pool.jsonl\nblocks: blocks.json\n")
    seen = capture(monkeypatch, anchors_module, function)
    with pytest.raises(Stop):
        cli("grpo", "anchors", stage, "--config", "../conf/anchors.yaml", "--bank", "bank1", "bank2", "--rules",
            "rules.jsonl", "--out", "anchors", *extra)
    call = seen[0]
    assert call["bank"] == [str(work / "bank1"), str(work / "bank2")]
    assert call["reference"] == str(conf / "base" / "condition.json")
    assert call["population"] == str(conf / "pool.jsonl") and call["blocks"] == str(conf / "blocks.json")
    if stage == "build":
        assert call["full_set"] == [str(work / "full")] and call["top_up"] == [str(work / "top")]


@pytest.mark.parametrize("resume", ["runs/x/checkpoint-5", "latest"])
def test_grpo_train_preflight_takes_its_adapter_and_resume_relative(layout, cli, monkeypatch, resume):
    conf, data, work = layout
    import ruleforge.grpo.trainer as trainer

    (conf / "train.yaml").write_text("model:\n  path: ../models/base\n  init_adapter: warm/checkpoint-1\n")
    seen = []
    monkeypatch.setattr(trainer, "run_grpo", lambda cfg: seen.append(cfg))
    status, out, _ = cli("grpo", "train", "--config", "../conf/train.yaml", "--preflight-only", "--run-name", "r1",
                      "--output-root", "out", "--resume", resume, "--", "data.train=d/train.jsonl", "data.replay=d/replay.jsonl", "data.rules=rules.jsonl",
                      "data.rules_sha256=abc", "data.anchor_condition=a/condition.json", "reward.truncated=zero")
    assert status == 0 and out["preflight_only"] is True
    cfg = seen[0]
    assert cfg["model"]["path"] == str(conf.parent / "models" / "base")
    assert cfg["model"]["init_adapter"] == str(conf / "warm" / "checkpoint-1")
    assert cfg["resume_from_checkpoint"] == (resume if resume == "latest" else str(work / resume))
    assert cfg["output_dir"] == str(work / "out" / "r1") and cfg["data"]["train"] == str(work / "d" / "train.jsonl")
    assert cfg["preflight_only"] is True and cfg["grpo"]["report_to"] == []
    assert (work / "out" / "r1.runs").is_dir()


@pytest.fixture
def evaluated(layout, monkeypatch):
    conf = layout[0]
    (conf / "run.yaml").write_text("sampling:\n  num_decoder_draws: 3\n  max_semantic_attempts: 7\n")
    return captured_config(monkeypatch)


def test_a_flag_beats_the_config_which_beats_the_packaged_default(evaluated, cli):
    with pytest.raises(Stop):
        cli("evaluate", "--config", "../conf/run.yaml", "--judge", "fake", "--rules", "r", "--scenarios", "s",
            "--draws", "4")
    sampling = evaluated[0].sampling
    assert (sampling.num_decoder_draws, sampling.max_semantic_attempts, sampling.retry_scope) == (4, 7, "component")


def test_a_key_set_by_a_flag_and_an_override_is_refused_naming_both(evaluated, cli):
    status, _, error = cli("evaluate", "--judge", "fake", "--rules", "r", "--scenarios", "s", "--draws", "4", "--",
                           "sampling.num_decoder_draws=6")
    assert status == 2
    assert "--draws 4 and -- sampling.num_decoder_draws=6 both set sampling.num_decoder_draws" in error
    status, _, error = cli("evaluate", "--judge", "fake", "--rules", "r", "--scenarios", "s", "--",
                           "model@judge_model=x")
    assert status == 2 and "--judge fake and -- model@judge_model=x both set judge_model" in error


def test_a_sweep_overrides_a_config_value_but_not_an_explicit_one(evaluated, cli):
    with pytest.raises(Stop):
        cli("evaluate", "--config", "../conf/run.yaml", "--judge", "fake", "--rules", "r", "--scenarios", "s",
            "--sweep", "draws=7,8")
    assert evaluated[0].sampling.num_decoder_draws == 7
    status, _, error = cli("evaluate", "--judge", "fake", "--rules", "r", "--scenarios", "s", "--draws", "4",
                           "--sweep", "draws=7,8")
    assert status == 2
    assert "--sweep draws=... sweeps sampling.num_decoder_draws, which --draws 4 also sets" in error


def test_a_path_still_relative_once_resolved_is_refused_naming_key_value_and_file(layout, cli, monkeypatch):
    conf = layout[0]
    monkeypatch.setenv("RELATIVE_RULES", "relative/rules.jsonl")
    status, _, error = cli("evaluate", "--judge", "fake", "--scenarios", "s", "--",
                           "rule_set_path=${oc.env:RELATIVE_RULES}")
    assert status == 2
    assert "rule_set_path = 'relative/rules.jsonl' (a flag, an override or an interpolation)" in error
    (conf / "run.yaml").write_text("rule_set_path: ${oc.env:RELATIVE_RULES}\n")
    status, _, error = cli("evaluate", "--config", "../conf/run.yaml", "--judge", "fake", "--scenarios", "s")
    assert status == 2 and f"({conf / 'run.yaml'})" in error


def test_a_composition_error_names_the_operation(layout, cli):
    status, _, error = cli("evaluate", "--judge", "fake", "--rules", "r", "--scenarios", "s", "--", "nonsense=1")
    assert status == 2 and "nonsense" in error and "the evaluate config does not compose" in error
    status, _, error = cli("evaluate", "--judge", "fake", "--", "a=[unclosed")
    assert status == 2 and "-- a=[unclosed: not a Hydra override" in error


def test_a_missing_or_unknown_judge_is_refused_in_the_command_s_words(layout, cli):
    conf = layout[0]
    status, _, error = cli("evaluate", "--rules", "r", "--scenarios", "s")
    assert status == 2 and "no judge given: --judge is required" in error and "openai_compatible:MODEL@URL" in error
    assert "model@judge_model" not in error
    (conf / "model" / "mine.yaml").write_text("model_type: fake\nmodel_name: fake-1\nseed: 1\n")
    (conf / "run.yaml").write_text("rule_set_path: rules.jsonl\n")
    status, _, error = cli("evaluate", "--config", "../conf/run.yaml", "--judge", "gpt-4o", "--scenarios", "s")
    assert status == 2 and "no model config named 'gpt-4o' (there are " in error and "mine" in error
    assert "openai:MODEL" in error and "Config search path" not in error and "ruleforge-config-" not in error
    (conf / "run.yaml").write_text("defaults:\n  - absent\n  - _self_\n")
    status, _, error = cli("evaluate", "--config", "../conf/run.yaml", "--judge", "fake", "--scenarios", "s")
    assert status == 2 and "the evaluate config does not compose" in error and "absent" in error


def test_a_config_file_must_be_a_mapping_that_exists(layout, cli):
    conf = layout[0]
    status, _, error = cli("evaluate", "--config", "../conf/absent.yaml")
    assert status == 2 and "no such file" in error
    (conf / "list.yaml").write_text("- a\n- b\n")
    status, _, error = cli("evaluate", "--config", "../conf/list.yaml")
    assert status == 2 and "is not a mapping" in error
    (conf / "bad.yaml").write_text("a: [\n")
    status, _, error = cli("evaluate", "--config", "../conf/bad.yaml")
    assert status == 2 and "cannot be read as YAML" in error


def test_rebasing_rewrites_literal_relative_paths_alone(tmp_path):
    rebasing = Rebasing(patterns={"out": PATH, "files": PATH, "root.*.path": PATH, "seeds": "mapping"},
                        special={"out": ("latest",)})
    node = OmegaConf.create({"out": "latest", "files": ["a", "/b", "${x}"], "root": {"one": {"path": "p"}},
                             "seeds": {"good": "g"}, "defaults": ["x"], "other": "o", "count": 3})
    rebasing.rebase(node, tmp_path, "file.yaml")
    assert OmegaConf.to_container(node, resolve=False) == {
        "out": "latest", "files": [str(tmp_path / "a"), "/b", "${x}"], "root": {"one": {"path": str(tmp_path / "p")}},
        "seeds": {"good": str(tmp_path / "g")}, "defaults": ["x"], "other": "o", "count": 3}
    rebasing.rebase(OmegaConf.create({"out": "${y}/z"}), tmp_path, "where.yaml")
    assert rebasing.origins == {"out": "where.yaml"}


def test_a_dict_override_of_a_path_mapping_is_rebased(tmp_path):
    operation = Operation("x", (), declared={"seeds": "mapping", "out": PATH})
    [mapping, plain, deleted] = parse_overrides(["seeds={good: g, bad: /b}", "+out=o", "~other"], operation, tmp_path)
    assert mapping.override == f"seeds={{good:'{tmp_path / 'g'}',bad:'/b'}}"
    assert plain.override == f"+out='{tmp_path / 'o'}'" and deleted.override == "~other"


def test_a_path_written_with_a_tilde_is_expanded_by_the_rebase(tmp_path, monkeypatch):
    from ruleforge.cli.common import is_literal_relative, rebased

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert rebased("~/x/../y", tmp_path / "base") == str(tmp_path / "home" / "y")
    assert rebased(["~", "b", "/c", "${x}/d", "artifact://e", "latest"], tmp_path, ("latest",)) == [
        str(tmp_path / "home"), str(tmp_path / "b"), "/c", "${x}/d", "artifact://e", "latest"]
    [plain] = parse_overrides(["+out='~/o'"], Operation("x", (), declared={"out": PATH}), tmp_path)
    assert plain.override == f"+out='{tmp_path / 'home' / 'o'}'"
    assert not is_literal_relative("~/x")  # absolute once expanded: an interpolation resolving to it is no relative path


def test_conflicts_are_found_between_layers_alone():
    refuse_conflicts([setting("a", 1, "--a 1")], [setting("b", 2, "b=2")], [setting("c", 3, "c=3")])
    with pytest.raises(CliError, match="--sweep c=4 sweeps c, which b=2 also sets"):
        refuse_conflicts([], [setting("c", 2, "b=2")], [setting("c", 4, "c=4")])


def test_the_packaged_configs_name_their_roots_by_the_package_s_resolvers_alone():
    for path in sorted(package_path("configs").rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert "${oc.env:" not in text, path
        assert "../" not in json.dumps(OmegaConf.to_container(OmegaConf.load(path), resolve=False)), path


def test_compose_takes_a_node_or_a_packaged_file(tmp_path):
    operation = Operation("x", (), declared={"out": PATH})
    cfg = compose(operation, OmegaConf.create({"out": "???", "n": 1}),
                  layers=([setting("out", str(tmp_path / "o"), "--out o")],), cwd=tmp_path)
    assert cfg.out == str(tmp_path / "o") and cfg.n == 1
    assert compose_module.check_paths(cfg, operation, {}) is None


def test_a_path_pattern_walks_only_the_mappings_it_can_read(monkeypatch):
    monkeypatch.delenv("SURELY_UNSET_VARIABLE_X", raising=False)
    cfg = OmegaConf.create({"a": 1, "rules": {"one": {"path": "/p"}, "two": "${oc.env:SURELY_UNSET_VARIABLE_X}"},
                            "broken": "${oc.env:SURELY_UNSET_VARIABLE_X}"})
    assert compose_module._keys(cfg, "a.b") == [] and compose_module._keys(cfg, "missing.b") == []
    assert compose_module._keys(cfg, "rules.*.path") == ["rules.one.path"]
    assert compose_module._keys(cfg, "broken.path") == []
    operation = Operation("x", (), declared={"broken": PATH, "rules.*.path": PATH})
    assert compose_module.check_paths(cfg, operation, {}) is None


def test_a_file_that_is_no_mapping_has_nothing_rebased(tmp_path):
    node = OmegaConf.create(["a", "b"])
    Rebasing(patterns={"a": PATH}).rebase(node, tmp_path, "list.yaml")
    assert OmegaConf.to_container(node) == ["a", "b"]


@pytest.mark.parametrize("variable", ["RULEFORGE_ARTIFACT_ROOT"])
def test_grpo_train_s_default_output_root_is_the_artifact_root_s(cli, monkeypatch, tmp_path, variable):
    import ruleforge.grpo.trainer as trainer

    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT")
    monkeypatch.setenv(variable, str(tmp_path / "artifacts"))
    seen = []
    monkeypatch.setattr(trainer, "run_grpo", lambda cfg: seen.append(cfg))
    status, _, _ = cli("grpo", "train", "--run-name", "r", "--model", "/m", "--preflight-only", "--",
                       "model.init_adapter=null", "data.train=/t", "data.replay=/r", "data.rules=/x",
                       "data.rules_sha256=s", "data.anchor_condition=/c", "reward.truncated=zero")
    assert status == 0 and seen[0]["output_dir"] == str(tmp_path / "artifacts" / "training" / "r")


def test_a_mapping_override_sets_each_of_its_entries(evaluated, cli):
    status, _, error = cli("evaluate", "--judge", "fake", "--rules", "r", "--scenarios", "s", "--draws", "4", "--",
                           "sampling={num_decoder_draws:6}")
    assert status == 2
    assert "--draws 4 and -- sampling={num_decoder_draws:6} both set sampling.num_decoder_draws" in error
    status, _, error = cli("evaluate", "--judge", "fake", "--rules", "r", "--scenarios", "s", "--sweep", "draws=2,3",
                           "--", "sampling={num_decoder_draws:6}")
    assert status == 2
    assert "--sweep draws=... sweeps sampling.num_decoder_draws, which sampling={num_decoder_draws:6} also sets" in error
    status, _, error = cli("evaluate", "--judge", "fake", "--rules", "r", "--scenarios", "s", "--draws", "4", "--",
                           "sampling=null")
    assert status == 2 and "both set sampling.num_decoder_draws" in error
    status, _, error = cli("evaluate", "--judge", "fake", "--rules", "r", "--scenarios", "s", "--",
                           "sampling.num_decoder_draws=4", "sampling.num_decoder_draws=6")
    assert status == 2
    assert "-- sampling.num_decoder_draws=4 and -- sampling.num_decoder_draws=6 both set" in error
    status, _, error = cli("evaluate", "--judge", "fake", "--rules", "r", "--scenarios", "s", "--sweep",
                           "extra.x=a,b", "--", "condition_extra={x: c}")
    assert status == 2 and "sweeps condition_extra.x, which condition_extra={x: c} also sets" in error


def test_a_value_inside_a_selected_judge_refines_it(evaluated, cli):
    with pytest.raises(Stop):
        cli("evaluate", "--judge", "fake", "--rules", "r", "--scenarios", "s", "--draws", "4", "--",
            "judge_model.seed=3", "sampling={max_semantic_attempts:9}")
    cfg = evaluated[0]
    assert (cfg.judge_model.seed, cfg.sampling.num_decoder_draws, cfg.sampling.max_semantic_attempts) == (3, 4, 9)


def test_overlap_reads_nesting_both_ways_and_a_group_as_refined():
    group = Assignment("judge_model", "model@judge_model=x", "--judge x", group=True)
    leaf = setting("judge_model.seed", 3, "judge_model.seed=3")
    assert overlap(group, leaf) is None and overlap(leaf, group) is None
    assert overlap(group, Assignment("judge_model", "judge_model=null", "judge_model=null")) == "judge_model"
    whole = Assignment("sampling", "sampling=null", "sampling=null")
    draws = setting("sampling.num_decoder_draws", 4, "--draws 4")
    assert overlap(whole, draws) == overlap(draws, whole) == "sampling.num_decoder_draws"
    mapping = Assignment("sampling", "sampling={a:{b:1},c:{}}", "s", leaves=("sampling.a.b", "sampling.c"))
    assert overlap(mapping, draws) is None and overlap(mapping, setting("sampling.c.d", 1, "x")) == "sampling.c.d"
    [parsed] = parse_overrides(["sampling={a: {b: 1}, c: {}}"], Operation("x", ()), Path("/"))
    assert parsed.leaves == ("sampling.a.b", "sampling.c") and not parsed.group
    [selected] = parse_overrides(["model@judge_model=qwen3-8b"], Operation("x", ()), Path("/"))
    assert selected.group and selected.key == "judge_model" and selected.sets() == ("judge_model",)


@pytest.mark.parametrize("value", ["a\\b", "C:\\dir\\", "\\", "a\\'b", "it's", "a,b=c", "${x}", "", ["a\\b", "c"],
                                   "a\\\n", "a\n"])
def test_a_quoted_value_reads_back_unchanged_through_hydra_s_override_grammar(value):
    from hydra.core.override_parser.overrides_parser import OverridesParser

    assert OverridesParser.create().parse_override(f"k={quoted(value)}").value() == value


def test_a_job_s_own_destination_overrides_are_left_out_of_its_argv():
    assert compose_module.without_keys(["campaign_root=/a", "++campaign_id=c", "x=1", "a=[unclosed"],
                                       ("campaign_root", "campaign_id")) == ["x=1", "a=[unclosed"]
    assert compose_module.override_key("model@judge_model=q") == "judge_model"


@pytest.mark.parametrize("name", ["qwen3_8b.yaml", "qwen3_8b_mix.yaml"])
def test_a_grpo_config_composes_the_files_its_defaults_name_beside_it(name, monkeypatch, tmp_path):
    from ruleforge.cli.grpo import train_operation
    from ruleforge.core.provenance import repo_root

    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    directory = repo_root() / "configs" / "training" / "grpo"
    base = OmegaConf.load(directory / "base.yaml")
    cfg = compose(train_operation(), "grpo/base.yaml", config_file=directory / name, cwd=tmp_path)
    assert cfg.model.path == "/models/Qwen3-8B" and cfg.grpo.num_generations == base.grpo.num_generations
    if name == "qwen3_8b_mix.yaml":
        assert OmegaConf.select(cfg, "wandb.group") == OmegaConf.load(directory / name).wandb.group


def test_a_config_with_defaults_beside_a_directory_keeps_its_keys_keys(layout, cli, monkeypatch):
    conf = layout[0]
    (conf / "base.yaml").write_text("prompt_draws: 7\n")
    (conf / "rows.yaml").write_text("defaults: [base, _self_]\nrules: rules.jsonl\n")
    seen = capture(monkeypatch, grpo_package, "rows")
    with pytest.raises(Stop):
        cli("grpo", "rows", "--config", "../conf/rows.yaml", "--anchors", "a", "--scenarios", "t", "--out", "o",
            "--model", "m")
    assert seen[0]["prompt_draws"] == 7 and seen[0]["rules"] == str(conf / "rules.jsonl")


def test_a_swept_node_that_does_not_resolve_here_is_compared_as_written(monkeypatch):
    from ruleforge.cli.sweep import composed_value

    monkeypatch.delenv("SURELY_UNSET_VARIABLE_X", raising=False)
    cfg = OmegaConf.create({"judge": {"path": "${oc.env:SURELY_UNSET_VARIABLE_X}/w", "n": 1}, "seed": 2})
    assert composed_value(cfg, "judge") == {"path": "${oc.env:SURELY_UNSET_VARIABLE_X}/w", "n": 1}
    assert composed_value(cfg, "seed") == 2


def test_a_config_file_s_recorded_block_reads_under_today_s_key_before_its_paths_are_rebased(tmp_path):
    from ruleforge.core.record_names import recorded_block_key

    recorded = recorded_block_key("ruleset_size")
    saved = tmp_path / "saved.yaml"
    saved.write_text(f"{recorded}:\n  subset_root: subsets\n  seeds: [0]\nn: 1\n", encoding="utf-8")
    rebasing = Rebasing(patterns={"ruleset_size.subset_root": PATH})
    node = compose_module.load_user_config(saved, rebasing)
    assert OmegaConf.to_container(node) == {"ruleset_size": {"subset_root": str(tmp_path / "subsets"), "seeds": [0]},
                                            "n": 1}
    saved.write_text(f"{recorded}:\n  seeds: [0]\nruleset_size:\n  seeds: [1]\n", encoding="utf-8")
    with pytest.raises(CliError, match=f"--config {saved}: the config holds '{recorded}' and 'ruleset_size'"):
        compose_module.load_user_config(saved, rebasing)
