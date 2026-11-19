from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ruleforge import grpo
from ruleforge.evaluate.api import EXECUTION_FNAME
from ruleforge.experiments.training.pins import PINNED_PROTECTED
from ruleforge.grpo import grpo_data as gd
from ruleforge.grpo.leakage import LeakageError, disjoint_marker_path
from ruleforge.grpo import PromptBuilder, RowsBuild
from ruleforge.rulesets import Ruleset
from ruleforge.evaluate.campaign import load_conditions
from ruleforge.scenarios import ScenarioSet
from tests.test_training_grpo_data import (  # noqa: F401 - the module fixture
    MarkersDouble,
    build_into,
    read_jsonl,
    world,
    write_jsonl,
)
from tests.test_tasks import DemoTask


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path, world):  # noqa: F811
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(world.base / "models"))


def bank(world):  # noqa: F811
    return load_conditions(world.root)["gemma/individual_bank"]


def test_the_builder_renders_a_built_in_condition_s_prompts_as_the_campaign_did(world):  # noqa: F811
    condition = bank(world)
    builder = PromptBuilder.for_condition(condition, world.root)
    assert (builder.locator, builder.reasoning_prompt_style, builder.individual_rule_context) == (
        "modelspec", "forced-grpo", "isolated")
    reference = gd.CampaignPrompts.build(task_spec=gd.condition_task_spec(condition),
                                         rules=read_jsonl(world.rules), judge_type=condition.judge_type)
    prompts = builder.prompts(Ruleset.load(world.rules))
    rows = read_jsonl(world.population)
    for component in ("full_ruleset", "Rule 1", "Rule 2"):
        assert prompts.entry(rows[1], component) == reference.entry(rows[1], component)
    assert builder.task_identity("No_Interpretation") == {
        "task": "modelspec", "prompt_style": "forced-grpo", "individual_rule_context": "isolated",
        "judge_type": "No_Interpretation", "response_contract": None}
    assert builder.rendering() == {"task": builder.task_identity("No_Interpretation"), "task_locator": "modelspec"}


def test_the_builder_reads_the_task_a_root_recorded_and_holds_a_given_task_to_the_condition(world, tmp_path):  # noqa: F811
    outside = dataclasses.replace(bank(world), task="demo@1.0", response_contract="forced_json", prompt_style="json")
    root = tmp_path / "root"
    (root / outside.condition_id).mkdir(parents=True)
    (root / outside.condition_id / EXECUTION_FNAME).write_text(json.dumps({"task_locator": "tests.test_tasks:DemoTask"}))
    builder = PromptBuilder.for_condition(outside, root)
    assert isinstance(builder.task, DemoTask) and builder.locator == "tests.test_tasks:DemoTask"
    assert builder.build_eval_prompt("Rule 1: x", {"prompt": "p", "response": "r"}, "No_Interpretation").startswith(
        "# Demo task\n")
    assert PromptBuilder.for_condition(outside, tmp_path / "elsewhere", task=DemoTask()).locator == (
        "tests.test_tasks:DemoTask")
    with pytest.raises(gd.GrpoDataError, match="the task loaded, 'modelspec', has no prompt style 'json'"):
        PromptBuilder.for_condition(outside, tmp_path / "elsewhere", task="modelspec")
    with pytest.raises(gd.GrpoDataError, match="its task cannot render its prompts .*no task 'demo'"):
        PromptBuilder.for_condition(outside, tmp_path / "elsewhere")
    (root / outside.condition_id / EXECUTION_FNAME).write_text("{")
    with pytest.raises(gd.GrpoDataError, match="its task cannot render its prompts"):
        PromptBuilder.for_condition(outside, root)


def test_the_builder_refuses_a_scenario_lacking_a_field_its_task_reads(world):  # noqa: F811
    """A Model Spec scenario with no response would be rendered with ``None`` in its place and trained on."""
    builder = PromptBuilder.for_condition(bank(world), world.root)
    with pytest.raises(gd.GrpoDataError, match=r"task 'modelspec' cannot render a scenario of this build.*"
                                               r"lacks \['response'\]"):
        builder.build_eval_prompt("Rule 1: x", {"prompt": "p"}, "No_Interpretation")


def test_a_condition_whose_style_answers_a_bare_label_is_refused(world, tmp_path):  # noqa: F811
    label = dataclasses.replace(bank(world), prompt_style="naive")
    with pytest.raises(gd.GrpoDataError, match="under the 'label' contract: GRPO scores 'forced_json' answers alone"):
        PromptBuilder.for_condition(label, tmp_path)


def test_rows_are_the_builder_s_rows_byte_for_byte_with_the_task_recorded_and_no_pins(world, tmp_path):  # noqa: F811
    kwargs = dict(prompt_draws=4, class_a_share_cap=0.75, allow_no_held_out=True)
    built = grpo.rows(world.anchors, ScenarioSet.load(world.population), world.rules, world.model, tmp_path / "rows",
                      markers=MarkersDouble(), **kwargs)
    reference = build_into(world, tmp_path / "reference", pinned_protected={}, **kwargs)
    assert isinstance(built, RowsBuild) and built.out == tmp_path / "rows"
    for name in (gd.GRPO_ROWS_FNAME, gd.REPLAY_ROWS_FNAME, gd.SCENARIOS_FNAME):
        assert (tmp_path / "rows" / name).read_bytes() == (tmp_path / "reference" / name).read_bytes()
    payload = built.payload
    assert payload["rendering"]["task"]["task"] == "modelspec" and payload["rendering"]["task_locator"] == "modelspec"
    assert "task_spec" not in payload["rendering"] and reference["rendering"]["task_spec"] == "modelspec"
    assert {key: value for key, value in payload["rendering"].items() if key not in ("task", "task_locator")} == {
        key: value for key, value in reference["rendering"].items() if key != "task_spec"}
    assert payload["pinned_protected"] == [] and payload["allow_no_pinned_protected"] is False
    assert built.train_overrides() == {
        "data.train": str(tmp_path / "rows" / gd.GRPO_ROWS_FNAME),
        "data.replay": str(tmp_path / "rows" / gd.REPLAY_ROWS_FNAME),
        "data.rules": str(world.rules.absolute()), "data.rules_sha256": built.rules_sha256,
        "data.anchor_condition": payload["anchor_condition"]}
    assert set(built.markers) == {gd.SCENARIOS_FNAME, gd.GRPO_ROWS_FNAME, gd.REPLAY_ROWS_FNAME}
    assert RowsBuild.read(tmp_path / "rows") == built


def test_rows_pin_what_the_caller_names_and_refuse_a_pin_list(world, tmp_path):  # noqa: F811
    with pytest.raises(LeakageError, match="pinned protected files"):
        grpo.rows(world.anchors, world.population, world.rules, world.model, tmp_path / "pinned",
                  markers=MarkersDouble(), prompt_draws=4, allow_no_held_out=True, pins=PINNED_PROTECTED)
    with pytest.raises(gd.GrpoDataError, match=r"pins name protected files by digest"):
        grpo.rows(world.anchors, world.population, world.rules, world.model, tmp_path / "bad", pins=["frozen_500"])
    assert set(PINNED_PROTECTED) == {"frozen_500", "frozen_500_fig5_grpo_disjoint", "wildchat_stress_weird_200x3"}


def test_the_attestation_marks_the_training_file_against_its_protected_files(world, tmp_path):  # noqa: F811
    train = tmp_path / "train.jsonl"
    shutil.copy(world.population, train)
    marker = grpo.exclusion.attest(train, protected=[world.protected])
    assert disjoint_marker_path(train).is_file() and [entry["rows"] for entry in marker["protected"]] == [1]
    assert marker["screen"] == grpo.exclusion.UNSCREENED  # the second pass alone: no near-copy screen ran
    with pytest.raises(LeakageError, match="second pass alone"):
        grpo.exclusion.check_prompt_marker(train)
    assert grpo.exclusion.check_prompt_marker(train, allow_unscreened=True) == marker
    leaked = tmp_path / "leaked.jsonl"
    leaked.write_text(train.read_text().splitlines(keepends=True)[0])
    with pytest.raises(LeakageError, match="is not prompt-disjoint from the protected set"):
        grpo.exclusion.attest(train, protected=[world.protected, leaked])


def test_a_pool_changed_in_memory_or_built_by_hand_is_refused_by_the_rows(world, tmp_path):  # noqa: F811
    pool = ScenarioSet.load(world.population)
    changed = dataclasses.replace(pool, rows=pool.rows[:-1])
    with pytest.raises(gd.GrpoDataError, match="holds rows other than its file's"):
        grpo.rows(world.anchors, changed, world.rules, world.model, tmp_path / "refused", markers=MarkersDouble(),
                  prompt_draws=4, allow_no_held_out=True)
    assert not (tmp_path / "refused").exists()


def test_a_pool_whose_file_is_gone_is_refused_by_the_rows(world, tmp_path):  # noqa: F811
    pool = dataclasses.replace(ScenarioSet.load(world.population), path=tmp_path / "gone.jsonl")
    with pytest.raises(gd.GrpoDataError, match="is not the file it names"):
        grpo.rows(world.anchors, pool, world.rules, world.model, tmp_path / "refused", markers=MarkersDouble(),
                  prompt_draws=4, allow_no_held_out=True)
    assert not (tmp_path / "refused").exists()


def test_rows_are_the_paper_build_byte_for_byte_when_its_pins_are_passed(world, tmp_path):  # noqa: F811
    scenarios = tmp_path / "paper" / "train.jsonl"
    scenarios.parent.mkdir()
    shutil.copy(world.population, scenarios)
    pinned = [{"path": f"/populations/{name}.jsonl", "sha256": digest, "rows": 500}
              for name, digest in PINNED_PROTECTED.items()]
    MarkersDouble().write(scenarios, {
        "kind": "prompt_disjoint", "policy": {"shingle_n": 5},
        "dataset": {"name": scenarios.name, "sha256": gd.sha256_file(scenarios), "rows": len(read_jsonl(scenarios))},
        "protected": [{"path": str(world.protected), "sha256": gd.sha256_file(world.protected), "rows": 1}, *pinned],
    })
    common = dict(prompt_draws=4, class_a_share_cap=0.75, allow_no_held_out=True)
    built = grpo.rows(world.anchors, scenarios, world.rules, world.model, tmp_path / "rows", markers=MarkersDouble(),
                      pins=PINNED_PROTECTED, **common)
    reference = gd.build_grpo_data(anchors=world.anchors, scenarios=scenarios, rules=world.rules, model=world.model,
                                   out=tmp_path / "reference", markers=MarkersDouble(),
                                   pinned_protected=PINNED_PROTECTED, **common)
    for name in (gd.GRPO_ROWS_FNAME, gd.REPLAY_ROWS_FNAME, gd.SCENARIOS_FNAME):
        assert (tmp_path / "rows" / name).read_bytes() == (tmp_path / "reference" / name).read_bytes()
    payload = built.payload
    assert payload["pinned_protected"] == reference["pinned_protected"] == [
        {"name": name, "sha256": digest} for name, digest in PINNED_PROTECTED.items()]
    assert payload["sampling_weights"] == reference["sampling_weights"]
    assert {key: value for key, value in payload["rendering"].items() if key not in ("task", "task_locator")} == {
        key: value for key, value in reference["rendering"].items() if key != "task_spec"}


#: The checkout, and its base training config (the paper's runs compose their model configs over it).
CHECKOUT = Path(__file__).resolve().parents[1]
REPOSITORY_BASE = CHECKOUT / "configs/training/grpo/base.yaml"


def _overrides(**more):
    return {"data.train": "/t", "data.replay": "/r", "data.rules": "/rules.jsonl", "data.rules_sha256": "ab",
            "data.anchor_condition": "/c.json", "run_name": "outsider-1", "output_root": "/runs/train",
            "model.path": "/models/Qwen3-0.6B", "reward.truncated": "zero", "model.init_adapter": None, **more}


def test_the_training_config_is_the_base_plus_exactly_its_unset_keys(tmp_path):
    from omegaconf import OmegaConf

    assert grpo.api.base_config_path() == CHECKOUT / "src/ruleforge/configs/grpo/base.yaml"
    with pytest.raises(gd.GrpoDataError, match=r"leaves \['data.anchor_condition', 'data.replay', 'data.rules', "
                                               r"'data.rules_sha256', 'data.train', 'model.path', "
                                               r"'reward.truncated', 'run_name'\] unset"):
        grpo.api.training_config()
    config = grpo.api.training_config(overrides=_overrides(**{"+model.master_dtype": "bfloat16"}))
    assert config["output_dir"] == "/runs/train/outsider-1" and "hydra" not in config
    assert config["grpo"]["report_to"] == []
    # the repository's base, the paper's runs' (reporting to W&B, which the library's defaults turn off)
    base = OmegaConf.to_container(OmegaConf.load(REPOSITORY_BASE), resolve=False)
    assert base["grpo"]["report_to"] == ["wandb"]
    assert config["model"] == {"path": "/models/Qwen3-0.6B", "attn_implementation": "sdpa", "init_adapter": None,
                               "master_dtype": "bfloat16"}
    with pytest.raises(gd.GrpoDataError, match="override 'reward.truncate': .*a key the config lacks takes a '\\+'"):
        grpo.api.training_config(overrides=_overrides(**{"reward.truncate": "zero"}))
    given = grpo.api.training_config({**base, "run_name": "r"}, overrides=_overrides())
    assert given["grpo"]["report_to"] == ["wandb"] and given["output_dir"] == "/runs/train/outsider-1"
    whole = tmp_path / "whole.yaml"
    whole.write_text(OmegaConf.to_yaml(OmegaConf.create(base)))
    assert grpo.api.training_config(whole, overrides=_overrides()) == given
    composed = tmp_path / "composed.yaml"
    composed.write_text("defaults:\n  - base\n")
    with pytest.raises(gd.GrpoDataError, match="composes others \\(its defaults list\\)"):
        grpo.api.training_config(composed)


def test_without_a_packaged_base_config_the_repositorys_is_read(tmp_path, monkeypatch):
    from importlib import resources

    monkeypatch.setattr(resources, "files", lambda package: tmp_path / "ruleforge")
    assert grpo.api.base_config_path() == REPOSITORY_BASE
    assert grpo.api.training_config(overrides=_overrides())["grpo"]["report_to"] == []


def test_a_packaged_base_config_comes_first(tmp_path, monkeypatch):
    from importlib import resources

    packaged = tmp_path / "ruleforge"
    (packaged / "configs" / "grpo").mkdir(parents=True)
    (packaged / "configs" / "grpo" / "base.yaml").write_text("run_name: x\n")
    monkeypatch.setattr(resources, "files", lambda package: packaged)
    assert grpo.api.base_config_path() == packaged / "configs" / "grpo" / "base.yaml"


def _save(output_dir, step, *, preset=None, files=("adapter_config.json", "adapter_model.safetensors")):
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in files:
        (checkpoint / name).write_text("{}")
    preset = preset or {"architecture": "Qwen3ForCausalLM", "r": 32, "alpha": 64, "dropout": 0.0, "bias": "none",
                        "target_modules": ["q_proj", "v_proj"], "master_dtype": "float32"}
    record = {"step": step, "run": {"identity": {"preset": preset}}}
    (checkpoint / "smrc_provenance.json").write_text(json.dumps(record))
    return checkpoint


def test_a_run_names_its_finished_saves_and_what_a_merge_of_one_must_find(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("ruleforge.grpo.trainer.run_grpo", lambda config: calls.append(config))
    run = grpo.train(overrides=_overrides(output_root=str(tmp_path)))
    assert calls == [run.config] and run.output_dir == tmp_path / "outsider-1"
    with pytest.raises(gd.GrpoDataError, match="holds no finished save"):
        run.merge_expectation()
    _save(run.output_dir, 100)
    _save(run.output_dir, 200, preset={"architecture": "Gemma4ForConditionalGeneration", "r": 16, "alpha": 32,
                                       "target_modules": ["q_proj"], "module_scope": "model.language_model."})
    unfinished = _save(run.output_dir, 300, files=("adapter_config.json",))
    (run.output_dir / "checkpoint-x").mkdir()
    assert run.checkpoints() == [100, 200] and run.checkpoint(100) == run.output_dir / "checkpoint-100"
    with pytest.raises(gd.GrpoDataError, match=f"{unfinished} is not a finished save .*adapter_model.safetensors"):
        run.checkpoint(300)
    newest, first = run.merge_expectation(), run.merge_expectation(100)
    assert (newest.rank, newest.alpha, newest.base_model, newest.target_modules, newest.module_scope) == (
        16, 32.0, "Qwen3-0.6B", ("q_proj",), "model.language_model.")
    assert (first.rank, first.target_modules, first.module_scope) == (32, ("q_proj", "v_proj"), None)


def test_merge_is_the_evaluation_merge_returning_its_directory(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("ruleforge.grpo.merging.merge_adapter",
                        lambda base, adapter, out, **kwargs: calls.append((base, adapter, out, kwargs)))
    expect = object()
    assert grpo.merge("/base", "/run/checkpoint-1", tmp_path / "merged", expect=expect) == tmp_path / "merged"
    assert calls == [("/base", "/run/checkpoint-1", tmp_path / "merged",
                      {"expect": expect, "threads": 8, "expect_base_sha256": None, "training_run": None})]


def test_a_merge_records_the_training_run_its_save_names(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("ruleforge.grpo.merging.merge_adapter",
                        lambda base, adapter, out, **kwargs: calls.append(kwargs["training_run"]))
    save = _save(tmp_path / "outsider-1", 1)
    record = json.loads((save / "smrc_provenance.json").read_text())
    grpo.merge("/base", save, tmp_path / "unnamed", expect=None)
    record["run"]["config"] = {"run_name": "outsider-1"}
    (save / "smrc_provenance.json").write_text(json.dumps(record))
    grpo.merge("/base", save, tmp_path / "named", expect=None)
    grpo.merge("/base", save, tmp_path / "given", expect=None, training_run="another")
    grpo.merge("/base", tmp_path / "peft-adapter", tmp_path / "foreign", expect=None)
    assert calls == [None, "outsider-1", "another", None]


def test_the_package_reads_its_names_on_first_use():
    import ruleforge.grpo as package

    assert package.train is grpo.api.train and package.TrainRun is grpo.api.TrainRun and package.merge is grpo.api.merge
    assert {"rows", "train", "merge", "PromptBuilder", "anchor_set", "exclusion", "report"} <= set(dir(package))
    with pytest.raises(AttributeError, match="has no attribute 'nothing'"):
        getattr(package, "nothing")


@pytest.mark.parametrize("name", ["anchor_set", "exclusion", "report"])
def test_the_package_reads_its_modules_on_first_use(name, monkeypatch):
    import importlib

    import ruleforge.grpo as package

    monkeypatch.delattr(package, name, raising=False)
    assert getattr(package, name) is importlib.import_module(f"ruleforge.grpo.{name}") is vars(package)[name]


def test_the_toolkit_s_names_and_modules_resolve_in_a_fresh_interpreter(repo_root):
    probe = ("from ruleforge import grpo\n"
             "names = {name: getattr(grpo, name) for name in grpo.__all__}\n"
             "for name in ('anchor_set.build', 'anchor_set.route', 'exclusion.attest', 'report.report'):\n"
             "    module, attribute = name.split('.')\n"
             "    assert callable(getattr(names[module], attribute)), name\n"
             "print(sorted(names))\n")
    result = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(repo_root / 'src')!r})\n"
                             + probe], capture_output=True, text=True, cwd=repo_root)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(sorted(grpo.__all__))
