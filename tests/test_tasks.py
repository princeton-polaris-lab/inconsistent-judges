from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import ClassVar, Mapping

import pytest

from ruleforge import tasks as tasks_package
from ruleforge.prompts.context import VALID_INDIVIDUAL_RULE_CONTEXTS
from ruleforge.scenarios import ScenarioSet
from ruleforge.tasks.spec import TaskSpecAdapter, task_identity_of
from ruleforge.tasks.spec import LegalTaskSpec
from ruleforge.tasks.spec import ModelSpecTaskSpec
from ruleforge.prompts.styles import (
    FORCED_GRPO,
    FORCED_JSON_STYLES,
    HIERARCHY_RULE_CONTEXT,
    HIERARCHY_STYLES,
    NAIVE,
    NAIVE_HIERARCHY,
    VALID_REASONING_PROMPT_STYLES,
    validate_forced_response,
)
from ruleforge.tasks.spec import build_task_spec
from ruleforge.tasks import (
    BUILTIN_TASKS,
    RewardWeights,
    ENTRY_POINT_GROUP,
    FORCED_JSON,
    LABEL,
    LegalTask,
    ModelSpecTask,
    OptimizableTask,
    Task,
    TaskError,
    TaskSettings,
    check_task_identity,
    load_task,
    register_task,
    task_locator,
)
from ruleforge.tasks import registry as registry_module
from ruleforge.tasks.base import is_task_identifier
from ruleforge.tasks.modelspec import render_modelspec_prompt
from ruleforge.prompts.interpreters import INTERPRETATION_RULES

SCENARIO = {"prompt": "May I have a recipe?", "response": "Here is one."}


class DemoTask(Task):
    """An outside task."""

    name: ClassVar[str] = "demo"
    version: ClassVar[str] = "1.0"
    scenario_fields: ClassVar[tuple[str, ...]] = ("prompt", "response")
    styles: ClassVar[Mapping[str, str]] = {"plain": LABEL, "json": FORCED_JSON, "referenced": LABEL}
    default_style: ClassVar[str] = "plain"

    def check(self, settings: TaskSettings) -> None:
        super().check(settings)
        if settings.prompt_style == "referenced" and settings.rule_context != "implicated":
            raise TaskError("referenced is judged with the implicated context alone", setting="rule_context")

    def render(self, rule_text, scenario, settings, *, context_rules=None):
        base = FORCED_GRPO if self.contract(settings.prompt_style) == FORCED_JSON else NAIVE
        return "# Demo task\n" + render_modelspec_prompt(base, rule_text, scenario, settings.judge_type,
                                                         context_rules=context_rules)


class ArchivedDemoTask(DemoTask):
    """An outside task that keeps a style only to read the records that ran it."""

    name: ClassVar[str] = "demo-archived"
    record_only_styles: ClassVar[frozenset[str]] = frozenset({"json"})


class DemoOptimizable(OptimizableTask, DemoTask):
    name: ClassVar[str] = "demo-opt"
    version: ClassVar[str] = ""
    scenario_block_pattern: ClassVar[re.Pattern[str]] = re.compile(r"(?P<prompt>.*)")

    def sg_environment_description(self):
        return "sg env"

    def ra_environment_description(self):
        return "ra env"

    def sg_prior(self):
        return "sg prior"

    def ra_prior(self, weights):
        return f"ra prior {weights.intent} {weights.entropy_method}"

    def sg_system_prompt(self, *, per_batch, seeds, already, history, rng):
        return f"sg {per_batch} {seeds['type']} {seeds['count']} {sorted(seeds['paths'])} {len(already)} {len(history)}"

    def ra_history(self, history, weights):
        return f"ra history {len(history)} {weights.intent} {weights.intent_method}"

    def sg_history(self, history):
        return f"sg history {len(history)}"


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(registry_module, "_REGISTERED", {})
    register_task("demo", DemoTask)
    register_task("demo-opt", DemoOptimizable)


def test_the_modelspec_task_declares_every_published_style_and_its_contract():
    task = ModelSpecTask()
    assert tuple(task.styles) == VALID_REASONING_PROMPT_STYLES and task.default_style == NAIVE
    assert {style for style, contract in task.styles.items() if contract == FORCED_JSON} == set(FORCED_JSON_STYLES)
    assert set(task.styles.values()) == {LABEL, FORCED_JSON}
    assert (task.identifier, task.builtin, task.scenario_fields) == ("modelspec", True, ("prompt", "response"))
    assert tuple(BUILTIN_TASKS) == tuple(registry_module.BUILTIN_TASK_CLASSES)


@pytest.mark.parametrize(
    ("settings", "setting", "message"),
    [
        (TaskSettings("forced-v2"), "prompt_style", "Invalid reasoning_prompt_style: 'forced-v2'"),
        (TaskSettings(NAIVE, "nearby"), "rule_context", "rule context 'nearby'"),
        (TaskSettings(NAIVE, judge_type="Telepathy"), "judge_type", "judge type 'Telepathy'"),
        (TaskSettings("encouraged", judge_type="Narrow"), "judge_type", "No_Interpretation template only"),
        (TaskSettings(NAIVE_HIERARCHY), "rule_context", "judges each rule with the rules that outrank it"),
        (TaskSettings(NAIVE, HIERARCHY_RULE_CONTEXT), "rule_context", "belongs to the hierarchy prompt"),
    ],
)
def test_the_modelspec_task_refuses_what_it_never_approved(settings, setting, message):
    with pytest.raises(TaskError, match=re.escape(message)) as raised:
        ModelSpecTask().check(settings)
    assert raised.value.setting == setting


def test_the_hierarchy_pairing_and_every_other_style_in_every_other_rule_context_are_approved():
    task = ModelSpecTask()
    task.check(TaskSettings(NAIVE_HIERARCHY, HIERARCHY_RULE_CONTEXT))
    assert HIERARCHY_STYLES == (NAIVE_HIERARCHY,)
    for style in VALID_REASONING_PROMPT_STYLES:
        for context in VALID_INDIVIDUAL_RULE_CONTEXTS - {HIERARCHY_RULE_CONTEXT}:
            if style not in HIERARCHY_STYLES:
                task.check(TaskSettings(style, context))


def test_a_record_only_style_reads_its_records_and_no_new_run_selects_it():
    task = ArchivedDemoTask()
    settings = TaskSettings("json")
    task.check(settings)
    assert task.identity(settings)["prompt_style"] == "json" and task.contract("json") == FORCED_JSON
    assert (task.new_run_refusal("plain"), DemoTask().new_run_refusal("json"), ModelSpecTask.record_only_styles) == (
        None, None, frozenset())
    assert task.new_run_refusal("json") == (
        "prompt style 'json' of task 'demo-archived@1.0' is recognised only to read the records that ran it; a new "
        "run cannot select it")
    with pytest.raises(TaskError) as raised:
        task.check(TaskSettings("nope"))
    assert str(raised.value) == (
        "task 'demo-archived@1.0' has no prompt style 'nope'; its styles: ['plain', 'referenced']; recognised only to "
        "read the records that ran them: ['json']")
    with pytest.raises(TaskError) as raised:
        ModelSpecTask().check(TaskSettings("nope"))
    assert str(raised.value) == (
        "Invalid reasoning_prompt_style: 'nope'; valid: ['encouraged', 'forced', 'forced-grpo', 'naive', "
        "'naive-hierarchy', 'naive-rules-above']")
    with pytest.raises(TaskError) as raised:
        DemoTask().check(TaskSettings("nope"))
    assert str(raised.value) == "task 'demo@1.0' has no prompt style 'nope'; its styles: ['json', 'plain', 'referenced']"


def test_a_built_in_identity_records_no_contract():
    assert ModelSpecTask().identity(TaskSettings(FORCED_GRPO, judge_type="No_Interpretation")) == {
        "task": "modelspec", "prompt_style": FORCED_GRPO, "individual_rule_context": "isolated",
        "judge_type": "No_Interpretation", "response_contract": None}
    assert LegalTask().identity(TaskSettings(NAIVE))["task"] == "legal"


@pytest.mark.parametrize(
    ("settings", "message"),
    [(TaskSettings(FORCED_GRPO), "supports only reasoning_prompt_style='naive'"),
     (TaskSettings(NAIVE, "nearby"), "rule context 'nearby'"),
     (TaskSettings(NAIVE, HIERARCHY_RULE_CONTEXT), "belongs to the hierarchy prompt")],
)
def test_the_legal_task_refuses_every_style_but_naive_an_unknown_context_and_the_hierarchy_context(settings, message):
    with pytest.raises(TaskError, match=re.escape(message)):
        LegalTask().check(settings)


#: The judge prompts the task specs render, by digest.
RENDER_TABLE = Path(__file__).resolve().parent / "fixtures" / "tasks" / "render_sha256.json"
RENDER_RULE = "Rule 1: Be kind."
#: No context list, an empty one (the same prompt) and two context rules.
RENDER_CONTEXTS = (None, [], ["Rule 2: Be brief.", "Meta-Rule 1: Rules apply in order."])
RENDER_SCENARIOS = {"modelspec": {"prompt": "May I have a recipe?", "response": "Here is one."},
                    "legal": {"facts_of_case": "F.", "legal_question": "Q?"}}


def test_the_tasks_render_the_prompts_the_task_specs_rendered_before_the_move(task_cfg):
    pinned = json.loads(RENDER_TABLE.read_text(encoding="utf-8"))["table"]
    rendered = {}
    for task in (ModelSpecTask(), LegalTask()):
        for style, context, judge_type in itertools.product(
                VALID_REASONING_PROMPT_STYLES, sorted(VALID_INDIVIDUAL_RULE_CONTEXTS),
                ("No_Interpretation", *INTERPRETATION_RULES)):
            settings = TaskSettings(style, context, judge_type)
            try:
                task.check(settings)
            except TaskError:
                continue
            if task.new_run_refusal(style) is not None:
                continue
            spec = build_task_spec(task_cfg(task_spec=task.name, reasoning_prompt_style=style,
                                            individual_rule_context=context, judge_types=[judge_type]))
            adapter = TaskSpecAdapter(task, spec.cfg)
            scenario = RENDER_SCENARIOS[task.name]
            for index, rules in enumerate(RENDER_CONTEXTS):
                prompts = {
                    task.render(RENDER_RULE, dict(scenario), settings, context_rules=rules),
                    spec.build_eval_prompt(RENDER_RULE, dict(scenario), judge_type, context_rules=rules),
                    adapter.build_eval_prompt(RENDER_RULE, dict(scenario), judge_type, context_rules=rules),
                }
                assert len(prompts) == 1, (task.name, style, context, judge_type, index)
                # implicated_ranges changes which rules are context, never the template: it renders implicated's prompt.
                pinned_as = "implicated" if context == "implicated_ranges" else context
                key = "|".join((task.name, style, pinned_as, judge_type, str(index)))
                digest = hashlib.sha256(prompts.pop().encode("utf-8")).hexdigest()
                assert rendered.setdefault(key, digest) == digest, key
    assert rendered == pinned
    assert isinstance(build_task_spec(task_cfg(task_spec="legal")), LegalTaskSpec)
    assert ModelSpecTaskSpec(task_cfg()).task.identifier == "modelspec"


def test_an_unknown_judge_type_is_refused_when_rendered():
    with pytest.raises(ValueError, match="Invalid judge type: Telepathy"):
        ModelSpecTask().render("Rule 1: x", SCENARIO, TaskSettings(NAIVE, judge_type="Telepathy"))
    with pytest.raises(ValueError, match="Invalid judge type: Telepathy"):
        LegalTask().render("Sec. 1", {}, TaskSettings(NAIVE, judge_type="Telepathy"))


def test_an_outside_task_records_its_versioned_name_and_its_style_s_contract():
    task = DemoTask()
    assert (task.identifier, task.builtin) == ("demo@1.0", False)
    assert task.identity(TaskSettings("json"))["response_contract"] == FORCED_JSON
    assert task.identity(TaskSettings("referenced", "implicated"))["response_contract"] == LABEL
    assert task.contract("plain") == LABEL
    with pytest.raises(TaskError, match="has no prompt style 'nope'"):
        task.contract("nope")
    with pytest.raises(TaskError, match="has no prompt style 'nope'"):
        task.check(TaskSettings("nope"))
    with pytest.raises(TaskError, match="referenced is judged with the implicated context"):
        task.identity(TaskSettings("referenced"))


def test_a_subclass_of_a_built_in_task_is_not_built_in():
    class Mine(ModelSpecTask):
        name = "mine"

    assert Mine().builtin is False
    assert Mine().identity(TaskSettings(FORCED_GRPO))["response_contract"] == FORCED_JSON


def test_scenario_rows_must_carry_the_task_s_fields(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"prompt": "p", "response": "r"}\n', encoding="utf-8")
    DemoTask().check_scenarios(ScenarioSet.load(path))
    DemoTask().check_scenarios([{"prompt": "p", "response": "r", "extra": 1}])
    with pytest.raises(TaskError, match=r"scenario row 1 lacks \['response'\]"):
        DemoTask().check_scenarios([{"prompt": "p", "response": "r"}, {"prompt": "p"}])


@pytest.mark.parametrize("task", [ModelSpecTask(), LegalTask()], ids=["modelspec", "legal"])
def test_a_built_in_task_is_optimizable_with_no_config(task):
    assert isinstance(task, OptimizableTask) and task.builtin
    assert set(task.scenario_fields) <= set(task.scenario_block_pattern.groupindex)
    weights = RewardWeights(intent=0.25)
    assert "0.25*(intent_penalty)" in task.ra_prior(weights) and "Conditions" in task.sg_prior()
    assert task.ra_history([], weights) == "" and task.sg_history([]) == ""
    unseeded = task.sg_system_prompt(per_batch=7, seeds={"type": "none", "count": 1, "paths": {}}, already=[],
                                     history=[], rng=random.Random(0))
    assert "You must generate 7 scenarios" in unseeded and "Example" not in unseeded
    assert unseeded.endswith(f"# Environment Description Given to the Agent\n{task.sg_environment_description()}")
    other = {"modelspec": "single_seed", "legal": "wildchat"}[task.name]
    assert task.sg_system_prompt(per_batch=7, seeds={"type": other, "count": 1, "paths": {}}, already=[],
                                 history=[], rng=random.Random(0)) == unseeded


def test_an_optimizable_task_reads_labels_and_forced_answers_by_contract():
    task = DemoOptimizable()
    assert task.parse_judgment(" Yes. ") == "yes"
    assert task.response_validator("json") is validate_forced_response
    assert task.response_validator("plain") is None


@pytest.mark.parametrize(("value", "ok"), [("demo", True), ("demo@1.0", True), ("a.b-c_d@2+x", True),
                                           ("", False), ("@1", False), ("demo@", False), (3, False), ("a b", False)])
def test_a_task_identifier_is_a_name_or_name_at_version(value, ok):
    assert is_task_identifier(value) is ok


def test_built_in_names_instances_and_classes_load(registered):
    assert isinstance(load_task(), ModelSpecTask) and isinstance(load_task("legal"), LegalTask)
    task = DemoTask()
    assert load_task(task) is task
    assert isinstance(load_task(DemoTask), DemoTask)
    assert isinstance(load_task("demo"), DemoTask) and isinstance(load_task("demo@1.0"), DemoTask)


def test_a_version_other_than_the_installed_one_is_refused(registered):
    with pytest.raises(TaskError, match="version '2.0' was asked for; the one installed is version '1.0'"):
        load_task("demo@2.0")
    with pytest.raises(TaskError, match="the one installed is version ''"):
        load_task("modelspec@1")


def test_a_dotted_path_loads_its_class(monkeypatch):
    module = types.ModuleType("demo_tasks_pkg")
    module.DemoTask = DemoTask
    module.NotATask = object
    module.Nested = types.SimpleNamespace(Task=DemoTask)
    monkeypatch.setitem(sys.modules, "demo_tasks_pkg", module)
    assert isinstance(load_task("demo_tasks_pkg:DemoTask"), DemoTask)
    assert isinstance(load_task("demo_tasks_pkg:Nested.Task@1.0"), DemoTask)
    assert isinstance(load_task("ruleforge.tasks.modelspec:ModelSpecTask"), ModelSpecTask)
    with pytest.raises(TaskError, match="which is not a Task"):
        load_task("demo_tasks_pkg:NotATask")
    with pytest.raises(TaskError, match="cannot load task 'demo_tasks_pkg:Missing'"):
        load_task("demo_tasks_pkg:Missing")
    with pytest.raises(TaskError, match="cannot load task 'no_such_module_x:Thing'"):
        load_task("no_such_module_x:Thing")


def test_an_installed_entry_point_is_found_by_its_name(monkeypatch):
    class Entry:
        def __init__(self, name, target):
            self.name, self._target = name, target

        def load(self):
            return self._target

    entries = [Entry("other", DemoOptimizable), Entry("demo", DemoTask), Entry("wrong", DemoTask)]
    monkeypatch.setattr(registry_module.metadata, "entry_points",
                        lambda group: entries if group == ENTRY_POINT_GROUP else [])
    assert isinstance(load_task("demo@1.0"), DemoTask)
    with pytest.raises(TaskError, match="'wrong' found a task named 'demo'"):
        load_task("wrong")
    with pytest.raises(TaskError, match="no task 'absent'"):
        load_task("absent")


def test_a_task_cannot_take_a_built_in_name(monkeypatch):
    monkeypatch.setattr(registry_module, "_REGISTERED", {})

    class Impostor(DemoTask):
        name = "modelspec"

    module = types.ModuleType("impostor_pkg")
    module.Impostor = Impostor
    monkeypatch.setitem(sys.modules, "impostor_pkg", module)
    with pytest.raises(TaskError, match="a built-in task's name"):
        register_task("modelspec", Impostor)
    with pytest.raises(TaskError, match="a built-in task's name"):
        load_task("impostor_pkg:Impostor")
    with pytest.raises(TaskError, match="is named 'demo', not 'other'"):
        register_task("other", DemoTask)
    with pytest.raises(TaskError, match="needs a Task subclass"):
        register_task("thing", object)
    with pytest.raises(TaskError, match="a task is named by a string"):
        load_task(3)
    with pytest.raises(TaskError, match="a task is named by a string"):
        load_task("")


def test_the_tasks_listed_are_the_ones_load_task_finds_by_name(registered, monkeypatch):
    class Entry:
        def __init__(self, name, target):
            self.name, self._target = name, target

        def load(self):
            if isinstance(self._target, Exception):
                raise self._target
            return self._target

    class Styleless(Task):
        """Loads and constructs (``render`` is its one abstract method) but declares no styles."""
        name = "styleless"

        def render(self, rule_text, scenario, settings, *, context_rules=None):
            return ""

    class Unbuildable(DemoTask):
        name = "unbuildable"

        def __init__(self):
            raise RuntimeError("needs a config")

    register_task("unbuildable", Unbuildable)
    module = types.ModuleType("listed_pkg")
    module.TASK = DemoTask()
    entries = [Entry("zz-broken", ImportError("no module named zz")), Entry("demo", DemoOptimizable),
               Entry("modelspec", DemoTask), Entry("wrong", DemoTask), Entry("plain", object),
               Entry("styleless", Styleless),
               Entry("other-opt", type("OtherOpt", (DemoOptimizable,), {"name": "other-opt",
                                                                     "record_only_styles": frozenset({"json"})}))]
    monkeypatch.setattr(registry_module.metadata, "entry_points",
                        lambda group: entries if group == ENTRY_POINT_GROUP else [])
    listed = tasks_package.list_tasks()
    assert [(info.name, info.source, info.locator) for info in listed] == [
        ("modelspec", "built-in", "modelspec"), ("legal", "built-in", "legal"),
        ("demo", "registered", "tests.test_tasks:DemoTask"),
        ("demo-opt", "registered", "tests.test_tasks:DemoOptimizable"),
        ("unbuildable", "registered",
         "tests.test_tasks:test_the_tasks_listed_are_the_ones_load_task_finds_by_name.<locals>.Unbuildable"),
        ("other-opt", "entry point", "other-opt"), ("plain", "entry point", "plain"),
        ("styleless", "entry point", "styleless"), ("wrong", "entry point", "wrong"),
        ("zz-broken", "entry point", "zz-broken")]
    by_name = {info.name: info for info in listed}
    modelspec = by_name["modelspec"]
    assert (modelspec.identifier, modelspec.optimizable, modelspec.default_style, modelspec.error) == (
        "modelspec", True, NAIVE, None)
    assert dict(modelspec.styles) == dict(ModelSpecTask.styles)
    assert dict(by_name["other-opt"].styles) == {"plain": LABEL, "referenced": LABEL}
    assert (by_name["demo"].identifier, by_name["demo"].optimizable) == ("demo@1.0", False)
    assert (by_name["other-opt"].identifier, by_name["other-opt"].optimizable) == ("other-opt", True)
    assert by_name["plain"].error == "TaskError: 'plain' names <class 'object'>, which is not a Task"
    assert by_name["wrong"].error == "TaskError: it loads a task named 'demo'"
    assert by_name["zz-broken"].error == "ImportError: no module named zz"
    assert (by_name["zz-broken"].identifier, by_name["zz-broken"].styles) == (None, {})
    assert by_name["styleless"].error == "AttributeError: 'Styleless' object has no attribute 'styles'"
    assert (by_name["styleless"].identifier, by_name["styleless"].styles) == (None, {})
    assert by_name["unbuildable"].error == "RuntimeError: needs a config"


def test_a_locator_is_the_spec_as_given_or_the_class_path():
    assert task_locator("pkg.mod:Cls") == "pkg.mod:Cls" and task_locator("modelspec") == "modelspec"
    assert task_locator(DemoTask()) == f"{__name__}:DemoTask"
    assert task_locator(ModelSpecTask) == "ruleforge.tasks.modelspec:ModelSpecTask"


def test_a_registered_task_s_locator_loads_it_where_nothing_is_registered(registered):
    """A registered name lives in this process's registry alone, so its locator is the class's path."""
    assert task_locator("demo") == f"{__name__}:DemoTask"
    assert task_locator("demo@1.0") == f"{__name__}:DemoTask@1.0"
    assert task_locator("demo-opt") == f"{__name__}:DemoOptimizable"
    locators = [task_locator(spec) for spec in ("demo", "demo@1.0", "demo@2.0")]
    registry_module._REGISTERED.clear()
    with pytest.raises(TaskError, match="no task 'demo'"):
        load_task("demo")
    assert [type(load_task(locator)) for locator in locators[:2]] == [DemoTask, DemoTask]
    with pytest.raises(TaskError, match="version '2.0' was asked for; the one installed is version '1.0'"):
        load_task(locators[2])


def test_a_loaded_task_is_held_to_the_identity_a_condition_recorded():
    check_task_identity(ModelSpecTask(), task_name="modelspec", response_contract=None, prompt_style=FORCED_GRPO,
                        where="c")
    check_task_identity(DemoTask(), task_name="demo@1.0", response_contract=FORCED_JSON, prompt_style="json",
                        where="c")
    with pytest.raises(TaskError, match=r"c: recorded task 'demo@1.0' \(response contract 'label'\), but the task "
                                        r"loaded is 'demo@1.0' \(response contract 'forced_json'\)"):
        check_task_identity(DemoTask(), task_name="demo@1.0", response_contract=LABEL, prompt_style="json",
                            where="c")
    with pytest.raises(TaskError, match="recorded task 'modelspec'"):
        check_task_identity(DemoTask(), task_name="modelspec", response_contract=None, prompt_style="plain", where="c")
    with pytest.raises(TaskError, match="has no prompt style 'naive'"):
        check_task_identity(DemoTask(), task_name="demo@1.0", response_contract=LABEL, prompt_style="naive",
                            where="c")


def test_the_package_exports_the_contract():
    assert set(tasks_package.__all__) <= set(dir(tasks_package))


@pytest.mark.parametrize("module", ["ruleforge.tasks", "ruleforge.tasks.spec"])
def test_the_task_packages_load_no_judge_metric_or_loop(module, repo_root):
    probe = (f"import importlib, sys\nimportlib.import_module({module!r})\n"
             "print(sorted(name for name in sys.modules if name.startswith(('ruleforge.judges', 'ruleforge.optimize', "
             "'ruleforge.evaluate', 'ruleforge.grpo'))))\n")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=repo_root,
                            env={**os.environ, "PYTHONPATH": str(repo_root / "src")})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def _cfg(task_cfg, task_spec="demo", **eval_overrides):
    return task_cfg(task_spec=task_spec, **eval_overrides)


def test_build_task_spec_puts_an_outside_task_behind_the_adapter(registered, task_cfg):
    spec = build_task_spec(_cfg(task_cfg, reasoning_prompt_style="JSON"))
    assert isinstance(spec, TaskSpecAdapter) and isinstance(spec.task, DemoTask)
    assert (spec.reasoning_prompt_style, spec.individual_rule_context, spec.judge_types) == (
        "json", "isolated", ["No_Interpretation"])
    assert spec.valid_scenario_keys == ("prompt", "response") and spec.interpretation_rules is INTERPRETATION_RULES
    assert spec.build_eval_prompt("Rule 1: x", dict(SCENARIO), "No_Interpretation").startswith("# Demo task\n")
    assert spec.eval_response_validator is validate_forced_response
    assert spec.parse_judgment("no") == "no"
    assert spec.task_identity("No_Interpretation") == {
        "task": "demo@1.0", "prompt_style": "json", "individual_rule_context": "isolated",
        "judge_type": "No_Interpretation", "response_contract": FORCED_JSON}
    assert not hasattr(spec, "scenario_block_pattern")


def test_the_adapter_takes_the_task_s_default_style_and_asks_the_task_to_approve(registered, task_cfg):
    cfg = _cfg(task_cfg)
    assert build_task_spec(cfg).reasoning_prompt_style == "plain"
    assert build_task_spec(cfg).eval_response_validator is None
    with pytest.raises(TaskError, match="referenced is judged with the implicated context"):
        build_task_spec(_cfg(task_cfg, reasoning_prompt_style="referenced"))
    referenced = build_task_spec(_cfg(task_cfg, reasoning_prompt_style="referenced",
                                      individual_rule_context="implicated"))
    assert referenced.task_identity("No_Interpretation")["individual_rule_context"] == "implicated"


def test_the_adapter_refuses_the_optimizer_s_half_for_a_task_that_has_none(registered, task_cfg):
    spec = build_task_spec(_cfg(task_cfg))
    weights = RewardWeights()
    for call in (spec.get_sg_environment_description, spec.get_ra_environment_description, spec.get_sg_prior,
                 lambda: spec.get_ra_prior(weights), lambda: spec.build_ra_history_string([], weights),
                 lambda: spec.build_sg_history_string([]),
                 lambda: spec.build_sg_posterior_sampling_system_prompt([], [], random.Random(0))):
        with pytest.raises(TaskError, match="is not an OptimizableTask"):
            call()
    assert not any(hasattr(spec, name) for name in ("ra_reward", "sg_reward", "intent_weight"))


def test_the_adapter_hands_the_optimizer_s_half_to_an_optimizable_task(registered, task_cfg):
    spec = build_task_spec(_cfg(task_cfg, task_spec="demo-opt"))
    assert spec.scenario_block_pattern is DemoOptimizable.scenario_block_pattern
    assert (spec.get_sg_environment_description(), spec.get_ra_environment_description(), spec.get_sg_prior()) == (
        "sg env", "ra env", "sg prior")
    weights = RewardWeights(intent=0, intent_method="rubric_orig_standards")
    assert spec.get_ra_prior(weights) == "ra prior 0.0 consensus"
    assert spec.build_ra_history_string([1, 2], weights) == "ra history 2 0.0 rubric_orig_standards"
    assert spec.build_sg_history_string([1]) == "sg history 1"
    assert spec.build_sg_posterior_sampling_system_prompt([{"prompt": "p"}], [], random.Random(0)) == (
        "sg 3 none 1 ['bad_seed_scenario_path', 'good_seed_scenario_path', 'seed_scenario_path'] 1 0")
    assert spec.parse_judgment("unsure") == "unsure" and spec.eval_response_validator is None


def test_an_unknown_task_spec_names_the_built_ins_and_why(task_cfg):
    with pytest.raises(ValueError, match=r"Unknown task_spec 'nope'\. Valid options: \['legal', 'modelspec'\]"):
        build_task_spec(task_cfg(task_spec="nope"))


def test_a_stand_in_task_spec_records_the_built_in_modelspec_task():
    stand_in = types.SimpleNamespace(reasoning_prompt_style=NAIVE, individual_rule_context="isolated")
    assert task_identity_of(stand_in, "No_Interpretation") == {
        "task": "modelspec", "prompt_style": NAIVE, "individual_rule_context": "isolated",
        "judge_type": "No_Interpretation", "response_contract": None}


def test_a_dotted_path_or_entry_point_may_name_an_instance(monkeypatch):
    module = types.ModuleType("demo_instance_pkg")
    module.TASK = DemoTask()
    monkeypatch.setitem(sys.modules, "demo_instance_pkg", module)
    assert load_task("demo_instance_pkg:TASK") is module.TASK
