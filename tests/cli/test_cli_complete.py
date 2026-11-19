from __future__ import annotations

import argparse
import importlib
import inspect
import types
from typing import Any, Iterator, Optional

import pytest

from ruleforge.cli.app import build_parser
from ruleforge.cli.mounted import subcommands

#: A level of ``reproduce``, in place of its experiment (a positional word: any of them, or ``all``).
EXPERIMENT = "<EXPERIMENT>"

#: Options a verb must take, by verb (``campaign run --preflight-only``: a run's checks alone, no judge started).
OPTIONS = {("campaign", "run"): ("--preflight-only",), ("reproduce",): ("--rerun", "--exact", "--check")}

#: The modules run as ``python -m`` programs (the Hub packing), each running its command line.
PROGRAMS = ("ruleforge.hub",)

#: Every verb the parser builds, and the API it runs (``module:attribute``; a mounted step's ``module:main``).
WIRING = {
    ("evaluate",): "ruleforge.evaluate.api:evaluate",
    ("campaign", "create"): "ruleforge.evaluate.api:Campaign.create",
    ("campaign", "add"): "ruleforge.evaluate.api:Campaign.condition",
    ("campaign", "run"): "ruleforge.evaluate.api:Campaign.run",
    ("campaign", "status"): "ruleforge.evaluate.api:Campaign.status",
    ("campaign", "analyze"): "ruleforge.evaluate.api:Campaign.analyze",
    ("campaign", "collect"): "ruleforge.judges.batch_collect:collect",
    ("campaign", "cost", "usage"): "ruleforge.evaluate.cost:campaign_usage",
    ("campaign", "merge-shards"): "ruleforge.evaluate.api:Campaign.merge_shards",
    ("analyze",): "ruleforge.evaluate.api:Campaign.analyze",
    ("bank", "run"): "ruleforge.evaluate.bank:run_job",
    ("bank", "analyze"): "ruleforge.evaluate.bank:analyze_into",
    ("scenarios", "concat"): "ruleforge.scenarios.subset:build_concatenation",
    ("scenarios", "subtract"): "ruleforge.scenarios.subset:build_difference",
    ("scenarios", "partition"): "ruleforge.scenarios.subset:build_blocks",
    ("optimize",): "ruleforge.optimize.api:optimize",
    ("optimize", "select"): "ruleforge.optimize.select:sweep_table",
    ("grpo", "rows"): "ruleforge.grpo.api:rows",
    ("grpo", "merge"): "ruleforge.grpo.merging:merge_adapter",
    ("grpo", "anchors", "build"): "ruleforge.grpo.anchor_set:build_anchor_set",
    ("grpo", "anchors", "route"): "ruleforge.grpo.anchor_set:route_top_up",
    ("grpo", "train"): "ruleforge.grpo.trainer:run_grpo",
    ("grpo", "exclusion", "attest"): "ruleforge.grpo.exclusion:attest",
    **{("grpo", "parity", step): "ruleforge.grpo.lora_parity:main"
       for step in ("make-adapter", "inspect", "vllm", "hf", "compare", "greedy-agreement", "plumbing")},
    ("grpo", "report", "saves"): "ruleforge.grpo.report:main",
    **{("grpo", "report", step): "ruleforge.grpo.report_tables:main" for step in ("populations", "read")},
    ("judges", "list"): "ruleforge.judges.base:model_config_names",
    ("judges", "show"): "ruleforge.cli.judging:judge_spec",
    ("judges", "check"): "ruleforge.judges:check",
    ("tasks", "list"): "ruleforge.tasks:list_tasks",
    ("prompts", "show"): "ruleforge.tasks.registry:load_task",
    ("version",): "ruleforge.core.provenance:code_state",
    **{("reproduce", EXPERIMENT, level): f"ruleforge.cli.reproduce:{level}"
       for level in ("render", "rebuild", "verify", "subsets")},
    **{("jobs", "launch", verb): f"ruleforge.launch.cli:LAUNCH_HANDLERS[{verb}]" for verb in ("job", "campaign", "train")},
    **{("jobs", verb): f"ruleforge.launch.cli:CONTROL_HANDLERS[{verb}]" for verb in ("status", "wait", "pull", "stop")},
    ("data", "fetch"): "ruleforge.hub:fetch",
    ("data", "verify"): "ruleforge.hub:verify",
}


def _children(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    found: dict[str, argparse.ArgumentParser] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            found.update(action.choices)
    return found


def _positional_choices(parser: argparse.ArgumentParser) -> list[tuple[str, ...]]:
    return [tuple(action.choices) for action in parser._actions
            if not action.option_strings and action.choices and not isinstance(action, argparse._SubParsersAction)]


def walk(parser: argparse.ArgumentParser, words: tuple[str, ...] = (), handler: Any = None
         ) -> Iterator[tuple[tuple[str, ...], argparse.ArgumentParser, Any]]:
    """Every command the parser builds."""
    handler = parser.get_default("handler") or handler
    children = _children(parser)
    if handler is not None and (not children or not any(
            action.required for action in parser._actions if isinstance(action, argparse._SubParsersAction))):
        positional = _positional_choices(parser)
        if words == ("reproduce",):
            experiments, levels = positional
            assert "all" in experiments
            for level in levels:
                yield (*words, EXPERIMENT, level), parser, handler
        else:
            yield words, parser, handler
    for name, child in children.items():
        yield from walk(child, (*words, name), handler)


@pytest.fixture(scope="module")
def built() -> dict[tuple[str, ...], tuple[argparse.ArgumentParser, Any]]:
    return {words: (parser, handler) for words, parser, handler in walk(build_parser())}


def test_every_built_verb_is_wired_and_none_is_unlisted(built):
    assert sorted(built) == sorted(WIRING)


def test_each_listed_option_is_its_verb_s(built):
    parsers = {words[:1] if words[0] == "reproduce" else words: parser for words, (parser, _) in built.items()}
    for words, options in OPTIONS.items():
        given = {option for action in parsers[words]._actions for option in action.option_strings}
        assert set(options) <= given, words


def test_each_listed_program_runs_its_command_line():
    for module in PROGRAMS:
        source = inspect.getsource(importlib.import_module(module))
        assert '\nif __name__ == "__main__":' in source, module


def _resolve(dotted: str) -> Any:
    module, _, attribute = dotted.partition(":")
    found: Any = importlib.import_module(module)
    for name in attribute.split("."):
        found = getattr(found, name)
    return found


def _closure(handler: Any) -> dict[str, Any]:
    return {name: cell.cell_contents for name, cell in zip(handler.__code__.co_freevars, handler.__closure__ or ())}


def _names(function: Any, seen: Optional[set] = None) -> set[str]:
    """The names ``function``'s code refers to, and those of its module's functions it refers to, transitively."""
    seen = set() if seen is None else seen
    seen.add(function)
    module = inspect.getmodule(function)

    def of(code: types.CodeType) -> set[str]:
        found = set(code.co_names)
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                found |= of(constant)
        return found

    names = of(function.__code__)
    for name in list(names):
        called = getattr(module, name, None)
        if inspect.isfunction(called) and called.__module__ == module.__name__ and called not in seen:
            names |= _names(called, seen)
    return names


def _refers_to_module(names: set[str], module: str) -> bool:
    parts = module.split(".")
    return module in names or any(".".join(parts[:cut]) in names and set(parts[cut:]) <= names
                                  for cut in range(1, len(parts)))


def wired(handler: Any, api: str) -> bool:
    """Whether ``handler`` runs ``api``."""
    module, _, attribute = api.partition(":")
    if handler.__name__ == "<lambda>":
        closure = _closure(handler)
        if "command" in closure:
            return closure["command"].function() is _resolve(api)
        tool, step = closure["tool"], closure["step"]
        return f"{tool.module}:{tool.main}" == api and step in subcommands(tool.build())
    names = _names(handler)
    table, _, key = attribute.partition("[")
    if key:
        return _refers_to_module(names, module) and table in names and key.rstrip("]") in _resolve(f"{module}:{table}")
    first = attribute.split(".")[0]
    imported = getattr(inspect.getmodule(handler), first, None) is _resolve(f"{module}:{first}")
    return (imported or _refers_to_module(names, module)) and set(attribute.split(".")) <= names and (
        _resolve(api) is not None)


def test_every_verb_runs_its_api_function(built):
    for words, (_, handler) in built.items():
        assert wired(handler, WIRING[words]), (" ".join(words), WIRING[words])


def test_every_verb_s_help_is_its_usage(built, capsys):
    parser = build_parser()
    for words in built:
        with pytest.raises(SystemExit) as raised:
            parser.parse_args(["reproduce", "--help"] if EXPERIMENT in words else [*words, "--help"])
        assert raised.value.code == 0 and f"usage: ruleforge {words[0]}" in capsys.readouterr().out, words
