from __future__ import annotations

import argparse
import dataclasses
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

from ruleforge.cli.common import subcommand
from ruleforge.cli.judging import judge_node, judge_spec
from ruleforge.cli.records import check_batch_roots, describe, guarded_batch_writes


def _identity(identity: Any) -> dict[str, Any]:
    return {name: value for name, value in dataclasses.asdict(identity).items()
            if value is not None and name != "truncation_source"}


def run_judges_list(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Print every name ``--judge NAME`` finds, then the shorthands ``--judge`` reads."""
    from ruleforge.judges.base import SHORTHANDS, model_config_names

    print("Model configs (--judge NAME):")
    for name in model_config_names():
        print(f"  {name}")
    print("Shorthands (--judge FORM):")
    width = max(map(len, SHORTHANDS))
    for form, meaning in SHORTHANDS.items():
        print(f"  {form:<{width}}  {meaning}")
    print("A YAML model config is a judge too: --judge path/to/judge.yaml")
    return 0


def run_judges_show(args: argparse.Namespace, argv: Sequence[str]) -> int:
    spec = judge_spec(judge_node(args.judge, cwd=Path.cwd()))
    print(describe({"model_type": spec.model_type, "model_name": spec.model_name, "options": spec.options.written(),
                    "capabilities": dataclasses.asdict(spec.capabilities()),
                    "request_settings": dict(spec.request_settings())}))
    return 0


def run_judges_check(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Print a judge's identity, capabilities and request settings and, by default for a free judge without a local
    engine or with ``--probe``, its answer to one probe prompt."""
    from ruleforge.judges import check
    from ruleforge.judges.batch_api import spend_confirmed

    spec = judge_spec(judge_node(args.judge, cwd=Path.cwd()))
    if args.probe:
        check_batch_roots(spec)
    with guarded_batch_writes(), spend_confirmed() if args.yes_spend else nullcontext():
        report = check(spec, probe=args.probe)
    probe = report.probe
    print(describe({
        "model_type": report.spec.model_type, "model_name": report.spec.model_name,
        "identity": _identity(report.identity), "capabilities": dataclasses.asdict(report.capabilities),
        "request_settings": dict(report.request_settings),
        "probe": None if probe is None else {"text": probe.text, "finish_reason": probe.finish_reason,
                                             "error": probe.error},
        "ok": report.ok,
    }))
    return 0 if report.ok else 1


def run_tasks_list(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from ruleforge.tasks import list_tasks

    found = list_tasks()
    shown = []
    for info in found:
        fields = {item.name: getattr(info, item.name) for item in dataclasses.fields(info)}
        shown.append({name: dict(value) if name == "styles" else value for name, value in fields.items()
                      if value is not None})
    print(describe(shown))
    return 1 if any(info.error is not None for info in found) else 0


def run_version(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from ruleforge.core.provenance import code_state, package_version

    print(describe({"version": package_version(), "code": code_state()}))
    return 0


PLACEHOLDER_RULESET = "<ruleset>"
PLACEHOLDER_FIELDS = {"prompt": "<original_prompt>", "response": "<ai_response>"}


def register_prompts(sub: argparse._SubParsersAction) -> None:
    """Add ``prompts show STYLE``."""
    prompts = sub.add_parser("prompts", help="a task's prompt styles", allow_abbrev=False)
    verbs = prompts.add_subparsers(dest="verb", required=True)
    text = "a prompt style's full-set judge prompt, its ruleset and scenario fields left as placeholders"
    show = subcommand(verbs, "show", run_prompts_show, text)
    show.add_argument("style", help="a style of the task (the Model Spec task's: naive, naive-rules-above, encouraged, "
                                     "forced, forced-grpo)")
    show.add_argument("--task", help="a task's name, or pkg.module:Class (default: modelspec)")
    show.add_argument("--interpretation", help="the interpretation the judge takes (default: No_Interpretation)")


def run_prompts_show(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Print the task's template for ``STYLE``, placeholder-filled."""
    from ruleforge.cli.common import CliError
    from ruleforge.prompts.styles import NO_INTERPRETATION
    from ruleforge.tasks.base import DEFAULT_TASK, TaskError, TaskSettings
    from ruleforge.tasks.registry import load_task

    task = load_task(args.task or DEFAULT_TASK)
    settings = TaskSettings(args.style, judge_type=args.interpretation or NO_INTERPRETATION)
    refusal = task.new_run_refusal(args.style)
    if refusal is not None:
        raise CliError(refusal)
    try:
        task.check(settings)
    except TaskError as error:
        raise CliError(f"{error} (a style that renders only over a levelled ruleset, under the superiors rule "
                       "context, has no placeholder template)" if error.setting == "rule_context" else str(error))
    scenario = {name: PLACEHOLDER_FIELDS.get(name, f"<{name}>") for name in task.scenario_fields}
    sys.stdout.write(task.render(PLACEHOLDER_RULESET, scenario, settings))
    return 0
