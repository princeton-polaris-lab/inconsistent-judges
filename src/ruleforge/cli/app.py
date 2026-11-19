from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from omegaconf.errors import OmegaConfBaseException

from ruleforge.cli import evaluate, grpo, info, optimize
from ruleforge.cli import data
from ruleforge.cli import jobs
from ruleforge.cli import reproduce
from ruleforge.cli import records
from ruleforge.cli import scenarios
from ruleforge.cli.common import CliError, subcommand
from ruleforge.cli.judging import JUDGE_HELP
from ruleforge.cli.slurm import SUBMISSION_OPTIONS
from ruleforge.core.credentials import refused_arguments
from ruleforge.core.errors import RuleForgeError, TracebackError

Handler = Callable[[argparse.Namespace, Sequence[str]], int]
REFUSALS = (RuleForgeError, ValueError, OSError, OmegaConfBaseException)


def _config_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="a YAML config over the packaged one (its relative paths are its "
                                                    "directory's; its defaults list reaches model/ beside it)")


def _slurm_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("SLURM (--slurm PROFILE submits the command instead of running it)")
    group.add_argument("--slurm", metavar="PROFILE", help="a cluster profile's name or YAML file")
    group.add_argument("--cluster-dir", type=Path, help="where profiles are looked for first")
    group.add_argument("--dry-run", action="store_true", help="print each job's request; submit nothing (without "
                                                              "--slurm: print what would run here; run nothing)")
    group.add_argument("--gpus", type=int, help="GPUs per job (default: the command's judge engines')")
    group.add_argument("--timeout-min", type=int, help="each job's wall (default: the profile's)")
    group.add_argument("--cpus", type=int, help="CPUs per job (default: the profile's)")
    group.add_argument("--mem-gb", type=int, help="memory per job (default: the profile's)")
    group.add_argument("--json", action="store_true", help="print the launch as one JSON object")


def _evaluate_parser(sub: argparse._SubParsersAction) -> None:
    parser = subcommand(sub, "evaluate", evaluate.run_evaluate,
                         "the inconsistency rate of a ruleset under a judge, with its confidence interval",
                         overrides=True)
    parser.add_argument("--rules", help="the ruleset (JSONL)")
    parser.add_argument("--scenarios", help="the scenarios (JSONL)")
    parser.add_argument("--judge", help=JUDGE_HELP)
    parser.add_argument("--task", help="a task's name, or pkg.module:Class (default: modelspec)")
    parser.add_argument("--prompt", help="the prompt style (a style of the task; naive, naive-rules-above, encouraged or forced)")
    parser.add_argument("--interpretation", help="the interpretation the judge takes (default: No_Interpretation)")
    parser.add_argument("--rule-context", help="isolated | implicated_ranges | implicated (a range read as its first "
                                               "rule, as the paper's records did) | superiors")
    parser.add_argument("--rule-levels", help="each rule's level (a superiors rule context's)")
    parser.add_argument("--mode", help="both | full | individual_bank")
    parser.add_argument("--draws", type=int, help="decoder draws per scenario (default 5, at least 2)")
    parser.add_argument("--attempts", type=int, help="attempts per scenario and draw (default 30)")
    parser.add_argument("--retry-scope", help="component | scenario_bundle")
    parser.add_argument("--out", help="the campaign root (default: under the artifact root's sampling/, the root "
                                      "$RULEFORGE_ARTIFACT_ROOT; if unset, under "
                                      "./ruleforge-runs)")
    parser.add_argument("--campaign-id", help="the campaign's id (default: a digest of population and settings)")
    parser.add_argument("--condition-id", help="the condition's id (default: named by its fingerprint)")
    parser.add_argument("--final", action="store_true", help="a final campaign (the approved population size)")
    parser.add_argument("--yes-spend", action="store_true", help="let a judge that bills its calls run")
    parser.add_argument("--relocate-batch-root", action="store_true",
                        help="record a batch judge's paid state's new place")
    parser.add_argument("--sweep", action="append", default=[], metavar="KEY=a,b",
                        help="one member per value (repeatable: their product); see `ruleforge.cli.sweep`")
    _config_option(parser)
    _slurm_options(parser)


def _analyze_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("root", type=Path, help="a campaign root this library created, or a configured root this "
                                                "version made (`ruleforge jobs launch campaign`)")
    parser.add_argument("--allow-incomplete", action="store_true", help="analyse the complete conditions alone")
    parser.add_argument("--allow-stack-mismatch", action="store_true")
    parser.add_argument("--rule-set", type=Path, help="the ruleset whose rules the per-rule estimates name")
    parser.add_argument("--exclude-exhausted", nargs=2, metavar=("JUDGE", "SCENARIO"),
                        help="the fallback: drop one exhausted scenario from every arm of one judge")
    parser.add_argument("--n-resamples", type=int)
    parser.add_argument("--ci-level", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-permutations", type=int,
                        help="the paired contrasts' permutations (default: the analysis's, 100,000)")
    parser.add_argument("--replace", action="store_true", help="replace the root's analysis")


def _campaign_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("campaign", help="a campaign root's operations", allow_abbrev=False)
    verbs = parser.add_subparsers(dest="verb", required=True)
    create = subcommand(verbs, "create", evaluate.run_campaign_create,
                         "create a campaign root over a population and its sampling (its conditions: campaign add)")
    create.add_argument("root", type=Path)
    create.add_argument("--scenarios", type=Path, required=True, help="the population the root freezes (JSONL)")
    create.add_argument("--campaign-id", help="the campaign's id (default: a digest of population and sampling)")
    create.add_argument("--draws", type=int, help="decoder draws per scenario (default 5)")
    create.add_argument("--attempts", type=int, help="attempts per scenario and draw (default 30)")
    create.add_argument("--retry-scope", help="component | scenario_bundle")
    create.add_argument("--final", action="store_true", help="a final campaign (the approved population size)")
    add = subcommand(verbs, "add", evaluate.run_campaign_add, "register a condition in a root this library created")
    add.add_argument("root", type=Path)
    add.add_argument("--scenarios", type=Path, required=True, help="the root's population file")
    add.add_argument("--rules", type=Path, required=True)
    add.add_argument("--judge", required=True, help=JUDGE_HELP)
    add.add_argument("--condition-id")
    for name in ("task", "prompt", "interpretation", "rule-context", "mode"):
        add.add_argument(f"--{name}")
    add.add_argument("--rule-levels", type=Path)
    add.add_argument("--extra", action="append", metavar="KEY=VALUE", help="a free identity axis (repeatable)")
    run = subcommand(verbs, "run", evaluate.run_campaign_run, "judge a registered condition's incomplete draws",
                     overrides=True)
    run.add_argument("root", type=Path)
    run.add_argument("condition_id")
    run.add_argument("--judge", required=True, help=JUDGE_HELP)
    run.add_argument("--rules", type=Path, required=True)
    run.add_argument("--scenarios", type=Path, required=True)
    run.add_argument("--rule-levels", type=Path)
    run.add_argument("--task", help="where the task is found (default: the condition's execution record)")
    run.add_argument("--draw-ids", type=int, nargs="+", help="these draws alone (default: every one)")
    run.add_argument("--yes-spend", action="store_true")
    run.add_argument("--relocate-batch-root", action="store_true")
    run.add_argument("--preflight-only", action="store_true",
                     help="every check the run makes before its judge starts, and what it would judge; write nothing")
    status = subcommand(verbs, "status", evaluate.run_campaign_status, "a root's gate, computed and written nowhere")
    status.add_argument("root", type=Path)
    status.add_argument("--check-ledgers", action="store_true")
    _analyze_options(subcommand(verbs, "analyze", evaluate.run_analyze, "write a root's analysis"))
    evaluate.register_campaign(verbs)


def _optimize_parser(sub: argparse._SubParsersAction) -> None:
    parser = subcommand(sub, "optimize", optimize.run_optimize,
                         "RuleForge: amend a ruleset until a judge panel reads it consistently", overrides=True)
    parser.add_argument("--rules", help="the initial ruleset (JSONL)")
    parser.add_argument("--task", help="modelspec | legal")
    parser.add_argument("--judge", help="the judge panel's model")
    parser.add_argument("--agent", help="the agents' model (default: the judge's engine)")
    parser.add_argument("--seeds-good", help="the wildchat seeding's good examples")
    parser.add_argument("--seeds-bad", help="the wildchat seeding's bad examples")
    parser.add_argument("--seeds", help="the single_seed seeding's examples")
    parser.add_argument("--out", help="the run's directory (new, or empty)")
    parser.add_argument("--episodes", type=int, help="at most this many episodes (default 25)")
    parser.add_argument("--wandb", action="store_true", help="log each episode to W&B (the wandb extra)")
    parser.add_argument("--yes-spend", action="store_true", help="let a judge that bills its calls run")
    parser.add_argument("--sweep", action="append", default=[], metavar="KEY=a,b",
                        help="one run per value under <out>/<key>.<value> (agent.seed=1..10)")
    _config_option(parser)
    _slurm_options(parser)
    optimize.register(parser)


def _grpo_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("grpo", help="GRPO: rows, training, merging, parity, the report on its saves", allow_abbrev=False)
    verbs = parser.add_subparsers(dest="verb", required=True)
    for name in ("rows", "merge"):
        _function_parser(verbs, name, grpo.COMMANDS[name])
    anchors = verbs.add_parser("anchors", help="anchor sets: route a top-up, build", allow_abbrev=False)
    stages = anchors.add_subparsers(dest="stage", required=True)
    for name in ("build", "route"):
        _function_parser(stages, name, grpo.COMMANDS[f"anchors {name}"])
    train = subcommand(verbs, "train", grpo.run_train, "train a judge with GRPO (configs/grpo/base.yaml)",
                       overrides=True)
    train.add_argument("--run-name")
    train.add_argument("--model", help="the base checkpoint directory (model.path)")
    train.add_argument("--output-root", help="where the run's directory is made")
    train.add_argument("--resume", help="latest, or a checkpoint directory (resume_from_checkpoint)")
    train.add_argument("--preflight-only", action="store_true", help="check everything; load no model")
    train.add_argument("--allow-unscreened-rows", action="store_true",
                       help="train on rows attested by the second pass alone (data.allow_unscreened_rows)")
    _config_option(train)
    _slurm_options(train)
    grpo.register(verbs)


def _function_handler(command: "grpo.FunctionCommand") -> Handler:
    return lambda args, argv: grpo.run_function(args, argv, command)


def _function_parser(group: argparse._SubParsersAction, name: str, command: "grpo.FunctionCommand") -> None:
    found = subcommand(group, name, _function_handler(command), command.summary,
                       overrides=True)
    grpo.add_function_arguments(found, command)
    _config_option(found)
    _slurm_options(found)


def _info_parsers(sub: argparse._SubParsersAction) -> None:
    judges = sub.add_parser("judges", help="the model configs a --judge names; a judge's identity; a probe",
                            allow_abbrev=False)
    verbs = judges.add_subparsers(dest="verb", required=True)
    subcommand(verbs, "list", info.run_judges_list,
               "the model configs' names (a checkout's configs/model, then the package's), and the shorthands")
    subcommand(verbs, "show", info.run_judges_show, "a judge's options and capabilities").add_argument("judge")
    check = subcommand(verbs, "check", info.run_judges_check, "a judge's identity, and one probe prompt's answer")
    check.add_argument("judge")
    check.add_argument("--probe", action=argparse.BooleanOptionalAction, default=None,
                       help="ask the probe (default: for a free judge with no local engine)")
    check.add_argument("--yes-spend", action="store_true", help="let the probe of a judge that bills its calls run")
    tasks = sub.add_parser("tasks", help="the tasks a --task finds", allow_abbrev=False)
    subcommand(tasks.add_subparsers(dest="verb", required=True), "list", info.run_tasks_list, "every task, by name")
    subcommand(sub, "version", info.run_version, "the package's version and code state")
    info.register_prompts(sub)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ruleforge", description="RuleForge's operations, each run here or "
                                     "submitted to SLURM (--slurm PROFILE); `ruleforge COMMAND --help` for each.",
                                     allow_abbrev=False)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging, and a refusal's traceback")
    sub = parser.add_subparsers(dest="command", required=True)
    _evaluate_parser(sub)
    _campaign_parser(sub)
    _analyze_options(subcommand(sub, "analyze", evaluate.run_analyze, "write a campaign root's analysis"))
    evaluate.register_bank(sub)
    scenarios.register(sub)
    _optimize_parser(sub)
    _grpo_parser(sub)
    _info_parsers(sub)
    reproduce.register(sub)
    jobs.register(sub)
    data.register(sub)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command ``argv`` names (default: this process's arguments) and return its exit status."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if refused_arguments(argv, "ruleforge"):
        return 2
    cut = argv.index("--") if "--" in argv else len(argv)
    parser = build_parser()
    args = parser.parse_args(argv[:cut])
    args.overrides = argv[cut + 1:]
    if args.overrides and not args.takes_overrides:
        parser.error(f"{args.command} takes no `-- key=value` overrides")
    opened = len(records.OPENED)
    try:
        _refuse_slurm_options_alone(args)
        return args.handler(args, argv)
    except REFUSALS as error:
        if args.verbose or isinstance(error, TracebackError):
            raise
        print(f"ruleforge {args.command}: {type(error).__name__}: {error}", file=sys.stderr)
        return 1 if len(records.OPENED) > opened or _unfinished(error) else 2


def _refuse_slurm_options_alone(args: argparse.Namespace) -> None:
    """Refuse a submission option given without --slurm; --dry-run is also the command's own."""
    if not hasattr(args, "slurm") or args.slurm is not None:
        return
    given = [flag for flag in SUBMISSION_OPTIONS
             if flag != "--dry-run" and getattr(args, flag[2:].replace("-", "_")) not in (None, False)]
    if given:
        one = len(given) == 1
        raise CliError(f"{' and '.join(given)} {'is' if one else 'are'} for --slurm PROFILE, and without --slurm the "
                       f"command runs here: give a profile, or drop {'it' if one else 'them'}")


def _unfinished(error: BaseException) -> bool:
    runner = sys.modules.get("ruleforge.evaluate.runner")
    batches = sys.modules.get("ruleforge.judges.batch_api")
    return ((runner is not None and isinstance(error, runner.TransportExhausted))
            or (batches is not None and isinstance(error, batches.BatchExecutionError)))
