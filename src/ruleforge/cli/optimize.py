from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from ruleforge.cli import records
from ruleforge.cli.common import Assignment, CliError, flag_settings, note_fake, rebased, setting, subcommand
from ruleforge.cli.compose import compose, parse_overrides, refuse_conflicts, without_keys
from ruleforge.cli.judging import engine_gpus, judge_flag, judge_spec
from ruleforge.cli.schema import Operation, Parameter, bind
from ruleforge.cli.sweep import (CONDITION, Axes, Member, Place, check_places, fill, label, members, parse,
                                 refuse_equal_values)

JUDGE_NODE, AGENT_NODE = "judge_model", "agent_model"
#: Each value flag: its config key, and whether it is a path.
FLAGS = {"rules": ("rule_amendment.initial_rule_set_path", True), "task": ("task_spec", False),
         "seeds_good": ("scenario_generation.good_seed_scenario_path", True),
         "seeds_bad": ("scenario_generation.bad_seed_scenario_path", True),
         "seeds": ("scenario_generation.seed_scenario_path", True), "out": ("out", True),
         "episodes": ("max_episodes", False)}
AXES = Axes(keys={}, prefixes={"judge.": (f"{JUDGE_NODE}.", CONDITION), "agent.": (f"{AGENT_NODE}.", CONDITION),
                               "scenario_generation.": ("scenario_generation.", CONDITION),
                               "rule_amendment.": ("rule_amendment.", CONDITION),
                               "eval_cfg.": ("eval_cfg.", CONDITION)})


def operation() -> Operation:
    """Return ``optimize``'s schema: its path keys bound to the API's path parameters."""
    from ruleforge.optimize import SEED_PATH_KEYS
    from ruleforge.optimize.api import optimize

    seeds = tuple(Parameter(f"scenario_generation.{key}", optimize, "seeds", entry=True)
                  for key in SEED_PATH_KEYS.values())
    return Operation("optimize", parameters=(
        *bind(optimize, only=("rules", "out"), keys={"rules": "rule_amendment.initial_rule_set_path"}), *seeds),
        judges=(JUDGE_NODE, AGENT_NODE))


def convert(cfg: DictConfig) -> dict[str, Any]:
    """Return ``optimize``'s arguments from a composed config."""
    from ruleforge.optimize import EVAL_FIELDS, OptimizeConfig, RAConfig, SGConfig, seed_paths

    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    generation = container["scenario_generation"]
    amendment = container["rule_amendment"]
    config = OptimizeConfig(
        max_episodes=container["max_episodes"],
        scenario_generation=SGConfig(**{name: value for name, value in generation.items()
                                        if not name.endswith("_path")}),
        rule_amendment=RAConfig(**{name: value for name, value in amendment.items()
                                   if name != "initial_rule_set_path"}),
        **{name: (tuple(value) if name == "judge_types" else value) for name, value in container["eval_cfg"].items()
           if name in EVAL_FIELDS},
        num_tries=container["num_tries"], max_retry_seconds=container["max_retry_seconds"])
    agent = cfg.get(AGENT_NODE)
    return {"rules": amendment["initial_rule_set_path"], "task": container["task_spec"],
            "judge": judge_spec(cfg[JUDGE_NODE]), "agent": None if agent is None else judge_spec(agent),
            "seeds": seed_paths(cfg.scenario_generation), "out": Path(container["out"]), "config": config}


def _flags(args: argparse.Namespace, cwd: Path) -> list[Assignment]:
    judges = (judge_flag(args, name, node, cwd) for name, node in (("judge", JUDGE_NODE), ("agent", AGENT_NODE)))
    found = [judge for judge in judges if judge is not None] + flag_settings(args, FLAGS, cwd)
    if args.wandb:
        found.append(setting("wandb.enabled", True, "--wandb"))
    return found


def _out(member: Member, out: str, *, swept: bool) -> Path:
    values = member.at(CONDITION)
    if not swept:
        return Path(out)
    if any(f"{{{axis.name}}}" in out for axis, _ in values):
        return Path(fill(out, values, AXES, what="--out"))
    return Path(out).joinpath(*(f"{axis.name}.{label(axis, value, AXES)}" for axis, value in values))


def _adding(key: str) -> str:
    return "++" if key.startswith((f"{JUDGE_NODE}.", f"{AGENT_NODE}.")) else ""


def run_optimize(args: argparse.Namespace, argv: Sequence[str]) -> int:
    cwd = Path.cwd()
    axes = parse(args.sweep, AXES)
    flags = _flags(args, cwd)
    overrides = parse_overrides(args.overrides, operation(), cwd)
    refuse_conflicts(flags, overrides, [Assignment(axis.key, "", f"{axis.name}=...") for axis in axes])
    if any(axis.key.startswith(f"{AGENT_NODE}.") for axis in axes):
        # Whether the run has its own agents is the composed config's to say, read before the sweep's own keys.
        unswept = compose(operation(), "optimize.yaml", config_file=args.config, layers=(flags + overrides,), cwd=cwd)
        if unswept.get(AGENT_NODE) is None:
            raise CliError("--sweep agent.<key> sweeps the agents' model config, and this run configures no agent "
                           "(no --agent, and neither the --config file nor an override selects model@agent_model), "
                           "so the judge's engine serves the agents: sweep judge.<key> (the one engine), or give "
                           "--agent")
    runs = []
    for member in members(axes):
        layer = [Assignment(axis.key, f"{_adding(axis.key)}{axis.key}={value}", f"{axis.name}={value}")
                 for axis, value in member.values]
        cfg = compose(operation(), "optimize.yaml", config_file=args.config, layers=(flags + overrides, layer),
                      cwd=cwd)
        call = convert(cfg)
        call["out"] = Path(rebased(str(_out(member, str(call["out"]), swept=bool(axes))), cwd))
        runs.append((member, cfg, call))
    refuse_equal_values([(member, cfg) for member, cfg, _ in runs], axes)
    check_places(places(runs, launching=args.slurm is not None and not args.dry_run),
                 way_out="give each run a directory of its own, apart from every other's, from the W&B directory "
                         "beside it, from the batch roots and from the launch records, and each batch root a place "
                         "apart from every W&B directory and record (an --out template naming {key} for each swept "
                         "key, through no '..' or link that brings two together)")
    if args.slurm is not None:
        return _submit(args, argv, runs)
    if args.dry_run:
        for _, cfg, call in runs:
            records.print_dry_run(call["out"], cfg)
        return 0
    from ruleforge.judges.batch_api import spend_confirmed
    from ruleforge.optimize.judging import refuse_unconfirmed

    with spend_confirmed() if args.yes_spend else nullcontext():
        for _, _, call in runs:  # every run's judges before any run's record, so a confirmed retry can take them
            refuse_unconfirmed(call["judge"], call["agent"])
        for member, cfg, call in runs:
            _run(call, cfg, argv, args.verbose)
    return 0


def _run(call: dict[str, Any], cfg: DictConfig, argv: Sequence[str], verbose: bool) -> None:
    from ruleforge.optimize.api import optimize

    record = records.open_record(call["out"], "optimize", argv, cfg)
    with records.logging_to(record, verbose=verbose), _wandb(cfg, call["out"]) as logger, \
            records.guarded_batch_writes():
        run = optimize(call["rules"], task=call["task"], judge=call["judge"], agent=call["agent"], seeds=call["seeds"],
                       out=call["out"], config=call["config"], logger=logger)
    print(records.describe({"out": str(run.out), "last_episode": run.last_episode, "stopped_by": list(run.stopped_by),
                            "ruleset": str(run.ruleset_path), "reward": run.reward}), flush=True)
    note_fake(spec.model_type for spec in (call["judge"], call["agent"]) if spec is not None)


WANDB_DIRNAME = "wandb"
WANDB_KIND = "W&B directory"


def _wandb_parent(out: Path) -> Path:
    return Path(out).parent


def places(runs: Sequence[tuple[Member, DictConfig, dict[str, Any]]], *, launching: bool) -> list[Place]:
    """Return every place the run writes, or its jobs do, each with its check."""
    found: list[Place] = []
    for member, cfg, call in runs:
        out = call["out"]
        found += records.target_places(member, out, "--out")
        if cfg.wandb.enabled:
            found.append(Place(member, "W&B directory", _wandb_parent(out) / WANDB_DIRNAME, own=False, kind=WANDB_KIND,
                               protect=lambda out=out: _check_wandb_dir(out)))
        found += records.batch_places(member, (call["judge"], call["agent"]), home=out)
    if launching:
        from ruleforge.cli.slurm import launch_places

        found += launch_places()
    return found


def _check_wandb_dir(out: Path) -> None:
    from ruleforge.evaluate.api import refuse_protected

    refuse_protected(_wandb_parent(out) / WANDB_DIRNAME, "the run's W&B directory")


@contextmanager
def _wandb(cfg: DictConfig, out: Path) -> Iterator[Optional[Any]]:
    if not cfg.wandb.enabled:
        yield None
        return
    import wandb

    from ruleforge.optimize import WandbLogger

    with wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, group=cfg.wandb.group, name=out.name,
                    dir=str(_wandb_parent(out)), config=OmegaConf.to_container(cfg, resolve=True)):
        yield WandbLogger()


def _gpus(judge: Any, agent: Optional[Any]) -> int:
    return engine_gpus(judge) + (0 if agent is None or agent == judge else engine_gpus(agent))


def _submit(args: argparse.Namespace, argv: Sequence[str], runs: Sequence[tuple[Member, DictConfig, dict]]) -> int:
    from ruleforge.cli import slurm
    from ruleforge.launch.profile import find_profile

    profile = find_profile(args.slurm, args.cluster_dir)
    base = slurm.passthrough_argv(argv, drop={**slurm.SUBMISSION_OPTIONS, "--sweep": 1, "--out": 1})
    specs = []
    for index, (member, _, call) in enumerate(runs):
        gpus = args.gpus if args.gpus is not None else _gpus(call["judge"], call["agent"])
        cut = base.index("--") if "--" in base else len(base)
        # The job's directory is its --out alone: an `-- out=...` override would set it twice.
        tail = [*without_keys(base[cut + 1:], ("out",)),
                *(f"{_adding(axis.key)}{axis.key}={value}" for axis, value in member.values)]
        argv_ = [*base[:cut], "--out", str(call["out"]), *(["--", *tail] if tail else [])]
        specs.append(slurm.job(f"optimize-{index}", argv_, profile, slurm.request(args, gpus)))
    return slurm.submit(specs, profile, verb="optimize", dry_run=args.dry_run, as_json=args.json)


def register(parser: argparse.ArgumentParser) -> None:
    """Add ``optimize select``: the best seed of RuleForge sweeps."""
    defaults = {action.dest: action.default for action in parser._actions
                if action.dest != "help" and not isinstance(action, argparse._SubParsersAction)}
    verbs = parser.add_subparsers(dest="verb", required=False)
    text = "pick the best seed of RuleForge sweep directories by their last episode's effective reward"
    select = subcommand(verbs, "select", run_select, text, optimize_defaults=defaults)
    select.add_argument("--sweep-dir", action="append", required=True, dest="sweep_dirs",
                        help="a sweep's directory: one run per seed, a job directory named by its number or an "
                             "optimize --sweep member (repeatable)")
    select.add_argument("--output", type=Path, default=None,
                        help="the selection table (default: <first sweep dir>/selection.json)")
    select.add_argument("--min-last-episode", type=int, default=1,
                        help="exclude runs whose last complete episode is below this; episode 0's ruleset is the "
                             "unamended initial one (default: 1)")


def run_select(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Write the selection table as JSON and print it."""
    import json

    from ruleforge.optimize.select import NOT_A_RUN, RUN_NAMING, sweep_table

    given = sorted(name for name, default in args.optimize_defaults.items() if getattr(args, name) != default)
    if given:
        raise CliError(f"optimize select takes none of optimize's own options (given: "
                       f"{', '.join('--' + name.replace('_', '-') for name in given)}): it reads sweeps here and "
                       "submits nothing")
    sweep_dirs = [records.absolute_path(directory) for directory in args.sweep_dirs]
    output = records.absolute_path(args.output) or sweep_dirs[0] / "selection.json"
    records.check_outputs([output], way_out="give --output a place of its own, apart from its run records")
    rows, notes = sweep_table(sweep_dirs, min_last_episode=args.min_last_episode)
    for note in notes:
        print(note)
    if not rows:
        found = any(f": {NOT_A_RUN}: " not in note for note in notes)
        raise CliError("No completed runs found in the given sweep dir(s)" + ("." if found else f": {RUN_NAMING}."))
    records.open_record(output, "optimize select", argv, None)
    winner = rows[0]
    output.write_text(json.dumps({"winner": winner, "all_seeds": rows}, indent=2))
    header = (f"{'seed':>4} {'job':>3} {'last_ep':>7} {'effective':>10} {'entropy':>10} "
              f"{'penalty':>10} {'peak_ep':>7} {'peak_eff':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        job = "-" if row["job_num"] is None else row["job_num"]
        print(f"{str(row['seed']):>4} {job:>3} {row['last_episode']:>7} "
              f"{row['effective_reward']:>10.4f} {row['entropy_reward']:>10.4f} "
              f"{row['regularization_penalty']:>10.4f} {row['peak_episode']:>7} "
              f"{row['peak_effective_reward']:>9.4f}")
    job = "" if winner["job_num"] is None else f"job {winner['job_num']}, "
    print(f"\nWinner: seed={winner['seed']} ({job}episode {winner['last_episode']})")
    print(f"Ruleset: {winner['optimal_rule_set_path']}")
    print(f"Selection table written to {output}")
    return 0
