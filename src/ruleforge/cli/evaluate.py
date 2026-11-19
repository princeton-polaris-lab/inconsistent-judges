from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ruleforge.cli import records
from ruleforge.cli.common import Assignment, CliError, flag_settings, note_fake, quoted, rebased, setting, subcommand
from ruleforge.cli.compose import compose, parse_overrides, refuse_conflicts, without_keys
from ruleforge.cli.judging import (JUDGE_HELP, engine_gpus, is_named, judge_assignment, judge_flag, judge_node,
                                   judge_spec)
from ruleforge.cli.schema import PATH, Operation, bind
from ruleforge.cli.sweep import (CAMPAIGN, CONDITION, Axes, Member, Place, check_places, fill, members, parse,
                                 refuse_collisions, refuse_equal_values)
from ruleforge.cli.sweep import segments as name_segments

#: Each value flag of ``evaluate``: its config key, and whether it is a path.
FLAGS = {
    "rules": ("rule_set_path", True), "scenarios": ("scenario_set_path", True), "task": ("task_spec", False),
    "prompt": ("eval_cfg.reasoning_prompt_style", False), "interpretation": ("interpretation", False),
    "rule_context": ("eval_cfg.individual_rule_context", False), "rule_levels": ("rule_levels_path", True),
    "mode": ("evaluation_mode", False), "draws": ("sampling.num_decoder_draws", False),
    "attempts": ("sampling.max_semantic_attempts", False), "retry_scope": ("sampling.retry_scope", False),
    "out": ("campaign_root", True),
    "campaign_id": ("campaign_id", False), "condition_id": ("condition_id", False),
}
RUN_OVERRIDES = frozenset({"execution", "max_transport_retries"})
SWITCHES = {"final": ("sampling.campaign_mode", "final"), "yes_spend": ("confirm_spend", True),
            "relocate_batch_root": ("relocate_batch_root", True)}
JUDGE_NODE = "judge_model"


def _judge_label(name: str, value: str) -> Optional[str]:
    if name != "judge":
        return None
    if is_named(value):
        return value
    path = Path(value)
    if path.suffix in (".yaml", ".yml") or path.is_file():
        return path.stem
    return judge_spec(judge_node(value, cwd=Path.cwd())).model_name


AXES = Axes(
    keys={
        "prompt": ("eval_cfg.reasoning_prompt_style", CONDITION), "interpretation": ("interpretation", CONDITION),
        "rule_context": ("eval_cfg.individual_rule_context", CONDITION), "mode": ("evaluation_mode", CONDITION),
        "task": ("task_spec", CONDITION), "judge": (JUDGE_NODE, CONDITION), "rules": ("rule_set_path", CONDITION),
        "rule_levels": ("rule_levels_path", CONDITION),
        "draws": ("sampling.num_decoder_draws", CAMPAIGN), "max_attempts": ("sampling.max_semantic_attempts", CAMPAIGN),
        "retry_scope": ("sampling.retry_scope", CAMPAIGN),
        "forced_json": ("sampling.forced_json_requirement", CAMPAIGN), "truncation": ("sampling.truncation", CAMPAIGN),
        "strict_think": ("sampling.strict_think", CAMPAIGN),
        "block_size": ("sampling.scenarios_per_request_block", CAMPAIGN),
        "seed_base": ("sampling.generation_seed_base", CAMPAIGN), "campaign_mode": ("sampling.campaign_mode", CAMPAIGN),
    },
    prefixes={"extra.": ("condition_extra.", CONDITION), "bootstrap.": ("sampling.bootstrap.", CAMPAIGN)},
    files=("rules", "rule_levels"), label=_judge_label,
)
#: The campaign-level sweep axes whose values a root's ``evaluate`` job takes as a flag (the rest as overrides).
AXIS_FLAGS = {"draws": "--draws", "max_attempts": "--attempts", "retry_scope": "--retry-scope"}


def operation() -> Operation:
    """Return ``evaluate``'s schema: its path keys bound to the API's path parameters."""
    from ruleforge.evaluate.api import ConditionSettings
    from ruleforge.evaluate.api import evaluate as evaluate_api

    return Operation("evaluate", parameters=(
        *bind(evaluate_api, only=("rules", "scenarios", "out"),
              keys={"rules": "rule_set_path", "scenarios": "scenario_set_path", "out": "campaign_root"}),
        *bind(ConditionSettings, only=("rule_levels",), keys={"rule_levels": "rule_levels_path"}),
    ), judges=(JUDGE_NODE,))


@dataclass
class Call:
    """One ``evaluate`` call, converted from a composed config."""

    cfg: DictConfig
    rules: Any
    scenarios: Any
    judge: Any
    sampling: Any
    condition: Any
    execution: Any
    out: Optional[str]
    campaign_id: Optional[str]
    condition_id: Optional[str]
    confirm_spend: bool
    relocate_batch_root: bool


def convert(cfg: DictConfig) -> Call:
    """Return the API settings a composed ``evaluate`` config describes."""
    from ruleforge.evaluate.api import Approvals, ConditionSettings, SamplingSettings
    from ruleforge.rulesets import Ruleset
    from ruleforge.evaluate.identity import BootstrapSettings
    from ruleforge.evaluate.packing import ExecutionSettings
    from ruleforge.judges.batch_api import config_confirms_spend
    from ruleforge.scenarios import ScenarioSet

    sampling_cfg = cfg.sampling
    sampling = SamplingSettings(
        draws=sampling_cfg.num_decoder_draws, max_attempts=sampling_cfg.max_semantic_attempts,
        retry_scope=sampling_cfg.retry_scope,
        forced_json=sampling_cfg.forced_json_requirement, truncation=sampling_cfg.truncation,
        strict_think=sampling_cfg.strict_think, block_size=sampling_cfg.scenarios_per_request_block,
        seed_base=sampling_cfg.generation_seed_base,
        bootstrap=BootstrapSettings(**OmegaConf.to_container(sampling_cfg.bootstrap, resolve=True)),
        campaign_mode=sampling_cfg.campaign_mode, approved_scenario_count=sampling_cfg.approved_scenario_count,
        max_transport_retries=cfg.max_transport_retries)
    condition = ConditionSettings(
        task=cfg.task_spec, prompt=cfg.eval_cfg.reasoning_prompt_style, interpretation=cfg.interpretation,
        rule_context=cfg.eval_cfg.individual_rule_context, rule_levels=cfg.rule_levels_path,
        mode=cfg.evaluation_mode, extra=OmegaConf.to_container(cfg.condition_extra, resolve=True),
        approvals=Approvals(rule_set_sha256=cfg.approved_rule_set_sha256,
                            rule_levels_sha256=cfg.approved_rule_levels_sha256,
                            adapter_sha256=cfg.approved_adapter_sha256,
                            model_weights_sha256=cfg.approved_model_weights_sha256))
    return Call(cfg=cfg, rules=Ruleset.load(cfg.rule_set_path),
                scenarios=ScenarioSet.load(cfg.scenario_set_path, expect_sha256=cfg.approved_scenario_sha256),
                judge=judge_spec(cfg.judge_model), sampling=sampling, condition=condition,
                execution=ExecutionSettings(**OmegaConf.to_container(cfg.execution, resolve=True)),
                out=cfg.campaign_root, campaign_id=cfg.campaign_id, condition_id=cfg.condition_id,
                confirm_spend=config_confirms_spend(cfg), relocate_batch_root=bool(cfg.relocate_batch_root))


def flag_assignments(args: argparse.Namespace, cwd: Path) -> list[Assignment]:
    """Return the assignments ``evaluate``'s flags make."""
    judge = judge_flag(args, "judge", JUDGE_NODE, cwd)
    found = ([] if judge is None else [judge]) + flag_settings(args, FLAGS, cwd)
    for name, (key, value) in SWITCHES.items():
        if getattr(args, name):
            found.append(setting(key, value, f"--{name.replace('_', '-')}"))
    return found


def member_assignments(member: Member, cwd: Path) -> list[Assignment]:
    """Return the assignments a sweep member's values make, after every other layer."""
    found = []
    for axis, value in member.values:
        spelling = f"{axis.name}={value}"
        if axis.key == JUDGE_NODE:
            found.append(judge_assignment(value, spelling, node=JUDGE_NODE, cwd=cwd,
                                          name=f"__ruleforge_judge_{member.index}__"))
        elif axis.name in AXES.files:
            found.append(setting(axis.key, rebased(value, cwd), spelling))
        elif axis.level == CONDITION:
            adding = "++" if axis.key.startswith("condition_extra.") else ""
            found.append(Assignment(key=axis.key, override=f"{adding}{axis.key}={quoted(value)}", spelling=spelling))
        else:
            found.append(Assignment(key=axis.key, override=f"{axis.key}={value}", spelling=spelling))
    return found


@dataclass
class Planned:
    """A member as it runs: its call, root, campaign id and condition id."""

    member: Member
    call: Call
    root: Path
    campaign_id: str
    condition_id: Optional[str]
    segments: str = ""
    judge_file: Optional[Path] = field(default=None, repr=False)
    #: The condition as a draft of its campaign named and checked it (:func:`name_conditions`).
    drafted: Any = field(default=None, repr=False)

    def entry(self) -> tuple[Optional[str], Any, Any, Any]:
        return (self.condition_id, self.call.rules, self.call.judge, self.call.condition)


def plan(member: Member, args: argparse.Namespace, layer: Sequence[Assignment], cwd: Path) -> Planned:
    """Compose and convert ``member``'s config, and name its root, campaign and condition."""
    from ruleforge.evaluate.api import default_campaign_id, default_root

    cfg = compose(operation(), "evaluate.yaml", config_file=args.config, cwd=cwd,
                  layers=(layer, member_assignments(member, cwd)))
    call = convert(cfg)
    campaign_level, condition_level = member.at(CAMPAIGN), member.at(CONDITION)
    if call.campaign_id is not None:
        campaign_id = fill(call.campaign_id, campaign_level, AXES, what="--campaign-id") if campaign_level \
            else call.campaign_id
    else:
        campaign_id = default_campaign_id(call.scenarios.manifest, call.sampling.config()) + name_segments(
            campaign_level, AXES, separator="-")
    out = call.out
    if out is not None and campaign_level:
        out = fill(out, campaign_level, AXES, what="--out")
    root = Path(rebased(str(out), cwd)) if out is not None else Path(rebased(str(default_root(campaign_id)), cwd))
    condition_id, segments = call.condition_id, ""
    if condition_level:
        if condition_id is not None:
            condition_id = fill(condition_id, condition_level, AXES, what="--condition-id")
        else:
            segments = name_segments(condition_level, AXES, separator="/")
    return Planned(member=member, call=call, root=root.resolve(), campaign_id=campaign_id,
                   condition_id=condition_id, segments=segments)


def roots_of(planned: Sequence[Planned]) -> dict[Path, list[Planned]]:
    """The members by root, refused when one root's members differ in what its campaign is."""
    roots: dict[Path, list[Planned]] = {}
    for item in planned:
        roots.setdefault(item.root, []).append(item)
    for root, group in roots.items():
        first = group[0]
        for other in group[1:]:
            if (other.campaign_id, other.call.sampling, other.call.scenarios.path) != (
                    first.campaign_id, first.call.sampling, first.call.scenarios.path):
                raise CliError(f"sweep members {first.member.describe()} and {other.member.describe()} share the root "
                               f"{root} but not its campaign (its id, population or sampling); name their roots apart "
                               "with an --out template naming each swept campaign-level key")
    return roots


def places(roots: Mapping[Path, Sequence[Planned]], *, launching: bool) -> list[Place]:
    """Return every place the command writes, or its jobs do, each with its check."""
    from ruleforge.evaluate.sharding import campaign_lock_path

    found: list[Place] = []
    for root, group in roots.items():
        first = group[0].member
        found += records.target_places(first, root, "root", campaign_root=True)
        lock = campaign_lock_path(root)
        found.append(Place(first, "campaign lock", lock, own=False,
                           protect=records.protected(lock, "the campaign lock beside the root")))
        for item in group:
            found += records.batch_places(item.member, (item.call.judge,), campaign=root)
    if launching:
        from ruleforge.cli.slurm import launch_places

        found += launch_places()
    return found


def name_conditions(roots: dict[Path, list[Planned]]) -> None:
    """Name every member's condition and check its registration, and its spend, on a draft of its campaign, writing
    nothing."""
    from ruleforge.evaluate.api import Campaign, refuse_unconfirmed_spend

    drafts = {}
    for root, group in roots.items():
        first = group[0]
        drafts[root] = draft = Campaign.draft(root, campaign_id=first.campaign_id, scenarios=first.call.scenarios,
                                              sampling=first.call.sampling)
        for item in group:
            item.drafted = draft.prepare(item.condition_id, rules=item.call.rules, judge=item.call.judge,
                                         settings=item.call.condition, segments=item.segments)
            item.condition_id = item.drafted.condition_id
    refuse_collisions([(item.member, (str(item.root), str(item.condition_id)))
                       for group in roots.values() for item in group],
                      way_out="name them apart with a template (--out, --campaign-id or --condition-id naming "
                              "{key} for each swept key) or give them distinct values")
    for root, group in roots.items():
        identities: dict[str, Planned] = {}
        for item in group:
            other = identities.setdefault(item.drafted.spec.fingerprint, item)
            if other is not item:
                raise CliError(f"sweep members {other.member.describe()} and {item.member.describe()} are one "
                               f"condition (fingerprint {item.drafted.spec.fingerprint[:12]}...): they judge the same "
                               "rules, judge, task settings, mode and extra under two names; sweep values that differ")
        draft = drafts[root]
        draft.register([item.drafted for item in group])
        unconfirmed = [item.drafted for item in group
                       if not item.call.confirm_spend and item.drafted.judge.capabilities().spends_money]
        if unconfirmed:  # the root's own refusals first, as creating it makes them
            Campaign.check_create(draft.root, campaign_id=draft.campaign_id, scenarios=draft.scenarios,
                                  sampling=draft.sampling)
        for condition in unconfirmed:
            refuse_unconfirmed_spend(condition, analysis_read=True)


def configs_text(group: Sequence[Planned]) -> Optional[DictConfig]:
    """Return what a run record keeps of a root's members' configs."""
    if len(group) == 1:
        return group[0].call.cfg
    return OmegaConf.create({item.member.describe(): _resolved(item.call.cfg) for item in group})


def _resolved(cfg: DictConfig) -> Any:
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except OmegaConfBaseException:
        return cfg


def eval_summary(result: Any) -> dict[str, Any]:
    """Return an ``EvalResult`` as the command prints it."""
    estimates = result.estimates
    return {
        "root": str(result.root), "campaign_id": result.campaign_id, "condition_id": result.condition_id,
        "mode": result.mode, "draws": {str(draw): state for draw, state in result.draws.items()},
        "estimates": None if estimates is None else {
            "rate": dataclasses.asdict(estimates.rate), "n_scenarios": estimates.n_scenarios,
            "draws": estimates.draws, "same_call": None if estimates.same_call is None else dataclasses.asdict(
                estimates.same_call), "file": str(estimates.estimates_path), "sha256": estimates.estimates_sha256},
        "excluded": None if result.excluded is None else dict(result.excluded),
    }


def campaign_summary(result: Any) -> dict[str, Any]:
    """Return a ``CampaignResult`` as the command prints it."""
    return {"estimates": None if result.path is None else str(result.path),
            "bootstrap": None if result.path is None else result.payload.get("bootstrap"),
            "conditions": {cid: eval_summary(item) for cid, item in result.conditions.items()},
            "excluded_conditions": dict(result.excluded_conditions),
            "scenario_exclusion": None if result.scenario_exclusion is None else dict(result.scenario_exclusion)}


def _print(payload: Any) -> None:
    print(records.describe(payload), flush=True)


def _print_result(result: Any) -> None:
    from ruleforge.evaluate.campaign import load_conditions
    from ruleforge.evaluate.results import CampaignResult

    whole = isinstance(result, CampaignResult)
    shown = list(result.conditions.values()) if whole else [result]
    _print(campaign_summary(result) if whole else eval_summary(result))
    registered: dict[str, Any] = {}
    for root in {item.root for item in shown}:
        registered.update(load_conditions(root))
    note_fake(registered[item.condition_id].model_type for item in shown)


def run_evaluate(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Run ``evaluate``: every member named and checked first, then run here or submitted."""
    cwd = Path.cwd()
    axes = parse(args.sweep, AXES)
    flags = flag_assignments(args, cwd)
    overrides = parse_overrides(args.overrides, operation(), cwd)
    refuse_conflicts(flags, overrides, [Assignment(axis.key, "", f"{axis.name}=...", group=axis.key == JUDGE_NODE)
                                        for axis in axes])
    planned = [plan(member, args, [*flags, *overrides], cwd) for member in members(axes)]
    refuse_equal_values([(item.member, item.call.cfg) for item in planned], axes)
    roots = roots_of(planned)
    check_places(places(roots, launching=args.slurm is not None and not args.dry_run),
                 way_out="give each campaign a root of its own, apart from every other root, its run records and "
                         "campaign lock, every other member's batch root and the launch records, and each batch root "
                         "a place apart from every lock and record (an --out template naming each swept "
                         "campaign-level key, through no '..' or link that brings two together)")
    swept_conditions = any(axis.level == CONDITION for axis in axes)
    name_conditions(roots)
    if args.slurm is not None:
        return _submit_evaluate(args, argv, roots, swept_conditions)
    if args.dry_run:
        for root, group in roots.items():
            records.print_dry_run(root, configs_text(group))
        return 0
    status = 0
    for root, group in roots.items():
        status = max(status, _evaluate_root(root, group, swept_conditions, argv, args.verbose))
    return status


def _complete(draws: Any) -> bool:
    return all(state == "complete" for state in draws.values())


def _evaluate_root(root: Path, group: Sequence[Planned], swept_conditions: bool, argv: Sequence[str],
                   verbose: bool) -> int:
    from ruleforge.evaluate.api import Campaign
    from ruleforge.evaluate.api import evaluate as evaluate_api

    first = group[0]
    campaign = Campaign.create(root, campaign_id=first.campaign_id, scenarios=first.call.scenarios,
                               sampling=first.call.sampling)
    record = records.open_record(root, "evaluate", argv, configs_text(group), campaign_root=True)
    with records.logging_to(record, verbose=verbose):
        if len(group) == 1 and not swept_conditions:
            call = first.call
            result = evaluate_api(call.rules, call.scenarios, call.judge, out=root, sampling=call.sampling,
                                  condition=call.condition, campaign_id=first.campaign_id,
                                  condition_id=first.condition_id, confirm_spend=call.confirm_spend,
                                  execution=call.execution, relocate_batch_root=call.relocate_batch_root)
            _print_result(result)
            return 0 if _complete(result.draws) else 1
        swept = campaign.sweep([item.entry() for item in group], execution=first.call.execution,
                               confirm_spend=first.call.confirm_spend,
                               relocate_batch_root=first.call.relocate_batch_root)
    _print_result(swept)
    return 0 if all(_complete(swept.conditions[item.condition_id].draws) for item in group) else 1


def _submit_evaluate(args: argparse.Namespace, argv: Sequence[str], roots: dict[Path, list[Planned]],
                     swept_conditions: bool) -> int:
    from ruleforge.cli import slurm
    from ruleforge.launch.profile import find_profile

    profile = find_profile(args.slurm, args.cluster_dir)
    base = slurm.passthrough_argv(argv, drop={**slurm.SUBMISSION_OPTIONS, "--sweep": 1})
    specs, analysed = [], {}
    for index, (root, group) in enumerate(roots.items()):
        if not swept_conditions:
            specs.append(slurm.job(f"evaluate-{index}", _member_argv(base, group[0]), profile,
                                   _request(args, group[0])))
            continue
        campaign, conditions = _registered(root, group, argv, dry_run=args.dry_run)
        # A root holding an accepted analysis has nothing left to judge, so it gets no job; any other is refused.
        accepted = campaign.accepted_analysis([condition.condition_id for condition in conditions])
        if accepted is not None:
            analysed[root] = accepted
            continue
        members_ = [slurm.job(f"evaluate-{index}-{n}", _run_argv(root, item, conditions[n]), profile,
                              _request(args, item))
                    for n, item in enumerate(group)]
        specs += members_
        if any(item.call.condition.mode == "both" for item in group):
            specs.append(slurm.job(f"analyze-{index}", ["analyze", str(root)], profile,
                                   slurm.Request(timeout_min=args.timeout_min),
                                   depends_on=[spec.job_id for spec in members_]))
    for root, result in analysed.items():
        if specs:  # stdout holds the launch's result alone (one JSON object under --json)
            print(f"== {root} holds an accepted analysis of its members: nothing of it to run", file=sys.stderr,
                  flush=True)
        else:
            _print_result(result)
    if not specs:
        return 0
    return slurm.submit(specs, profile, verb="evaluate", dry_run=args.dry_run, as_json=args.json)


def _request(args: argparse.Namespace, item: Planned) -> Any:
    from ruleforge.cli import slurm

    gpus = args.gpus if args.gpus is not None else engine_gpus(item.call.judge)
    return slurm.request(args, gpus, resumable=True)


#: Keys a root's job gets from its own flags; an override of them is left out of its argv.
DESTINATION_KEYS = ("campaign_root", "campaign_id")


def _member_argv(base: Sequence[str], item: Planned) -> list[str]:
    cut = base.index("--") if "--" in base else len(base)
    head, tail = _without(base[:cut], ("--out", "--campaign-id")), without_keys(base[cut + 1:], DESTINATION_KEYS)
    head += ["--out", str(item.root), "--campaign-id", item.campaign_id]
    for axis, value in item.member.values:
        if axis.name in AXIS_FLAGS:
            head += [AXIS_FLAGS[axis.name], value]
        else:
            tail.append(f"{axis.key}={value}")
    return head + (["--", *tail] if tail else [])


def _without(argv: Sequence[str], options: Sequence[str]) -> list[str]:
    from ruleforge.cli.slurm import passthrough_argv

    return passthrough_argv(argv, drop=dict.fromkeys(options, 1))


def _registered(root: Path, group: Sequence[Planned], argv: Sequence[str], *,
                dry_run: bool) -> tuple[Any, list[Any]]:
    from ruleforge.evaluate.api import Campaign

    first = group[0]
    if dry_run:
        conditions = [item.drafted for item in group]
        campaign = first.drafted.campaign
    else:
        campaign = Campaign.create(root, campaign_id=first.campaign_id, scenarios=first.call.scenarios,
                                   sampling=first.call.sampling)
        conditions = list(campaign.register([item.entry() for item in group]))
    record = None if dry_run else records.open_record(root, "evaluate", argv, configs_text(group),
                                                       campaign_root=True)
    for item, condition in zip(group, conditions):
        place = (record or records.runs_dir(root) / "dry-run") / "judges" / f"{item.member.index}.yaml"
        if record is not None:
            place.parent.mkdir(parents=True, exist_ok=True)
            place.write_text(OmegaConf.to_yaml(OmegaConf.create(condition.judge.options.written())),
                             encoding="utf-8")
        item.judge_file = place
    return campaign, conditions


def _run_argv(root: Path, item: Planned, condition: Any) -> list[str]:
    call = item.call
    argv = ["campaign", "run", str(root), condition.condition_id, "--judge", str(item.judge_file),
            "--rules", str(call.rules.path), "--scenarios", str(call.scenarios.path)]
    if call.condition.rule_levels is not None:
        argv += ["--rule-levels", str(call.condition.rule_levels)]
    argv += ["--task", condition.task_locator]
    if call.confirm_spend:
        argv.append("--yes-spend")
    if call.relocate_batch_root:
        argv.append("--relocate-batch-root")
    execution = {name: value for name, value in dataclasses.asdict(call.execution).items()
                 if value != getattr(type(call.execution)(), name)}
    overrides = [f"execution.{name}={quoted(value)}" for name, value in execution.items()]
    if call.sampling.max_transport_retries != _default_retries():
        overrides.append(f"max_transport_retries={call.sampling.max_transport_retries}")
    return argv + (["--", *overrides] if overrides else [])


def _default_retries() -> int:
    from ruleforge.evaluate.api import SamplingSettings

    return SamplingSettings().max_transport_retries


def _recorded_claim(root: Path, scenarios: Path, *, max_transport_retries: Optional[int] = None,
                    own: bool = False) -> dict[str, Any]:
    """Return the ``Campaign.create``/``draft`` arguments that bind ``root`` as recorded (``own``: a root this library
    created)."""
    from ruleforge.evaluate.api import Campaign, SamplingSettings
    from ruleforge.scenarios import ScenarioSet

    recorded = Campaign.reopen(root) if own else Campaign.open(root)
    approved = (recorded.scenario_manifest.record_count
                if recorded.config.expected_scenario_count is not None else None)
    retries = _default_retries() if max_transport_retries is None else max_transport_retries
    sampling = dataclasses.replace(SamplingSettings.from_config(recorded.config, max_transport_retries=retries),
                                   approved_scenario_count=approved)
    return dict(root=recorded.root if own else root, campaign_id=recorded.campaign_id,
                scenarios=ScenarioSet.load(scenarios), sampling=sampling)


def _given_settings(args: argparse.Namespace) -> Any:
    from ruleforge.evaluate.api import ConditionSettings

    given = {name: getattr(args, name) for name in ("task", "prompt", "interpretation", "rule_context", "mode")
             if getattr(args, name) is not None}
    return ConditionSettings(**given, rule_levels=args.rule_levels, extra=_extra(args.extra))


def _recorded_settings(spec: Any, task: str, rule_levels: Optional[Path]) -> Any:
    from ruleforge.evaluate.api import ConditionSettings

    return ConditionSettings(task=task, prompt=spec.prompt_style, interpretation=spec.judge_type,
                             rule_context=spec.individual_rule_context, rule_levels=rule_levels,
                             mode=spec.evaluation_mode, extra=dict(spec.extra))


def _extra(items: Sequence[str]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for item in items or ():
        name, separator, value = item.partition("=")
        if not separator or not name:
            raise CliError(f"--extra {item!r}: expected KEY=VALUE")
        if name in found:
            raise CliError(f"--extra {name}={found[name]} and --extra {item} both set {name}: give it once")
        found[name] = value
    return found


def _judge(value: str) -> Any:
    return judge_spec(judge_node(value, cwd=Path.cwd()))


CREATE_FLAGS = {"draws": "draws", "attempts": "max_attempts", "retry_scope": "retry_scope"}


def run_campaign_create(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Create a campaign root over a population and its sampling, or find it there again."""
    from ruleforge.core.paths import one_place
    from ruleforge.evaluate.api import Campaign, SamplingSettings, default_campaign_id
    from ruleforge.evaluate.sharding import campaign_lock_path
    from ruleforge.scenarios import ScenarioSet

    try:
        root, source = one_place(args.root), one_place(args.scenarios, kind="file")
    except ValueError as error:
        raise CliError(f"campaign create: {error}") from error
    scenarios = ScenarioSet.load(source)
    given = {field_: getattr(args, flag) for flag, field_ in CREATE_FLAGS.items() if getattr(args, flag) is not None}
    sampling = SamplingSettings(**given, **({"campaign_mode": "final"} if args.final else {}))
    campaign_id = args.campaign_id or default_campaign_id(scenarios.manifest, sampling.config())
    run, lock = Member(0, ()), campaign_lock_path(root)
    check_places([*records.target_places(run, root, "root", campaign_root=True),
                  Place(run, "campaign lock", lock, own=False,
                        protect=records.protected(lock, "the campaign lock beside the root"))],
                 way_out="give the campaign a root of its own, apart from its run records and the campaign lock "
                         "beside it (through no '..' or link that brings two together)")
    campaign = Campaign.create(root, campaign_id=campaign_id, scenarios=scenarios, sampling=sampling)
    records.open_record(campaign.root, "campaign create", argv, None, campaign_root=True)
    _print({"root": str(campaign.root), "campaign_id": campaign.campaign_id, "scenarios": len(scenarios.ids),
            "draws": sampling.draws, "campaign_mode": sampling.campaign_mode})
    return 0


def run_campaign_add(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Register a condition in a root this library created, its run recorded beside the root."""
    from ruleforge.evaluate.api import Campaign
    from ruleforge.evaluate.campaign import check_new_condition_id
    from ruleforge.rulesets import Ruleset

    campaign = Campaign.create(**_recorded_claim(args.root, args.scenarios))
    if args.condition_id is not None:  # one named by its fingerprint holds no recorded key
        check_new_condition_id(campaign.root, args.condition_id)
    given = dict(rules=Ruleset.load(args.rules), judge=_judge(args.judge), settings=_given_settings(args))
    record = records.open_record(campaign.root, "campaign add", argv, None, campaign_root=True)
    with records.logging_to(record, verbose=args.verbose):
        condition = campaign.condition(args.condition_id, **given)
    _print({"root": str(campaign.root), "condition_id": condition.condition_id,
            "fingerprint": condition.spec.fingerprint})
    return 0


def run_campaign_run(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from ruleforge.evaluate.api import EXECUTION_FNAME, Campaign, refuse_unconfirmed_spend
    from ruleforge.rulesets import Ruleset
    from ruleforge.evaluate.campaign import load_conditions
    from ruleforge.evaluate.packing import ExecutionSettings

    keys = [item.partition("=")[0] for item in args.overrides]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise CliError(f"campaign run: -- {', '.join(repeated)} given more than once: give each once")
    given = OmegaConf.to_container(OmegaConf.from_dotlist(list(args.overrides)), resolve=True)
    try:
        if set(given) - RUN_OVERRIDES:
            raise TypeError(f"{sorted(set(given) - RUN_OVERRIDES)} are no execution settings")
        execution = ExecutionSettings(**given.get("execution", {}))
        retries = given.get("max_transport_retries")
        if retries is not None and (isinstance(retries, bool) or not isinstance(retries, int)):
            raise TypeError(f"max_transport_retries={retries!r} is no whole number")
    except (TypeError, ValueError) as error:
        raise CliError("campaign run takes `-- execution.<setting>=<value>` and `-- max_transport_retries=<n>` "
                       f"overrides alone ({error})") from error
    claim = _recorded_claim(args.root, args.scenarios, max_transport_retries=retries, own=args.preflight_only)
    if not args.preflight_only:
        Campaign.check_create(**claim)  # the root's own refusals first, as creating it makes them
    draft = Campaign.draft(**claim)
    spec = load_conditions(draft.root).get(args.condition_id)
    if spec is None:
        raise CliError(f"{draft.root} registers no condition {args.condition_id!r}; register it with "
                       "`ruleforge campaign add`")
    execution_record = draft.root / args.condition_id / EXECUTION_FNAME
    task = args.task or (json.loads(execution_record.read_text(encoding="utf-8"))["task_locator"]
                         if execution_record.is_file() else spec.task)
    bound = dict(rules=Ruleset.load(args.rules), judge=_judge(args.judge),
                 settings=_recorded_settings(spec, task, args.rule_levels))
    if args.preflight_only:
        _print(draft.preflight(args.condition_id, **bound, draw_ids=args.draw_ids, execution=execution,
                               confirm_spend=args.yes_spend, relocate_batch_root=args.relocate_batch_root))
        return 0
    if not args.yes_spend and bound["judge"].capabilities().spends_money:
        refuse_unconfirmed_spend(draft.prepare(args.condition_id, **bound), draw_ids=args.draw_ids)
    campaign = Campaign.create(**claim)
    record = records.open_record(campaign.root, "campaign run", argv, None, campaign_root=True)
    with records.logging_to(record, verbose=args.verbose):
        campaign.condition(args.condition_id, **bound)
        campaign.run(args.condition_id, draw_ids=args.draw_ids, execution=execution, confirm_spend=args.yes_spend,
                     relocate_batch_root=args.relocate_batch_root)
    draws = campaign.status()["conditions"][args.condition_id]["draws"]
    states = {str(draw): record_["state"] for draw, record_ in draws.items()
              if args.draw_ids is None or int(draw) in args.draw_ids}
    _print({"root": str(campaign.root), "condition_id": args.condition_id, "draws": states})
    return 0 if all(state == "complete" for state in states.values()) else 1


def run_campaign_status(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from ruleforge.evaluate.api import Campaign

    _print(Campaign.open(args.root).status(check_ledgers=args.check_ledgers))
    return 0


def run_analyze(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Analyse a root: this library's through ``Campaign.analyze``, a configured root through its own writers."""
    from ruleforge.evaluate import configured
    from ruleforge.evaluate.aggregate import refuse_stripped
    from ruleforge.evaluate.api import Campaign

    refuse_stripped(args.root)
    settings = dict(allow_incomplete=args.allow_incomplete, allow_stack_mismatch=args.allow_stack_mismatch,
                    rule_set=args.rule_set, exclude_exhausted=args.exclude_exhausted, n_resamples=args.n_resamples,
                    ci_level=args.ci_level, seed=args.seed, n_permutations=args.n_permutations, replace=args.replace)
    if configured.is_configured(args.root):
        root = configured.checked_root(args.root)
        record = records.open_record(root, "analyze", argv, None, campaign_root=True)
        with records.logging_to(record, verbose=args.verbose):
            configured.analyze(root, **settings)
        result = Campaign.open(root).result()
    else:
        campaign = Campaign.reopen(args.root)
        root = campaign.root
        record = records.open_record(root, "analyze", argv, None, campaign_root=True)
        with records.logging_to(record, verbose=args.verbose):
            result = campaign.analyze(**settings)
    _print_result(result)
    return 0


#: The kinds of request a cost reading may keep (``ruleforge.evaluate.identity.COMPONENT_KINDS``).
COMPONENT_KINDS = ("full_set", "rule")


def register_campaign(verbs: argparse._SubParsersAction) -> None:
    """Add ``campaign collect``, ``cost usage`` and ``merge-shards``."""
    collect = subcommand(verbs, "collect", run_collect,
                         "collect a Batch that ended without completing: its outputs, usage.json and partial.json. A "
                         "work directory inside a campaign root is collected only from that root's own batch/ (the "
                         "one batch-root rule); anywhere else in a campaign root it is refused")
    collect.add_argument("directories", nargs="+", type=Path, metavar="batch_dir",
                         help="a Batch work directory ($RULEFORGE_BATCH_ROOT/<key>, or <root>/batch/<key> in a "
                              "campaign root) whose batch ended without completing")
    cost = verbs.add_parser("cost", help="what a Batch API campaign's batches used and cost",
                            allow_abbrev=False)
    readings = cost.add_subparsers(dest="reading", required=True)
    usage = subcommand(readings, "usage", run_cost_usage, "attribute and price the batches of campaign roots")
    usage.add_argument("--root", type=Path, action="append", required=True, help="a campaign root to read (repeatable)")
    usage.add_argument("--batch-root", type=Path, help="the batch state directory (default: $RULEFORGE_BATCH_ROOT, else the one --root's batch/)")
    usage.add_argument("--rates", type=Path, help="the rate file (default: the package's "
                                                  "configs/pricing/batch_rates.yaml)")
    usage.add_argument("--price", action="append", help="a model id of the rate file to price at (repeatable)")
    usage.add_argument("--out", type=Path, help="also write the record here, as JSON")
    usage.add_argument("--scenarios", type=Path,
                       help="a scenario JSONL (a population or a cut of one): count each condition's requests for "
                            "these alone, read per request from its batches' outputs")
    usage.add_argument("--condition", action="append",
                       help="read only this condition of the --root campaigns, by condition id (repeatable)")
    usage.add_argument("--group-by", default="condition",
                       help="condition (default), campaign, or a condition field such as prompt_style or "
                            "evaluation_mode")
    usage.add_argument("--component-kind", choices=COMPONENT_KINDS,
                       help="count each condition's requests of this kind alone (the full-set judgment or the "
                            "individual rules'), read per request from its batches' outputs")
    merge = subcommand(verbs, "merge-shards", run_merge_shards,
                       "merge a sharded condition's shard roots into its campaign root (one this library created, or "
                       "a configured root this version's jobs made: `ruleforge jobs launch campaign --shards`)")
    merge.add_argument("--campaign-root", required=True, type=Path)
    merge.add_argument("--condition-id", required=True)
    merge.add_argument("--shards-dir", type=Path,
                       help="where the shard roots are, if not <campaign root>.shards/<condition id>")
    merge.add_argument("--num-shards", type=int, help="the layout to merge, when several were launched")


def run_collect(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Run :func:`ruleforge.judges.batch_collect.collect` for each directory; exit 2 when one was refused, the others
    still collected."""
    from ruleforge.judges.batch_api import BatchDestinationRefused, BatchExecutionError, guard_work_directories
    from ruleforge.judges.batch_collect import collect, summary

    from ruleforge.evaluate.api import CampaignError, refuse_work_directory

    def guard(batch_root: Path, work: Path) -> None:
        try:
            refuse_work_directory(work, "the batch work directory")
        except CampaignError as error:
            raise BatchDestinationRefused(str(error)) from error

    refused = 0
    for directory in [records.absolute_path(given) for given in args.directories]:
        try:
            refuse_work_directory(directory, "the batch work directory")
            with guard_work_directories(guard):
                marker = collect(directory)
        except (BatchExecutionError, CampaignError, ValueError) as error:
            print(f"ruleforge campaign collect: {error}", file=sys.stderr)
            refused += 1
            continue
        print(summary(directory, marker))
    return 2 if refused else 0


def _batch_root(given: Optional[Path], roots: Sequence[Path]) -> Path:
    from ruleforge import env
    from ruleforge.evaluate.api import DEFAULT_BATCH_DIRNAME

    found = given or env.batch_root()
    if found is not None:
        return Path(found)
    if len(roots) == 1:
        return roots[0] / DEFAULT_BATCH_DIRNAME
    raise CliError("--batch-root is required with several --root when RULEFORGE_BATCH_ROOT is not set")


RATES = "configs/pricing/batch_rates.yaml"


def _rates(given: Optional[Path]) -> Path:
    from ruleforge.env import package_path

    return package_path(RATES) if given is None else given


def _refuse_out_in(out: Optional[Path], batch_roots: Sequence[Path]) -> None:
    if out is None:
        return
    place = Path(os.path.realpath(out))
    for batch_root in batch_roots:
        if place.is_relative_to(os.path.realpath(batch_root)):
            raise CliError(f"--out {out} is inside the batch root {batch_root}, whose directories a cost reading reads "
                           "as batches: write the record outside it")


READING_WAY_OUT = "give --out a place of its own, apart from its run records"


def _reading_out(out: Optional[Path], batch_roots: Sequence[Path]) -> Optional[Path]:
    out = records.absolute_path(out)
    _refuse_out_in(out, batch_roots)
    if out is not None:
        records.check_outputs([out], way_out=READING_WAY_OUT)
    return out


def _write_record(out: Optional[Path], record: Any, argv: Sequence[str], verb: str) -> None:
    from ruleforge.core.artifacts import write_text_atomically

    if out is not None:
        records.open_record(out, verb, argv, None)
        write_text_atomically(out, json.dumps(record, indent=1, sort_keys=True) + "\n")


def run_cost_usage(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Print :func:`ruleforge.evaluate.cost.campaign_usage`'s table; exit 1 when a batch the ledgers name has no
    usage.json."""
    from ruleforge.evaluate.cost import campaign_usage

    if args.scenarios is not None and not args.scenarios.is_file():
        raise CliError(f"--scenarios {args.scenarios} is not a file")
    batch_root = _batch_root(args.batch_root, args.root)
    out = _reading_out(args.out, [batch_root])
    record, lines, attribution = campaign_usage(
        args.root, batch_root, rates=_rates(args.rates), price=args.price or (),
        scenarios=args.scenarios, component_kind=args.component_kind, conditions=args.condition or (),
        group_by=args.group_by)
    print("\n".join(lines))
    if attribution.missing:
        print(f"MISSING: {len(attribution.missing)} batch(es) the ledgers name have no usage.json under "
              f"{attribution.batch_root}; their cost is not counted: {dict(sorted(attribution.missing.items()))}")
    _write_record(out, record, argv, "campaign cost usage")
    if out is not None:
        print(f"record: {out}")
    return 1 if attribution.missing else 0


def run_merge_shards(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Merge a campaign's condition shards into a root this library created, every place checked first."""
    from ruleforge.evaluate import configured
    from ruleforge.evaluate.api import Campaign

    if configured.merges_into(args.campaign_root):
        root = Path(os.path.realpath(records.absolute_path(args.campaign_root)))
        configured.check_merge(root, args.condition_id, shards_dir=args.shards_dir, num_shards=args.num_shards)
        record = records.open_record(root, "campaign merge-shards", argv, None, campaign_root=True)
        with records.logging_to(record, verbose=args.verbose):
            report = configured.merge_shards(root, args.condition_id, shards_dir=args.shards_dir,
                                             num_shards=args.num_shards)
        print(json.dumps(report, indent=1))
        return 0
    campaign = Campaign.reopen(args.campaign_root)
    campaign.check_shard_merge(args.condition_id)
    record = records.open_record(campaign.root, "campaign merge-shards", argv, None, campaign_root=True)
    with records.logging_to(record, verbose=args.verbose):
        report = campaign.merge_shards(args.condition_id, shards_dir=args.shards_dir, num_shards=args.num_shards)
    print(json.dumps(report, indent=1))
    return 0


BANK_PATHS = {"scenario_set_path": PATH, "rule_set_path": PATH, "campaign_root": PATH, "job_root": PATH,
              "rule_levels_path": PATH}
BANK_FLAGS = {"out": ("campaign_root", True), "campaign_id": ("campaign_id", False)}


def bank_operation() -> Operation:
    """Return ``bank run``'s schema."""
    from ruleforge.evaluate.bank import DESIGN_CONFIG_KEY as block, verify_subsets

    return Operation("bank run", parameters=bind(
        verify_subsets, only=("subset_root", "manifest_dir"),
        keys={"subset_root": f"{block}.subset_root", "manifest_dir": f"{block}.subset_manifest_dir"}),
        judges=(JUDGE_NODE, "agent_model"), declared=BANK_PATHS)


def register_bank(sub: argparse._SubParsersAction) -> None:
    """Add ``bank run`` and ``bank analyze``."""
    parser = sub.add_parser("bank", help="a shared-bank campaign: run a job of it, analyse it",
                            allow_abbrev=False)
    verbs = parser.add_subparsers(dest="verb", required=True)
    text = ("run one job of a shared-bank campaign -- the bank, then one full-set condition per (subset seed, "
            "size) -- with its judge loaded once, into a root this library creates or created")
    run = subcommand(verbs, "run", run_bank_run, text, overrides=True)
    run.add_argument("--config", type=Path, required=True,
                     help="the campaign's config (a configured campaign's, with its ruleset-size design under the key "
                          "`ruleset_size`: configs/sampling_ruleset_size.yaml's shape); its relative paths are its directory's, but for a "
                          "file of the checkout's configs/, written for its jobs' working directory: read from the "
                          "checkout's top")
    run.add_argument("--judge", help=JUDGE_HELP)
    run.add_argument("--out", help="the campaign root (default: the config's campaign_root)")
    run.add_argument("--campaign-id", help="the campaign's id (default: the config's)")
    run.add_argument("--preflight-only", action="store_true",
                     help="check the job and the root it claims, and print the job's plan; write nothing (so does "
                          "that design block's preflight_only)")
    run.add_argument("--yes-spend", action="store_true", help="let a judge that bills its calls run")
    analyze = subcommand(verbs, "analyze", run_bank_analyze,
                         "write a shared-bank root's analysis: the inconsistency curve over the subset sizes")
    analyze.add_argument("root", type=Path, help="a shared-bank campaign root this library created (any shared-bank "
                                                 "root with --out)")
    analyze.add_argument("--out", type=Path,
                         help="write the analysis to this file, outside the root, instead: any shared-bank root, "
                              "one this library did not create among them, is read and never written")
    analyze.add_argument("--n-resamples", type=int, help="override the manifest's bootstrap count")
    analyze.add_argument("--ci-level", type=float, help="override the manifest's interval level")
    analyze.add_argument("--seed", type=int, help="override the manifest's bootstrap seed")
    analyze.add_argument("--allow-stack-mismatch", action="store_true",
                         help="report a model whose conditions were judged on different inference stacks, marked as "
                              "such")
    analyze.add_argument("--replace", action="store_true", help="replace the root's analysis")
    analyze.add_argument("--bank-root", type=Path,
                         help="a shared-bank root that supplies each model's bank and every size the root's own "
                              "design lacks; it is only read. The two campaigns must agree on every sampling setting "
                              "in force, and the retry cap is the one setting a recorded amendment may have raised")


def checkout_configs() -> dict[Path, Path]:
    """Return the checkout's Hydra configs mapped to the working directory their jobs run in; none outside a checkout."""
    from ruleforge.core.errors import ArtifactRootError
    from ruleforge.core.paths import repository_root

    try:
        checkout = repository_root()
    except ArtifactRootError:
        return {}
    return {checkout / "configs": checkout}


def bank_config(args: argparse.Namespace, cwd: Path) -> DictConfig:
    """Return ``bank run``'s config: ``--config`` composed with the flags and the overrides."""
    operation = bank_operation()
    judge = judge_flag(args, "judge", JUDGE_NODE, cwd)
    flags = ([] if judge is None else [judge]) + flag_settings(args, BANK_FLAGS, cwd)
    overrides = parse_overrides(args.overrides, operation, cwd)
    refuse_conflicts(flags, overrides, ())
    return compose(operation, OmegaConf.create({}), config_file=args.config, layers=(flags + overrides,), cwd=cwd,
                   config_bases=checkout_configs())


def bank_places(root: Path, batch_root: Optional[str]) -> list[Place]:
    """Return every place ``bank run`` writes, as one inventory."""
    from ruleforge.evaluate.api import bound_batch_root, refuse_batch_root
    from ruleforge.evaluate.sharding import campaign_lock_path

    found = records.target_places(None, root, "root", campaign_root=True)
    lock = campaign_lock_path(root)
    found.append(Place(None, "campaign lock", lock, own=False,
                       protect=records.protected(lock, "the campaign lock beside the root")))
    if batch_root:
        place = Path(bound_batch_root(Path(os.path.realpath(root)), str(batch_root)))
        found.append(Place(None, "the batch judge's batch_root", place, own=False, home=root, holds=True,
                           kind=records.BATCH_ROOT_KIND, protect=lambda: refuse_batch_root(root, place)))
    return found


def run_bank_run(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Run one job of a shared-bank campaign from its composed config."""
    from ruleforge.evaluate.api import Campaign
    from ruleforge.evaluate.bank import design_block, incomplete_error, plan_job, run_job
    from ruleforge.judges.batch_api import config_confirms_spend, spend_confirmed
    from ruleforge.evaluate.identity import SamplingConfig
    from ruleforge.scenarios import ScenarioSet

    cfg = bank_config(args, Path.cwd())
    design, planned, report = plan_job(cfg)
    root = Path(str(cfg.campaign_root))
    claim = dict(campaign_id=str(cfg.campaign_id),
                 scenarios=ScenarioSet.load(str(cfg.scenario_set_path), expect_sha256=cfg.get("approved_scenario_sha256")),
                 sampling=SamplingConfig.from_mapping(OmegaConf.to_container(cfg.sampling, resolve=True)))
    check_places(bank_places(root, cfg.judge_model.get("batch_root")),
                 way_out="give the root a place of its own, apart from its run records and the campaign lock, and "
                         "the judge's batch_root a place of its own in it or above it")
    Campaign.check_create(root, **claim)
    print(json.dumps(report, indent=2), flush=True)
    confirmed = spend_confirmed() if config_confirms_spend(cfg) or args.yes_spend else nullcontext()
    if args.preflight_only or bool(design_block(cfg).get("preflight_only", False)):
        return 0
    campaign = Campaign.create(root, **claim)
    record = records.open_record(campaign.root, "bank run", argv, cfg, campaign_root=True)
    with records.logging_to(record, verbose=args.verbose), confirmed:
        incomplete = run_job(campaign, cfg, design, planned, report)
    if incomplete:
        print(incomplete_error(incomplete), file=sys.stderr)
        return 1
    return 0


def run_bank_analyze(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Analyse a shared-bank root; with ``--out``, any such root into that file."""
    from ruleforge.evaluate import bank
    from ruleforge.evaluate.api import Campaign, CampaignError
    from ruleforge.evaluate.campaign import load_conditions
    from ruleforge.evaluate.identity import refuse_published

    settings = dict(n_resamples=args.n_resamples, ci_level=args.ci_level, seed=args.seed,
                    allow_stack_mismatch=args.allow_stack_mismatch, replace=args.replace, bank_root=args.bank_root)
    for place in (args.root, *(() if args.bank_root is None else (args.bank_root,))):
        refuse_published(place, "bank analyze", "its manifests and each draw's completion record in full")
    out = records.absolute_path(args.out)
    if out is not None:
        records.check_outputs([out], way_out=READING_WAY_OUT)
        Campaign.open(args.root)
        record = records.open_record(out, "bank analyze", argv, None)
        with records.logging_to(record, verbose=args.verbose):
            payload = bank.analyze_into(args.root, out, **settings)
    else:
        try:
            campaign = Campaign.reopen(args.root)
        except CampaignError as error:
            raise CliError(f"{error}; `--out FILE` analyses it into a file outside it") from error
        record = records.open_record(campaign.root, "bank analyze", argv, None, campaign_root=True)
        with records.logging_to(record, verbose=args.verbose):
            payload = bank.analyze(args.root, **settings)
    for model, entry in payload["models"].items():
        first, last = entry["points"][0], entry["points"][-1]
        head = f"{model}: K={first['size']} {first['estimate']:.3f} [{first['ci_low']:.3f}, {first['ci_high']:.3f}]"
        if len(entry["points"]) == 1:
            print(f"{head}; one size, so no contrast")
            continue
        rise = entry["contrasts"]["net_rise"]
        print(f"{head} -> K={last['size']} {last['estimate']:.3f} [{last['ci_low']:.3f}, {last['ci_high']:.3f}];"
              f" net rise {rise['estimate']:+.3f} [{rise['ci_low']:+.3f}, {rise['ci_high']:+.3f}];"
              f" {entry['adjacent_rises']} of {len(entry['points']) - 1} adjacent steps rise")
    print(json.dumps({key: payload[key] for key in ("campaign_id", "state", "n_scenarios", "n_decoder_draws",
                                                    "n_subset_seeds")}))
    note_fake(spec.model_type for spec in load_conditions(args.root).values())
    return 0
