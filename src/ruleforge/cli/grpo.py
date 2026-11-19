from __future__ import annotations

import argparse
import dataclasses
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from ruleforge.cli import records
from ruleforge.cli.common import Assignment, CliError, flag_settings, rebased, setting, subcommand
from ruleforge.cli.compose import compose, parse_overrides, refuse_conflicts
from ruleforge.cli.mounted import Step, Tool, lazy_group, mount, mount_step
from ruleforge.cli.schema import PATH, REQUIRED, SEQUENCE, Operation, annotations_of, bind, defaults_of, path_kind
from ruleforge.cli.sweep import Member, Place, check_places
from ruleforge.core.optional import require


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    return value


@dataclass(frozen=True)
class FunctionCommand:
    """A command whose config is an API function's parameters; ``defaults`` override theirs."""

    name: str
    load: Callable[[], Callable[..., Any]] = field(repr=False)
    target: str
    summary: str
    nested: Mapping[str, Callable[[], type]] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)
    flags: Mapping[str, str] = field(default_factory=dict)
    beside: tuple[str, ...] = ()

    def function(self) -> Callable[..., Any]:
        return self.load()

    def keys(self) -> dict[str, tuple[Any, str]]:
        """Each config key, with the callable and parameter it is bound to."""
        function = self.function()
        found = {name: (function, name) for name in defaults_of(function) if name not in self.nested}
        for name, load in self.nested.items():
            found.update({f"{name}.{field_}": (load(), field_) for field_ in defaults_of(load())})
        return found

    def operation(self) -> Operation:
        return Operation(self.name, parameters=tuple(
            parameter for key, (target, name) in self.keys().items()
            for parameter in bind(target, only=(name,), keys={name: key})))

    def config(self) -> DictConfig:
        """The packaged config: every key at its parameter's default (``???`` when it has none)."""
        node: dict[str, Any] = {}
        for key, (target, name) in self.keys().items():
            value = self.defaults.get(key, defaults_of(target)[name])
            head, _, rest = key.partition(".")
            if rest:
                node.setdefault(head, {})[rest] = _plain(value)
            else:
                node[key] = _plain(value)
        return OmegaConf.create(node)

    def flag(self, key: str) -> str:
        return self.flags.get(key, "--" + key.replace(".", "-").replace("_", "-"))


def _anchors() -> Any:
    from ruleforge.grpo import anchor_set

    return anchor_set


def _merge_expectation() -> type:
    from ruleforge.grpo.merging import MergeExpectation

    return MergeExpectation


def _merge_adapter() -> Callable[..., Any]:
    from ruleforge.grpo.merging import merge_adapter

    return merge_adapter


def _grpo(name: str) -> Callable[[], Callable[..., Any]]:
    def load() -> Callable[..., Any]:
        import ruleforge.grpo

        return getattr(ruleforge.grpo, name)

    return load


#: What a build writes beside its target before renaming it there.
PARTIAL_SUFFIX = ".partial"

COMMANDS = {
    "rows": FunctionCommand("grpo rows", _grpo("rows"), target="out", defaults={"pins": {}},
                            summary="build the GRPO rows and the KL-anchor replay rows", beside=(PARTIAL_SUFFIX,)),
    "anchors build": FunctionCommand("grpo anchors build", lambda: _anchors().build_anchor_set, target="out",
                                     summary="build an anchor set from single-rule banks", beside=(PARTIAL_SUFFIX,)),
    "anchors route": FunctionCommand("grpo anchors route", lambda: _anchors().route_top_up, target="out",
                                     summary="route the bank rows that need more draws to a top-up",
                                     beside=(PARTIAL_SUFFIX,)),
    "merge": FunctionCommand("grpo merge", _merge_adapter, target="out_dir", nested={"expect": _merge_expectation},
                             summary="merge an adapter into its base model's weights in fp32",
                             defaults={"threads": 8},
                             flags={"base_dir": "--base", "adapter_dir": "--adapter", "out_dir": "--out",
                                    "training_step_override": "--training-step"}, beside=(PARTIAL_SUFFIX,)),
}


def places(target: Path, what: str, *, beside: Sequence[str] = (), launching: bool) -> list[Place]:
    """Return every place a GRPO command writes, or its job does, each with its check."""
    run = Member(0, ())
    found = records.target_places(run, target, what)
    for suffix in beside:
        partial = target.with_name(target.name + suffix)
        found.append(Place(run, f"{suffix} directory", partial,
                           protect=records.protected(partial, f"its {suffix} directory")))
    if launching:
        from ruleforge.cli.slurm import launch_places

        found += launch_places()
    return found


WAY_OUT = ("give its target a place of its own, apart from its run records, its partial build and the launch records "
           "(through no '..' or link that brings two together)")


def _number(annotation: Any, default: Any, kind: type) -> bool:
    return (isinstance(default, kind) and not isinstance(default, bool)) or annotation is kind or (
        kind in typing.get_args(annotation))


def add_function_arguments(parser: argparse.ArgumentParser, command: FunctionCommand) -> None:
    """Add one flag per config key of ``command``."""
    for key, (target, name) in command.keys().items():
        default = command.defaults.get(key, defaults_of(target)[name])
        if isinstance(default, Mapping):
            continue
        annotation = annotations_of(target).get(name, Any)
        kind = path_kind(annotation)
        options: dict[str, Any] = {"dest": key.replace(".", "__"), "default": None}
        if isinstance(default, bool):
            options["action"] = argparse.BooleanOptionalAction
        elif kind == SEQUENCE or isinstance(default, (tuple, list)):
            options["nargs"] = "+"
        elif _number(annotation, default, int):
            options["type"] = int
        elif _number(annotation, default, float):
            options["type"] = float
        shown = "required" if default == REQUIRED else f"default {default!r}"
        named = {PATH: "a path, ", SEQUENCE: "paths, "}.get(kind, "")
        parser.add_argument(command.flag(key), help=f"{named}{shown}", **options)


def _function_flags(args: argparse.Namespace, command: FunctionCommand, cwd: Path) -> list[Assignment]:
    found = []
    kinds = command.operation().path_keys()
    for key in command.keys():
        value = getattr(args, key.replace(".", "__"), None)
        if value is not None:
            found.append(setting(key, rebased(value, cwd) if key in kinds else value, f"{command.flag(key)} {value}"))
    return found


def _arguments(command: FunctionCommand, cfg: DictConfig) -> dict[str, Any]:
    arguments = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    for name, load in command.nested.items():
        arguments[name] = load()(**arguments[name])
    return arguments


def run_function(args: argparse.Namespace, argv: Sequence[str], command: FunctionCommand) -> int:
    cwd = Path.cwd()
    operation = command.operation()
    flags = _function_flags(args, command, cwd)
    overrides = parse_overrides(args.overrides, operation, cwd)
    refuse_conflicts(flags, overrides, ())
    cfg = compose(operation, command.config(), config_file=args.config, layers=(flags + overrides,), cwd=cwd)
    target = Path(cfg[command.target])
    check_places(places(target, command.flag(command.target), beside=command.beside,
                        launching=args.slurm is not None and not args.dry_run), way_out=WAY_OUT)
    if args.slurm is not None:
        return _submit(args, argv, command.name, gpus=args.gpus or 0)
    if args.dry_run:
        records.print_dry_run(target, cfg)
        return 0
    arguments = _arguments(command, cfg)
    record = records.open_record(target, command.name, argv, cfg)
    with records.logging_to(record, verbose=args.verbose):
        result = command.function()(**arguments)
    print(records.describe(_result(result)), flush=True)
    return 0


def _result(result: Any) -> Any:
    if dataclasses.is_dataclass(result):
        return {item.name: _plain(getattr(result, item.name)) for item in dataclasses.fields(result) if item.repr}
    return result


TRAIN_PATHS = {"output_root": PATH, "output_dir": PATH, "model.path": PATH,
               "model.init_adapter": PATH, "resume_from_checkpoint": PATH, "data.train": PATH, "data.replay": PATH,
               "data.rules": PATH, "data.anchor_condition": PATH}
TRAIN_FLAGS = {"run_name": ("run_name", False), "model": ("model.path", True), "output_root": ("output_root", True),
               "resume": ("resume_from_checkpoint", True)}


def train_operation() -> Operation:
    return Operation("grpo train", parameters=(), declared=TRAIN_PATHS, special={"resume_from_checkpoint": ("latest",)})


def run_train(args: argparse.Namespace, argv: Sequence[str]) -> int:
    cwd = Path.cwd()
    operation = train_operation()
    flags = flag_settings(args, TRAIN_FLAGS, cwd, special=("latest",))
    if args.preflight_only:
        flags.append(setting("preflight_only", True, "--preflight-only"))
    if args.allow_unscreened_rows:
        flags.append(setting("data.allow_unscreened_rows", True, "--allow-unscreened-rows"))
    overrides = parse_overrides(args.overrides, operation, cwd)
    refuse_conflicts(flags, overrides, ())
    cfg = compose(operation, "grpo/base.yaml", config_file=args.config, layers=(flags + overrides,), cwd=cwd)
    target = Path(cfg.output_dir)
    check_places(places(target, "output_dir", launching=args.slurm is not None and not args.dry_run), way_out=WAY_OUT)
    if args.slurm is not None:
        return _submit(args, argv, "grpo train", gpus=args.gpus or 1, ranks=True)
    if args.dry_run:
        records.print_dry_run(target, cfg)
        return 0
    # The trainer's module needs torch; TRL and PEFT build the trainer, which a preflight does not.
    require("training", "torch", *(() if cfg.preflight_only else ("trl", "peft")))
    from ruleforge.grpo.trainer import export_submitit_distributed_env, run_grpo

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    record = records.open_record(target, "grpo train", argv, cfg)
    with records.logging_to(record, verbose=args.verbose):
        export_submitit_distributed_env(os.environ)
        run_grpo(config)
    print(records.describe({"output_dir": str(cfg.output_dir), "preflight_only": bool(cfg.preflight_only)}),
          flush=True)
    return 0


def _submit(args: argparse.Namespace, argv: Sequence[str], verb: str, *, gpus: int, ranks: bool = False) -> int:
    from ruleforge.cli import slurm
    from ruleforge.launch.profile import find_profile

    profile = find_profile(args.slurm, args.cluster_dir)
    request = slurm.request(args, gpus)
    if ranks:
        from ruleforge.launch.train import training_resources

        request = dataclasses.replace(request, **training_resources({"cpus": args.cpus, "mem_gb": args.mem_gb}, gpus))
    spec = slurm.job(verb.replace(" ", "-"), slurm.passthrough_argv(argv, drop=slurm.SUBMISSION_OPTIONS), profile,
                     request)
    options = {"options": {"rank_per_gpu": True}} if ranks else {}
    return slurm.submit([spec], profile, verb=verb, dry_run=args.dry_run, as_json=args.json, **options)


def _identity_sidecar(output: Path) -> list[Path]:
    from ruleforge.grpo.report import identity_path

    return [identity_path(output)]


PARITY = Tool("ruleforge.grpo.lora_parity", steps={
    "make-adapter": Step("write a seeded random adapter (CPU, no weights loaded)", ("out",)),
    "inspect": Step("static adapter checks (CPU)"),
    "vllm": Step("freeze sequences (unless given) and score one engine's vLLM arms (GPU)",
                 ("out", "decode_out", "decode_reference_out", "decode_reference", "greedy_out"),
                 creates=("sequences", "decode_sequences")),
    "hf": Step("score the two HF arms, or a plain model alone without --adapter (GPU)",
               ("out", "decode_out", "greedy_out")),
    "compare": Step("gate HF against vLLM (CPU); exit 1 on failure", ("report",)),
    "greedy-agreement": Step("the greedy full-set judgment agreement of two engines (CPU); exit 1 under the minimum",
                             ("report",)),
    "plumbing": Step("M0's plumbing checks on a merged directory (CPU); exit 1 on failure", ("report",)),
})
REPORT = Tool("ruleforge.grpo.report", steps={
    "saves": Step("the tolerances, every save's report and the reported save with the lowest m1: JSON and a table "
                  "(CPU)", ("out",)),
})
TABLES = Tool("ruleforge.grpo.report_tables", steps={
    "populations": Step("the scenario table and each dev campaign's scenario file", ("out_dir",)),
    "read": Step("a response table (and its identity) from one condition of a dev campaign", ("out",),
                 beside=_identity_sidecar),
})


def register(verbs: argparse._SubParsersAction) -> None:
    """Add ``grpo parity``, ``exclusion attest`` and ``report``'s commands."""
    mount(verbs, "parity", PARITY, "HF-PEFT against vLLM logprob parity for one adapter, served as LoRA or merged",
          path=("grpo",))
    exclusion = verbs.add_parser("exclusion", help="the training scenarios' prompt-disjointness marker",
                                 allow_abbrev=False)
    stages = exclusion.add_subparsers(dest="stage", required=True)
    text = "the second pass on a dataset against the protected files, writing its marker beside it"
    attest = subcommand(stages, "attest", run_attest, text)
    attest.add_argument("--dataset", required=True, type=Path)
    attest.add_argument("--protected", required=True, action="append")
    attest.add_argument("--carved", action="append", default=[],
                        help="a held-out file carved from the pool: read by the second pass, attested by no "
                             "screen (repeatable)")
    report = verbs.add_parser("report", help="a training run's saves against the base: tolerances, m1, gate statistics",
                              allow_abbrev=False)
    steps = lazy_group(report, "step")
    for tool in (TABLES, REPORT):
        for step in tool.steps:
            mount_step(steps, step, tool, step, path=("grpo", "report"))


def run_attest(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Run the prompt-exclusion second pass and write its unscreened marker beside the dataset; a leak is refused."""
    from ruleforge.grpo.exclusion import ExclusionError, attest
    from ruleforge.grpo.leakage import LeakageError, disjoint_marker_path

    dataset = records.absolute_path(args.dataset)
    records.check_outputs([disjoint_marker_path(dataset)],
                          way_out="give the dataset a place where its marker and run records are its own")
    record = records.open_record(disjoint_marker_path(dataset), "grpo exclusion attest", argv, None)
    with records.logging_to(record, verbose=args.verbose):
        try:
            marker = attest(dataset, protected=args.protected, carved=args.carved)
        except (LeakageError, ExclusionError) as error:
            raise CliError(str(error)) from error
        carved = (f"; carved (second pass only): {', '.join(entry['path'] for entry in marker['carved'])}"
                  if args.carved else "")
        print(f"disjoint: {marker['dataset']['rows']} rows ({marker['dataset']['prompt_keys']} prompts) against "
              f"{marker['protected_rows']} protected rows{carved}", flush=True)
        print(f"screen: {marker['screen']} (the second pass alone; the trainer takes these rows only with "
              "--allow-unscreened-rows)")
        print(f"marker: {disjoint_marker_path(dataset)}")
    return 0
