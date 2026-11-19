from __future__ import annotations

import argparse
from typing import Any, Optional

from ruleforge.launch.guards import switch
from ruleforge.launch.environment import LaunchError

SPECS_DIR = "configs/launch"
#: A job's CPUs and memory (GB) per GPU when not given, and its time limit in minutes.
DEFAULT_CPUS_PER_GPU, DEFAULT_MEM_GB_PER_GPU, DEFAULT_TIMEOUT_MIN = 4, 24, 60
LAUNCH_BACKENDS, TRAIN_BACKENDS = ("local", "slurm"), ("slurm",)
PREFLIGHT_WHERE = ("local", "none")
#: A campaign's `--root-template` placeholder: a block file's stem.
BLOCK = "block"
LAUNCH_VERBS = {
    "job": ("launch one command", "the command goes after --, after every option: {program} job [options] -- python "
                                  "x.py ..."),
    "campaign": ("launch a stochastic-decoding campaign: its conditions, shards and blocks",
                 "the campaign job's Hydra arguments (python -m ruleforge.launch.programs.run_campaign) go after --: "
                 "{program} campaign [options] -- --config-name ..."),
    "train": ("launch a GRPO training run: preflighted and checked first",
              "the training program's Hydra arguments (python -m ruleforge.launch.programs.train_grpo) go after --: "
              "{program} train --run-name NAME [options] -- KEY=VALUE"),
}
VERBS = {
    "status": "every run's state",
    "wait": "wait for every run to end: exit 0 all succeeded, 1 one did not, 3 timed out",
    "pull": "take the outputs of the runs that succeeded and merge each sharded condition whose shard roots are then "
            "all here",
    "stop": "stop every run that has not ended (--after-save N: a training run, right after its checkpoint-N lands)",
}


class SwitchAction(argparse.BooleanOptionalAction):
    """`--NAME` (on), `--NAME=SWITCH` or `--no-NAME` (off); a spec gives true or false."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.nargs = "?"
        self.metavar = "SWITCH"

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any,
                 option_string: Optional[str] = None) -> None:
        negative = str(option_string).startswith("--no-")
        try:
            if negative and values is not None:
                raise LaunchError(f"{option_string} takes no value")
            value = not negative if values is None else switch(values, what=str(option_string))
        except LaunchError as error:
            raise argparse.ArgumentError(self, str(error)) from None
        setattr(namespace, self.dest, value)


def job_options(parser: argparse.ArgumentParser, *, verb: str = "job") -> None:
    """Add the options of a launch verb to `parser`; `campaign` and `train` leave out what their plans decide."""
    parser.add_argument("--spec", help=f"a launch spec: a path, or a name under {SPECS_DIR}/")
    parser.add_argument("--var", action="append", metavar="NAME=VALUE", help="fill a spec's {NAME} (repeatable)")
    parser.add_argument("--backend", choices=TRAIN_BACKENDS if verb == "train" else LAUNCH_BACKENDS)
    if verb != "train":
        parser.add_argument("--name", help="the job's name (default: the spec's file name, else job)")
    parser.add_argument("--gpu-count", type=int, help="GPUs for the job (default 0)")
    parser.add_argument("--cpus", type=int, help=f"CPUs (default {DEFAULT_CPUS_PER_GPU} per GPU; a CPU job's too)")
    parser.add_argument("--mem-gb", type=int,
                        help=f"GiB of memory (default {DEFAULT_MEM_GB_PER_GPU} per GPU; a CPU job's too)")
    parser.add_argument("--timeout-min", type=int, help=f"the job's limit in minutes (default {DEFAULT_TIMEOUT_MIN})")
    parser.add_argument("--qos", help="the SLURM QOS (default: the spec's, else the --profile's, else a CPU job's "
                                      "normal)")
    if verb == "job":
        parser.add_argument("--workdir", help="the directory under the checkout the command runs in (default .)")
    parser.add_argument("--env", action="append", metavar="NAME=VALUE",
                        help="export after the launcher's setup lines (repeatable; taken literally)")
    if verb == "job":
        parser.add_argument("--output", action="append", metavar="PATH",
                            help="an artifact path the job writes: another launch writing it while this one runs is "
                                 "refused (repeatable)")
    if verb != "train":
        parser.add_argument("--axis", action="append", metavar="KEY=VALUES",
                            help="one job per value, KEY=VALUE appended to its command: a,b,c or A..B, KEY+KEY2 for "
                                 "keys taking each value together (repeatable, at most twice: their product); "
                                 "{member} is each job's index from 0")
        parser.add_argument("--resumable", action=argparse.BooleanOptionalAction, default=None,
                            help="the command resumes when run again: SLURM may requeue it (after a preemption, or "
                                 "its wall)")
    parser.add_argument("--expect-commit", metavar="SHA", help="refuse unless this checkout is at that commit")
    parser.add_argument("--allow-dirty", action="store_true", default=None,
                        help="launch a checkout with uncommitted changes (recorded as dirty)")
    parser.add_argument("--profile", metavar="PROFILE",
                        help="the SLURM cluster of this cluster profile (a YAML file, or PROFILE.yaml in "
                             "$RULEFORGE_CLUSTER_DIR, configs/cluster/ or the package's) in place of the launcher "
                             "config's RULEFORGE_SLURM_* cluster: its partition, account, setup lines (a preflight's too), "
                             "excluded nodes and default QOS")
    parser.add_argument("--dry-run", action="store_true", default=None,
                        help="run the free checks and print what would be submitted; submit nothing")
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=None,
                        help="wait for the runs to end")
    parser.add_argument("--pull", action=argparse.BooleanOptionalAction, default=None,
                        help="after --wait, pull (run the shard merges)")
    parser.add_argument("--ignore-launch", action="append", metavar="LAUNCH_ID",
                        help="launch although this other launch, which writes an output of this one (a campaign: "
                             "a job of one of its conditions), has a run its backend cannot say has ended (look by "
                             "hand first; repeatable, recorded)")
    parser.add_argument("--json", action="store_true", default=None, help="print one JSON object on stdout")


def campaign_options(parser: argparse.ArgumentParser) -> None:
    """Add the campaign verb's own options."""
    parser.add_argument("--campaign-id", metavar="ID",
                        help="campaign_id; its root is sampling/<ID> under the artifact root, whatever the config's")
    parser.add_argument("--campaign-root", metavar="PATH",
                        help="another root (an artifact path), which must hold a campaign unless --new-root")
    parser.add_argument("--new-root", action="store_true", default=None,
                        help="--campaign-root names a root to start, not one to resume")
    parser.add_argument("--population", metavar="PATH",
                        help="the scenario file to judge (an artifact path), with --expect-population-sha256")
    parser.add_argument("--expect-population-sha256", metavar="SHA256", help="the population's approved digest")
    parser.add_argument("--pack-draws", action=SwitchAction, default=None,
                        help="run each condition's draws through shared judge calls (execution.pack_draws)")
    parser.add_argument("--pack-max-prompts", type=int, help="execution.pack_max_prompts, with --pack-draws")
    parser.add_argument("--pack-max-draws", type=int, help="execution.pack_max_draws, with --pack-draws")
    parser.add_argument("--shards", type=int, help="cut each condition into this many jobs, merged back")
    parser.add_argument("--no-shards", action="store_true", default=None,
                        help="run each condition as one job writing its root, whatever the spec's shards (the resume "
                             "of a rebased root)")
    parser.add_argument("--only", metavar="I,J",
                        help="launch only these shards of the campaign's one condition (a relaunch: each resumes)")
    parser.add_argument("--blocks", action="append", metavar="PATH",
                        help="one condition per block file, or per *.jsonl in a directory (an artifact path; "
                             "repeatable), each in the root --root-template names")
    parser.add_argument("--root-template", metavar="PATH",
                        help=f"each block's root, an artifact path holding {{{BLOCK}}} (its stem)")
    parser.add_argument("--preflight", choices=PREFLIGHT_WHERE,
                        help="where each condition's preflight runs: here (local, the default) or not at all (none)")
    parser.add_argument("--preflight-only", action="store_true", default=None,
                        help="preflight every condition, then stop: submit nothing")


def train_options(parser: argparse.ArgumentParser) -> None:
    """Add the train verb's own options."""
    parser.add_argument("--run-name", required=True, metavar="NAME",
                        help="the run (the training program's run_name): its directory is training/<NAME> under "
                             "the artifact root")
    parser.add_argument("--resume", metavar="none|latest|CHECKPOINT",
                        help="none (the default): a fresh run, refused over existing checkpoints; latest: from the "
                             "newest checkpoint, else fresh (on SLURM, requeued after a preemption or its wall); or a "
                             "checkpoint directory (an artifact path)")
    parser.add_argument("--expect", action="append", metavar="KEY=VALUE",
                        help="refuse unless the config composed here, then the preflight's, holds KEY=VALUE "
                             "(repeatable)")
    parser.add_argument("--preflight-only", action="store_true", default=None,
                        help="preflight and check --expect, then stop: submit nothing")
    parser.add_argument("--completions-per-step", type=int, metavar="N",
                        help="refuse unless one optimizer step trains on N completions over the ranks (ranks x "
                             "per-device batch x gradient accumulation)")
    parser.add_argument("--replay-rows-per-step", type=int, metavar="N",
                        help="refuse unless one step replays N anchor rows over the ranks (ranks x per-device "
                             "batch x anchor_accum)")


def add_launch_verbs(sub: Any, *, program: str, **kwargs: Any) -> dict[str, argparse.ArgumentParser]:
    """Add the launch verbs as subcommands of `sub` and return their parsers by name."""
    own = {"campaign": campaign_options, "train": train_options}
    parsers = {}
    for verb, (text, epilog) in LAUNCH_VERBS.items():
        parser = sub.add_parser(verb, help=text, epilog=epilog.format(program=program), **kwargs)
        job_options(parser, verb=verb)
        if verb in own:
            own[verb](parser)
        parser.set_defaults(options_parser=parser)
        parsers[verb] = parser
    return parsers


def add_control_verbs(sub: Any, **kwargs: Any) -> dict[str, argparse.ArgumentParser]:
    """Add the control verbs as subcommands of `sub` and return their parsers by name."""
    parsers = {}
    for verb, text in VERBS.items():
        parser = sub.add_parser(verb, help=text, description=text, **kwargs)
        parser.add_argument("launch", help="a launch id, or a launch directory")
        parser.add_argument("--json", action="store_true", help="print one JSON object on stdout")
        parsers[verb] = parser
    parsers["wait"].add_argument("--timeout-min", type=float,
                                 help="give up after this long (default: the longest job's limit, plus 30)")
    parsers["wait"].add_argument("--pull", action="store_true", help="then pull, unless it timed out")
    stopping = parsers["stop"]
    stopping.add_argument("--reason", help="why, logged in the launch's stop.json")
    stopping.add_argument("--after-save", type=int, metavar="N",
                          help="wait for the training run's checkpoint-N, then stop it (it resumes with --resume "
                               "latest: no STOP.json is written); exit 3, stopping nothing, once its next save has "
                               "begun")
    stopping.add_argument("--timeout-min", type=float,
                          help="with --after-save: give up after this long (default: the run's limit, plus 30)")
    return parsers
