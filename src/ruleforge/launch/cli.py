from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Optional, Sequence

import yaml
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ruleforge.launch import campaign, train
from ruleforge.launch.backends import ROOTS, Backend, backend_for, preflight_request, run_preflight
from ruleforge.launch.control import (
    BLOCK,
    DEFAULT_CPUS_PER_GPU,
    DEFAULT_MEM_GB_PER_GPU,
    DEFAULT_TIMEOUT_MIN,
    SPECS_DIR,
    add_control_verbs,
    add_launch_verbs,
)
from ruleforge.core.credentials import refused_arguments
from ruleforge.core.provenance import repo_root
from ruleforge.launch.environment import artifact_relative, run_workdir
from ruleforge.launch.guards import CAMPAIGN_MARKER, check_code, check_name, code_record
from ruleforge.launch.jobs import JobSpec, Resources, fill, members
from ruleforge.launch.environment import LaunchError
from ruleforge.launch.profile import Profile, find_profile
from ruleforge.launch.records import (
    WAIT_FAILED,
    WAIT_SUCCEEDED,
    WAIT_TIMED_OUT,
    LaunchRecord,
    check_claim_places,
    check_records_place,
    check_writes,
    claim_roots,
    later_records,
    launches_writing,
    live_launches,
    live_writers,
    lock_path,
    locked,
    new_launch_id,
    pull,
    records_root,
    stop,
    succeeded,
    summary,
    wait,
    writers_dir,
)
from ruleforge.core.errors import ArtifactRootError
from ruleforge.env import artifact_root

#: What a spec may not set: it would apply to every launch of the spec.
NOT_IN_SPECS = ("spec", "var", "dry-run", "json", "allow-dirty", "new-root", "preflight", "preflight-only", "only",
                "no-shards", "run-name", "resume", "ignore-launch", "profile")
#: Options that mean something on SLURM only; refused on this machine.
SLURM_ONLY = ("profile", "qos", "resumable", "cpus", "mem_gb", "gpu_count", "timeout_min")
_VAR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
BUILT_IN_VARS = ("launch_id", "member", BLOCK, *ROOTS)
_NOT_SETTINGS = ("args", "spec", "var", "verb", "handler", "options_parser", "command_line")
ENTRY_POINT = ("python", "-m", "ruleforge.launch.cli")
PROGRAM = " ".join(ENTRY_POINT)


class Output:
    """Human lines on stdout, or on stderr under `--json`, whose one JSON object is then all stdout holds."""

    def __init__(self, as_json: bool, *, follow: str = PROGRAM) -> None:
        self.as_json = as_json
        self.follow = follow

    def say(self, line: str) -> None:
        print(line, file=sys.stderr if self.as_json else sys.stdout, flush=True)

    def result(self, payload: Mapping[str, Any], lines: Sequence[str] = ()) -> None:
        if self.as_json:
            print(json.dumps(payload, sort_keys=True, default=str), flush=True)
            return
        for line in lines:
            self.say(line)


class SpecLoader(yaml.SafeLoader):
    """YAML with only its unambiguous plain scalars typed: booleans, null and whole numbers; everything else stays the
    text written."""


SpecLoader.yaml_implicit_resolvers = {}
for _tag, _pattern, _first in (("bool", r"^(?:true|false)$", "tf"), ("null", r"^(?:~|null|)$", ["~", "n", ""]),
                               ("int", r"^(?:0|-?[1-9][0-9]*)$", "-0123456789")):
    SpecLoader.add_implicit_resolver(f"tag:yaml.org,2002:{_tag}", re.compile(_pattern), list(_first))


def _text(value: Any, where: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise LaunchError(f"{where}: {value!r} is not text: quote it")
    return str(value)


def parse_vars(items: Sequence[str]) -> dict[str, str]:
    given: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not _VAR_NAME.fullmatch(name):
            raise LaunchError(f"--var {item!r}: expected NAME=VALUE")
        if name in given:
            raise LaunchError(f"--var gives {name} twice")
        given[name] = value
    return given


def spec_options(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """Return the long options a spec may set, by name."""
    return {option[2:]: action for action in parser._actions for option in action.option_strings
            if option.startswith("--") and option[2:] not in (*NOT_IN_SPECS, "help")
            and not (isinstance(action, argparse.BooleanOptionalAction) and option.startswith("--no-"))}


def resolve_spec(value: str, repo: Path) -> Path:
    """Return a spec's path: as given, else a name under `configs/launch/`."""
    path = Path(value).expanduser()
    if path.is_file():
        return path
    named = repo / SPECS_DIR / (value if value.endswith((".yaml", ".yml")) else f"{value}.yaml")
    if named.is_file():
        return named
    raise LaunchError(f"no launch spec {value!r}: a path, or a name under {SPECS_DIR}/")


def _typed(action: argparse.Action, key: str, value: Any, values: Mapping[str, str]) -> Any:
    text = fill(_text(value, key), values)
    try:
        result = action.type(text) if action.type else text
    except ValueError as error:
        raise LaunchError(f"{key}: {text!r} is not a {action.type.__name__}") from error
    if action.choices is not None and result not in action.choices:
        raise LaunchError(f"{key}: {result!r} is not one of {', '.join(map(str, action.choices))}")
    return result


def _spec_value(action: argparse.Action, key: str, value: Any, values: Mapping[str, str]) -> Any:
    if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
        if not isinstance(value, bool):
            raise LaunchError(f"{key}: true or false, not {value!r}")
        return value
    if isinstance(action, argparse._AppendAction):
        return [_typed(action, key, item, values) for item in (value if isinstance(value, list) else [value])]
    return _typed(action, key, value, values)


PRESET_KEY = "preset"


def _read_spec(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    data = yaml.load(text, Loader=SpecLoader)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise LaunchError(f"{path}: a launch spec is a mapping of options")
    data = dict(data)
    declared = data.pop("vars", None) or {}
    if isinstance(declared, list):
        declared = dict.fromkeys(declared)
    if not isinstance(declared, dict):
        raise LaunchError(f"{path}: vars is a mapping of name to default (null: --var must give it), or a list")
    return data, declared, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _spec_file(path: Path, sha256: str, repo: Path) -> dict[str, str]:
    shown = path.resolve()
    return {"path": str(shown.relative_to(repo.resolve()) if shown.is_relative_to(repo.resolve()) else shown),
            "sha256": sha256}


def _preset(path: Path, data: dict[str, Any], declared: dict[str, Any], repo: Path) -> tuple[
        dict[str, str], dict[str, Any], dict[str, Any]]:
    name = data.pop(PRESET_KEY)
    if data:
        raise LaunchError(f"{path}: a preset gives its spec ({name}) and values of the spec's vars, nothing else "
                          f"(not {', '.join(map(str, sorted(data)))})")
    base = resolve_spec(_text(name, PRESET_KEY), repo)
    base_data, base_declared, base_sha256 = _read_spec(base)
    if PRESET_KEY in base_data:
        raise LaunchError(f"{path}: its spec {base} is a preset itself: name the spec it presets")
    unknown = sorted(set(declared) - set(base_declared))
    unset = sorted(variable for variable, value in declared.items() if value is None)
    if unknown or unset:
        raise LaunchError(f"{path}: a preset gives values of {base}'s vars ({', '.join(base_declared)})"
                          + (f"; it declares no {', '.join(unknown)}" if unknown else "")
                          + (f"; {', '.join(unset)} given no value" if unset else ""))
    return _spec_file(base, base_sha256, repo), base_data, {**base_declared, **declared}


def load_spec(
    path: Path, given: Mapping[str, str], launch_id: str, options: Mapping[str, argparse.Action], *, repo: Path,
    verb: str = "job",
) -> tuple[dict[str, Any], list[str], dict[str, Any], dict[str, str]]:
    """Return a spec's option values, its args, its record and the values its placeholders were filled with; a preset is
    read as its spec with the preset's values as defaults."""
    data, declared, sha256 = _read_spec(path)
    preset_of = None
    if PRESET_KEY in data:
        preset_of, data, declared = _preset(path, data, declared, repo)
    for name in declared:
        if not isinstance(name, str) or not _VAR_NAME.fullmatch(name) or name in BUILT_IN_VARS:
            raise LaunchError(f"{path}: vars.{name}: a name ({_VAR_NAME.pattern}), and not a built-in "
                              f"({', '.join(BUILT_IN_VARS)})")
    undeclared = sorted(set(given) - set(declared))
    missing = sorted(name for name, default in declared.items() if default is None and name not in given)
    if undeclared:
        raise LaunchError(f"--var {', '.join(undeclared)}: not declared in {path}'s vars ({', '.join(declared)})")
    if missing:
        raise LaunchError(f"{path} needs --var for {', '.join(missing)}")
    values = {name: given[name] if name in given else _text(default, f"vars.{name}")
              for name, default in declared.items()}
    values["launch_id"] = launch_id
    raw_args = data.pop("args", None) or []
    if not isinstance(raw_args, list):
        raise LaunchError(f"{path}: args is a list (the command, or the overrides)")
    arguments = [fill(_text(item, "args"), values) for item in raw_args]
    unknown = sorted(set(data) - set(options))
    if unknown:
        raise LaunchError(f"{path}: unknown key(s) {', '.join(map(str, unknown))} (a spec's keys are the {verb} "
                          "verb's long options, vars and args: is it a spec for another verb?)")
    parsed = {options[key].dest: _spec_value(options[key], key, value, values) for key, value in data.items()}
    record = {**_spec_file(path, sha256, repo),
              "vars": {name: value for name, value in values.items() if name != "launch_id"},
              **({"preset_of": preset_of} if preset_of else {})}
    return parsed, arguments, record, values


def job_settings(
    args: argparse.Namespace, parser: argparse.ArgumentParser, launch_id: str, repo: Path
) -> tuple[dict[str, Any], list[str], Optional[dict[str, Any]]]:
    """Return the job's settings (the command line's over the spec's), its command and the spec's record, with the
    profile read and the QOS resolved."""
    given = parse_vars(args.var or [])
    values: dict[str, str] = {"launch_id": launch_id}
    spec_values: dict[str, Any] = {}
    spec_args: list[str] = []
    spec_record = None
    if args.spec is not None:
        path = resolve_spec(args.spec, repo)
        spec_values, spec_args, spec_record, values = load_spec(path, given, launch_id, spec_options(parser),
                                                                repo=repo, verb=args.verb)
        spec_values.setdefault("name", path.stem)
    elif given:
        raise LaunchError("--var fills a spec's placeholders: give --spec")
    command_line = {dest: value for dest, value in vars(args).items()
                    if value is not None and dest not in _NOT_SETTINGS}
    settings = {**spec_values, **command_line}
    if settings.get("profile") is not None:
        settings["profile"] = find_profile(settings["profile"])
        if settings.get("qos") is None:
            settings["qos"] = settings["profile"].qos(settings.get("gpu_count") or 0)
    return settings, [*spec_args, *(fill(item, values) for item in args.args)], spec_record


def _backend_name(settings: Mapping[str, Any]) -> str:
    backend = settings.get("backend")
    if backend is None:
        raise LaunchError("--backend is required (or the spec's backend)")
    return backend


def build_job(settings: Mapping[str, Any], arguments: Sequence[str]) -> tuple[str, JobSpec, dict[str, Any]]:
    """Return the backend's name, the job and the backend's options; settings that disagree are refused."""
    backend = _backend_name(settings)
    for option in SLURM_ONLY:
        if settings.get(option) not in (None, False) and backend != "slurm":
            raise LaunchError(f"--{option.replace('_', '-')} applies to slurm only, not {backend}")
    if settings.get("pull") and not settings.get("wait"):
        raise LaunchError("--pull comes with --wait (or run the pull verb once the runs have ended)")
    if not arguments:
        raise LaunchError("no command: put it after -- (or in the spec's args)")
    count = settings.get("gpu_count", 0)
    each = max(count, 1)
    profile: Optional[Profile] = settings.get("profile")
    given = {"cpus": settings.get("cpus"), "mem_gb": settings.get("mem_gb"), "timeout_min": settings.get("timeout_min"),
             "qos": settings.get("qos"), "resumable": bool(settings.get("resumable"))}
    if profile is not None:
        resources = profile.resources(count, **given)
    else:
        resources = Resources(
            kind="gpu" if count else "cpu", gpus=count,
            cpus=DEFAULT_CPUS_PER_GPU * each if given["cpus"] is None else given["cpus"],
            mem_gb=DEFAULT_MEM_GB_PER_GPU * each if given["mem_gb"] is None else given["mem_gb"],
            timeout_min=DEFAULT_TIMEOUT_MIN if given["timeout_min"] is None else given["timeout_min"],
            qos=given["qos"], resumable=given["resumable"],
        )
    if count < 0 or min(resources.cpus, resources.mem_gb, resources.timeout_min) < 1:
        raise LaunchError(f"--gpu-count {count}, --cpus {resources.cpus}, --mem-gb {resources.mem_gb} and "
                          f"--timeout-min {resources.timeout_min}: GPUs at least 0, the rest at least 1")
    spec = JobSpec(
        check_name(settings.get("name", "job"), what="--name"), "job", 0, tuple(arguments), resources,
        outputs=tuple(artifact_relative(path) for path in settings.get("output") or ()),
        workdir=run_workdir(settings.get("workdir", ".")), env=tuple(settings.get("env") or ()),
    )
    options = {"allow_dirty": bool(settings.get("allow_dirty"))}
    return backend, spec, options


def make_backend(name: str, repo: Path, options: Mapping[str, Any], settings: Mapping[str, Any], out: "Output",
                 backend_kwargs: Mapping[str, Any]) -> Backend:
    """Return the verb's backend, on the settings' cluster profile when one is named."""
    profile = settings.get("profile")
    return backend_for(name, repo, options, echo=out.say, **({"profile": profile} if profile is not None else {}),
                       **backend_kwargs)


def _profile_record(settings: Mapping[str, Any]) -> dict[str, Any]:
    profile: Optional[Profile] = settings.get("profile")
    return {} if profile is None else {"profile": str(profile.path)}


def _request_lines(job: Mapping[str, Any]) -> list[str]:
    request = job["request"]
    return [f"== {job['job_id']}", shlex.join(request["argv"]),
            *([json.dumps(request["submitit"], sort_keys=True)] if "submitit" in request else [])]


def _wait_and_pull(
    record: LaunchRecord, backend: Backend, out: Output, *, timeout_min: Optional[float] = None,
    pull_after: bool = False,
) -> tuple[int, dict[str, Any]]:
    if pull_after:
        check_writes(record, merges=True)
    status, states = wait(record, backend, timeout_min=timeout_min, echo=out.say)
    if status == WAIT_FAILED and record.data.get("campaign"):
        _relaunch_hint(record, {job_id for job_id, entry in states.items() if not succeeded(entry)}, out)
    result: dict[str, Any] = {"wait": status, "states": states}
    if pull_after and status != WAIT_TIMED_OUT:
        result["pull"] = pull(record, backend)
        if status == WAIT_SUCCEEDED and _pull_incomplete(result["pull"]):
            status = WAIT_FAILED
    return status, result


def _pull_incomplete(report: Mapping[str, Any]) -> bool:
    return any(report.get(key) for key in ("pull_errors", "held", "merge_skipped", "merge_errors"))


def refuse_live_writers(paths: Collection[str], repo: Path, backend_kwargs: Mapping[str, Any],
                        settings: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a launch while another launch that writes one of `paths` has a run that has not ended, unless
    `--ignore-launch` names it; return what the record keeps of those."""
    ignored = list(settings.get("ignore_launch") or ())
    if settings.get("dry_run") or not paths:
        return {"ignored_launches": ignored} if ignored else {}
    live = live_writers(records_root(), paths, _states_of(repo, backend_kwargs), ignore=ignored)
    if live:
        named = "; ".join(f"{entry['launch_id']} writes {', '.join(entry['paths'])} ({summary(entry['states'])})"
                          for entry in live)
        _refuse_live(named, "output")
    return {"ignored_launches": ignored} if ignored else {}


def refuse_live_campaign_jobs(conditions: Sequence[campaign.Condition], indices: Sequence[int],
                              roots: Sequence[Path], repo: Path, backend_kwargs: Mapping[str, Any],
                              settings: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a campaign launch while another launch has a live run of one of this launch's jobs (same campaign root,
    condition id and shard layout), unless `--ignore-launch` names it; return what the record keeps of those."""
    ignored = list(settings.get("ignore_launch") or ())
    if settings.get("dry_run"):
        return {"ignored_launches": ignored} if ignored else {}
    base = artifact_root()
    layouts = [(condition.to_json(), {job.job_id for job in condition.jobs(indices)},
                f"{condition.condition_id} in {condition.campaign_root}", campaign.local_root(condition.campaign_root, base))
               for condition in conditions]

    def shared_of(other: LaunchRecord) -> dict[str, str]:
        theirs = other.directory.parent.parent
        return {job_of_theirs: label for recorded, jobs, label, mine in layouts
                for job, job_of_theirs in campaign.shared_jobs(
                    other.data, recorded, is_root=lambda root, mine=mine: campaign.local_root(root, theirs) == mine
                ).items() if job in jobs}

    live = live_launches(launches_writing(records_root(), roots), shared_of, _states_of(repo, backend_kwargs),
                         ignore=ignored)
    if live:
        named = "; ".join(f"{entry['launch_id']} runs {', '.join(sorted(set(entry['shared'].values())))} "
                          f"({summary(entry['states'])}{'; its record is gone' if entry['record_gone'] else ''})"
                          for entry in live)
        _refuse_live(named, "root")
    return {"ignored_launches": ignored} if ignored else {}


def _states_of(repo: Path, backend_kwargs: Mapping[str, Any]) -> Callable[[LaunchRecord], dict[str, dict[str, Any]]]:
    def states_of(other: LaunchRecord) -> dict[str, dict[str, Any]]:
        return backend_for(other.data["backend"], repo, other.data["options"], echo=lambda line: None,
                           **backend_kwargs).states(other)
    return states_of


def _refuse_live(named: str, what: str) -> None:
    raise LaunchError(f"{named}: its run has not ended, and two runs would write one {what} at once. Wait for it "
                      "(launch wait) or stop it (launch stop) first; if its backend cannot say it has ended (a "
                      "launch killed while submitting stays submitting; a local job whose launch failed while it ran, "
                      "and a SLURM job whose submission raised after sbatch was asked, are unknown), look by hand, "
                      "then pass --ignore-launch <its id>")


def launch(
    backend: Backend, specs: Sequence[JobSpec], *, verb: str, launch_id: str, code: Mapping[str, Any],
    spec_record: Optional[Mapping[str, Any]], out: Output, dry_run: bool = False, wait_after: bool = False,
    pull_after: bool = False, merges: Sequence[Mapping[str, Any]] = (), extra: Optional[Mapping[str, Any]] = None,
    refuse: Callable[[], Mapping[str, Any]], claims: Sequence[Path] = (),
    before_submit: Optional[Callable[[], Optional[int]]] = None, command_line: Optional[Sequence[str]] = None,
) -> int:
    """Render the jobs, run `before_submit`, record, submit, then wait and pull when asked. `refuse` is asked before
    anything is spent and again under the records lock right before the record is written."""
    if not specs:
        raise LaunchError("nothing to launch: no job")
    if not dry_run:
        check_records_place()
        check_claim_places(claims)
    refused = dict(refuse())
    specs = [backend.prepare(spec) for spec in specs]
    jobs = [{**spec.to_json(), "request": request} for spec, request in zip(specs, backend.render(specs))]

    def payload() -> dict[str, Any]:
        return {"launch_id": launch_id, "verb": verb, "backend": backend.name, "options": backend.options,
                "repo": str(backend.repo), "code": dict(code), "spec": spec_record, "jobs": jobs,
                "merges": [dict(merge) for merge in merges], **(extra or {}), **refused,
                **({"command_line": list(command_line)} if command_line is not None else {})}

    if dry_run:
        out.result({**payload(), "dry_run": True, "preview": None},
                   [line for job in jobs for line in _request_lines(job)] + ["== dry run: nothing was submitted"])
        return 0
    if before_submit is not None:
        status = before_submit()
        if status is not None:
            return status
    with locked([lock_path(records_root()), *(lock_path(writers_dir(root)) for root in claims)]):
        check_records_place()
        check_claim_places(claims)
        refused = dict(refuse())
        record = LaunchRecord.create(records_root() / launch_id, payload())
        claim_roots(record, claims)
    out.say(f"== launch {launch_id} ({backend.name}, commit {code['commit'][:12]}"
            f"{', DIRTY' if code['dirty'] else ''}): record {record.path}")
    result: dict[str, Any] = {"launch_id": launch_id, "record": str(record.directory), "runs": record.runs}
    try:
        backend.submit(record)
    except BaseException as error:
        record.finish(error)
        out.say(f"== launch {launch_id} stopped part-way; its record names the {len(record.runs)} run(s) it started "
                "or was starting")
        out.result({**result, "error": record.data["error"]})
        if isinstance(error, Exception) and not isinstance(error, LaunchError):
            raise LaunchError(f"submitting failed: {type(error).__name__}: {error}") from error
        raise
    record.finish()
    status, lines = 0, []
    if wait_after or backend.foreground:
        try:
            status, followed = _wait_and_pull(record, backend, out, pull_after=pull_after)
        except Exception as error:
            out.result({**result, "runs": record.runs, "error": f"{type(error).__name__}: {error}"})
            raise
        result.update(followed, runs=record.runs)
        lines = [*_state_lines(record, followed["states"]), *_pull_lines(followed.get("pull"))]
    out.result(result, [*lines, f"== launch {launch_id}: {len(record.runs)} run(s); follow it with "
                                f"{out.follow} status {launch_id}"])
    return status


def _job(args: argparse.Namespace, out: Output, repo: Path, backend_kwargs: Mapping[str, Any]) -> int:
    code = code_record(repo)
    launch_id = new_launch_id(code["commit"])
    settings, arguments, spec_record = job_settings(args, args.options_parser, launch_id, repo)
    check_code(code, expect_commit=settings.get("expect_commit"), allow_dirty=bool(settings.get("allow_dirty")))
    name, spec, options = build_job(settings, arguments)
    specs = members(spec, settings.get("axis") or ())
    backend = make_backend(name, repo, options, settings, out, backend_kwargs)
    written = {path for job in specs for path in job.outputs}
    return launch(backend, specs, verb="job", launch_id=launch_id, code=code, spec_record=spec_record, out=out,
                  dry_run=bool(settings.get("dry_run")), wait_after=bool(settings.get("wait")),
                  pull_after=bool(settings.get("pull")), extra=_profile_record(settings),
                  refuse=lambda: refuse_live_writers(written, repo, backend_kwargs, settings),
                  command_line=getattr(args, "command_line", None))


def _campaign(args: argparse.Namespace, out: Output, repo: Path, backend_kwargs: Mapping[str, Any]) -> int:
    code = code_record(repo)
    launch_id = new_launch_id(code["commit"])
    settings, arguments, spec_record = job_settings(args, args.options_parser, launch_id, repo)
    no_shards = settings.pop("no_shards", None)
    if no_shards:
        if args.shards is not None or args.only is not None:
            raise LaunchError("--no-shards runs each condition as one job: not with --shards or --only")
        settings["shards"] = None
    check_code(code, expect_commit=settings.get("expect_commit"), allow_dirty=bool(settings.get("allow_dirty")))
    name, template, options = build_job({"name": "campaign", **settings}, list(campaign.PROGRAM))
    where = settings.get("preflight") or "local"
    if where == "none" and settings.get("preflight_only"):
        raise LaunchError("--preflight-only with --preflight none would check nothing")
    backend = make_backend(name, repo, options, settings, out, backend_kwargs)
    choice = campaign.CampaignOptions.from_settings(settings)
    indices = campaign.selected_shards(choice.only, choice.shards)

    def holds_campaign(root: campaign.Place) -> bool:
        return (root.local / CAMPAIGN_MARKER).is_file()

    conditions = campaign.plan(repo, template, arguments, settings.get("axis") or (), choice, exists=holds_campaign,
                               allocates_gpus=name == "slurm")
    specs = [job for condition in conditions for job in condition.jobs(indices)]
    requests = [] if where == "none" else [preflight_request(backend, condition.preflight(indices[0]))
                                           for condition in conditions]
    merges = [merge for merge in (condition.merge(indices) for condition in conditions) if merge is not None]
    extra = {"campaign": {"conditions": [condition.to_json() for condition in conditions], "preflight": where,
                          "preflights": [request["shown"] for request in requests]}, **_profile_record(settings)}
    dry_run = bool(settings.get("dry_run"))
    if dry_run:
        for condition, request in zip(conditions, requests):
            out.say(f"== preflight {condition.spec.job_id} ({where}): {shlex.join(request['shown'])}")
    if not dry_run:
        campaign.check_roots(conditions)
    roots = [] if dry_run else sorted({campaign.local_root(condition.campaign_root, artifact_root())
                                       for condition in conditions})

    def preflight() -> Optional[int]:
        for condition, request in zip(conditions, requests):
            out.say(f"== preflight {condition.spec.job_id} ({where}): {condition.condition_id} in "
                    f"{condition.campaign_root}")
            status = run_preflight(backend, request, echo=out.say)
            if status != 0:
                out.say(f"== the preflight of {condition.spec.job_id} refused it (exit {status}): nothing was "
                        "submitted")
                return WAIT_FAILED
        if settings.get("preflight_only"):
            out.say(f"== --preflight-only: {len(conditions)} condition(s) passed their preflight; nothing was "
                    "submitted")
            return 0
        return None

    try:
        return launch(backend, specs, verb="campaign", launch_id=launch_id, code=code, spec_record=spec_record,
                      out=out, dry_run=dry_run, wait_after=bool(settings.get("wait")),
                      pull_after=bool(settings.get("pull")), merges=merges, extra=extra,
                      refuse=lambda: refuse_live_campaign_jobs(conditions, indices, roots, repo, backend_kwargs,
                                                               settings),
                      claims=roots, before_submit=preflight, command_line=getattr(args, "command_line", None))
    except LaunchError:
        _launch_failure_hint(launch_id, out)
        raise


def _launch_failure_hint(launch_id: str, out: Output) -> None:
    try:
        record = LaunchRecord.load(launch_id)
    except LaunchError:
        return
    planned = {job["job_id"] for job in record.jobs}
    # a SLURM run recorded before sbatch answered: sbatch may have queued it
    unread = sorted(run["job_id"] for run in record.runs if "slurm_job" in run and run["slurm_job"] is None)
    missed = planned - {run["job_id"] for run in record.runs if run["job_id"] not in unread}
    then = ""
    if unread:
        out.say(f"== {', '.join(unread)}: the launch failed after sbatch was asked to queue it, and no job id was "
                f"read: look by hand (squeue --me), scancel what it queued; it is launched again below only once "
                f"none is queued or running, with --ignore-launch {launch_id} (the record holds it unknown, never "
                "ended)")
        then = f" once that look found none of it queued or running, with --ignore-launch {launch_id}"
    if missed == planned:
        out.say(f"== {launch_id} launched no job{' for certain' if unread else ''}: launch again with the same "
                f"command{then}, from commit {record.data['code']['commit'][:12]} (--expect-commit)")
        return
    _relaunch_hint(record, missed, out)


def _relaunch_hint(record: LaunchRecord, missed: Collection[str], out: Output) -> None:
    conditions = record.data["campaign"]["conditions"]
    commit, alone = record.data["code"]["commit"][:12], len(conditions) == 1
    later = later_records(record)
    for condition in conditions:
        followed = {job: other.launch_id for other in later for job in campaign.relaunched_jobs(other.data, condition)}
        elsewhere = {job: followed[job] for job in missed if job in followed}
        if elsewhere:
            out.say(f"== {condition['job_id']} ({condition['condition_id']}): {', '.join(sorted(elsewhere))} "
                    f"launched again by {', '.join(sorted(set(elsewhere.values())))}: follow that launch (launch "
                    "wait, then pull, it), and launch none of them again")
        found = campaign.relaunch(condition, set(missed) - elsewhere.keys(), alone=alone)
        if found is None:
            continue
        indices, options = found
        sharded = condition["shards"] is not None
        what = f"shard(s) {','.join(map(str, indices))}" if sharded else "it"
        how = (f" and {shlex.join(options)}" if alone else
               f", narrowed to this condition: {shlex.join(options)} in place of its --axis and --blocks options")
        out.say(f"== {condition['job_id']} ({condition['condition_id']}): launch {what} again with the same command"
                f"{how if options else ''}, from commit {commit} (--expect-commit): a relaunch resumes from its "
                "ledger")
    if record.data["merges"]:
        out.say(f"== pull {record.launch_id} once its runs have ended, and each relaunch once its runs and "
                f"{record.launch_id}'s have: a pull merges each sharded condition whose shards all succeeded in "
                "its launch")


def _train(args: argparse.Namespace, out: Output, repo: Path, backend_kwargs: Mapping[str, Any]) -> int:
    code = code_record(repo)
    launch_id = new_launch_id(code["commit"])
    settings, arguments, spec_record = job_settings(args, args.options_parser, launch_id, repo)
    check_code(code, expect_commit=settings.get("expect_commit"), allow_dirty=bool(settings.get("allow_dirty")))
    ranks = settings.get("gpu_count") or 0
    run, composed = train.plan(repo, arguments, run_name=settings["run_name"],
                               resume=settings.get("resume") or train.RESUME_FRESH, ranks=ranks,
                               expected_per_step={key: settings.get(f"{key}_per_step")
                                                  for key in train.PER_STEP_OPTIONS},
                               env=settings.get("env") or ())
    expected = train.expectations(settings.get("expect") or ())
    train.check_expectations(composed, expected, where="the config composed here")
    if not settings.get("dry_run"):
        train.check_root(run, launch_id, str(composed.output_dir))
    job = {**settings, **train.training_resources(settings, ranks), "name": run.name, "workdir": train.WORKDIR,
           "output": None}
    name, spec, options = build_job(job, run.command(launch_id))
    backend = make_backend(name, repo, {**options, "rank_per_gpu": True}, settings, out, backend_kwargs)
    limits = backend.qos_limits()
    train.check_wall(spec.resources.qos, limits, spec.resources.timeout_min)
    spec = replace(spec, resources=replace(spec.resources,
                                           resumable=train.requeued(spec.resources.qos, limits, run.resume)))
    request = preflight_request(backend, replace(spec, job_id=f"{run.name}-preflight", kind="preflight",
                                                 argv=run.preflight(launch_id), resources=train.PREFLIGHT,
                                                 outputs=()))
    hydra_dir = run.hydra_dir(launch_id, "preflight")
    geometry = run.per_step
    out.say(f"== {run.name}: {run.ranks} rank(s); each step {geometry['completions']} completions "
            f"({geometry['prompts']} prompts) and {geometry['replay_rows']} replay rows; a save every "
            f"{run.save_steps} steps to {run.max_steps}")

    def refuse_writers() -> dict[str, Any]:
        ignored = refuse_live_writers({run.output_dir}, repo, backend_kwargs, settings)
        if not settings.get("dry_run") and run.resume == train.RESUME_FRESH and train.started(backend, run):
            raise LaunchError(f"{run.output_dir} holds a run that has started (its code snapshot): a fresh run "
                              "would mix with it. Resume it (--resume latest), or give another --run-name")
        return ignored

    extra = {"train": {**run.to_json(), "expect": expected, "push": [],
                       "preflight": {"where": "local", "command": request["shown"], "hydra_dir": hydra_dir}},
             **_profile_record(settings)}
    if settings.get("dry_run"):
        out.say(f"== preflight (local): {shlex.join(request['shown'])}")

    def before_submit() -> Optional[int]:
        out.say(f"== preflight (local) of {run.name}: {hydra_dir}")
        status = run_preflight(backend, request, echo=out.say)
        if status != 0:
            out.say(f"== the preflight refused {run.name} (exit {status}): nothing was submitted")
            return WAIT_FAILED
        found = train.preflight_config(hydra_dir)
        train.check_expectations(OmegaConf.create(found["config"]), expected, where="the preflight's config")
        extra["train"]["preflight"].update(found)
        if settings.get("preflight_only"):
            out.say(f"== --preflight-only: {run.name} passed its preflight; nothing was submitted")
            return 0
        extra["train"]["saved_before"] = train.saved_before(backend, run)
        return None

    return launch(backend, [spec], verb="train", launch_id=launch_id, code=code, spec_record=spec_record, out=out,
                  dry_run=bool(settings.get("dry_run")), wait_after=bool(settings.get("wait")),
                  pull_after=bool(settings.get("pull")), extra=extra, refuse=refuse_writers,
                  before_submit=before_submit, command_line=getattr(args, "command_line", None))


def _loaded(args: argparse.Namespace, out: Output, repo: Path,
            backend_kwargs: Mapping[str, Any]) -> tuple[LaunchRecord, Backend]:
    record = LaunchRecord.load(args.launch)
    backend = backend_for(record.data["backend"], repo, record.data["options"], echo=out.say, **backend_kwargs)
    return record, backend


def _state_lines(record: LaunchRecord, states: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = []
    for job_id, entry in states.items():
        run = record.run_for(job_id) or {}
        where = " ".join(f"{key}={run[key]}" for key in ("slurm_job", "run_id", "exit_code") if key in run)
        notes = [note for note in (entry.get("error"), entry.get("refused") and f"refused: {entry['refused']}")
                 if note]
        lines.append(f"{job_id}  {entry['state']:<12}  {where}" + "".join(f"  ({note})" for note in notes))
    return lines


def _status(args: argparse.Namespace, out: Output, repo: Path,
            backend_kwargs: Mapping[str, Any]) -> int:
    record, backend = _loaded(args, out, repo, backend_kwargs)
    states = backend.states(record)
    launch_state, error = record.data["state"], record.data.get("error")
    out.result({"launch_id": record.launch_id, "backend": record.data["backend"], "launch_state": launch_state,
                "error": error, "states": states},
               [f"== {record.launch_id} ({record.data['backend']}): {summary(states)}",
                *([f"== the launch is {launch_state}" + (f": {error}" if error else "")]
                  if launch_state != "submitted" else []),
                *_state_lines(record, states)])
    return 0


def _wait(args: argparse.Namespace, out: Output, repo: Path,
          backend_kwargs: Mapping[str, Any]) -> int:
    record, backend = _loaded(args, out, repo, backend_kwargs)
    status, followed = _wait_and_pull(record, backend, out, timeout_min=args.timeout_min, pull_after=args.pull)
    out.result({"launch_id": record.launch_id, **followed},
               _state_lines(record, followed["states"]) + _pull_lines(followed.get("pull"))
               + (["== still running after the wait's limit: wait again, or stop it"] if status == WAIT_TIMED_OUT
                  else []))
    return status


def _pull_lines(report: Optional[Mapping[str, Any]]) -> list[str]:
    if report is None:
        return []
    return ([f"== pulled {len(report['pulled'])} path(s)"
             + (f"; merged {len(report['merged'])}" if report["merged"] else "")]
            + [f"== not pulled: {path}: {error}" for path, error in report.get("pull_errors", {}).items()]
            + [f"== held: its run did not succeed and no later launch's pull took it (its relaunch's pull does): "
               f"{path}" for path in report.get("held", [])]
            + [f"== merge of {entry['condition_id']} into {entry['campaign_root']}: merged already by "
               f"{entry['launch_id']}" for entry in report.get("merged_elsewhere", [])]
            + [f"== merge of {entry['condition_id']} into {entry['campaign_root']} skipped: {entry['reason']}"
               for entry in report.get("merge_skipped", [])]
            + [f"== merge of {entry['condition_id']} into {entry['campaign_root']} refused: {entry['error']}"
               for entry in report.get("merge_errors", [])])


def _pull(args: argparse.Namespace, out: Output, repo: Path,
          backend_kwargs: Mapping[str, Any]) -> int:
    record, backend = _loaded(args, out, repo, backend_kwargs)
    report = pull(record, backend)
    out.result({"launch_id": record.launch_id, **report}, _pull_lines(report))
    return WAIT_FAILED if _pull_incomplete(report) else 0


def _stop(args: argparse.Namespace, out: Output, repo: Path,
          backend_kwargs: Mapping[str, Any]) -> int:
    record, backend = _loaded(args, out, repo, backend_kwargs)
    if args.after_save is None:
        if args.timeout_min is not None:
            raise LaunchError("--timeout-min bounds the wait of --after-save")
        entry, found = stop(record, backend, reason=args.reason), {}
        status = None
    else:
        status, found = train.stop_after_save(record, backend, args.after_save, reason=args.reason,
                                              timeout_min=args.timeout_min, echo=out.say)
        if "stop" not in found:
            out.result({"launch_id": record.launch_id, **found})
            return status
        entry = found.pop("stop")
    errors, submitting = entry.get("stop_errors", {}), entry.get("submitting", [])
    out.result({"launch_id": record.launch_id, **found, **entry},
               [f"== stop requested for {', '.join(entry['stopped']) or 'nothing (every named run has ended)'}",
                *(f"== not stopped: {job_id}: {error}" for job_id, error in errors.items()),
                *([f"== still being submitted: {', '.join(submitting)}; the launch submits no later job: stop again "
                   "once status names their runs"] if submitting else [])])
    if status is not None:
        return status
    return WAIT_FAILED if errors or submitting else 0


LAUNCH_HANDLERS: dict[str, Callable[..., int]] = {"job": _job, "campaign": _campaign, "train": _train}
CONTROL_HANDLERS: dict[str, Callable[..., int]] = {"status": _status, "wait": _wait, "pull": _pull, "stop": _stop}


#: What a verb refuses with (exit 2).
REFUSALS = (LaunchError, ArtifactRootError, ValueError, yaml.YAMLError, OmegaConfBaseException, OSError, KeyError)


def run_verb(args: argparse.Namespace, handler: Callable[..., int], *, program: str, repo: Optional[Path] = None,
             follow: str = PROGRAM, refusals: tuple[type[BaseException], ...] = REFUSALS, **backend_kwargs: Any) -> int:
    """Run a verb's parsed `args` with its `handler` and return its exit status; a refusal among `refusals` is
    `<program> <verb>: <reason>` on stderr and 2 (`ruleforge jobs` passes none: its own boundary answers them)."""
    try:
        return handler(args, Output(bool(args.json), follow=follow), Path(repo) if repo else repo_root(),
                       backend_kwargs)
    except refusals as error:
        detail = f"{type(error).__name__}: {error}" if isinstance(error, (OSError, KeyError)) else str(error)
        print(f"{program} {args.verb}: {detail}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    """Return the parser; `job`'s command is everything after the first `--`, split off by `main`."""
    parser = argparse.ArgumentParser(prog=PROGRAM, description="Launch a job on SLURM or this machine, and follow it.")
    sub = parser.add_subparsers(dest="verb", required=True)
    for verb, launching in add_launch_verbs(sub, program=PROGRAM).items():
        launching.set_defaults(handler=LAUNCH_HANDLERS[verb])
    for verb, control in add_control_verbs(sub).items():
        control.set_defaults(handler=CONTROL_HANDLERS[verb])
    return parser


def main(argv: Optional[Sequence[str]] = None, *, repo: Optional[Path] = None,
         entry_point: Sequence[str] = ENTRY_POINT, **backend_kwargs: Any) -> int:
    """Run the command line; everything after the first `--` is `job`'s command."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if refused_arguments(argv, PROGRAM):
        return 2
    command_line = [*entry_point, *argv]
    command: Optional[list[str]] = None
    if "--" in argv:
        split = argv.index("--")
        argv, command = argv[:split], argv[split + 1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    if command is not None and args.verb not in ("job", "campaign", "train"):
        parser.error(f"{args.verb} takes no command after --")
    args.args = command or []
    args.command_line = command_line
    return run_verb(args, args.handler, program=PROGRAM, repo=repo, **backend_kwargs)


if __name__ == "__main__":
    sys.exit(main())
