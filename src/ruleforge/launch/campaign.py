from __future__ import annotations

import dataclasses
import itertools
import os
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Optional, Sequence

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ruleforge.launch.backends import PREFLIGHT_RESOURCES
from ruleforge.launch.control import BLOCK
from ruleforge.launch.environment import artifact_relative, relative_subpath
from ruleforge.launch.guards import (
    check_campaign_root,
    check_exports,
    check_name,
    check_overrides,
    check_packing,
    check_population,
    check_spelled,
    compose,
    hydra_arguments,
    override_keys,
    reaches,
    refuse_linked_way,
    unset_variable,
)
from ruleforge.launch.jobs import JobSpec, fill, members, parse_axis
from ruleforge.launch.environment import LaunchError
from ruleforge.core.errors import ArtifactRootError
from ruleforge.env import artifact_root
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.sharding import SHARDS_SUFFIX, job_root_override, shard_dirname, shard_root, shards_root

#: Where the campaign's programs run: the checkout's top.
WORKDIR = "."
PROGRAM = ("python", "-m", "ruleforge.launch.programs.run_campaign")
PREFLIGHT_PROGRAM = ("python", "-m", "ruleforge.launch.programs.preflight_campaign")
DEFAULT_CONFIG = "sampling_campaign"
#: A preflight's own Hydra directory, quoted because Hydra's override grammar nests an interpolation only in a quoted value.
PREFLIGHT_RUN_DIR = ("hydra.run.dir='${rf.unpinned:${campaign_root}}/hydra/preflight/${condition_id}/"
                     "${now:%Y-%m-%d_%H-%M-%S}'")
SHARD_PREFLIGHT_RUN_DIR = (
    "hydra.run.dir='${rf.unpinned:${campaign_root}" + SHARDS_SUFFIX + "/${condition_id}}/preflight/"
    "${now:%Y-%m-%d_%H-%M-%S}'"
)
LOCAL_JUDGE = "local"
#: Keys the verb's own options set, so a job's arguments may not.
POPULATION_KEYS = ("scenario_set_path", "approved_scenario_sha256")
PACKING_KEYS = ("execution.pack_draws", "execution.pack_max_prompts", "execution.pack_max_draws")


@dataclass(frozen=True)
class CampaignOptions:
    """The campaign verb's own options, as its command line and a spec name them."""

    campaign_id: Optional[str] = None
    campaign_root: Optional[str] = None
    new_root: bool = False
    population: Optional[str] = None
    expect_population_sha256: Optional[str] = None
    pack_draws: bool = False
    pack_max_prompts: Optional[int] = None
    pack_max_draws: Optional[int] = None
    shards: Optional[int] = None
    only: Optional[str] = None
    blocks: tuple[str, ...] = ()
    root_template: Optional[str] = None

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "CampaignOptions":
        values = {field.name: settings[field.name] for field in dataclasses.fields(cls)
                  if settings.get(field.name) is not None}
        return cls(**{**values, "blocks": tuple(values.get("blocks", ()))})


@dataclass(frozen=True)
class Place:
    """A path a campaign names: where it is here, and where it is below the artifact root (None: outside it)."""

    local: Path
    relative: Optional[str]

    @property
    def named(self) -> str:
        """Return the path as a job names it."""
        return str(self.local)

    @property
    def recorded(self) -> str:
        """Return the path as the record keeps it: below the artifact root, else absolute."""
        return self.relative or str(self.local)


def place(value: str, *, what: str) -> Place:
    """Return an artifact path (relative to the artifact root, or absolute); an interpolation is refused."""
    text = str(value).strip()
    if "$" in text:
        raise LaunchError(f"{what} {text}: an artifact path (relative to $RULEFORGE_ARTIFACT_ROOT, or absolute), not an "
                          "interpolation: the launcher names it as each backend's jobs read it")
    path = Path(text).expanduser()
    if not path.is_absolute():
        relative = relative_subpath(text, what=what)
        return Place(artifact_root() / relative, relative)
    try:
        return Place(path, artifact_relative(str(path)))
    except (ValueError, ArtifactRootError):
        return Place(path, None)


def job_name(job_id: str, index: int, shards: Optional[int]) -> str:
    """Return the name of a condition's job: its own, or its shard `index`'s."""
    return job_id if shards is None else f"{job_id}-{shard_dirname(index, shards)}"


def shard_overrides(index: int, count: int) -> list[str]:
    """Return the overrides that make a campaign job shard `index` of `count`."""
    return [f"execution.num_shards={count}", f"execution.shard_index={index}", job_root_override(index, count)]


def selected_shards(only: Optional[str], count: Optional[int]) -> list[int]:
    """Return `--only 3,7` as shard indices (every shard without it); one outside the layout is refused."""
    if count is not None and count < 1:
        raise LaunchError(f"--shards {count}: at least 1")
    if only is None:
        return list(range(count or 1))
    if count is None:
        raise LaunchError("--only selects shards: give --shards")
    try:
        indices = sorted({int(item) for item in only.split(",") if item.strip()})
    except ValueError as error:
        raise LaunchError(f"--only {only!r}: comma-separated shard indices") from error
    if not indices or not all(0 <= index < count for index in indices):
        raise LaunchError(f"--only {only!r}: shard indices in [0, {count})")
    return indices


def resolve_condition(repo: Path, arguments: Sequence[str], *, env: Sequence[str] = ()) -> dict[str, Any]:
    """Return a campaign job's campaign id, root, condition id, judge model type and its batch judge's `batch_root`, from
    its config composed as Hydra composes it."""
    cfg = compose(repo, arguments, default_config=DEFAULT_CONFIG, env=env, what="a campaign job")
    check_exports(cfg, env, what="a campaign job")
    try:
        judge = str(OmegaConf.select(cfg, "judge_model.model_type", default=""))
        return {**{key: str(cfg[key]) for key in ("campaign_id", "campaign_root", "condition_id")},
                "judge_model_type": judge, "batch_root": job_batch_root(cfg, judge)}
    except OmegaConfBaseException as error:
        raise LaunchError(f"{shlex.join(arguments)}: the config does not compose: {type(error).__name__}: "
                          f"{error}") from None


def job_batch_root(cfg: Any, judge: str) -> Optional[str]:
    """Return where a campaign job's batch judge keeps its batches' state, or None for a judge of another kind; one that
    does not resolve here is refused."""
    from ruleforge.judges.backend import BATCH_MODEL_TYPES

    if judge not in BATCH_MODEL_TYPES:
        return None
    try:
        value = OmegaConf.select(cfg, "judge_model.batch_root", default=None)
    except OmegaConfBaseException as error:
        unset = unset_variable(error)
        refusal = "the batch judge's batch_root does not resolve where the launch runs"
        if unset is not None:
            raise LaunchError(f"{refusal}: {unset}, which it reads, is unset here, and the launch checks the place it "
                              f"names before the job's transport writes its paid state there; set {unset} (the "
                              "variable it reads) where the launch runs, before the launch -- its --preflight-only "
                              "too -- or give batch_root as an absolute path") from None
        raise LaunchError(f"{refusal} ({type(error).__name__}: {str(error).splitlines()[0]}): it would resolve only in "
                          "the job, at a place no check of the launch saw, where the job's transport would write its "
                          "paid state; give batch_root as an absolute path, or one read from a variable set where the "
                          "launch runs") from None
    return str(value) if value else None


@dataclass(frozen=True)
class Condition:
    """One condition of a campaign: its job, where it writes, how it is cut into shards, the options that select it alone
    and its batch judge's `batch_root`."""

    spec: JobSpec
    campaign_id: str
    campaign_root: str
    condition_id: str
    shards: Optional[int] = None
    selection: tuple[str, ...] = ()
    named_root: Optional[str] = None
    batch_root: Optional[str] = None

    @property
    def arguments(self) -> tuple[str, ...]:
        """Return its job's Hydra arguments."""
        return self.spec.argv[len(PROGRAM):]

    def jobs(self, indices: Sequence[int]) -> list[JobSpec]:
        """Return its jobs: itself, or its shards `indices`."""
        if self.shards is None:
            return [self.spec]
        return [replace(self.spec, job_id=job_name(self.spec.job_id, index, self.shards),
                        argv=(*self.spec.argv, *shard_overrides(index, self.shards)))
                for index in indices]

    def preflight(self, first: int) -> JobSpec:
        """Return its preflight: its arguments (a shard's when sharded) with a Hydra directory in none of its jobs'
        roots."""
        shard = [] if self.shards is None else shard_overrides(first, self.shards)
        run_dir = PREFLIGHT_RUN_DIR if self.shards is None else SHARD_PREFLIGHT_RUN_DIR
        return replace(self.spec, job_id=f"{self.spec.job_id}-preflight", kind="preflight",
                       argv=(*PREFLIGHT_PROGRAM, *self.arguments, *shard, run_dir), resources=PREFLIGHT_RESOURCES,
                       outputs=())

    def merge(self, indices: Sequence[int]) -> Optional[dict[str, Any]]:
        """Return the record's merge of its shards into its campaign root (None unsharded)."""
        if self.shards is None:
            return None
        return {"campaign_root": self.campaign_root, "condition_id": self.condition_id, "num_shards": self.shards,
                "job_ids": [job.job_id for job in self.jobs(indices)]}

    def to_json(self) -> dict[str, Any]:
        """Return what the record keeps of it: enough to relaunch its jobs alone."""
        return {"job_id": self.spec.job_id, "campaign_id": self.campaign_id, "campaign_root": self.campaign_root,
                "condition_id": self.condition_id, "shards": self.shards, "selection": list(self.selection)}


def relaunch(condition: Mapping[str, Any], missed: Collection[str], *,
             alone: bool) -> Optional[tuple[list[int], list[str]]]:
    """Return a recorded condition's shards whose jobs are `missed` and the options that relaunch those alone; None when
    none is missed."""
    shards = condition["shards"]
    indices = [index for index in range(shards or 1) if job_name(condition["job_id"], index, shards) in missed]
    if not indices:
        return None
    cut = [] if shards is None else ["--shards", str(shards), "--only", ",".join(map(str, indices))]
    return indices, [*([] if alone else condition["selection"]), *cut]


def check_roots(conditions: Sequence[Condition]) -> None:
    """Refuse a launch of `conditions` that would write where nothing may (a root through a symlink, inside another
    campaign root, on a pinned root's path or nested in the launch, or a symlink on the way to a job's place); nothing
    is written."""
    from ruleforge.evaluate.api import refuse_protected
    from ruleforge.evaluate.campaign import check_new_condition_id

    _refuse_nested_roots(conditions)
    for condition in conditions:
        cid = condition.condition_id
        root = artifact_root() / condition.campaign_root
        check_spelled(condition.named_root or root, root, what=f"the campaign root of {cid}")
        refuse_protected(root, f"the campaign root of {cid}", campaign_root=True)
        check_new_condition_id(root, cid)
        own: list[Path] = []
        if condition.shards is None:
            base, ways, files = root, [root / "hydra" / "preflight" / cid, *job_destinations(root, cid)], JOB_FILES
        else:
            base, files = shards_root(root), ()
            refuse_protected(base, f"the shard root of {cid}")
            own = [shard_root(root, cid, index, condition.shards) for index in range(condition.shards)]
            for shard in own:
                check_new_condition_id(shard, cid)
            ways = [base / cid / "preflight", *own]
        for way in ways:
            refuse_linked_way(base, way, what=f"a directory {cid}'s jobs write", below=True, roots=True, own=own)
        for name in files:
            if (base / name).is_symlink():
                raise LaunchError(f"{base / name}, a file {cid}'s job writes in its root, is a symlink: replace it "
                                  "with a file of the root's own")
        if condition.batch_root is not None:
            check_batch_root(condition.batch_root, condition, conditions,
                             what=f"the batch_root of {cid}'s batch judge")


def _refuse_nested_roots(conditions: Sequence[Condition]) -> None:
    places = [(condition, root, spelled) for condition in conditions
              for root in [artifact_root() / condition.campaign_root]
              for own in (root, shards_root(root)) for spelled in dict.fromkeys((own, Path(os.path.realpath(own))))]
    for condition, root, spelled in places:
        for other, other_root, place in places:
            if other_root != root and spelled.is_relative_to(place):
                raise LaunchError(f"{spelled}, the campaign root or shard root of {condition.condition_id}, is or lies "
                                  f"in {place}, the campaign root or shard root of {other.condition_id} in this "
                                  "launch, which only its own jobs write: give each campaign a root apart from the "
                                  "others'")


def check_batch_root(batch_root: str, condition: Condition, conditions: Sequence[Condition], *, what: str) -> None:
    """Refuse a batch judge's `batch_root` that is relative, through a symlink, on a pinned root's path, inside another
    campaign root, or in or above a root of the launch; nothing is written."""
    from ruleforge.evaluate.api import DEFAULT_BATCH_DIRNAME, refuse_batch_root, refuse_pinned

    if not os.path.isabs(batch_root):
        raise LaunchError(f"{what} {batch_root!r} is relative: its job's transport takes it as given, from the job's "
                          "working directory (the checkout's top, or Hydra's run directory under hydra.job.chdir), "
                          "a '~' unexpanded; name it absolute")
    base = artifact_root()
    home = base / condition.campaign_root
    refuse_pinned(batch_root, what)  # a spelling that names two places is refused before any check takes one
    written = Path(os.path.abspath(batch_root))
    if written.is_symlink():
        raise LaunchError(f"{what} {written} is a symlink: a write would follow it to {os.path.realpath(written)}; "
                          "replace it with a directory of its own")
    refuse_batch_root(home, batch_root, what=what)
    if written.is_relative_to(base):
        refuse_linked_way(base, written, what=what)
    places = {written, Path(os.path.realpath(written))}
    homes = {home, Path(os.path.realpath(home))}
    for other in conditions:
        root = base / other.campaign_root
        for own in (root, shards_root(root)):
            for spelled in {own, Path(os.path.realpath(own))}:
                # the one exemption: the condition's own root's batch/
                batch = spelled / DEFAULT_BATCH_DIRNAME if spelled in homes else None
                for place in places:
                    if spelled.is_relative_to(place) or (place.is_relative_to(spelled) and not (
                            batch is not None and place.is_relative_to(batch))):
                        raise LaunchError(f"{what} {batch_root} is, lies in or holds {spelled}, "
                                          f"{other.condition_id}'s campaign root or shard root in this launch, which "
                                          "only its own jobs write (its batch judge's batch root may be its own "
                                          f"root's {DEFAULT_BATCH_DIRNAME}/): choose a batch root outside it")


#: Files a campaign job writes in the root it runs in, besides the directories of `job_destinations`.
JOB_FILES = ("scenario_manifest.json", "sampling_manifest.json", "completion.json", ".ruleforge-configured")


def job_destinations(root: Path, condition_id: str) -> list[Path]:
    """Return the directories a campaign job of `condition_id` writes in `root`."""
    from ruleforge.core.provenance import SNAPSHOT_DIRNAME

    return [root / "hydra" / condition_id, root / condition_id, root / SNAPSHOT_DIRNAME]


def local_root(recorded: str, base: Path) -> Path:
    """Return a recorded campaign root as a path here, resolved."""
    path = Path(recorded)
    return (path if path.is_absolute() else base / path).resolve()


def shared_jobs(launch: Mapping[str, Any], condition: Mapping[str, Any], *,
                is_root: Optional[Callable[[str], bool]] = None) -> dict[str, str]:
    """Return the jobs of a recorded `condition` that another `launch` planned too, as {this condition's job name: that
    launch's}."""
    planned = {job["job_id"] for job in launch["jobs"]}
    shards, layout = condition["shards"], (condition["condition_id"], condition["shards"])
    same_root = is_root or condition["campaign_root"].__eq__
    return {job_name(condition["job_id"], index, shards): job_name(other["job_id"], index, shards)
            for other in (launch.get("campaign") or {}).get("conditions", ())
            if (other["condition_id"], other["shards"]) == layout and same_root(other["campaign_root"])
            for index in range(shards or 1) if job_name(other["job_id"], index, shards) in planned}


def relaunched_jobs(launch: Mapping[str, Any], condition: Mapping[str, Any]) -> set[str]:
    """Return the jobs of a recorded `condition` that another `launch` launched again."""
    taken = {run["job_id"] for run in launch["runs"] if not run.get("refused")}
    return {job for job, theirs in shared_jobs(launch, condition).items() if theirs in taken}


def _check_not_set(arguments: Sequence[str], owned: Sequence[str], option: str) -> None:
    for argument in hydra_arguments(arguments)[1]:
        if any(reaches(key, owned) for key in override_keys(argument)):
            raise LaunchError(f"{argument!r} sets what {option} sets: give it once, as {option}")


def _knobs(options: CampaignOptions, *, exists: Callable[[Place], bool]) -> list[str]:
    overrides: list[str] = []
    if options.campaign_id is not None:
        overrides.append(f"campaign_id={check_name(options.campaign_id, what='--campaign-id')}")
    root = None
    if options.campaign_root is not None:
        if options.blocks:
            raise LaunchError("--campaign-root with --blocks: each block has a root of its own (--root-template)")
        root = place(options.campaign_root, what="--campaign-root")
        if not options.new_root:
            check_campaign_root(str(root.local), holds_campaign=exists(root))
    elif options.campaign_id is not None and not options.blocks:
        root = place(f"sampling/{options.campaign_id}", what="the campaign root")
    if root is not None:
        overrides.append(f"campaign_root={root.named}")
    if options.population is not None and options.blocks:
        raise LaunchError("--population with --blocks: each block is its own population")
    population = None if options.population is None else place(options.population, what="--population")
    digest = check_population(None if population is None else population.local, options.expect_population_sha256)
    if population is not None:
        overrides += [f"scenario_set_path={population.named}", f"approved_scenario_sha256={digest}"]
    check_packing(options.pack_draws, options.pack_max_prompts, options.pack_max_draws)
    if options.pack_draws:
        overrides.append("execution.pack_draws=true")
        for key in ("pack_max_prompts", "pack_max_draws"):
            value = getattr(options, key)
            overrides += [] if value is None else [f"execution.{key}={value}"]
    return overrides


def block_files(values: Sequence[str]) -> list[Place]:
    """Return `--blocks` as files: each file, or a directory's `*.jsonl` blocks sorted, named once."""
    found: list[Place] = []
    for value in values:
        where = place(value, what="--blocks")
        if where.local.is_dir():
            files = sorted(where.local.glob("*.jsonl"))
            if not files:
                raise LaunchError(f"--blocks {value}: no *.jsonl block in it")
            found += [place(str(path), what="--blocks") for path in files]
        elif where.local.is_file():
            found.append(where)
        else:
            raise LaunchError(f"--blocks {value}: no such file or directory")
    stems = [block.local.stem for block in found]
    repeated = sorted({stem for stem in stems if stems.count(stem) > 1})
    if repeated:
        raise LaunchError(f"--blocks names {', '.join(repeated)} twice: each block's root is named by its stem")
    return found


def _blocks(options: CampaignOptions) -> list[tuple[str, list[str], tuple[str, ...]]]:
    if not options.blocks:
        if options.root_template is not None:
            raise LaunchError("--root-template names each block's root: give --blocks")
        return [("", [], ())]
    template = options.root_template
    if template is None or f"{{{BLOCK}}}" not in template:
        raise LaunchError(f"--blocks needs --root-template, a root holding {{{BLOCK}}} (each block's stem)")
    entries = []
    for block in block_files(options.blocks):
        root = place(fill(template, {BLOCK: block.local.stem}), what="--root-template")
        entries.append((block.local.stem, [f"campaign_root={root.named}",
                                           f"scenario_set_path={block.named}",
                                           f"approved_scenario_sha256={sha256_file(block.local)}"],
                        ("--blocks", block.recorded)))
    return entries


def _points(axes: Sequence[str]) -> list[tuple[str, ...]]:
    parsed = [parse_axis(axis) for axis in axes]
    return [tuple(item for (key, _), value in zip(parsed, point) for item in ("--axis", f"{key}={value}"))
            for point in itertools.product(*(values for _, values in parsed))]


def plan(
    repo: Path, template: JobSpec, arguments: Sequence[str], axes: Sequence[str], options: CampaignOptions, *,
    exists: Callable[[Place], bool], allocates_gpus: bool = True,
) -> list[Condition]:
    """Return the campaign's conditions, the axes' points times the blocks cut into shards; `--only` needs one
    condition."""
    knobs = _knobs(options, exists=exists)
    blocks = _blocks(options)
    base = replace(template, kind="campaign", argv=(*PROGRAM, *arguments), workdir=WORKDIR)
    conditions: list[Condition] = []
    for member, point in zip(members(base, axes), _points(axes), strict=True):
        given = member.argv[len(PROGRAM):]
        check_overrides(given)
        if options.population is not None or options.blocks:
            _check_not_set(given, POPULATION_KEYS, "--population" if options.population else "--blocks")
        if options.pack_draws:
            _check_not_set(given, PACKING_KEYS, "--pack-draws")
        for name, overrides, block in blocks:
            job_id = check_name(f"{member.job_id}-{name}" if name else member.job_id, what="a campaign job's name")
            spec = replace(member, job_id=job_id, argv=(*member.argv, *knobs, *overrides))
            resolved = resolve_condition(repo, spec.argv[len(PROGRAM):], env=spec.env)
            if not Path(resolved["campaign_root"]).is_absolute():
                raise LaunchError(f"{job_id}'s campaign_root {resolved['campaign_root']!r} is relative: name it "
                                  "absolute (${oc.env:RULEFORGE_ARTIFACT_ROOT}/...), or give --campaign-root")
            if allocates_gpus and resolved["judge_model_type"] == LOCAL_JUDGE and not spec.resources.gpus:
                raise LaunchError(f"{job_id}'s judge is a local model, served by vLLM on the job's GPUs: give "
                                  "--gpu-count (a job with none fails when its engine starts)")
            root = place(resolved["campaign_root"], what=f"{job_id}'s campaign_root")
            condition_id = resolved["condition_id"]
            if relative_subpath(condition_id, what=f"{job_id}'s condition_id") != condition_id:
                raise LaunchError(f"{job_id}'s condition_id {condition_id!r} is not a plain relative path")
            conditions.append(Condition(spec, resolved["campaign_id"], root.recorded, condition_id,
                                        selection=(*point, *block), named_root=root.named,
                                        batch_root=resolved["batch_root"]))
    if options.only is not None and len(conditions) > 1:
        raise LaunchError(f"--only relaunches shards of one condition, and this campaign has {len(conditions)}: "
                          "the same shards of every condition would launch again, running ones among them. "
                          "Narrow --axis (or --blocks) to the condition whose shards to relaunch and give its "
                          "--shards (a launch that stopped part-way, or a wait that failed, prints these options)")
    return _cut(conditions, options.shards)


def _cut(conditions: list[Condition], shards: Optional[int]) -> list[Condition]:
    keys = [(condition.campaign_root, condition.condition_id) for condition in conditions]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise LaunchError(f"two jobs of the campaign would write condition {repeated[0][1]} of {repeated[0][0]}")
    return [replace(condition, shards=shards) for condition in conditions]
