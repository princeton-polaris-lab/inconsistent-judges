from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from collections import Counter
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass, field, replace
from itertools import takewhile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import yaml

from ruleforge.core.artifacts import umask
from ruleforge.core.constants import FULL_RULESET_KEY, PUBLISHED_COPY_FNAME, PUBLISHED_LEDGER_FNAME
from ruleforge.core.digests import sha256_file
from ruleforge.core.errors import RuleForgeError
from ruleforge.core.optional import require
from ruleforge.core.paths import ARTIFACT_SCHEME, second_place
from ruleforge.core.record_names import PER_SCENARIO_FNAME, public_segment
from ruleforge.env import env_value, package_path

__all__ = ["CompactRule", "Entry", "HubError", "Index", "Omit", "Publication", "fetch", "load_index", "load_inventory",
           "main_pack", "members_digest", "pack", "pack_models", "published_record", "read_sums", "reference", "refuse_campaign",
           "refuse_in_destinations", "resolve_commit", "run_record", "temporary_parent", "verify"]

INDEX, SUMS, CARD = "index.json", "SHA256SUMS", "README.md"
FORMAT = 1
KINDS = ("root", "selection", "input", "adapter")
FILE_KINDS = ("selection", "input")
FILE, ARCHIVE = "file", "tar.zst"
# What a Hub snapshot holds beside the files a repository lists.
SNAPSHOT_EXTRAS = (".gitattributes", ".cache")
# Per-draw text files the compact tier leaves out.
DRAW_TEXT = ("judge_outputs.jsonl", "raw_outputs.jsonl", "evaluations.jsonl", PER_SCENARIO_FNAME,
             "attempts.jsonl")
# Run records the compact tier leaves out of every root: Hydra's run directories and the code snapshots.
RUN_RECORDS = ("hydra/**", "code_snapshots/**")
# Fields the published copy names without the account's paths or the job's host: (file name, field path, rule), `[]`
# stepping into each item of a list. None is in a stack's identity.
PUBLISHED_FIELDS = (
    ("runs.jsonl", "runtime.engine_args.model", "name"),
    ("runs.jsonl", "runtime.hostname", "host"),
    ("draw_provenance.json", "runtime.engine_args.model", "name"),
    ("draw_provenance.json", "runtime.hostname", "host"),
    ("draw_provenance.json", "model_weights.path", "location"),
    ("draw_provenance.json", "model_weights.manifest_sha256", "digest"),
    ("condition_stack.json", "runtime.engine_args.model", "name"),
    ("condition_stack.json", "runtime.hostname", "host"),
    ("rebase.json", "chain[].parent.root", "location"),
    *(("grpo_data_manifest.json", path, "location") for path in (
        "anchor_condition", "anchors[].path", "by_anchor_set[].path", "campaigns.bank[].root",
        "campaigns.calibration.root", "campaigns.confirm[].root", "campaigns.full_set[].root", "campaigns.top_up[].root",
        "campaigns.yes_side[].root", "protected_extra[].path", "required_protected[].path",
        "sampling_weights.rest_filter.tables[].path", "scenarios.path")),
    ("grpo_data_manifest.json", "rules.path", "repo"),
    ("grpo_data_manifest.json", "tokenizer.model", "name"),
    ("grpo_data_manifest.json", "tokenizer.campaign_model_dir", "name"),
    ("merge_manifest.json", "base.path", "name"),
    ("merge_manifest.json", "adapter.base_model", "name"),
    ("merge_manifest.json", "adapter.path", "location"),
    ("decode_report.json", "inputs.hf_merged.model_path", "location"),
    ("prefill_report.json", "inputs.hf_merged.model_path", "location"),
    ("adapter_config.json", "base_model_name_or_path", "hub"),
)
HOST_PLACEHOLDER = "host"


@dataclass(frozen=True)
class Projection:
    """What a campaign root's published copy keeps of a file: the rows whose ``row`` fields hold those values (every
    row when there are none) and, of each, the ``fields``: dotted paths, ``*`` standing for every key at its level and
    ``key[]`` for each object of a list. A named field's value is kept as it is when it is a scalar or a list of
    scalars, or when ``whole`` names it (a reader checks all of it); ``without`` names fields a ``*`` does not cover. A
    field no path names is left out, at every level."""

    fields: tuple[str, ...]
    row: tuple[tuple[str, str], ...] = ()
    whole: tuple[str, ...] = ()
    without: tuple[str, ...] = ()


# The engine arguments the stack checks leave out (ruleforge.evaluate.stack.ENGINE_PLACEMENT_ARGS).
ENGINE_PLACEMENT_ARGS = ("model", "tensor_parallel_size", "disable_log_stats", "max_num_seqs", "gpu_memory_utilization",
                         "max_num_batched_tokens")
# What the stack checks read of a recorded runtime: the package versions, the precision settings and every binding
# engine argument, whole (ruleforge.evaluate.stack.judgment_signature, runtime_changes).
_RUNTIME = (*(f"runtime.{name}_version" for name in ("vllm", "torch", "transformers", "tokenizers", "triton")),
            *(f"runtime.precision_env.{name}" for name in ("VLLM_FLOAT32_MATMUL_PRECISION", "TRITON_F32_DEFAULT",
                                                         "VLLM_BATCH_INVARIANT")),
            "runtime.engine_args.*")
_RUNTIME_WHOLE = ("runtime.engine_args.*",)
_RUNTIME_WITHOUT = tuple(f"runtime.engine_args.{name}" for name in ENGINE_PLACEMENT_ARGS)


@dataclass(frozen=True)
class Profile:
    """What one kind of campaign root's published copy keeps, by that root's own readers: per file name, the rows and
    fields (:class:`Projection`; a kind it does not name keeps all of them), and the parts of each draw ledger's
    published form (ruleforge.evaluate.ledger.FORM_PARTS)."""

    projections: Mapping[str, Projection]
    ledger: tuple[str, ...] = ()


def _outcomes(*fields: str) -> Projection:
    return Projection(("scenario_id", "decoder_draw_id", *fields))


# Of a draw's completion record, what every outcome reader holds the draw to (inference.read_draw_outcomes) ...
_GATE = ("state", "decoder_draw_id", "scenario_manifest_sha256", "condition_fingerprint")
# ... what leniency and RuleForge hold its materialization to, and what a local judge's stack check holds its run
# log to.
_MATERIALIZED, _RUN_LOG = ("materialized", "campaign_id", "condition_id"), ("run_log_rows",)
# A local judge's draws: what the stack checks read (model_type decides whether an unlogged stack change is refused).
_LOCAL = {"draw_provenance.json": Projection(("model_type", *_RUNTIME), whole=_RUNTIME_WHOLE, without=_RUNTIME_WITHOUT),
          "runs.jsonl": Projection(("stack_change", *_RUNTIME), whole=("stack_change", *_RUNTIME_WHOLE),
                                   without=_RUNTIME_WITHOUT)}
_LEDGER_STACK = ("has_attempts", "judged_on")
# The prompt-ablation roots' readers: the rates, leniency's forced arms and the hierarchy's flat arms.
_PROMPT_OUTCOMES = _outcomes("contradiction", "full_set_judgment", "rule_judgments.*", "in_context_rule_judgments.*",
                             "semantic_attempt_id")
_HIERARCHY_OUTCOMES = _outcomes("contradiction", "full_set_judgment", "rule_judgments.*")
_LENIENCY_LEDGER = ("accepted_attempt", "judgment", "rule_judgments")
_RAW = ("scenario_id", "semantic_attempt_id", "request_block_id", "component_id", "raw_response")
# Every root's conditions: each field their fingerprint covers (the outcome readers recompute it), their id and
# fingerprint; not the envelope's schema version, which nothing reads (identity.recorded_fingerprint).
_CONDITION = Projection(("*",), whole=("*",), without=("schema_version",))


def _manifest(*fields: str) -> Projection:
    """What a kind of root's readers read of its campaign's sampling manifest, beside its id, mode, schema and cap."""
    return Projection(("campaign_id", "campaign_mode", "schema_version", "max_semantic_attempts", *fields))


_REQUEST_DESIGN = ("generation_seed_base", "retry_scope", "scenarios_per_request_block")
_PROMPT_MANIFEST = _manifest(*_REQUEST_DESIGN, "expected_decoder_draws", "scenario_manifest_sha256")
# What the training readout reads of a rebase: its parent's identity, the population it moved to, the scenarios it
# swapped, and of each link of its chain the parent's identity, the target and the swap (evaluate.chain).
_HOP_PARENT = ("campaign_id", "scenario_manifest_sha256", "scenario_source_sha256", "sampling_manifest_sha256")
_REBASE = Projection(("removed_ids", "added_ids", "parent.campaign_id", "parent.scenario_manifest_sha256",
                      "parent.sampling_manifest_sha256", "target.scenario_manifest_sha256",
                      *(f"chain[].parent.{name}" for name in _HOP_PARENT), "chain[].target", "chain[].removed_ids",
                      "chain[].added_ids"), whole=("chain[].target",))


def _training(raw: Sequence[str]) -> Profile:
    return Profile({
        "bundle_outcomes.jsonl": _outcomes("contradiction", "full_set_judgment", "rule_judgments.*", "full_set_phi",
                                           "max_rule_phi"),
        "completion.json": Projection((*_GATE, *_RUN_LOG)),
        # the merged checkpoint is compared whole; the engine at one tensor-parallel size and attention backend
        "draw_provenance.json": Projection(("model_type", "model_weights", "attention_backend", "tp_size", *_RUNTIME),
                                           whole=("model_weights", *_RUNTIME_WHOLE), without=_RUNTIME_WITHOUT),
        "runs.jsonl": Projection(("code_revision", "code_dirty", "code_state_sha256", "stack_change", *_RUNTIME),
                                 whole=("stack_change", *_RUNTIME_WHOLE), without=_RUNTIME_WITHOUT),
        "evaluations.jsonl": Projection(("scenario_id", "scenario.source")),
        "raw_outputs.jsonl": Projection(tuple(raw), row=(("component_id", FULL_RULESET_KEY),)),
        "condition.json": _CONDITION,
        # the family's design check compares every key of its roots' manifests but the campaign's run fields
        # (family_readout.check_design); of those, the readout reads the code state, never the conditions or the time
        "sampling_manifest.json": Projection(("*",), whole=("*",), without=("conditions", "created_at")),
        "rebase.json": _REBASE,
    }, ledger=("call", "judgment", "raw_output_sha256", *_LEDGER_STACK))


# Each kind of campaign root by its readers; the inventory names each root's (``profile``, ``profiles``).
PROFILES = {
    "ruleset-size": Profile({"bundle_outcomes.jsonl": _outcomes("full_set_judgment", "rule_judgments.*",
                                                               "ruleset_rule_ids"),
                             "completion.json": Projection(_GATE), "condition.json": _CONDITION}),
    "prompt-ablation-local": Profile({"bundle_outcomes.jsonl": _PROMPT_OUTCOMES,
                                      "completion.json": Projection((*_GATE, *_MATERIALIZED, *_RUN_LOG)), **_LOCAL,
                                      "condition.json": _CONDITION, "sampling_manifest.json": _PROMPT_MANIFEST},
                                     ledger=(*_LENIENCY_LEDGER, *_LEDGER_STACK)),
    "prompt-ablation-api": Profile({"bundle_outcomes.jsonl": _PROMPT_OUTCOMES,
                                    "completion.json": Projection((*_GATE, *_MATERIALIZED)),
                                    "request_blocks.jsonl": Projection(("recorded_at",)),
                                    "condition.json": _CONDITION, "sampling_manifest.json": _PROMPT_MANIFEST},
                                   ledger=(*_LENIENCY_LEDGER, "started_at")),
    "hierarchy-local": Profile({"bundle_outcomes.jsonl": _HIERARCHY_OUTCOMES,
                                "completion.json": Projection((*_GATE, *_RUN_LOG)), **_LOCAL,
                                "condition.json": _CONDITION, "sampling_manifest.json": _manifest(*_REQUEST_DESIGN)},
                               ledger=_LEDGER_STACK),
    # an API judge's hierarchy arms keep no ledger: their request dates are read from their request blocks
    "hierarchy-api": Profile({"bundle_outcomes.jsonl": _HIERARCHY_OUTCOMES, "completion.json": Projection(_GATE),
                              "request_blocks.jsonl": Projection(("recorded_at",)), "condition.json": _CONDITION,
                              "sampling_manifest.json": _manifest(*_REQUEST_DESIGN)}),
    "ruleforge": Profile({"bundle_outcomes.jsonl": _outcomes("contradiction"),
                          "completion.json": Projection((*_GATE, *_MATERIALIZED, "accepted")),
                          "condition.json": _CONDITION,
                          "sampling_manifest.json": _manifest("expected_decoder_draws", "expected_scenarios",
                                                              "forced_json_requirement", "scenario_manifest_sha256",
                                                              "temperature", "truncation")}),
    # Gemma-4-31B-it's same-call reading takes the completion, Qwen3-8B's (strict thinking) the raw response alone
    "training-gemma": _training((*_RAW, "completion")),
    "training-qwen": _training(_RAW),
}
# Every file kind a campaign root's published copy holds, as globs over its paths: the packer refuses any other file it
# would keep, naming it (a new kind needs a decision: published here, or left out by an omit glob).
PUBLISHED_KINDS = (
    "**/condition.json", "scenario_manifest.json", "sampling_manifest.json", "figure2_design.json", "label_replay.json",
    "rebase.json", "aggregate/estimates.json", "aggregate/figure2_estimates.json",
    *(f"**/draw-*/{name}" for name in ("bundle_outcomes.jsonl", "completion.json", "attempts.jsonl", "raw_outputs.jsonl",
                                       "evaluations.jsonl", "request_blocks.jsonl", "runs.jsonl",
                                       "draw_provenance.json")))
# A campaign root's attempt ledger is published as its strict replay's published form, under a name of its own
# (ruleforge.evaluate.ledger.published_form): each accepted scenario's full-ruleset call, and of every attempt the
# first and last start time and the stacks the rows name.
LEDGER_PUBLISHED_AS = {"attempts.jsonl": PUBLISHED_LEDGER_FNAME}
# A campaign root's raw outputs keep only the rows of their draw's accepted full-ruleset calls, by its ledger's replay.
RAW_OUTPUTS = "raw_outputs.jsonl"
# Fields whose values are scenario, judge or model text, by file name: they may quote any path or name, so the check
# for local paths and names (:func:`refuse_private`) reads every field but these. A field whose value has a path's
# shape and is none (an endpoint such as ``/v1``, a route) is exempt here too, by its field, never by its value.
# A path segment ``*`` stands for any key (the RuleForge run's rewards are keyed by seed).
_SCENARIO = ("prompt", "response")
TEXT_FIELDS = {
    "scenarios.jsonl": _SCENARIO, "frozen_500_r3.jsonl": _SCENARIO, "grpo_v2_dev.jsonl": _SCENARIO,
    "grpo_rows.jsonl": ("prompt[].content",), "raw_outputs.jsonl": ("raw_response", "completion"),
    # the RuleForge run's records of the scenarios it judged
    "evaluations.jsonl": ("scenario.prompt", "scenario.response"), "entropy.jsonl": ("scenario.prompt", "scenario.response"),
    "posterior_sample.jsonl": _SCENARIO, "optimal_scenario_set.jsonl": _SCENARIO,
    "accumulated_scenario_set.jsonl": _SCENARIO,
    "reward_all.jsonl": ("*.per_scenario_rewards[].scenario.prompt", "*.per_scenario_rewards[].scenario.response",
                         "scenario_list[].scenario.prompt", "scenario_list[].scenario.response"),
    "reward_optimal.jsonl": ("reward.per_scenario_rewards[].scenario.prompt",
                             "reward.per_scenario_rewards[].scenario.response", "scenario_list[].scenario.prompt",
                             "scenario_list[].scenario.response"),
}
# Where a local path begins: a home directory (the packing account's own, wherever it lies, is added).
LOCAL_ROOTS = ("/home/", "/Users/")
# An absolute path in a value, whatever its mount: a directory under the root, a home-relative path or a drive
# (``/srv/x``, ``/tmp/x``, ``~/x``, ``~user/x``, ``C:\x``); a URL's or a relative path's slashes are not one.
ABSOLUTE_PATH = re.compile(r"(?<![\w.:/~-])(?:/[\w.@+-]+/|~[\w.-]*/|[A-Za-z]:[\\/])")
_ABSOLUTE_PATH_BYTES = re.compile(ABSOLUTE_PATH.pattern.encode("utf-8"))
# A whole value that is an absolute path of one component or none: the root, a directory under it, a home or a drive
# (``/``, ``/srv``, ``/srv/``, ``~``, ``~user``, ``C:``, ``C:\``).
WHOLE_ABSOLUTE_PATH = re.compile(r"\s*(?:/[\w.@+-]*/?|~[\w.-]*/?|[A-Za-z]:[\\/]?)\s*")
_WHOLE_ABSOLUTE_PATH_BYTES = re.compile(rb'"\s*(?:/[\w.@+-]*/?|~[\w.-]*/?|[A-Za-z]:(?:\\\\|/)?)\s*"')
# An artifact root's top-level directories: a recorded path is published from the first of them on.
ARTIFACT_DIRS = ("sampling", "selection", "anchors", "populations", "verdict_variance", "grpo_data", "batch", "training")
# A caller's own check of a place about to be written (the place, what it is); raises to refuse it.
Guard = Callable[[Path, str], None]
DOWNLOAD = "the directory the download is held in"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SUMS_LINE = re.compile(r"([0-9a-f]{64}) [ *](.+)")
_CHUNK = 1 << 20


class HubError(RuntimeError, RuleForgeError):
    """A repository or a destination that cannot be used as it is: nothing is placed for it."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise HubError(message)


def safe_path(path: Any, what: str) -> str:
    """Return ``path`` if it stays where it is put: relative POSIX, with no ``\\``, NUL, empty, ``.`` or ``..``
    component."""
    _require(isinstance(path, str) and path != "", f"{what}: {path!r} is not a path")
    _require(not path.startswith("/") and "\\" not in path and "\0" not in path
             and not {"", ".", ".."} & set(path.split("/")), f"{what}: {path!r} is not a relative path inside it")
    return path


def members_digest(members: Mapping[str, str]) -> str:
    """Return one digest over an archive's members by path and SHA-256, whatever the compression or manifest order."""
    digest = hashlib.sha256()
    for path in sorted(members):
        digest.update(f"{path}\x1e{members[path]}\x1d".encode("utf-8"))
    return digest.hexdigest()


def read_sums(text: str, what: str = SUMS) -> dict[str, str]:
    """Return a ``sha256sum`` listing as ``{path: sha256}``, refusing a malformed line, unsafe path or repeated path."""
    sums: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = _SUMS_LINE.fullmatch(line)
        _require(match is not None, f"{what}:{number}: not a sha256sum line")
        path = match.group(2).removeprefix("./")
        safe_path(path, f"{what}:{number}")
        _require(path not in sums, f"{what}: {path} is listed twice")
        sums[path] = match.group(1)
    return sums


def write_sums(sums: Mapping[str, str]) -> str:
    return "".join(f"{sums[path]}  {path}\n" for path in sorted(sums))


def overlapping(dests: Iterable[str]) -> list[tuple[str, str]]:
    """Return every pair of destinations where one is the other or lies inside it."""
    counts = Counter(dests)
    pairs = {(dest, dest) for dest, count in counts.items() if count > 1}
    pairs |= {(parent, dest) for dest in counts for parent in map(str, PurePosixPath(dest).parents) if parent in counts}
    return sorted(pairs)


@dataclass(frozen=True)
class Entry:
    path: str
    kind: str
    group: str
    dest: str
    sha256: str
    bytes: int
    format: str = FILE
    prefix: str | None = None
    members: str | None = None
    members_sha256: str | None = None
    members_digest: str | None = None
    root: str | None = None
    omit: tuple[str, ...] | None = None

    @property
    def archive(self) -> bool:
        return self.format == ARCHIVE

    def to_json(self) -> dict[str, Any]:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in self.__dict__.items()
                if value is not None}


@dataclass(frozen=True)
class Index:
    entries: tuple[Entry, ...]
    documents: tuple[str, ...]
    compact_rule: str

    def listed(self) -> set[str]:
        """Every file of the repository the index accounts for, itself included."""
        return ({INDEX, *self.documents} | {entry.path for entry in self.entries}
                | {entry.members for entry in self.entries if entry.members})

    def select(self, only: Sequence[str]) -> list[Entry]:
        """Return every entry, or those of each named experiment and root."""
        if not only:
            return list(self.entries)
        groups = {entry.group for entry in self.entries}
        roots = {entry.root: entry.group for entry in self.entries if entry.root}
        unknown = sorted(set(only) - groups - set(roots))
        _require(not unknown, f"--only {', '.join(unknown)}: neither an experiment ({', '.join(sorted(groups))}) "
                              "nor a root of this repository")
        chosen = set(only)
        return [entry for entry in self.entries
                if entry.group in chosen or entry.root in chosen
                or (entry.kind != "root" and entry.group in {roots[name] for name in chosen if name in roots})]


_ENTRY_KEYS = {"path", "kind", "group", "dest", "sha256", "bytes", "format", "prefix", "members", "members_sha256",
               "members_digest", "root", "omit"}


def _omit_list(value: Any, what: str) -> tuple[str, ...]:
    _require(isinstance(value, list) and bool(value), f"{what}: omit is a list of globs")
    for pattern in value:
        safe_path(pattern, f"{what} omit")
    _require(len(set(value)) == len(value), f"{what}: omit names a glob twice")
    return tuple(value)


def _entry(raw: Any, number: int) -> Entry:
    where = f"{INDEX} files[{number}]"
    _require(isinstance(raw, dict), f"{where}: not an object")
    _require(not set(raw) - _ENTRY_KEYS, f"{where}: unknown keys {sorted(set(raw) - _ENTRY_KEYS)}")
    missing = sorted({"path", "kind", "group", "dest", "sha256", "bytes"} - set(raw))
    _require(not missing, f"{where}: missing {missing}")
    entry = Entry(**{"format": FILE, **raw})
    where = f"{INDEX} {entry.path!r}"
    safe_path(entry.path, where)
    safe_path(entry.dest, f"{where} dest")
    _require(entry.kind in KINDS, f"{where}: kind {entry.kind!r} is not one of {', '.join(KINDS)}")
    _require(isinstance(entry.group, str) and bool(entry.group), f"{where}: no group")
    _require(isinstance(entry.sha256, str) and bool(_SHA256.fullmatch(entry.sha256)), f"{where}: no sha256")
    _require(type(entry.bytes) is int and entry.bytes >= 0, f"{where}: bytes must be a count")
    _require(entry.format in (FILE, ARCHIVE), f"{where}: format {entry.format!r} is not {FILE} or {ARCHIVE}")
    archive_keys = (entry.prefix, entry.members, entry.members_sha256, entry.members_digest)
    if entry.archive:
        _require(None not in archive_keys, f"{where}: an archive names its prefix, members, members_sha256 and "
                                           "members_digest")
        safe_path(entry.prefix, f"{where} prefix")
        safe_path(entry.members, f"{where} members")
        _require(bool(_SHA256.fullmatch(entry.members_sha256)) and bool(_SHA256.fullmatch(entry.members_digest)),
                 f"{where}: members_sha256 and members_digest are SHA-256 digests")
    else:
        _require(archive_keys == (None,) * 4 and entry.omit is None, f"{where}: a plain file has no prefix, members "
                                                                     "or omit list")
    if entry.omit is not None:
        entry = replace(entry, omit=_omit_list(entry.omit, where))
    if entry.kind == "root":
        _require(entry.archive and entry.root is not None and entry.dest == f"sampling/{entry.root}"
                 and entry.prefix == entry.dest, f"{where}: a root is an archive of sampling/<root>/, placed there")
    else:
        _require(entry.root is None, f"{where}: only a root names a root")
    return entry


def load_index(data: Any, sums: Mapping[str, str]) -> Index:
    """The index, checked against itself and the repository's ``SHA256SUMS``."""
    _require(isinstance(data, dict) and data.get("format") == FORMAT, f"{INDEX}: not an index of format {FORMAT}")
    _require(isinstance(data.get("files"), list) and isinstance(data.get("documents", []), list),
             f"{INDEX}: files and documents are lists")
    entries = tuple(_entry(raw, number) for number, raw in enumerate(data["files"]))
    documents = tuple(safe_path(path, f"{INDEX} documents") for path in data.get("documents", []))
    index = Index(entries, documents, str(data.get("compact_rule", "")))
    paths = [entry.path for entry in entries] + [entry.members for entry in entries if entry.members]
    repeated = sorted({path for path in paths if paths.count(path) > 1} | set(paths) & {INDEX, SUMS, *documents})
    _require(not repeated, f"{INDEX}: {', '.join(repeated)} listed more than once")
    clashes = overlapping(entry.dest for entry in entries)
    _require(not clashes, f"{INDEX}: destinations {clashes[:3]} overlap")
    _require(SUMS not in sums, f"{SUMS} lists itself")
    listed = index.listed()
    _require(set(sums) == listed, f"{SUMS} and {INDEX} disagree: only in {SUMS} {sorted(set(sums) - listed)[:5]}, "
                                  f"only in {INDEX} {sorted(listed - set(sums))[:5]}")
    for entry in entries:
        _require(sums[entry.path] == entry.sha256, f"{entry.path}: {INDEX} and {SUMS} give two digests")
        _require(entry.members is None or sums[entry.members] == entry.members_sha256,
                 f"{entry.members}: {INDEX} and {SUMS} give two digests")
    return index


def check_root(root: Path) -> Path:
    """``root`` as an absolute path to an existing directory."""
    _require(root.is_dir(), f"{root} is not a directory")
    return Path(os.path.abspath(root))


def check_destination(root: Path, dest: str) -> str:
    """Return ``absent`` or ``present`` for ``dest`` under ``root``; refuses a symlink or non-directory among its
    ancestors."""
    path = root
    parts = PurePosixPath(dest).parts
    for part in parts[:-1]:
        path = path / part
        if not os.path.lexists(path):
            return "absent"
        _require(not path.is_symlink() and path.is_dir(), f"{dest}: {path.relative_to(root)} is a symlink or not a "
                                                          "directory")
    return "present" if os.path.lexists(root / dest) else "absent"


def refuse_campaign(place: Path | str, what: str) -> tuple[Path, Path]:
    """Return ``(written, resolved)`` for ``place``, writing nothing and refusing a place that is, or lies inside, a
    campaign root, or whose spelling names two places."""
    from ruleforge.evaluate.api import CampaignError, _enclosing_root, _is_campaign_root, _spellings

    try:
        written, resolved = _spellings(place, what)
    except CampaignError as error:
        raise HubError(str(error)) from error
    inside = (next((path for path in (resolved, written) if _is_campaign_root(path)), None)
              or _enclosing_root(resolved, written))
    if inside is not None:
        raise HubError(f"{what} {written} is {'' if inside in (written, resolved) else 'in '}the campaign root "
                       f"{inside}, which nothing but its own campaign writes: choose a place outside it")
    return written, resolved


def refuse_in_destinations(place: Path | str, root: Path, entries: Iterable[Entry], what: str) -> None:
    """Refuse a download directory that is, or lies inside, the destination of one of ``entries``; nothing is
    written."""
    written = Path(os.path.abspath(place))
    resolved = Path(os.path.realpath(written))
    for entry in entries:
        dest = root / entry.dest
        for top in dict.fromkeys((dest, Path(os.path.realpath(dest)))):
            _require(not (written.is_relative_to(top) or resolved.is_relative_to(top)),
                     f"{what} {written} is {'' if top in (written, resolved) else 'inside '}{dest}, where "
                     f"{entry.path} is placed: choose a place outside every entry's destination")


def _check(place: Path, what: str, guard: Guard | None) -> None:
    if guard is not None:
        guard(place, what)
    refuse_campaign(place, what)


def _copy_hashing(source: Iterable[bytes], handle) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    for chunk in source:
        digest.update(chunk)
        handle.write(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _chunks(handle) -> Iterator[bytes]:
    return iter(lambda: handle.read(_CHUNK), b"")


def _still_absent(root: Path, entry: Entry, when: str, guard: Guard | None = None) -> None:
    _require(check_destination(root, entry.dest) == "absent", f"{entry.dest}: appeared while it was {when}")
    _check(root / entry.dest, f"the destination of {entry.path}", guard)


def _make_directories(path: Path, made: list[Path], *, exist_ok: bool = True) -> None:
    made[:] = takewhile(lambda directory: not os.path.lexists(directory), (path, *path.parents))
    path.mkdir(parents=True, exist_ok=exist_ok)


def _remove_made(made: Sequence[Path]) -> None:
    for directory in made:
        with suppress(OSError):
            directory.rmdir()


@contextmanager
def _new_layout(out: Path) -> Iterator[None]:
    # Removes ``out`` and the ancestors made for it when the block fails, so the same command can be run again.
    made: list[Path] = []
    owned = False  # set only after this run's own mkdir of ``out`` returned
    try:
        _make_directories(out, made, exist_ok=False)
        owned = True
        yield
    except BaseException:
        if owned:
            shutil.rmtree(out, ignore_errors=True)
        _remove_made(made if owned else made[1:])
        raise


def place_file(source: Path, root: Path, entry: Entry, guard: Guard | None = None) -> int:
    """Copy one plain file into place through a staging file beside it; return the bytes written."""
    dest = root / entry.dest
    _still_absent(root, entry, "fetched", guard)
    dest.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.fetching-")
    staging = Path(staged)
    try:
        with os.fdopen(descriptor, "wb") as writer, source.open("rb") as reader:
            digest, size = _copy_hashing(_chunks(reader), writer)
        _require(digest == entry.sha256, f"{entry.path}: changed while it was copied")
        staging.chmod(0o666 & ~umask())  # mkstemp's 0600 is for the staging file only
        _still_absent(root, entry, "copied", guard)
        try:
            os.link(staging, dest)  # refuses an existing destination
        except FileExistsError as error:
            raise HubError(f"{entry.dest}: appeared while it was copied") from error
    finally:
        staging.unlink(missing_ok=True)
    return size


def _member_path(name: str, prefix: str, where: str) -> str:
    safe_path(name, where)
    if name == prefix:
        return ""
    _require(name.startswith(prefix + "/"), f"{where}: outside the archive's prefix {prefix}/")
    return name[len(prefix) + 1:]


def _checked_members(source: Path, prefix: str, what: str) -> Iterator[tuple[tarfile.TarInfo, str, tarfile.TarFile]]:
    import zstandard

    seen: set[str] = set()
    files: set[str] = set()
    dirs: set[str] = set()
    with source.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader, \
            tarfile.open(fileobj=reader, mode="r|") as archive:
        for member in archive:
            where = f"{what} member {member.name!r}"
            name = member.name.rstrip("/") if member.isdir() else member.name
            relative = _member_path(name, prefix, where)
            _require(name not in seen, f"{where}: appears twice")
            seen.add(name)
            _require(member.type in (tarfile.DIRTYPE, tarfile.REGTYPE, tarfile.AREGTYPE),
                     f"{where}: not a regular file or a directory")
            ancestors = {str(parent) for parent in PurePosixPath(name).parents}
            _require(not ancestors & files and (member.isdir() or (relative != "" and name not in dirs)),
                     f"{where}: collides with another member")
            (dirs if member.isdir() else files).add(name)
            dirs |= ancestors
            yield member, relative, archive


@contextmanager
def _unpacking(what: str, after: str = "") -> Iterator[None]:
    import zstandard

    try:
        yield
    except (tarfile.TarError, zstandard.ZstdError) as error:
        raise HubError(f"{what} cannot be unpacked ({type(error).__name__}: {error}){after}") from error


def _same_members(found: Mapping[str, str], listed: Mapping[str, str], what: str) -> None:
    unexpected, absent = sorted(set(found) - set(listed)), sorted(set(listed) - set(found))
    wrong = sorted(path for path in set(found) & set(listed) if found[path] != listed[path])
    _require(not (unexpected or absent or wrong), f"{what}: its members are not its manifest's (not listed "
                                                  f"{unexpected[:3]}, missing {absent[:3]}, other bytes {wrong[:3]})")


def extract_archive(source: Path, root: Path, entry: Entry, members: Mapping[str, str],
                    guard: Guard | None = None) -> int:
    """Extract one ``.tar.zst`` into a staging directory, check every member against ``members``, move it into place;
    return the bytes written."""
    dest = root / entry.dest
    _still_absent(root, entry, "fetched", guard)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}.fetching-"))
    found: dict[str, str] = {}
    written = 0
    try:
        with closing(_checked_members(source, entry.prefix, entry.path)) as stream:
            for member, relative, archive in stream:
                target = staging / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as writer:
                    found[member.name], size = _copy_hashing(_chunks(archive.extractfile(member)), writer)
                written += size
                os.utime(target, (member.mtime, member.mtime))
        _same_members(found, members, entry.path)
        staging.chmod(0o777 & ~umask())  # mkdtemp's 0700 is for the staging directory only
        _still_absent(root, entry, "extracted", guard)
        os.rename(staging, dest)  # refuses a directory that is not empty
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return written


def _download(repo: str, repo_type: str, revision: str | None, target: Path, files: Sequence[str]) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, repo_type=repo_type, revision=revision, local_dir=target,
                      allow_patterns=list(files), token=False)


def _remote_listing(repo: str, repo_type: str, revision: str | None) -> set[str]:
    from huggingface_hub import list_repo_files

    return {path for path in list_repo_files(repo, repo_type=repo_type, revision=revision, token=False)
            if path.split("/", 1)[0] not in SNAPSHOT_EXTRAS}


def _listing(base: Path) -> set[str]:
    found = set()
    for directory, subdirectories, files in os.walk(base):
        here = Path(directory)
        if here == base:
            subdirectories[:] = [name for name in subdirectories if name not in SNAPSHOT_EXTRAS]
            files = [name for name in files if name not in SNAPSHOT_EXTRAS]
        for name in subdirectories + files:
            _require(not (here / name).is_symlink(), f"{(here / name).relative_to(base)} is a symlink")
        found.update((here / name).relative_to(base).as_posix() for name in files)
    return found


def _manifest(base: Path, entry: Entry) -> dict[str, str]:
    _require(sha256_file(base / entry.members) == entry.members_sha256, f"{entry.members}: not the bytes {SUMS} lists")
    members = read_sums((base / entry.members).read_text(encoding="utf-8"), entry.members)
    _require(members_digest(members) == entry.members_digest, f"{entry.members}: not the members {INDEX} names")
    for name in members:
        _member_path(name, entry.prefix, f"{entry.members} {name!r}")
    return members


def _holds_members(root: Path, entry: Entry, members: Mapping[str, str]) -> bool:
    top = root / entry.dest
    if top.is_symlink() or not top.is_dir():
        return False
    symlinks: list[str] = []
    try:
        files = {f"{entry.prefix}/{relative}": relative for relative, _ in _tree_files(top, symlinks)}
    except HubError:  # a special file
        return False
    return (not symlinks and set(files) == set(members)
            and all(sha256_file(top / files[name]) == members[name] for name in sorted(files)))


def fetch(repo: str | Path, root: str | Path, *, only: Sequence[str] = (), dry_run: bool = False,
          repo_type: str = "dataset", revision: str | None = None, download_dir: str | Path | None = None,
          scratch_dir: str | Path | None = None, guard: Guard | None = None,
          echo: Callable[[str], None] = print) -> dict[str, Any]:
    """Place a repository's files (a Hub id or a local directory) under the artifact root ``root`` and return what was
    placed, ``dry_run`` checking and listing without writing under ``root``, a Hub download held in a new directory
    inside ``download_dir`` (a dry run's in ``scratch_dir``)."""
    root = check_root(Path(root))
    local = Path(repo).is_dir()
    base = Path(repo)
    download: Path | None = None
    made: list[Path] = []
    refused = False
    try:
        if not local:
            if dry_run:
                parent = Path(scratch_dir) if scratch_dir else temporary_parent(root, guard)
            else:
                parent = Path(download_dir or root)
            _check(parent, DOWNLOAD, guard)
            if dry_run:
                _refuse_in_root(parent, root, DOWNLOAD, "a dry run writes nothing under the root; give it a "
                                                       "scratch_dir outside it")
            elif download_dir is not None:
                _make_directories(parent, made)
            download = base = Path(tempfile.mkdtemp(prefix=".hub-download-", dir=parent))
            _download(str(repo), repo_type, revision, base, [INDEX, SUMS])
        sums = read_sums((base / SUMS).read_text(encoding="utf-8"))
        _require(sha256_file(base / INDEX) == sums.get(INDEX), f"{INDEX} is not the one {SUMS} lists")
        index = load_index(json.loads((base / INDEX).read_text(encoding="utf-8")), sums)
        listing = (_listing(base) if local else _remote_listing(str(repo), repo_type, revision)) - {SUMS}
        _require(listing == set(sums), f"{repo}: files not in {SUMS} {sorted(listing - set(sums))[:5]}, "
                                       f"missing {sorted(set(sums) - listing)[:5]}")
        selected = index.select(only)
        if not local and not dry_run:
            refuse_in_destinations(parent, root, selected, DOWNLOAD)
        states = {entry.path: check_destination(root, entry.dest) for entry in selected}
        for entry in selected:
            _require(entry.kind != "root" or states[entry.path] == "absent",
                     f"{entry.dest} exists under {root}: nothing is fetched over a root")
        present_archives = [entry for entry in selected if entry.archive and states[entry.path] == "present"]
        if not local and present_archives:
            _check(base, DOWNLOAD, guard)  # after a network call: what appeared meanwhile is refused before a write
            _download(str(repo), repo_type, revision, base, [entry.members for entry in present_archives])
        plan = []
        for entry in selected:
            state = states[entry.path]
            if state == "present":
                dest = root / entry.dest
                same = (_holds_members(root, entry, _manifest(base, entry)) if entry.archive
                        else dest.is_file() and not dest.is_symlink() and sha256_file(dest) == entry.sha256)
                _require(same, f"{entry.dest} exists under {root} and is not this entry's "
                               f"{'members' if entry.archive else 'bytes'}: nothing is fetched over it")
            plan.append((entry, state))
        for entry, state in plan:
            echo(f"{'present' if state == 'present' else 'fetch':8} {entry.kind:9} {entry.dest} ({entry.bytes:,} "
                 f"bytes{', ' + entry.format if entry.archive else ''})")
        report = {"root": str(root), "placed": [], "bytes_written": 0, "dry_run": dry_run,
                  "present": [entry.dest for entry, state in plan if state != "absent"]}
        if dry_run:
            echo(f"dry run: {len(plan)} entries checked, nothing written")
            return report
        todo = [entry for entry, state in plan if state == "absent"]
        if not local:
            _check(base, DOWNLOAD, guard)
            _download(str(repo), repo_type, revision, base, [entry.path for entry in todo]
                      + [entry.members for entry in todo if entry.archive] + list(index.documents))
        for document in index.documents:
            _require(sha256_file(base / document) == sums[document], f"{document}: not the bytes {SUMS} lists")
        manifests = {}
        for entry in todo:
            _require(sha256_file(base / entry.path) == entry.sha256, f"{entry.path}: not the bytes {SUMS} lists")
            if entry.archive:
                manifests[entry.path] = _manifest(base, entry)
        for entry in todo:
            source = base / entry.path
            if entry.archive:
                with _unpacking(f"{repo}: {entry.path}, an archive with the bytes its {SUMS} lists,",
                                ": the repository is damaged; nothing of that archive is placed, and what was placed "
                                "before it stays"):
                    written = extract_archive(source, root, entry, manifests[entry.path], guard)
            else:
                written = place_file(source, root, entry, guard)
            report["placed"].append(entry.dest)
            report["bytes_written"] += written
            echo(f"placed   {entry.dest} ({written:,} bytes)")
        echo(f"placed {len(report['placed'])} entries under {root}: {report['bytes_written'] / 1e9:.2f} GB written"
             + (f"; {len(report['present'])} already present" if report["present"] else ""))
        return report
    except BaseException:
        refused = True
        raise
    finally:
        if download is not None:
            shutil.rmtree(download, ignore_errors=True)
        if refused:
            _remove_made(made)


# How many wrong files a verification's refusal names.
SHOWN = 20
TEMPORARY = "the temporary directory (TMPDIR)"


def resolve_commit(repository: str | Path, repo_type: str, revision: str | None) -> str | None:
    """Return the commit ``revision`` of a Hub repository names now, or None for a local directory."""
    if Path(repository).is_dir():
        return None
    from huggingface_hub import repo_info

    commit = repo_info(str(repository), repo_type=repo_type, revision=revision, token=False).sha
    _require(bool(commit), f"{repository}: the Hub names no commit for revision {revision or 'main'}")
    return commit


def temporary_parent(root: Path | None, guard: Guard | None = None) -> Path:
    """Return the temporary directory to hold a Hub reference in, without writing a probe file or going under
    ``root``."""
    candidates = [tempfile.tempdir] if tempfile.tempdir is not None else tempfile._candidate_tempdir_list()
    for candidate in candidates:
        written = Path(os.path.abspath(candidate))
        _check(written, TEMPORARY, guard)
        _refuse_in_root(written, root, TEMPORARY, _OUTSIDE_TMPDIR)
        if written.is_dir() and os.access(written, os.W_OK | os.X_OK):
            return written
    raise HubError(f"no temporary directory to hold the repository's reference: none of "
                   f"{', '.join(map(str, candidates))} is a directory this process may write; set TMPDIR to one")


_OUTSIDE_TMPDIR = ("the repository's reference is downloaded there, and a dry run and a verification write nothing "
                   "under the root; set TMPDIR to a directory outside it")


def _refuse_in_root(place: Path | str, root: Path | None, what: str, remedy: str) -> None:
    written = Path(os.path.abspath(place))
    resolved = Path(os.path.realpath(written))
    for top in () if root is None else dict.fromkeys((Path(os.path.abspath(root)), Path(os.path.realpath(root)))):
        _require(not (written.is_relative_to(top) or resolved.is_relative_to(top)),
                 f"{what} {written} is {'' if top in (written, resolved) else 'inside '}the artifact root {root}: "
                 f"{remedy}")


@contextmanager
def reference(repository: str | Path, repo_type: str, revision: str | None, only: Sequence[str],
              scratch: Path | None, guard: Guard | None = None, *,
              root: Path | None = None) -> Iterator[tuple[Path, list[Entry]]]:
    """Yield where a repository's reference files are and the index entries ``only`` names, a Hub repository's
    ``index.json`` and ``SHA256SUMS`` downloaded into a new directory in ``scratch`` that is removed after the block."""
    repository = str(repository)
    local = Path(repository).is_dir()
    base = Path(repository)
    download: Path | None = None
    try:
        if not local:
            parent = Path(scratch) if scratch is not None else temporary_parent(root, guard)
            _check(parent, TEMPORARY, guard)
            _refuse_in_root(parent, root, TEMPORARY, _OUTSIDE_TMPDIR)
            download = base = Path(tempfile.mkdtemp(prefix="ruleforge-hub-reference-", dir=parent))
            _download(repository, repo_type, revision, base, [INDEX, SUMS])
        sums = read_sums((base / SUMS).read_text(encoding="utf-8"))
        _require(sha256_file(base / INDEX) == sums.get(INDEX), f"{INDEX} is not the one {SUMS} lists")
        try:
            index = load_index(json.loads((base / INDEX).read_text(encoding="utf-8")), sums)
            groups = {entry.group for entry in index.entries}
            names = groups | {entry.root for entry in index.entries if entry.root is not None}
        except TypeError as error:
            raise HubError(f"{repository}: its {INDEX}, the one {SUMS} lists, is not an index ({error}): the "
                           "repository is damaged; nothing is read from it") from error
        unknown = sorted(set(only) - names)
        _require(not unknown, f"{', '.join(unknown)}: neither an experiment of {repository} "
                              f"({', '.join(sorted(groups))}) nor one of its campaign roots")
        yield base, index.select(only)
    finally:
        if download is not None:
            shutil.rmtree(download, ignore_errors=True)


def _read_root(root: str | Path) -> Path:
    text = os.fspath(root)
    written = Path(os.path.abspath(os.path.join(os.getcwd(), os.path.expanduser(text))))
    resolved = second_place(text, written)
    _require(resolved is None,
             f"the artifact root {root} names two places: {written} as written out, and {resolved} where it is read "
             "(a '..' after a symlink, which the kernel takes from the link's target); give it without that '..'")
    return check_root(written)


def _file_problems(root: Path, entry: Entry) -> list[str]:
    path = root / entry.dest
    if path.is_symlink() or not path.is_file():
        return [f"{entry.dest}: not a regular file"]
    return [] if sha256_file(path) == entry.sha256 else [f"{entry.dest}: other bytes than {entry.path}'s"]


def _archive_problems(root: Path, entry: Entry, members: Mapping[str, str]) -> tuple[list[str], int]:
    top = root / entry.dest
    if top.is_symlink() or not top.is_dir():
        return [f"{entry.dest}: not a directory"], 0
    symlinks: list[str] = []
    found = {relative for relative, _ in _tree_files(top, symlinks)}
    expected = {_member_path(name, entry.prefix, entry.members): digest for name, digest in members.items()}
    problems = [f"{entry.dest}/{name}: a symlink" for name in symlinks]
    problems += [f"{entry.dest}/{name}: not a file of {entry.path}" for name in sorted(found - set(expected))]
    problems += [f"{entry.dest}/{name}: missing" for name in sorted(set(expected) - found)]
    problems += [f"{entry.dest}/{name}: other bytes than {entry.path}'s" for name in sorted(found & set(expected))
                 if sha256_file(top / name) != expected[name]]
    return problems, len(found)


def verify(repository: str | Path, root: str | Path, *, only: Sequence[str] = (), repo_type: str = "dataset",
           revision: str | None = None, guard: Guard | None = None,
           echo: Callable[[str], None] = print) -> dict[str, Any]:
    """Check a repository's entries present under ``root`` against its reference, writing nothing there; return those
    verified and those absent, refusing a selection none of which is there and any file that differs."""
    root = _read_root(root)
    local = Path(repository).is_dir()
    scratch = None if local else temporary_parent(root, guard)
    with reference(repository, repo_type, revision, only, scratch, guard, root=root) as (base, selected):
        present = {entry.path for entry in selected if check_destination(root, entry.dest) == "present"}
        _require(bool(present), f"none of the {len(selected)} entries named is under {root}: nothing to verify")
        archives = [entry for entry in selected if entry.archive and entry.path in present]
        if not local and archives:
            _check(base, "the directory the repository's reference is held in", guard)
            _download(str(repository), repo_type, revision, base, [entry.members for entry in archives])
        report: dict[str, Any] = {"root": str(root), "verified": [], "absent": [], "files": 0}
        problems: list[str] = []
        for entry in selected:
            if entry.path not in present:
                report["absent"].append(entry.dest)
                echo(f"absent   {entry.dest}")
                continue
            if entry.archive:
                found, files = _archive_problems(root, entry, _manifest(base, entry))
            else:
                found, files = _file_problems(root, entry), 1
            problems += found
            report["files"] += files
            if not found:
                report["verified"].append(entry.dest)
            echo(f"{'differs' if found else 'verified':8} {entry.dest}")
    if problems:
        more = f"; and {len(problems) - SHOWN} more" if len(problems) > SHOWN else ""
        raise HubError(f"{len(problems)} file(s) under {root} are not the repository's: "
                       f"{'; '.join(problems[:SHOWN])}{more}")
    echo(f"verified {len(report['verified'])} entries under {root} ({report['files']:,} files)"
         + (f"; {len(report['absent'])} not there" if report["absent"] else ""))
    return report


@dataclass(frozen=True)
class CompactRule:
    """Which files of a campaign root the compact tier keeps; ``attempts_conditions`` are globs over public condition
    names."""

    attempts_conditions: tuple[str, ...] = ("forced*",)
    draw0_evaluations: bool = False
    all_attempts: bool = False
    raw_outputs: bool = False

    def keeps(self, relative: str) -> bool:
        """Return whether the rule keeps ``relative``, a path inside the root; a run record is never kept."""
        parts = relative.split("/")
        if run_record(relative):
            return False
        if parts[-1] not in DRAW_TEXT:
            return True
        if parts[-1] == "attempts.jsonl":
            return self.all_attempts or (len(parts) >= 3 and any(
                fnmatch.fnmatchcase(public_segment(parts[1]), pattern) for pattern in self.attempts_conditions))
        if parts[-1] == "evaluations.jsonl":
            return self.draw0_evaluations and len(parts) >= 2 and parts[-2] == "draw-000"
        return self.raw_outputs and parts[-1] == "raw_outputs.jsonl"


def run_record(relative: str) -> bool:
    """Whether ``relative``, a path inside a campaign root, is one of its run records (:data:`RUN_RECORDS`)."""
    parts = tuple(relative.split("/"))
    return any(_glob(tuple(pattern.split("/")), parts) for pattern in RUN_RECORDS)


def _glob(pattern: tuple[str, ...], parts: tuple[str, ...]) -> bool:
    if not pattern:
        return not parts
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        return any(_glob(rest, parts[index:]) for index in range(len(parts) + 1))
    return bool(parts) and fnmatch.fnmatchcase(parts[0], head) and _glob(rest, parts[1:])


class Omit:
    """A tree's omit list: globs over its file paths (``**`` any number of segments); one that leaves out nothing is
    refused."""

    def __init__(self, patterns: Sequence[str] = ()) -> None:
        self.patterns = tuple(patterns)
        self._split = [tuple(pattern.split("/")) for pattern in self.patterns]
        self.counts = Counter({pattern: 0 for pattern in self.patterns})

    def leaves_out(self, relative: str) -> bool:
        """Whether a glob matches ``relative`` (a path inside the tree), counted for each that does."""
        parts = tuple(map(public_segment, relative.split("/")))
        hits = [pattern for pattern, split in zip(self.patterns, self._split) if _glob(split, parts)]
        self.counts.update(hits)
        return bool(hits)

    def keeping(self, keeps: Callable[[str], bool]) -> Callable[[str], bool]:
        """Return ``keeps`` with this list's files left out."""
        return lambda relative: not self.leaves_out(relative) and keeps(relative)

    def check(self, what: str, echo: Callable[[str], None]) -> None:
        """Refuse a glob that left out nothing of ``what``; then say how many files each left out."""
        unused = [pattern for pattern, count in self.counts.items() if count == 0]
        _require(not unused, f"{what}: its omit list's {unused} leave out no file: it names what the tree does not "
                             "hold")
        if self.patterns:
            echo(f"{what}: {sum(self.counts.values())} file(s) omitted ("
                 + ", ".join(f"{pattern}: {count}" for pattern, count in self.counts.items()) + ")")


def _tree_files(top: Path, symlinks: list[str]) -> list[tuple[str, os.stat_result]]:
    _require(top.is_dir() and not top.is_symlink(), f"{top} is not a directory")
    found = []
    for directory, subdirectories, files in os.walk(top, onerror=_raise):
        for name in sorted(subdirectories + files):
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                symlinks.append(path.relative_to(top).as_posix())
                continue
            _require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), f"{path}: not a regular file or a directory")
            if stat.S_ISREG(mode):
                found.append((path.relative_to(top).as_posix(), path.lstat()))
    return sorted(found, key=lambda item: item[0])


def _raise(error: OSError) -> None:
    raise error


def _archive_members(archive: Path, prefix: str, what: str) -> dict[str, str]:
    members = {}
    with _unpacking(what), closing(_checked_members(archive, prefix, what)) as stream:
        for member, _, tar in stream:
            if not member.isdir():
                digest = hashlib.sha256()
                for chunk in _chunks(tar.extractfile(member)):
                    digest.update(chunk)
                members[member.name] = digest.hexdigest()
    return members


@dataclass(frozen=True)
class Publication:
    """What a published copy is made with beyond its own bytes: the published SHA-256 of each file the package
    rewrites, by its recorded SHA-256, and an adapter's base model as the Hub names it."""

    digests: Mapping[str, str] = field(default_factory=dict)
    base_model: str | None = None
    private: tuple[str, ...] = ()  # names (hosts, accounts) no published file may hold outside text fields


def _absolute_parts(value: Any) -> tuple[str, ...] | None:
    return PurePosixPath(value).parts if isinstance(value, str) and value.startswith("/") else None


def _from(parts: tuple[str, ...] | None, start: int | None, prefix: str = "") -> Any:
    return parts[-1] if start is None else prefix + "/".join(parts[start:])


def _location(value: Any, _: Publication) -> Any:
    parts = _absolute_parts(value)
    if parts is None:
        return value
    return _from(parts, next((at for at, part in enumerate(parts) if part in ARTIFACT_DIRS), None), ARTIFACT_SCHEME)


def _repo(value: Any, _: Publication) -> Any:
    parts = _absolute_parts(value)
    return value if parts is None else _from(parts, next((at for at, part in enumerate(parts) if part == "data"), None))


def _hub(value: Any, publication: Publication) -> Any:
    _require(publication.base_model is not None, "an adapter's base model is published as the Hub names it, and the "
                                                 "inventory names none for it")
    return publication.base_model


_RULES: dict[str, Callable[[Any, Publication], Any]] = {
    "name": lambda value, _: PurePosixPath(value).name if isinstance(value, str) else value,
    "host": lambda value, _: HOST_PLACEHOLDER,
    "location": _location,  # an artifact:// URI from the path's first artifact-root directory, else its last segment
    "repo": _repo,  # the path from its data/ segment: where the code repository holds the file
    "digest": lambda value, publication: publication.digests.get(value, value),
    "hub": _hub,
}


def _apply(node: Any, steps: Sequence[str], rule: str, publication: Publication) -> bool:
    key, rest = steps[0], steps[1:]
    if key.endswith("[]"):
        items = node.get(key[:-2]) if isinstance(node, dict) else None
        changed = False
        for item in items if isinstance(items, list) else ():
            changed |= _apply(item, rest, rule, publication)
        return changed
    if not isinstance(node, dict) or key not in node:
        return False
    if rest:
        return _apply(node[key], rest, rule, publication)
    value = _RULES[rule](node[key], publication)
    changed, node[key] = value != node[key], value
    return changed


def _paths(paths: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(path.split(".")) for path in paths)


def _names(patterns: tuple[tuple[str, ...], ...], here: tuple[str, ...]) -> bool:
    # ``*`` stands for a key, never for a list's objects (``key[]``)
    return any(len(pattern) == len(here) and all(part == key or (part == "*" and not key.endswith("[]"))
                                                 for part, key in zip(pattern, here))
               for pattern in patterns)


def _beneath(patterns: tuple[tuple[str, ...], ...], here: tuple[str, ...]) -> bool:
    return any(len(pattern) > len(here) and _names((pattern[:len(here)],), here) for pattern in patterns)


def _scalar(value: Any) -> bool:
    plain = (str, int, float, bool, type(None))
    return isinstance(value, plain) or (isinstance(value, list) and all(isinstance(item, plain) for item in value))


def _project(node: Mapping[str, Any], projection: Projection, here: tuple[str, ...], where: str) -> dict[str, Any]:
    fields, whole, without = _paths(projection.fields), _paths(projection.whole), _paths(projection.without)
    kept: dict[str, Any] = {}
    for key, value in node.items():
        path = (*here, key)
        if _names(without, path):
            continue
        if _names(fields, path):  # a named field
            _require(_scalar(value) or _names(whole, path), f"{where}: {'.'.join(path)} holds an object, and its "
                                                            "projection neither names its fields nor keeps it whole")
            kept[key] = value
        elif _beneath(fields, (*here, f"{key}[]")):  # a list of objects, whose fields it names
            _require(isinstance(value, list) and all(isinstance(item, dict) for item in value),
                     f"{where}: {'.'.join(path)} is not a list of objects, and its projection names their fields")
            kept[key] = [_project(item, projection, (*here, f"{key}[]"), where) for item in value]
        elif _beneath(fields, path):
            _require(value is None or isinstance(value, dict), f"{where}: {'.'.join(path)} is not an object, and its "
                                                               "projection names fields of it")
            kept[key] = None if value is None else _project(value, projection, path, where)
    return kept


def _projected(record: Any, projection: Projection, where: str) -> Any:
    """``record`` as ``projection`` keeps it, the fields it names at every level: None for a row it leaves out."""
    _require(isinstance(record, dict), f"{where} is not a JSON object, so the fields its published copy keeps cannot "
                                       "be chosen")
    if any(record.get(key) != value for key, value in projection.row):
        return None
    return _project(record, projection, (), where)


def _published_text(raw: bytes, fields: Sequence[tuple[list[str], str]], forms: Sequence[Callable[[Any], str]],
                    where: str, publication: Publication, projection: Projection | None = None,
                    keep: Callable[[Any], bool] | None = None) -> bytes | None:
    try:
        record = json.loads(raw)
    except ValueError as error:
        raise HubError(f"{where} is not JSON ({error}): its published copy cannot be made") from error
    if keep is not None and not keep(record):
        return None
    changed = False
    if projection is not None:
        kept = _projected(record, projection, where)
        if kept is None:
            return None
        changed, record = kept != record, kept
    for steps, rule in fields:
        changed |= _apply(record, steps, rule, publication)
    if not changed:
        return raw
    recorded = json.loads(raw)  # the rewrites changed the parsed record in place
    form = next((form for form in forms if form(recorded).encode("utf-8") == raw), None)
    _require(form is not None, f"{where} is not written as the code writes it, so its published copy would differ "
                               "beyond the fields it publishes")
    return form(record).encode("utf-8")


def published_record(name: str, data: bytes, where: str = "a run record",
                     publication: Publication = Publication(), projection: Projection | None = None,
                     keep: Callable[[Any], bool] | None = None) -> bytes:
    """Return a file's bytes with the fields :data:`PUBLISHED_FIELDS` names for file ``name`` made public and, under a
    ``projection``, only the rows and fields it keeps, and of a JSONL file only the rows ``keep`` holds to (the same
    bytes when nothing changes), in the form its recorded bytes are written in, refusing a file whose rewrite would
    change anything else."""
    from ruleforge.core.files import json_sidecar_text

    fields = [(path.split("."), rule) for file, path, rule in PUBLISHED_FIELDS if file == name]
    keys = {f'"{steps[-1]}"'.encode("utf-8") for steps, _ in fields}
    if projection is None and keep is None and not any(key in data for key in keys):  # copied as it is
        return data
    if name.endswith(".jsonl"):
        lines = (lambda record: json.dumps(record, sort_keys=True), json.dumps,
                 lambda record: json.dumps(record, ensure_ascii=False))  # the ledger's and its views' lines
        published = (_published_text(line, fields, lines, f"{where} line {number}", publication, projection, keep)
                     if line and (projection is not None or keep is not None or any(key in line for key in keys))
                     else line
                     for number, line in enumerate(data.split(b"\n"), 1))
        return b"\n".join(line for line in published if line is not None)
    documents = (json_sidecar_text, lambda record: json.dumps(record, indent=1) + "\n",
                 lambda record: json.dumps(record, indent=2), json.dumps, lambda record: json.dumps(record) + "\n")
    published = _published_text(data, fields, documents, where, publication, projection)
    _require(published is not None, f"{where}: a projection's row condition applies to a JSONL file's rows only")
    return published


def _text_field(where: str, text: Sequence[tuple[str, ...]]) -> bool:
    return _names(text, tuple(where.split(".")))


def _holding(node: Any, where: str, needles: Sequence[str], text: Sequence[tuple[str, ...]]) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _holding(value, f"{where}.{key}" if where else key, needles, text)
    elif isinstance(node, list):
        for value in node:
            yield from _holding(value, where + "[]", needles, text)
    elif isinstance(node, str) and not _text_field(where, text) and (
            any(needle in node for needle in needles) or ABSOLUTE_PATH.search(node)
            or WHOLE_ABSOLUTE_PATH.fullmatch(node)):
        yield where


def refuse_private(name: str, data: bytes, where: str, private: Sequence[str] = ()) -> None:
    """Refuse a JSON or JSONL file that holds, in a field outside :data:`TEXT_FIELDS`, an absolute path
    (:data:`ABSOLUTE_PATH`, or a whole value :data:`WHOLE_ABSOLUTE_PATH`), a local path or one of the ``private``
    names; the refusal names the file, line and field, never the value."""
    needles = (*LOCAL_ROOTS, str(Path.home()), *private)
    encoded = [needle.encode("utf-8") for needle in needles]

    def suspect(chunk: bytes) -> bool:
        return (any(needle in chunk for needle in encoded) or _ABSOLUTE_PATH_BYTES.search(chunk) is not None
                or _WHOLE_ABSOLUTE_PATH_BYTES.search(chunk) is not None)

    if not name.endswith((".json", ".jsonl")) or not suspect(data):
        return
    text = _paths(TEXT_FIELDS.get(name, ()))
    lines = data.split(b"\n") if name.endswith(".jsonl") else [data]
    for number, line in enumerate(lines, 1):
        if not suspect(line):
            continue
        at = f"{where} line {number}" if name.endswith(".jsonl") else where
        try:
            record = json.loads(line)
        except ValueError as error:
            raise HubError(f"{at} is not JSON ({error}), so its fields cannot be held to the local-name check") from error
        held = sorted(set(_holding(record, "", needles, text)))
        _require(not held, f"{at}: {', '.join(held)} name an absolute or local path or a private name in a field no "
                           "publication rule or text field covers; add the field to PUBLISHED_FIELDS or TEXT_FIELDS")


def _publishing(where: str, publication: Publication, profile: Profile | None = None,
                campaign: Path | None = None) -> Callable[[str, Path], bytes | None]:
    """Return a tree's files as :func:`published_record` publishes them (None: as recorded), each under its
    ``profile``'s projection and each JSON file held to :func:`refuse_private`; in a ``campaign`` root, each attempt
    ledger as its published form (:data:`LEDGER_PUBLISHED_AS`) of the profile's parts."""
    projections = {} if profile is None else profile.projections
    names = {name for name, _, _ in PUBLISHED_FIELDS} | set(projections)

    def published(relative: str, path: Path) -> bytes | None:
        name = PurePosixPath(relative).name
        if campaign is not None and name in LEDGER_PUBLISHED_AS:
            data = _published_ledger(campaign, relative, () if profile is None else profile.ledger)
            refuse_private(LEDGER_PUBLISHED_AS[name], data, f"{where}/{relative}", publication.private)
            return data
        if campaign is not None and name == RAW_OUTPUTS:
            recorded = path.read_bytes()
            data = _published_raw_outputs(campaign, relative, recorded, f"{where}/{relative}", publication,
                                          projections.get(name))
            refuse_private(name, data, f"{where}/{relative}", publication.private)
            return None if data == recorded else data
        if name not in names and not name.endswith((".json", ".jsonl")):
            return None
        data = path.read_bytes()
        rewritten = (published_record(name, data, f"{where}/{relative}", publication, projections.get(name))
                     if name in names else data)
        refuse_private(name, rewritten, f"{where}/{relative}", publication.private)
        return None if rewritten == data else rewritten

    return published


def _draw_identity(campaign: Path, relative: str) -> tuple[Path, dict[str, Any]]:
    """The draw directory ``relative`` (a file of a campaign root's draw) lies in, and its ledger's identity: the draw
    named by its place, the campaign by the root's sampling manifest."""
    draw = PurePosixPath(relative).parent
    manifest = campaign / "sampling_manifest.json"
    _require(manifest.is_file(), f"{campaign} holds draw records but no sampling_manifest.json naming its campaign")
    _require(bool(re.fullmatch(r"draw-\d+", draw.name)) and len(draw.parts) >= 2,
             f"{campaign.name}/{relative}: a ledger or raw outputs file lies in a condition's draw directory")
    return campaign / draw, {"campaign_id": json.loads(manifest.read_text())["campaign_id"],
                             "condition_id": draw.parent.as_posix(),
                             "decoder_draw_id": int(draw.name.removeprefix("draw-"))}


def _published_ledger(campaign: Path, relative: str, parts: Sequence[str]) -> bytes:
    """A campaign root's draw ledger ``relative`` as its strict replay publishes it, of ``parts``."""
    from ruleforge.evaluate.ledger import published_form_text

    draw, identity = _draw_identity(campaign, relative)
    return published_form_text(draw, **identity, parts=tuple(parts))


def _published_raw_outputs(campaign: Path, relative: str, data: bytes, where: str, publication: Publication,
                           projection: Projection | None) -> bytes:
    """A campaign root's draw raw outputs ``relative`` as published: the rows of the draw's accepted full-ruleset calls
    alone, by its ledger's strict replay, under ``projection``."""
    from ruleforge.evaluate.ledger import LEDGER_FNAME, accepted_calls

    draw, identity = _draw_identity(campaign, relative)
    _require((draw / LEDGER_FNAME).is_file(), f"{campaign.name}/{relative}: a draw's raw outputs are published by its "
                                              f"ledger's replay, and the draw holds no {LEDGER_FNAME}")
    calls = accepted_calls(draw, **identity)

    def accepted(row: Any) -> bool:
        return isinstance(row, dict) and (
            row.get("scenario_id"), row.get("semantic_attempt_id"), row.get("request_block_id")) in calls

    return published_record(RAW_OUTPUTS, data, where, publication, projection, keep=accepted)


def _published_digests(artifact_root: Path, inventory: Mapping[str, Any]) -> dict[str, str]:
    """Return ``{recorded SHA-256: published SHA-256}`` of each plain file of the inventory the package rewrites."""
    names = {name for name, _, _ in PUBLISHED_FIELDS}
    found = {}
    for experiment in inventory["experiments"].values():
        for item in experiment.get("files", []):
            source = artifact_root / item["dest"]
            if item.get("format", FILE) == FILE and source.name in names and source.is_file():
                data = source.read_bytes()
                published = published_record(source.name, data, item["dest"])
                if published != data:
                    found[hashlib.sha256(data).hexdigest()] = hashlib.sha256(published).hexdigest()
    return found


def pack_tree(top: Path, prefix: str, archive: Path, manifest: Path, *, keeps: Callable[[str], bool] = lambda _: True,
              level: int = 19, threads: int = 0, symlinks: list[str] | None = None,
              published: Callable[[str, Path], bytes | None] | None = None,
              extra: Mapping[str, Callable[[Sequence[str]], bytes]] | None = None,
              renames: Mapping[str, str] | None = None) -> dict[str, str]:
    """Write ``top``'s files, and the ``extra`` files no tree file may share a path with (each made from the tree's
    kept files), as a reproducible ``.tar.zst`` and their SHA-256s as ``manifest`` and return the members, symlinks
    left out and appended to ``symlinks``, ``published`` giving a file's bytes as packed (None: as on disk) and
    ``renames`` the name a file of each name is packed under (``published`` gives its bytes)."""
    import zstandard

    members: dict[str, str] = {}
    archive.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstandard.ZstdCompressor(level=level, threads=threads)
    had_manifest = manifest.exists()
    raw = archive.open("xb")  # refuses an existing archive
    try:
        with raw, compressor.stream_writer(raw, closefd=False) as writer:
            with tarfile.open(fileobj=writer, mode="w|", format=tarfile.GNU_FORMAT) as tar:
                files = _tree_files(top, [] if symlinks is None else symlinks)
                renames = renames or {}
                clash = sorted(set(extra or {}) & {relative for relative, _ in files}) + sorted(
                    relative for relative, _ in files if PurePosixPath(relative).name in set(renames.values()))
                _require(not clash, f"{top} already holds {clash}, which its published copy writes")
                kept: list[str] = []
                for relative, info in files:
                    if not keeps(relative):
                        continue
                    kept.append(relative)
                    path = PurePosixPath(relative)
                    name = f"{prefix}/{path.with_name(renames.get(path.name, path.name))}"
                    data = None if published is None else published(relative, top / relative)
                    _require(data is not None or path.name not in renames, f"{top / relative}: a renamed file is "
                                                                           "packed as its published bytes")
                    member = tarfile.TarInfo(name)
                    member.size, member.mtime = info.st_size if data is None else len(data), int(info.st_mtime)
                    member.mode = stat.S_IMODE(info.st_mode)
                    member.uid = member.gid = 0
                    member.uname = member.gname = ""
                    if data is None:
                        with (top / relative).open("rb") as handle:
                            tar.addfile(member, handle)
                        members[name] = sha256_file(top / relative)
                    else:
                        tar.addfile(member, io.BytesIO(data))
                        members[name] = hashlib.sha256(data).hexdigest()
                for relative, make in sorted((extra or {}).items()):
                    data = make(kept)
                    member = tarfile.TarInfo(f"{prefix}/{relative}")
                    member.size, member.mode, member.uname, member.gname = len(data), 0o644, "", ""
                    tar.addfile(member, io.BytesIO(data))
                    members[member.name] = hashlib.sha256(data).hexdigest()
        # Read back from the archive: its bytes must be the tree's as hashed, and the manifest must parse to the same
        # members.
        archived = _archive_members(archive, prefix, str(archive))
        _same_members(archived, members, f"{archive} (a file changed while it was packed)")
        text = write_sums(archived)
        _require(read_sums(text) == archived, f"{archive}: a member name its manifest cannot hold")
        manifest.write_text(text, encoding="utf-8")
    except BaseException:
        archive.unlink(missing_ok=True)
        if not had_manifest:
            manifest.unlink(missing_ok=True)
        raise
    return archived


@dataclass
class _Built:
    out: Path
    entries: list[Entry] = field(default_factory=list)

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)

    def finish(self, compact_rule: str, card: Path | None) -> Index:
        documents = []
        if card is not None:
            shutil.copyfile(card, self.out / CARD)
            documents.append(CARD)
        payload = {"format": FORMAT, "compact_rule": compact_rule, "documents": documents,
                   "files": [entry.to_json() for entry in sorted(self.entries, key=lambda entry: entry.path)]}
        (self.out / INDEX).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        sums = {path: sha256_file(self.out / path) for path in sorted(_listing(self.out) - {SUMS})}
        (self.out / SUMS).write_text(write_sums(sums), encoding="utf-8")
        return load_index(payload, sums)


def _roots(experiment: Mapping[str, Any]) -> dict[str, str | None]:
    return dict(experiment.get("roots", {}))


def _omits(experiment: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    return {root: tuple(patterns) for root, patterns in experiment.get("omit", {}).items()}


def _published_copy(root: str, kept: Sequence[str] = ()) -> bytes:
    """What a published copy of campaign root ``root`` holds at its top: being there says it is one, and it names the
    conditions whose draws hold their ledger's published form (of the root's ``kept`` files)."""
    from ruleforge.core.files import json_sidecar_text
    from ruleforge.evaluate.ledger import LEDGER_CONDITIONS_KEY

    conditions = sorted({str(PurePosixPath(relative).parent.parent) for relative in kept
                         if PurePosixPath(relative).name in LEDGER_PUBLISHED_AS})
    return json_sidecar_text({LEDGER_CONDITIONS_KEY: conditions}).encode("utf-8")


def _named_kinds(root: str, keeps: Callable[[str], bool]) -> Callable[[str], bool]:
    """``keeps``, refusing a file it keeps whose kind :data:`PUBLISHED_KINDS` does not name."""
    kinds = _paths_split(PUBLISHED_KINDS)

    def named(relative: str) -> bool:
        kept = keeps(relative)
        parts = tuple(relative.split("/"))
        _require(not kept or any(_glob(kind, parts) for kind in kinds),
                 f"{root}/{relative}: a file kind the package's rules neither publish (hub.PUBLISHED_KINDS) nor leave "
                 "out (an omit glob or the compact rule): decide which")
        return kept

    return named


def _paths_split(globs: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(pattern.split("/")) for pattern in globs)


def _profile(experiment: Mapping[str, Any], root: str) -> Profile:
    """The :data:`PROFILES` entry the inventory names for ``root`` (its own, else its experiment's); a root without one
    is refused."""
    name = (experiment.get("profiles") or {}).get(root, experiment.get("profile"))
    _require(name in PROFILES, f"{root}: the inventory names no profile of {sorted(PROFILES)} for it, so what its "
                               "published copy keeps is not decided")
    return PROFILES[name]


def _every_root(inventory: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(inventory.get("compact", {}).get("omit", ()))


def load_inventory(path: Path | str) -> dict[str, Any]:
    """Return the inventory (``data/hub_inventory.yaml``), refusing unsafe paths, repeats, bad pins and overlapping
    destinations."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(data, dict) and data.get("format") == FORMAT, f"{path}: not an inventory of format {FORMAT}")
    _require(isinstance(data.get("experiments"), dict) and data["experiments"], f"{path}: no experiments")
    if "omit" in data.get("compact", {}):
        _omit_list(data["compact"]["omit"], f"{path} compact")
    seen: set[str] = set()
    dests: list[str] = []
    for name, experiment in data["experiments"].items():
        safe_path(name, f"{path} experiment")
        _require(isinstance(experiment.get("roots", {}), dict), f"{path} {name} roots: each root names its "
                                                                "members_digest")
        for root, pinned in _roots(experiment).items():
            safe_path(root, f"{path} {name} root")
            _require(pinned is None or (isinstance(pinned, str) and bool(_SHA256.fullmatch(pinned))),
                     f"{path} {name} {root}: its members_digest is not a SHA-256 digest")
            _require(root not in seen, f"{path}: {root} is listed twice")
            seen.add(root)
            dests.append(f"sampling/{root}")
        _require(isinstance(experiment.get("omit", {}), dict), f"{path} {name} omit: each root names its omit list")
        for key in ("all_attempts", "raw_outputs", "omit", "profiles"):
            stray = sorted(set(experiment.get(key, [])) - set(_roots(experiment)))
            _require(not stray, f"{path} {name} {key}: {stray} are not its roots")
        unnamed = sorted(root for root in _roots(experiment) if (experiment.get("profiles") or {}).get(
            root, experiment.get("profile")) not in PROFILES)
        _require(not unnamed, f"{path} {name}: {unnamed} name no profile of {sorted(PROFILES)} (profile, profiles)")
        for root, patterns in experiment.get("omit", {}).items():
            _omit_list(patterns, f"{path} {name} {root}")
        for item in experiment.get("files", []):
            for key in ("path", "dest"):
                safe_path(item.get(key), f"{path} {name} {key}")
            _require(item.get("kind") in FILE_KINDS, f"{path} {name} {item['path']}: kind is selection or input")
            archive = item.get("format", FILE) == ARCHIVE
            _require(archive == ("prefix" in item),
                     f"{path} {name} {item['path']}: an archive, and only one, names its prefix")
            _require(not set(item) & ({"sha256"} if archive else {"members_digest", "omit"}),
                     f"{path} {name} {item['path']}: an archive is pinned by members_digest, a plain file by sha256 "
                     "(and only an archive has an omit list)")
            if "omit" in item:
                _omit_list(item["omit"], f"{path} {name} {item['path']}")
            dests.append(item["dest"])
    for model in data.get("models", []):
        _require(isinstance(model.get("base_model", ""), str), f"{path} model {model.get('dir')}: base_model is the "
                                                               "base model's Hub id")
        safe_path(model.get("dir"), f"{path} model dir")
        safe_path(model.get("dest"), f"{path} model dest")
        dests += [f"{model['dest']}/{safe_path(name, f'{path} model file')}" for name in model.get("files", {})]
    clashes = overlapping(dests)
    _require(not clashes, f"{path}: destinations {clashes[:3]} overlap")
    return data


def _rule(inventory: Mapping[str, Any], experiment: Mapping[str, Any], root: str | None = None) -> CompactRule:
    compact = inventory.get("compact", {})
    return CompactRule(tuple(compact.get("attempts_conditions", ("forced*",))),
                       bool(experiment.get("draw0_evaluations", False)), root in experiment.get("all_attempts", []),
                       root in experiment.get("raw_outputs", []))


def _roots_by_profile(inventory: Mapping[str, Any]) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for experiment in inventory["experiments"].values():
        for root in _roots(experiment):
            by.setdefault((experiment.get("profiles") or {}).get(root, experiment.get("profile")), []).append(root)
    return by


def compact_text(inventory: Mapping[str, Any]) -> str:
    """The compact rule as the index states it."""
    conditions = ", ".join(_rule(inventory, {}).attempts_conditions)
    draw0 = sorted(name for name, experiment in inventory["experiments"].items()
                   if experiment.get("draw0_evaluations"))
    def listed(key: str) -> list[str]:
        return sorted(root for experiment in inventory["experiments"].values() for root in experiment.get(key, []))
    every, raw = listed("all_attempts"), listed("raw_outputs")
    return (f"every file of a campaign root but its run records ({', '.join(RUN_RECORDS)}) and the per-draw "
            f"{', '.join(DRAW_TEXT)}, except attempts.jsonl of the "
            f"conditions {conditions} (each condition directory matched by its public style name)"
            + (f", every attempts.jsonl of {', '.join(every)}" if every else "")
            + (f", raw_outputs.jsonl of {', '.join(raw)}" if raw else "")
            + (f" and draw-000/evaluations.jsonl in {', '.join(draw0)}" if draw0 else "")
            + (f"; every root also leaves out {', '.join(_every_root(inventory))}" if _every_root(inventory) else "")
            + f"; a root publishes each attempts.jsonl it keeps as {PUBLISHED_LEDGER_FNAME}, its strict replay's "
              "accepted scenarios, each with the full-ruleset call its bundle holds, and of every attempt the first and "
              f"last start time and the stacks the rows name, and of each {RAW_OUTPUTS} it keeps only the rows of the "
              "draw's accepted full-ruleset calls, by that replay"
            + "; " + "; ".join(
                f"a {profile_name} root ({', '.join(roots)}) keeps "
                + ", ".join(f"of each {name} {', '.join(projection.fields)}"
                            + "".join(f" of the rows whose {key} is {value}" for key, value in projection.row)
                            for name, projection in PROFILES[profile_name].projections.items())
                + (f", of each ledger's form its {', '.join(PROFILES[profile_name].ledger)}"
                   if PROFILES[profile_name].ledger else "")
                for profile_name, roots in _roots_by_profile(inventory).items())
            + "; " + ", ".join(f"{path} of {name} ({rule})" for name, path, rule in PUBLISHED_FIELDS)
            + f" are published, in every root, input archive and file, as the path's last segment (name), "
              f"{HOST_PLACEHOLDER!r} (host), an artifact:// URI from the path's first artifact-root directory "
              "(location), the path from its data/ segment (repo), the published SHA-256 of a file the package "
              "rewrites (digest) and the adapter's base model as the Hub names it (hub)")


def _write_published(source: Path, target: Path, where: str, publication: Publication) -> None:
    """Write ``source`` at ``target`` as :func:`published_record` publishes it, held to :func:`refuse_private`."""
    if source.name in {name for name, _, _ in PUBLISHED_FIELDS} or source.name.endswith((".json", ".jsonl")):
        data = source.read_bytes()
        published = published_record(source.name, data, where, publication)
        refuse_private(source.name, published, where, publication.private)
        target.write_bytes(published)
    else:
        shutil.copyfile(source, target)


def _members_path(path: str) -> str:
    return path.removesuffix(".tar.zst") + ".files.sha256"


def _file_entry(out: Path, path: str, kind: str, group: str, dest: str) -> Entry:
    return Entry(path=path, kind=kind, group=group, dest=dest, sha256=sha256_file(out / path),
                 bytes=(out / path).stat().st_size)


def _archive_entry(out: Path, path: str, kind: str, group: str, dest: str, prefix: str,
                   members: Mapping[str, str], root: str | None = None, omit: Sequence[str] = ()) -> Entry:
    manifest = _members_path(path)
    return Entry(path=path, kind=kind, group=group, dest=dest, sha256=sha256_file(out / path),
                 bytes=(out / path).stat().st_size, format=ARCHIVE, prefix=prefix, members=manifest,
                 members_sha256=sha256_file(out / manifest), members_digest=members_digest(members), root=root,
                 omit=tuple(omit) or None)


def _pinned(item: Mapping[str, Any], entry: Entry) -> None:
    if entry.archive:
        expected, actual = item.get("members_digest"), entry.members_digest
    else:
        expected, actual = item.get("sha256"), entry.sha256
    if expected is not None:
        _require(expected == actual, f"{entry.path}: {actual[:12]}... is not the {expected[:12]}... the inventory "
                                     "pins")


def pack(artifact_root: Path | str, inventory: Mapping[str, Any], out: Path | str, *, only: Sequence[str] = (),
         card: Path | None = None, level: int = 19, threads: int = 0, private: Sequence[str] = (),
         echo: Callable[[str], None] = print) -> Index:
    """Build the dataset repository's layout (roots, inputs, plain files, index, ``SHA256SUMS``) in ``out``, which must
    not exist and a failed pack removes, each source held to its pinned digest and published under
    :data:`PUBLISHED_FIELDS`; ``only`` names experiments or roots."""
    artifact_root, out = Path(artifact_root), Path(out)
    known = set(inventory["experiments"]) | {root for experiment in inventory["experiments"].values()
                                              for root in _roots(experiment)}
    unknown = sorted(set(only) - known)
    _require(not unknown, f"--only {', '.join(unknown)}: neither an experiment nor a root of the inventory")
    selection = [(group, experiment, [root for root in _roots(experiment) if not only or group in only or root in only],
                  experiment.get("files", []) if not only or group in only else [])
                 for group, experiment in inventory["experiments"].items()]
    missing = [f"sampling/{root} (a directory)" for _, _, roots, _ in selection for root in roots
               if not (artifact_root / "sampling" / root).is_dir()]
    for _, _, _, items in selection:
        for item in items:
            archive, source = item.get("format", FILE) == ARCHIVE, artifact_root / item["dest"]
            if not (source.is_dir() if archive else source.is_file()):
                missing.append(f"{item['dest']} ({'a directory' if archive else 'a file'})")
    _require(not missing, f"{artifact_root} does not hold what the selection packs: {'; '.join(missing[:SHOWN])}"
                          f"{f'; and {len(missing) - SHOWN} more' if len(missing) > SHOWN else ''}; pack from a root "
                          "that holds them, or name with --only what this one holds; nothing was written")
    refuse_campaign(out, "the directory the layout is written into")
    publication = Publication(digests=_published_digests(artifact_root, inventory), private=tuple(private))
    every_root = Omit(_every_root(inventory))
    with _new_layout(out):
        built = _Built(out)
        for group, experiment, roots, items in selection:
            for root in roots:
                pinned = _roots(experiment)[root]
                path, symlinks = f"{group}/roots/{root}.tar.zst", []
                rule, omit = _rule(inventory, experiment, root), Omit(_omits(experiment).get(root, ()))
                top = artifact_root / "sampling" / root
                members = pack_tree(top, f"sampling/{root}", out / path, out / _members_path(path),
                                    keeps=_named_kinds(root, every_root.keeping(omit.keeping(rule.keeps))),
                                    level=level, threads=threads, symlinks=symlinks,
                                    published=_publishing(root, publication, _profile(experiment, root), top),
                                    extra={PUBLISHED_COPY_FNAME: lambda kept, root=root: _published_copy(root, kept)},
                                    renames=LEDGER_PUBLISHED_AS)
                if symlinks:
                    echo(f"{root}: {len(symlinks)} symlink(s) left out ({', '.join(symlinks[:3])}"
                         f"{', ...' if len(symlinks) > 3 else ''})")
                omit.check(root, echo)
                entry = _archive_entry(out, path, "root", group, f"sampling/{root}", f"sampling/{root}", members, root,
                                       omit.patterns)
                _pinned({"members_digest": pinned}, entry)
                built.add(entry)
            for item in items:
                path = f"{group}/{item['path']}"
                (out / path).parent.mkdir(parents=True, exist_ok=True)
                if item.get("format", FILE) == ARCHIVE:
                    omit = Omit(item.get("omit", ()))
                    members = pack_tree(artifact_root / item["dest"], item["prefix"], out / path,
                                        out / _members_path(path), keeps=omit.keeping(lambda _: True), level=level,
                                        threads=threads, published=_publishing(item["dest"], publication))
                    omit.check(item["dest"], echo)
                    entry = _archive_entry(out, path, item["kind"], group, item["dest"], item["prefix"], members,
                                           omit=omit.patterns)
                else:
                    _write_published(artifact_root / item["dest"], out / path, item["dest"], publication)
                    entry = _file_entry(out, path, item["kind"], group, item["dest"])
                _pinned(item, entry)
                built.add(entry)
        if not only:
            every_root.check("every campaign root", echo)
        return built.finish(compact_text(inventory), card)


def pack_models(artifact_root: Path | str, inventory: Mapping[str, Any], out: Path | str, *,
                card: Path | None = None, private: Sequence[str] = ()) -> Index:
    """Build the model repository's layout in ``out``, which must not exist: each adapter file from under
    ``artifact_root``, held to its pinned SHA-256, then the index and ``SHA256SUMS``; a failed pack removes ``out``."""
    artifact_root, out = Path(artifact_root), Path(out)
    models = inventory.get("models", [])
    _require(bool(models), "the inventory names no model")
    missing = [f"{model['dest']}/{name}" for model in models for name in model["files"]
               if not (artifact_root / model["dest"] / name).is_file()]
    _require(not missing, f"{artifact_root} does not hold the adapters' files: {'; '.join(missing[:SHOWN])}; nothing "
                          "was written")
    refuse_campaign(out, "the directory the layout is written into")
    with _new_layout(out):
        built = _Built(out)
        for model in models:
            for name, pinned in model["files"].items():
                path = f"{model['dir']}/{name}"
                (out / path).parent.mkdir(parents=True, exist_ok=True)
                _write_published(artifact_root / model["dest"] / name, out / path, path,
                                 Publication(base_model=model.get("base_model"), private=tuple(private)))
                entry = _file_entry(out, path, "adapter", model["dir"], f"{model['dest']}/{name}")
                _pinned({"sha256": pinned}, entry)
                built.add(entry)
        return built.finish("", card)


def main_pack(argv: Sequence[str] | None = None, *, prog: str = "python -m ruleforge.hub") -> int:
    """Run ``python -m ruleforge.hub pack``: 0 done, 2 a refusal."""
    parser = argparse.ArgumentParser(
        prog=prog, description="Build a Hub repository's layout from an artifact root. Uploads nothing.")
    verbs = parser.add_subparsers(dest="verb", required=True)
    build = verbs.add_parser("pack", help="pack the inventory's roots and files from an artifact root")
    build.add_argument("--artifact-root", help="default: $RULEFORGE_ARTIFACT_ROOT")
    build.add_argument("--only", nargs="+", default=[], metavar="NAME",
                       help="experiments, or roots (each archive alone)")
    build.add_argument("--level", type=int, default=19, help="zstd level (default 19)")
    build.add_argument("--threads", type=int, default=0, help="zstd worker threads (default 0: single-threaded)")
    build.add_argument("--model", action="store_true",
                       help="the model repository: the inventory's adapters, from where a fetch places them")
    build.add_argument("--card", type=Path, help="the repository's card, copied as README.md")
    build.add_argument("--inventory", type=Path, help="default: the checkout's data/hub_inventory.yaml")
    build.add_argument("--private", type=Path, metavar="FILE",
                       help="names (hosts, accounts) no published file may hold outside its text fields, one a line")
    build.add_argument("--out", type=Path, required=True, help="a directory that does not exist yet")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        _require(not (args.model and args.only), "--only names the dataset's experiments or roots; --model packs every "
                                                 "adapter")
        if not args.model:
            require(f"{prog} pack", "zstandard")  # a dataset layout reads and writes .tar.zst archives
        inventory = load_inventory(args.inventory or package_path("data/hub_inventory.yaml"))
        _require(not os.path.lexists(args.out), f"{args.out} exists: the layout is written into a new directory")
        # an empty --artifact-root is refused, not replaced
        artifact_root = args.artifact_root if args.artifact_root is not None else env_value("ARTIFACT_ROOT")
        _require(bool(artifact_root), "no artifact root: pass --artifact-root or set RULEFORGE_ARTIFACT_ROOT")
        private = [line.strip() for line in args.private.read_text(encoding="utf-8").splitlines()
                   if line.strip()] if args.private else []
        if args.model:
            indexes = {"model": pack_models(artifact_root, inventory, args.out, card=args.card, private=private)}
        else:
            indexes = {"dataset": pack(artifact_root, inventory, args.out, only=args.only, card=args.card,
                                       level=args.level, threads=args.threads, private=private)}
    except (RuleForgeError, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    for name, index in indexes.items():
        size = sum(entry.bytes for entry in index.entries)
        print(f"{name}: {len(index.entries)} entries, {size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main_pack(prog="python -m ruleforge.hub"))
