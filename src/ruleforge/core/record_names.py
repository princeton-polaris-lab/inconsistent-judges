from __future__ import annotations

import json
import re
from importlib import resources
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, TypeVar, Union

from ruleforge.core.errors import SamplingDesignError

TABLE_FNAME = "record_names.json"
TABLE_SCHEMA = "ruleforge.core.record_names/1"


def read_table(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the parsed name table after checking its schema, task list, style rows and retired keys."""
    try:
        rows = [(row["recorded"], row["public"], row["record_only"]) for row in payload["styles"]]
        recorded, public = [row[0] for row in rows], [row[1] for row in rows]
        names = recorded + public + list(payload["retired"])
        own_keys = [(task, name, key) for task, keys in payload["task_keys"].items() for name, key in keys.items()]
        well_formed = (payload["schema"] == TABLE_SCHEMA and payload["tasks"]
                       and all(isinstance(payload[key], list) for key in ("styles", "tasks", "retired"))
                       and all(isinstance(name, str) and name for name in [*names, *payload["tasks"]])
                       and all(isinstance(row[2], bool) for row in rows) and len(set(names)) == len(names)
                       and all(task in payload["tasks"] and name in public and key in recorded
                               for task, name, key in own_keys))
    except (AttributeError, KeyError, TypeError) as error:
        raise SamplingDesignError(f"{TABLE_FNAME} is malformed: {error!r}") from error
    if not well_formed:
        raise SamplingDesignError(
            f"{TABLE_FNAME} is malformed: it must declare schema {TABLE_SCHEMA!r}, the built-in tasks, style rows and "
            "retired keys that are non-empty names, each named once, with a record_only flag per row, and task keys "
            "that name a built-in task, a public name and a recorded key")
    return {
        "tasks": tuple(payload["tasks"]),
        "task_keys": payload["task_keys"],
        "keys": {public: key for key, public, _ in rows},
        "order": tuple(public),
        "record_only": tuple(public for _, public, only in rows if only),
        "retired": frozenset(payload["retired"]),
    }


_TABLE = read_table(json.loads(resources.files(__package__).joinpath(TABLE_FNAME).read_text(encoding="utf-8")))

# The built-in tasks the table applies to; a task of one's own names styles of its own, hashed as written.
IDENTITY_TASKS: tuple[str, ...] = _TABLE["tasks"]
# task -> public style -> recorded key, for a task's prompt whose key is not the one the rows give its name.
TASK_KEYS: Mapping[str, Mapping[str, str]] = MappingProxyType(_TABLE["task_keys"])
# public style -> recorded key, and back
IDENTITY_KEYS: Mapping[str, str] = MappingProxyType(_TABLE["keys"])
PUBLIC_NAMES: Mapping[str, str] = MappingProxyType({key: style for style, key in IDENTITY_KEYS.items()})
ORDER: tuple[str, ...] = _TABLE["order"]
# Styles read from the records alone, never selected for a new run.
RECORD_ONLY: tuple[str, ...] = _TABLE["record_only"]
RECORD_KEYS: frozenset[str] = frozenset(PUBLIC_NAMES)
# Keys of styles with no public name: never read, never written.
RETIRED_KEYS: frozenset[str] = _TABLE["retired"]
# Keys of the same-call statistics inside the published estimates files; changing one breaks reading them.
SEPARATE_CALL_KEY = "method_a"
SAME_CALL_KEY = "method_b"
SAME_CALL_RATE_KEY = "method_b_rate"
SEPARATE_VS_SAME_CALL_KEY = "method_a_vs_b"
# Launches recorded under this directory name stay readable, so a live one still guards its outputs.
RECORDED_LAUNCHES_DIRNAME = ".smrc-launch"
# The two files every ruleset-size root holds, as its published runs named them; changing one breaks reading them.
SHARED_BANK_DESIGN_FNAME = "figure2_design.json"
SHARED_BANK_ESTIMATES_FNAME = "figure2_estimates.json"
# Each experiment block's config key as the published runs' saved configs spell it. The RuleForge and training
# conditions' ids and condition_extra.figure hold the same word; their configs read it as ${rf.recorded:<block>}.
RECORDED_BLOCK_KEYS: Mapping[str, str] = MappingProxyType({"ruleset_size": "figure2", "ruleforge": "figure4",
                                                           "training": "figure5"})
# Strings inside the published data, kept in what new runs write so they read alike; changing one breaks reading them.
PER_SCENARIO_FNAME = "smrc_per_scenario.jsonl"
SUMMARY_FNAME = "smrc_summary.json"
PROVENANCE_FNAME = "smrc_provenance.json"
WEIGHTS_DIGEST_VERSION = "smrc-weights-sha256-v1"
ANCHOR_REPLAY_SEED_LABEL = "smrc-anchor-replay"
ROWS_DERIVATION = "smrc.finetuning.grpo_data/v1"
STOPPED_KIND = "smrc-grpo-early-stopped"
WANDB_RUN_KIND = "smrc-grpo-wandb-run"
PARTIAL_KIND = "smrc-partial-batch"
BATCH_METADATA_SOURCE = "smrc"
_BLOCK_TAIL = re.compile(r"draw-\d+/attempt-\d+/block-\d+(?:@[^/]+)?")

Condition = TypeVar("Condition")


def _maps(task: Optional[str]) -> bool:
    return task is None or task in IDENTITY_TASKS


def _replace_segments(condition_id: str, old: str, new: str) -> str:
    if old == new:
        return condition_id
    return "/".join(new if segment == old else segment for segment in condition_id.split("/"))


def identity_key(style: str, task: Optional[str] = None) -> str:
    """Return the key a condition of ``style`` is hashed under: its recorded key for a built-in task, else as
    written."""
    if not _maps(task):
        return style
    return TASK_KEYS.get(task, {}).get(style, IDENTITY_KEYS.get(style, style))


def public_style(style: str, task: Optional[str] = None) -> str:
    """Return the public name of ``style``: a built-in task's recorded key maps to its public name, else as written."""
    if not _maps(task):
        return style
    own = {key: public for public, key in TASK_KEYS.get(task, {}).items()}
    return own.get(style, PUBLIC_NAMES.get(style, style))


def identity_condition_id(condition_id: str, prompt_style: str, task: Optional[str] = None) -> str:
    """Return the id seeds, block ids and draw grids are hashed over: the condition id with its own style's segment
    keyed."""
    return _replace_segments(condition_id, prompt_style, identity_key(prompt_style, task))


def public_condition_id(condition_id: str, prompt_style: str, task: Optional[str] = None) -> str:
    """Return the id a golden names a condition by: the condition id with its own style's recorded key made public."""
    style = public_style(prompt_style, task)
    return _replace_segments(condition_id, identity_key(style, task), style)


def identity_block_id(
    block_id: str, *, campaign_id: str, condition_id: str, prompt_style: str, task: Optional[str] = None,
) -> str:
    """Return the id a provider namespace is taken over: ``block_id`` with its condition id replaced by the identity
    id."""
    prefix = f"{campaign_id}/{condition_id}/"
    tail = block_id[len(prefix):]
    if not (block_id.startswith(prefix) and _BLOCK_TAIL.fullmatch(tail)):
        raise SamplingDesignError(
            f"request block {block_id!r} is not a block of condition {condition_id!r} in campaign {campaign_id!r}")
    return f"{campaign_id}/{identity_condition_id(condition_id, prompt_style, task)}/{tail}"


def record_key_segments(condition_id: str, prompt_style: Optional[str] = None, task: Optional[str] = None,
                        ) -> tuple[str, ...]:
    """Return the segments of ``condition_id`` that are recorded keys other than the condition's own style's."""
    own = identity_key(public_style(prompt_style, task), task) if prompt_style is not None and _maps(task) else None
    return tuple(segment for segment in condition_id.split("/") if segment in RECORD_KEYS and segment != own)


def refuse_record_keys(condition_id: str) -> None:
    """Raise ``SamplingDesignError`` when a new condition id holds a recorded or retired key as a segment."""
    held = [segment for segment in condition_id.split("/") if segment in RECORD_KEYS or segment in RETIRED_KEYS]
    if held:
        named = ", ".join(f"{key!r} (the records' key of {PUBLIC_NAMES[key]!r})" if key in PUBLIC_NAMES
                          else f"{key!r} (a retired style's key)" for key in held)
        raise SamplingDesignError(
            f"condition id {condition_id!r} holds {named} as a segment: a key the published records name a style by "
            "names a recorded condition alone; name a new condition by its style's public name")


def is_retired(style: Any, task: Optional[str] = None) -> bool:
    """Return whether ``style`` is a retired style's key of a built-in task: a condition recorded under it is not
    read."""
    return _maps(task) and style in RETIRED_KEYS


def public_segment(segment: str) -> str:
    """Return a path segment under its public name; a segment that is no recorded key is returned as written."""
    return PUBLIC_NAMES.get(segment, segment)


def renamed_to(key: str) -> Optional[str]:
    """Return the public name of the style recorded under ``key``, or None when ``key`` is no recorded key."""
    return PUBLIC_NAMES.get(key)


def recorded_block_key(block: str) -> str:
    """Return the key the published runs' saved configs name experiment block ``block`` by."""
    try:
        return RECORDED_BLOCK_KEYS[block]
    except KeyError:
        raise SamplingDesignError(f"{block!r} is no experiment block with a recorded key (the blocks: "
                                  f"{', '.join(RECORDED_BLOCK_KEYS)})") from None


def public_blocks(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``config`` with each experiment block it names by its recorded key under today's key, and every reference
    to the block with it; a config naming one block under both keys is refused."""
    held = {recorded: public for public, recorded in RECORDED_BLOCK_KEYS.items() if recorded in config}
    both = [f"{recorded!r} and {public!r}" for recorded, public in held.items() if public in config]
    if both:
        raise SamplingDesignError(f"the config holds {'; '.join(both)}: one experiment block under its recorded and "
                                  "its current key; keep one")

    def moved(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: moved(item) for key, item in value.items()}
        if isinstance(value, list):
            return [moved(item) for item in value]
        if isinstance(value, str):
            for recorded, public in held.items():
                value = value.replace("${" + recorded + ".", "${" + public + ".").replace(
                    "${" + recorded + "}", "${" + public + "}")
        return value

    return {held.get(key, key): moved(value) for key, value in config.items()} if held else dict(config)


def locate_condition(conditions: Union[Mapping[str, Condition], Iterable[Condition]], public_id: str) -> Condition:
    """Return the one condition whose public id is ``public_id``; raise when none or several match."""
    pool = list(conditions.values() if isinstance(conditions, Mapping) else conditions)
    found = [condition for condition in pool if condition.public_id == public_id]
    if len(found) != 1:
        held = sorted(condition.condition_id for condition in (found or pool))
        raise SamplingDesignError(
            f"no registered condition has the public id {public_id!r} (registered: {held})" if not found else
            f"{len(found)} registered conditions have the public id {public_id!r} ({held}): a root registers one "
            "condition per public id")
    return found[0]
