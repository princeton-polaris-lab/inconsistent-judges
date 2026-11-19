from __future__ import annotations

import fcntl
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from ruleforge.evaluate.identity import RUNS_FNAME, run_log_damage, run_log_problem
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.resume import count_complete_jsonl_records

SHARDS_SUFFIX = ".shards"
CAMPAIGN_LOCK_SUFFIX = ".lock"
SHARD_RECORD_FNAME = "shard.json"
SHARD_DIR_PATTERN = re.compile(r"^shard-(\d+)-of-(\d+)$")
SHARD_BLOCK_SEPARATOR = "@"

Unit = tuple[int, int]
# (commit, dirty, digest)
CodeState = tuple[Optional[str], Optional[bool], Optional[str]]


class ShardError(SamplingDesignError):
    """Raised when a shard layout is malformed or cannot be merged."""


def blocks_per_draw(record_count: int, scenarios_per_request_block: Optional[int]) -> int:
    """How many first-attempt request blocks one draw of ``record_count`` scenarios holds."""
    if not scenarios_per_request_block:
        return 1
    return -(-record_count // scenarios_per_request_block)


def block_members(
    scenario_ids: Sequence[str], scenarios_per_request_block: Optional[int], block_index: int
) -> tuple[str, ...]:
    """Return the manifest scenarios of first-attempt request block ``block_index``, in manifest order."""
    if not scenarios_per_request_block:
        return tuple(scenario_ids)
    start = block_index * scenarios_per_request_block
    return tuple(scenario_ids[start : start + scenarios_per_request_block])


def assign_units(draw_ids: Sequence[int], n_blocks: int, num_shards: int) -> list[tuple[Unit, ...]]:
    """Return every shard's ``(draw, block)`` units, cut into contiguous near-equal runs; more shards than units is
    refused."""
    units = [(int(draw), block) for draw in draw_ids for block in range(n_blocks)]
    if num_shards > len(units):
        raise ShardError(
            f"{num_shards} shards for {len(units)} unit(s) ({len(draw_ids)} draw(s) x {n_blocks} request "
            "block(s)); a shard must hold at least one unit"
        )
    base, extra = divmod(len(units), num_shards)
    shards: list[tuple[Unit, ...]] = []
    start = 0
    for index in range(num_shards):
        size = base + (1 if index < extra else 0)
        shards.append(tuple(units[start : start + size]))
        start += size
    return shards


def shard_dirname(index: int, count: int) -> str:
    """``shard-03-of-20``: the index zero-padded to the count's width."""
    width = len(str(count))
    return f"shard-{index:0{width}d}-of-{count}"


def parse_shard_dirname(name: str) -> Optional[tuple[int, int]]:
    """``(index, count)`` of a :func:`shard_dirname`, or None for any other name."""
    match = SHARD_DIR_PATTERN.match(name)
    if match is None:
        return None
    index, count = int(match.group(1)), int(match.group(2))
    return (index, count) if shard_dirname(index, count) == name else None


def shards_root(campaign_root: Path | str) -> Path:
    """``<campaign_root>.shards``: where a campaign's shards write, beside the root."""
    root = Path(campaign_root)
    return root.with_name(root.name + SHARDS_SUFFIX)


def shard_root(campaign_root: Path | str, condition_id: str, index: int, count: int) -> Path:
    """The root shard ``index`` of ``count`` of ``condition_id`` writes."""
    return shards_root(campaign_root) / condition_id / shard_dirname(index, count)


def campaign_lock_path(campaign_root: Path | str) -> Path:
    """``.<root name>.lock`` beside the campaign root (resolved, so every spelling of the root takes one lock)."""
    root = Path(campaign_root).resolve()
    return root.with_name(f".{root.name}{CAMPAIGN_LOCK_SUFFIX}")


@contextmanager
def campaign_lock(campaign_root: Path | str) -> Iterator[Path]:
    """Hold the campaign's exclusive lock from reading its roots to writing, or refuse at once. Shard merges, amendment
    records and analyses take it; jobs and readers do not. The holder must not take it twice."""
    path = campaign_lock_path(campaign_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Never through a link: a planted symlink would create its target.
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666)
    except OSError as error:
        raise SamplingDesignError(
            f"{path}, the campaign's lock, cannot be opened as a file of its own ({error}); a symlink there is "
            "never followed: remove it"
        ) from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SamplingDesignError(
                f"another process holds {path}, the campaign's lock: a shard merge, a cap amendment's record or an "
                "analysis of this campaign is running. Run this once it has ended"
            ) from None
        try:
            yield path
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def shard_records(campaign_root: Path | str) -> list[Path]:
    """Return every shard root's ``shard.json`` beside a campaign root, in path order."""
    return sorted(shards_root(campaign_root).glob(f"**/{SHARD_RECORD_FNAME}"))


def job_root_override(index: int, count: int) -> str:
    """The Hydra override that points a shard job's ``job_root`` at its root, resolved by the config itself."""
    return f"job_root=${{campaign_root}}{SHARDS_SUFFIX}/${{condition_id}}/{shard_dirname(index, count)}"


def shard_block_id(block_id: str, index: int, count: int) -> str:
    """The id of a retry block a shard plans over its own scenarios (``<block id>@shard-03-of-20``)."""
    return f"{block_id}{SHARD_BLOCK_SEPARATOR}{shard_dirname(index, count)}"


def code_state_key(state: Mapping[str, Any]) -> CodeState:
    """``(code_revision, code_dirty, code_state_sha256)`` of a ``code_state()`` or a ``runs.jsonl`` row."""
    return (state.get("code_revision"), state.get("code_dirty"), state.get("code_state_sha256"))


def describe_code_state(state: CodeState) -> str:
    """Return a code state as its abbreviated commit, a dirty marker and its abbreviated source digest."""
    revision, dirty, digest = state
    return f"commit {str(revision)[:12]}{' (dirty)' if dirty else ''}, sources {str(digest)[:12]}"


def recorded_code_states(root: Path | str, condition_id: str) -> dict[CodeState, list[str]]:
    """Return ``{state: [draw-NNN, ...]}``, the code states recorded for a condition; a damaged run log is refused."""
    found: dict[CodeState, list[str]] = {}
    for draw_dir in sorted((Path(root) / condition_id).glob("draw-*")):
        damage = run_log_damage(draw_dir)
        cut = run_log_problem(draw_dir) or (damage[0] if damage and not damage[1] else None)
        if cut:
            raise ShardError(f"{draw_dir}: {cut}; the code states its runs recorded cannot be read from it")
        runs = draw_dir / RUNS_FNAME
        count = count_complete_jsonl_records(runs)
        if not count:
            continue
        with runs.open("r", encoding="utf-8") as handle:
            for _, line in zip(range(count), handle):
                row = json.loads(line)
                if "code_state_sha256" in row:
                    draws = found.setdefault(code_state_key(row), [])
                    if draw_dir.name not in draws:
                        draws.append(draw_dir.name)
    return found


def layout_code_states(
    roots: Iterable[tuple[str, Path | str]], condition_id: str
) -> dict[CodeState, list[tuple[str, str]]]:
    """Return ``{state: [(shard label, draw-NNN), ...]}`` over the shard roots of one layout."""
    found: dict[CodeState, list[tuple[str, str]]] = {}
    for label, root in roots:
        for state, draws in recorded_code_states(root, condition_id).items():
            found.setdefault(state, []).extend((label, draw) for draw in draws)
    return found


def _check_index(index: Any, count: Any) -> None:
    for name, value in (("index", index), ("count", count)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ShardError(f"shard {name} must be an integer, got {value!r}")
    if count < 1 or not 0 <= index < count:
        raise ShardError(f"shard index {index} of {count}: the index must be in [0, count) and the count at least 1")


@dataclass(frozen=True)
class DrawShard:
    """The part of one draw a shard runs: its first-attempt request blocks."""

    index: int
    count: int
    blocks: tuple[int, ...]

    def __post_init__(self) -> None:
        _check_index(self.index, self.count)
        if not self.blocks or list(self.blocks) != sorted(set(self.blocks)) or self.blocks[0] < 0:
            raise ShardError(f"a draw shard's blocks must be distinct, ascending and non-negative: {self.blocks}")

    @property
    def label(self) -> str:
        return shard_dirname(self.index, self.count)

    def scenario_ids(self, manifest_ids: Sequence[str], scenarios_per_request_block: Optional[int]) -> tuple[str, ...]:
        """This shard's scenarios of the draw, in manifest order; refuses a block past the manifest."""
        n_blocks = blocks_per_draw(len(manifest_ids), scenarios_per_request_block)
        if self.blocks[-1] >= n_blocks:
            raise ShardError(
                f"{self.label} names request block {self.blocks[-1]}, but a draw of {len(manifest_ids)} "
                f"scenarios holds {n_blocks}"
            )
        return tuple(
            sid for block in self.blocks for sid in block_members(manifest_ids, scenarios_per_request_block, block)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "count": self.count, "request_blocks": list(self.blocks)}


@dataclass(frozen=True)
class ShardAssignment:
    """What one shard of a condition runs: its units among the plan's draws."""

    index: int
    count: int
    draw_ids: tuple[int, ...]
    blocks_per_draw: int
    units: tuple[Unit, ...]

    @classmethod
    def build(cls, index: int, count: int, draw_ids: Sequence[int], n_blocks: int) -> "ShardAssignment":
        _check_index(index, count)
        return cls(
            index=index,
            count=count,
            draw_ids=tuple(int(draw) for draw in draw_ids),
            blocks_per_draw=n_blocks,
            units=assign_units(draw_ids, n_blocks, count)[index],
        )

    @property
    def label(self) -> str:
        return shard_dirname(self.index, self.count)

    @property
    def draws(self) -> tuple[int, ...]:
        """The draws this shard holds units of, ascending."""
        return tuple(sorted({draw for draw, _ in self.units}))

    def draw_shard(self, draw: int) -> DrawShard:
        return DrawShard(self.index, self.count, tuple(block for d, block in self.units if d == draw))

    def record(self, *, campaign_id: str, condition_id: str, condition_fingerprint: str) -> dict[str, Any]:
        """The shard root's :data:`SHARD_RECORD_FNAME` payload."""
        return {
            "campaign_id": campaign_id,
            "condition_id": condition_id,
            "condition_fingerprint": condition_fingerprint,
            "shard_index": self.index,
            "num_shards": self.count,
            "plan_draw_ids": list(self.draw_ids),
            "blocks_per_draw": self.blocks_per_draw,
            "units": [list(unit) for unit in self.units],
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any]) -> "ShardAssignment":
        try:
            assignment = cls(
                index=payload["shard_index"],
                count=payload["num_shards"],
                draw_ids=tuple(int(draw) for draw in payload["plan_draw_ids"]),
                blocks_per_draw=int(payload["blocks_per_draw"]),
                units=tuple((int(draw), int(block)) for draw, block in payload["units"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ShardError(f"malformed shard record: {error!r}") from error
        _check_index(assignment.index, assignment.count)
        return assignment
