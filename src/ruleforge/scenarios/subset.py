from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


from ruleforge.core.errors import RuleForgeError
from ruleforge.core.digests import sha256_file
from ruleforge.core.files import write_json_sidecar
from ruleforge.core.artifacts import load_jsonl_records, write_text_atomically
from ruleforge.scenarios.manifest import scenario_id

SUBSET_SCHEMA_VERSION = 1
SUBSET_SIDECAR_SUFFIX = ".manifest.json"
class SubsetError(ValueError, RuleForgeError):
    """Raised for a cut that cannot be made exactly as asked."""


def sidecar_path(output: Path | str) -> Path:
    """Return the path of the sidecar that describes ``output``."""
    return Path(str(output) + SUBSET_SIDECAR_SUFFIX)


def _population(source: Path) -> tuple[list[str], list[dict[str, Any]]]:
    records = load_jsonl_records(source)
    lines = [line for line in source.read_text(encoding="utf-8").split("\n") if line.strip()]
    if not records:
        raise SubsetError(f"{source} holds no scenarios")
    return lines, records


def _read_source(
    source: Path | str, expected_source_sha256: Optional[str]
) -> tuple[Path, str, list[str], list[dict[str, Any]]]:
    source_path = Path(source)
    if not source_path.is_file():
        raise SubsetError(f"source {source_path} does not exist")
    source_sha256 = sha256_file(source_path)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise SubsetError(
            f"source {source_path.name} has digest {source_sha256}, expected "
            f"{expected_source_sha256}"
        )
    lines, records = _population(source_path)
    return source_path, source_sha256, lines, records


def _refuse_overwrite(path: Path, text: str, what: str) -> None:
    if path.exists() and path.read_bytes() != text.encode("utf-8"):
        raise SubsetError(
            f"{path} exists with different content; refusing to overwrite a {what} a campaign may have frozen"
        )


SELECTION_SLICE = "contiguous rows [start, stop) of the source, in its order"
BLOCK_INFIX = ".block-"


def block_name(source: Path | str, index: int) -> str:
    """Return the file name of block ``index`` of ``source`` (``<stem>.block-003<suffix>``)."""
    path = Path(source)
    return f"{path.stem}{BLOCK_INFIX}{index:03d}{path.suffix}"


def _slice_text(lines: Sequence[str], start: int, stop: int) -> str:
    return "".join(line + "\n" for line in lines[start:stop])


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scenario_ids(records: Sequence[Mapping[str, Any]], source: Path) -> list[str]:
    ids = [scenario_id(record, f"{source.name} row {row}") for row, record in enumerate(records, start=1)]
    duplicates = sorted(sid for sid, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise SubsetError(f"{source.name} holds {len(duplicates)} duplicate scenario id(s): {duplicates[:3]}")
    return ids


def build_blocks(
    source: Path | str,
    out_dir: Path | str,
    *,
    block_size: int,
    expected_source_sha256: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Cut ``source`` into contiguous blocks of ``block_size`` rows under ``out_dir``, each with a sidecar."""
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise SubsetError(f"block_size must be a positive integer, got {block_size!r}")
    source_path, source_sha256, lines, records = _read_source(source, expected_source_sha256)
    ids = _scenario_ids(records, source_path)
    ranges = [(start, min(start + block_size, len(lines))) for start in range(0, len(lines), block_size)]
    directory = Path(out_dir)
    outputs = [directory / block_name(source_path, index) for index in range(len(ranges))]
    texts = [_slice_text(lines, start, stop) for start, stop in ranges]
    for output, text in zip(outputs, texts):
        _refuse_overwrite(output, text, "block")
    payloads = []
    for index, ((start, stop), output, text) in enumerate(zip(ranges, outputs, texts)):
        write_text_atomically(output, text)
        payload = {
            "schema_version": SUBSET_SCHEMA_VERSION,
            "selection": SELECTION_SLICE,
            "source_name": source_path.name,
            "source_sha256": source_sha256,
            "population_count": len(records),
            "count": stop - start,
            "row_range": [start, stop],
            "block": {"index": index, "of": len(ranges), "size": block_size},
            "scenario_ids": ids[start:stop],
            "output_name": output.name,
            "output_sha256": sha256_file(output),
        }
        write_json_sidecar(sidecar_path(output), payload)
        payloads.append(payload)
    return payloads


def read_slice_sidecar(path: Path | str) -> dict[str, Any]:
    """Read a block's sidecar, refusing one that does not describe a slice."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SubsetError(f"cannot read the block sidecar {source}: {error}") from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != SUBSET_SCHEMA_VERSION
        or payload.get("selection") != SELECTION_SLICE
    ):
        raise SubsetError(f"{source} is not the sidecar of a block (a contiguous slice of its source)")
    return dict(payload)


def check_slices(population: Path | str, sidecars: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check that ``sidecars`` tile ``population`` exactly, recomputing ids and digests; return the population's ids."""
    source = Path(population)
    source_sha256 = sha256_file(source)
    lines, records = _population(source)
    ids = _scenario_ids(records, source)
    ranges = []
    for sidecar in sidecars:
        name = sidecar.get("output_name")
        if sidecar.get("source_sha256") != source_sha256:
            raise SubsetError(f"block {name} is not a slice of {source.name} (it names another source digest)")
        start, stop = sidecar["row_range"]
        if not 0 <= start < stop <= len(lines):
            raise SubsetError(f"block {name}'s rows [{start}, {stop}) are not rows of {source.name} ({len(lines)})")
        if sidecar.get("scenario_ids") != ids[start:stop]:
            raise SubsetError(f"block {name}'s scenario ids are not rows [{start}, {stop}) of {source.name}")
        if _text_sha256(_slice_text(lines, start, stop)) != sidecar.get("output_sha256"):
            raise SubsetError(f"block {name}'s digest is not that of rows [{start}, {stop}) of {source.name}")
        ranges.append((start, stop, name))
    covered = 0
    for start, stop, name in sorted(ranges):
        if start != covered:
            detail = (
                f"rows [{covered}, {start}) are in no block" if start > covered
                else f"block {name} overlaps rows [{start}, {covered}) of another block"
            )
            raise SubsetError(f"the blocks do not tile {source.name}: {detail}")
        covered = stop
    if covered != len(lines):
        raise SubsetError(f"the blocks do not tile {source.name}: rows [{covered}, {len(lines)}) are in no block")
    return ids


SELECTION_CONCATENATION = "the sources' rows, each source's in its order, the sources in the order given"


def build_concatenation(
    sources: Sequence[Path | str],
    output: Path | str,
    *,
    expected_source_sha256s: Optional[Sequence[Optional[str]]] = None,
) -> dict[str, Any]:
    """Write ``sources`` one after another to ``output`` with a sidecar; a scenario in two sources is refused."""
    paths = [Path(source) for source in sources]
    if len(paths) < 2:
        raise SubsetError(f"a concatenation needs at least two sources, got {len(paths)}")
    if len({path.resolve() for path in paths}) != len(paths):
        raise SubsetError(f"a source is named twice among {[str(path) for path in paths]}")
    expected = list(expected_source_sha256s) if expected_source_sha256s is not None else [None] * len(paths)
    if len(expected) != len(paths):
        raise SubsetError(f"{len(expected)} expected digest(s) for {len(paths)} source(s): give one per source")
    read = [_read_source(path, digest) for path, digest in zip(paths, expected)]
    output_path = Path(output)
    lines = [line for _, _, source_lines, _ in read for line in source_lines]
    _scenario_ids([record for *_, records in read for record in records], output_path)
    text = _slice_text(lines, 0, len(lines))
    _refuse_overwrite(output_path, text, "concatenated population")
    write_text_atomically(output_path, text)
    entries, start = [], 0
    for path, digest, source_lines, _ in read:
        entries.append({"name": path.name, "sha256": digest, "count": len(source_lines),
                        "row_range": [start, start + len(source_lines)]})
        start += len(source_lines)
    payload = {
        "schema_version": SUBSET_SCHEMA_VERSION,
        "selection": SELECTION_CONCATENATION,
        "sources": entries,
        "population_count": len(lines),
        "output_name": output_path.name,
        "output_sha256": sha256_file(output_path),
    }
    write_json_sidecar(sidecar_path(output_path), payload)
    return payload


SELECTION_DIFFERENCE = "the source's rows whose scenario id no excluded population holds, in the source's order"


def build_difference(
    source: Path | str,
    output: Path | str,
    *,
    exclude: Sequence[Path | str],
    expected_source_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Write the rows of ``source`` no ``exclude`` population holds (by scenario id) to ``output``, with a sidecar."""
    source_path, source_sha256, lines, records = _read_source(source, expected_source_sha256)
    ids = _scenario_ids(records, source_path)
    paths = [Path(path) for path in exclude]
    if not paths:
        raise SubsetError("a difference needs at least one population to exclude")
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved) or source_path.resolve() in resolved:
        raise SubsetError(f"an excluded population is named twice, or is the source, among {[str(p) for p in paths]}")
    held = set(ids)
    removed: set[str] = set()
    excluded = []
    for path in paths:
        if not path.is_file():
            raise SubsetError(f"excluded population {path} does not exist")
        excluded_records = load_jsonl_records(path)
        if not excluded_records:
            raise SubsetError(f"excluded population {path} holds no rows: nothing carved from {source_path.name}")
        excluded_ids = set(_scenario_ids(excluded_records, path))
        strangers = sorted(excluded_ids - held)
        if strangers:
            raise SubsetError(
                f"{path.name} holds {len(strangers)} row(s) {source_path.name} does not (e.g. {strangers[0][:12]}): "
                "a difference subtracts populations carved from its source"
            )
        excluded.append({"name": path.name, "sha256": sha256_file(path), "count": len(excluded_ids),
                         "removed": len(excluded_ids - removed)})
        removed |= excluded_ids
    indices = [index for index, sid in enumerate(ids) if sid not in removed]
    if not indices:
        raise SubsetError(f"the excluded populations hold every row of {source_path.name}: nothing is left")
    output_path = Path(output)
    text = "".join(lines[index] + "\n" for index in indices)
    _refuse_overwrite(output_path, text, "population difference")
    write_text_atomically(output_path, text)
    payload = {
        "schema_version": SUBSET_SCHEMA_VERSION,
        "selection": SELECTION_DIFFERENCE,
        "source_name": source_path.name,
        "source_sha256": source_sha256,
        "population_count": len(records),
        "excluded": excluded,
        "count": len(indices),
        "indices": indices,
        "scenario_ids": [ids[index] for index in indices],
        "output_name": output_path.name,
        "output_sha256": sha256_file(output_path),
    }
    write_json_sidecar(sidecar_path(output_path), payload)
    return payload
