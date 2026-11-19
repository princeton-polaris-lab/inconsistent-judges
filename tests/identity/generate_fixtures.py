from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

if __package__ in (None, ""):  # pragma: no cover - run as a file rather than with -m
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ruleforge.evaluate.campaign import registers_retired
from tests.identity import published as pub

#: The population every root froze (``scenario_source_sha256``), committed in the repository.
POPULATION = pub.REPO_ROOT / "data" / "populations" / "frozen_500_r3.jsonl"
#: Scenarios the prompt table renders: the shortest of the population, so the fixture stays small.
PROMPT_SCENARIOS = 6
#: Prompt rows per (style, rule context, judge type).
PROMPTS_PER_COMBO = 50
SEED_ROWS = 100
BLOCK_ROWS = 20
SETTLEMENT_PER_CATEGORY = 2
SETTLEMENT_CANDIDATES = 8
#: The bytes of raw rows, as stored, a settlement row may carry.
SETTLEMENT_MAX_RAW_BYTES = 120_000
BATCH_MAX_INPUT_BYTES = 1_500_000
BATCH_PROVIDERS = ("openai_batch", "anthropic_batch")
#: What closes a thinking block in the raw output of the judges ``strict_think`` serves.
THINK_CLOSE = "</think>"


def _source(artifact_root: Path, path: Path, line: int, data: bytes) -> dict[str, Any]:
    return {"path": path.relative_to(artifact_root).as_posix(), "line": line, "sha256": pub.sha256_bytes(data)}


def _conditions(artifact_root: Path, roots: Iterable[str]) -> list[tuple[str, Path, dict[str, Any]]]:
    found = []
    for root in roots:
        for path in sorted((artifact_root / "sampling" / root).glob("**/condition.json")):
            payload = json.loads(path.read_bytes())
            if not registers_retired(payload, path, artifact_root / "sampling" / root):  # as load_conditions reads it
                found.append((root, path, payload))
    return found


def _is_subset(payload: Mapping[str, Any]) -> bool:
    """A ruleset-size subset condition: its ruleset is a generated subset file, not one the repository commits."""
    return (payload.get("extra") or {}).get("role") == "subset"


def _draws(condition_json: Path) -> list[Path]:
    return sorted(path for path in condition_json.parent.glob("draw-*") if (path / "attempts.jsonl").is_file())


def condition_rows(artifact_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"kind": "header", "generator": "tests/identity/generate_fixtures.py",
                                   "roots": list(pub.ROOTS), "schema": 1}]
    scenario_manifests: dict[str, dict[str, Any]] = {}
    for root in pub.ROOTS:
        base = artifact_root / "sampling" / root
        manifest = (base / "sampling_manifest.json").read_bytes()
        scenario = (base / "scenario_manifest.json").read_bytes()
        scenario_sha = pub.sha256_bytes(scenario)
        scenario_manifests.setdefault(scenario_sha, json.loads(scenario))
        rows.append({"kind": "sampling_manifest", "root": root, "sha256": pub.sha256_bytes(manifest),
                     "payload": json.loads(manifest), "scenario_manifest_sha256": scenario_sha})
    for root, path, payload in _conditions(artifact_root, pub.ROOTS):
        rows.append({"kind": "condition", "root": root,
                     "path": path.relative_to(artifact_root / "sampling" / root).as_posix(),
                     "sha256": pub.sha256_bytes(path.read_bytes()), "payload": payload})
    rows += [{"kind": "scenario_manifest", "sha256": sha, "payload": payload}
             for sha, payload in sorted(scenario_manifests.items())]
    return rows


def _sanitized(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """An attempt row with each component's batch work directory reference reduced to the directory's name."""
    row = json.loads(json.dumps(attempt))
    for component in row["components"]:
        ref = component.get("request_block_ref")
        if isinstance(ref, str) and "/" in ref:
            component["request_block_ref"] = ref.rstrip("/").rsplit("/", 1)[-1]
    return row


def _scan_lines(path: Path) -> Iterable[tuple[int, bytes, dict[str, Any]]]:
    for number, line in pub.numbered_lines(path):
        yield number, line, json.loads(line)


def _seed_row(row: Mapping[str, Any], component: Mapping[str, Any], generation_seed_base: Any,
              source: Mapping[str, Any]) -> dict[str, Any]:
    """A seed-table row: an attempt row's tuple, one component's recorded seed and the campaign's seed base."""
    return {
        "kind": "seed", "campaign_id": row["campaign_id"], "condition_id": row["condition_id"],
        "scenario_id": row["scenario_id"], "decoder_draw_id": row["decoder_draw_id"],
        "semantic_attempt_id": row["semantic_attempt_id"], "component_id": component["component_id"],
        "generation_seed_base": generation_seed_base, "generation_seed": component["generation_seed"],
        "source": dict(source),
    }


def _block_row(row: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """A block-table row: an attempt row's block id and the provider custom id its first component carried."""
    return {
        "kind": "block", "campaign_id": row["campaign_id"], "condition_id": row["condition_id"],
        "decoder_draw_id": row["decoder_draw_id"], "semantic_attempt_id": row["semantic_attempt_id"],
        "request_block_id": row["request_block_id"],
        "provider_request_ref": row["components"][0].get("provider_request_ref"), "source": dict(source),
    }


def _prompt_row(root: str, payload: Mapping[str, Any], row: Mapping[str, Any], component: Mapping[str, Any],
                source: Mapping[str, Any]) -> dict[str, Any]:
    """A prompt-table row: one component's recorded prompt digest and what its condition renders it from."""
    return {
        "kind": "prompt", "prompt_style": payload["prompt_style"],
        "individual_rule_context": payload["individual_rule_context"], "judge_type": payload["judge_type"],
        "root": root, "condition_id": payload["condition_id"], "rule_set_name": payload["rule_set_name"],
        "rule_set_sha256": payload["rule_set_sha256"],
        # A superiors condition's levels file, by digest (no other condition records one, and no row of its carries it).
        **({"rule_levels_sha256": payload["rule_levels_sha256"]} if payload.get("rule_levels_sha256") else {}),
        "scenario_id": row["scenario_id"], "component_id": component["component_id"],
        "prompt_sha256": component["prompt_sha256"], "source": dict(source),
    }


#: The condition fields a settlement row carries (what its validity schema is built from, and its category's).
SETTLEMENT_CONDITION_FIELDS = (
    "prompt_style", "individual_rule_context", "forced_json_requirement", "label_rule", "evaluation_mode",
    "strict_think", "retry_scope", "rule_set_name", "rule_set_sha256", "model_type",
)


def _attempt_row(root: str, payload: Mapping[str, Any], row: Mapping[str, Any], bundle_size: int,
                 source: Mapping[str, Any], raws: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "attempt", "root": root, "condition_id": payload["condition_id"],
        "condition": {name: payload.get(name) for name in SETTLEMENT_CONDITION_FIELDS},
        "category": json.loads(json.dumps(_category(payload, row, bundle_size))), "attempt": _sanitized(row),
        "source": dict(source), "raw_path": (Path(source["path"]).parent / "raw_outputs.jsonl").as_posix(),
        "raw": raws,
    }


def seed_rows(artifact_root: Path, conditions: list[tuple[str, Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Recorded seeds of local judges (the only judges sent one) and recorded block ids of batch judges."""
    manifests = {root: json.loads((artifact_root / "sampling" / root / "sampling_manifest.json").read_bytes())
                 for root in {root for root, _, _ in conditions}}
    seeds: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    subsets_taken: dict[str, int] = defaultdict(int)
    for root, path, payload in conditions:
        if _is_subset(payload):
            subsets_taken[root] += 1
            if subsets_taken[root] > 2:
                continue
        local = payload["seed_capability"] == "local_deterministic"
        draws = _draws(path)
        for draw_dir in ([draws[0], draws[-1]] if len(draws) > 1 else draws):
            attempts = draw_dir / "attempts.jsonl"
            first = later = None
            for number, line, row in _scan_lines(attempts):
                if first is None and row["semantic_attempt_id"] == 0:
                    first = (number, line, row)
                if later is None and row["semantic_attempt_id"] > 0:
                    later = (number, line, row)
                if first is not None and later is not None:
                    break
            for found, positions in ((first, (0, -1)), (later, (0,))):
                if found is None:
                    continue
                number, line, row = found
                source = _source(artifact_root, attempts, number, line)
                if local:
                    for position in sorted({position % len(row["components"]) for position in positions}):
                        seeds.append(_seed_row(row, row["components"][position],
                                               manifests[root]["generation_seed_base"], source))
                else:
                    ref = row["components"][0].get("provider_request_ref")
                    if isinstance(ref, str) and ref.startswith("ns"):
                        blocks.append(_block_row(row, source))
    return _spread(seeds, SEED_ROWS) + _spread(blocks, BLOCK_ROWS)


def _spread(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    return [rows[(index * len(rows)) // count] for index in range(count)]


def _population() -> dict[str, dict[str, Any]]:
    from ruleforge.core.artifacts import compute_scenario_id, iter_jsonl_records

    return {compute_scenario_id(row): dict(row) for row in iter_jsonl_records(POPULATION)}


def prompt_rows(artifact_root: Path, conditions: list[tuple[str, Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Recorded component prompt digests over a few short scenarios, at most fifty per style, context and judge."""
    population = _population()
    chosen = sorted(population, key=lambda sid: (len(population[sid].get("prompt", "")) +
                                                  len(population[sid].get("response", "")), sid))[:PROMPT_SCENARIOS]
    wanted = set(chosen)
    rulesets = pub.repo_rule_files()
    combos: dict[tuple[str, str, str], list[tuple[str, Path, dict[str, Any]]]] = defaultdict(list)
    for root, path, payload in conditions:
        if (payload["rule_set_name"], payload["rule_set_sha256"]) in rulesets:
            combos[(payload["prompt_style"], payload["individual_rule_context"], payload["judge_type"])].append(
                (root, path, payload))
    rows: list[dict[str, Any]] = []
    for combo, members in sorted(combos.items()):
        per_condition: list[list[dict[str, Any]]] = []
        for root, path, payload in members:
            entries: list[dict[str, Any]] = []
            draws = _draws(path)
            attempts = draws[0] / "attempts.jsonl"
            seen: set[str] = set()
            for number, line, row in _scan_lines(attempts):
                if row["scenario_id"] not in wanted or row["scenario_id"] in seen:
                    continue
                seen.add(row["scenario_id"])
                components = row["components"]
                for position in dict.fromkeys((0, 1, len(components) // 2, len(components) - 1)):
                    if position >= len(components):
                        continue
                    entries.append(_prompt_row(root, payload, row, components[position],
                                               _source(artifact_root, attempts, number, line)))
                if len(seen) == len(wanted):
                    break
            per_condition.append(entries)
        taken: list[dict[str, Any]] = []
        for index in range(max((len(entries) for entries in per_condition), default=0)):
            for entries in per_condition:
                if index < len(entries) and len(taken) < PROMPTS_PER_COMBO:
                    taken.append(entries[index])
        rows += taken
    used = sorted({row["scenario_id"] for row in rows})
    return [{"kind": "scenario", "scenario_id": sid, "row": population[sid]} for sid in used] + rows


def _category(payload: Mapping[str, Any], row: Mapping[str, Any], bundle_size: int) -> tuple[Any, ...]:
    reasons = sorted({c["failure_reason"] for c in row["components"] if c["failure_reason"]})
    return (payload["prompt_style"], payload["forced_json_requirement"], row.get("retry_scope", "scenario_bundle"),
            bool(payload.get("strict_think", False)), payload["label_rule"], payload["evaluation_mode"],
            row["status"], tuple(reasons), len(row["components"]) < bundle_size)


def settlement_rows(
    artifact_root: Path, conditions: list[tuple[str, Path, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    """One or two recorded attempt rows per category, each with the raw rows of its components."""
    candidates: dict[tuple[Any, ...], list[tuple[Any, ...]]] = defaultdict(list)
    batch_dirs: dict[str, set[str]] = defaultdict(set)
    order = 0
    for root, path, payload in conditions:
        if _is_subset(payload):
            continue
        api = payload["model_type"] != "local"
        for draw_dir in _draws(path):
            attempts = draw_dir / "attempts.jsonl"
            bundle_size = 0
            for number, line, row in _scan_lines(attempts):
                bundle_size = max(bundle_size, len(row["components"]))
                if payload["model_type"] in BATCH_PROVIDERS:
                    ref = row["components"][0].get("request_block_ref")
                    if isinstance(ref, str):
                        batch_dirs[payload["model_type"]].add(ref)
                key = _category(payload, row, bundle_size)
                order += 1
                # Fewest components first, then a provider judge (no thinking text), then ledger order.
                rank = (len(row["components"]), not api, order)
                heap = candidates[key]
                entry = (tuple(-x for x in rank), root, str(path), number, line, bundle_size)
                if len(heap) < SETTLEMENT_CANDIDATES:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
    by_draw: dict[Path, list[tuple[Any, ...]]] = defaultdict(list)
    for key, heap in candidates.items():
        for entry in heap:
            row = json.loads(entry[4])
            by_draw[Path(entry[2]).parent / f"draw-{row['decoder_draw_id']:03d}"].append((key, entry, row))
    measured: dict[tuple[Any, ...], list[tuple[Any, ...]]] = defaultdict(list)
    for draw_dir, items in sorted(by_draw.items()):
        raw = _raw_lines(draw_dir / "raw_outputs.jsonl", [row for _, _, row in items])
        for key, entry, row in items:
            lines = raw[_attempt_key(row)]
            size = sum(len(json.dumps(stored)) for stored in _stored_rows(row, lines, strict_think=key[3]))
            measured[key].append((tuple(-x for x in entry[0]), size, entry, row, lines))
    out: list[dict[str, Any]] = []
    for key in sorted(measured, key=lambda k: json.dumps(k)):
        kept = 0
        for rank, size, entry, row, lines in sorted(measured[key], key=lambda item: (item[1], item[0])):
            if size > SETTLEMENT_MAX_RAW_BYTES or kept == SETTLEMENT_PER_CATEGORY:
                continue
            kept += 1
            _, root, path, number, line, bundle_size = entry
            condition_path = Path(path)
            payload = json.loads(condition_path.read_bytes())
            draw_dir = condition_path.parent / f"draw-{row['decoder_draw_id']:03d}"
            raws = [{"row": stored, "line": raw_number, "sha256": pub.sha256_bytes(raw_line)}
                    for (raw_number, raw_line, _), stored in zip(lines, _stored_rows(row, lines, strict_think=key[3]))]
            out.append(_attempt_row(root, payload, row, bundle_size,
                                    _source(artifact_root, draw_dir / "attempts.jsonl", number, line), raws))
    return out, batch_dirs


def stored_raw(raw: Mapping[str, Any], *, strict_think: bool, truncated: bool) -> dict[str, Any]:
    """A raw row as the settlement fixture stores it."""
    row = dict(raw)
    text = row.pop("raw_response")
    row["raw_response_elided"] = True
    if truncated:
        row.update(completion=None, text_elided=True)
    elif strict_think and isinstance(text, str):
        row.update(raw_answer=text.rsplit(THINK_CLOSE, 1)[-1], raw_think_closed=THINK_CLOSE in text)
    return row


def _stored_rows(attempt: Mapping[str, Any], lines: list[tuple[int, bytes, dict]], *, strict_think: bool) -> list[dict]:
    truncated = {c["component_id"] for c in attempt["components"] if c["failure_reason"] == "truncated"}
    return [stored_raw(raw, strict_think=strict_think, truncated=raw["component_id"] in truncated)
            for _, _, raw in lines]


def _attempt_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return row["scenario_id"], int(row["semantic_attempt_id"]), row["request_block_id"]


def _raw_lines(path: Path, rows: list[Mapping[str, Any]]) -> dict[tuple[str, int, str], list[tuple[int, bytes, dict]]]:
    """The raw rows of each attempt row, read in one pass: ``{(scenario, attempt, block): [(line, bytes, row)]}``."""
    wanted = {_attempt_key(row) for row in rows}
    prefixes = {f'{{"scenario_id": "{sid}"'.encode() for sid, _, _ in wanted}
    width = len(next(iter(prefixes)))
    found: dict[tuple[str, int, str], list[tuple[int, bytes, dict]]] = {key: [] for key in wanted}
    for number, line in pub.numbered_lines(path):
        if line[:width] not in prefixes:
            continue
        raw = json.loads(line)
        key = (raw["scenario_id"], int(raw["semantic_attempt_id"]), raw["request_block_id"])
        if key in found:
            found[key].append((number, line, raw))
    return found


def batch_rows(artifact_root: Path, batch_dirs: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    """The smallest recorded work directory per batch provider that a published row names, its request lines whole."""
    rows = []
    for provider in BATCH_PROVIDERS:
        best: Optional[tuple[int, str, list[Path]]] = None
        for ref in sorted(batch_dirs.get(provider, ())):
            directory = artifact_root / "batch" / Path(ref).name
            parts = sorted((directory / "inputs").glob("*.jsonl"))
            if not parts:
                continue
            size = sum(part.stat().st_size for part in parts)
            if size <= BATCH_MAX_INPUT_BYTES and (best is None or (size, directory.name) < best[:2]):
                best = (size, directory.name, parts)
        if best is None:
            raise SystemExit(f"no {provider} work directory under {BATCH_MAX_INPUT_BYTES} bytes")
        _, name, parts = best
        state = json.loads((artifact_root / "batch" / name / "state.json").read_bytes())
        rows.append({
            "kind": "batch", "provider": provider, "work_dir": name, "endpoint": state.get("endpoint"),
            "parts": [{"name": part.name, "sha256": pub.sha256_bytes(part.read_bytes()),
                       "lines": [line.decode("utf-8") for _, line in pub.numbered_lines(part)]} for part in parts],
        })
    return rows


def build(artifact_root: Path) -> dict[str, list[dict[str, Any]]]:
    conditions = _conditions(artifact_root, pub.ROOTS)
    settlement, batch_dirs = settlement_rows(artifact_root, conditions)
    return {
        pub.CONDITIONS: condition_rows(artifact_root),
        pub.SEEDS: seed_rows(artifact_root, conditions),
        pub.PROMPTS: prompt_rows(artifact_root, conditions),
        pub.SETTLEMENT: settlement,
        pub.BATCH: batch_rows(artifact_root, batch_dirs),
    }


def _lines_at(path: Path, numbers: Iterable[int], *, bundle_sizes: bool = False) -> tuple[dict[int, bytes],
                                                                                        dict[int, int]]:
    wanted = set(numbers)
    found: dict[int, bytes] = {}
    sizes: dict[int, int] = {}
    size = 0
    for index, line in (pub.numbered_lines(path) if path.is_file() else ()):
        if bundle_sizes:
            size = max(size, len(json.loads(line)["components"]))
        if index in wanted:
            found[index], sizes[index] = line, size
            if len(found) == len(wanted):
                break
    return found, sizes


#: The row kinds each table fixture holds; a row of any other kind is one no check reads, and is refused.
TABLE_KINDS = {pub.SEEDS: ("seed", "block"), pub.PROMPTS: ("prompt", "scenario"), pub.SETTLEMENT: ("attempt",)}


def _derived(row: Mapping[str, Any], record: Mapping[str, Any], *, root: str, payload: Mapping[str, Any],
             manifest: Mapping[str, Any], bundle_size: Optional[int], source: Mapping[str, Any]
             ) -> Optional[dict[str, Any]]:
    if row["kind"] == "block":
        return _block_row(record, source)
    if row["kind"] == "attempt":
        return _attempt_row(root, payload, record, bundle_size or 0, source, row["raw"])
    component = next((c for c in record["components"] if c["component_id"] == row.get("component_id")), None)
    if component is None:
        return None
    if row["kind"] == "seed":
        return _seed_row(record, component, manifest.get("generation_seed_base"), source)
    return _prompt_row(root, payload, record, component, source)


def _join_problems(name: str, row: Mapping[str, Any]) -> list[str]:
    attempt = row["attempt"]
    components = {c["component_id"]: c for c in attempt["components"]}
    key = (attempt["scenario_id"], attempt["semantic_attempt_id"], attempt["request_block_id"])
    problems, seen = [], set()
    for raw in row["raw"]:
        stored = raw["row"]
        cid = stored.get("component_id")
        if ((stored.get("scenario_id"), stored.get("semantic_attempt_id"), stored.get("request_block_id")) != key
                or cid not in components or cid in seen
                or stored.get("raw_output_sha256") != components[cid]["raw_output_sha256"]):
            problems.append(f"{name}: {row['source']['path']}:{row['source']['line']}: the raw row at line "
                            f"{raw['line']} is not one of its components' outputs")
        seen.add(cid)
    missing = sorted(cid for cid, component in components.items()
                     if component.get("raw_output_sha256") and cid not in seen)
    if missing:
        problems.append(f"{name}: {row['source']['path']}:{row['source']['line']}: no raw row for the recorded "
                        f"output of {', '.join(missing)}")
    return problems


def check_tables(artifact_root: Path, tables: Mapping[str, Iterable[Mapping[str, Any]]],
                 population: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Every difference between table fixture rows (``{fixture name: rows}``) and their sources."""
    problems: list[str] = []
    #: {ledger path: {line number: [(fixture, row)]}} and {raw path: {line number: [(fixture, attempt row, raw)]}}
    rows_at: dict[str, dict[int, list[tuple[str, Mapping[str, Any]]]]] = defaultdict(lambda: defaultdict(list))
    raws_at: dict[str, dict[int, list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]]] = defaultdict(
        lambda: defaultdict(list))
    for name, rows in tables.items():
        for row in rows:
            if row.get("kind") not in TABLE_KINDS[name]:
                problems.append(f"{name}: a row of kind {row.get('kind')!r}, which no check reads")
            elif row["kind"] == "scenario":
                if population.get(row["scenario_id"]) != row["row"]:
                    problems.append(f"{name}: scenario {row['scenario_id']} is not the population's row")
            else:
                rows_at[row["source"]["path"]][row["source"]["line"]].append((name, row))
                if row["kind"] == "attempt":
                    problems += _join_problems(name, row)
                    raw_path = (Path(row["source"]["path"]).parent / "raw_outputs.jsonl").as_posix()
                    for raw in row["raw"]:  # read where the attempt's own draw keeps them, whatever raw_path says
                        raws_at[raw_path][raw["line"]].append((name, row, raw))
    payloads: dict[Path, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for path, numbers in sorted(rows_at.items()):
        parts = Path(path).parts
        if len(parts) < 5 or parts[0] != "sampling" or not parts[-2].startswith("draw-") or parts[-1] != (
                "attempts.jsonl"):
            problems.append(f"{path} is not a draw's attempt ledger under sampling/<root>/")
            continue
        root = parts[1]
        condition = artifact_root / Path(path).parent.parent / "condition.json"
        campaign = artifact_root / "sampling" / root / "sampling_manifest.json"
        if not (condition.is_file() and campaign.is_file()):
            problems.append(f"{path}: no condition.json beside its draw, or no sampling_manifest.json in its root")
            continue
        payload = payloads.setdefault(condition, json.loads(condition.read_bytes()))
        manifest = manifests.setdefault(root, json.loads(campaign.read_bytes()))
        settlement = any(row["kind"] == "attempt" for wants in numbers.values() for _, row in wants)
        lines, sizes = _lines_at(artifact_root / path, numbers, bundle_sizes=settlement)
        for number, wants in sorted(numbers.items()):
            line = lines.get(number)
            for name, row in wants:
                where = f"{name}: {path}:{number}"
                if line is None:
                    problems.append(f"{where}: the ledger has no such line")
                    continue
                if pub.sha256_bytes(line) != row["source"]["sha256"]:
                    problems.append(f"{where} is not the recorded line")
                    continue
                derived = _derived(row, json.loads(line), root=root, payload=payload, manifest=manifest,
                                   bundle_size=sizes.get(number),
                                   source=_source(artifact_root, artifact_root / path, number, line))
                if derived is None:
                    problems.append(f"{where} holds no component {row.get('component_id')!r}")
                    continue
                differing = sorted(key for key in set(derived) | set(row) if derived.get(key) != row.get(key))
                if differing:
                    problems.append(f"{where}: {', '.join(differing)} not what the line, its condition.json and its "
                                    "sampling_manifest.json give")
    for path, numbers in sorted(raws_at.items()):
        lines, _ = _lines_at(artifact_root / path, numbers)
        for number, raw_wants in sorted(numbers.items()):
            line = lines.get(number)
            for name, row, raw in raw_wants:
                where = f"{name}: {path}:{number}"
                # Stored as its condition (the file, not the row's copy) and its attempt's recorded verdicts say.
                payload = payloads.get(artifact_root / Path(path).parent.parent / "condition.json", {})
                truncated = {c["component_id"] for c in row["attempt"]["components"]
                             if c["failure_reason"] == "truncated"}
                if line is None:
                    problems.append(f"{where}: the raw ledger has no such line")
                elif pub.sha256_bytes(line) != raw["sha256"]:
                    problems.append(f"{where} is not the recorded line")
                elif stored_raw(json.loads(line), strict_think=bool(payload.get("strict_think")),
                                truncated=raw["row"].get("component_id") in truncated) != raw["row"]:
                    problems.append(f"{where} is not the stored raw row")
    return problems


def check_batch(artifact_root: Path, rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Every difference between batch fixture rows and their work directories."""
    problems = []
    for row in rows:
        directory = artifact_root / "batch" / row["work_dir"]
        where = f"{pub.BATCH}: {row['work_dir']}"
        names = sorted(path.name for path in (directory / "inputs").glob("*.jsonl"))
        if [part["name"] for part in row["parts"]] != names:
            problems.append(f"{where}: its parts are not the work directory's inputs {names}")
        state = json.loads((directory / "state.json").read_bytes())
        if state.get("endpoint") != row["endpoint"]:
            problems.append(f"{where}: endpoint {row['endpoint']!r}, but the work directory's is "
                            f"{state.get('endpoint')!r}")
        for part in row["parts"]:
            path = directory / "inputs" / part["name"]
            data = path.read_bytes() if path.is_file() else None
            if data is None or pub.sha256_bytes(data) != part["sha256"]:
                problems.append(f"{where}: {part['name']} is not the recorded input")
            elif "".join(part["lines"]).encode("utf-8") != data:
                problems.append(f"{where}: {part['name']}'s lines are not the recorded input's")
    return problems


def check(artifact_root: Path) -> list[str]:
    """Every difference between the committed fixtures and the published roots (an empty list: none)."""
    problems: list[str] = []
    if pub.fixture_text(condition_rows(artifact_root)) != pub.committed_text(pub.CONDITIONS):
        problems.append(f"{pub.CONDITIONS} is not the conditions and manifests the roots hold")
    tables = {name: pub.read_fixture(name) for name in TABLE_KINDS}
    problems += check_tables(artifact_root, tables, _population())
    problems += check_batch(artifact_root, pub.read_fixture(pub.BATCH))
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    from ruleforge.env import artifact_root

    parser = argparse.ArgumentParser(
        description="Generate, or check, the identity fixtures from the published campaign roots.")
    parser.add_argument("--check", action="store_true", help="compare the committed fixtures with the roots")
    args = parser.parse_args(argv)
    root = artifact_root()
    if args.check:
        problems = check(root)
        print("\n".join(problems) or "fixtures match the published roots")
        return 1 if problems else 0
    pub.FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, rows in build(root).items():
        data = pub.fixture_bytes(rows)
        (pub.FIXTURES / name).write_bytes(data)
        print(f"{name}: {len(rows)} rows, {len(data)} bytes, sha256 {pub.sha256_bytes(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
