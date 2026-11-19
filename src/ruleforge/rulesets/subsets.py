from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ruleforge.core.digests import sha256_file

# Independence across sizes, not within a subset; every strategy draws without replacement inside a subset.
WITH_REPLACEMENT = "with_replacement"
WITHOUT_REPLACEMENT = "without_replacement"
# As with_replacement, then the full set last in an order of each seed's own; the other two strategies write the full
# set in the file's order.
WITH_REPLACEMENT_SHUFFLED_FULL = "with_replacement_shuffled_full"
VALID_SAMPLING_STRATEGIES = (WITH_REPLACEMENT, WITHOUT_REPLACEMENT, WITH_REPLACEMENT_SHUFFLED_FULL)

MANIFEST_META_FNAME = "manifest.json"


def strategy_root(output_root: Path, sampling_strategy: str) -> Path:
    """Return the directory holding every seed of one sampling strategy; the manifest keys files relative to it."""
    return Path(output_root) / sampling_strategy


def seed_dir(output_root: Path, sampling_strategy: str, seed: int) -> Path:
    """Return the directory holding one seed's subset files."""
    return strategy_root(output_root, sampling_strategy) / f"seed_{seed}"


def load_manifest(manifest_dir: Path) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Read ``manifest.json`` and the checksum file it names; return ``(metadata, {relative_path: digest})``."""
    manifest_dir = Path(manifest_dir)
    meta = json.loads(
        (manifest_dir / MANIFEST_META_FNAME).read_text(encoding="utf-8")
    )
    entries = parse_manifest(
        (manifest_dir / meta["files_manifest"]).read_text(encoding="utf-8")
    )
    return meta, entries


def sample_independent(all_rules: Sequence[Dict], seed: int, *, full_set: bool = False) -> List[Dict]:
    """Draw an independent subset per size (size k need not contain size k-1); ``full_set`` also draws the full set,
    last."""
    rng = random.Random(seed)
    return [
        {"subset_id": f"subset_{size}", "size": size, "rules": rng.sample(list(all_rules), size)}
        for size in range(2, len(all_rules) + (1 if full_set else 0))
    ]


def sample_nested(all_rules: Sequence[Dict], seed: int) -> List[Dict]:
    """Draw nested subsets: size k is size k-1 plus one rule."""
    shuffled = random.Random(seed).sample(list(all_rules), len(all_rules))
    return [
        {"subset_id": f"subset_{size}", "size": size, "rules": shuffled[:size]}
        for size in range(2, len(all_rules))
    ]


def create_subsets(
    sampling_strategy: str,
    all_rules: Sequence[Dict],
    seed: int,
) -> List[Dict]:
    """Draw the subsets of ``sampling_strategy``."""
    if sampling_strategy == WITH_REPLACEMENT:
        return sample_independent(all_rules, seed)
    if sampling_strategy == WITHOUT_REPLACEMENT:
        return sample_nested(all_rules, seed)
    if sampling_strategy == WITH_REPLACEMENT_SHUFFLED_FULL:
        return sample_independent(all_rules, seed, full_set=True)
    raise ValueError(
        f"Invalid sampling_strategy {sampling_strategy!r}; "
        f"valid: {list(VALID_SAMPLING_STRATEGIES)}"
    )


def generate_subset_files(
    rule_set_path: Path,
    output_dir: Path,
    sampling_strategy: str,
    seed: int,
) -> List[Path]:
    """Write one rule JSONL per subset size 2..n for one seed; return the paths sorted. The full set at n is in the
    file's order unless the strategy draws it."""
    all_rules = _read_rules(rule_set_path)
    if len(all_rules) < 2:
        raise ValueError(
            f"Rule set must contain at least 2 rules, got {len(all_rules)} "
            f"from {rule_set_path}"
        )

    subsets = create_subsets(sampling_strategy, all_rules, seed)
    if all(subset["size"] < len(all_rules) for subset in subsets):
        subsets.append({"size": len(all_rules), "rules": all_rules})
    generated: List[Path] = []
    for subset in subsets:
        path = output_dir / f"subset_size_{subset['size']:02d}.jsonl"
        _write_rules(path, subset["rules"])
        generated.append(path)
    return sorted(generated)


def _read_rules(rule_set_path: Path) -> List[Dict[str, Any]]:
    with Path(rule_set_path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_rules(path: Path, rules: Sequence[Dict[str, Any]]) -> None:
    # The serialization is part of the checksum contract.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(rule, ensure_ascii=False) + "\n" for rule in rules),
        encoding="utf-8",
    )


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    return sha256_file(path)


def build_manifest(root: Path) -> Dict[str, str]:
    """Map each subset file's path (relative to ``root``) to its digest."""
    root = Path(root)
    return {
        str(path.relative_to(root)): file_digest(path)
        for path in sorted(root.rglob("subset_size_*.jsonl"))
    }


def parse_manifest(text: str) -> Dict[str, str]:
    """Read a ``sha256sum``-format manifest back into ``{path: digest}``."""
    entries: Dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, _, relative_path = line.partition("  ")
        if not relative_path:
            raise ValueError(f"Malformed manifest line: {line!r}")
        entries[relative_path] = digest
    return entries


def diff_manifest(
    expected: Dict[str, str],
    actual: Dict[str, str],
) -> Dict[str, List[str]]:
    """Compare two manifests; return the sorted ``missing``, ``unexpected`` and ``changed`` paths."""
    expected_paths, actual_paths = set(expected), set(actual)
    return {
        "missing": sorted(expected_paths - actual_paths),
        "unexpected": sorted(actual_paths - expected_paths),
        "changed": sorted(
            path
            for path in expected_paths & actual_paths
            if expected[path] != actual[path]
        ),
    }
