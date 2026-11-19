from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from ruleforge.core.errors import RuleForgeError
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.artifacts import compute_scenario_id, load_jsonl_hashed


class ScenarioSetError(ValueError, RuleForgeError):
    """Raised for a scenario file that is not the one expected, or changed while it was read."""


@dataclass(frozen=True)
class ScenarioSet:
    """Every row of one scenario file, in order, with the manifest a campaign freezes over it."""

    rows: tuple[Mapping[str, Any], ...]
    path: Path
    manifest: ScenarioManifest

    @classmethod
    def load(
        cls, path: Path | str, *, expect_sha256: Optional[str] = None, expect_count: Optional[int] = None,
    ) -> "ScenarioSet":
        """Load the scenario set at ``path``, checking its SHA-256 and row count when given."""
        source = Path(path)
        manifest = ScenarioManifest.build(source, expected_count=expect_count)
        if expect_sha256 is not None and manifest.source_sha256 != expect_sha256:
            raise ScenarioSetError(f"{source.name} has SHA-256 {manifest.source_sha256}, not the expected "
                                   f"{expect_sha256}")
        records, digest = load_jsonl_hashed(source)
        rows = tuple(dict(row) for row in records)
        if digest != manifest.source_sha256 or tuple(map(compute_scenario_id, rows)) != manifest.scenario_ids:
            raise ScenarioSetError(f"{source.name} changed while it was read")
        return cls(rows=rows, path=source, manifest=manifest)

    @property
    def sha256(self) -> str:
        return self.manifest.source_sha256

    @property
    def ids(self) -> tuple[str, ...]:
        return self.manifest.scenario_ids

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self.rows)

    def by_id(self) -> dict[str, Mapping[str, Any]]:
        """Return each row by its scenario id."""
        return dict(zip(self.ids, self.rows))


__all__ = ["ScenarioSet", "ScenarioSetError"]
