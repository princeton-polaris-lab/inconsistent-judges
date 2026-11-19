from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ruleforge.core.digests import _HEX64, canonical_digest, sha256_file
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.files import write_json_sidecar
from ruleforge.core.artifacts import compute_scenario_id, iter_jsonl_records

# A campaign sidecar of another version is refused, not reinterpreted.
SAMPLING_SCHEMA_VERSION = 2
# Joins a scenario's prompt and response in its id.
_ID_SEPARATOR = "\x1e"


def scenario_id(record: Mapping[str, Any], where: str) -> str:
    """Return a scenario's id (``compute_scenario_id``), refusing a prompt or response that holds the character the id
    joins them with: two such rows, split differently around it, would share one id."""
    for field in ("prompt", "response"):
        value = record.get(field)
        if isinstance(value, str) and _ID_SEPARATOR in value:
            raise SamplingDesignError(f"{where}: its {field} holds U+001E, which a scenario id joins prompt and response "
                                      "with, so another row could share its id; remove it from the scenario file")
    return compute_scenario_id(record)


@dataclass(frozen=True)
class ScenarioManifest:
    """The ordered, hashed scenario population of one comparison; joins use ``scenario_id``, never row position."""

    scenario_ids: tuple[str, ...]
    source_name: str
    source_sha256: str
    record_count: int
    manifest_sha256: str
    schema_version: int = SAMPLING_SCHEMA_VERSION

    @staticmethod
    def digest(
        scenario_ids: Sequence[str], source_sha256: str, schema_version: int
    ) -> str:
        return canonical_digest(
            {
                "schema_version": schema_version,
                "scenario_ids": list(scenario_ids),
                "source_sha256": source_sha256,
            }
        )

    @classmethod
    def build(
        cls, source: Path | str, *, expected_count: Optional[int] = None
    ) -> "ScenarioManifest":
        """Freeze a scenario JSONL file, read strictly."""
        path = Path(source)
        ids = tuple(scenario_id(record, f"{path.name} row {row}")
                    for row, record in enumerate(iter_jsonl_records(path), start=1))
        source_sha256 = sha256_file(path)
        manifest = cls(
            scenario_ids=ids,
            source_name=path.name,
            source_sha256=source_sha256,
            record_count=len(ids),
            manifest_sha256=cls.digest(ids, source_sha256, SAMPLING_SCHEMA_VERSION),
        )
        manifest.validate(expected_count=expected_count)
        return manifest

    def validate(self, *, expected_count: Optional[int] = None) -> None:
        """Refuse duplicates, a wrong count, or a digest that does not match."""
        if self.schema_version != SAMPLING_SCHEMA_VERSION:
            raise SamplingDesignError(
                f"scenario manifest schema_version {self.schema_version!r} is not "
                f"{SAMPLING_SCHEMA_VERSION}"
            )
        if not self.scenario_ids:
            raise SamplingDesignError("scenario manifest holds no scenarios")
        bad = [sid for sid in self.scenario_ids if not (isinstance(sid, str) and _HEX64.match(sid))]
        if bad:
            raise SamplingDesignError(
                f"scenario manifest holds {len(bad)} malformed scenario id(s): {bad[:3]}"
            )
        counts: dict[str, int] = {}
        for sid in self.scenario_ids:
            counts[sid] = counts.get(sid, 0) + 1
        duplicates = sorted(sid for sid, count in counts.items() if count > 1)
        if duplicates:
            raise SamplingDesignError(
                f"scenario manifest holds {len(duplicates)} duplicate scenario id(s): "
                f"{duplicates[:3]}"
            )
        if self.record_count != len(self.scenario_ids):
            raise SamplingDesignError(
                f"scenario manifest record_count {self.record_count} does not match "
                f"its {len(self.scenario_ids)} ids"
            )
        expected_digest = self.digest(
            self.scenario_ids, self.source_sha256, self.schema_version
        )
        if self.manifest_sha256 != expected_digest:
            raise SamplingDesignError(
                "scenario manifest digest mismatch: the ids, their order or the "
                "source digest changed after the manifest was frozen"
            )
        if expected_count is not None and len(self.scenario_ids) != expected_count:
            raise SamplingDesignError(
                f"scenario manifest holds {len(self.scenario_ids)} scenarios; this "
                f"campaign requires exactly {expected_count}"
            )

    def verify_source(self, source: Path | str) -> None:
        """Refuse a scenario file that is not the one this manifest froze."""
        rebuilt = ScenarioManifest.build(source)
        if rebuilt.source_sha256 != self.source_sha256:
            raise SamplingDesignError(
                f"{Path(source).name} has digest {rebuilt.source_sha256[:12]}..., the "
                f"manifest froze {self.source_sha256[:12]}..."
            )
        if rebuilt.scenario_ids != self.scenario_ids:  # pragma: no cover - defensive
            raise SamplingDesignError("scenario ids no longer match the frozen manifest")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scenario_ids"] = list(self.scenario_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScenarioManifest":
        try:
            manifest = cls(
                scenario_ids=tuple(payload["scenario_ids"]),
                source_name=str(payload["source_name"]),
                source_sha256=str(payload["source_sha256"]),
                record_count=int(payload["record_count"]),
                manifest_sha256=str(payload["manifest_sha256"]),
                schema_version=int(payload.get("schema_version", 0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SamplingDesignError(f"malformed scenario manifest: {error}") from error
        manifest.validate()
        return manifest

    def write(self, path: Path | str) -> Path:
        return write_json_sidecar(path, self.to_dict())

    @classmethod
    def load(cls, path: Path | str) -> "ScenarioManifest":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SamplingDesignError(f"cannot read scenario manifest {source}: {error}") from error
        if not isinstance(payload, Mapping):
            raise SamplingDesignError(f"{source} is not a JSON object")
        return cls.from_dict(payload)
