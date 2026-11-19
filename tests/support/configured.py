from __future__ import annotations

import json
from pathlib import Path

from ruleforge.core.artifacts import link_new
from ruleforge.evaluate.configured import MARKER_FNAME, marker_bytes
from ruleforge.scenarios.manifest import ScenarioManifest


def mark_configured(root: Path) -> Path:
    """``root`` marked as a configured root of the campaign and population its manifests record."""
    manifest = json.loads((root / "sampling_manifest.json").read_text(encoding="utf-8"))
    population = ScenarioManifest.load(root / "scenario_manifest.json")
    link_new(root / MARKER_FNAME, marker_bytes(str(manifest["campaign_id"]), population.manifest_sha256))
    return root / MARKER_FNAME
