from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from ruleforge.evaluate.amendment import REPLAYED_DIGESTS, draw_digests
from ruleforge.evaluate.campaign import load_conditions
from ruleforge.evaluate.identity import REPLAY_RECORD_FNAME, SAMPLING_MANIFEST_FNAME, draw_dirname
from ruleforge.evaluate.ledger import LEDGER_FNAME


def _ledgers(directories: list[Path]) -> list[dict[str, Any]]:
    return [{"path": str(directory), **draw_digests(directory)} for directory in directories
            if (directory / LEDGER_FNAME).is_file()]


def derive(source: Path, destination: Path, *, shards: Optional[Path] = None) -> dict[str, Any]:
    """``source`` copied to ``destination`` with the ``label_replay.json`` a re-score of it would have written."""
    shutil.copytree(source, destination, symlinks=True)
    manifest = json.loads((source / SAMPLING_MANIFEST_FNAME).read_text(encoding="utf-8"))
    conditions = {}
    for condition_id, condition in sorted(load_conditions(destination).items()):
        draws = {}
        for draw_dir in sorted((destination / condition_id).glob("draw-*")):
            draw = int(draw_dir.name.split("-")[1])
            sources = ([shard / condition_id / draw_dirname(draw) for shard in sorted(shards.iterdir())]
                       if shards is not None else [source / condition_id / draw_dirname(draw)])
            draws[str(draw)] = {"source_ledgers": _ledgers(sources),
                                "replayed_draw": draw_digests(draw_dir, REPLAYED_DIGESTS)}
        conditions[condition_id] = {"source": str(shards if shards is not None else source),
                                    "from_shards": shards is not None, "source_fingerprint": condition.fingerprint,
                                    "fingerprint": condition.fingerprint, "draws": draws}
    record = {"tool": "smrc.sampling.label_replay", "label_rule": manifest["label_rule"],
              "source_label_rule": manifest["label_rule"], "campaign_id": manifest["campaign_id"],
              "conditions": conditions}
    (destination / REPLAY_RECORD_FNAME).write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return record
