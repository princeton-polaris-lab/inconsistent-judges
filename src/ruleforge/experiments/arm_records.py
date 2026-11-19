from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ruleforge.evaluate.identity import REPLAY_RECORD_FNAME

#: A condition's ``individual_rule_context`` when it judged every rule alone, each in a separate single-rule call.
ISOLATED = "isolated"


def _identity_without_style(condition: Any, exempt: Sequence[str]) -> Dict[str, Any]:
    payload = condition.to_dict()
    for key in ("condition_id", *exempt):
        payload.pop(key, None)
    return payload


def differences_beyond_style(a: Any, b: Any, exempt: Sequence[str] = ("prompt_style",)) -> List[str]:
    """Return the identity fields in which conditions ``a`` and ``b`` differ but for their label and ``exempt``, sorted:
    empty exactly when the two can be one comparison's arms."""
    first, other = _identity_without_style(a, exempt), _identity_without_style(b, exempt)
    return sorted(key for key in set(first) | set(other) if first.get(key) != other.get(key))


def bootstrap_settings(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """The campaign's bootstrap and randomization settings, as ``contrast_row`` takes them."""
    bootstrap = payload["bootstrap"]
    return dict(
        n_resamples=int(bootstrap["n_resamples"]),
        ci_level=float(bootstrap["ci_level"]),
        seed=int(bootstrap["seed"]),
        n_permutations=int(payload["randomization_test"]["n_permutations"]),
    )


def read_replay_record(root: Path | str) -> Optional[tuple[dict[str, Any], str]]:
    """Return a derived root's ``label_replay.json`` with the SHA-256 of the bytes it was parsed from; None for a
    campaign root as it ran."""
    path = Path(root) / REPLAY_RECORD_FNAME
    if not path.is_file():
        return None
    data = path.read_bytes()
    return json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest()


def replay_provenance(root: Path) -> Optional[Dict[str, Any]]:
    """Return what a golden records of a root's label replay: the record's name and sha256 and what it re-scored; None
    for a campaign root as it ran."""
    read = read_replay_record(root)
    if read is None:
        return None
    record, sha256 = read
    return {
        "name": REPLAY_RECORD_FNAME, "sha256": sha256,
        **{key: record.get(key) for key in ("tool", "code_state_sha256", "source_label_rule", "label_rule")},
    }
