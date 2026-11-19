from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple


def _text(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result)


def traced(results: Sequence[Any], key: str = "result") -> Tuple[List[Any], List[Dict[str, Any]]]:
    """``results`` and one minimal trace each, keyed by ``key`` (``judgment`` for the validated form)."""
    return list(results), [
        {
            key: result,
            "selected_attempt": None if result is None else 1,
            "attempts": [{"attempt": 1, "seed": None, "candidates": [
                {"raw_response": _text(result), "completion": _text(result), "finish_reason": None,
                 "accepted": result is not None}]}],
        }
        for result in results
    ]
