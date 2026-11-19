from __future__ import annotations

from typing import Any, Mapping

from ruleforge.core.errors import SamplingDesignError

CHAIN_KEY = "chain"
_HOP_TARGET_FIELDS = ("scenario_manifest_sha256", "source_sha256")
_HOP_KEYS = frozenset({"parent", "target", "removed_ids", "added_ids"})


class RebaseError(SamplingDesignError):
    """The parent root, the population or the destination cannot be rebased."""


def _refuse(condition: bool, message: str) -> None:
    if not condition:
        raise RebaseError(message)


def rebase_hop(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return one link of a chain as a rebase record describes it: ``parent``, ``target``, ``removed_ids``,
    ``added_ids``."""
    return {
        "parent": dict(record["parent"]),
        "target": {key: record["target"][key] for key in _HOP_TARGET_FIELDS},
        "removed_ids": list(record["removed_ids"]),
        "added_ids": list(record["added_ids"]),
    }


def rebase_chain(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the rebases from the first root to ``record``'s, oldest first; a damaged ``chain`` is refused."""
    chain = record.get(CHAIN_KEY)
    if chain is None:
        return [rebase_hop(record)]
    _refuse(isinstance(chain, list) and all(isinstance(hop, Mapping) and _HOP_KEYS <= set(hop) for hop in chain),
            f"{CHAIN_KEY} is not a list of hops (each with {list(_HOP_KEYS)}): the record cannot be read as a chain")
    return [dict(hop) for hop in chain]
