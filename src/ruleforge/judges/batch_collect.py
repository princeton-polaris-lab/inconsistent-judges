from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ruleforge.core.record_names import PARTIAL_KIND
from ruleforge.judges import anthropic_batch, openai_batch
from ruleforge.judges.batch_api import (
    PARTIAL_FNAME,
    PARTIAL_SCHEMA_VERSION,
    TERMINAL_STATUSES,
    TEXT_EXTRACTORS,
    BatchExecutionError,
    BatchRequest,
    custom_id_prefix,
    sha256_bytes,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

OPENAI_BATCH, ANTHROPIC_BATCH = "openai_batch", "anthropic_batch"
NO_RESULT = "no_result"
OPENAI_COMPLETED = "completed"


class BatchCollectRefused(BatchExecutionError):
    """Raised for a batch the collect will not collect: still running, completed, or with nothing to read."""


def _read_lines(path: Path, *, expected_sha256: str) -> List[Dict[str, Any]]:
    payload = path.read_bytes()
    if sha256_bytes(payload) != expected_sha256:
        raise BatchCollectRefused(f"{path} is not the input file its state.json recorded (its sha256 differs)")
    return [json.loads(line) for line in payload.decode("utf-8").split("\n") if line.strip()]


def _openai_requests(directory: Path, state: Mapping[str, Any]) -> List[BatchRequest]:
    requests = []
    for shard in state["shards"]:
        for record in _read_lines(directory / "inputs" / shard["path"], expected_sha256=shard["sha256"]):
            if record.get("url") != state["endpoint"]:
                raise BatchCollectRefused(f"{directory.name}: an input line of {shard['path']} is for "
                                          f"{record.get('url')!r}, not the batch's {state['endpoint']!r}")
            requests.append(BatchRequest(custom_id=record["custom_id"], body=record["body"]))
    return requests


def _inline_requests(directory: Path, state: Mapping[str, Any], id_key: str, body_key: str) -> List[BatchRequest]:
    return [BatchRequest(custom_id=record[id_key], body=record[body_key])
            for shard in state["shards"]
            for record in _read_lines(directory / "inputs" / shard["path"], expected_sha256=shard["sha256"])]


def transport_of(state: Mapping[str, Any]) -> str:
    """Return the transport a ``state.json`` belongs to: ``anthropic_batch`` or ``openai_batch``."""
    if state.get("endpoint") == anthropic_batch.ANTHROPIC_BATCH_ENDPOINT and state.get("model"):
        return ANTHROPIC_BATCH
    if state.get("endpoint") in TEXT_EXTRACTORS and "shards" in state and "model" not in state:
        return OPENAI_BATCH
    raise BatchCollectRefused(f"a state.json of endpoint {state.get('endpoint')!r} is neither an OpenAI Batch "
                              f"work directory's nor an Anthropic one's; only those are collected")


def _error_reason(record: Mapping[str, Any]) -> str:
    error = record.get("error")
    if isinstance(error, Mapping) and error.get("code"):
        return str(error["code"])
    status = (record.get("response") or {}).get("status_code")
    return f"status {status}" if status is not None else "error"


def _openai_error_reasons(outputs: Path) -> Dict[str, str]:
    reasons: Dict[str, str] = {}
    for path in sorted(outputs.glob("part-*.errors.jsonl")):
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                record = json.loads(line)
                reasons[str(record.get("custom_id"))] = _error_reason(record)
    return reasons


def partial_marker(
    *, transport: str, endpoint: str, requests: Sequence[BatchRequest], outcomes: Mapping[str, Any],
    usage: Mapping[str, int], parts: Mapping[str, Mapping[str, Any]], error_reasons: Mapping[str, str],
) -> Dict[str, Any]:
    """Build the partial marker: the requests expected, returned and missing (with reasons), and each part's end
    state."""
    expected = [request.custom_id for request in requests]
    reasons: Dict[str, str] = {}
    for custom_id in expected:
        outcome = outcomes.get(custom_id)
        if outcome is not None and outcome.error is None:
            continue
        reasons[custom_id] = error_reasons.get(custom_id) or (outcome.error if outcome is not None else NO_RESULT)
    return {
        "kind": PARTIAL_KIND,
        "schema_version": PARTIAL_SCHEMA_VERSION,
        "transport": transport,
        "endpoint": endpoint,
        "requests_expected": len(expected),
        "requests_returned": usage["requests"],  # = the answers without an error: the known ids' billed ones
        "requests_missing": len(reasons),
        "missing_reasons": dict(sorted(Counter(reasons.values()).items())),
        "missing_custom_ids": sorted(reasons),
        "custom_id_prefix": custom_id_prefix(expected),
        "parts": {name: dict(part) for name, part in sorted(parts.items())},
        "usage": dict(usage),
    }


def _refuse_unless_ended(directory: Path, states: Mapping[str, Optional[str]], *, terminal: frozenset,
                         completed: str) -> None:
    running = {name: state for name, state in states.items() if state not in terminal}
    if running:
        raise BatchCollectRefused(
            f"{directory.name}: part(s) {running} have not ended. The collect never cancels a batch: wait until it "
            f"ends (an OpenAI batch expires at the end of its completion window, an Anthropic Message Batch's "
            f"requests 24 h after it was created) and run it again")
    if all(state == completed for state in states.values()):
        raise BatchCollectRefused(
            f"{directory.name}: every part {completed}. This batch is the normal path's: relaunch its campaign with "
            f"the same command, which re-attaches to it, downloads it, records its block and writes usage.json; "
            f"a collect would leave the block unrecorded")


def _collect_openai(directory: Path, state: Dict[str, Any], client_factory: Callable[[], Any]) -> Dict[str, Any]:
    requests = _openai_requests(directory, state)
    runner = openai_batch.OpenAIBatchRunner(root=directory.parent, endpoint=state["endpoint"],
                                            client_factory=client_factory)
    if runner.workdir(requests) != directory:
        raise BatchCollectRefused(f"{directory}: its input files address {runner.workdir(requests).name}, not "
                                  f"this directory; it is not the batch its name says")
    state = runner.refresh(requests)
    parts = state["parts"]
    _refuse_unless_ended(directory, {name: part.get("status") for name, part in parts.items()},
                         terminal=TERMINAL_STATUSES, completed=OPENAI_COMPLETED)
    for name, part in parts.items():
        completed = openai_batch.completed_count(part)
        if part.get("output_file_id") and completed is None:
            raise BatchCollectRefused(
                f"{directory.name}: OpenAI names an output file for {name} ({part.get('status')}) but gives no "
                f"request counts for it, so its output cannot be held to the requests it billed")
        if completed and not part.get("output_file_id"):
            raise BatchCollectRefused(
                f"{directory.name}: OpenAI counts {completed} completed request(s) in {name} ({part.get('status')}) "
                f"and names no output file for it, so a collect now would bill none of them; run it again later")
    outputs = runner.download(requests)
    outcomes, usage = runner.read_results(requests)
    return partial_marker(
        transport=OPENAI_BATCH, endpoint=state["endpoint"], requests=requests, outcomes=outcomes,
        usage=usage.to_dict(), error_reasons=_openai_error_reasons(outputs),
        parts={name: {"state": part.get("status"), "id": part.get("batch_id"),
                      "provider_counts": part.get("request_counts")} for name, part in parts.items()})


def _collect_anthropic(directory: Path, state: Dict[str, Any], client_factory: Callable[[], Any]) -> Dict[str, Any]:
    requests = _inline_requests(directory, state, "custom_id", "params")
    runner = anthropic_batch.AnthropicBatchRunner(root=directory.parent, model=state["model"],
                                                  client_factory=client_factory)
    if runner.workdir(requests) != directory:
        raise BatchCollectRefused(f"{directory}: its input files address {runner.workdir(requests).name}, not "
                                  f"this directory; it is not the batch its name says")
    state = runner.refresh(requests)
    parts = state["parts"]
    _refuse_unless_ended(directory, {name: anthropic_batch.part_state(part) for name, part in parts.items()},
                         terminal=anthropic_batch.TERMINAL_PART_STATES, completed=anthropic_batch.PART_COMPLETED)
    runner.download(requests)
    outcomes, usage = runner.read_results(requests, partial=True)
    return partial_marker(
        transport=ANTHROPIC_BATCH, endpoint=state["endpoint"], requests=requests, outcomes=outcomes,
        usage=usage.to_dict(), error_reasons={},
        parts={name: {"state": anthropic_batch.part_state(part), "id": part.get("batch_id"),
                      "provider_counts": part.get("request_counts")} for name, part in parts.items()})


def collect(
    directory: Path | str, *, openai_client: Optional[Callable[[], Any]] = None,
    anthropic_client: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Collect the batch work directory ``directory``; return its partial marker, also written beside its usage."""
    directory = Path(directory)
    state_path = directory / "state.json"
    if not state_path.is_file():
        raise BatchCollectRefused(f"{directory} holds no state.json: it is not a Batch work directory")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    transport = transport_of(state)
    if transport == OPENAI_BATCH:
        marker = _collect_openai(directory, state, openai_client or openai_batch.default_client)
    else:
        marker = _collect_anthropic(directory, state, anthropic_client or anthropic_batch.default_client)
    write_json_atomic(directory / PARTIAL_FNAME, marker)
    return marker


def summary(directory: Path, marker: Mapping[str, Any]) -> str:
    """Return one summary line: the batch, how its parts ended, how many requests came back."""
    states = ", ".join(sorted({str(part["state"]) for part in marker["parts"].values()}))
    reasons = ", ".join(f"{reason} {count}" for reason, count in marker["missing_reasons"].items()) or "none"
    return (f"{directory.name}: {marker['transport']} ({states}): {marker['requests_returned']} of "
            f"{marker['requests_expected']} request(s) returned, {marker['requests_missing']} missing ({reasons}); "
            f"usage.json and {PARTIAL_FNAME} written")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="batch_collect",
        description="Collect a paid batch that ended without completing: what came back, its usage and a partial marker.",
        epilog="Read-only toward the provider (status GETs and file downloads); never creates, submits or cancels.")
    parser.add_argument("directories", nargs="+", type=Path, metavar="batch_dir",
                        help="a Batch work directory ($RULEFORGE_BATCH_ROOT/<key>) whose batch ended without completing")
    args = parser.parse_args(argv)
    refused = 0
    for directory in args.directories:
        try:
            marker = collect(directory)
        except (BatchExecutionError, ValueError) as error:
            print(f"batch_collect: {error}", file=sys.stderr)
            refused += 1
            continue
        print(summary(directory, marker))
    return 2 if refused else 0
