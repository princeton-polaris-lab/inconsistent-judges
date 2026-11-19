from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ruleforge.core.artifacts import compute_scenario_id, load_jsonl_records, write_jsonl_records, write_text_atomically
from ruleforge.core.digests import canonical_digest
from ruleforge.core.errors import SamplingDesignError
from ruleforge.evaluate.campaign import SCENARIO_MANIFEST_FNAME, campaign_completion, load_conditions
from ruleforge.evaluate.identity import (
    CAMPAIGN_MODE_PILOT,
    CAMPAIGN_MODE_SCREENING,
    EVALUATION_MODE_BANK,
    EVALUATION_MODE_FULL,
    FULL_RULESET_COMPONENT,
    RETRY_SCOPE_COMPONENT,
    SAMPLING_MANIFEST_FNAME,
    ConditionSpec,
    draw_dirname,
    load_sampling_manifest,
)
from ruleforge.evaluate.ledger import AttemptRecord, ComponentRecord, DrawLedger, LedgerState, raw_output_sha256
from ruleforge.evaluate.runner import STATE_COMPLETE, draw_ended
from ruleforge.evaluate.validation import TRUNCATED, ComponentSchema, strict_think_answer
from ruleforge.grpo.anchor_set import SIDE_DEV, condition_projection
from ruleforge.grpo.report import (
    DEV,
    DEV_NONYES,
    PROBE,
    SCENARIOS_FNAME,
    STAGE1,
    ReportError,
    Scenarios,
    Settings,
    probe_scenarios,
    read_scenarios,
    read_settings,
    rule_ids_of,
    scenario_sets,
    stage1_scenarios,
    write_table_identity,
)
from ruleforge.judges.completion_format import ThinkingFamily, directory_family
from ruleforge.prompts.parse import THINK_CLOSE, THINK_OPEN
from ruleforge.scenarios.manifest import ScenarioManifest

POPULATIONS_FNAME = "populations.json"
#: The campaigns a table is read from: pilots, and the one-draw probe's screening campaign.
DEV_CAMPAIGN_MODES = (CAMPAIGN_MODE_PILOT, CAMPAIGN_MODE_SCREENING)
#: The population the base's draws run on.
REFERENCES = "references"


class TableError(ReportError):
    """A campaign that is not the table it would be read as."""


@dataclass(frozen=True)
class TableKind:
    """One response table: the campaign's evaluation mode, the objects its rows carry and its scenario set."""

    name: str
    evaluation_mode: str
    full: bool
    rules: bool
    scenario_set: str


KINDS = {kind.name: kind for kind in (
    TableKind("bank", EVALUATION_MODE_BANK, False, True, REFERENCES),
    TableKind("full", EVALUATION_MODE_FULL, True, False, REFERENCES),
    TableKind(STAGE1, EVALUATION_MODE_FULL, True, False, STAGE1),
    TableKind(PROBE, EVALUATION_MODE_BANK, False, True, PROBE),
)}


def expected_draws(kind: TableKind, settings: Settings) -> int:
    """Return the draws a kind's campaign runs, as the settings declare them."""
    return {"bank": settings["probe"]["bank_draws"], "full": settings["stage1"]["base_draws"],
            STAGE1: settings["stage1"]["draws"], PROBE: 1}[kind.name]


def scenario_table(dev: Path | str, dev_nonyes: Path | str | None = None) -> list[dict[str, str]]:
    """Return the scenario table: every dev and dev_nonyes row by campaign scenario id with its source; a row without a
    source or a scenario in both files is refused."""
    rows: list[dict[str, str]] = []
    for subset, path in ((DEV, dev), (DEV_NONYES, dev_nonyes)):
        for row in load_jsonl_records(path) if path is not None else ():
            source = row.get("source")
            if not isinstance(source, str) or not source:
                raise TableError(f"a row of {path} has no source")
            rows.append({"scenario_id": compute_scenario_id(row), "source": source, "subset": subset})
    ids = [row["scenario_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise TableError("a scenario appears twice across the dev and dev_nonyes populations")
    return rows


def _scenarios_of(table: Sequence[Mapping[str, str]]) -> Scenarios:
    return Scenarios(tuple(row["scenario_id"] for row in table), tuple(row["source"] for row in table),
                     tuple(row["subset"] for row in table))


def stage1_table(table: Sequence[Mapping[str, str]], settings: Settings) -> list[dict[str, str]]:
    """Return the scenario table cut to stage 1's scenarios, so the base is drawn on them alone; refused unless stage 1
    and the probe fix the same scenarios on the cut table as on the whole one."""
    whole = _scenarios_of(table)
    keep = set(stage1_scenarios(whole, settings))
    cut = [dict(row) for row in table if row["scenario_id"] in keep]
    part = _scenarios_of(cut)
    probe = settings["probe"]
    moved = [name for name, of_whole, of_cut in (
        ("stage 1", stage1_scenarios(whole, settings), stage1_scenarios(part, settings)),
        ("the probe", probe_scenarios(whole, probe["scenarios"], probe["seed"]),
         probe_scenarios(part, probe["scenarios"], probe["seed"])),
    ) if of_whole != of_cut]
    if moved:
        raise TableError(f"cut to stage 1, the table would fix other scenarios for {' and '.join(moved)} than the "
                         "whole table does, since its sources' quotas shift: draw --dev and --dev-nonyes again, at "
                         "random, so the sources keep their shares")
    return cut


def stage_sets(scenarios: Scenarios, settings: Settings) -> dict[str, list[str]]:
    """Return every dev campaign's scenario set (sorted ids)."""
    return {REFERENCES: sorted(scenarios.ids), **scenario_sets(scenarios, settings)}


def write_populations(sets: Mapping[str, Sequence[str]], populations: Sequence[Path | str], out_dir: Path | str
                      ) -> dict[str, Any]:
    """Write each stage's population file, rows copied byte for byte from `populations`, and `populations.json` naming
    each with its SHA-256 and count; an id no file holds is refused."""
    lines: dict[str, str] = {}
    for path in populations:
        # a newline alone ends a line: a row may hold a raw U+2028 in a JSON string
        for line in Path(path).read_text(encoding="utf-8").split("\n"):
            if line.strip():
                lines.setdefault(compute_scenario_id(json.loads(line)), line + "\n")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {}
    for name, ids in sets.items():
        missing = sorted(set(ids) - set(lines))
        if missing:
            raise TableError(f"the {name} set names {len(missing)} scenario(s) no population holds, e.g. {missing[:2]}")
        wanted = set(ids)
        path, text = out / f"{name}.jsonl", "".join(line for sid, line in lines.items() if sid in wanted)
        write_text_atomically(path, text)
        record[name] = {"path": str(path.absolute()), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "count": len(wanted)}
    write_text_atomically(out / POPULATIONS_FNAME, json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


@dataclass(frozen=True)
class DevCondition:
    """One condition of a pilot or screening campaign root, as a table is read from it."""

    root: Path
    manifest: Mapping[str, Any]
    scenario_ids: tuple[str, ...]
    condition: ConditionSpec
    draw_states: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return all(state == STATE_COMPLETE for state in self.draw_states)


def find_condition(root: Path | str, *, policy: str, evaluation_mode: str) -> str:
    """Return the id of the root's one condition in `evaluation_mode` whose declared policy is `policy`."""
    found = sorted(cid for cid, spec in load_conditions(Path(root)).items()
                   if spec.evaluation_mode == evaluation_mode and dict(spec.extra).get("policy") == policy)
    if len(found) != 1:
        raise TableError(f"{root} holds {len(found)} {evaluation_mode!r} condition(s) of policy {policy!r}: {found}")
    return found[0]


def open_dev_condition(root: Path | str, condition_id: str, *, evaluation_mode: str) -> DevCondition:
    """Open a condition of a pilot or screening campaign root whose draws have all ended; another campaign mode, a
    foreign scenario manifest or a draw still running is refused."""
    base = Path(root)
    manifest = load_sampling_manifest(base / SAMPLING_MANIFEST_FNAME)
    if manifest["campaign_mode"] not in DEV_CAMPAIGN_MODES:
        raise TableError(f"{base} is a {manifest['campaign_mode']!r} campaign; the dev campaigns are "
                         f"{list(DEV_CAMPAIGN_MODES)}")
    scenario_manifest = ScenarioManifest.load(base / SCENARIO_MANIFEST_FNAME)
    if scenario_manifest.manifest_sha256 != manifest["scenario_manifest_sha256"]:
        raise TableError(f"{base}: the scenario manifest is not the one the campaign manifest froze")
    conditions = load_conditions(base)
    from ruleforge.grpo.anchor_set import named_condition

    try:
        condition = named_condition(conditions, condition_id)
    except SamplingDesignError as error:
        raise TableError(f"{base} registers no condition {condition_id!r}: {error}") from error
    condition_id = condition.condition_id
    if condition.evaluation_mode != evaluation_mode:
        raise TableError(f"{base}/{condition_id} is registered as {condition.evaluation_mode!r}, not {evaluation_mode!r}")
    gate = campaign_completion(base, write=False)["conditions"][condition_id]["draws"]
    draws = [gate[str(draw)] for draw in range(int(manifest["expected_decoder_draws"]))]
    states = tuple(record["state"] for record in draws)
    if not all(map(draw_ended, draws)):
        raise TableError(f"{base}/{condition_id}: draw states {list(states)}; a table is read from draws that ended "
                         "(complete, or out of attempts with nothing pending)")
    return DevCondition(base, manifest, scenario_manifest.scenario_ids, condition, states)


def _settled(state: LedgerState, sid: str) -> dict[str, tuple[AttemptRecord, ComponentRecord]]:
    if state.retry_scope == RETRY_SCOPE_COMPONENT:
        return dict(state.settled.get(sid, {}))
    return state.accepted_components(sid) if state.is_accepted(sid) else {}


def _attempts(records: Sequence[AttemptRecord]) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = {}
    for record in records:
        if not record.consumes_attempt:
            continue
        for component in record.components:
            entry = counts.setdefault(component.component_id, [0, 0])
            entry[0] += 1
            entry[1] += int(component.failure_reason == TRUNCATED)
    return {cid: (n, truncated) for cid, (n, truncated) in counts.items()}


def think_stats(raw: str, count_tokens: Callable[[str], int], family: Optional[ThinkingFamily] = None
                ) -> tuple[int, bool]:
    """Return the thinking's token count (the text before the last closing marker of `family`, Qwen3's `</think>` when
    None) and whether it was empty."""
    close, opening = (THINK_CLOSE, THINK_OPEN) if family is None else (family.think_close, family.raw_think_open)
    reasoning = raw.rsplit(close, 1)[0] if close in raw else ""
    return count_tokens(reasoning), not reasoning.strip().removeprefix(opening).strip()


def _full_object(
    where: str, settled: Optional[tuple[AttemptRecord, ComponentRecord]], counts: tuple[int, int], raw: Mapping[str, Any],
    condition: ConditionSpec, verdict_schema: ComponentSchema, map_schema: ComponentSchema,
    count_tokens: Callable[[str], int], thinking: Optional[ThinkingFamily] = None,
) -> dict[str, Any]:
    attempts, truncated = counts
    full: dict[str, Any] = {"verdict": None, "attempts": attempts, "truncated_attempts": truncated}
    if settled is None:
        return full
    record, component = settled
    row = raw.get((record.semantic_attempt_id, record.request_block_id))
    if row is None or raw_output_sha256(row.get("raw_response")) != component.raw_output_sha256:
        raise TableError(f"{where}: the accepted full-set output is missing from the raw outputs or is not the text "
                         "the ledger recorded")
    text = strict_think_answer(row["raw_response"]) if condition.strict_think else row["completion"]
    labelled = verdict_schema.classify_forced(text, require_map=False)
    if not labelled.valid or labelled.judgment != component.judgment:
        raise TableError(f"{where}: the stored full-set output reads as {labelled.judgment!r}, the ledger says "
                         f"{component.judgment!r}")
    mapped = map_schema.classify_forced(text, require_map=True)
    think, empty = think_stats(row["raw_response"], count_tokens, thinking)
    return {**full, "verdict": component.judgment, "map": dict(mapped.rule_judgments) if mapped.valid else {},
            "map_complete": mapped.valid, "map_failure": None if mapped.valid else mapped.failure_reason,
            "think_tokens": think, "empty_think": empty}


def _rules_object(
    settled: Mapping[str, tuple[AttemptRecord, ComponentRecord]], counts: Mapping[str, tuple[int, int]],
    rule_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}
    for rule in rule_ids:
        attempts, truncated = counts.get(rule, (0, 0))
        rules[rule] = {"label": settled[rule][1].judgment if rule in settled else None, "attempts": attempts,
                       "truncated_attempts": truncated}
    return rules


def read_table(
    source: DevCondition,
    kind: TableKind,
    *,
    rules: Sequence[Mapping[str, Any]],
    rule_ids: Sequence[str],
    draws: int,
    scenario_ids: Sequence[str],
    ruleset_sha256: str,
    count_tokens: Optional[Callable[[str], int]] = None,
    thinking: Optional[ThinkingFamily] = None,
) -> list[dict[str, Any]]:
    """Return the response table of `source` as `kind`, one row per scenario and draw; another scenario set, draw count
    or ruleset, a draw still running, a changed raw output, or a disagreement with the materialised views is refused.
    """
    from ruleforge.evaluate.discrimination import map_schema

    if kind.evaluation_mode != source.condition.evaluation_mode:
        raise TableError(f"a {kind.name} table is read from an {kind.evaluation_mode!r} condition, not "
                         f"{source.condition.evaluation_mode!r}")
    if source.condition.rule_set_sha256 != ruleset_sha256:
        raise TableError(f"{source.root}/{source.condition.condition_id} judged ruleset "
                         f"{source.condition.rule_set_sha256[:12]}..., not the settings' ({ruleset_sha256[:12]}...)")
    if sorted(source.scenario_ids) != sorted(scenario_ids):
        raise TableError(f"{source.root} judged {len(source.scenario_ids)} scenario(s), not the {kind.name} set's "
                         f"{len(scenario_ids)}")
    if int(source.manifest["expected_decoder_draws"]) != draws:
        raise TableError(f"{source.root} ran {source.manifest['expected_decoder_draws']} draw(s); a {kind.name} "
                         f"table has {draws}")
    if kind.full and count_tokens is None:
        raise TableError(f"a {kind.name} table counts thinking tokens: it needs the model's tokenizer")
    condition = source.condition
    verdict_schema = ComponentSchema.for_condition(condition, rules)
    mapping = map_schema(condition, rules)
    max_attempts = int(source.manifest["max_semantic_attempts"])
    table: list[dict[str, Any]] = []
    for draw in range(draws):
        where = f"{source.root}/{condition.condition_id}/{draw_dirname(draw)}"
        ledger = DrawLedger(source.root / condition.condition_id / draw_dirname(draw),
                            campaign_id=source.manifest["campaign_id"], condition_id=condition.condition_id,
                            decoder_draw_id=draw)
        state = ledger.load()
        raw: dict[str, dict[tuple[int, str], Mapping[str, Any]]] = {}
        for entry in ledger.iter_raw() if kind.full else ():
            attempt = (int(entry["semantic_attempt_id"]), entry["request_block_id"])
            raw.setdefault(entry["scenario_id"], {})[attempt] = entry
        records: dict[str, list[AttemptRecord]] = {}
        for record in state.records:
            records.setdefault(record.scenario_id, []).append(record)
        for sid in source.scenario_ids:
            label = f"{where} {sid[:12]}"
            if not state.is_accepted(sid) and state.next_attempt(sid) < max_attempts:
                raise TableError(f"{label}: neither accepted nor exhausted -- the draw is still running")
            settled, counts = _settled(state, sid), _attempts(records.get(sid, ()))
            row: dict[str, Any] = {"scenario_id": sid, "draw": draw}
            if kind.full:
                row["full"] = _full_object(label, settled.get(FULL_RULESET_COMPONENT),
                                           counts.get(FULL_RULESET_COMPONENT, (0, 0)), raw.get(sid, {}), condition,
                                           verdict_schema, mapping, count_tokens, thinking)
            if kind.rules:
                row["rules"] = _rules_object(settled, counts, rule_ids)
            table.append(row)
    if source.complete and source.manifest["campaign_mode"] == CAMPAIGN_MODE_PILOT:
        _cross_check(source, kind, rule_ids, table)
    return table


def _cross_check(source: DevCondition, kind: TableKind, rule_ids: Sequence[str], table: Sequence[Mapping[str, Any]]
                 ) -> None:
    from ruleforge.grpo.anchor_set import AnchorError, open_condition, read_bank, read_full_set

    views: dict[tuple[str, int], dict[str, Any]] = {}
    try:
        anchored = open_condition(source.root, source.condition.condition_id, evaluation_mode=kind.evaluation_mode)
        if kind.evaluation_mode == EVALUATION_MODE_FULL:
            for sid, verdicts in read_full_set(anchored, rule_ids).items():
                for draw, verdict in enumerate(verdicts):
                    views[(sid, draw)] = {"verdict": verdict}
        else:
            for sid, draws in read_bank(anchored, rule_ids).items():
                for draw, judgments in enumerate(draws):
                    views[(sid, draw)] = {"labels": {r: j.label for r, j in judgments.items()}}
    except AnchorError as error:
        raise TableError(str(error)) from error
    for row in table:
        view = views[(row["scenario_id"], row["draw"])]
        read = ({"verdict": row["full"]["verdict"]} if kind.full
                else {"labels": {rule: entry["label"] for rule, entry in row["rules"].items()}})
        if read != view:
            raise TableError(f"{source.root}/{source.condition.condition_id}: {row['scenario_id'][:12]} draw "
                             f"{row['draw']} reads {read} from the ledger and {view} from the materialised views")


def table_identity(condition: ConditionSpec) -> dict[str, Any]:
    """Return the identity a table's sidecar records: the condition's weights and adapter digests and dtype, and the
    digest of the condition less its id and evaluation mode, which a base's bank and full-set campaigns share."""
    if condition.serving_dtype is None:
        raise TableError(f"{condition.condition_id} declares no serving dtype: the report's tables name the engine "
                         "they were served on (dtype float32 or bfloat16 in the judge's config)")
    return {"model_weights_sha256": condition.model_weights_sha256, "adapter_sha256": condition.adapter_sha256,
            "serving_dtype": condition.serving_dtype,
            "condition_sha256": canonical_digest(condition_projection(condition, side=SIDE_DEV))}


def write_table(path: Path | str, table: Sequence[Mapping[str, Any]], condition: ConditionSpec) -> dict[str, Any]:
    """Write a table and its identity sidecar: the rows to a temporary file, the sidecar, then the rename, so a table
    that exists always has its identity."""
    identity = table_identity(condition)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    os.close(descriptor)
    partial = Path(name)
    try:
        write_jsonl_records(partial, table)
        written = write_table_identity(path, **identity, contents=partial)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return written


def _tokens(path: Path) -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path))
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grpo_report_tables",
                                     description="Prepare and read the report's dev campaigns.")
    sub = parser.add_subparsers(dest="command", required=True)
    pops = sub.add_parser("populations", help="the scenario table and each dev campaign's scenario file")
    pops.add_argument("--settings", type=Path, required=True,
                      help="the report's settings (configs/training/report.yaml)")
    pops.add_argument("--dev", type=Path, required=True, help="the development scenarios")
    pops.add_argument("--dev-nonyes", type=Path, help="development scenarios whose base judgment is not yes")
    pops.add_argument("--out-dir", type=Path, required=True)
    read = sub.add_parser("read", help="a response table (and its identity) from one condition of a dev campaign")
    read.add_argument("--settings", type=Path, required=True,
                      help="the report's settings (configs/training/report.yaml)")
    read.add_argument("--scenarios", type=Path, required=True, help=f"the scenario table ({SCENARIOS_FNAME})")
    read.add_argument("--root", type=Path, required=True, help="the campaign root")
    read.add_argument("--policy", required=True, help="the condition's declared policy: a save's label, or base")
    read.add_argument("--kind", choices=sorted(KINDS), required=True)
    read.add_argument("--rules", type=Path, required=True, help="the settings' ruleset")
    read.add_argument("--tokenizer", type=Path, help="the model directory whose tokenizer counts thinking tokens")
    read.add_argument("--out", type=Path, required=True)
    return parser


def _run_populations(args: argparse.Namespace) -> dict[str, Any]:
    settings = read_settings(args.settings)
    rows = stage1_table(scenario_table(args.dev, args.dev_nonyes), settings)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_records(args.out_dir / SCENARIOS_FNAME, rows)
    sources = [path for path in (args.dev, args.dev_nonyes) if path is not None]
    return write_populations(stage_sets(_scenarios_of(rows), settings), sources, args.out_dir)


def _run_read(args: argparse.Namespace) -> dict[str, Any]:
    kind = KINDS[args.kind]
    settings = read_settings(args.settings)
    sets = stage_sets(read_scenarios(args.scenarios), settings)
    rule_ids = rule_ids_of(args.rules, settings["ruleset_sha256"])
    condition_id = find_condition(args.root, policy=args.policy, evaluation_mode=kind.evaluation_mode)
    source = open_dev_condition(args.root, condition_id, evaluation_mode=kind.evaluation_mode)
    table = read_table(source, kind, rules=load_jsonl_records(args.rules), rule_ids=rule_ids,
                       draws=expected_draws(kind, settings), scenario_ids=sets[kind.scenario_set],
                       ruleset_sha256=settings["ruleset_sha256"],
                       count_tokens=None if args.tokenizer is None else _tokens(args.tokenizer),
                       thinking=None if args.tokenizer is None else directory_family(args.tokenizer))
    identity = write_table(args.out, table, source.condition)
    return {"table": str(args.out), "rows": len(table), "identity": identity,
            "exhausted_draws": [d for d, state in enumerate(source.draw_states) if state != STATE_COMPLETE]}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = _run_populations(args) if args.command == "populations" else _run_read(args)
    print(json.dumps(summary, indent=1))
    return 0
