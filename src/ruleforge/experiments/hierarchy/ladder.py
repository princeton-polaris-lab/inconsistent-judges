from __future__ import annotations

from functools import partial
from itertools import accumulate
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Sequence, Tuple

import numpy as np

from ruleforge.evaluate.inference import (INFERENCE_METHOD, PRIMARY_INTERVAL_BASIS, grid_from_cells,
                                          hierarchical_bootstrap)
from ruleforge.evaluate.scoring import PHI_MAP
from ruleforge.experiments.checks import require
from ruleforge.experiments.goldens import INTERVAL_COLUMNS, format_interval, interval_record
from ruleforge.experiments.hierarchy import analysis as ha
from ruleforge.experiments.hierarchy.analysis import HierarchyError, JudgePair, bundle_labels, disclosures
from ruleforge.experiments import arm_records as rp
from ruleforge.rulesets.levels import LEVEL_FIELD, LEVEL_ORDER, LEVEL_SOURCE, RuleLevels

__all__ = ["CSV_COLUMNS", "LAST_RUNG", "RUNGS", "SCHEMA_VERSION", "CellLadder", "build_payload", "cell_ladder",
           "csv_rows", "rung_label", "summary_markdown"]

SCHEMA_VERSION = 1
#: Rung k keeps the top k levels; the last keeps every level and is the arm's inconsistency rate.
RUNGS = tuple(range(1, len(LEVEL_ORDER) + 1))
LAST_RUNG = f"rung {RUNGS[-1]}"
RUNG_GRIDS = tuple(f"rung_{k}" for k in RUNGS)
UNSURE_GRIDS = tuple(f"under_unsure_{k}" for k in RUNGS)
OVER_GRID = "over"

_require = partial(require, error=HierarchyError)

DEFINITION = (
    "phi(yes)=1, phi(unsure)=0.5, phi(no)=0. Per (scenario, decoder draw) of a judge's flat or hierarchy arm in the "
    "hierarchy comparison: F its full-set judgment, M the maximum of phi over its per-rule judgments, and M_k the "
    f"maximum over the rules of the top k levels ({' > '.join(LEVEL_ORDER)}). "
    "The per-rule judgments are the arm's own: "
    "on the flat arm each rule judged alone; on the hierarchy arm, whose full set shows the levels, each rule judged "
    "with its superiors (a root rule with none). Every per-rule prompt of a hierarchy arm, a root rule's too, also explains "
    "the ranking and names the rule's level, so even its root rung's per-rule judgments are not its flat arm's: at "
    "every rung, a judge's hierarchy - flat difference mixes the full-set prompt's effect with the per-rule prompts'. "
    "At rung k the cell is inconsistent iff phi(F) < M_k or phi(F) > M: the full set misses a violation of a rule at "
    "the top k levels, or exceeds every per-rule judgment. A miss of a rule below the top k levels is excused. "
    f"{LAST_RUNG.capitalize()} is the arm's stored inconsistency phi(F) != M cell "
    "for cell and takes its published estimate and interval: the campaign's own scenario-cluster bootstrap at the "
    "same settings, run over the campaign's conditions. Every other rung, the unsure misses and the over-flag share "
    "one scenario-cluster bootstrap over the arm's grids, so a difference between rungs carries no interval here. "
    "under_unsure is the rate of misses at rung k with M_k = 0.5: no rule of the top k levels says yes, and an unsure "
    "one exceeds a no full set."
)


def rung_label(levels: Sequence[str]) -> str:
    """Return a rung's name: the first rung's one level, then the level each rung adds (``+user``)."""
    return levels[-1] if len(levels) == 1 else f"+{levels[-1]}"


class CellLadder(NamedTuple):
    """One cell's over-flag (phi(F) > M) and, per rung, its under-flag (phi(F) < M_k) and that flag where M_k is 0.5."""

    over: bool
    under: Tuple[bool, ...]
    under_unsure: Tuple[bool, ...]

    @property
    def inconsistent(self) -> Tuple[bool, ...]:
        """Return, per rung, whether the cell is inconsistent there: its over-flag or that rung's under-flag."""
        return tuple(self.over or under for under in self.under)


def cell_ladder(full_judgment: str, rule_judgments: Mapping[str, str], rank: Mapping[str, int]) -> CellLadder:
    """Return a cell's ladder from its labels; ``rank`` is each rule's level index (0 the root)."""
    phi_full = PHI_MAP[full_judgment]
    tops = [0.0] * len(LEVEL_ORDER)
    for rule_id, label in rule_judgments.items():
        tops[rank[rule_id]] = max(tops[rank[rule_id]], PHI_MAP[label])
    m_k = list(accumulate(tops, max))
    under = tuple(phi_full < m for m in m_k)
    return CellLadder(phi_full > m_k[-1], under, tuple(flag and m == PHI_MAP["unsure"] for flag, m in zip(under, m_k)))


def _arm_ladder(judge: str, name: str, arm: ha.Arm, levels: RuleLevels, settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Return one arm's rates down the rungs, refused unless the last rung is its stored grid cell for cell and its
    estimates' rate, on the primary interval basis."""
    root, spec, stored = arm.root, arm.spec, arm.grid
    where = f"{judge} {name} ({spec.public_id} in {root.uri})"
    summary = root.payload["conditions"][spec.condition_id]
    _require([(summary.get("interval_basis") == PRIMARY_INTERVAL_BASIS,
               f"its interval basis is {summary.get('interval_basis')!r}, not {PRIMARY_INTERVAL_BASIS!r}"),
              (root.payload.get("inference_method") == INFERENCE_METHOD,
               f"its campaign's inference method is {root.payload.get('inference_method')!r}, not "
               f"{INFERENCE_METHOD!r}")], where)
    rank, rule_ids = levels.rank, frozenset(levels.rule_ids)
    names = (*RUNG_GRIDS, *UNSURE_GRIDS, OVER_GRID)
    values: Dict[str, Dict[Tuple[str, int], float]] = {grid: {} for grid in names}
    for key, row in arm.cells.items():
        ladder = cell_ladder(*bundle_labels(row, rule_ids, f"{where} {key[0][:12]}/draw {key[1]}"), rank)
        for grid, flag in zip(names, (*ladder.inconsistent, *ladder.under_unsure, ladder.over)):
            values[grid][key] = float(flag)
    grids = {grid: grid_from_cells(spec.condition_id, f"hierarchy_ladder:{grid}", root.scenario_ids, stored.n_draws, cell)
             for grid, cell in values.items()}
    last = grids.pop(RUNG_GRIDS[-1])
    _require([(np.array_equal(last.values, stored.values),
               f"{LAST_RUNG} is not its stored inconsistency (the stored contradiction) cell for cell"),
              (last.rate == summary["estimate"],
               f"{LAST_RUNG}'s rate {last.rate!r} is not its estimates' {summary['estimate']!r}")], where)
    boot = hierarchical_bootstrap(grids, shared_inner=[list(grids)], **settings)
    rates = {grid: interval_record(boot["conditions"][grid]) for grid in grids}
    rates[RUNG_GRIDS[-1]] = interval_record(summary)
    return {
        "judge": judge,
        "model_name": spec.model_name,
        "arm": name,
        "prompt_style": spec.prompt_style,
        "individual_rule_context": spec.individual_rule_context,
        "condition_id": spec.public_id,
        "root": root.uri,
        "campaign_id": root.payload["campaign_id"],
        "label_rule": spec.label_rule,
        "n_scenarios": stored.n_scenarios,
        "n_decoder_draws": stored.n_draws,
        "grid_digest": stored.digest,
        "rungs": [{"rung": k, "levels": list(LEVEL_ORDER[:k]), "rate": rates[grid], "under_unsure": rates[unsure]}
                  for k, grid, unsure in zip(RUNGS, RUNG_GRIDS, UNSURE_GRIDS)],
        "over": rates[OVER_GRID],
    }


def build_payload(*, pairs: Sequence[JudgePair], rules_path: Path | str, levels_path: Path | str,
                  allow_incomplete: bool = False) -> Dict[str, Any]:
    """Return the ladder of every judge's flat and hierarchy arms the hierarchy comparison's ``pairs`` name, in paper
    order, each arm read and checked as that comparison reads it."""
    levels, roots, read = ha.read_pairs(pairs, rules_path, levels_path, allow_incomplete=allow_incomplete,
                                        where="hierarchy ladder")
    arms = [(judge, name, arm) for judge, *sides in read for name, arm in zip((ha.FLAT, ha.HIERARCHY), sides)]
    settings = {key: value for key, value in rp.bootstrap_settings(next(iter(roots.values())).payload).items()
                if key != "n_permutations"}
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "hierarchy_ladder",
        "definition": DEFINITION,
        "levels": {"order": list(LEVEL_ORDER), "source": LEVEL_SOURCE, "field": LEVEL_FIELD,
                   "rules": levels.rules_by_level()},
        "rungs": [{"rung": k, "levels": list(LEVEL_ORDER[:k])} for k in RUNGS],
        "arms": [_arm_ladder(*arm, levels, settings) for arm in arms],
        "provenance": {**ha.pairs_provenance(pairs, levels, roots, allow_incomplete=allow_incomplete),
                       "bootstrap": settings},
    }


CSV_COLUMNS = ("judge", "arm", "condition_id", "root", "rung", "levels",
               *(f"{name}{suffix}" for name in ("rate", "under_unsure", "over") for suffix, _ in INTERVAL_COLUMNS))


def csv_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return one row per arm and rung, in :data:`CSV_COLUMNS`."""
    rows = []
    for arm in payload["arms"]:
        for rung in arm["rungs"]:
            row = {"judge": arm["judge"], "arm": arm["arm"], "condition_id": arm["condition_id"], "root": arm["root"],
                   "rung": rung["rung"], "levels": "+".join(rung["levels"])}
            for name, interval in (("rate", rung["rate"]), ("under_unsure", rung["under_unsure"]), ("over", arm["over"])):
                for suffix, key in INTERVAL_COLUMNS:
                    row[f"{name}{suffix}"] = interval[key]
            rows.append(row)
    return rows


def _rung_table(payload: Mapping[str, Any], quantity: str, *, over: bool) -> List[str]:
    head = " | ".join(rung_label(rung["levels"]) for rung in payload["rungs"])
    lines = [f"| Judge | Arm | {head} |" + (" over-flag |" if over else ""),
             "|---|---|" + "---|" * (len(payload["rungs"]) + over)]
    for arm in payload["arms"]:
        lines.append(f"| {arm['judge']} | {arm['arm']} | "
                     + " | ".join(format_interval(rung[quantity]) for rung in arm["rungs"]) + " |"
                     + (f" {format_interval(arm['over'])} |" if over else ""))
    return lines


def summary_markdown(payload: Mapping[str, Any]) -> str:
    """Return the payload's Markdown summary: each arm's rate and unsure misses at every rung, and its over-flag."""
    lines = ["# Inconsistency rate by hierarchy level", "", payload["definition"], "",
             *_rung_table(payload, "rate", over=True),
             "", "## Misses where the top levels' highest label is unsure (under_unsure)", "",
             *_rung_table(payload, "under_unsure", over=False)]
    amendments = disclosures(payload)
    if amendments:
        lines += ["", "## Disclosures", "", *amendments]
    return "\n".join(lines) + "\n"
