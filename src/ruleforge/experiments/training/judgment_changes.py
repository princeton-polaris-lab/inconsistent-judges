from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Mapping, Sequence

from ruleforge.core.errors import RuleForgeError
from ruleforge.evaluate.report import TRANSITION_LABELS, ScoringError, anchor_stratum, judgment_transitions
from ruleforge.experiments.training.family_arms import family_golden_main, read_family_arms, read_family_golden
from ruleforge.experiments.training.frozen_anchors import KINDS, arm_labels
from ruleforge.experiments.training.paper_families import PAPER_OUTPUTS, PaperFamily

SOURCE = "judgment_changes"
SCHEMA_VERSION = 1
#: The decoder draw the panels read from each arm.
DRAW = 0
COMPARISONS = KINDS
#: Each kind's key in ``judgment_transitions``, which calls the aggregate the anchor.
TRANSITIONS = {"per_rule": "per_rule", "aggregate": "anchors", "full_set": "full_set"}
DEFINITION = (
    f"Decoder draw {DRAW} of each arm, as its accepted bundle outcomes hold it. Change rate: the share of paired "
    "judgments whose label differs between the base arm and the trained arm (ruleforge.evaluate.report."
    "judgment_transitions, rows the base, columns the trained): individual-rule judgments paired by scenario and "
    "rule, full-ruleset judgments by scenario, over the pairs whose two labels are both valid (yes, unsure or no; "
    "every pair, since an accepted bundle holds no other label). Counts: the trained arm's labels at the same draw. "
    "Each arm's draw is its own temperature-1 sample, so a change rate holds the decoder's sampling variation beside "
    "the training's. Draw to draw: each arm's change rate between two of its own draws, paired and counted alike, "
    "for every pair of its draws, pooled over the pairs (each pair of draws holds the same n, so the pooled rate is "
    "their mean), with the lowest and the highest pair's rate: the change sampling alone gives. The aggregated rule "
    "judgment (aggregate; anchors in transitions) is a bundle's largest rule judgment under no < unsure < yes, paired "
    "by scenario and counted like the full-ruleset judgments; a bundle whose stored max_rule_phi or contradiction is "
    "not its labels' is refused."
)


class JudgmentChangesError(ValueError, RuleForgeError):
    """A family's arms cannot be read as this golden reads them."""


def _rows(cells: Mapping[tuple[str, int], Mapping[str, Any]], draw: int) -> dict[str, dict[str, Any]]:
    return {
        scenario: {"full_set_judgment": row["full_set_judgment"],
                   "rules": [{"rule_id": rule, "judgment": label} for rule, label in row["rule_judgments"].items()]}
        for (scenario, index), row in cells.items() if index == draw
    }


def _counts(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    labels = {"per_rule": [rule["judgment"] for row in rows.values() for rule in row["rules"]],
              "aggregate": [anchor_stratum(row) for row in rows.values()],
              "full_set": [row["full_set_judgment"] for row in rows.values()]}
    return {kind: {**{label: found.count(label) for label in TRANSITION_LABELS}, "n": len(found)}
            for kind, found in labels.items()}


def _changes(base: Mapping[str, Mapping[str, Any]], other: Mapping[str, Mapping[str, Any]],
             where: str) -> dict[str, Any]:
    try:
        transitions = judgment_transitions(base, other)
    except ScoringError as error:
        raise JudgmentChangesError(f"{where}: {error}") from error
    summaries = {kind: transitions["summaries"][TRANSITIONS[kind]] for kind, _ in COMPARISONS}
    return {"transitions": transitions,
            **{kind: {"n": summary["n"], "changed": summary["n"] - summary["unchanged"]}
               for kind, summary in summaries.items()}}


def _draw_to_draw(cells: Mapping[tuple[str, int], Mapping[str, Any]], n_draws: int, where: str) -> dict[str, Any]:
    draws = [_rows(cells, draw) for draw in range(n_draws)]
    pairs = []
    for first, second in itertools.combinations(range(n_draws), 2):
        changes = _changes(draws[first], draws[second], f"{where} draws {first} and {second}")
        pairs.append({"draws": [first, second], **{kind: changes[kind] for kind, _ in COMPARISONS}})
    pooled = {}
    for kind, _ in COMPARISONS:
        n, changed = sum(pair[kind]["n"] for pair in pairs), sum(pair[kind]["changed"] for pair in pairs)
        rates = [pair[kind]["changed"] / pair[kind]["n"] for pair in pairs]
        pooled[kind] = {"n": n, "changed": changed, "change_rate": changed / n, "min_change_rate": min(rates),
                        "max_change_rate": max(rates)}
    return {"n_pairs": len(pairs), **pooled, "pairs": pairs}


def _require_stored_aggregates(cells: Mapping[tuple[str, int], Mapping[str, Any]], n_draws: int, where: str) -> None:
    scenarios = sorted({scenario for scenario, _ in cells})
    arm_labels(cells, scenarios, n_draws, list(cells[(scenarios[0], DRAW)]["rule_judgments"]), where,
               error=JudgmentChangesError)


def family_changes(cells: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]], reference: str, member: str,
                   n_draws: int, where: str) -> dict[str, Any]:
    """Return the changes from the arm ``reference`` to the arm ``member`` at :data:`DRAW`, each arm's draw-to-draw
    rates and the trained arm's counts, from the two arms' bundle outcomes by arm label."""
    rows = {label: _rows(cells[label], DRAW) for label in (reference, member)}
    changes = _changes(rows[reference], rows[member], where)
    draw_to_draw = {label: _draw_to_draw(cells[label], n_draws, f"{where} arm {label}") for label in (reference, member)}
    for label in (reference, member):
        _require_stored_aggregates(cells[label], n_draws, f"{where} arm {label}")
    return {
        "changes": {kind: {"label": label, **changes[kind], "change_rate": changes[kind]["changed"] / changes[kind]["n"]}
                    for kind, label in COMPARISONS},
        "draw_to_draw": draw_to_draw,
        "member_counts": _counts(rows[member]),
        "transitions": changes["transitions"],
    }


def family_readout(family: PaperFamily, where: str) -> dict[str, Any]:
    """Return one family's judgment changes, draw-to-draw rates and trained-arm counts at :data:`DRAW`."""
    arms = read_family_arms(family, where, error=JudgmentChangesError)
    reference, member = arms.reference.label, arms.member.label
    return {
        "reference": reference,
        "member": member,
        "labels": arms.labels(),
        **family_changes(arms.cells, reference, member, arms.n_draws, where),
        "provenance": arms.provenance(),
    }


def build_golden(names: Sequence[str], population: str) -> dict[str, Any]:
    """Return the golden of the paper families ``names``, each read from its roots renamed for ``population``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "definition": DEFINITION,
        "draw": DRAW,
        "population": population,
        "families": {name: family_readout(PAPER_OUTPUTS[name].family.on_population(population), f"family {name}")
                     for name in names},
    }


def read_golden(path: Path) -> dict[str, Any]:
    """Return the golden at ``path``, refusing one this module did not write."""
    return read_family_golden(path, source=SOURCE, schema_version=SCHEMA_VERSION, noun="judgment-changes",
                              error=JudgmentChangesError)


def main(argv: list[str] | None = None) -> int:
    return family_golden_main(argv, noun="judgment-changes", build_golden=build_golden, read_golden=read_golden,
                              figure_module="ruleforge.experiments.training.judgment_changes_figure",
                              panels={"changes": "the changes", "counts": "the counts"})


if __name__ == "__main__":
    raise SystemExit(main())
