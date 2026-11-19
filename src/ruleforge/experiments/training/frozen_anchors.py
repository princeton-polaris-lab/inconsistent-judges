from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ruleforge.core.errors import RuleForgeError
from ruleforge.evaluate.inference import scenario_cluster_bootstrap, statistic_interval
from ruleforge.evaluate.report import TRANSITION_LABELS
from ruleforge.evaluate.scoring import PHI_MAP
from ruleforge.experiments.checks import require
from ruleforge.experiments import arm_records as rp
from ruleforge.experiments.training.family_arms import family_golden_main, read_family_arms, read_family_golden
from ruleforge.experiments.training.paper_families import PAPER_OUTPUTS, PaperFamily

SOURCE = "frozen_anchors"
SCHEMA_VERSION = 1
#: The labels in the order the aggregate takes their maximum in: no < unsure < yes.
ORDER = ("no", "unsure", "yes")
KINDS = (("per_rule", "Individual-rule judgments"), ("aggregate", "Aggregated rule judgments"),
         ("full_set", "Full-ruleset judgments"))
#: The paired differences of the rates: (first, second) for first minus second. always_yes is an always-yes judge's
#: rate against the frozen anchors.
DIFFERENCES = {"base_minus_trained_frozen": ("base", "trained_frozen"),
               "base_minus_trained_updated": ("base", "trained_updated"),
               "trained_frozen_minus_trained_updated": ("trained_frozen", "trained_updated"),
               "trained_frozen_minus_always_yes": ("trained_frozen", "always_yes")}
ANCHORS = ("yes", "not_yes")
#: The first key of the features only the statistic reads: the pairs behind each split rate.
PAIRS = "anchor_pairs"
DEFINITION = (
    "Each family pairs its base arm (the reference) with its trained arm over the same scenarios and decoder draws, "
    "from each arm's accepted bundle outcomes. A bundle holds one full-ruleset judgment F and one separate-call "
    "judgment R_k per rule; its aggregate A is the largest R_k under no < unsure < yes. Every label is yes, unsure or "
    "no, and a bundle whose stored max_rule_phi or contradiction is not its labels' is refused. Each number is a mean "
    "over scenarios of a per-scenario value, or (by_anchor) a ratio of two such means. A rate between two sets of "
    "labels averages, per scenario, over all D x D pairs of a draw of one with a draw of the other; a rate within one "
    "arm's labels, over the D(D-1) ordered pairs of two different draws of it: exact averages over the draws at hand, "
    "each estimating the rate between independent temperature-1 draws (an arm's F against its own A counts the pairs "
    "within one bundle too, which estimate the same rate because each call of a bundle is sampled on a seed of its "
    "own). Inconsistency rates, the share of pairs in which F differs from A: base, the base arm's F against its own "
    "A, which are the frozen anchors; trained_updated, the trained arm's F against its own A; trained_frozen, the "
    "trained arm's F against the base arm's A. differences: base minus trained_frozen (the reduction left when "
    "evaluation keeps the frozen anchors), base minus trained_updated, trained_frozen minus trained_updated, and "
    "trained_frozen minus always_yes, an always-yes judge's rate against the frozen anchors (the base arm's "
    "always_yes), which trained_frozen compares with directly. by_anchor: base and trained_frozen again over only the "
    "pairs whose frozen anchor is yes, or is not yes: the share of those pairs in which F differs from A (undefined in "
    "a resample that holds none), with share, those pairs' share of all pairs, and base minus trained_frozen. "
    "same_draw: each arm's F against its own A at the same draw, averaged over draws, which is its root's own flat "
    "rate (its interval is this golden's bootstrap's, not the root's). Change rates, the share of pairs whose labels "
    "differ, for the individual-rule judgments (R_k paired by rule, averaged over the rules), the aggregate A and the "
    "full-ruleset judgment F: base_to_trained, the base arm's labels against the trained arm's; draw_to_draw, each "
    "arm's labels against its own at a different draw; resampling, the mean of the two arms' draw_to_draw; excess, "
    "base_to_trained minus resampling, whose expectation for one scenario is half the squared distance between the two "
    "arms' label distributions: the part of the change that is a shift in the judgment's distribution, not resampling "
    "(a per-scenario value may be negative). label_shares: the share of each arm's bundles whose F, or whose A, is "
    "each label. always_yes: an always-yes judge's inconsistency rate against each arm's aggregate, the share of its "
    "bundles whose A is not yes. Intervals: one scenario-cluster bootstrap per family over all its per-scenario "
    "values, at the reference root's bootstrap settings (bootstrap), so that every difference's interval is paired; "
    "percentile intervals, no tests."
)


class FrozenAnchorsError(ValueError, RuleForgeError):
    """A family's arms cannot be read as this golden reads them."""


def arm_labels(cells: Mapping[tuple[str, int], Mapping[str, Any]], scenario_ids: Sequence[str], n_draws: int,
               rule_ids: Sequence[str], where: str, *,
               error: type[Exception] = FrozenAnchorsError) -> tuple[np.ndarray, np.ndarray]:
    """Return an arm's labels as indices into :data:`ORDER`, draws last: the full-ruleset judgments ``(N, D)`` and the
    rule judgments ``(N, K, D)``; refused as ``error`` unless every bundle holds ``rule_ids``' judgments in yes, unsure
    or no, and its stored aggregate and inconsistency are its labels'."""
    index = {label: position for position, label in enumerate(ORDER)}
    full = np.empty((len(scenario_ids), n_draws), dtype=np.int8)
    rules = np.empty((len(scenario_ids), len(rule_ids), n_draws), dtype=np.int8)
    for row_index, scenario in enumerate(scenario_ids):
        for draw in range(n_draws):
            row, at = cells[(scenario, draw)], f"scenario {scenario[:12]} draw {draw}"
            judgments = row.get("rule_judgments")
            require([(isinstance(judgments, Mapping) and set(judgments) == set(rule_ids),
                      f"{at} does not hold one judgment for each of the family's {len(rule_ids)} rules")],
                    where, error=error)
            labels = [row.get("full_set_judgment"), *(judgments[rule] for rule in rule_ids)]
            if unknown := [label for label in labels if label not in index]:
                raise error(f"{where}: {at} holds the label {unknown[0]!r}, not yes, unsure or no")
            full[row_index, draw] = index[labels[0]]
            rules[row_index, :, draw] = [index[label] for label in labels[1:]]
            top = int(rules[row_index, :, draw].max())
            require([(row.get("max_rule_phi") == PHI_MAP[ORDER[top]],
                      f"{at}: its max_rule_phi {row.get('max_rule_phi')!r} is not its rule judgments' largest "
                      f"({ORDER[top]})"),
                     (row.get("contradiction") == float(full[row_index, draw] != top),
                      f"{at}: its contradiction {row.get('contradiction')!r} is not whether its full-ruleset "
                      f"judgment ({labels[0]}) differs from its aggregate ({ORDER[top]})")],
                    where, error=error)
    return full, rules


def cross(first: np.ndarray, second: np.ndarray, counted: np.ndarray | None = None) -> np.ndarray:
    """Return each scenario's share of draw pairs, every draw of ``first`` with every draw of ``second``, whose labels
    differ (and, given ``counted``, whose draw of ``second`` it marks); draws are the last axis, and the axes between
    it and the scenarios' are averaged over."""
    differ = first[..., :, None] != second[..., None, :]
    if counted is not None:
        differ &= counted[..., None, :]
    return differ.mean(axis=tuple(range(1, differ.ndim)))


def within(labels: np.ndarray) -> np.ndarray:
    """Return each scenario's share of ordered pairs of two different draws of ``labels`` whose labels differ."""
    differ = labels[..., :, None] != labels[..., None, :]
    return differ.reshape(differ.shape[0], -1).sum(axis=1) / (labels[0].size * (labels.shape[-1] - 1))


def _features(full: Mapping[str, np.ndarray], rules: Mapping[str, np.ndarray], reference: str,
              member: str) -> dict[tuple[str, ...], np.ndarray]:
    aggregate = {label: arm_rules.max(axis=1) for label, arm_rules in rules.items()}
    kinds = {"per_rule": rules, "aggregate": aggregate, "full_set": full}
    yes = ORDER.index("yes")
    features = {("inconsistency", "base"): cross(full[reference], aggregate[reference]),
                ("inconsistency", "trained_updated"): cross(full[member], aggregate[member]),
                ("inconsistency", "trained_frozen"): cross(full[member], aggregate[reference])}
    for label in (reference, member):
        features[("inconsistency", "same_draw", label)] = (full[label] != aggregate[label]).mean(axis=1)
    for kind, _ in KINDS:
        features[("changes", kind, "base_to_trained")] = cross(kinds[kind][member], kinds[kind][reference])
        for label in (reference, member):
            features[("changes", kind, "draw_to_draw", label)] = within(kinds[kind][label])
    for label in (reference, member):
        for which, labels in (("full_set", full[label]), ("aggregate", aggregate[label])):
            for share in TRANSITION_LABELS:
                features[("label_shares", label, which, share)] = (labels == ORDER.index(share)).mean(axis=1)
        features[("always_yes", label)] = (aggregate[label] != yes).mean(axis=1)
    frozen_yes = aggregate[reference] == yes
    for anchor, counted in zip(ANCHORS, (frozen_yes, ~frozen_yes)):
        for rate, judged in (("base", reference), ("trained_frozen", member)):
            features[(PAIRS, anchor, rate)] = cross(full[judged], aggregate[reference], counted)
    return features


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def _statistics(names: Sequence[tuple[str, ...]], reference: str,
                member: str) -> tuple[tuple[tuple[str, ...], ...], Callable[[np.ndarray], np.ndarray]]:
    """Return the golden's places in order, the features' first (but for :data:`PAIRS`), and the statistic that
    computes them from feature means: the features themselves, the rates' differences, the rates by frozen anchor,
    and each kind's resampling and excess."""
    shown = [name for name in names if name[0] != PAIRS]
    derived = [("inconsistency", "differences", name) for name in DIFFERENCES]
    derived += [("inconsistency", "by_anchor", anchor, part) for anchor in ANCHORS
                for part in ("share", "base", "trained_frozen", "base_minus_trained_frozen")]
    derived += [("changes", kind, part) for kind, _ in KINDS for part in ("resampling", "excess")]
    column = {name: position for position, name in enumerate(names)}

    def statistic(means: np.ndarray) -> np.ndarray:
        values = {name: means[:, position] for name, position in column.items()}
        rates = {name: values[("inconsistency", name)] for name in ("base", "trained_updated", "trained_frozen")}
        rates["always_yes"] = values[("always_yes", reference)]
        for name, (first, second) in DIFFERENCES.items():
            values[("inconsistency", "differences", name)] = rates[first] - rates[second]
        shares = {"yes": values[("label_shares", reference, "aggregate", "yes")],
                  "not_yes": values[("always_yes", reference)]}
        for anchor in ANCHORS:
            place = ("inconsistency", "by_anchor", anchor)
            values[(*place, "share")] = shares[anchor]
            for rate in ("base", "trained_frozen"):
                values[(*place, rate)] = _ratio(values[(PAIRS, anchor, rate)], shares[anchor])
            values[(*place, "base_minus_trained_frozen")] = (values[(*place, "base")]
                                                             - values[(*place, "trained_frozen")])
        for kind, _ in KINDS:
            resampling = (values[("changes", kind, "draw_to_draw", reference)]
                          + values[("changes", kind, "draw_to_draw", member)]) / 2
            values[("changes", kind, "resampling")] = resampling
            values[("changes", kind, "excess")] = values[("changes", kind, "base_to_trained")] - resampling
        return np.column_stack([values[name] for name in (*shown, *derived)])

    return (*shown, *derived), statistic


def _nest(entries: Mapping[tuple[str, ...], Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for path, value in entries.items():
        node = nested
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return nested


def family_numbers(cells: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]], reference: str, member: str,
                   scenario_ids: Sequence[str], n_draws: int, bootstrap: Mapping[str, Any],
                   where: str) -> dict[str, Any]:
    """Return a family's rates, changes and shares with their intervals from one joint scenario-cluster bootstrap
    (``bootstrap``: ``n_resamples``, ``ci_level``, ``seed``), its two arms' bundle outcomes by arm label."""
    expected = {(scenario, draw) for scenario in scenario_ids for draw in range(n_draws)}
    require([(reference != member, f"both arms are labelled {reference!r}"),
             (n_draws >= 2, f"{n_draws} decoder draw; the draw-to-draw rates need two or more"),
             *[(set(cells[label]) == expected, f"arm {label}'s bundles are not the {len(scenario_ids)} scenarios x "
                                               f"{n_draws} draws both arms are read over") for label in (reference, member)]],
            where, error=FrozenAnchorsError)
    first = cells[reference][(scenario_ids[0], 0)].get("rule_judgments")
    require([(isinstance(first, Mapping) and bool(first), f"arm {reference}'s scenario {scenario_ids[0][:12]} draw 0 "
                                                          "holds no rule judgments")], where, error=FrozenAnchorsError)
    rule_ids = list(first)
    full, rules = {}, {}
    for label in (reference, member):
        full[label], rules[label] = arm_labels(cells[label], scenario_ids, n_draws, rule_ids, f"{where} arm {label}")
    features = _features(full, rules, reference, member)
    names, statistic = _statistics(list(features), reference, member)
    point, replicates = scenario_cluster_bootstrap(np.column_stack(list(features.values())), statistic, **bootstrap)
    numbers = _nest({name: statistic_interval(point[position], replicates[:, position], bootstrap["ci_level"])
                     for position, name in enumerate(names)})
    for kind, label in KINDS:
        numbers["changes"][kind]["label"] = label
    return {"n_scenarios": len(scenario_ids), "n_rules": len(rule_ids), **numbers}


def family_readout(family: PaperFamily, where: str) -> dict[str, Any]:
    """Return one family's inconsistency rates against frozen and updated anchors and its expected change rates."""
    arms = read_family_arms(family, where, error=FrozenAnchorsError)
    reference, member = arms.reference, arms.member
    first = arms.roots[reference.root]
    settings = rp.bootstrap_settings(first.payload)
    bootstrap = {key: settings[key] for key in ("n_resamples", "ci_level", "seed")}
    numbers = family_numbers(arms.cells, reference.label, member.label, first.scenario_ids, arms.n_draws, bootstrap,
                             where)
    same_draw = numbers["inconsistency"]["same_draw"]
    flat = {arm.label: arms.roots[arm.root].payload["conditions"][arms.specs[arm.label].condition_id]["estimate"]
            for arm in family.arms}
    require([(math.isclose(same_draw[label]["estimate"], flat[label], rel_tol=0.0, abs_tol=1e-12),
              f"arm {label}'s same-draw rate {same_draw[label]['estimate']} is not its root's flat rate {flat[label]}")
             for label in flat], where, error=FrozenAnchorsError)
    return {
        "reference": reference.label,
        "member": member.label,
        "labels": arms.labels(),
        **numbers,
        "bootstrap": {**bootstrap, "root": reference.root},
        "provenance": arms.provenance(),
    }


def build_golden(names: Sequence[str], population: str) -> dict[str, Any]:
    """Return the golden of the paper families ``names``, each read from its roots renamed for ``population``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "definition": DEFINITION,
        "population": population,
        "families": {name: family_readout(PAPER_OUTPUTS[name].family.on_population(population), f"family {name}")
                     for name in names},
    }


def read_golden(path: Path) -> dict[str, Any]:
    """Return the golden at ``path``, refusing one this module did not write."""
    return read_family_golden(path, source=SOURCE, schema_version=SCHEMA_VERSION, noun="frozen-anchors",
                              error=FrozenAnchorsError)


def main(argv: list[str] | None = None) -> int:
    return family_golden_main(argv, noun="frozen-anchors", build_golden=build_golden, read_golden=read_golden,
                              figure_module="ruleforge.experiments.training.frozen_anchors_figure",
                              panels={"rates": "the inconsistency rates against each anchor"})


if __name__ == "__main__":
    raise SystemExit(main())
