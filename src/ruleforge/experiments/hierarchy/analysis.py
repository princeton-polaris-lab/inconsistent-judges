from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ruleforge.experiments import arm_records as rp
from ruleforge.experiments.campaign_roots import (PROVIDER, Root, arm_outcomes, arm_stack, check_campaign_settings,
                                                open_roots, registered_condition)
from ruleforge.experiments.checks import require
from ruleforge.experiments.goldens import format_interval, interval_record
from ruleforge.experiments.labels import campaign_label, display_order
from ruleforge.evaluate.estimates import (
    CampaignEstimatesError,
    amendment_sentence,
    estimates_source,
)
from ruleforge.evaluate.multiplicity import ALPHA
from ruleforge.evaluate.inference import (
    INFERENCE_METHOD,
    PRIMARY_INTERVAL_BASIS,
    TEST_METHOD,
    DrawGrid,
    grid_from_cells,
    hierarchical_bootstrap,
    holm_adjust,
)
from ruleforge.evaluate.scoring import PHI_MAP
from ruleforge.core.paths import is_artifact_uri
from ruleforge.evaluate.aggregate import contrast_row
from ruleforge.evaluate.identity import MANIFEST_AMENDMENTS_KEY, ConditionSpec
from ruleforge.evaluate.stack import contrast_conflict
from ruleforge.prompts.styles import HIERARCHY_RULE_CONTEXT, NAIVE_HIERARCHY, NAIVE_RULES_ABOVE
from ruleforge.rulesets.levels import LEVEL_FIELD, LEVEL_ORDER, LEVEL_SOURCE, RuleLevels, RuleLevelsError, load_rule_levels

__all__ = [
    "APPROVED_PROMPT_SHA256",
    "CSV_COLUMNS",
    "Arm",
    "CellOutcome",
    "HierarchyError",
    "JudgePair",
    "build_payload",
    "bundle_labels",
    "cell_outcome",
    "compare",
    "csv_rows",
    "disclosures",
    "pairs_provenance",
    "read_pairs",
    "summary_markdown",
]

SCHEMA_VERSION = 2
FLAT, HIERARCHY = "flat", "hierarchy"
#: Each side's prompt style and rule context.
SIDES = {FLAT: (NAIVE_RULES_ABOVE, rp.ISOLATED), HIERARCHY: (NAIVE_HIERARCHY, HIERARCHY_RULE_CONTEXT)}
#: The condition fields a judge's two arms may differ in; everything else must be equal.
ARM_FIELDS = ("prompt_style", "individual_rule_context", "rule_levels_sha256")
#: The approved hierarchy prompts (test fixtures) by sha256, recorded in the provenance; prompts are not re-rendered.
APPROVED_PROMPT_SHA256 = {
    "approved_full_set_hier.txt": "e42c8e6069b02f2aab7885ad8bce3a3b772d8c2554e66a06a00551e920cb1afe",
    "approved_rule43_user_hier.txt": "de1f51c90170b7293cf2d4c7ad9dba9184342002dad2a4bb1102b47024b45f33",
    "approved_rule28_root_hier.txt": "3497da55e9c69501a61cf5e2a7b805a2eeb12a6b1667eacf8d29f5e5b965d99c",
}
#: The 2x2's cells, (full-set side, reference side) to name; F against C is the mismatched cell.
TWO_BY_TWO = {("F", "I"): FLAT, ("F_H", "C"): HIERARCHY, ("F_H", "I"): "hierarchy_full_set_isolated_rules",
              ("F", "C"): "mismatched_flat_full_set_conditioned_rules"}
MISMATCHED = TWO_BY_TWO[("F", "C")]
#: The root level's index in LEVEL_ORDER.
ROOT_RANK = 0

DEFINITION = (
    "phi(yes)=1, phi(unsure)=0.5, phi(no)=0. Flat (separate-call): F the full-set judgment, I_r rule r's judgment "
    "judged alone, M = max_r phi(I_r); a (scenario, decoder draw) is inconsistent iff phi(F) != M. Hierarchy: F_H "
    "the full-set judgment with every rule's level shown and the ranking explained, C_r rule r's judgment judged with "
    "its superiors (the rules at strictly higher levels of the Model Spec's chain of command; rules of its own level "
    "never), M_H = max_r phi(C_r); hierarchy-inconsistent iff phi(F_H) != M_H. Both are the separate-call computation "
    "(calculate_inconsistency, each accepted bundle's stored contradiction) on their own condition's bundles: unsure "
    "handling, validity, attrition, the label rule and the clustering by scenario are the flat rate's."
)
PAIRING = (
    "The flat and hierarchy arms are paired by (scenario, decoder draw index). The primary contrast's unit is the "
    "scenario (its mean over its draws), each arm's draws resampled independently (two generations); the secondary "
    "cells pair draw d of one arm with draw d of the other, by index only (each condition's seeds derive from its "
    "own id), and share their decoder resample by that index."
)
MULTIPLICITY = (
    "primary: one family, the paired differences hierarchy - flat of the judges analysed, each with its scenario-"
    "clustered paired randomization p, Holm-adjusted across them (alpha 0.05). secondary: descriptive, intervals "
    "only, no test and no multiplicity family"
)
SECONDARY = (
    "Descriptive, no test: one joint scenario-cluster bootstrap per judge over the 2x2, the over/under split, the "
    "root floor and the per-level shift (their decoder resample shared by draw index). The 2x2's flat and hierarchy "
    "cells are the primary rates, with this bootstrap's intervals; the primary intervals are the estimates'."
)


class HierarchyError(CampaignEstimatesError):
    """The arms, the ruleset or the levels cannot make the comparison as stated."""


_require = partial(require, error=HierarchyError)


def bundle_labels(row: Mapping[str, Any], rule_ids: frozenset, where: str) -> Tuple[str, Mapping[str, str]]:
    """Return a bundle's full-set label and rule labels, refusing any label outside yes/no/unsure or any rule set
    other than the ruleset's."""
    full, judgments = row.get("full_set_judgment"), row.get("rule_judgments")
    if not (isinstance(judgments, Mapping) and set(judgments) == rule_ids):
        raise HierarchyError(f"{where}: its rule_judgments do not name exactly the ruleset's rules")
    if full not in PHI_MAP or any(label not in PHI_MAP for label in judgments.values()):
        raise HierarchyError(f"{where}: a judgment outside {sorted(PHI_MAP)}")
    return full, judgments


class CellOutcome(NamedTuple):
    """One bundle's outcome: phi of the full set, the maximum over the rules, and for an under-flag the level index of
    the highest-ranked rule that reached the maximum."""

    phi_full: float
    phi_max: float
    top_rank: Optional[int]

    @property
    def inconsistent(self) -> bool:
        return self.phi_full != self.phi_max

    @property
    def over(self) -> bool:
        return self.phi_full > self.phi_max

    @property
    def under(self) -> bool:
        return self.phi_full < self.phi_max

    @property
    def root_floor(self) -> bool:
        """Return whether this is an over-flag or an under-flag a root rule reached: one no hierarchy can excuse."""
        return self.over or self.top_rank == ROOT_RANK


def cell_outcome(full_judgment: str, rule_judgments: Mapping[str, str], rank: Mapping[str, int]) -> CellOutcome:
    """Return the outcome of a full-set label against rule labels (``rank``: each rule's level index, 0 the root)."""
    phi_full = PHI_MAP[full_judgment]
    phi = {rule_id: PHI_MAP[label] for rule_id, label in rule_judgments.items()}
    phi_max = max(phi.values())
    if phi_full >= phi_max:
        return CellOutcome(phi_full, phi_max, None)
    return CellOutcome(phi_full, phi_max, min(rank[rule_id] for rule_id, value in phi.items() if value == phi_max))


@dataclass(frozen=True)
class Arm:
    """One arm as read: its root, its registered condition, its bundles by (scenario, draw) and its stored grid."""

    root: Root
    spec: ConditionSpec
    cells: Mapping[Tuple[str, int], Mapping[str, Any]]
    grid: DrawGrid
    stack: Mapping[str, Any]


@dataclass(frozen=True)
class JudgePair:
    """A judge's two arms, each a (root URI, public condition id)."""

    flat: Tuple[str, str]
    hierarchy: Tuple[str, str]


def _check_pair(flat: Any, hierarchy: Any, levels: RuleLevels, where: str) -> None:
    differing = rp.differences_beyond_style(flat, hierarchy, exempt=ARM_FIELDS)
    _require(
        [
            ((flat.prompt_style, flat.individual_rule_context) == SIDES[FLAT] and flat.rule_levels_sha256 is None,
             f"the flat arm {flat.public_id} is {flat.prompt_style!r} with its rules judged "
             f"{flat.individual_rule_context!r}, not {SIDES[FLAT]}"),
            ((hierarchy.prompt_style, hierarchy.individual_rule_context) == SIDES[HIERARCHY],
             f"the hierarchy arm {hierarchy.public_id} is {hierarchy.prompt_style!r} with its rules judged "
             f"{hierarchy.individual_rule_context!r}, not {SIDES[HIERARCHY]}"),
            (hierarchy.rule_levels_sha256 == levels.levels_file["sha256"],
             f"the hierarchy arm read levels file {str(hierarchy.rule_levels_sha256)[:12]}..., not "
             f"{levels.levels_file['name']} ({levels.levels_file['sha256'][:12]}...)"),
            (flat.rule_set_sha256 == levels.ruleset["sha256"],
             f"judged against ruleset {flat.rule_set_name} ({flat.rule_set_sha256[:12]}), not "
             f"{levels.ruleset['name']} ({levels.ruleset['sha256'][:12]})"),
            (not differing, f"its arms differ in {differing}, not only in {list(ARM_FIELDS)}"),
        ],
        where,
    )


def _read_arms(pair: JudgePair, roots: Mapping[str, Root], levels: RuleLevels, n_draws: int,
               ) -> Tuple[str, Arm, Arm]:
    specs = {side: registered_condition(roots[uri], cid, f"{side} arm {cid} in {uri}")
             for side, (uri, cid) in ((FLAT, pair.flat), (HIERARCHY, pair.hierarchy))}
    judge = campaign_label(specs[FLAT].model_name)
    where = f"{judge}: flat {pair.flat[1]} in {pair.flat[0]}, hierarchy {pair.hierarchy[1]} in {pair.hierarchy[0]}"
    _check_pair(specs[FLAT], specs[HIERARCHY], levels, where)
    check_campaign_settings([roots[uri] for uri in dict.fromkeys((pair.flat[0], pair.hierarchy[0]))], where)
    arms = {}
    for side, (uri, _) in ((FLAT, pair.flat), (HIERARCHY, pair.hierarchy)):
        root, spec = roots[uri], specs[side]
        cells, grid = arm_outcomes(root, spec, n_draws, where)
        arms[side] = Arm(root, spec, cells, grid, arm_stack(root, spec, where))
    conflict = contrast_conflict(arms[FLAT].stack["signatures"], arms[HIERARCHY].stack["signatures"])
    _require([(conflict is None, f"its arms' judgments come from different inference stacks or serving settings "
                                 f"({conflict}): the flat control is a same-stack control")], where)
    return judge, arms[FLAT], arms[HIERARCHY]


def _arm_record(arm: Arm) -> Dict[str, Any]:
    spec = arm.spec
    return {"root": arm.root.uri, "campaign_id": arm.root.payload["campaign_id"], "condition_id": spec.public_id,
            "prompt_style": spec.prompt_style, "individual_rule_context": spec.individual_rule_context,
            "rule_levels_sha256": spec.rule_levels_sha256, "label_rule": spec.label_rule,
            "condition_fingerprint": spec.fingerprint, "grid_digest": arm.grid.digest,
            "stack": {key: value for key, value in arm.stack.items() if key != "signatures"}}


def compare(judge: str, flat: Arm, hierarchy: Arm, levels: RuleLevels, settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Return one judge's record (the primary rates and difference, and the secondary quantities), refused unless each
    arm's cells equal its stored grid and its rate its estimates' rate."""
    where = f"{judge} ({flat.spec.public_id} against {hierarchy.spec.public_id})"
    ids, n_draws = flat.grid.scenario_ids, flat.grid.n_draws
    _require([(hierarchy.grid.scenario_ids == ids and hierarchy.grid.n_draws == n_draws,
               "the two arms do not cover the same scenarios and draws")], where)
    rank, rule_ids = levels.rank, frozenset(levels.rule_ids)
    by_level = {level: ids_ for level, ids_ in levels.rules_by_level().items() if ids_}
    values: Dict[str, Dict[Tuple[str, int], float]] = {}
    under_levels: Counter = Counter()

    def put(name: str, key: Tuple[str, int], value: float) -> None:
        values.setdefault(name, {})[key] = float(value)

    for key in flat.cells:
        at = f"{where} {key[0][:12]}/draw {key[1]}"
        full, alone = bundle_labels(flat.cells[key], rule_ids, f"{at} (flat)")
        full_h, conditioned = bundle_labels(hierarchy.cells[key], rule_ids, f"{at} (hierarchy)")
        outcomes = {(side, reference): cell_outcome(verdict, judgments, rank)
                    for side, verdict in (("F", full), ("F_H", full_h))
                    for reference, judgments in (("I", alone), ("C", conditioned))}
        for cell, name in TWO_BY_TWO.items():
            put(name, key, outcomes[cell].inconsistent)
        for side, cell in ((FLAT, outcomes[("F", "I")]), (HIERARCHY, outcomes[("F_H", "C")])):
            put(f"{side}_over", key, cell.over)
            put(f"{side}_under", key, cell.under)
        hier = outcomes[("F_H", "C")]
        put("root_floor", key, hier.root_floor)
        if hier.under:
            under_levels[LEVEL_ORDER[hier.top_rank]] += 1
        for level, members in by_level.items():
            put(f"shift_{level}", key, float(np.mean([PHI_MAP[conditioned[r]] - PHI_MAP[alone[r]] for r in members])))
    grids = {name: grid_from_cells(f"{judge}/{name}", f"hierarchy:{name}", ids, n_draws, cells)
             for name, cells in values.items()}
    for side, arm in ((FLAT, flat), (HIERARCHY, hierarchy)):
        summary = arm.root.payload["conditions"][arm.spec.condition_id]
        _require([(np.array_equal(grids[side].values, arm.grid.values),
                   f"the {side} arm's recomputed inconsistency is not its stored contradiction cell for cell"),
                  (grids[side].rate == summary["estimate"],
                   f"the {side} arm's rate {grids[side].rate!r} is not its estimates' {summary['estimate']!r}")], where)
    primary = contrast_row({HIERARCHY: hierarchy.grid, FLAT: flat.grid}, HIERARCHY, FLAT, shared=False, **settings)
    boot = hierarchical_bootstrap(grids, shared_inner=[list(grids)],
                                  **{key: settings[key] for key in ("n_resamples", "ci_level", "seed")})
    secondary = {name: interval_record(boot["conditions"][name]) for name in grids}
    n_under = sum(under_levels.values())
    n_cells = len(ids) * n_draws
    return {
        "judge": judge,
        "model_name": flat.spec.model_name,
        "flat": _arm_record(flat),
        "hierarchy": _arm_record(hierarchy),
        "n_scenarios": len(ids),
        "n_decoder_draws": n_draws,
        "primary": {
            "flat_rate": interval_record(flat.root.payload["conditions"][flat.spec.condition_id]),
            "hierarchy_rate": interval_record(hierarchy.root.payload["conditions"][hierarchy.spec.condition_id]),
            "difference": {key: primary[key] for key in (
                "estimate", "ci_low", "ci_high", "sd", "se_analytic", "p_value", "n_permutations", "inner_resample",
                "grid_digest_a", "grid_digest_b")},
            "difference_of": f"{HIERARCHY} - {FLAT}",
        },
        "secondary": {
            "two_by_two": {name: {**secondary[name], "full_set": side, "reference": reference,
                                  "mismatched": name == MISMATCHED}
                           for (side, reference), name in TWO_BY_TWO.items()},
            "over_under": {side: {"over": secondary[f"{side}_over"], "under": secondary[f"{side}_under"]}
                           for side in (FLAT, HIERARCHY)},
            "root_floor": secondary["root_floor"],
            "level_shift": {level: {**secondary[f"shift_{level}"], "n_rules": len(members),
                                    "placebo": level == LEVEL_ORDER[0]}
                            for level, members in by_level.items()},
            "under_flags_by_top_level": {
                level: {"count": under_levels[level], "rate": under_levels[level] / n_cells,
                        "share_of_under_flags": under_levels[level] / n_under if n_under else None}
                for level in LEVEL_ORDER},
        },
    }


def read_pairs(pairs: Sequence[JudgePair], rules_path: Path | str, levels_path: Path | str, *, allow_incomplete: bool,
               where: str) -> Tuple[RuleLevels, Dict[str, Root], List[Tuple[str, Arm, Arm]]]:
    """Return the levels, the roots ``pairs`` name, and each judge's label and two arms in paper order, read and
    checked; a judge given twice is refused."""
    _require([(bool(pairs), "no judge's arms given")], where)
    uris = [uri for pair in pairs for uri, _ in (pair.flat, pair.hierarchy)]
    _require([(is_artifact_uri(uri), f"root {uri!r} is not an artifact:// URI") for uri in uris], where)
    try:
        levels = load_rule_levels(rules_path, levels_path)
    except RuleLevelsError as error:
        raise HierarchyError(str(error)) from error
    roots = open_roots({uri: uri for uri in uris}, where, allow_incomplete=allow_incomplete,
                          allow_stack_mismatch=False)
    n_draws = int(next(iter(roots.values())).payload["n_decoder_draws"])
    read = [_read_arms(pair, roots, levels, n_draws) for pair in pairs]
    judges = [judge for judge, *_ in read]
    repeated = sorted({judge for judge in judges if judges.count(judge) > 1})
    _require([(not repeated, f"{repeated} given twice: a judge has one flat and one hierarchy arm")], where)
    order = display_order(judges)
    return levels, roots, sorted(read, key=lambda entry: order.index(entry[0]))


def pairs_provenance(pairs: Sequence[JudgePair], levels: RuleLevels, roots: Mapping[str, Root], *,
                     allow_incomplete: bool) -> Dict[str, Any]:
    """Return the provenance every build from ``pairs`` records: the pairs, levels, prompts, campaigns and settings."""
    first = next(iter(roots.values())).payload
    used: Dict[str, List[str]] = {}
    for pair in pairs:
        for uri, cid in (pair.flat, pair.hierarchy):
            used.setdefault(uri, []).append(cid)
    return {
        "pairs": [{FLAT: list(pair.flat), HIERARCHY: list(pair.hierarchy)} for pair in pairs],
        "ruleset": levels.ruleset,
        "levels": levels.levels_file,
        "approved_prompts_sha256": dict(APPROVED_PROMPT_SHA256),
        "campaigns": [
            {**estimates_source(root.payload), "root": root.uri,
             "label_replay": rp.replay_provenance(root.path), "conditions": sorted(used[root.uri])}
            for root in roots.values()
        ],
        "scenario_manifest_sha256": first["scenario_manifest_sha256"],
        "n_scenarios": first["n_scenarios"],
        "n_decoder_draws": int(first["n_decoder_draws"]),
        "inference_method": INFERENCE_METHOD,
        "interval_basis": PRIMARY_INTERVAL_BASIS,
        "allow_incomplete": allow_incomplete,
    }


def build_payload(
    *,
    pairs: Sequence[JudgePair],
    rules_path: Path | str,
    levels_path: Path | str,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Return the comparison of every judge's ``pairs`` of arms, in paper order: each judge's record, Holm over
    the judges' differences, and the provenance."""
    levels, roots, read = read_pairs(pairs, rules_path, levels_path, allow_incomplete=allow_incomplete,
                                     where="hierarchy rate")
    settings = rp.bootstrap_settings(next(iter(roots.values())).payload)
    records = [compare(judge, flat, hierarchy, levels, settings) for judge, flat, hierarchy in read]
    adjusted = holm_adjust([{"p_value": record["primary"]["difference"]["p_value"]} for record in records],
                           alpha=ALPHA)
    for record, row in zip(records, adjusted, strict=True):
        record["primary"]["difference"].update(
            p_holm=row["p_holm"], significant_holm=row["significant_holm"], test_method=row["test_method"],
            family_size=len(records))
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "hierarchy",
        "definition": DEFINITION,
        "pairing": PAIRING,
        "multiplicity": MULTIPLICITY,
        "secondary": SECONDARY,
        "levels": {"order": list(LEVEL_ORDER), "source": LEVEL_SOURCE, "field": LEVEL_FIELD,
                   "rules": levels.rules_by_level()},
        "judges": records,
        "provenance": {**pairs_provenance(pairs, levels, roots, allow_incomplete=allow_incomplete),
                       "bootstrap": settings, "test_method": TEST_METHOD},
    }


CSV_COLUMNS = ("judge", "model_name", "n_scenarios", "family", "quantity", "estimate", "ci_low", "ci_high", "p_value",
               "p_holm", "significant_holm", "flat_condition", "hierarchy_condition")


def _quantities(record: Mapping[str, Any]) -> List[Tuple[str, str, Mapping[str, Any]]]:
    primary, secondary = record["primary"], record["secondary"]
    rows = [("primary", "flat_rate", primary["flat_rate"]), ("primary", "hierarchy_rate", primary["hierarchy_rate"]),
            ("primary", "difference_hierarchy_minus_flat", primary["difference"])]
    rows += [("secondary", f"two_by_two:{name}", cell) for name, cell in secondary["two_by_two"].items()]
    rows += [("secondary", f"{direction}:{side}", split[direction])
             for side, split in secondary["over_under"].items() for direction in ("over", "under")]
    rows += [("secondary", "root_floor", secondary["root_floor"])]
    rows += [("secondary", f"level_shift:{level}", shift) for level, shift in secondary["level_shift"].items()]
    rows += [("secondary", f"under_flags_top_level:{level}", {"estimate": top["rate"], "ci_low": "", "ci_high": ""})
             for level, top in secondary["under_flags_by_top_level"].items()]
    return rows


def csv_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return one row per judge and quantity, in :data:`CSV_COLUMNS`."""
    return [
        {"judge": record["judge"], "model_name": record["model_name"], "n_scenarios": record["n_scenarios"],
         "family": family, "quantity": quantity,
         **{key: interval[key] for key in ("estimate", "ci_low", "ci_high")},
         **{key: interval.get(key, "") for key in ("p_value", "p_holm", "significant_holm")},
         "flat_condition": record["flat"]["condition_id"], "hierarchy_condition": record["hierarchy"]["condition_id"]}
        for record in payload["judges"] for family, quantity, interval in _quantities(record)
    ]


def _control(record: Mapping[str, Any]) -> str:
    flat, hierarchy = record[FLAT]["stack"], record[HIERARCHY]["stack"]
    if flat["kind"] != PROVIDER:
        return f"same stack ({flat['short']})"

    def span(stack: Mapping[str, Any]) -> str:
        dates = stack["dates"]
        return f"{str(dates['first_request_block'])[:10]} to {str(dates['last_request_block'])[:10]}"

    return f"published arm at the provider (requests {span(flat)}; hierarchy {span(hierarchy)})"


def summary_markdown(payload: Mapping[str, Any]) -> str:
    """Return the payload's Markdown summary: the primary family, then the secondary quantities."""
    order = payload["levels"]["order"]
    lines = [
        "# Hierarchy inconsistency rate",
        "",
        f"{payload['definition']} {payload['pairing']}",
        "",
        "## Primary (Holm over the judges)",
        "",
        "| Judge | flat rate | hierarchy rate | hierarchy - flat | p | p (Holm) | flat control |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in payload["judges"]:
        primary = record["primary"]
        difference = primary["difference"]
        lines.append(f"| {record['judge']} | {format_interval(primary['flat_rate'])} | {format_interval(primary['hierarchy_rate'])} | "
                     f"{format_interval(difference, True)} | {difference['p_value']:.4g} | {difference['p_holm']:.4g}"
                     f"{' *' if difference['significant_holm'] else ''} | {_control(record)} |")
    lines += ["", "## Secondary (descriptive)", "", payload["secondary"], "",
              "| Judge | F vs I (flat) | F_H vs C (hierarchy) | F_H vs I | F vs C (mismatched) | root floor |",
              "|---|---|---|---|---|---|"]
    for record in payload["judges"]:
        cells = record["secondary"]["two_by_two"]
        lines.append(f"| {record['judge']} | " + " | ".join(format_interval(cells[name]) for name in TWO_BY_TWO.values())
                     + f" | {format_interval(record['secondary']['root_floor'])} |")
    lines += ["", "Over/under flags (flat; hierarchy), and the hierarchy's under-flags by the level of the "
                  "highest-ranked rule that set M_H:", "",
              "| Judge | over | under | under-flags by level |", "|---|---|---|---|"]
    for record in payload["judges"]:
        split, tops = record["secondary"]["over_under"], record["secondary"]["under_flags_by_top_level"]
        levels = ", ".join(f"{level} {tops[level]['count']}" for level in order if tops[level]["count"])
        lines.append(f"| {record['judge']} | {format_interval(split[FLAT]['over'])}; {format_interval(split[HIERARCHY]['over'])} | "
                     f"{format_interval(split[FLAT]['under'])}; {format_interval(split[HIERARCHY]['under'])} | {levels or '-'} |")
    lines += ["", "Per-level shift, mean phi(C_r) - phi(I_r) over each level's rules (root: the placebo):", "",
              "| Judge | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for record in payload["judges"]:
        shifts = record["secondary"]["level_shift"]
        lines.append(f"| {record['judge']} | "
                     + " | ".join(format_interval(shifts[level], True) if level in shifts else "-" for level in order) + " |")
    amendments = disclosures(payload)
    if amendments:
        lines += ["", "## Disclosures", "", *amendments]
    return "\n".join(lines) + "\n"


def disclosures(payload: Mapping[str, Any]) -> List[str]:
    """Return one line per retry-cap amendment of a campaign read."""
    return [f"- {amendment_sentence(campaign['campaign_id'], amendment)}"
            for campaign in payload["provenance"]["campaigns"]
            for amendment in campaign.get(MANIFEST_AMENDMENTS_KEY) or ()]
