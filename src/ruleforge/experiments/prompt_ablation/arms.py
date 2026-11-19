from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ruleforge.experiments import arm_records as rp
from ruleforge.evaluate.estimates import (
    SHARED_DESIGN_FIELDS,
    CampaignEstimatesError,
    estimates_source,
    is_complete,
)
from ruleforge.experiments.checks import require
from ruleforge.experiments.campaign_roots import (
    Root,
    arm_outcomes,
    arm_stack,
    check_campaign_settings,
    open_roots,
    public_id_of,
    registered_condition,
    short_stack,
)
from ruleforge.experiments.labels import NAIVE_SLOT, SLOTS, STYLE_LABELS, campaign_label, display_order
from ruleforge.evaluate.multiplicity import ALPHA
from ruleforge.experiments.prompt_ablation.analysis import (
    BH_METHOD,
    FORCED_B,
    PROMPT_BARS,
    BAR_POPULATION,
    HOLM_MC_MARGIN_NOTE,
    HOLM_METHOD,
    SAME_VS_SEPARATE_KEYS,
    SLOT_STYLES,
    TABLE_CONTRASTS,
    TABLE_METRIC,
    TABLE_SCHEMA_VERSION,
    TABLE_SOURCE,
    add_table_multiplicity,
    rate_bar,
    table_row,
)
from ruleforge.experiments.paper_outputs import SLOT_NAMES
from ruleforge.prompts.styles import FORCED
from ruleforge.evaluate.inference import (
    INFERENCE_METHOD,
    TEST_METHOD,
    DrawGrid,
)
from ruleforge.core.record_names import SAME_CALL_RATE_KEY, SEPARATE_VS_SAME_CALL_KEY
from ruleforge.core.digests import file_record
from ruleforge.core.paths import is_artifact_uri
from ruleforge.evaluate.aggregate import contrast_row
from ruleforge.evaluate.identity import (
    CAMPAIGN_MODE_FINAL,
    RETRY_SCOPE_SCENARIO_BUNDLE,
    ConditionSpec,
)
from ruleforge.evaluate.stack import (
    STACK_PACKAGES,
    across_conditions,
    contrast_conflict,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ARM_MAP_VERSION",
    "ArmMap",
    "ArmMapError",
    "add_arm_map_var_argument",
    "arm_map_var_refusal",
    "arm_map_variables_record",
    "build_payload",
    "check_arm",
    "judges_by_label",
    "load_arm_map",
    "paper_table_refusal",
    "parse_variables",
    "validate_cross_stack",
]

ARM_MAP_VERSION = 1
#: The one arm whose contrasts an arm map may declare cross-stack.
CROSS_STACK_ARM = NAIVE_SLOT
TOP_KEYS = ("version", "roots", "judges", "cross_stack_reason", "variables")
#: A root URI's placeholder for a variable (``${name}``), a variable's name, and a value: a fragment of a root's name.
PLACEHOLDER = re.compile(r"\$\{([^}]*)\}")
VARIABLE_NAME = re.compile(r"[a-z][a-z0-9_]*")
VARIABLE_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
JUDGE_KEYS = (*SLOT_NAMES, "cross_stack")
ARM_KEYS = ("root", "condition")

DEFINITION = (
    "The prompt-ablation table, built from an arm map: per judge, its naive, encouraged and forced arms, each a "
    "condition of a named campaign root; the rows are keyed by the prompt slots (naive, encouraged, forced) and "
    "each names the prompt style, condition and root of its two arms. Every arm's flat grid is rebuilt from its "
    "draws and matched to its estimates by digest; each contrast is baseline minus treatment, the scenario-cluster "
    "bootstrap interval and the scenario-clustered paired randomization p at the campaigns' shared settings "
    "(contrast_row, the aggregate's own function), equal to the aggregate's row where both arms share a root that "
    "pairs them; Holm (p_holm) and Benjamini-Hochberg (p_bh) over every row of the table. A judge's arms share "
    "one condition identity but the prompt style (the label rule included). cross_stack marks a row whose arms ran on "
    "different local inference stacks (their packages only, never a serving setting), allowed only for the naive arm "
    "against the judge's encouraged or forced arm and only where the arm map declares it (cross_stack_reason); every "
    "other contrast is held to one stack and serving setup. A judge's arms' roots share the campaign settings that "
    "shape an arm's outcomes (attempts, retry scope, request block, seed base)."
)


class ArmMapError(CampaignEstimatesError):
    """The arm map cannot make the table as stated."""


_require = partial(require, error=ArmMapError)


@dataclass(frozen=True)
class Arm:
    """One condition of one root, as the arm map names it."""

    root: str
    condition_id: str


@dataclass(frozen=True)
class JudgeArms:
    """One judge's arms: ``arms`` by slot (:data:`~ruleforge.experiments.paper_outputs.SLOTS`)."""

    model_name: str
    arms: Dict[str, Arm]
    cross_stack: bool


@dataclass(frozen=True)
class ArmMap:
    """A parsed arm map: its roots (key -> ``artifact://`` URI) and its judges."""

    path: Path
    roots: Dict[str, str]
    judges: Dict[str, JudgeArms]
    cross_stack_reason: Optional[str]
    named: Tuple[Arm, ...]
    #: The values its declared variables were given (empty for a map that declares none).
    variables: Dict[str, str] = field(default_factory=dict)


def _mapping(value: Any, where: str, keys: Sequence[str], required: Sequence[str]) -> Mapping[str, Any]:
    _require([(isinstance(value, Mapping), f"{type(value).__name__} is not a mapping")], where)
    unknown = sorted(str(key) for key in value if key not in keys)
    missing = [key for key in required if key not in value]
    _require([(not unknown, f"unknown key(s) {unknown} (it takes {list(keys)})"),
              (not missing, f"no {missing}")], where)
    return value


def _arm(value: Any, where: str, roots: Mapping[str, str]) -> Arm:
    arm = _mapping(value, where, ARM_KEYS, ARM_KEYS)
    root, condition = arm["root"], arm["condition"]
    _require(
        [
            (isinstance(root, str) and root in roots, f"root {root!r} is not one of the map's roots {sorted(roots)}"),
            (isinstance(condition, str) and bool(condition), f"condition {condition!r} is not a condition id"),
        ],
        where,
    )
    return Arm(root, condition)


def _load_yaml(text: str, where: str) -> Any:
    """Return ``yaml.safe_load``, refusing a mapping key that is not a string or that one mapping holds twice."""
    import yaml

    class StrictLoader(yaml.SafeLoader):
        def construct_mapping(self, node: Any, deep: bool = False) -> Any:
            self.flatten_mapping(node)
            seen: List[str] = []
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                at = f"line {key_node.start_mark.line + 1}"
                _require([(isinstance(key, str), f"{at}: key {key!r} is not a name")], where)
                _require([(key not in seen, f"{at}: key {key!r} appears twice in one mapping (an arm map names "
                                            "each root, judge and slot once)")], where)
                seen.append(key)
            return super().construct_mapping(node, deep=deep)

    try:
        return yaml.load(text, Loader=StrictLoader)
    except yaml.YAMLError as error:
        raise ArmMapError(f"{where}: cannot read it: {error}") from error


def _resolve_variables(top: Mapping[str, Any], given: Mapping[str, str], where: str) -> Dict[str, str]:
    """Return the map's roots with each ``${name}`` filled from ``given``."""
    declared = top.get("variables", [])
    if "variables" in top:
        _require([(isinstance(declared, list) and bool(declared) and all(isinstance(name, str) for name in declared),
                   "variables is not a non-empty list of names")], where)
    undeclared = sorted(set(given) - set(declared))
    unfilled = [name for name in declared if name not in given]
    _require(
        [
            *((VARIABLE_NAME.fullmatch(name) is not None, f"variable {name!r} is not a name ([a-z][a-z0-9_]*)")
              for name in declared),
            (len(set(declared)) == len(declared), "a variable is declared twice"),
            *((isinstance(value, str) and VARIABLE_VALUE.fullmatch(value) is not None,
               f"variable {name}'s value {value!r} is not a root-name fragment ([A-Za-z0-9][A-Za-z0-9._-]*)")
              for name, value in given.items()),
            (not undeclared, f"values given for {undeclared}, which the map does not declare"),
            (not unfilled, f"no value for {unfilled} (--arm-map-var NAME=VALUE)"),
        ],
        where,
    )
    roots = top["roots"]
    used = {name for uri in roots.values() if isinstance(uri, str) for name in PLACEHOLDER.findall(uri)}
    unknown, unused = sorted(used - set(declared)), sorted(set(declared) - used)
    _require([(not unknown, f"root placeholders {unknown} name no declared variable"),
              (not unused, f"variables {unused} fill no root")], where)
    return {key: PLACEHOLDER.sub(lambda match: given[match.group(1)], uri) if isinstance(uri, str) else uri
            for key, uri in roots.items()}


def parse_variables(words: Sequence[str]) -> Dict[str, str]:
    """Return ``NAME=VALUE`` words (``--arm-map-var``) as ``{name: value}``; a word without ``=`` or a name given twice
    is refused."""
    parsed: Dict[str, str] = {}
    for word in words:
        name, sep, value = word.partition("=")
        _require([(bool(sep), f"{word!r} is not NAME=VALUE"),
                  (name not in parsed, f"variable {name!r} is given twice")], "--arm-map-var")
        parsed[name] = value
    return parsed


def arm_map_variables_record(arm_map: "ArmMap") -> Dict[str, Any]:
    """Return what a readout records of the variables' values: ``{"variables": {...}}``, or nothing for a map that
    declares none."""
    return {"variables": dict(sorted(arm_map.variables.items()))} if arm_map.variables else {}


def add_arm_map_var_argument(parser: argparse.ArgumentParser, example: str) -> None:
    """Add the repeatable ``--arm-map-var NAME=VALUE`` flag; ``example`` ends its help."""
    parser.add_argument("--arm-map-var", action="append", default=[], metavar="NAME=VALUE",
                        help=f"a value for a variable the arm map declares (repeatable: {example})")


def arm_map_var_refusal(arm_map: Optional[Path], words: Sequence[str]) -> Optional[str]:
    """Why ``--arm-map-var`` cannot be taken (no ``--arm-map`` to give it to), or None."""
    return "--arm-map-var gives an arm map's variables; it needs --arm-map" if words and arm_map is None else None


def load_arm_map(path: Path | str, variables: Optional[Mapping[str, str]] = None) -> ArmMap:
    """Read an arm map (YAML) and check its schema; :func:`build_payload` checks it against the roots."""
    given = dict(variables or {})
    path = Path(path)
    where = f"arm map {path.name}"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ArmMapError(f"{where}: cannot read it: {error}") from error
    raw = _load_yaml(text, where)
    top = _mapping(raw, where, TOP_KEYS, ("version", "roots", "judges"))
    roots, judges = top["roots"], top["judges"]
    _require(
        [
            (type(top["version"]) is int and top["version"] == ARM_MAP_VERSION,
             f"version {top['version']!r}, not {ARM_MAP_VERSION}"),
            (isinstance(roots, Mapping) and bool(roots), "roots is not a non-empty mapping"),
            (isinstance(judges, Mapping) and bool(judges), "judges is not a non-empty mapping"),
        ],
        where,
    )
    roots = _resolve_variables(top, given, where)
    _require(
        [
            (isinstance(uri, str) and is_artifact_uri(uri), f"root {key!r}: {uri!r} is not an artifact:// URI")
            for key, uri in roots.items()
        ],
        where,
    )
    _require([(len(set(roots.values())) == len(roots), "two roots name one URI")], where)
    parsed: Dict[str, JudgeArms] = {}
    named: List[Arm] = []
    for model_name, entry in judges.items():
        at = f"{where} judge {model_name}"
        entry = _mapping(entry, at, JUDGE_KEYS, tuple(SLOT_NAMES))
        arms = {slot: _arm(entry[name], f"{at} {name}", roots) for name, slot in SLOT_NAMES.items()}
        cross_stack = entry.get("cross_stack")
        _require([(cross_stack in (None, CROSS_STACK_ARM),
                   f"cross_stack {cross_stack!r}: only the {CROSS_STACK_ARM} arm's contrasts may be declared "
                   "cross-stack")], at)
        parsed[str(model_name)] = JudgeArms(str(model_name), arms, cross_stack == CROSS_STACK_ARM)
        named.extend(arms.values())
    repeated = sorted({f"{arm.root}:{arm.condition_id}" for arm in named if named.count(arm) > 1})
    declared = [name for name, judge in parsed.items() if judge.cross_stack]
    reason = top.get("cross_stack_reason")
    _require(
        [
            (not repeated, f"{repeated} named more than once: an arm is one slot of one judge"),
            (not declared or (isinstance(reason, str) and bool(reason.strip())),
             f"{declared} declare cross_stack, so the map must state cross_stack_reason"),
            (bool(declared) or reason is None, "cross_stack_reason without a judge that declares cross_stack"),
            (not (set(roots) - {arm.root for arm in named}),
             f"root(s) {sorted(set(roots) - {arm.root for arm in named})} hold no arm"),
        ],
        where,
    )
    return ArmMap(path, dict(roots), parsed, reason, tuple(named), given)


def check_arm(arm: Arm, slot: str, judge: JudgeArms, roots: Mapping[str, Root], where: str) -> ConditionSpec:
    """Return the arm's registered condition, refused unless it is the judge's, analysed, and of its slot's style."""
    spec = registered_condition(roots[arm.root], arm.condition_id, where)
    _require(
        [
            (spec.model_name == judge.model_name, f"{arm.condition_id} is {spec.model_name}'s, not {judge.model_name}'s"),
            (spec.prompt_style == SLOT_STYLES[slot],
             f"{arm.condition_id}'s style {spec.prompt_style!r} is not its {STYLE_LABELS[slot]} slot's "
             f"{SLOT_STYLES[slot]!r}"),
        ],
        where,
    )
    return spec


def _check_identities(named: Mapping[str, Any], where: str) -> None:
    rules = {name: spec.label_rule for name, spec in named.items()}
    _require([(len(set(rules.values())) == 1,
               "its arms were judged under different label rules (" +
               ", ".join(f"{name} {rule}" for name, rule in rules.items()) + "): one rule per judge's arms")], where)
    first_name, first = next(iter(named.items()))
    _require(
        [
            (not (differing := rp.differences_beyond_style(first, spec)),
             f"its {name} arm differs from its {first_name} arm in {differing}, not only in the prompt style")
            for name, spec in named.items()
        ],
        where,
    )


def _arm_grid(root: Root, spec: ConditionSpec, n_draws: int, where: str) -> DrawGrid:
    return arm_outcomes(root, spec, n_draws, where)[1]


#: The fields a recomputed within-root contrast must share with the aggregate's row.
AGGREGATE_FIELDS = ("estimate", "ci_low", "ci_high", "sd", "p_value", "n_permutations", "grid_digest_a",
                    "grid_digest_b")


def _aggregate_row(payload: Mapping[str, Any], a: str, b: str) -> Optional[Mapping[str, Any]]:
    return next(
        (row for family in (payload.get("style_families") or {}).values() for row in family["contrasts"]
         if (row["a"], row["b"]) == (a, b)),
        None,
    )


def _contrast(
    grids: Mapping[str, DrawGrid], a: Tuple[Root, str], b: Tuple[Root, str], common: Mapping[str, Any], where: str
) -> Dict[str, Any]:
    """``rate(a) - rate(b)`` by the aggregate's function; equal to the aggregate's row when one root pairs them."""
    (root_a, cid_a), (root_b, cid_b) = a, b
    row = contrast_row({cid_a: grids[cid_a], cid_b: grids[cid_b]}, cid_a, cid_b, shared=False, **common)
    within = root_a.key == root_b.key
    stored = _aggregate_row(root_a.payload, cid_a, cid_b) if within else None
    if stored is not None:
        _require(
            [(stored[key] == row[key], f"{cid_a} vs {cid_b}: the recomputed {key} {row[key]!r} is not the "
                                       f"aggregate's {stored[key]!r}") for key in AGGREGATE_FIELDS],
            where,
        )
    return {**row, "paired_within_root": within}


def _crossed_fields(
    first: Sequence[Mapping[str, Optional[str]]], second: Sequence[Mapping[str, Optional[str]]]
) -> List[str]:
    return sorted({
        name
        for a in first
        for b in second
        for name, value in across_conditions(a).items()
        if value is not None and b.get(name) is not None and b[name] != value
    })


def _cross_stack(
    stacks: Mapping[str, Mapping[str, Any]], a: str, b: str, *, excepted: bool, where: str, names: Tuple[str, str]
) -> Optional[Dict[str, Any]]:
    """Return the cross-stack record of contrast ``a`` vs ``b``, None on one stack; refused unless excepted and
    differing in the stack's packages only."""
    first, second = stacks[a]["signatures"], stacks[b]["signatures"]
    conflict = contrast_conflict(first, second)
    if conflict is None:
        return None
    _require([(excepted, f"{names[0]} vs {names[1]} would compare judgments from different inference stacks or "
                         f"serving settings ({conflict}); only the {CROSS_STACK_ARM} arm's contrasts may cross stacks, "
                         "and only where the arm map declares cross_stack for the judge")], where)
    fields = _crossed_fields(first, second)
    beyond = [name for name in fields if name not in STACK_PACKAGES]
    _require([(bool(fields) and not beyond,
               f"{names[0]} vs {names[1]} differs in {beyond or 'nothing it can name'} beyond the inference stack's "
               f"packages ({conflict}): the declared exception covers {list(STACK_PACKAGES)} only, never a serving "
               "setting")], where)
    return {
        "declared": CROSS_STACK_ARM,
        "baseline_stack": short_stack(first, fields),
        "treatment_stack": short_stack(second, fields),
        "fields": fields,
        "conflict": conflict,
    }


def _table_row(
    label: str, model_name: str, slots: Tuple[str, str], arms: Mapping[str, Tuple[Root, Any]], contrast: Mapping[str, Any],
    cross_stack: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    baseline, treatment = slots
    (root_a, spec_a), (root_b, spec_b) = arms[baseline], arms[treatment]
    return {
        **table_row(label, model_name, baseline, treatment, contrast, test_method=TEST_METHOD),
        "baseline_style": spec_a.prompt_style,
        "treatment_style": spec_b.prompt_style,
        "baseline_condition": spec_a.public_id,
        "treatment_condition": spec_b.public_id,
        "baseline_root": root_a.uri,
        "treatment_root": root_b.uri,
        "paired_within_root": contrast["paired_within_root"],
        "cross_stack": None if cross_stack is None else dict(cross_stack),
    }


def _arm_record(root: Root, spec: ConditionSpec, stack: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "root": root.uri,
        "campaign_id": root.payload["campaign_id"],
        "condition_id": spec.public_id,
        "prompt_style": spec.prompt_style,
        "label_rule": spec.label_rule,
        "model_type": spec.model_type,
        "stack": {key: value for key, value in stack.items() if key != "signatures"},
    }


def judges_by_label(arm_map: ArmMap) -> Dict[str, JudgeArms]:
    """The arm map's judges by golden label (``campaign_label``), refused if two judges share one."""
    labels = {model_name: campaign_label(model_name) for model_name in arm_map.judges}
    _require([(len(set(labels.values())) == len(labels), f"two judges share a golden label: {labels}")],
             f"arm map {arm_map.path.name}")
    return {labels[model_name]: judge for model_name, judge in arm_map.judges.items()}


def build_payload(
    arm_map_path: Path | str, *, alpha: float = ALPHA, allow_incomplete: bool = False, allow_stack_mismatch: bool = False,
    variables: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return the golden the arm map at ``arm_map_path`` describes, its variables given ``variables``."""
    arm_map = load_arm_map(arm_map_path, variables)
    roots = open_roots(arm_map.roots, f"arm map {arm_map.path.name}", allow_incomplete=allow_incomplete,
                       allow_stack_mismatch=allow_stack_mismatch)
    first = next(iter(roots.values())).payload
    n_draws = int(first["n_decoder_draws"])
    common = rp.bootstrap_settings(first)
    by_label = judges_by_label(arm_map)
    rows: List[Dict[str, Any]] = []
    bar_block: Dict[str, Any] = {}
    rates: Dict[str, Any] = {}
    conditions: Dict[str, Any] = {}
    same_vs_separate: Dict[str, Any] = {}
    arm_records: Dict[str, Any] = {}
    label_rules: Dict[str, str] = {}
    retry_scopes: Dict[str, str] = {}
    grids: Dict[str, DrawGrid] = {}
    stacks: Dict[str, Dict[str, Any]] = {}
    # Each arm's spec by (root key, the id it is registered and analysed under): what the golden reads of a root.
    used: Dict[Tuple[str, str], ConditionSpec] = {}
    for label in display_order(by_label):
        judge = by_label[label]
        where = f"{label} ({judge.model_name})"
        arms = {slot: (roots[arm.root], check_arm(arm, slot, judge, roots, where))
                for slot, arm in judge.arms.items()}
        _check_identities({STYLE_LABELS[slot]: arms[slot][1] for slot in SLOTS}, where)
        check_campaign_settings([roots[key] for key in dict.fromkeys(arm.root for arm in judge.arms.values())], where)
        for slot, (root, spec) in arms.items():
            used[(root.key, spec.condition_id)] = spec
            stacks[spec.condition_id] = arm_stack(root, spec, where)
            grids[spec.condition_id] = _arm_grid(root, spec, n_draws, where)
        crossed = []
        for baseline, treatment in TABLE_CONTRASTS:
            (root_a, spec_a), (root_b, spec_b) = arms[baseline], arms[treatment]
            cross = _cross_stack(stacks, spec_a.condition_id, spec_b.condition_id,
                                 excepted=judge.cross_stack and baseline == NAIVE_SLOT, where=where,
                                 names=(STYLE_LABELS[baseline], STYLE_LABELS[treatment]))
            crossed.append(cross)
            contrast = _contrast(grids, (root_a, spec_a.condition_id), (root_b, spec_b.condition_id), common, where)
            rows.append(_table_row(label, judge.model_name, (baseline, treatment), arms, contrast, cross))
        _require([(not judge.cross_stack or any(crossed),
                   f"the arm map declares cross_stack {CROSS_STACK_ARM}, but its {CROSS_STACK_ARM} arm shares its "
                   "other arms' stack (or the judge is a provider's): remove the stale declaration")], where)
        forced_root, forced = arms[FORCED]
        forced_entry = (forced_root.payload.get("forced_conditions") or {}).get(forced.condition_id)
        _require([(forced_entry is not None,
                   f"{forced.condition_id} has no same-call rate in root {forced_root.key}'s estimates (forced_conditions)")],
                 where)
        summaries = {slot: root.payload["conditions"][spec.condition_id] for slot, (root, spec) in arms.items()}
        bar_block[label] = {BAR_POPULATION: {**{slot: rate_bar(summaries[slot]) for slot in SLOTS},
                                               FORCED_B: rate_bar(forced_entry[SAME_CALL_RATE_KEY])}}
        rates[label] = {"rates": {slot: float(summaries[slot]["estimate"]) for slot in sorted(SLOTS)},
                        "n_common": int(summaries[NAIVE_SLOT]["n_scenarios"])}
        for slot, (root, spec) in arms.items():
            summary = summaries[slot]
            conditions[spec.public_id] = {
                "label": label, "prompt_style": spec.prompt_style, "slot": slot, "root": root.uri,
                "campaign_id": root.payload["campaign_id"],
                **{key: summary[key] for key in ("variance_components", "attempt_statistics", "condition_fingerprint",
                                                 "grid_digest")},
            }
        same_vs_separate[label] = {key: forced_entry[SEPARATE_VS_SAME_CALL_KEY][key] for key in SAME_VS_SEPARATE_KEYS}
        arm_records[label] = {slot: _arm_record(root, spec, stacks[spec.condition_id])
                              for slot, (root, spec) in arms.items()}
        label_rules[label] = arms[NAIVE_SLOT][1].label_rule
        retry_scopes[label] = arms[NAIVE_SLOT][1].retry_scope   # one per judge too: _check_identities holds the identity
    ignored = {root.uri: sorted(public_id_of(root, cid) for cid in root.payload["conditions"] if (key, cid) not in used)
               for key, root in roots.items()}
    ignored = {uri: cids for uri, cids in ignored.items() if cids}
    for uri, cids in ignored.items():
        logger.info("%s: conditions the arm map does not name, so no row: %s", uri, cids)
    bases = {summary["interval_basis"] for root in roots.values() for cid, summary in root.payload["conditions"].items()
             if (root.key, cid) in used}
    _require([(len(bases) == 1, f"the arms carry {len(bases)} interval bases {sorted(bases)}; expected one")],
             "the figure")
    table = add_table_multiplicity(rows, alpha=alpha)
    flagged = [row for row in table if row["cross_stack"] is not None]
    return {
        "schema_version": TABLE_SCHEMA_VERSION,
        "table": table,
        "rates": rates,
        "bars": bar_block,
        "diagnostics": {
            "conditions": conditions,
            "separate_vs_same_call": same_vs_separate,
            "excluded_conditions": {public_id_of(root, cid): entry for root in roots.values()
                                    for cid, entry in (root.payload.get("excluded_conditions") or {}).items()},
        },
        "provenance": {
            "schema_version": TABLE_SCHEMA_VERSION,
            "source": TABLE_SOURCE,
            # A strict-gate final campaign each: a scratch golden is never re-rendered into the paper's files.
            "complete": all(is_complete(root.payload) and root.manifest.get("campaign_mode") == CAMPAIGN_MODE_FINAL
                            for root in roots.values()),
            "arm_map": {**file_record(arm_map.path), "version": ARM_MAP_VERSION, **arm_map_variables_record(arm_map)},
            "campaigns": [
                {**estimates_source(root.payload), "root": root.uri,
                 "conditions": sorted(spec.public_id for (key, _), spec in used.items() if key == root.key)}
                for root in roots.values()
            ],
            "arms": arm_records,
            "label_rules": label_rules,
            # Each judge's retry scope, held equal within a judge and free across judges; named only where a judge's is
            # not the bundle default.
            **({"retry_scopes": retry_scopes}
               if any(scope != RETRY_SCOPE_SCENARIO_BUNDLE for scope in retry_scopes.values()) else {}),
            "label_replay": {root.uri: rp.replay_provenance(root.path) for root in roots.values()},
            "cross_stack": {
                "declared": sorted(label for label, judge in by_label.items() if judge.cross_stack),
                "contrasts": [[row["label"], row["baseline"], row["treatment"]] for row in flagged],
            },
            "ignored_conditions": ignored,
            "models": display_order(by_label),
            "model_labels": {judge.model_name: label for label, judge in by_label.items()},
            **{key: first.get(key) for key in SHARED_DESIGN_FIELDS},
            "alpha": alpha,
            "metric": TABLE_METRIC,
            "pairing": "scenario_id over one frozen manifest shared by every root; a judge's arms joined across roots "
                       "by the arm map",
            "multiplicity": (
                f"{BH_METHOD} in p_bh and {HOLM_METHOD} in p_holm, each recomputed over every contrast in this "
                "table; the per-root p_holm in the estimates files is a per-campaign family and is not reused; "
                f"{HOLM_MC_MARGIN_NOTE}"
            ),
            "bar_settings": {
                "population": BAR_POPULATION,
                "interval_basis": bases.pop(),
                "inference_method": first.get("inference_method", INFERENCE_METHOD),
                "bootstrap": first.get("bootstrap"),
                "n_scenarios": first.get("n_scenarios"),
                "n_decoder_draws": first.get("n_decoder_draws"),
                "bars": list(PROMPT_BARS),
            },
            "definition": DEFINITION,
        },
    }


def validate_cross_stack(table: Sequence[Mapping[str, Any]], provenance: Optional[Mapping[str, Any]] = None) -> None:
    """Refuse a table whose ``cross_stack`` flags the arm-map build would not have written, or that its ``provenance``
    contradicts."""
    for row in table:
        record = row.get("cross_stack")
        if record is None:
            continue
        where = f"{row.get('label')} {row.get('baseline')}->{row.get('treatment')}"
        _require(
            [
                (row.get("baseline") == NAIVE_SLOT, f"cross_stack on a row whose baseline is not the {CROSS_STACK_ARM} slot"),
                (isinstance(record, Mapping) and record.get("declared") == CROSS_STACK_ARM
                 and all(isinstance(record.get(key), str) for key in ("baseline_stack", "treatment_stack", "conflict"))
                 and isinstance(record.get("fields"), list) and bool(record["fields"])
                 and all(name in STACK_PACKAGES for name in record["fields"]),
                 "its cross_stack record is not a declared exception naming both stacks, the stack packages they "
                 "differ in, and the conflict"),
            ],
            where,
        )
    if provenance is None:
        return
    flagged = [row for row in table if row.get("cross_stack") is not None]
    declared = provenance.get("cross_stack")
    contrasts = declared.get("contrasts") if isinstance(declared, Mapping) else []
    judges = declared.get("declared") if isinstance(declared, Mapping) else []
    _require(
        [
            (contrasts == [[row["label"], row["baseline"], row["treatment"]] for row in flagged],
             f"the rows flagged cross_stack ({[row['label'] + ' ' + row['treatment'] for row in flagged]}) are not the "
             "ones provenance.cross_stack.contrasts lists"),
            (isinstance(judges, list) and all(row["label"] in judges for row in flagged),
             "a flagged row's judge is not one provenance.cross_stack.declared lists"),
        ],
        "the golden",
    )


def paper_table_refusal(payload: Mapping[str, Any]) -> Optional[str]:
    """Return why a golden may not be written into the paper's files, or None: every arm must be of its slot's template
    style."""
    off = sorted({
        f"{row['label']} {STYLE_LABELS[slot]} {style}"
        for row in payload["table"]
        for slot, style in ((row["baseline"], row.get("baseline_style", SLOT_STYLES[row["baseline"]])),
                            (row["treatment"], row.get("treatment_style", SLOT_STYLES[row["treatment"]])))
        if style != SLOT_STYLES[slot]
    })
    if off:
        return (f"arms {off} are not their slots' template styles {list(SLOT_STYLES.values())}: the paper's prompt "
                "templates must be the prompts its table contrasts")
    return None
