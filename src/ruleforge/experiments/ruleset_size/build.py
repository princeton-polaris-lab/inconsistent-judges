from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ruleforge.core.errors import RuleForgeError
from ruleforge.experiments.ruleset_size.curves import CONTRADICTION, INCONSISTENCY_RATE
from ruleforge.experiments.ruleset_size.figure import draw, legend_order
from ruleforge.core.paths import resolve, to_uri
from ruleforge.evaluate.estimates import CampaignEstimatesError, public_amendments
from ruleforge.evaluate.identity import (
    MANIFEST_AMENDMENTS_KEY,
    RETRY_SCOPE_SCENARIO_BUNDLE,
    retry_scope_record,
)
from ruleforge.evaluate.bank import (
    BANK_ROOT_KEY,
    DESIGN_SHARED_BANK,
    ONE_RULESET_SIZES_KEY,
    SHARED_BANK_ESTIMATES_FNAME,
    group_by_model,
    load_design,
    one_ruleset_sizes,
)
from ruleforge.evaluate.campaign import load_conditions
from ruleforge.evaluate.identity import SHARED_BANK_DESIGN_FNAME
from ruleforge.evaluate.stack import (
    STACK_MISMATCHES_KEY,
    recorded_mismatch_refusal,
    recorded_mismatches,
)
from ruleforge.experiments import driver
from ruleforge.experiments.labels import MODEL_DISPLAY_LABELS, campaign_label
from ruleforge.rulesets.subsets import WITH_REPLACEMENT_SHUFFLED_FULL

#: The curve schema: every point rests on the complete scenario x subset-seed x draw grid, and ``n_obs`` counts
#: judgments.
CURVE_VERSION = 5
DERIVED_FROM = f"stochastic campaign {DESIGN_SHARED_BANK}"

#: Top-level estimates fields every joined root must agree on.
SHARED_DESIGN_FIELDS = (
    "design",
    "inference_method",
    "interval_basis",
    "n_scenarios",
    "n_decoder_draws",
    "n_subset_seeds",
    "subset_sizes",
    "temperature",
    "truncation",
    "label_rule",
    "scenario_manifest_sha256",
    "bootstrap",
    "contrasts",
)

#: Curve fields that must agree across one figure (checked at the build and again when a golden is read).
CURVE_SHARED_FIELDS = (
    "ci_estimator",
    "inference_method",
    "ci_level",
    "n_resamples",
    "rng_seed",
    "n_expected",
    "n_scenarios",
    "n_subset_seeds",
    "n_decoder_draws",
    "strategy",
    "derived_from",
)
CAMPAIGN_SHARED_FIELDS = (
    "design",
    "scenario_manifest_sha256",
    "temperature",
    "truncation",
    "label_rule",
    "max_semantic_attempts",
)

CONTRAST_NAMES = {
    "net_rise": "K{max} - K{min}",
    "upper_vs_middle": "K56 - mean(K20..K30)",
    "last_step": "K{max} - K{max_minus_1}",
}


class CampaignCurveError(ValueError, RuleForgeError):
    """The estimates cannot make one ruleset-size figure."""


def load_estimates(
    paths: Sequence[Path], *, allow_incomplete: bool, allow_stack_mismatch: bool = False
) -> List[Dict[str, Any]]:
    """Read every estimates file, refusing a set that is not one experiment."""
    if not paths:
        raise CampaignCurveError("at least one --estimates file is required")
    payloads: List[Dict[str, Any]] = []
    for path in paths:
        data = Path(path).read_bytes()
        payload = json.loads(data.decode("utf-8"))
        payload["_path"], payload["_sha256"] = Path(path), hashlib.sha256(data).hexdigest()
        design = payload.get("design", {}).get("design")
        if design != DESIGN_SHARED_BANK:
            raise CampaignCurveError(
                f"{path}: design {design!r} is not {DESIGN_SHARED_BANK!r}; this driver reads "
                "the shared-bank campaign"
            )
        if payload.get("state") != "complete" and not allow_incomplete:
            raise CampaignCurveError(
                f"{path}: campaign state {payload.get('state')!r}; the paper's figure needs a "
                "complete campaign (pass --allow-incomplete for a scratch build)"
            )
        refusal = recorded_mismatch_refusal(payload, str(path))
        if refusal is not None and not allow_stack_mismatch:
            raise CampaignCurveError(refusal)
        payloads.append(payload)
    first = payloads[0]
    for payload in payloads[1:]:
        for field in SHARED_DESIGN_FIELDS:
            if payload.get(field) != first.get(field):
                raise CampaignCurveError(
                    f"{payload['_path']}: {field} {payload.get(field)!r} differs from "
                    f"{first['_path']}: {first.get(field)!r}; these roots are not one experiment"
                )
    seen: Dict[str, Path] = {}
    for payload in payloads:
        for model in payload["models"]:
            if model in seen:
                raise CampaignCurveError(f"model {model} appears in both {seen[model]} and {payload['_path']}")
            seen[model] = payload["_path"]
    return payloads


def judge_label(model_name: str) -> str:
    """Return a campaign judge's legend label, its golden label's display name; a judge with none is refused."""
    try:
        return MODEL_DISPLAY_LABELS[campaign_label(model_name)]
    except CampaignEstimatesError as error:
        raise CampaignCurveError(str(error)) from None


def _contrast_label(name: str, sizes: Sequence[int]) -> str:
    sizes = sorted(sizes)
    template = CONTRAST_NAMES.get(name, name)
    return template.format(max=sizes[-1], min=sizes[0], max_minus_1=sizes[-2])


def estimates_root(payload: Mapping[str, Any]) -> Path:
    """Return the root an analysis read: the one joined estimates name, else the root their file lies in."""
    return resolve(payload["root"]) if BANK_ROOT_KEY in payload else Path(payload["_path"]).parent.parent


def shared_ruleset_sizes(payload: Mapping[str, Any]) -> Dict[str, List[int]]:
    """Return each model's sizes at which every subset seed judged one ruleset: as joined estimates record them, else
    from the conditions of the root their file lies in."""
    recorded = {model: entry.get(ONE_RULESET_SIZES_KEY) for model, entry in payload["models"].items()}
    if all(sizes is not None for sizes in recorded.values()):
        return recorded
    root = estimates_root(payload)
    if not (root / SHARED_BANK_DESIGN_FNAME).is_file():
        # Away from their root, a design that judges the full set in file order is taken to share one ruleset at its
        # largest size alone.
        if payload["design"]["sampling_strategy"] == WITH_REPLACEMENT_SHUFFLED_FULL:
            raise CampaignCurveError(f"{payload['_path']} lies away from its root, so the sizes whose subset seeds "
                                     "judged one ruleset cannot be read; build from the root's aggregate/ instead")
        return {model: [max(int(size) for size in payload["subset_sizes"])] for model in payload["models"]}
    design = load_design(root)
    by_model = group_by_model(load_conditions(root), design)
    return {model: one_ruleset_sizes(by_model[model], design) for model in payload["models"]}


def joined_roots(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return, for estimates that joined a bank root, the root of the curve's bank and, by size, the campaign and root
    each other size came from with its retry scope and cap in force; nothing for estimates of one root."""
    bank = payload.get(BANK_ROOT_KEY)
    if bank is None:
        return {}
    root = {"campaign_id": payload["campaign_id"], "root": payload["root"],
            "retry_scope": payload.get("retry_scope", RETRY_SCOPE_SCENARIO_BUNDLE),
            "max_semantic_attempts": payload["max_semantic_attempts"]}
    return {"root": bank["root"], "sizes_from": {str(size): dict(root) for size in payload["subset_sizes"]
                                                 if size not in bank["subset_sizes"]}}


def build_curve(payload: Mapping[str, Any], model_name: str, one_ruleset: Sequence[int]) -> Dict[str, Any]:
    """Return one model's curve from its estimates entry; ``one_ruleset`` holds the sizes at which its subset seeds
    judged one ruleset."""
    entry = payload["models"][model_name]
    if not entry.get("judge_type"):
        raise CampaignCurveError(f"{model_name}: the estimates record no judge_type; the curve's strategy is not a guess")
    sizes = [int(size) for size in payload["subset_sizes"]]
    n_scenarios = int(payload["n_scenarios"])
    n_seeds = int(payload["n_subset_seeds"])
    n_draws = int(payload["n_decoder_draws"])
    n_obs = n_scenarios * n_seeds * n_draws
    judge = entry.get("model_name", model_name)
    # The campaign that judged the bank: the root's own, or the bank root's for estimates that joined one.
    bank = payload.get(BANK_ROOT_KEY, payload)
    points = []
    for point in entry["points"]:
        size = int(point["size"])
        points.append(
            {
                "size": size,
                "mean": float(point["estimate"]),
                "ci_low": float(point["ci_low"]),
                "ci_high": float(point["ci_high"]),
                "sd": float(point["sd"]),
                "two_stage": dict(point["two_stage"]),
                "n_obs": n_obs,
                "n_scenarios": n_scenarios,
                "n_seeds": n_seeds,
                "n_draws": n_draws,
                "coverage": 1.0,
                "drops": {},
                # Where every seed judged one ruleset, the seeds are replicate draws of it.
                "shares_one_ruleset": size in one_ruleset,
                "zero_seed_variance": False,
            }
        )
    if [point["size"] for point in points] != sizes:
        raise CampaignCurveError(f"{model_name}: points cover {[p['size'] for p in points]}, the design {sizes}")
    contrasts = {
        name: {
            "contrast": _contrast_label(name, sizes),
            **row,
            "excludes_zero": bool(row["ci_low"] > 0.0 or row["ci_high"] < 0.0),
        }
        for name, row in entry["contrasts"].items()
    }
    return {
        "label": judge_label(judge),
        "metric": CONTRADICTION,
        "quantity": INCONSISTENCY_RATE,
        "code_version": CURVE_VERSION,
        "strategy": str(entry["judge_type"]),
        "derived_from": DERIVED_FROM,
        "ci_estimator": payload["interval_basis"],
        "inference_method": payload["inference_method"],
        "ci_level": float(payload["bootstrap"]["ci_level"]),
        "n_resamples": int(payload["bootstrap"]["n_resamples"]),
        "rng_seed": int(payload["bootstrap"]["seed"]),
        "n_expected": n_obs,
        "n_scenarios": n_scenarios,
        "n_subset_seeds": n_seeds,
        "n_decoder_draws": n_draws,
        "full_set_key": None,
        "points": points,
        "contrasts": contrasts,
        "adjacent_rises": int(entry["adjacent_rises"]),
        "bank": dict(entry["bank"]),
        "subsets": dict(entry["subsets"]),
        "campaign": {
            "campaign_id": bank["campaign_id"],
            "model_name": judge,
            "condition_prefix": model_name,
            "estimates": to_uri(payload["_path"]),
            "estimates_sha256": payload["_sha256"],
            "state": payload.get("state"),
            "design": dict(payload["design"]),
            "scenario_manifest_sha256": payload["scenario_manifest_sha256"],
            "temperature": payload["temperature"],
            "truncation": payload.get("truncation"),
            "label_rule": payload.get("label_rule"),
            "max_semantic_attempts": frozen_cap({**bank, "_path": payload["_path"]}),
            **retry_scope_record(payload),
            **public_amendments(bank),
            "created_at": payload.get("created_at"),
            **_model_mismatches(payload, model_name),
            **joined_roots(payload),
        },
    }


def frozen_cap(payload: Mapping[str, Any]) -> Any:
    """Return the retry cap a root's campaign froze: its ``max_semantic_attempts``, or the old cap of the one
    amendment that raised it."""
    amendments = payload.get(MANIFEST_AMENDMENTS_KEY) or []
    if not amendments:
        return payload.get("max_semantic_attempts")
    if len(amendments) != 1 or not isinstance(amendments[0], Mapping) or (
            amendments[0].get("new_cap") != payload.get("max_semantic_attempts")):
        raise CampaignCurveError(
            f"{payload.get('_path')}: its estimates record max_semantic_attempts {payload.get('max_semantic_attempts')!r}"
            f" beside {MANIFEST_AMENDMENTS_KEY} {amendments!r}: one amendment raising the cap to it is what an amended "
            "root's analysis records (ruleforge.evaluate.bank)"
        )
    return amendments[0]["old_cap"]


def _model_mismatches(payload: Mapping[str, Any], model_name: str) -> Dict[str, Any]:
    mine = [
        record for record in recorded_mismatches(payload)
        if any(str(cid).startswith(model_name + "/") for cid in record["conditions"])
    ]
    return {STACK_MISMATCHES_KEY: mine} if mine else {}


def build_curves(
    paths: Sequence[Path], *, allow_incomplete: bool = False, allow_stack_mismatch: bool = False
) -> List[Dict[str, Any]]:
    """Return every model's curve, in legend order."""
    payloads = load_estimates(paths, allow_incomplete=allow_incomplete, allow_stack_mismatch=allow_stack_mismatch)
    curves = [build_curve(payload, model, sizes)
              for payload in payloads for model, sizes in shared_ruleset_sizes(payload).items()]
    if any("retry_scope" in curve["campaign"] for curve in curves):
        # Estimates that name no retry scope ran under the bundle default; once one curve names a scope, all do.
        for curve in curves:
            curve["campaign"].setdefault("retry_scope", RETRY_SCOPE_SCENARIO_BUNDLE)
    return hold_joined(legend_order(curves, lambda curve: curve["label"]))


def hold_joined(curves: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    """Return the curves, refusing them unless their shared fields agree (they are one experiment)."""
    for field in CURVE_SHARED_FIELDS:
        if len({json.dumps(curve.get(field), sort_keys=True) for curve in curves}) != 1:
            raise CampaignCurveError(f"curves disagree on {field}")
    for field in CAMPAIGN_SHARED_FIELDS:
        if len({json.dumps(curve.get("campaign", {}).get(field), sort_keys=True) for curve in curves}) != 1:
            raise CampaignCurveError(f"curves disagree on campaign.{field}; they are not one experiment")
    return curves


def validate_curves(curves: Sequence[Mapping[str, Any]], *, allow_stack_mismatch: bool = False) -> List[Dict[str, Any]]:
    """Return the golden's curves, refusing one this module did not write or one that mixes experiments."""
    if not curves:
        raise CampaignCurveError("the golden holds no curves")
    for curve in curves:
        if curve.get("code_version") != CURVE_VERSION:
            raise CampaignCurveError(
                f"curve {curve.get('label')!r} has code_version {curve.get('code_version')!r}, not "
                f"{CURVE_VERSION}; this driver renders only the campaign golden"
            )
        if curve.get("quantity") != INCONSISTENCY_RATE or curve.get("metric") != CONTRADICTION:
            raise CampaignCurveError(f"curve {curve.get('label')!r} is not the {INCONSISTENCY_RATE}")
        if curve.get("campaign", {}).get("state") != "complete":
            raise CampaignCurveError(
                f"curve {curve.get('label')!r} was built from an incomplete campaign; a scratch "
                "golden is not re-rendered into the paper's figure"
            )
        refusal = recorded_mismatch_refusal(curve.get("campaign", {}), f"curve {curve.get('label')!r}")
        if refusal is not None and not allow_stack_mismatch:
            raise CampaignCurveError(refusal)
    hold_joined(curves)
    for curve in curves:
        if [point["size"] for point in curve["points"]] != [point["size"] for point in curves[0]["points"]]:
            raise CampaignCurveError(f"curve {curve['label']!r} covers different sizes from {curves[0]['label']!r}")
    labels = [curve["label"] for curve in curves]
    if len(set(labels)) != len(labels):
        raise CampaignCurveError(f"duplicate curve labels {labels}")
    return [dict(curve) for curve in curves]


def _summary(curves: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for curve in curves:
        first, last = curve["points"][0], curve["points"][-1]
        rise = curve["contrasts"].get("net_rise", {})
        lines.append(
            f"  {curve['label']:<13} K={first['size']} {first['mean']:.3f} [{first['ci_low']:.3f}, {first['ci_high']:.3f}]"
            f" -> K={last['size']} {last['mean']:.3f} [{last['ci_low']:.3f}, {last['ci_high']:.3f}]"
            f"  net rise {rise.get('estimate', float('nan')):+.3f}"
            f" [{rise.get('ci_low', float('nan')):+.3f}, {rise.get('ci_high', float('nan')):+.3f}]"
            f"  {curve['adjacent_rises']} of {len(curve['points']) - 1} steps rise"
        )
    lines.append(
        f"  {len(curves)} curve(s); N={curves[0]['n_scenarios']} x J={curves[0]['n_subset_seeds']} "
        f"x D={curves[0]['n_decoder_draws']}; {curves[0]['ci_estimator']}"
    )
    return "\n".join(lines)


def read_golden(path: Path) -> List[Dict[str, Any]]:
    """Return the curves of the golden at ``path``, refusing a file that is not a list of curves this driver wrote."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise CampaignCurveError(f"{path} is not a list of curves; not a golden this driver wrote")
    return payload


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the ruleset-size golden from campaign estimates, or redraw the figure from a golden.")
    parser.add_argument("--estimates", type=Path, action="append", default=[],
                        help=f"a campaign root's aggregate/{SHARED_BANK_ESTIMATES_FNAME} (repeat per root)")
    driver.add_from_golden(parser)
    driver.add_golden(parser)
    driver.add_figure(parser, help="the figure (.png or .pdf)", repeatable=False)
    driver.add_dpi(parser)
    driver.add_allow_incomplete(parser, "build from an incomplete campaign (scratch previews only)")
    driver.add_allow_stack_mismatch(parser, "build from (or re-render) estimates that report inference-stack "
                                    "mismatches; the curves name them")
    args = driver.parse(parser, argv)
    driver.one_source(parser, args)
    driver.require_drawing([args.figure])
    curves = driver.golden_or_build(
        parser, args,
        lambda path: validate_curves(read_golden(path), allow_stack_mismatch=args.allow_stack_mismatch),
        lambda: build_curves(args.estimates, allow_incomplete=args.allow_incomplete,
                             allow_stack_mismatch=args.allow_stack_mismatch),
        (CampaignCurveError,))
    # The golden first, so a figure that fails to write sits beside this run's golden, not an older one.
    driver.write_golden(args.golden, curves, sort_keys=False)
    driver.draw_each([args.figure], lambda out: draw(curves, out, dpi=args.dpi))
    print(_summary(curves))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
