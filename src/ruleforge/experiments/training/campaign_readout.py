from __future__ import annotations

import hashlib
import json
import math
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ruleforge.core.errors import RuleForgeError
from ruleforge.evaluate.inference import (
    INFERENCE_METHOD,
    TEST_METHOD,
    holm_adjust,
    load_condition_outcomes,
    outcome_grid,
)
from ruleforge.core.paths import to_uri
from ruleforge.core.record_names import SAME_CALL_KEY, SAME_CALL_RATE_KEY
from ruleforge.experiments.checks import require
from ruleforge.hub import published_record
from ruleforge.evaluate.aggregate import POLICY_ORIENTATION, PolicyFamily, policy_families
from ruleforge.evaluate.campaign import load_conditions
from ruleforge.evaluate.identity import (
    CAMPAIGN_MODE_FINAL,
    DRAW_PROVENANCE_FNAME,
    ConditionSpec,
    draw_dirname,
    load_sampling_manifest,
)
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.evaluate.discrimination import (
    KAPPA_ORIENTATION,
    KAPPA_TEST_METHOD,
    VerdictGrids,
    YES_PHI,
    kappa_features,
    kappa_parts,
    read_strata,
)
from ruleforge.evaluate.stack import STACK_MISMATCHES_KEY, recorded_mismatch_refusal, recorded_mismatches

GOLDEN_SCHEMA_VERSION = 1
GOLDEN_SOURCE = "stochastic_campaign"
#: The binary per-scenario inconsistency: the full-ruleset judgment against the strongest single-rule judgment.
METRIC = "flat"
CLAIM_RULE = (
    "Two Holm families over the reference-against-each pairs, flat inconsistency and the kappa "
    "difference; 'more consistent' only when both are significant in the policy's favour (an "
    "intersection-union test, so no correction across the two)"
)
#: The engine settings the disclosure names, as each draw's provenance records them.
PRECISION_KEYS = ("VLLM_FLOAT32_MATMUL_PRECISION", "TRITON_F32_DEFAULT", "VLLM_BATCH_INVARIANT")
#: How Triton computes an fp32 attention dot product under each ``TRITON_F32_DEFAULT``.
ATTENTION_PRECISION = {
    "unset": "Triton's default precision for fp32 dot products (TF32; TRITON_F32_DEFAULT unset)",
    "ieee": "IEEE fp32 (TRITON_F32_DEFAULT=ieee)",
    "tf32x3": "three-pass TF32 (TRITON_F32_DEFAULT=tf32x3)",
    "tf32": "TF32 (TRITON_F32_DEFAULT=tf32)",
}
REGIMES = {"float32": "merged_fp32", "bfloat16": "bf16_fallback"}
#: LoRA serving: each policy's adapter unmerged on the bf16 engine, the base arm without one.
LORA_REGIME = "lora_bf16"
LORA_DTYPE = "bfloat16"
#: Parity rho: a gate under fp32 serving (where the interval's half-width is at most the precision), a reporting
#: threshold under the bf16 fallback, a two-sided window under LoRA serving.
RHO_GATE_FP32 = 0.99
RHO_PRECISION_FP32 = 0.01
RHO_REPORT_BF16 = 0.90
RHO_WINDOW_LORA = (0.95, 1.05)
RHO_PRECISION_LORA = 0.01
#: The parity compare mode each regime's reports must have been compared in.
PARITY_MODES = {**{regime: "merged" for regime in REGIMES.values()}, LORA_REGIME: "lora"}
#: Parity's two arms by the report's ``segment``: prefill scores every position, decode the completion positions.
PARITY_ARMS = {"all": "prefill", "completion": "decode"}


class TrainingReadoutError(ValueError, RuleForgeError):
    """The campaign, its estimates or a parity report cannot make the training figure."""


_require = partial(require, error=TrainingReadoutError)


def policy_pairs(items: Optional[Sequence[str]], flag: str, *, repeat: bool = False) -> dict[str, list[str]]:
    """Return a flag's ``POLICY=VALUE`` entries by policy; a policy repeats only where ``repeat``."""
    pairs: dict[str, list[str]] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key or not value or (key in pairs and not repeat) or value in pairs.get(key, []):
            raise TrainingReadoutError(f"{flag} takes {'distinct' if repeat else 'unique'} POLICY=VALUE entries, got {item!r}")
        pairs.setdefault(key, []).append(value)
    return pairs


def _close(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(a, b, atol=1e-12, rtol=0))


def _the_family(estimates: Mapping[str, Any], registered: Mapping[str, ConditionSpec], where: str):
    families = policy_families(registered)
    blocks = estimates.get("policy_families") or {}
    _require(
        [
            (len(families) == 1 and len(blocks) == 1, f"{len(blocks)} policy families; a training readout is one"),
            (set(families) == set(blocks), "the estimates' policy family is not the registered one"),
        ],
        where,
    )
    (name, family), block = next(iter(families.items())), next(iter(blocks.values()))
    _require(
        [
            (not family.missing and "missing_policies" not in block, f"declared policies {family.missing} are missing"),
            ("excluded_policies" not in block, f"policies {block.get('excluded_policies')} were excluded"),
            (set(family.members) == set(estimates["conditions"]), "the root holds conditions outside the family"),
            (block.get("orientation") == POLICY_ORIENTATION, "the flat family's orientation is not the policy one"),
            (block.get("kappa_orientation") == KAPPA_ORIENTATION, "the kappa family's orientation is not the policy one"),
        ],
        where,
    )
    return name, family, block


def _check_family_rows(rows: list, pairs: list, *, key: str, test_method: str, where: str) -> None:
    _require(
        [
            ([(row["a"], row["b"]) for row in rows] == pairs, f"{key} are not the reference-against-each pairs"),
            (all(row.get("family_size") == len(pairs) for row in rows), f"{key} family_size is not {len(pairs)}"),
            (all(row.get("test_method") == test_method for row in rows), f"{key} carry another test method"),
            (all(row.get("inference_method") == INFERENCE_METHOD for row in rows), f"{key} carry another inference method"),
        ],
        where,
    )
    for row, fresh in zip(rows, holm_adjust(rows), strict=True):
        _require(
            [
                (0 <= float(row["p_value"]) <= 1, f"{key} p-value {row['p_value']} is not a probability"),
                (_close(row["p_holm"], fresh["p_holm"]), f"{key} Holm p disagrees with a recomputation over the family"),
                (
                    row.get("significant_holm") == fresh["significant_holm"],
                    f"{key} significance disagrees with a recomputation of Holm over the family",
                ),
            ],
            where,
        )


def _engine_disclosure(condition_dirs: Sequence[Path], n_draws: int, where: str) -> dict[str, Any]:
    seen: dict[str, set] = {"vllm_version": set(), "attention_backend": set(), **{k: set() for k in PRECISION_KEYS}}
    for condition_dir in condition_dirs:
        for draw in range(n_draws):
            record = json.loads((condition_dir / draw_dirname(draw) / DRAW_PROVENANCE_FNAME).read_text(encoding="utf-8"))
            runtime = record.get("runtime") or {}
            precision = runtime.get("precision_env") or {}
            seen["vllm_version"].add(runtime.get("vllm_version"))
            seen["attention_backend"].add(record.get("attention_backend"))
            for key in PRECISION_KEYS:
                seen[key].add(precision.get(key))
    differing = {key: sorted(map(str, values)) for key, values in seen.items() if len(values) != 1}
    _require([(not differing, f"the arms' draws ran on different engine settings {differing}")], where)
    return {key: next(iter(values)) for key, values in seen.items()}


def _number(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else repr(value)


#: Judges without parity: their thinking does not close with a literal ``</think>``, which the greedy judgment gate
#: reads, so their campaigns validate with strict_think off.
PARITY_B_NOT_BUILT = frozenset({"gemma-4-31B-it"})


def parity_b_built(model_name: str, strict_think: bool, where: str) -> bool:
    """Return whether parity exists for the judge, refusing a root whose ``strict_think`` does not match its judge."""
    built = model_name not in PARITY_B_NOT_BUILT
    wrong = (
        f"{model_name}'s campaign validated with strict_think off, which only a judge parity B was never built for "
        "runs (PARITY_B_NOT_BUILT: its thinking closes otherwise than with </think>)"
    ) if built else (
        f"{model_name}'s campaign validated with strict_think on, but its thinking closes otherwise than with </think> "
        "(PARITY_B_NOT_BUILT)"
    )
    _require([(strict_think is built, wrong)], where)
    return built


def read_parity(path: Path | str, *, regime: str, policy: str) -> dict[str, Any]:
    """Return one policy's parity record for one arm (rho, interval, verdict, the report by digest); the report must
    be a passing compare of the regime's mode and, outside the bf16 fallback, a gated rho."""
    path = Path(path)
    data = path.read_bytes()
    report = json.loads(data.decode("utf-8"))
    overall = (report.get("buckets") or {}).get("all") or {}
    rho = overall.get("rho")
    rho_ci = overall.get("rho_ci")
    segment = report.get("segment")
    finite = isinstance(rho, (int, float)) and math.isfinite(rho)
    where = f"parity report {path.name} for {policy!r}"
    mode = PARITY_MODES[regime]
    _require(
        [
            (report.get("mode") == mode, f"mode {report.get('mode')!r}, not {mode!r}"),
            (report.get("passed") is True, f"it did not pass ({report.get('failures')})"),
            (finite, f"rho {rho!r} is not a number"),
            (segment in PARITY_ARMS, f"segment {segment!r} is neither parity arm {sorted(PARITY_ARMS)}"),
        ],
        where,
    )
    gated = report.get("rho_gated") or {}
    if regime != REGIMES["bfloat16"]:
        _check_rho_gate(report, overall, rho, gated, regime, where)
    return {
        "report": to_uri(path),
        "report_sha256": hashlib.sha256(published_record(path.name, data)).hexdigest(),  # as the Hub publishes it
        "mode": report["mode"],
        "arm": PARITY_ARMS[segment],
        "segment": segment,
        "passed": True,
        "rho": float(rho),
        "rho_ci": None if rho_ci is None else [float(v) for v in rho_ci],
        "rho_gated": gated,
    }


def _check_rho_gate(report: Mapping[str, Any], overall: Mapping[str, Any], rho: float, gated: Mapping[str, Any],
                    regime: str, where: str) -> None:
    """Check that rho was compared with the regime's range and precision, that the gate ran, and that rho is inside
    the gate."""
    lora = regime == LORA_REGIME
    label = "b_lora" if lora else "fp32"
    low, high = RHO_WINDOW_LORA if lora else (RHO_GATE_FP32, math.inf)
    wanted = RHO_PRECISION_LORA if lora else RHO_PRECISION_FP32
    thresholds = report.get("thresholds") or {}
    rho_range, precision = thresholds.get("rho_range"), thresholds.get("rho_precision")
    # An infinite bound is recorded as the string "inf"/"-inf".
    bounds = ([float(b) if b in ("inf", "-inf") else b for b in rho_range]
              if isinstance(rho_range, list) and len(rho_range) == 2 else [None, None])
    _require(
        [
            (
                all(isinstance(bound, (int, float)) for bound in bounds) and bounds[0] >= low and bounds[1] <= high,
                f"it was compared with rho range {rho_range}, not {label}'s gate (--rho-range {low} {high})",
            ),
            (
                isinstance(precision, (int, float)) and precision <= wanted,
                f"its rho precision {precision!r} is not {label}'s {wanted} (--rho-precision)",
            ),
            (
                gated.get("all") is True and all(gated.values()),
                f"its rho gate did not run (rho_gated {gated}): the interval's half-width "
                f"{_number(overall.get('rho_halfwidth'))} is over {precision!r}, or the effect was too weak to "
                f"test; add prompts until it is at most {wanted}",
            ),
            (low <= rho <= high, f"rho {rho:.4f} is outside the {label} gate [{low}, {high}]"),
        ],
        where,
    )


def read_parity_arms(
    reports: Sequence[Path | str], *, regime: str, policy: str
) -> dict[str, dict[str, Any]]:
    """Return one policy's parity records by arm, at most one per arm."""
    arms: dict[str, dict[str, Any]] = {}
    for path in reports:
        record = read_parity(path, regime=regime, policy=policy)
        _require(
            [(record["arm"] not in arms, f"two {record['arm']} parity reports for {policy!r}")],
            f"parity reports for {policy!r}",
        )
        arms[record["arm"]] = record
    return {arm: arms[arm] for arm in PARITY_ARMS.values() if arm in arms}


def _rhos(parity: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> str:
    return "; ".join(
        f"{policy}: "
        + ", ".join(
            f"{arm} rho = {entry['rho']:.4f}"
            + (f" [{entry['rho_ci'][0]:.4f}, {entry['rho_ci'][1]:.4f}]" if entry.get("rho_ci") else "")
            for arm, entry in arms.items()
        )
        for policy, arms in parity.items()
    ) or "no parity report (scratch build)"


def parity_not_built(model_name: str) -> str:
    """Return the disclosure's sentence in place of parity rho, for a judge parity was never built for."""
    return (
        f"Parity B was not built for {model_name}: no rho is reported. Each policy arm is named by content in its "
        "condition identity, which every job recomputed from the served files before its first request."
    )


def disclosure(
    *, regime: str, engine: Mapping[str, Any], parity: Mapping[str, Mapping[str, Mapping[str, Any]]],
    policies: Sequence[str], reference: str, model_name: str, parity_built: bool = True,
) -> list[str]:
    """Return the disclosure sentences printed beside the result: the serving regime, the engine, parity rho per
    policy (or that parity was not built), and the base arm's serving."""
    version = engine.get("vllm_version") or "an unrecorded vLLM version"
    attention = ATTENTION_PRECISION.get(str(engine.get("TRITON_F32_DEFAULT")), f"TRITON_F32_DEFAULT={engine.get('TRITON_F32_DEFAULT')}")
    arms = ", ".join(policies)
    rhos = _rhos(parity)

    def measured(sentence: str) -> str:
        return sentence if parity_built else parity_not_built(model_name)

    if regime == REGIMES["float32"]:
        lines = [
            f"Each GRPO policy arm ({arms}) is its checkpoint merged into the {model_name} weights in fp32 and "
            "served in fp32.",
            f"Every arm runs on one fp32 engine: vLLM {version}, dtype float32, attention backend "
            f"{engine.get('attention_backend')} with {attention}, fp32 GEMMs at VLLM_FLOAT32_MATMUL_PRECISION="
            f"{engine.get('VLLM_FLOAT32_MATMUL_PRECISION')}; only the weights differ between arms.",
            measured(
                f"Parity B (HF-PEFT in fp32 against the merged checkpoint served in fp32, in prefill and decode arms), "
                f"a merge check expected at about 1 and gated at rho >= {RHO_GATE_FP32} where its 95% interval's "
                f"half-width is at most {RHO_PRECISION_FP32}: {rhos}."
            ),
        ]
        rollouts = (
            "Training sampled its rollouts from TRL's bf16 colocated engine, a rounded copy of the policy, with "
            "token-level importance sampling. The evaluation measures the merged policy itself, not that engine's copy."
        )
    elif regime == LORA_REGIME:
        low, high = RHO_WINDOW_LORA
        lines = [
            f"Under LoRA serving (serving.mode lora_bf16), each GRPO policy arm "
            f"({arms}) is its checkpoint's LoRA adapter, unmerged, served through vLLM's LoRA path in bf16, named by "
            "its SHA-256.",
            f"Every arm serves the {model_name} weights on one bf16 engine configuration: vLLM {version}, dtype "
            f"bfloat16, attention backend {engine.get('attention_backend')}. Each policy arm adds its adapter; the "
            "base arm serves none, and its engine is built without LoRA (LoRA is enabled only where an adapter is "
            "served).",
            measured(
                f"Parity B (HF-PEFT in fp32 against the adapter served by vLLM's LoRA path in bf16, in prefill and "
                f"decode arms): rho, the share of the adapter's effect on log-probabilities that the served adapter "
                f"reproduces, gated within [{low}, {high}] where its 95% interval's half-width is at most "
                f"{RHO_PRECISION_LORA}: {rhos}."
            ),
        ]
        rollouts = (
            "Training sampled its rollouts from TRL's bf16 colocated engine, a rounded copy of the policy; "
            "token-level importance sampling corrected the gap. The evaluation serves the adapter unmerged: vLLM adds "
            "its output to each layer's bf16 output, where a contribution below bf16 resolution rounds away, and "
            "parity B's rho measures how much of the adapter's effect survives."
        )
    else:
        below = [
            policy for policy, records in parity.items()
            if any(entry["rho"] < RHO_REPORT_BF16 for entry in records.values())
        ]
        lines = [
            f"Under the bf16 fallback, each GRPO policy arm ({arms}) is its checkpoint merged "
            "in fp32 from its fp32 LoRA factors and cast to bf16, served in bf16.",
            f"Every arm runs on one bf16 engine: vLLM {version}, dtype bfloat16, attention backend "
            f"{engine.get('attention_backend')}; only the weights differ between arms.",
            measured(
                f"Parity B's rho is the share of the adapter's effect on log-probabilities that survives the bf16 "
                f"merge (reporting threshold {RHO_REPORT_BF16}, in prefill and decode arms): {rhos}."
            ),
        ]
        if below:
            lines.append(
                f"For {', '.join(below)}, rho is below {RHO_REPORT_BF16}: the result is stated for the merged bf16 "
                "checkpoint only, not for the unmerged adapter."
            )
        rollouts = (
            "The evaluated bf16 checkpoints are also the bf16 model TRL's colocated engine sampled the training "
            "rollouts from, bf16(W + dW) up to fp32 rounding in the merge: they carry the same rounding of the "
            "adapter as that engine, which token-level importance sampling corrected for in training."
        )
    dtype = "fp32" if regime == REGIMES["float32"] else "bf16"
    return lines + [
        f"The base arm ({reference}: {model_name} with no adapter) is served in {dtype} on vLLM {version}; it is not "
        f"the bf16 {model_name} of the ruleset-size experiment (vLLM 0.19, naive-rules-above prompt).",
        rollouts,
    ]


def serving_regime(specs: Mapping[str, ConditionSpec], reference: str, where: str) -> str:
    """Return the one serving regime the arms' condition identities declare (merged fp32, merged bf16, or LoRA
    bf16), refusing a root that mixes them."""
    dtypes = {spec.serving_dtype for spec in specs.values()}
    _require(
        [
            (
                len(dtypes) == 1 and next(iter(dtypes)) in {*REGIMES, LORA_DTYPE},
                f"serving dtypes {sorted(map(str, dtypes))}; a training readout serves every arm in float32 or bfloat16 "
                "(the fallback, or LoRA serving), declared by the judge config's dtype key (a bf16 serving option "
                "declares dtype bfloat16: qwen3-8b.yaml leaves it to vLLM's auto, recorded as None)",
            )
        ],
        where,
    )
    dtype = next(iter(dtypes))
    adapters = {policy: spec.adapter_sha256 for policy, spec in specs.items()}
    if not any(adapters.values()):
        weights = specs[reference].model_weights_sha256
        underived = sorted(policy for policy, spec in specs.items()
                           if policy != reference and spec.model_weights_sha256 in (None, weights))
        _require(
            [
                (
                    not underived,
                    f"policies {underived} serve the reference's weights with no adapter: under merged serving each "
                    "policy arm is its merged checkpoint, named by its weights digest (a policy declared as a merged "
                    "checkpoint under LoRA serving is served as the base)",
                )
            ],
            where,
        )
        return REGIMES[dtype]
    bare = sorted(policy for policy, adapter in adapters.items() if policy != reference and not adapter)
    _require(
        [
            (dtype == LORA_DTYPE, f"the adapters were served in {dtype}; LoRA serving is bf16"),
            (adapters[reference] is None, f"the reference {reference!r} serves an adapter; the base serves none"),
            (not bare, f"policies {bare} serve no adapter beside policies that do: one regime serves every arm"),
            (
                all(spec.model_weights_sha256 is None for spec in specs.values())
                and len({spec.model_path for spec in specs.values()}) == 1,
                "under LoRA serving every arm serves the base weights, and these arms name other weights",
            ),
        ],
        where,
    )
    return LORA_REGIME


def claim(flat_row: Mapping[str, Any], kappa_row: Mapping[str, Any]) -> str:
    """Return the claim the two families' rows allow for one policy, a family counting only when its Holm p is
    significant: ``higher_inconsistency`` (flat inconsistency up), ``null`` (no flat change), ``more_consistent``
    (flat down and kappa up) or ``yes_shift`` (flat down alone)."""
    flat_down = flat_row["significant_holm"] and flat_row["estimate"] > 0
    flat_up = flat_row["significant_holm"] and flat_row["estimate"] < 0
    kappa_up = bool(kappa_row["significant_holm"]) and (kappa_row["estimate"] or 0.0) > 0
    if flat_up:
        return "higher_inconsistency"
    if not flat_down:
        return "null"
    return "more_consistent" if kappa_up else "yes_shift"


def _interval(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {key: summary[key] for key in ("estimate", "ci_low", "ci_high")}


def _same_call(recorded: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if recorded is None:
        return None
    return {"same_call_rate" if key == SAME_CALL_RATE_KEY else key: value for key, value in recorded.items()}


def _family_row(row: Mapping[str, Any], public_ids: Mapping[str, str]) -> dict[str, Any]:
    keys = (
        "a", "b", "policy_a", "policy_b", "estimate", "ci_low", "ci_high", "sd", "interval_basis", "p_value",
        "p_holm", "significant_holm", "family_size", "n_permutations", "test_method", "p_randomization",
        "n_undefined_replicates", "bootstrap_seed",
    )
    return {key: public_ids[row[key]] if key in ("a", "b") else row[key] for key in keys if key in row}


def build_from_campaign(
    estimates_path: Path | str,
    *,
    parity: Mapping[str, Path | str | Sequence[Path | str]],
    labels: Optional[Mapping[str, str]] = None,
    allow_incomplete: bool = False,
    allow_stack_mismatch: bool = False,
) -> dict[str, Any]:
    """Return the golden of one analysed campaign root, ``parity`` mapping each non-reference policy to its parity
    reports (one per arm; empty for a judge without parity) and ``labels`` giving display names."""
    estimates_path = Path(estimates_path)
    where = str(estimates_path)
    root = estimates_path.parent.parent
    data = estimates_path.read_bytes()
    estimates = json.loads(data.decode("utf-8"))
    refusal = recorded_mismatch_refusal(estimates, where)
    if refusal is not None and not allow_stack_mismatch:
        raise TrainingReadoutError(refusal)
    manifest = load_sampling_manifest(root / "sampling_manifest.json")
    scenarios = ScenarioManifest.load(root / "scenario_manifest.json")
    registered = load_conditions(root)
    complete = estimates.get("state") == "complete" and not estimates.get("excluded_conditions")
    final = manifest.get("campaign_mode") == CAMPAIGN_MODE_FINAL
    _require(
        [
            (complete or allow_incomplete, "the campaign is not complete (pass allow_incomplete for a scratch build)"),
            (final or allow_incomplete, f"campaign_mode {manifest.get('campaign_mode')!r}; the paper needs a final campaign"),
            (estimates.get("inference_method") == INFERENCE_METHOD, "unsupported inference method"),
            *[
                (estimates.get(key) == manifest.get(source), f"the estimates' {key} is not the campaign manifest's")
                for key, source in (
                    ("campaign_id", "campaign_id"), ("scenario_manifest_sha256", "scenario_manifest_sha256"),
                    ("n_decoder_draws", "expected_decoder_draws"), ("temperature", "temperature"),
                    ("truncation", "truncation"), ("max_semantic_attempts", "max_semantic_attempts"),
                    ("forced_json_requirement", "forced_json_requirement"),
                )
            ],
            (scenarios.manifest_sha256 == manifest["scenario_manifest_sha256"], "the scenario manifest is not the campaign's"),
            (estimates.get("n_scenarios") == scenarios.record_count, "the estimates' n_scenarios is not the manifest's"),
        ],
        where,
    )
    _, family, block = _the_family(estimates, registered, where)
    n_draws = int(manifest["expected_decoder_draws"])
    strict_think = bool(manifest.get("strict_think", False))
    result, missing_parity = family_golden(
        estimates["conditions"], block, family, registered, {cid: root / cid for cid in family.members}, scenarios,
        n_draws=n_draws, strict_think=strict_think, parity=parity, labels=labels, allow_incomplete=allow_incomplete,
        where=where,
    )
    result["provenance"] = {
        "complete": bool(complete and final and not missing_parity),
        "campaign_id": manifest["campaign_id"],
        "campaign_mode": manifest.get("campaign_mode"),
        "campaign_root": to_uri(root),
        "estimates": {"uri": to_uri(estimates_path), "sha256": hashlib.sha256(data).hexdigest()},
        "policy_family": "+".join(registered[cid].public_id for cid in family.members),
        **design_provenance(manifest, estimates, scenarios, n_draws=n_draws, strict_think=strict_think,
                            reference_spec=registered[family.policies[family.reference]]),
    }
    mismatches = recorded_mismatches(estimates)
    if mismatches:
        result["provenance"][STACK_MISMATCHES_KEY] = mismatches
    return result


def design_provenance(
    manifest: Mapping[str, Any], estimates: Mapping[str, Any], scenarios: ScenarioManifest, *, n_draws: int,
    strict_think: bool, reference_spec: ConditionSpec,
) -> dict[str, Any]:
    """Return the design a golden's provenance records: population, sampling design, inference settings, judge and
    prompt."""
    return {
        "scenario_manifest_sha256": scenarios.manifest_sha256,
        "scenario_source_sha256": scenarios.source_sha256,
        "scenario_set": scenarios.source_name,
        "n_scenarios": scenarios.record_count,
        "n_decoder_draws": n_draws,
        "temperature": manifest["temperature"],
        "truncation": manifest["truncation"],
        "max_semantic_attempts": manifest["max_semantic_attempts"],
        "forced_json_requirement": manifest["forced_json_requirement"],
        "retry_scope": manifest.get("retry_scope"),
        "strict_think": strict_think,
        "inference_method": estimates["inference_method"],
        "bootstrap": estimates["bootstrap"],
        "randomization_test": estimates["randomization_test"],
        "judge": {
            "model_name": reference_spec.model_name,
            "judge_type": reference_spec.judge_type,
            "prompt_style": reference_spec.prompt_style,
            "individual_rule_context": reference_spec.individual_rule_context,
            "rule_set_name": reference_spec.rule_set_name,
            "rule_set_sha256": reference_spec.rule_set_sha256,
        },
    }


def family_golden(
    rows: Mapping[str, Mapping[str, Any]],
    block: Mapping[str, Any],
    family: PolicyFamily,
    registered: Mapping[str, ConditionSpec],
    condition_dirs: Mapping[str, Path],
    scenarios: ScenarioManifest,
    *,
    n_draws: int,
    strict_think: bool,
    parity: Mapping[str, Path | str | Sequence[Path | str]],
    labels: Optional[Mapping[str, str]],
    allow_incomplete: bool,
    where: str,
) -> tuple[dict[str, Any], list[str]]:
    """Return the golden without its provenance and the parity reports it lacks, every estimate checked against the
    draws; ``registered`` and ``condition_dirs`` are keyed by the id the estimates analysed each member under."""
    ids = scenarios.scenario_ids
    reference = family.reference
    members = dict(family.policies)
    reference_spec = registered[members[reference]]
    model_name = reference_spec.model_name
    public_ids = {cid: registered[cid].public_id for cid in members.values()}
    labels = {policy: policy for policy in members} | dict(labels or {})
    shared = sorted({text for text in labels.values() if list(labels.values()).count(text) > 1})
    _require(
        [
            (set(labels) == set(members), f"display labels {sorted(labels)} are not the policies {sorted(members)}"),
            (not shared, f"display labels {shared} name more than one arm: the panels draw an arm by its "
                         "display label, so two arms of one label would be drawn as one"),
        ],
        where,
    )

    strata = read_strata([condition_dirs[cid] for cid in members.values()], ids)
    _require([(block.get("strata") == strata.to_dict(), "the recorded strata are not the scenario rows'")], where)
    rates: dict[str, float] = {}
    kappas: dict[str, Optional[float]] = {}
    arms: dict[str, Any] = {}
    for policy, cid in members.items():
        spec, row, arm = registered[cid], rows[cid], block["arms"][policy]
        cells, provenance = load_condition_outcomes(
            condition_dirs[cid], expected_scenario_ids=ids, expected_draws=n_draws,
            scenario_manifest_sha256=scenarios.manifest_sha256,
        )
        flat = outcome_grid(cid, cells, scenario_ids=ids, n_draws=n_draws, identity_id=spec.identity_id)
        verdicts = VerdictGrids.from_cells(cid, cells, scenario_ids=ids, n_draws=n_draws, identity_id=spec.identity_id)
        parts = kappa_parts(kappa_features(verdicts, strata).mean(axis=0, keepdims=True), len(strata.names))[0]
        kappa = None if not np.isfinite(parts[0]) else float(parts[0])
        _require(
            [
                (row["condition_fingerprint"] == spec.fingerprint == provenance["condition_fingerprint"], f"{cid}: stale fingerprint"),
                (row["grid_digest"] == flat.digest and _close(row["estimate"], flat.rate), f"{cid}: the flat rate is not its draws'"),
                (arm["grid_digests"] == verdicts.digests, f"{cid}: the judgment grids are not its draws'"),
                (_close(arm["kappa"]["estimate"], kappa), f"{cid}: kappa is not its draws'"),
                (_close(arm["yes_rate"]["estimate"], float(verdicts.shares("full", YES_PHI).mean())), f"{cid}: the yes-rate is not its draws'"),
                (
                    _close(arm["always_yes_floor"]["estimate"], float(1 - verdicts.shares("rule_max", YES_PHI).mean())),
                    f"{cid}: the always-yes floor is not its draws'",
                ),
            ],
            where,
        )
        rates[policy], kappas[policy] = flat.rate, kappa
        arms[policy] = {
            "condition_id": public_ids[cid],
            "label": labels[policy],
            "reference": policy == reference,
            "model_path": spec.model_path,
            "model_weights_sha256": spec.model_weights_sha256,
            **({"adapter_sha256": spec.adapter_sha256} if spec.adapter_sha256 else {}),
            "serving_dtype": spec.serving_dtype,
            "flat": {**_interval(row), "grid_digest": row["grid_digest"]},
            "kappa": {**_interval(arm["kappa"]), "p_observed": arm["kappa"]["p_observed"], "p_chance": arm["kappa"]["p_chance"]},
            "yes_rate": _interval(arm["yes_rate"]),
            "always_yes_floor": _interval(arm["always_yes_floor"]),
            "m_stratified": {
                stratum: {**_interval(entry), "n_draws": entry["n_draws"]} for stratum, entry in arm["m_stratified"].items()
            },
            "same_call": _same_call(arm.get(SAME_CALL_KEY)),
            "attempt_statistics": block["attempt_statistics"][policy],
            "first_attempt_acceptance_share": block["first_attempt_acceptance_share"][policy],
        }

    pairs = family.pairs
    flat_rows, kappa_rows = block["contrasts"], block["discrimination_contrasts"]
    _check_family_rows(flat_rows, pairs, key="flat contrasts", test_method=TEST_METHOD, where=where)
    _check_family_rows(kappa_rows, pairs, key="kappa contrasts", test_method=KAPPA_TEST_METHOD, where=where)
    label_of = {cid: policy for policy, cid in members.items()}
    for row in flat_rows:
        a, b = label_of[row["a"]], label_of[row["b"]]
        _require([(_close(row["estimate"], rates[a] - rates[b]), f"{row['b']}: the flat contrast is not rate(a) - rate(b)")], where)
    for row in kappa_rows:
        a, b = label_of[row["a"]], label_of[row["b"]]
        expected = None if kappas[a] is None or kappas[b] is None else kappas[b] - kappas[a]
        _require([(_close(row["estimate"], expected), f"{row['b']}: the kappa contrast is not kappa(b) - kappa(a)")], where)

    regime = serving_regime({policy: registered[cid] for policy, cid in members.items()}, reference, where)
    built = parity_b_built(model_name, strict_think, where)
    engine = _engine_disclosure([condition_dirs[cid] for cid in members.values()], n_draws, where)
    policies = [policy for policy in members if policy != reference]
    _require(
        [
            (set(parity) <= set(policies), f"parity reports for {sorted(set(parity) - set(policies))}, which are not policies"),
            (built or not parity, f"parity reports for {sorted(parity)}, but parity B was never built for {model_name}: pass none"),
        ],
        where,
    )
    parity_records = {
        policy: read_parity_arms(
            [parity[policy]] if isinstance(parity[policy], (str, Path)) else parity[policy], regime=regime, policy=policy,
        )
        for policy in policies if policy in parity
    }
    missing_parity = [
        f"{policy} ({arm})" for policy in policies for arm in PARITY_ARMS.values()
        if arm not in parity_records.get(policy, {})
    ] if built else []
    _require([(not missing_parity or allow_incomplete, f"no parity report for {missing_parity}")], where)

    comparisons = {}
    kappa_by_member = {row["b"]: row for row in kappa_rows}
    for row in flat_rows:
        policy = label_of[row["b"]]
        verdict = claim(row, kappa_by_member[row["b"]])
        comparisons[policy] = {
            "claim": verdict,
            "flat": _family_row(row, public_ids),
            "kappa": _family_row(kappa_by_member[row["b"]], public_ids),
            "harm_flag": block["stratified_differences"][policy]["harm_flag"],
            "stratified_differences": block["stratified_differences"][policy],
            "decomposition": block["decompositions"][policy],
        }
    result = {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "source": GOLDEN_SOURCE,
        "metric": METRIC,
        "claim_rule": CLAIM_RULE,
        "reference_policy": reference,
        "policies": list(members),
        "flat_orientation": POLICY_ORIENTATION,
        "kappa_orientation": KAPPA_ORIENTATION,
        "strata": strata.to_dict(),
        "arms": arms,
        "comparisons": comparisons,
        "disclosure": {
            "regime": regime,
            "serving_dtype": reference_spec.serving_dtype,
            "engine": engine,
            "parity_b": parity_records,
            **({} if built else {"parity_b_not_built": parity_not_built(model_name)}),
            "lines": disclosure(
                regime=regime, engine=engine, parity=parity_records, policies=policies, reference=reference,
                model_name=model_name, parity_built=built,
            ),
        },
    }
    return result, missing_parity

