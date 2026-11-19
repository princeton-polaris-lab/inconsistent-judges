from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ruleforge.core.errors import RuleForgeError
from ruleforge.core.paths import to_uri
from ruleforge.evaluate.amendment import AMENDED_FIELD, cap_amendment
from ruleforge.evaluate.identity import MANIFEST_AMENDMENTS_KEY, SCENARIO_EXCLUSION_KEY, retry_scope_record
from ruleforge.evaluate.stack import STACK_MISMATCHES_KEY, recorded_mismatch_refusal, recorded_mismatches

__all__ = [
    "CONTRAST_CAMPAIGN_FIELDS",
    "SHARED_DESIGN_FIELDS",
    "CampaignEstimatesError",
    "amendment_sentence",
    "public_amendment",
    "public_amendments",
    "campaign_settings_differences",
    "estimates_source",
    "frozen_settings",
    "is_complete",
    "load_estimates",
    "recorded_stack_mismatch_refusal",
]

SHARED_DESIGN_FIELDS = (
    "scenario_manifest_sha256",
    "n_scenarios",
    "n_decoder_draws",
    "temperature",
    "truncation",
    "forced_json_requirement",
    "inference_method",
    "bootstrap",
    "randomization_test",
)


# Manifest settings that two arms of one judge contrasted across roots must share.
CONTRAST_CAMPAIGN_FIELDS = (
    "max_semantic_attempts",
    "retry_scope",
    "scenarios_per_request_block",
    "generation_seed_base",
)


class CampaignEstimatesError(ValueError, RuleForgeError):
    """The estimates files cannot be read as one experiment."""


def is_complete(payload: Mapping[str, Any]) -> bool:
    """A strict-gate campaign: state ``complete`` and no excluded condition."""
    return payload.get("state") == "complete" and not payload.get("excluded_conditions")


def load_estimates(
    paths: Sequence[Path], *, allow_incomplete: bool, allow_stack_mismatch: bool = False,
) -> List[Dict[str, Any]]:
    """Read every estimates.json of one experiment, each with its path (``_path``) and bytes' digest (``_sha256``);
    stack mismatches need ``allow_stack_mismatch``, a fallback's exclusion ``allow_incomplete``."""
    if not paths:
        raise CampaignEstimatesError("at least one --estimates file is required")
    payloads: List[Dict[str, Any]] = []
    for path in paths:
        data = Path(path).read_bytes()
        payload = json.loads(data.decode("utf-8"))
        payload["_path"], payload["_sha256"] = Path(path), hashlib.sha256(data).hexdigest()
        recorded = payload.get(SCENARIO_EXCLUSION_KEY)
        if not allow_incomplete:
            if recorded is not None:
                raise CampaignEstimatesError(
                    f"{path}: its estimates drop scenario {recorded['scenario_id'][:12]} from the arms of judge "
                    f"{recorded['judge']!r} (the out-of-attempts fallback), which no paper build reads: the toolkit "
                    "reads them with their disclosures (Campaign.open(root).result(), ruleforge.evaluate.api), or "
                    "analyse the root again without the fallback once its exhausted draw is recovered (pass "
                    "--allow-incomplete for a scratch build)"
                )
            if not is_complete(payload):
                raise CampaignEstimatesError(
                    f"{path}: campaign state {payload.get('state')!r} with "
                    f"{len(payload.get('excluded_conditions') or {})} excluded condition(s); the "
                    "paper's table needs a complete campaign (pass --allow-incomplete for a scratch build)"
                )
        refusal = recorded_mismatch_refusal(payload, str(path))
        if refusal is not None and not allow_stack_mismatch:
            raise CampaignEstimatesError(refusal)
        payloads.append(payload)
    first = payloads[0]
    for payload in payloads[1:]:
        for field in SHARED_DESIGN_FIELDS:
            if payload.get(field) != first.get(field):
                raise CampaignEstimatesError(
                    f"{payload['_path']}: {field} {payload.get(field)!r} differs from "
                    f"{first['_path']}: {first.get(field)!r}; these roots are not one experiment"
                )
    seen: Dict[str, Path] = {}
    for payload in payloads:
        for condition_id in payload["conditions"]:
            if condition_id in seen:
                raise CampaignEstimatesError(
                    f"condition {condition_id} appears in both {seen[condition_id]} and "
                    f"{payload['_path']}"
                )
            seen[condition_id] = payload["_path"]
    return payloads


def frozen_settings(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a sampling manifest as its campaign froze it: an amendment's old cap restored and its record set aside."""
    amendment = cap_amendment(manifest)
    if amendment is None:
        return dict(manifest)
    frozen = {key: value for key, value in manifest.items() if key != MANIFEST_AMENDMENTS_KEY}
    frozen[AMENDED_FIELD] = amendment.old_cap
    return frozen


def campaign_settings_differences(
    manifests: Mapping[str, Mapping[str, Any]], fields: Sequence[str] = CONTRAST_CAMPAIGN_FIELDS
) -> List[str]:
    """Return each of ``fields`` the sampling manifests do not share, as ``field (name value, ...)``."""
    frozen = {name: frozen_settings(manifest) for name, manifest in manifests.items()}
    return [
        f"{field} (" + ", ".join(f"{name} {manifest.get(field)!r}" for name, manifest in frozen.items()) + ")"
        for field in fields
        if any(manifest.get(field) != next(iter(frozen.values())).get(field) for manifest in frozen.values())
    ]


def public_amendment(amendment: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recorded cap amendment as a golden quotes it: its caps, bundles and code states, without who decided
    it, why, or the wording of its checks (their counts stay)."""
    kept = {key: value for key, value in amendment.items() if key not in ("decided_by", "ledger", "reason")}
    if isinstance(kept.get("checks"), Mapping):
        kept["checks"] = {name: value for name, value in kept["checks"].items() if not isinstance(value, str)}
    return kept


def public_amendments(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``{amendments key: the amendments as a golden quotes them}``, or nothing for a root without any."""
    amendments = payload.get(MANIFEST_AMENDMENTS_KEY)
    return {MANIFEST_AMENDMENTS_KEY: [public_amendment(amendment) for amendment in amendments]} if amendments else {}


def amendment_sentence(campaign_id: str, amendment: Mapping[str, Any]) -> str:
    """Return one sentence disclosing a cap amendment as an analysis records it."""
    bundles = "; ".join(
        f"{bundle['condition_id']} draw {bundle['decoder_draw_id']} scenario {bundle['scenario_id'][:12]}: "
        f"{bundle['outcome']}" + (f" after {bundle['attempts']} attempts" if "attempts" in bundle else "")
        for bundle in amendment["bundles"])
    states = "; ".join(f"{cid}: " + ", ".join(f"{state} ({', '.join(draws)})" for state, draws in found.items())
                       for cid, found in amendment["code_states"].items())
    sentence = (
        f"Campaign {campaign_id}: {amendment['field']} raised from {amendment['old_cap']} to "
        f"{amendment['new_cap']} for bundles exhausted at {amendment['old_cap']}. {bundles}. "
        f"Code states: {states}.")
    raised = (amendment.get("checks") or {}).get("raised_carried")
    if raised:
        sentence += f" Its record: {raised}."
    held = amendment.get("held") or ()
    if held:
        sentence += (" Held at the old cap, as the rebase carried them over exhausted: "
                     + "; ".join(f"{item['condition_id']} draw {item['decoder_draw_id']} scenario "
                                 f"{item['scenario_id'][:12]}" for item in held) + ".")
    return sentence


def estimates_source(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return one joined root as provenance records it: an ``artifact://`` URI, its digest and the campaign fields."""
    mismatches = recorded_mismatches(payload)
    return {
        "campaign_id": payload["campaign_id"],
        "state": payload.get("state"),
        "estimates": to_uri(payload["_path"]),
        "estimates_sha256": payload["_sha256"],
        "created_at": payload.get("created_at"),
        "max_semantic_attempts": payload.get("max_semantic_attempts"),
        **retry_scope_record(payload),
        **({STACK_MISMATCHES_KEY: mismatches} if mismatches else {}),
        **public_amendments(payload),
        **({SCENARIO_EXCLUSION_KEY: payload[SCENARIO_EXCLUSION_KEY]} if payload.get(SCENARIO_EXCLUSION_KEY) else {}),
    }


def recorded_stack_mismatch_refusal(provenance: Mapping[str, Any], where: str) -> str | None:
    """Return why a result is not re-rendered: a campaign it joined reports stack mismatches; None when none does."""
    flagged = [
        campaign.get("campaign_id")
        for campaign in provenance.get("campaigns") or []
        if isinstance(campaign, Mapping) and campaign.get(STACK_MISMATCHES_KEY)
    ]
    if not flagged:
        return None
    return (
        f"{where} was built from campaign(s) {flagged} that report inference-stack mismatches "
        "(provenance.campaigns[].stack_mismatches); it is not re-rendered without --allow-stack-mismatch"
    )
