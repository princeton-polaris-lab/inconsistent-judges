from __future__ import annotations

import json
from pathlib import Path

import pytest

from ruleforge.evaluate.estimates import (
    CONTRAST_CAMPAIGN_FIELDS,
    amendment_sentence,
    campaign_settings_differences,
    frozen_settings,
    public_amendment,
)
from ruleforge.evaluate.amendment import AmendedBundle, AmendmentError, CapAmendment
from ruleforge.evaluate.identity import MANIFEST_AMENDMENTS_KEY

SCENARIO = "c" * 64


def record(*, old_cap=20, new_cap=50, checks=None, held=()):
    """A cap amendment's record as the amendment tool writes it into a sampling manifest."""
    bundle = AmendedBundle(condition_id="m/naive", decoder_draw_id=3, scenario_id=SCENARIO, shard=None,
                           attempts=old_cap, prompts_sha256="0" * 64, files={})
    return CapAmendment(old_cap=old_cap, new_cap=new_cap, recorded_at="2026-10-14T00:00:00+00:00",
                        decided_by="a maintainer", decided_on="2026-10-14", reason="R exhausted", ledger="note 143",
                        code_state=("b" * 40, False, "2" * 64), prior_code_states=(("a" * 40, False, "1" * 64),),
                        bundles=(bundle,), conditions=("m/naive",), checks=checks or {"prompts": "match"},
                        held=held).to_dict()


def manifest(cap=20, **fields):
    return {"campaign_id": "x", "max_semantic_attempts": cap, "retry_scope": "scenario_bundle",
            "scenarios_per_request_block": None, "generation_seed_base": 16, **fields}


def test_an_amended_manifest_reads_at_its_frozen_cap_and_any_other_as_it_is():
    plain = manifest()
    assert frozen_settings(plain) == plain and frozen_settings(plain) is not plain
    amended = manifest(50, **{MANIFEST_AMENDMENTS_KEY: [record()]})
    assert frozen_settings(amended) == plain
    assert amended["max_semantic_attempts"] == 50 and MANIFEST_AMENDMENTS_KEY in amended
    with pytest.raises(AmendmentError, match="one cap amendment"):
        frozen_settings(manifest(50, **{MANIFEST_AMENDMENTS_KEY: [record(), record()]}))


def test_a_root_that_took_a_ladder_shares_its_frozen_cap_and_another_cap_still_differs():
    amended = manifest(50, **{MANIFEST_AMENDMENTS_KEY: [record()]})
    assert campaign_settings_differences({"raw": manifest(), "laddered": amended}) == []
    every_key = sorted(set(manifest()) | set(amended))
    assert campaign_settings_differences({"raw": manifest(), "laddered": amended}, fields=every_key) == []
    assert campaign_settings_differences({"raw": manifest(), "other": manifest(40)}) == [
        "max_semantic_attempts (raw 20, other 40)"]
    from_five = manifest(50, **{MANIFEST_AMENDMENTS_KEY: [record(old_cap=5)]})
    assert campaign_settings_differences({"raw": manifest(), "laddered": from_five}) == [
        "max_semantic_attempts (raw 20, laddered 5)"]
    assert "max_semantic_attempts" in CONTRAST_CAMPAIGN_FIELDS


def disclosure(**kwargs):
    """An analysis's disclosure of the record (``amendment.disclosure``)."""
    summary = CapAmendment.from_dict(record(**kwargs)).summary()
    return {**summary, "bundles": [{"condition_id": "m/naive", "decoder_draw_id": 3, "scenario_id": SCENARIO,
                                    "held_by": "campaign root", "attempts_at_old_cap": 20, "outcome": "accepted",
                                    "accepted_at_attempt": 23, "attempts": 24}],
            "code_states": {"m/naive": {"commit bbbbbbbbbbbb, sources 222222222222": ["draw-003"]}}}


def test_the_disclosure_quotes_the_record_s_raised_carried_count_and_held_bundles():
    plain = amendment_sentence("x", disclosure())
    assert plain == (
        "Campaign x: max_semantic_attempts raised from 20 to 50 for bundles exhausted at 20. "
        f"m/naive draw 3 scenario {SCENARIO[:12]}: accepted after 24 attempts."
        " Code states: m/naive: commit bbbbbbbbbbbb, sources 222222222222 (draw-003).")
    raised = amendment_sentence("x", disclosure(checks={"raised_carried": "1 bundle(s) a rebase carried over"}))
    assert raised == plain + " Its record: 1 bundle(s) a rebase carried over."
    held = amendment_sentence("x", disclosure(held=(("m/naive", 0, "d" * 64),)))
    assert held == plain + (" Held at the old cap, as the rebase carried them over exhausted: "
                            f"m/naive draw 0 scenario {'d' * 12}.")
    pending = disclosure()
    pending["bundles"][0] = {key: value for key, value in pending["bundles"][0].items() if key != "attempts"}
    assert "m/naive draw 3 scenario cccccccccccc: accepted. Code states" in amendment_sentence("x", pending)


def test_an_amendment_is_quoted_without_who_decided_it_why_or_its_checks_wording():
    amendment = record(new_cap=50, checks={"prompts": "match", "max_accepted_attempt": {"m/naive": 7}})
    quoted = public_amendment(amendment)
    assert quoted["new_cap"] == 50 and quoted["checks"] == {"max_accepted_attempt": {"m/naive": 7}}
    assert not {"decided_by", "ledger", "reason"} & set(quoted)
    assert public_amendment({"old_cap": 20, "reason": "R"}) == {"old_cap": 20}


def test_the_published_amendment_is_quoted_as_its_golden_quotes_it():
    golden = json.loads((Path(__file__).resolve().parents[1] / "results/ruleset_size/curves.json").read_text("utf-8"))
    [quoted] = [row["campaign"]["amendments"][0] for row in golden if row["campaign"].get("amendments")]
    recorded = {**quoted, "reason": "as the record words it", "decided_by": "a maintainer", "ledger": "note 1",
                "checks": {**quoted["checks"], "prompts": "as the record words it"}}
    assert public_amendment(recorded) == quoted
