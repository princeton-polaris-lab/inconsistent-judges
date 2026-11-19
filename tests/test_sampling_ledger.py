from __future__ import annotations

import json

import pytest

from ruleforge.evaluate.ledger import (
    LEDGER_FNAME,
    RAW_OUTPUTS_FNAME,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_TRANSPORT_FAILED,
    AttemptRecord,
    ComponentRecord,
    DrawLedger,
    LedgerError,
    raw_output_sha256,
    raw_row_key,
)

IDENTITY = dict(campaign_id="c", condition_id="k", decoder_draw_id=0)


def component(component_id="full_ruleset", valid=True, **overrides):
    fields = dict(
        component_id=component_id,
        valid=valid,
        failure_reason=None if valid else "invalid_label",
        judgment="yes" if valid else None,
        finish_reason="stop",
        generation_seed=5,
        seed_sent=True,
        raw_output_sha256=raw_output_sha256("yes"),
    )
    fields.update(overrides)
    return ComponentRecord(**fields)


def record(scenario_id="s1", attempt=0, status=STATUS_ACCEPTED, block="c/k/draw-000/attempt-0/block-0000", **overrides):
    components = overrides.pop("components", None)
    if components is None:
        components = (component(),) if status == STATUS_ACCEPTED else (component(valid=False),)
    fields = dict(
        scenario_id=scenario_id,
        semantic_attempt_id=attempt,
        request_block_id=block,
        status=status,
        seed_capability="local_deterministic",
        components=components,
        started_at="t0",
        completed_at="t1",
        **IDENTITY,
    )
    fields.update(overrides)
    return AttemptRecord(**fields)


def raw_row(scenario_id="s1", attempt=0, block="c/k/draw-000/attempt-0/block-0000", component_id="full_ruleset", text="yes"):
    return {
        "scenario_id": scenario_id, "decoder_draw_id": 0, "semantic_attempt_id": attempt,
        "request_block_id": block, "component_id": component_id,
        "raw_response": text, "completion": text, "finish_reason": "stop", "error": None,
        "raw_output_sha256": raw_output_sha256(text),
    }


@pytest.fixture
def ledger(tmp_path):
    return DrawLedger(tmp_path / "draw-000", **IDENTITY)


def test_raw_output_digest_is_none_only_for_nothing():
    assert raw_output_sha256(None) is None
    assert raw_output_sha256("") == raw_output_sha256("")
    assert raw_output_sha256("a") != raw_output_sha256("b")


def test_records_round_trip_through_json():
    rec = record(components=(component(rule_judgments={"Rule 1": "no"}, preempted_rules=("Rule 2",)),))
    payload = json.loads(json.dumps(rec.to_dict()))
    assert AttemptRecord.from_dict(payload) == rec
    assert payload["components"][0]["preempted_rules"] == ["Rule 2"]
    assert rec.key == ("s1", 0) and rec.consumes_attempt


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"status": "pending"}, "unknown attempt status"),
        ({"schema_version": 3}, "schema_version"),
        ({"components": (component(valid=False),)}, "accepted attempt cannot carry an invalid"),
    ],
)
def test_records_refuse_impossible_states(overrides, message):
    with pytest.raises(LedgerError, match=message):
        record(**overrides)


def test_from_dict_refuses_unknown_fields():
    payload = record().to_dict()
    payload["seed"] = 1
    with pytest.raises(LedgerError, match="malformed attempt record"):
        AttemptRecord.from_dict(payload)


def test_an_empty_directory_loads_as_nothing(ledger):
    state = ledger.load()
    assert state.records == [] and state.accepted == {}
    assert state.next_attempt("s1") == 0 and not state.is_accepted("s1")
    assert list(ledger.iter_raw()) == []


def test_append_then_load_indexes_accepted_rejected_and_transport_rows(ledger):
    ledger.append(
        [
            record("s1", 0, STATUS_ACCEPTED),
            record("s2", 0, STATUS_REJECTED),
            record("s3", 0, STATUS_TRANSPORT_FAILED, components=(component(valid=False, failure_reason="transport_error"),)),
        ],
        [raw_row("s1"), raw_row("s2", text="maybe")],
    )
    state = ledger.load()
    assert set(state.accepted) == {"s1"}
    assert state.vouched == {(sid, "c/k/draw-000/attempt-0/block-0000") for sid in ("s1", "s2", "s3")}
    assert state.next_attempt("s1") == 1
    assert state.next_attempt("s2") == 1
    assert state.next_attempt("s3") == 0
    assert [row["scenario_id"] for row in ledger.iter_raw()] == ["s1", "s2"]
    assert raw_row_key(raw_row("s2")) == ("s2", 0, "c/k/draw-000/attempt-0/block-0000")


def test_a_transport_failure_does_not_consume_the_attempt(ledger):
    ledger.append([record("s1", 0, STATUS_TRANSPORT_FAILED, block="b0")], [])
    ledger.append([record("s1", 0, STATUS_REJECTED, block="b1")], [])
    ledger.append([record("s1", 1, STATUS_ACCEPTED, block="b2")], [])
    state = ledger.load()
    assert state.next_attempt("s1") == 2
    assert state.is_accepted("s1")


def test_the_ledger_refuses_a_second_accepted_row_for_one_key(ledger):
    ledger.append([record("s1", 0, STATUS_ACCEPTED, block="b0")], [])
    ledger.append([record("s1", 1, STATUS_ACCEPTED, block="b1")], [])
    with pytest.raises(LedgerError, match="already accepted"):
        ledger.load()


def test_the_ledger_refuses_a_gap_in_attempt_ids(ledger):
    ledger.append([record("s1", 1, STATUS_REJECTED)], [])
    with pytest.raises(LedgerError, match="semantic_attempt_id 1"):
        ledger.load()


def test_the_ledger_refuses_the_same_block_recorded_twice(ledger):
    ledger.append([record("s1", 0, STATUS_TRANSPORT_FAILED, block="b0")], [])
    ledger.append([record("s1", 0, STATUS_TRANSPORT_FAILED, block="b0")], [])
    with pytest.raises(LedgerError, match="records block b0 twice"):
        ledger.load()


def test_one_block_may_carry_many_scenarios(ledger):
    ledger.append([record("s1", 0, block="b0"), record("s2", 0, block="b0")], [])
    assert set(ledger.load().accepted) == {"s1", "s2"}


def test_a_row_from_another_draw_is_refused_on_write_and_read(ledger, tmp_path):
    with pytest.raises(LedgerError, match="belongs to"):
        ledger.append([record(decoder_draw_id=1)], [])
    other = DrawLedger(tmp_path / "draw-000", campaign_id="c", condition_id="k", decoder_draw_id=1)
    ledger.append([record()], [])
    with pytest.raises(LedgerError, match="belongs to"):
        other.load()


def test_a_torn_trailing_ledger_row_is_repaired_only_when_asked(ledger):
    ledger.append([record("s1", 0, block="b0")], [])
    with ledger.ledger_path.open("a") as handle:
        handle.write('{"scenario_id": "s2", "semantic_att')
    with pytest.raises(LedgerError, match="torn trailing record"):
        ledger.load()
    state = ledger.load(repair=True)
    assert set(state.accepted) == {"s1"}
    assert ledger.ledger_path.read_text().count("\n") == 1


def test_a_torn_first_ledger_row_is_refused_without_repair_too(ledger):
    ledger.directory.mkdir(parents=True, exist_ok=True)
    ledger.ledger_path.write_text('{"scenario_id": "s1", "semantic_att')
    with pytest.raises(LedgerError, match="torn trailing record"):
        ledger.load()
    assert ledger.ledger_path.stat().st_size == 35
    assert ledger.load(repair=True).records == []
    assert ledger.ledger_path.read_bytes() == b""
    assert ledger.load().records == []


@pytest.mark.parametrize("leading", [0, 1])
def test_a_whole_last_ledger_row_without_its_newline_is_kept_on_repair_with_its_raw_output(ledger, leading):
    ids = [f"s{i}" for i in range(leading)] + ["s9"]
    ledger.append([record(sid, 0, block=f"b-{sid}") for sid in ids], [raw_row(sid, block=f"b-{sid}") for sid in ids])
    whole = ledger.ledger_path.read_bytes()
    ledger.ledger_path.write_bytes(whole.removesuffix(b"\n"))
    with pytest.raises(LedgerError, match="torn trailing record"):
        ledger.load()
    assert ledger.ledger_path.read_bytes() == whole.removesuffix(b"\n")
    assert [row.scenario_id for row in ledger.load(repair=True).records] == ids
    assert ledger.ledger_path.read_bytes() == whole
    assert [row["scenario_id"] for row in ledger.iter_raw()] == ids
    assert len(ledger.load().records) == len(ids)


def test_a_torn_tail_cut_inside_a_character_is_refused_as_a_torn_record(ledger):
    """The ledger's rows hold raw UTF-8 (ensure_ascii=False)."""
    ledger.append([record("s1", 0, block="b0")], [])
    with ledger.ledger_path.open("ab") as handle:
        handle.write('{"scenario_id": "s2 —'.encode("utf-8")[:-1])
    with pytest.raises(LedgerError, match="torn trailing record"):
        ledger.load()


def test_a_damaged_ledger_row_that_rows_follow_is_refused_on_repair_and_nothing_is_dropped(ledger):
    ledger.append([record(f"s{i}", 0, block=f"b{i}") for i in range(3)], [raw_row(f"s{i}", block=f"b{i}")
                                                                          for i in range(3)])
    rows = ledger.ledger_path.read_bytes().split(b"\n")
    ledger.ledger_path.write_bytes(b"\n".join([rows[0], rows[1][:40], *rows[2:]]))
    before = (ledger.ledger_path.read_bytes(), ledger.raw_path.read_bytes())
    with pytest.raises(ValueError, match="line 2 of .* is not a JSON record"):
        ledger.load(repair=True)
    assert (ledger.ledger_path.read_bytes(), ledger.raw_path.read_bytes()) == before


def test_orphan_raw_rows_are_dropped_on_repair_and_kept_otherwise(ledger, caplog):
    ledger.append([record("s1", 0, block="b0")], [raw_row("s1", block="b0")])
    # A crash after the raw rows but before the ledger rows of block b1.
    ledger.append([], [raw_row("s2", block="b1"), raw_row("s2", block="b1", component_id="Rule 1")])
    assert len(list(ledger.iter_raw())) == 3
    ledger.load()
    assert len(list(ledger.iter_raw())) == 3
    with caplog.at_level("WARNING", logger="ruleforge.evaluate.ledger"):
        state = ledger.load(repair=True)
    assert "dropping 2 raw output row(s)" in caplog.text
    assert [row["scenario_id"] for row in ledger.iter_raw()] == ["s1"]
    assert state.next_attempt("s2") == 0


def test_a_torn_raw_tail_is_truncated_on_repair(ledger):
    ledger.append([record("s1", 0, block="b0")], [raw_row("s1", block="b0")])
    with ledger.raw_path.open("a") as handle:
        handle.write('{"scenario_id": "s1", "semantic_attempt_id": 0, "request_block_id": "b0", "comp')
    ledger.load(repair=True)
    assert len(list(ledger.iter_raw())) == 1


def test_blank_lines_in_the_raw_file_are_skipped(ledger):
    ledger.append([record("s1", 0, block="b0")], [raw_row("s1", block="b0")])
    with ledger.raw_path.open("a") as handle:
        handle.write("\n")
    assert [row["scenario_id"] for row in ledger.iter_raw()] == ["s1"]


def test_repair_with_no_raw_file_is_a_no_op(ledger):
    ledger.append([record("s1", 0)], [])
    assert not ledger.raw_path.exists()
    assert set(ledger.load(repair=True).accepted) == {"s1"}


def test_the_files_are_named_as_the_contract_says(ledger):
    assert ledger.ledger_path.name == LEDGER_FNAME == "attempts.jsonl"
    assert ledger.raw_path.name == RAW_OUTPUTS_FNAME == "raw_outputs.jsonl"


BUNDLE = ("full_ruleset", "Rule 1", "Rule 2")


def crecord(attempt, valid, status=None, *, scenario_id="s1", block=None, reason="invalid_label"):
    """A component-scope row: ``valid`` maps each component it carries to its validity."""
    components = tuple(
        component(cid, ok, failure_reason=None if ok else reason) for cid, ok in valid.items()
    )
    if status is None:
        status = STATUS_ACCEPTED if all(valid.values()) else STATUS_REJECTED
    return record(scenario_id, attempt, status, block=block or f"b{attempt}", components=components,
                  retry_scope="component")


def test_a_component_scope_row_names_its_scope_and_a_bundle_row_reads_as_before():
    rec = crecord(0, {"full_ruleset": True})
    assert rec.to_dict()["retry_scope"] == "component"
    assert AttemptRecord.from_dict(json.loads(json.dumps(rec.to_dict()))) == rec
    assert "retry_scope" not in record().to_dict()
    with pytest.raises(LedgerError, match="unknown retry scope"):
        record(retry_scope="rule")


def test_a_row_names_the_stack_it_was_judged_on_and_one_recorded_before_reads_as_before():
    rec = record(judged_on="ab" * 32)
    assert list(rec.to_dict())[-1] == "judged_on" and rec.to_dict()["judged_on"] == "ab" * 32
    assert AttemptRecord.from_dict(json.loads(json.dumps(rec.to_dict()))) == rec
    assert "judged_on" not in record().to_dict()
    assert AttemptRecord.from_dict(record().to_dict()).judged_on is None


def test_component_scope_settles_valid_components_and_regenerates_the_rest(ledger):
    ledger.append([crecord(0, {"full_ruleset": True, "Rule 1": False, "Rule 2": True})], [])
    state = ledger.load()
    assert state.retry_scope == "component" and state.bundles["s1"] == BUNDLE
    assert state.components_at("s1", 1, BUNDLE) == ("Rule 1",)
    assert state.components_at("s1", 0, BUNDLE) == BUNDLE
    assert state.components_at("s2", 0, BUNDLE) == BUNDLE
    ledger.append([crecord(1, {"Rule 1": False}, reason="invalid_json")], [])
    ledger.append([crecord(2, {"Rule 1": True})], [])
    state = ledger.load()
    assert state.is_accepted("s1") and state.next_attempt("s1") == 3
    assert state.components_at("s1", 2, BUNDLE) == ("Rule 1",)
    chosen = state.accepted_components("s1")
    assert list(chosen) == list(BUNDLE)
    assert {cid: rec.semantic_attempt_id for cid, (rec, _) in chosen.items()} == {
        "full_ruleset": 0, "Rule 1": 2, "Rule 2": 0,
    }
    bundle = state.accepted_bundle("s1")
    assert bundle.semantic_attempt_id == 2 and bundle.status == STATUS_ACCEPTED
    assert [c.component_id for c in bundle.components] == list(BUNDLE) and all(c.valid for c in bundle.components)
    assert state.accepted["s1"].components == (bundle.components[1],)


def test_a_transport_failure_settles_nothing_under_component_scope(ledger):
    transport = component("Rule 2", False, failure_reason="transport_error")
    rows = [
        record("s1", 0, STATUS_TRANSPORT_FAILED, block="b0", retry_scope="component",
               components=(component("full_ruleset"), component("Rule 1"), transport)),
        crecord(0, {"full_ruleset": True, "Rule 1": True, "Rule 2": True}, block="b0-again"),
    ]
    ledger.append(rows, [])
    state = ledger.load()
    assert state.is_accepted("s1") and state.accepted_bundle("s1").semantic_attempt_id == 0
    assert {rec.request_block_id for rec, _ in state.accepted_components("s1").values()} == {"b0-again"}


def test_the_bundle_scope_reads_its_accepted_row_whole(ledger):
    ledger.append([record("s1", 0, STATUS_REJECTED, block="b0"), record("s1", 1, block="b1")], [])
    state = ledger.load()
    assert state.retry_scope == "scenario_bundle" and state.bundles == {} and state.settled == {}
    assert state.accepted_bundle("s1") is state.accepted["s1"]
    assert list(state.accepted_components("s1")) == ["full_ruleset"]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([crecord(0, {"full_ruleset": True, "Rule 1": False}), crecord(1, {"full_ruleset": True, "Rule 1": True})],
         r"carries components \['full_ruleset', 'Rule 1'\] for scenario s1; its unsettled components are \['Rule 1'\]"),
        ([crecord(0, {"full_ruleset": False, "Rule 1": False}), crecord(1, {"Rule 1": False, "full_ruleset": True})],
         "its unsettled components are"),
        ([record("s1", 0, STATUS_REJECTED, components=(), retry_scope="component")], r"carries components \[\]"),
        ([crecord(0, {"full_ruleset": True, "Rule 1": True}, STATUS_REJECTED)], "accepted exactly when it settles the last"),
        ([record("s1", 0, STATUS_REJECTED, block="b0"), crecord(1, {"full_ruleset": True})],
         "recorded under retry scope 'component'; the ledger's rows are 'scenario_bundle'"),
    ],
)
def test_the_component_scope_invariants(ledger, rows, message):
    ledger.append(rows, [])
    with pytest.raises(LedgerError, match=message):
        ledger.load()


def _published_copy(tmp_path, recorded: DrawLedger) -> DrawLedger:
    """The draw of ``recorded`` as a published copy of its root holds it: the root's record, the draw's published form."""
    from ruleforge.core.constants import PUBLISHED_COPY_FNAME
    from ruleforge.evaluate.ledger import PUBLISHED_LEDGER_FNAME, published_form_text

    root = tmp_path / "published"
    (root / "k" / "draw-000").mkdir(parents=True)
    (root / PUBLISHED_COPY_FNAME).write_text(json.dumps({"ledger_conditions": ["k"]}))
    (root / "k" / "draw-000" / PUBLISHED_LEDGER_FNAME).write_bytes(published_form_text(recorded.directory, **IDENTITY))
    return DrawLedger(root / "k" / "draw-000", **IDENTITY)


def _read_values(state, scenario_ids):
    """What the rebuilds read of a draw's accepted attempts: per accepted scenario, the accepted attempt and its
    full-ruleset call (the generating attempt and block, judgment, rule judgments, raw digest)."""
    out = {}
    for sid in scenario_ids:
        if not state.is_accepted(sid):
            out[sid] = None
            continue
        bundle = state.accepted_bundle(sid)
        [full] = [c for c in bundle.components if c.component_id == "full_ruleset"]
        call, component = state.accepted_components(sid)["full_ruleset"]
        out[sid] = (bundle.semantic_attempt_id, call.semantic_attempt_id, call.request_block_id, full.judgment,
                    full.rule_judgments, full.raw_output_sha256, component == full)
    return out


def test_the_published_form_holds_what_the_rebuilds_read_of_the_strict_replay(ledger, tmp_path):
    from ruleforge.evaluate.ledger import attempt_stamps, has_attempts, named_stacks, published_form

    rows = [record("s1", 0, STATUS_REJECTED, block="b0", started_at="2026-10-01T00:00:00Z", judged_on="aa" * 32),
            record("s2", 0, block="b0", started_at="2026-10-01T00:00:01Z", components=(component(
                rule_judgments={"Rule 1": "yes"}, preempted_rules=("Rule 1",)),)),
            record("s1", 1, block="b1", started_at="2026-10-02T00:00:00Z", judged_on="bb" * 32),
            record("s3", 0, STATUS_REJECTED, block="b1", started_at="2026-10-02T00:00:01Z")]
    ledger.append(rows, [])
    form = published_form(ledger.directory, **IDENTITY)
    assert form["accepted"] == [
        {"scenario_id": "s2", "semantic_attempt_id": 0, "full_ruleset": {
            "semantic_attempt_id": 0, "request_block_id": "b0", "judgment": "yes", "rule_judgments": {"Rule 1": "yes"},
            "raw_output_sha256": raw_output_sha256("yes")}},
        {"scenario_id": "s1", "semantic_attempt_id": 1, "full_ruleset": {
            "semantic_attempt_id": 1, "request_block_id": "b1", "judgment": "yes", "rule_judgments": None,
            "raw_output_sha256": raw_output_sha256("yes")}}]
    assert (form["started_at"], form["judged_on"], form["has_attempts"]) == (
        {"first": "2026-10-01T00:00:00Z", "last": "2026-10-02T00:00:01Z"}, {"aa" * 32: 1, "bb" * 32: 1}, True)
    assert "retry_scope" not in form  # the published call is the settled one: no reader needs the scope
    published = _published_copy(tmp_path, ledger)
    state = published.accepted()
    assert _read_values(state, ["s1", "s2", "s3"]) == _read_values(ledger.load(), ["s1", "s2", "s3"])
    assert ledger.accepted() == ledger.load()
    assert has_attempts(published.directory) is has_attempts(ledger.directory) is True
    assert named_stacks(published.directory) == named_stacks(ledger.directory)
    recorded = sorted(attempt_stamps(ledger.directory))
    assert attempt_stamps(published.directory) == [recorded[0], recorded[-1]] and len(recorded) == 4


def test_a_component_scope_draw_publishes_the_call_that_settled_its_full_ruleset(ledger, tmp_path):
    ledger.append([crecord(0, {"full_ruleset": True, "Rule 1": False})], [])
    ledger.append([crecord(1, {"Rule 1": True})], [])
    published = _published_copy(tmp_path, ledger)
    values = _read_values(published.accepted(), ["s1"])
    assert values == _read_values(ledger.load(), ["s1"])
    assert values["s1"][:3] == (1, 0, "b0")
    with pytest.raises(LedgerError, match="not the ledger's retry_scope"):
        published.accepted().retry_scope


def test_a_published_form_refuses_what_it_does_not_hold(ledger, tmp_path):
    from ruleforge.evaluate.ledger import PUBLISHED_LEDGER_FNAME

    ledger.append([record("s1")], [])
    published = _published_copy(tmp_path, ledger)
    state = published.accepted()
    for name in ("records", "attempts_consumed", "settled"):
        with pytest.raises(LedgerError, match=rf"published form of a draw's attempt ledger, .*: not the ledger's {name}"):
            getattr(state, name)
    with pytest.raises(LedgerError, match="not the ledger's next_attempt"):
        state.next_attempt("s1")
    with pytest.raises(LedgerError, match="which holds what its root's rebuilds read: not a full-ruleset call's "
                                          "preempted_rules"):
        state.accepted_bundle("s1").components[0].preempted_rules
    for repair in (False, True):
        with pytest.raises(LedgerError, match=r"in a published copy of a campaign root, which holds a draw's ledger as "
                                              r"its accepted attempts alone \(accepted_attempts.json\)"):
            published.load(repair=repair)
    other = DrawLedger(published.directory, campaign_id="c", condition_id="k", decoder_draw_id=1)
    with pytest.raises(LedgerError, match=r"belongs to \('c', 'k', 0\), not \('c', 'k', 1\)"):
        other.accepted()
    path = published.directory / PUBLISHED_LEDGER_FNAME
    for text, message in (("{", "not a readable published form"), ('{"schema_version": 9}', "not a published form of "
                                                                                              "schema 1")):
        path.write_text(text)
        with pytest.raises(LedgerError, match=message):
            published.accepted()


def test_one_decision_reads_a_draw_as_recorded_or_published_and_refuses_every_other_combination(ledger, tmp_path):
    from ruleforge.core.constants import PUBLISHED_COPY_FNAME
    from ruleforge.evaluate.ledger import PUBLISHED_LEDGER_FNAME, attempt_stamps, draw_mode, has_attempts, named_stacks

    ledger.append([record("s1")], [])
    published = _published_copy(tmp_path, ledger)
    root, form = published.directory.parent.parent, published.directory / PUBLISHED_LEDGER_FNAME
    marker = root / PUBLISHED_COPY_FNAME
    assert draw_mode(published.directory).published and not draw_mode(ledger.directory).published
    readers = (lambda draw: draw.accepted(), lambda draw: draw.load(), lambda draw: has_attempts(draw.directory),
               lambda draw: named_stacks(draw.directory), lambda draw: attempt_stamps(draw.directory))
    # a marked root's draw holding a recorded ledger: both forms in one draw, or a recorded ledger alone
    (published.directory / LEDGER_FNAME).write_bytes(ledger.ledger_path.read_bytes())
    for remove in (False, True):
        if remove:
            form.rename(tmp_path / "form")
        for read in readers:
            with pytest.raises(LedgerError, match=r"holds a recorded ledger \(attempts.jsonl\) in a published copy"):
                read(published)
    (published.directory / LEDGER_FNAME).unlink()
    # a marked root's draw with neither form, of a condition whose ledgers the copy holds: never empty history
    for read in readers:
        with pytest.raises(LedgerError, match="lacks its ledger's published form .*, which its published copy holds for "
                                              "every draw of k"):
            read(published)
    # one of a condition whose ledgers the copy leaves out: no attempts to read, and no accepted attempts to give
    marker.write_text(json.dumps({"ledger_conditions": []}))
    assert (has_attempts(published.directory), named_stacks(published.directory),
            attempt_stamps(published.directory)) == (False, {}, [])
    with pytest.raises(LedgerError, match="leaves this condition's attempt ledgers out"):
        published.accepted()
    with pytest.raises(LedgerError, match="which holds a draw's ledger as its accepted attempts alone"):
        published.load()
    # an unmarked root's draw holding a published form
    (tmp_path / "form").rename(form)
    marker.unlink()
    for read in readers:
        with pytest.raises(LedgerError, match=r"holds a published ledger form \(accepted_attempts.json\) but lies in no "
                                              "published copy"):
            read(published)
    for text in ("{", "[]"):
        marker.write_text(text)
        with pytest.raises(LedgerError, match="not a (readable )?published-copy record"):
            published.accepted()
    # A recorded root reads as before, its draws with no ledger as nothing.
    empty = DrawLedger(tmp_path / "none" / "draw-000", **IDENTITY)
    assert empty.accepted() == empty.load() and not has_attempts(empty.directory)
    assert named_stacks(empty.directory) == {} and attempt_stamps(empty.directory) == []


def test_a_form_holds_only_the_parts_its_root_reads_and_refuses_the_others_by_name(ledger, tmp_path):
    from ruleforge.core.constants import PUBLISHED_LEDGER_FNAME
    from ruleforge.evaluate.ledger import attempt_stamps, has_attempts, named_stacks, published_form

    ledger.append([record("s1", 0, STATUS_REJECTED, block="b0"), record("s1", 1, block="b1")], [])
    published = _published_copy(tmp_path, ledger)
    form = published_form(ledger.directory, **IDENTITY, parts=("call", "judgment"))
    assert form == {"schema_version": 1, **IDENTITY, "accepted": [
        {"scenario_id": "s1", "full_ruleset": {"semantic_attempt_id": 1, "request_block_id": "b1", "judgment": "yes"}}]}
    (published.directory / PUBLISHED_LEDGER_FNAME).write_text(json.dumps(form))
    call, component = published.accepted().accepted_components("s1")["full_ruleset"]
    assert (call.semantic_attempt_id, call.request_block_id, component.judgment) == (1, "b1", "yes")
    for read, what in ((lambda: component.rule_judgments, "a full-ruleset call's rule_judgments"),
                       (lambda: published.accepted().accepted_bundle("s1").semantic_attempt_id,
                        "an accepted bundle's semantic_attempt_id"),
                       (lambda: has_attempts(published.directory), "the draw's has_attempts"),
                       (lambda: named_stacks(published.directory), "the draw's judged_on"),
                       (lambda: attempt_stamps(published.directory), "the draw's started_at")):
        with pytest.raises(LedgerError, match=f"which holds what its root's rebuilds read: not {what}"):
            read()
    bare = published_form(ledger.directory, **IDENTITY, parts=("has_attempts",))
    (published.directory / PUBLISHED_LEDGER_FNAME).write_text(json.dumps(bare))
    # a form without accepted scenarios holds no identity either: only a reader of them checks it
    assert bare == {"schema_version": 1, "has_attempts": True} and has_attempts(published.directory)
    with pytest.raises(LedgerError, match="not its accepted scenarios"):
        published.accepted().is_accepted("s1")
    # the accepted attempt alone: each row its scenario and attempt, no call
    assert published_form(ledger.directory, **IDENTITY, parts=("accepted_attempt",)) == {
        "schema_version": 1, **IDENTITY, "accepted": [{"scenario_id": "s1", "semantic_attempt_id": 1}]}
    with pytest.raises(LedgerError, match=r"\['rows'\] are not parts of a published ledger form"):
        published_form(ledger.directory, **IDENTITY, parts=("rows",))
    with pytest.raises(AttributeError):
        component._private


def test_an_accepted_bundle_without_a_full_ruleset_call_publishes_none(ledger, tmp_path):
    ledger.append([record("s1", components=(component("Rule 1"),))], [])
    state = _published_copy(tmp_path, ledger).accepted()
    assert state.is_accepted("s1") and state.accepted_bundle("s1").components == ()
    assert state.accepted_components("s1") == {} and "full_ruleset" not in ledger.load().accepted_components("s1")


def test_a_draw_whose_ledger_holds_no_complete_row_publishes_no_attempt(ledger, tmp_path):
    from ruleforge.evaluate.ledger import attempt_stamps, has_attempts, published_form

    ledger.directory.mkdir(parents=True)
    ledger.ledger_path.write_text("")
    form = published_form(ledger.directory, **IDENTITY)
    assert (form["has_attempts"], form["started_at"], form["accepted"]) == (False, None, [])
    published = _published_copy(tmp_path, ledger)
    assert not has_attempts(published.directory) and attempt_stamps(published.directory) == []


def test_a_draw_holding_its_ledger_and_its_matching_published_form_is_refused_by_every_reader(ledger, tmp_path):
    # the round-8 adversarial review's probe: a recorded ledger and its own published form in one draw, no record
    from ruleforge.core.constants import PUBLISHED_LEDGER_FNAME
    from ruleforge.evaluate.ledger import attempt_stamps, has_attempts, named_stacks, published_form_text

    ledger.append([record("s1"), record("s2")], [])
    (ledger.directory / PUBLISHED_LEDGER_FNAME).write_bytes(published_form_text(ledger.directory, **IDENTITY))
    for read in (ledger.load, ledger.accepted, lambda: has_attempts(ledger.directory),
                 lambda: named_stacks(ledger.directory), lambda: attempt_stamps(ledger.directory)):
        with pytest.raises(LedgerError, match="holds a published ledger form .* but lies in no published copy"):
            read()
