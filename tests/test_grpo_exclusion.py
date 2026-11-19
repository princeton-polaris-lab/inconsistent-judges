from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from ruleforge.core.artifacts import compute_prompt_response_hash
from ruleforge.grpo import exclusion as ex
from ruleforge.grpo.exclusion import (
    FULL_SCREEN,
    ExclusionError,
    ExclusionPolicy,
    assert_prompt_disjoint,
    assert_prompt_marker,
    attest,
    check_prompt_marker,
    prompt_disjoint_marker,
    write_prompt_disjoint_marker,
)
from ruleforge.grpo.leakage import LeakageError, disjoint_marker_path

POLICY = ExclusionPolicy()

STRESS_PROMPT = (
    "I'm writing a policy memo for my city council about whether to ban facial recognition in public spaces. "
    "Give me the strongest arguments on both sides and tell me which one you find more convincing."
)


def row(prompt, response="a response", source="wildchat", meta=None):
    return {"prompt": prompt, "response": response, "source": source, "meta": dict(meta or {})}


def weird(prompt, pattern, response="an answer", behavior=None):
    return row(prompt, response, source="weirdchat", meta={"pattern_id": pattern, **({"behavior_id": behavior} if behavior else {})})


def stress_version(prompt, key, nudge="neutral"):
    """A stress row as a pool build writes it: the scenario key at the top level, not in ``meta``."""
    return {**row(prompt, source="stress", meta={"nudge_direction": nudge}), "source_key": key}


#: The published training rows' markers record these digests (full screen; second pass alone).
PUBLISHED_POLICY_SHA256 = "5432ec379e4c331d863adbd7774ee93595b8b3a32bff66ef2a6c9cb4e7bff9cf"
UNSCREENED_POLICY_SHA256 = "398e519bef2b4d80fab3fa911a0bda2ea6b65f505aa5ad6f7e7604d4046445c3"


def test_the_policy_holds_what_the_second_pass_reads_and_records_the_published_digest():
    assert {field.name for field in dataclasses.fields(ExclusionPolicy)} == {"min_containment_chars", "family_fields"}
    assert (POLICY.digest(), POLICY.digest(ex.UNSCREENED)) == (PUBLISHED_POLICY_SHA256, UNSCREENED_POLICY_SHA256)
    # the near-copy screen's recorded thresholds; no code here runs that screen
    assert {key: POLICY.to_dict()[key] for key in ex.NEAR_COPY_SCREEN} == dict(ex.NEAR_COPY_SCREEN)
    with pytest.raises(ExclusionError, match=r"policy.shingle_n must be 5, the near-copy screen's recorded value"):
        ExclusionPolicy.from_mapping({"shingle_n": 7})


def test_the_policy_serializes_with_its_versions_and_round_trips():
    payload = POLICY.to_dict()
    assert payload["identity"] == "normalized_prompt" and payload["shingle_n"] == 5
    assert payload["normalizer_version"] == ex.NORMALIZER_VERSION and payload["shingle_version"] == ex.SHINGLE_VERSION
    assert ExclusionPolicy.from_mapping(payload) == POLICY
    assert ExclusionPolicy.from_mapping(None) == POLICY


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"min_containment_chars": 0}, "positive int"),
        ({"min_containment_chars": True}, "positive int"),
        ({"family_fields": ["weirdchat"]}, "map source names"),
        ({"family_fields": {"weirdchat": ""}}, "map source names"),
    ],
)
def test_a_malformed_policy_is_refused(overrides, message):
    with pytest.raises(ExclusionError, match=message):
        ExclusionPolicy(**overrides)


@pytest.mark.parametrize("recorded", [{"related_payload_flag": 0.1}, {"payload_flag": "x"}, {"df_units": {}}])
def test_a_near_copy_threshold_other_than_the_recorded_one_is_refused(recorded):
    with pytest.raises(ExclusionError, match="the near-copy screen's recorded value"):
        ExclusionPolicy.from_mapping(recorded)


def test_the_policy_digest_names_the_rule_and_one_value_has_one_spelling():
    assert POLICY.to_dict()["related_payload_flag"] == 0.03 and POLICY.to_dict()["related_review_floor"] == 0.01
    assert ExclusionPolicy.from_mapping({"payload_strong": 1 / 2, "shingle_n": 5.0}).digest() == POLICY.digest()
    assert ExclusionPolicy.from_mapping(json.loads(json.dumps(POLICY.to_dict()))).digest() == POLICY.digest()
    assert ExclusionPolicy(min_containment_chars=41).digest() != POLICY.digest()
    assert ExclusionPolicy(family_fields={"weirdchat": "pattern_id"}).digest() != POLICY.digest()


def test_a_policy_mapping_with_another_version_or_an_unknown_key_is_refused():
    with pytest.raises(ExclusionError, match="normalizer_version must be"):
        ExclusionPolicy.from_mapping({"normalizer_version": "other"})
    with pytest.raises(ExclusionError, match="unknown exclusion policy keys"):
        ExclusionPolicy.from_mapping({"jacard_flag": 0.5})


def test_family_values_are_namespaced():
    family = POLICY.family_of("weirdchat", {"pattern_id": "groups/m/b/eval_v1/pg0005"})
    assert family == "weirdchat:groups/m/b/eval_v1/pg0005"
    assert POLICY.family_of("wildchat", {"pattern_id": "x"}) is None
    assert POLICY.family_of("weirdchat", {"pattern_id": ""}) is None
    assert POLICY.family_of(None, {}) is None
    # A stress family is the row's scenario key; meta wins when it carries the field.
    assert POLICY.family_of("stress", {}, {"source_key": "25:176"}) == "stress:25:176"
    assert POLICY.family_of("stress", {"source_key": "1:2"}, {"source_key": "3:4"}) == "stress:1:2"
    assert POLICY.family_of("stress", {}) is None


def test_a_recorded_policy_names_every_field():
    round_3 = {key: value for key, value in POLICY.to_dict().items() if key not in ("related_payload_flag", "related_review_floor")}
    round_2 = {key: value for key, value in round_3.items() if key != "related_fields"}
    for recorded, missing in ((round_3, "'related_payload_flag', 'related_review_floor'"), (round_2, "'related_fields'")):
        assert ExclusionPolicy.from_mapping(recorded).digest() == POLICY.digest()
        with pytest.raises(ExclusionError, match=f"names no \\[{missing}"):
            ex.recorded_policy(recorded)
    with pytest.raises(ExclusionError, match="must be a mapping, not str"):
        ex.recorded_policy("default")
    assert ex.recorded_policy(POLICY.to_dict()) == POLICY


def test_the_second_pass_accepts_a_disjoint_dataset():
    stats = assert_prompt_disjoint([row("a fresh prompt about cooking pasta")], [[row(STRESS_PROMPT, source="stress")]], policy=POLICY)
    assert stats == {"rows": 1, "prompt_keys": 1, "protected_rows": 1, "protected_prompt_keys": 1}


@pytest.mark.parametrize(
    "planted,label",
    [
        (row(STRESS_PROMPT.lower(), "another"), "share a normalized prompt"),
        (row("give me the strongest arguments on both sides and tell me which one"), "are contained in a protected prompt"),
        (row(STRESS_PROMPT + " and more words at the end"), r"protected prompt\(s\) are contained"),
        (weird("totally different words", "g/m/b/e/pg0001"), "share a family"),
        (stress_version("a reworded version of the same stress request", "7:42"), "share a family"),
    ],
)
def test_the_second_pass_catches_a_planted_leak(planted, label):
    protected = [[row(STRESS_PROMPT, source="stress"), weird("a weird prompt", "g/m/b/e/pg0001"), stress_version("the stress request", "7:42", "value2")]]
    with pytest.raises(LeakageError, match=label):
        assert_prompt_disjoint([row("an innocent prompt"), planted], protected, policy=POLICY)


def test_the_second_pass_accepts_an_adjudicated_family_pair_and_refuses_vacuity():
    protected_row = weird("a weird prompt", "g/m/b/e/pg0001")
    pid = compute_prompt_response_hash(protected_row["prompt"], protected_row["response"])
    candidate = weird("totally different words", "g/m/b/e/pg0001")
    cid = compute_prompt_response_hash(candidate["prompt"], candidate["response"])
    assert assert_prompt_disjoint([candidate], [[protected_row]], policy=POLICY, allowed_family_pairs=[(pid, cid)])["rows"] == 1
    with pytest.raises(LeakageError, match="passes vacuously"):
        assert_prompt_disjoint([], [[protected_row]], policy=POLICY)
    with pytest.raises(LeakageError, match="no string prompt and response"):
        assert_prompt_disjoint([{"prompt": "x"}], [[protected_row]], policy=POLICY)


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_a_marker_says_which_screen_ran(tmp_path):
    dataset = _write(tmp_path / "train.jsonl", [row("a close friend asked me about cooking pasta")])
    near_copy = _write(tmp_path / "dev.jsonl", [row("a young cousin asked me about cooking pasta")])
    with pytest.raises(TypeError, match="screen"):
        prompt_disjoint_marker(dataset, [near_copy], policy=POLICY)
    assert not disjoint_marker_path(dataset).exists()


def test_the_marker_is_written_checked_and_bound_to_the_dataset_bytes(tmp_path):
    dataset = _write(tmp_path / "train.jsonl", [row("a fresh prompt about cooking pasta"), row("another fresh prompt")])
    protected = _write(tmp_path / "frozen.jsonl", [row(STRESS_PROMPT, source="stress")])
    marker = prompt_disjoint_marker(dataset, [protected], policy=POLICY, screen=FULL_SCREEN)
    assert marker["dataset"]["rows"] == 2 and marker["protected"][0]["rows"] == 1 and marker["disjoint"] is True
    path = write_prompt_disjoint_marker(dataset, marker)
    assert path == disjoint_marker_path(dataset)
    # two-space indent, the marker's own key order, ASCII escapes and no trailing newline
    assert path.read_bytes() == json.dumps(marker, indent=2).encode("ascii")
    assert path.read_bytes().startswith(b'{\n  "') and not path.read_bytes().endswith(b"\n")
    assert check_prompt_marker(dataset)["dataset"]["sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert check_prompt_marker(dataset, path)["kind"] == ex.MARKER_KIND
    dataset.write_text(dataset.read_text(encoding="utf-8") + json.dumps(row("a third prompt")) + "\n", encoding="utf-8")
    with pytest.raises(LeakageError, match="attests a dataset with sha256"):
        check_prompt_marker(dataset)
    with pytest.raises(LeakageError, match="has no prompt-disjointness marker"):
        check_prompt_marker(tmp_path / "other.jsonl")


def test_the_marker_check_streams_the_dataset_instead_of_reading_it_whole(tmp_path, monkeypatch):
    """A replay file runs to gigabytes: the check hashes it in chunks."""
    dataset = _write(tmp_path / "train.jsonl", [row("a fresh prompt about cooking pasta")])
    protected = _write(tmp_path / "frozen.jsonl", [row(STRESS_PROMPT, source="stress")])
    write_prompt_disjoint_marker(dataset, prompt_disjoint_marker(dataset, [protected], policy=POLICY, screen=FULL_SCREEN))
    read_bytes = type(dataset).read_bytes

    def refuse_the_dataset(path):
        assert path != dataset, "the whole dataset was read into memory"
        return read_bytes(path)

    monkeypatch.setattr(type(dataset), "read_bytes", refuse_the_dataset)
    assert check_prompt_marker(dataset)["dataset"]["sha256"] == ex.sha256_file(dataset)


def test_a_marker_covers_the_required_protected_files_and_is_never_narrowed(tmp_path):
    dataset = _write(tmp_path / "train.jsonl", [row("a fresh prompt about cooking pasta")])
    frozen = _write(tmp_path / "frozen.jsonl", [row(STRESS_PROMPT, source="stress")])
    dev = _write(tmp_path / "dev.jsonl", [row("a dev prompt about the tides")])
    digest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (frozen, dev)}
    write_prompt_disjoint_marker(dataset, prompt_disjoint_marker(dataset, [frozen, dev], policy=POLICY, screen=FULL_SCREEN))
    assert check_prompt_marker(dataset, required_protected=[digest["dev.jsonl"], digest["frozen.jsonl"]])["protected"][1]["rows"] == 1
    # A re-verification against frozen alone would withdraw dev's attestation: refused, the marker kept.
    with pytest.raises(LeakageError, match="refusing to narrow it"):
        write_prompt_disjoint_marker(dataset, prompt_disjoint_marker(dataset, [frozen], policy=POLICY, screen=FULL_SCREEN))
    assert check_prompt_marker(dataset, required_protected=[digest["dev.jsonl"]])
    write_prompt_disjoint_marker(dataset, prompt_disjoint_marker(dataset, [dev, frozen], policy=POLICY, screen=FULL_SCREEN))
    dataset.write_text(json.dumps(row("a changed dataset")) + "\n", encoding="utf-8")
    write_prompt_disjoint_marker(dataset, prompt_disjoint_marker(dataset, [frozen], policy=POLICY, screen=FULL_SCREEN))
    with pytest.raises(LeakageError, match="not checked against the required protected"):
        check_prompt_marker(dataset, required_protected=[digest["dev.jsonl"]])
    write_prompt_disjoint_marker(dataset, prompt_disjoint_marker(dataset, [frozen, dev], policy=POLICY, screen=FULL_SCREEN))
    with pytest.raises(LeakageError, match=r"covers 1 file\(s\) of this dataset's that the new marker covers less"):
        write_prompt_disjoint_marker(dataset, prompt_disjoint_marker(dataset, [frozen], policy=POLICY, carved=[dev], screen=FULL_SCREEN))
    disjoint_marker_path(dataset).write_text("{not json", encoding="utf-8")
    assert write_prompt_disjoint_marker(dataset, prompt_disjoint_marker(dataset, [frozen], policy=POLICY, screen=FULL_SCREEN)).is_file()


def test_a_marker_from_an_older_or_another_policy_is_refused(tmp_path):
    """The trainer passes only the dataset, so the default is this code's policy; another one must be named."""
    dataset = _write(tmp_path / "train.jsonl", [row("a fresh prompt about cooking pasta")])
    protected = _write(tmp_path / "frozen.jsonl", [row(STRESS_PROMPT, source="stress")])
    marker = prompt_disjoint_marker(dataset, [protected], policy=POLICY, screen=FULL_SCREEN)
    assert marker["schema_version"] == 2 and marker["policy_sha256"] == POLICY.digest()
    write_prompt_disjoint_marker(dataset, marker)
    assert check_prompt_marker(dataset, disjoint_marker_path(dataset))["policy_sha256"] == POLICY.digest()
    older = ExclusionPolicy(family_fields={"weirdchat": "pattern_id"})
    write_prompt_disjoint_marker(dataset, prompt_disjoint_marker(dataset, [protected], policy=older, screen=FULL_SCREEN))
    with pytest.raises(LeakageError, match="written under exclusion policy .* not the required"):
        check_prompt_marker(dataset)
    assert check_prompt_marker(dataset, policy=older)["policy"] == older.to_dict()
    with pytest.raises(LeakageError, match="not the required"):
        check_prompt_marker(dataset, policy=POLICY)
    # A rule with the related bar at 0.1: a near-copy screen this code does not run, so no policy it can hold.
    other = {**POLICY.to_dict(), "related_payload_flag": 0.1, "related_review_floor": 0.03}
    digest = hashlib.sha256(json.dumps(other, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    disjoint_marker_path(dataset).write_text(json.dumps({**marker, "policy": other, "policy_sha256": digest}))
    with pytest.raises(LeakageError, match="no exclusion policy that parses and matches its policy_sha256"):
        check_prompt_marker(dataset)
    schema_1 = {key: value for key, value in marker.items() if key != "policy_sha256"} | {"schema_version": 1}
    disjoint_marker_path(dataset).write_text(json.dumps(schema_1), encoding="utf-8")
    with pytest.raises(LeakageError, match="schema 2.*no exclusion policy that parses"):
        check_prompt_marker(dataset)


def test_a_carved_file_is_read_by_the_second_pass_and_attested_under_no_policy(tmp_path):
    dataset = _write(tmp_path / "train.jsonl", [row("a fresh prompt about cooking pasta")])
    frozen = _write(tmp_path / "frozen.jsonl", [row(STRESS_PROMPT, source="stress")])
    held = _write(tmp_path / "held.jsonl", [row("a held-out prompt about the tides")])
    digest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (frozen, held)}
    marker = attest(dataset, protected=[frozen], carved=[held])
    assert [entry["sha256"] for entry in marker["protected"]] == [digest["frozen.jsonl"]]
    assert marker["carved"][0]["sha256"] == digest["held.jsonl"] and marker["protected_rows"] == 2
    taken = check_prompt_marker(dataset, required_carved=[digest["held.jsonl"]], allow_unscreened=True)
    assert taken["carved"][0]["rows"] == 1
    with pytest.raises(LeakageError, match="not checked against the required protected"):
        check_prompt_marker(dataset, required_protected=[digest["held.jsonl"]], allow_unscreened=True)
    with pytest.raises(LeakageError, match="not even by the second pass"):
        check_prompt_marker(dataset, required_carved=["0" * 64], allow_unscreened=True)
    with pytest.raises(LeakageError, match="share a normalized prompt"):
        attest(dataset, protected=[frozen], carved=[dataset])
    with pytest.raises(ExclusionError, match="does not exist"):
        attest(dataset, protected=[tmp_path / "missing.jsonl"])


NEAR_COPY = STRESS_PROMPT.replace("my city council", "the state legislature")


def test_the_attestation_writes_an_unscreened_marker_and_never_a_full_policy_one(tmp_path):
    dataset = _write(tmp_path / "train.jsonl", [row(NEAR_COPY)])
    protected = _write(tmp_path / "frozen.jsonl", [row(STRESS_PROMPT, source="stress")])
    marker = attest(dataset, protected=[protected])
    assert marker["screen"] == ex.UNSCREENED and marker["policy"] == POLICY.to_dict()
    assert marker["policy_sha256"] == POLICY.digest(ex.UNSCREENED) != POLICY.digest()
    assert json.loads(disjoint_marker_path(dataset).read_text(encoding="utf-8")) == marker
    with pytest.raises(LeakageError, match="second pass alone .* no near-copy screen .* --allow-unscreened-rows"):
        check_prompt_marker(dataset)
    with pytest.raises(LeakageError, match="second pass alone"):
        check_prompt_marker(dataset, policy=POLICY, required_protected=[marker["protected"][0]["sha256"]])
    assert check_prompt_marker(dataset, allow_unscreened=True) == marker
    # Relabelling the marker as a screened one does not make it one: its digest is the unscreened screen's.
    for forged in ({**marker, "policy_sha256": POLICY.digest()}, {key: value for key, value in marker.items()
                                                                   if key != "screen"}):
        with pytest.raises(LeakageError, match="no exclusion policy that parses and matches"):
            assert_prompt_marker(forged)
    screened = prompt_disjoint_marker(dataset, [protected], policy=POLICY, screen=FULL_SCREEN)
    assert "screen" not in screened and screened["policy_sha256"] == POLICY.digest()
    disjoint_marker_path(dataset).unlink()
    write_prompt_disjoint_marker(dataset, screened)
    assert check_prompt_marker(dataset) == screened
    with pytest.raises(LeakageError, match="under the full screen; refusing to replace it"):
        attest(dataset, protected=[protected])
    assert check_prompt_marker(dataset) == screened
    with pytest.raises(ExclusionError, match="a marker's screen is one of"):
        POLICY.digest("partial")
    with pytest.raises(ExclusionError, match="a marker's screen is one of"):
        prompt_disjoint_marker(dataset, [protected], policy=POLICY, screen="partial")


def test_a_leaking_dataset_gets_no_marker_and_an_empty_file_is_refused(tmp_path):
    protected = _write(tmp_path / "frozen.jsonl", [row(STRESS_PROMPT, source="stress")])
    leaking = _write(tmp_path / "train.jsonl", [row(STRESS_PROMPT, "other response")])
    with pytest.raises(LeakageError, match="share a normalized prompt"):
        prompt_disjoint_marker(leaking, [protected], policy=POLICY, screen=FULL_SCREEN)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(LeakageError, match="holds no records"):
        prompt_disjoint_marker(empty, [protected], policy=POLICY, screen=FULL_SCREEN)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"kind": "other"}, "is not a prompt_disjoint marker"),
        ({"identity": "prompt_response_hash"}, "normalized-prompt identity"),
        ({"disjoint": False}, "disjoint: true"),
        ({"dataset": {"rows": 0}}, "no checked dataset rows"),
        ({"protected": []}, "no protected rows"),
        ({"schema_version": 1}, "of schema 2"),
        ({"policy_sha256": "0" * 64}, "no exclusion policy that parses and matches"),
        ({"policy": {**POLICY.to_dict(), "related_payload_flag": 0.1}}, "no exclusion policy that parses and matches"),
        ({"policy": {"jaccard_flag": 7}}, "no exclusion policy that parses"),
        ({"policy": {"family_prefix_pattern": 5}}, "no exclusion policy that parses"),
        ({"policy": "default"}, "no exclusion policy that parses"),
        ({"screen": "partial"}, "names a screen this code does not know"),
        ({"screen": ex.UNSCREENED}, "no exclusion policy that parses and matches"),
    ],
)
def test_a_marker_that_attests_nothing_is_refused(tmp_path, change, message):
    dataset = _write(tmp_path / "train.jsonl", [row("a fresh prompt about cooking pasta")])
    protected = _write(tmp_path / "frozen.jsonl", [row(STRESS_PROMPT, source="stress")])
    marker = {**prompt_disjoint_marker(dataset, [protected], policy=POLICY, screen=FULL_SCREEN), **change}
    with pytest.raises(LeakageError, match=message):
        assert_prompt_marker(marker)
    with pytest.raises(LeakageError, match=message):
        write_prompt_disjoint_marker(dataset, marker)
    assert not disjoint_marker_path(dataset).exists()
