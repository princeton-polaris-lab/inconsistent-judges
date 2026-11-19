from __future__ import annotations

import json

import pytest

from ruleforge.experiments import campaign_roots as cr
from ruleforge.experiments.prompt_ablation import arms as ta
from ruleforge.experiments.prompt_ablation.analysis import TABLE_CONTRASTS
from ruleforge.experiments.labels import MODEL_DISPLAY_LABELS, NAIVE_SLOT
from ruleforge.prompts.styles import ENCOURAGED, FORCED
from ruleforge.evaluate.estimates import public_amendment
from ruleforge.evaluate.identity import MANIFEST_AMENDMENTS_KEY
from tests import test_prompt_ablation_arms as arm_tests
from tests.test_campaign_estimates import disclosure, record
from tests.test_leniency_campaign import artifact_root
from tests.test_prompt_ablation_arms import (
    GEMMA,
    LABELS,
    LUNA,
    OLMO,
    NEW,
    NEW_SHORT,
    OSS,
    ROOTS,
    _build,
    _rebuild,
    arm_map,
    build_study_root,
    write_map,
)

HAIKU = "claude-haiku-5-5"
PAPER_MAP = "configs/paper/prompt_ablation_arms.yaml"
#: The styles Haiku's root records: its three arms' and the revised naive one, which no slot names.
HAIKU_STYLES = ("naive-rules-above", "naive", "encouraged", "forced")
HAIKU_ROOT = {"haiku": ("prompt-ablation-v2-2026-10-haiku-pop-6f528dcb",
                        [(HAIKU, HAIKU_STYLES, "anthropic_batch", None)])}
HAIKU_DESIGN = {"label_rule": "lenient_last_line", "retry_scope": "component"}
GLIMMER, GLIMMER_LABEL = "Muse-Glimmer-30B", "muse-glimmer-30b"
GLIMMER_STYLES = ("naive-rules-above", "encouraged", "forced")
GLIMMER_ROOT = {"glimmer": ("prompt-ablation-v2-2026-10-glimmer-pop-6f528dcb",
                            [(GLIMMER, GLIMMER_STYLES, "local", NEW)])}


study, driver = arm_tests.study, arm_tests.driver


def _templated():
    payload = arm_map()
    payload["roots"] = {key: uri.replace("-rescored-", "-${tag}-") for key, uri in payload["roots"].items()}
    payload["variables"] = ["tag"]
    return payload


def test_an_arm_map_names_its_roots_by_variable_and_the_golden_records_the_values(study, tmp_path):
    base, _ = _rebuild(study, tmp_path)
    path = write_map(base / "arms.yaml", _templated())
    with artifact_root(base):
        payload = ta.build_payload(path, variables={"tag": "rescored"})
    assert payload["table"] == study["payload"]["table"] and payload["bars"] == study["payload"]["bars"]
    record_ = payload["provenance"]["arm_map"]
    assert record_["variables"] == {"tag": "rescored"} and record_["name"] == "arms.yaml"
    assert [c["root"] for c in payload["provenance"]["campaigns"]] == [
        c["root"] for c in study["payload"]["provenance"]["campaigns"]]
    # A map without variables records none (the committed goldens' provenance keeps its bytes).
    assert "variables" not in study["payload"]["provenance"]["arm_map"]
    assert ta.load_arm_map(path, {"tag": "rescored"}).variables == {"tag": "rescored"}


@pytest.mark.parametrize(("edit", "given", "message"), [
    (None, {}, r"no value for \['tag'\] \(--arm-map-var NAME=VALUE\)"),
    (None, {"tag": "rescored", "pop": "x"}, r"values given for \['pop'\], which the map does not declare"),
    (None, {"tag": "../rescored"}, r"variable tag's value '../rescored' is not a root-name fragment"),
    (None, {"tag": ""}, r"variable tag's value '' is not a root-name fragment"),
    (lambda p: p.update(variables=["tag", "Pop"]), {"tag": "x", "Pop": "y"}, r"variable 'Pop' is not a name"),
    (lambda p: p.update(variables=["tag", "tag"]), {"tag": "x"}, "a variable is declared twice"),
    (lambda p: p.update(variables=[]), {}, "variables is not a non-empty list of names"),
    (lambda p: p.update(variables="tag"), {"tag": "x"}, "variables is not a non-empty list of names"),
    (lambda p: p.update(variables=["tag", "spare"]), {"tag": "x", "spare": "y"}, r"variables \['spare'\] fill no root"),
    (lambda p: p["roots"].update(luna=p["roots"]["luna"] + "${run}"), {"tag": "x"},
     r"root placeholders \['run'\] name no declared variable"),
])
def test_a_variable_that_is_not_declared_given_and_used_once_each_is_refused(tmp_path, edit, given, message):
    payload = _templated()
    if edit is not None:
        edit(payload)
    with pytest.raises(ta.ArmMapError, match=message):
        ta.load_arm_map(write_map(tmp_path / "arms.yaml", payload), given)
    with pytest.raises(ta.ArmMapError, match=r"values given for \['tag'\]"):
        ta.load_arm_map(write_map(tmp_path / "plain.yaml", arm_map()), {"tag": "x"})


def test_name_value_words_parse_once_each():
    assert ta.parse_variables(["pop=pop-6f528dcb", "tag=a=b"]) == {"pop": "pop-6f528dcb", "tag": "a=b"}
    assert ta.parse_variables([]) == {}
    with pytest.raises(ta.ArmMapError, match="--arm-map-var: 'pop' is not NAME=VALUE"):
        ta.parse_variables(["pop"])
    with pytest.raises(ta.ArmMapError, match="--arm-map-var: variable 'pop' is given twice"):
        ta.parse_variables(["pop=a", "pop=b"])


def test_main_takes_the_variables_with_the_arm_map_only(study, driver, tmp_path, capsys):
    base, _ = _rebuild(study, tmp_path)
    path = write_map(base / "arms.yaml", _templated())
    golden = tmp_path / "t.json"
    with artifact_root(base):
        assert driver.main(["--arm-map", str(path), "--arm-map-var", "tag=rescored", "--golden", str(golden)]) == 0
        assert json.loads(golden.read_text())["provenance"]["arm_map"]["variables"] == {"tag": "rescored"}
        for argv, message in ((["--arm-map", str(path), "--arm-map-var", "tag"], "'tag' is not NAME=VALUE"),
                              (["--arm-map", str(path)], "no value for ['tag']"),
                              (["--from_golden", str(golden), "--arm-map-var", "tag=x"], "it needs --arm-map")):
            with pytest.raises(SystemExit):
                driver.main(argv)
            assert message in capsys.readouterr().err


def _ladder(base, key, cap=50):
    """The root ``key`` as one whose cap a recorded amendment raised from 20 (a retry ladder)."""
    root = base / "sampling" / ROOTS[key][0]
    manifest = json.loads((root / "sampling_manifest.json").read_text())
    (root / "sampling_manifest.json").write_text(json.dumps({**manifest, "max_semantic_attempts": cap,
                                                             MANIFEST_AMENDMENTS_KEY: [record(new_cap=cap)]}))
    estimates = json.loads((root / "aggregate" / "estimates.json").read_text())
    (root / "aggregate" / "estimates.json").write_text(json.dumps({
        **estimates, "max_semantic_attempts": cap, MANIFEST_AMENDMENTS_KEY: [disclosure()]}))


def test_a_judge_s_arm_from_a_root_that_took_the_ladder_pairs_with_one_that_did_not(study, tmp_path):
    base, path = _rebuild(study, tmp_path, change=lambda b: _ladder(b, "luna"))
    payload = _build(base, path)
    assert payload["table"] == study["payload"]["table"]
    campaigns = {c["root"]: c for c in payload["provenance"]["campaigns"]}
    laddered = campaigns[f"artifact://sampling/{ROOTS['luna'][0]}"]
    assert laddered["max_semantic_attempts"] == 50 and laddered[MANIFEST_AMENDMENTS_KEY] == [public_amendment(disclosure())]
    assert all(MANIFEST_AMENDMENTS_KEY not in c for uri, c in campaigns.items() if uri != laddered["root"])
    base, path = _rebuild(study, tmp_path / "other", change=lambda b: _ladder(b, "luna"))
    root = base / "sampling" / ROOTS["published_luna"][0]
    for name in ("sampling_manifest.json", "aggregate/estimates.json"):
        (root / name).write_text(json.dumps({**json.loads((root / name).read_text()), "max_semantic_attempts": 5}))
    with pytest.raises(cr.CampaignRootError, match=r"max_semantic_attempts \(published_luna 5, luna 20\)"):
        _build(base, path)


def _with_haiku(study, tmp_path, designs=None):
    base, _ = _rebuild(study, tmp_path)
    with artifact_root(base):
        build_study_root(base, "haiku", roots=HAIKU_ROOT,
                         designs={**{f"{HAIKU}/{style}": HAIKU_DESIGN for style in HAIKU_STYLES}, **(designs or {})})
    manifest = base / "sampling" / HAIKU_ROOT["haiku"][0] / "sampling_manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "retry_scope": "component"}))
    payload = arm_map()
    payload["roots"]["haiku"] = f"artifact://sampling/{HAIKU_ROOT['haiku'][0]}"
    payload["judges"][HAIKU] = {
        "naive": {"root": "haiku", "condition": f"{HAIKU}/naive-rules-above"},
        "encouraged": {"root": "haiku", "condition": f"{HAIKU}/encouraged"},
        "forced": {"root": "haiku", "condition": f"{HAIKU}/forced"},
    }
    return base, write_map(base / "arms.yaml", payload)


def test_haiku_is_the_fifth_judge_after_the_published_four_its_rule_and_scope_named(study, tmp_path):
    base, path = _with_haiku(study, tmp_path)
    payload = _build(base, path)
    order = [LABELS[m] for m in (OLMO, GEMMA, OSS, LUNA)] + [HAIKU]
    assert [(r["label"], r["baseline"], r["treatment"]) for r in payload["table"]] == [
        (label, a, b) for label in order for a, b in TABLE_CONTRASTS]
    # The published four's rows keep their numbers; every row's families are recomputed over the fifteen.
    published = {(r["label"], r["baseline"], r["treatment"]): r for r in study["payload"]["table"]}
    for row in payload["table"][:12]:
        before = published[(row["label"], row["baseline"], row["treatment"])]
        assert {k: row[k] for k in ("difference", "ci_low", "ci_high", "p_value")} == {
            k: before[k] for k in ("difference", "ci_low", "ci_high", "p_value")}
        assert row["adjustment_family_n"] == 15
    provenance = payload["provenance"]
    assert provenance["models"] == order and provenance["model_labels"][HAIKU] == HAIKU
    assert provenance["label_rules"][HAIKU] == "lenient_last_line"
    assert provenance["retry_scopes"] == {**{LABELS[m]: "scenario_bundle" for m in (OLMO, GEMMA, OSS, LUNA)},
                                          HAIKU: "component"}
    arms = provenance["arms"][HAIKU]
    assert {slot: arms[slot]["prompt_style"] for slot in (NAIVE_SLOT, ENCOURAGED, FORCED)} == {
        NAIVE_SLOT: "naive-rules-above", ENCOURAGED: "encouraged", FORCED: "forced"}
    assert arms[NAIVE_SLOT]["stack"]["kind"] == cr.PROVIDER and HAIKU not in provenance["cross_stack"]["declared"]
    haiku_root = f"artifact://sampling/{HAIKU_ROOT['haiku'][0]}"
    assert provenance["ignored_conditions"][haiku_root] == [f"{HAIKU}/naive"]
    assert provenance["ignored_conditions"][f"artifact://sampling/{ROOTS['gemma'][0]}"] == [f"{GEMMA}/naive"]
    assert {c["root"]: c["conditions"] for c in provenance["campaigns"]}[haiku_root] == [
        f"{HAIKU}/encouraged", f"{HAIKU}/forced", f"{HAIKU}/naive-rules-above"]  # public ids, sorted
    assert set(payload["bars"]) == set(order) and MODEL_DISPLAY_LABELS[HAIKU] == "Claude Haiku 5.5"
    assert ta.paper_table_refusal(payload) is None


def test_haiku_s_arms_under_two_label_rules_are_refused(study, tmp_path):
    base, path = _with_haiku(study, tmp_path, designs={
        f"{HAIKU}/encouraged": {**HAIKU_DESIGN, "label_rule": "lenient_markup"}})
    with pytest.raises(ta.ArmMapError, match="claude-haiku-5-5 .*label rule"):
        _build(base, path)


def test_the_paper_s_arm_map_is_the_study_s_plan_under_one_label_rule_with_haiku_and_glimmer(repo_root, tmp_path):
    study_map = ta.load_arm_map(write_map(tmp_path / "arms.yaml", arm_map()))
    paper = ta.load_arm_map(repo_root / PAPER_MAP, {"pop": "pop-a22af097"})
    assert paper.variables == {"pop": "pop-a22af097"}
    assert set(paper.roots) == {*ROOTS, "haiku", "glimmer"}
    for key, (name, _) in ROOTS.items():
        raw = name.removesuffix("-rescored-lenient_markup")
        assert paper.roots[key] == f"artifact://sampling/{raw}-pop-a22af097-rescored-lenient_last_line"
    assert paper.roots["haiku"] == "artifact://sampling/prompt-ablation-v2-2026-10-haiku-sum"
    assert paper.roots["glimmer"] == "artifact://sampling/prompt-ablation-v2-2026-10-glimmer-pop-a22af097"
    assert set(paper.judges) == {*study_map.judges, HAIKU, GLIMMER}
    assert {model: paper.judges[model] for model in study_map.judges} == study_map.judges
    haiku = paper.judges[HAIKU]
    assert haiku.arms == {NAIVE_SLOT: ta.Arm("haiku", f"{HAIKU}/naive-rules-above"), ENCOURAGED: ta.Arm("haiku", f"{HAIKU}/encouraged"),
                          FORCED: ta.Arm("haiku", f"{HAIKU}/forced")}
    assert not haiku.cross_stack
    glimmer = paper.judges[GLIMMER]
    assert glimmer.arms == {NAIVE_SLOT: ta.Arm("glimmer", f"{GLIMMER}/naive-rules-above"),
                            ENCOURAGED: ta.Arm("glimmer", f"{GLIMMER}/encouraged"),
                            FORCED: ta.Arm("glimmer", f"{GLIMMER}/forced")}
    assert not glimmer.cross_stack
    assert all(not arm.condition_id.endswith("/naive") for arm in paper.named) and len(paper.named) == 18
    assert list(ta.judges_by_label(paper)) == [*(LABELS[m] for m in (OLMO, GEMMA, OSS)), GLIMMER_LABEL, LABELS[LUNA],
                                            HAIKU]
    assert "naive arms are the published ones" in paper.cross_stack_reason
    assert all(f"{model}'s" in paper.cross_stack_reason for model in (OLMO, GEMMA, OSS))
    assert {model for model, judge in paper.judges.items() if judge.cross_stack} == {OLMO, GEMMA, OSS}


def test_glimmer_is_the_sixth_judge_with_the_open_judges_its_rule_scope_and_label_named(study, tmp_path):
    import yaml

    base, path = _with_haiku(study, tmp_path)
    with artifact_root(base):
        build_study_root(base, "glimmer", roots=GLIMMER_ROOT,
                         designs={f"{GLIMMER}/{style}": HAIKU_DESIGN for style in GLIMMER_STYLES})
    manifest = base / "sampling" / GLIMMER_ROOT["glimmer"][0] / "sampling_manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "retry_scope": "component"}))
    payload = yaml.safe_load(path.read_text())
    payload["roots"]["glimmer"] = f"artifact://sampling/{GLIMMER_ROOT['glimmer'][0]}"
    payload["judges"][GLIMMER] = {slot: {"root": "glimmer", "condition": f"{GLIMMER}/{style}"}
                                  for slot, style in zip(("naive", "encouraged", "forced"), GLIMMER_STYLES)}
    payload = _build(base, write_map(path, payload))
    order = [*(LABELS[m] for m in (OLMO, GEMMA, OSS)), GLIMMER_LABEL, LABELS[LUNA], HAIKU]
    assert [(r["label"], r["baseline"], r["treatment"]) for r in payload["table"]] == [
        (label, a, b) for label in order for a, b in TABLE_CONTRASTS]
    published = {(r["label"], r["baseline"], r["treatment"]): r for r in study["payload"]["table"]}
    for row in payload["table"]:
        assert row["adjustment_family_n"] == 18
        before = published.get((row["label"], row["baseline"], row["treatment"]))
        if before is not None:
            assert {k: row[k] for k in ("difference", "ci_low", "ci_high", "p_value")} == {
                k: before[k] for k in ("difference", "ci_low", "ci_high", "p_value")}
    glimmer_rows = [row for row in payload["table"] if row["label"] == GLIMMER_LABEL]
    assert all(row["cross_stack"] is None and row["paired_within_root"] for row in glimmer_rows)
    provenance = payload["provenance"]
    assert provenance["models"] == order and provenance["model_labels"][GLIMMER] == GLIMMER_LABEL
    assert provenance["label_rules"][GLIMMER_LABEL] == "lenient_last_line"
    assert provenance["retry_scopes"][GLIMMER_LABEL] == "component"
    arms = provenance["arms"][GLIMMER_LABEL]
    assert {slot: arms[slot]["prompt_style"] for slot in (NAIVE_SLOT, ENCOURAGED, FORCED)} == dict(
        zip((NAIVE_SLOT, ENCOURAGED, FORCED), GLIMMER_STYLES))
    assert {arm["stack"]["short"] for arm in arms.values()} == {NEW_SHORT}
    assert GLIMMER_LABEL not in provenance["cross_stack"]["declared"]
    glimmer_root = f"artifact://sampling/{GLIMMER_ROOT['glimmer'][0]}"
    assert glimmer_root not in provenance["ignored_conditions"]
    assert {c["root"]: c["conditions"] for c in provenance["campaigns"]}[glimmer_root] == [
        f"{GLIMMER}/encouraged", f"{GLIMMER}/forced", f"{GLIMMER}/naive-rules-above"]
    assert set(payload["bars"]) == set(order) and MODEL_DISPLAY_LABELS[GLIMMER_LABEL] == "Muse-Glimmer-30B"
    assert ta.paper_table_refusal(payload) is None
