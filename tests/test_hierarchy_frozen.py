from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ruleforge.experiments import table
from ruleforge.core.digests import canonical_json
from ruleforge.core.paths import resolve
from ruleforge.evaluate.identity import LABEL_RULE
from ruleforge.evaluate.inference import holm_adjust
from ruleforge.experiments.hierarchy import analysis as ha
from ruleforge.experiments.hierarchy import figure as hf
from ruleforge.experiments.hierarchy import build as build_hierarchy

pytestmark = pytest.mark.frozen

GOLDEN = "results/hierarchy/hierarchy.json"
PROGRAM = "ruleforge.experiments.hierarchy.build"
PAIRS = "configs/paper/hierarchy_pairs.yaml"
FIGURES = ("figures/hierarchy.pdf", "figures/hierarchy.png")
#: The judges in paper order.
JUDGES = ("Olmo3-32B", "gemma4-31b", "gpt-oss-120b", "muse-glimmer-30b", "gpt-5.6-luna", "claude-haiku-5-5")
PRIMARY = {
    "Olmo3-32B": (
        (0.7556, 0.726, 0.784, 0.014766081390809142),
        (0.7448, 0.7188, 0.7704099999999999, 0.01323495642691732),
        (-0.010800000000000032, -0.03959999999999986, 0.018400000000000083, 0.01490395910912265, 0.4864651353486465,
         0.4864651353486465, False)),
    "gemma4-31b": (
        (0.7212000000000001, 0.6856, 0.7556, 0.017804196843845554),
        (0.622, 0.58479, 0.658, 0.018666332898306513),
        (-0.09920000000000007, -0.12439999999999996, -0.0739900000000001, 0.01277906715318454, 9.99990000099999e-06,
         5.999940000599994e-05, True)),
    "gpt-oss-120b": (
        (0.626, 0.5892000000000001, 0.6607999999999999, 0.018143757548600564),
        (0.5775999999999999, 0.5408, 0.6112000000000001, 0.018168571751417336),
        (-0.04840000000000011, -0.06879999999999997, -0.028800000000000048, 0.01018300520757993, 9.99990000099999e-06,
         5.999940000599994e-05, True)),
    "muse-glimmer-30b": (
        (0.44239999999999996, 0.4076, 0.47879999999999995, 0.018106622098933857),
        (0.6, 0.5624, 0.6376, 0.019165067317846812),
        (0.15760000000000002, 0.13319999999999999, 0.18280000000000007, 0.01274735074593933, 9.99990000099999e-06,
         5.999940000599994e-05, True)),
    "gpt-5.6-luna": (
        (0.436, 0.3976, 0.4736, 0.019373423445534867),
        (0.4088, 0.3728, 0.4448000000000001, 0.018486056279260865),
        (-0.027200000000000002, -0.050000000000000044, -0.0036000000000000476, 0.011903130935749637,
         0.022309776902230977, 0.044619553804461955, True)),
    "claude-haiku-5-5": (
        (0.5856, 0.5475999999999999, 0.624, 0.019758246147874563),
        (0.6167999999999999, 0.5788, 0.654, 0.01923118157060559),
        (0.031199999999999894, 0.010000000000000009, 0.05279999999999996, 0.010917373653328899, 0.004919950800491995,
         0.014759852401475985, True)),
}
#: Per judge, the sha256 of its descriptive block (``secondary``) as canonical JSON.
SECONDARY_SHA256 = {
    "Olmo3-32B": "10e1fb183aeb080d3c783995e84354875331436a91a40919312ef7e01c2bece0",
    "gemma4-31b": "b25e61469515ea08f6ce8300bdd0aa1ce536de9b9985c990ab01869ea63d1fad",
    "gpt-oss-120b": "c586ab2bb6d39478b7ace94d5935c1168ccd47537f161ae46ffb2d9c0dd55393",
    "muse-glimmer-30b": "9a99178133af3575d454fa5cbeb3eedda6940f725872766c64e9e07568c14170",
    "gpt-5.6-luna": "a673de43415b5720e2157c6f0a2b5abf819dd9a326276991a7a2c098d737165a",
    "claude-haiku-5-5": "14c2c007b3ac0a423fda1dbbc0af1b0c0d841c26a66034de306c619ef14aa32e",
}
OPEN_ROOT = "hierarchy-aware-2026-10-pop-a22af097"
ARMS = {
    "Olmo3-32B": [
        (OPEN_ROOT, "Olmo-3-32B-Think/naive-rules-above", "5a94234f9e8b0c2923a4e15e2588b17ca10b7fe1610d0d6378f308ea558de207",
         "003791f08bdccb9b273e556c6460f63923258b028cfe395a3bdc7ec22c15bdce", "lenient_markup", "local"),
        (OPEN_ROOT, "Olmo-3-32B-Think/naive-hierarchy",
         "7f6d2420d895460cb558f7493769253368c9b2516d814d90bf870eb1462d2bc9",
         "5d67b9d95db5ab297d6daa15a9ec95786ad4d736c7dc0d4d87c5d543fe05e8f7", "lenient_markup", "local")],
    "gemma4-31b": [
        (OPEN_ROOT, "gemma-4-31B-it/naive-rules-above", "17646ddc22e27d2c9698c6db0f35c328d7f25c55f06df6d0b156e67a83a9dbd5",
         "24033196adad081be802098ede3404f4d12b84964c0c3de274c295b8e34f3328", "lenient_markup", "local"),
        (OPEN_ROOT, "gemma-4-31B-it/naive-hierarchy",
         "133406f222433317e4a4ce012b7abd50c2ac75e22cb99bf4f6275bff3e0c8b76",
         "ded9c4056b6d42c5d79f9b12d94f6a33c8b1449f899d810663a663802c33594f", "lenient_markup", "local")],
    "gpt-oss-120b": [
        (OPEN_ROOT, "gpt-oss-120b/naive-rules-above", "5cdb1d75ef852ef840d4157cf8ad9475856e2cf9c92d96927f6b795a8e6e9a7e",
         "81308fd79f4a8e7200eb568db67e701a5c72adcfcc58035c48c0f58854d91ca6", "lenient_markup", "local"),
        (OPEN_ROOT, "gpt-oss-120b/naive-hierarchy", "5c43a079569d05f9b2c2c67a10ad130ba47ebb2c85494db94eda87daff1cd2e0",
         "de13c48407a484dfd6aab0e69fd174b1b4dd689b42414ea5e5e7350a62d393c0", "lenient_markup", "local")],
    "muse-glimmer-30b": [
        ("prompt-ablation-v2-2026-10-glimmer-pop-a22af097", "Muse-Glimmer-30B/naive-rules-above",
         "409dcd61f25450aca7aaf07739e873251fa7a6a2d1df2167477434ec8d164559",
         "73332927c93a3a3d076f5ba43dd11c85604ec9614743f2ae2e0a49793a22bcb0", "lenient_last_line", "local"),
        ("hierarchy-aware-2026-10-glimmer", "Muse-Glimmer-30B/naive-hierarchy",
         "ac7dc6980f5e721a3a7d8654f8c1d3ff9a33c9918b15f13aa417eca3b3e94c2a",
         "415b1b66195f7b9aae7a60cdf4b2930f1c1a05fc08d4eaad49fc654f17dee8a3", "lenient_last_line", "local")],
    "gpt-5.6-luna": [
        ("final-2026-09-luna-pop-a22af097-rescored-lenient_last_line", "gpt-5.6-luna/naive-rules-above",
         "0fcaa2ed99d95253a0bcd2cb2ba92af7ab344e780db2317c33cb89d7509fb0ae",
         "d27a2badfe387de889dab985ece516930b1af3a3426d2f4fe52b47666311c17c", "lenient_last_line", "provider"),
        ("hierarchy-aware-2026-10-luna-sum", "gpt-5.6-luna/naive-hierarchy",
         "8a3576aa23aa0bd7f72aa347b3a34feb87811e7569236d0fd439c721cdfb916c",
         "9a9d14b65ee1b65d8c1a9a0667c16bb8204d7d5f2ffa856ed4aba185acf0927d", "lenient_last_line", "provider")],
    "claude-haiku-5-5": [
        ("prompt-ablation-v2-2026-10-haiku-sum", "claude-haiku-5-5/naive-rules-above",
         "806fc5b850b61bed1a5d4d5dafa07400a970b82118ade8450e1bbba397b4392a",
         "f0dadf17ac4f10dbb449fdcae4ccee56f75c28594caef8e518d389541e891e95", "lenient_last_line", "provider"),
        ("hierarchy-aware-2026-10-haiku-sum", "claude-haiku-5-5/naive-hierarchy",
         "cf3206a9b45c049e46bc2601d0ce73f96c18948490e3a637685d8b6dcc2bfaab",
         "816c2daac6cef9568e6b309c5c791b3514c8fe4a27159bb7d5ba8e644f9e613b", "lenient_last_line", "provider")],
}
ROOTS = tuple(f"artifact://sampling/{name}" for name in (
    "hierarchy-aware-2026-10-pop-a22af097", "hierarchy-aware-2026-10-glimmer", "hierarchy-aware-2026-10-haiku-sum",
    "hierarchy-aware-2026-10-luna-sum",
    "prompt-ablation-v2-2026-10-glimmer-pop-a22af097", "prompt-ablation-v2-2026-10-haiku-sum",
    "final-2026-09-luna-pop-a22af097-rescored-lenient_last_line"))
#: Each joined campaign's retry scope as its estimates record it (None: the bundle scope, which they name not).
RETRY_SCOPES = {
    "hierarchy-aware-2026-10": None, "prompt-ablation-v2-2026-10-glimmer": "component",
    "hierarchy-aware-2026-10-glimmer": "component",
    "prompt-ablation-v2-2026-10-haiku-sum": "component", "hierarchy-aware-2026-10-haiku-sum": "component",
    "final-2026-09-luna": None, "hierarchy-aware-2026-10-luna-sum": None,
}
@pytest.fixture(scope="module")
def payload(repo_root):
    return json.loads((repo_root / GOLDEN).read_text(encoding="utf-8"))


def test_the_golden_is_a_hierarchy_payload_of_six_judges_in_paper_order(payload):
    assert build_hierarchy.load_golden(Path(__file__).resolve().parents[1] / GOLDEN) == payload
    assert (payload["analysis"], payload["schema_version"]) == ("hierarchy", ha.SCHEMA_VERSION) == (
        "hierarchy", 2)
    assert (payload["definition"], payload["pairing"], payload["multiplicity"], payload["secondary"]) == (
        ha.DEFINITION, ha.PAIRING, ha.MULTIPLICITY, ha.SECONDARY)
    assert tuple(record["judge"] for record in payload["judges"]) == JUDGES
    for record in payload["judges"]:
        assert (record["n_scenarios"], record["n_decoder_draws"]) == (500, 5)
        assert record["primary"]["difference"]["family_size"] == len(JUDGES)
        assert record["primary"]["difference_of"] == "hierarchy - flat"


@pytest.mark.parametrize("judge", JUDGES)
def test_each_judge_s_rates_and_paired_difference(payload, judge):
    record = next(record for record in payload["judges"] if record["judge"] == judge)
    primary = record["primary"]
    flat, hierarchy, difference = PRIMARY[judge]
    rate = ("estimate", "ci_low", "ci_high", "sd")
    assert tuple(primary["flat_rate"][key] for key in rate) == flat
    assert tuple(primary["hierarchy_rate"][key] for key in rate) == hierarchy
    assert tuple(primary["difference"][key] for key in (*rate, "p_value", "p_holm", "significant_holm")) == difference
    assert (primary["difference"]["n_permutations"], primary["difference"]["test_method"],
            primary["difference"]["inner_resample"]) == (100_000, ha.TEST_METHOD, "independent")
    assert (primary["difference"]["grid_digest_a"], primary["difference"]["grid_digest_b"]) == (
        record["hierarchy"]["grid_digest"], record["flat"]["grid_digest"])
    assert hashlib.sha256(canonical_json(record["secondary"]).encode("utf-8")).hexdigest() == SECONDARY_SHA256[judge]


def test_holm_over_the_six_differences_gives_the_golden_s_adjusted_p_values(payload):
    rows = holm_adjust([{"p_value": record["primary"]["difference"]["p_value"]} for record in payload["judges"]],
                       alpha=0.05)
    assert [(row["p_holm"], row["significant_holm"]) for row in rows] == [
        (record["primary"]["difference"]["p_holm"], record["primary"]["difference"]["significant_holm"])
        for record in payload["judges"]]
    assert [record["judge"] for record in payload["judges"]
            if not record["primary"]["difference"]["significant_holm"]] == ["Olmo3-32B"]


@pytest.mark.parametrize("judge", JUDGES)
def test_each_judge_s_arms_by_public_id_identity_and_label_reader(payload, judge):
    record = next(record for record in payload["judges"] if record["judge"] == judge)
    for side, (root, condition, fingerprint, grid, label_rule, stack) in zip(("flat", "hierarchy"), ARMS[judge]):
        arm = record[side]
        assert (arm["root"], arm["condition_id"], arm["condition_fingerprint"], arm["grid_digest"], arm["label_rule"],
                arm["stack"]["kind"]) == (f"artifact://sampling/{root}", condition, fingerprint, grid, label_rule,
                                          stack)
        assert (arm["prompt_style"], arm["individual_rule_context"]) == ha.SIDES[side]
        assert arm["rule_levels_sha256"] == (None if side == "flat" else payload["provenance"]["levels"]["sha256"])
    assert record["flat"]["label_rule"] == record["hierarchy"]["label_rule"]
    assert (record["flat"]["label_rule"] == LABEL_RULE) == (record["flat"]["root"] != f"artifact://sampling/{OPEN_ROOT}")


def test_the_caption_notes_name_each_judge_s_flat_control_as_its_records_make_it(payload):
    judges = hf.drawn_judges(payload)
    assert {record["judge"]: hf.flat_control(record) for record in judges} == {
        "Olmo3-32B": "rerun", "gemma4-31b": "rerun", "gpt-oss-120b": "rerun", "muse-glimmer-30b": "published",
        "gpt-5.6-luna": "provider", "claude-haiku-5-5": "provider"}
    assert hf.caption_notes(payload).endswith("\n\n" + ". ".join([
        "Flat arm re-run beside the hierarchy arm, on its inference stack: Olmo-3-32B, Gemma-4-31B, GPT-OSS-120B",
        "Flat arm the judge's published naive arm, on the hierarchy arm's inference stack: Muse-Glimmer-30B",
        "Flat arm the API judge's published naive arm, at the provider (a provider change since is not observable): "
        "GPT-5.6 Luna, Claude Haiku 5.5"]) + ".\n")
    assert "†" not in hf.caption_notes(payload)


def test_the_provenance_names_the_pairs_the_roots_and_the_settings(payload, repo_root):
    provenance = payload["provenance"]
    assert provenance["pairs"] == [{"flat": list(pair.flat), "hierarchy": list(pair.hierarchy)}
                                   for pair in build_hierarchy.load_pairs(repo_root / PAIRS)]
    assert provenance["levels"] == {"name": "openai_rules_all_full.jsonl",
                                    "sha256": "99eac16d5fc4c8cd028b091b6cab23badcd1573bd33d9765f0ca436e91f7130a"}
    assert provenance["ruleset"] == {"name": "openai_rules_all.jsonl",
                                     "sha256": "a99062f00a9ff616cba4d548e9cd59460dc4ee52fc46e3de4a4eb2579a260228"}
    assert provenance["approved_prompts_sha256"] == ha.APPROVED_PROMPT_SHA256 == {
        "approved_full_set_hier.txt": "e42c8e6069b02f2aab7885ad8bce3a3b772d8c2554e66a06a00551e920cb1afe",
        "approved_rule28_root_hier.txt": "3497da55e9c69501a61cf5e2a7b805a2eeb12a6b1667eacf8d29f5e5b965d99c",
        "approved_rule43_user_hier.txt": "de1f51c90170b7293cf2d4c7ad9dba9184342002dad2a4bb1102b47024b45f33"}
    assert provenance["bootstrap"] == {"seed": 16, "n_resamples": 10_000, "n_permutations": 100_000, "ci_level": 0.95}
    assert (provenance["n_scenarios"], provenance["n_decoder_draws"], provenance["scenario_manifest_sha256"]) == (
        500, 5, "a22af097639cfdaa15b4508c66631af1cb6d45780d8ada93a9fb1d5a29390462")
    assert (provenance["inference_method"], provenance["interval_basis"], provenance["test_method"],
            provenance["allow_incomplete"]) == (ha.INFERENCE_METHOD, ha.PRIMARY_INTERVAL_BASIS, ha.TEST_METHOD, False)
    campaigns = provenance["campaigns"]
    assert [campaign["root"] for campaign in campaigns] == list(ROOTS)[:1] + [
        "artifact://sampling/prompt-ablation-v2-2026-10-glimmer-pop-a22af097",
        "artifact://sampling/hierarchy-aware-2026-10-glimmer", "artifact://sampling/prompt-ablation-v2-2026-10-haiku-sum",
        "artifact://sampling/hierarchy-aware-2026-10-haiku-sum",
        "artifact://sampling/final-2026-09-luna-pop-a22af097-rescored-lenient_last_line",
        "artifact://sampling/hierarchy-aware-2026-10-luna-sum"]
    assert {campaign["campaign_id"]: campaign.get("retry_scope") for campaign in campaigns} == RETRY_SCOPES
    assert {campaign["max_semantic_attempts"] for campaign in campaigns} == {20}
    assert {campaign["state"] for campaign in campaigns} == {"complete"}
    assert [campaign["root"] for campaign in campaigns if campaign["label_replay"]] == [
        "artifact://sampling/final-2026-09-luna-pop-a22af097-rescored-lenient_last_line"]
    assert all("root_digest" not in campaign and "amendments" not in campaign for campaign in campaigns)
    assert "scenario_exclusion" not in provenance


def test_the_table_builds_and_draws_it_from_the_seven_roots():
    (row,) = [row for row in table.TABLE if row.name == "hierarchy"]
    assert row.runs == ROOTS
    (step,) = [step for step in row.builds if step.id == "hierarchy"]
    assert step.words[2:] == [PROGRAM, "--pairs", f"repo://{PAIRS}", "--golden", f"{{out}}/{GOLDEN}"]
    assert step.golden.path == GOLDEN and step.golden.rerun_tolerance == 0.005
    drawn = sorted(word for plot in row.plots if f"repo://{GOLDEN}" in plot.words for word in plot.words
                   if word.startswith("{out}/"))
    assert drawn == [f"{{out}}/{Path(path).name}" for path in FIGURES]


TICKS = ("Olmo-3-32B", "Gemma-4-31B*", "GPT-OSS-120B*", "Muse-Glimmer-30B*", "GPT-5.6 Luna*", "Claude Haiku 5.5*")


def test_the_figure_draws_the_golden_s_rates_and_holm_calls(payload):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    figure = hf.build_figure(payload)
    try:
        [axes] = figure.axes
        ticks = [tick.get_text() for tick in axes.get_xticklabels()]
        assert ticks == list(TICKS)
        assert [tick.endswith("*") for tick in ticks] == [PRIMARY[judge][2][-1] for judge in JUDGES]
        assert sorted(text.get_text() for text in axes.texts) == sorted(
            f"{PRIMARY[judge][side][0]:.3f}" for judge in JUDGES for side in (0, 1))
        assert not axes.artists and {line.get_marker() for line in axes.lines} == {"_"}
        assert not [text for text in figure.findobj(Text) if "†" in text.get_text()]
        legend = axes.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == [
            "flat", "hierarchy", "Holm-significant (hierarchy − flat)"]
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        rows = {round(box.y0 + box.y1) for box in (text.get_window_extent(renderer) for text in legend.get_texts())}
        assert len(rows) == 1
    finally:
        plt.close(figure)


def test_the_caption_notes_mark_the_golden_s_holm_calls_as_the_ticks_do(payload):
    notes = hf.caption_notes(payload)
    assert ("An asterisk after a judge's name marks a paired difference hierarchy - flat that is Holm-significant "
            "(alpha 0.05, Holm over the 6 judges drawn); a name without one, a difference that is not.") in notes
    assert [line.split(": ")[0] for line in notes.splitlines() if line.startswith("- ")] == [
        f"- {tick}" for tick in TICKS]


def test_the_figures_render_from_the_golden(repo_root, tmp_path, monkeypatch, capsys):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    out = [tmp_path / Path(path).name for path in FIGURES]
    assert build_hierarchy.main(["--from_golden", str(repo_root / GOLDEN),
                             *(word for path in out for word in ("--figure", str(path)))]) == 0
    capsys.readouterr()
    assert all(path.stat().st_size > 0 for path in out)


@pytest.mark.cluster
def test_the_golden_rebuilds_byte_for_byte_from_the_paper_root(repo_root, tmp_path):
    if not os.environ.get("RULEFORGE_ARTIFACT_ROOT"):
        pytest.skip("no artifact root is set")
    if not all((resolve(root) / "aggregate" / "estimates.json").is_file() for root in ROOTS):
        pytest.skip("the hierarchy appendix's roots are not in this artifact root")
    out = tmp_path / "hierarchy.json"
    built = subprocess.run([sys.executable, "-m", PROGRAM, "--pairs", PAIRS, "--golden", str(out)], cwd=repo_root,
                           env={**os.environ, "SOURCE_DATE_EPOCH": "0", "OMP_NUM_THREADS": "4",
                                "OPENBLAS_NUM_THREADS": "4"}, capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr[-2000:]
    assert out.read_bytes() == (repo_root / GOLDEN).read_bytes()
