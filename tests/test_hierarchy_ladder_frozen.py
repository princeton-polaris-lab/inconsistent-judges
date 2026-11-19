from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from ruleforge.core.digests import canonical_json
from ruleforge.core.paths import resolve
from ruleforge.experiments import table
from ruleforge.experiments.hierarchy import build_ladder
from ruleforge.experiments.hierarchy import ladder as hl
from ruleforge.rulesets.levels import LEVEL_ORDER

pytestmark = pytest.mark.frozen

PROGRAM = "ruleforge.experiments.hierarchy.build_ladder"
GOLDEN = "results/hierarchy/ladder.json"
HIERARCHY = "results/hierarchy/hierarchy.json"
PAIRS = "configs/paper/hierarchy_pairs.yaml"
#: The hierarchy comparison's judges in paper order, and each one's two arms.
JUDGES = ("Olmo3-32B", "gemma4-31b", "gpt-oss-120b", "muse-glimmer-30b", "gpt-5.6-luna", "claude-haiku-5-5")
SIDES = ("flat", "hierarchy")
#: Each arm's rate at every rung, to four places.
RUNG_RATES = {
    ("Olmo3-32B", "flat"): (0.4148, 0.67, 0.6732, 0.7212, 0.7556),
    ("Olmo3-32B", "hierarchy"): (0.442, 0.676, 0.6796, 0.7256, 0.7448),
    ("gemma4-31b", "flat"): (0.2552, 0.2888, 0.2912, 0.6116, 0.7212),
    ("gemma4-31b", "hierarchy"): (0.2324, 0.2612, 0.2652, 0.59, 0.622),
    ("gpt-oss-120b", "flat"): (0.178, 0.4416, 0.4556, 0.5512, 0.626),
    ("gpt-oss-120b", "hierarchy"): (0.2744, 0.3492, 0.3572, 0.4952, 0.5776),
    ("muse-glimmer-30b", "flat"): (0.254, 0.4264, 0.4264, 0.4384, 0.4424),
    ("muse-glimmer-30b", "hierarchy"): (0.3996, 0.5848, 0.5856, 0.5972, 0.6),
    ("gpt-5.6-luna", "flat"): (0.1612, 0.3068, 0.3116, 0.3728, 0.436),
    ("gpt-5.6-luna", "hierarchy"): (0.2084, 0.2868, 0.2952, 0.3792, 0.4088),
    ("claude-haiku-5-5", "flat"): (0.2852, 0.5144, 0.5168, 0.5528, 0.5856),
    ("claude-haiku-5-5", "hierarchy"): (0.2528, 0.538, 0.5392, 0.5872, 0.6168),
}
#: Per (judge, arm), the sha256 of the arm's record as canonical JSON: every rung's interval, the unsure misses, the
#: over-flag and its identity.
ARM_SHA256 = {
    ("Olmo3-32B", "flat"): "9eb92cfa02b83f7ae7700435141d9b62422eff819919dfac5c39e63d72e6ca85",
    ("Olmo3-32B", "hierarchy"): "0aa2062cefc7e729fe3713d652cb190c91db2987ff0469d55193d5028ba82eef",
    ("gemma4-31b", "flat"): "5d60081bc8413e33de75d67955a485351f57e7c1807a591ecdd9e957b505cec3",
    ("gemma4-31b", "hierarchy"): "3a13a5af0fb6d710d590e3f45e316b25ce6876ecc3c908a5c025e3678bef8b6a",
    ("gpt-oss-120b", "flat"): "07f41cc89d0e892ce0969713a966e0c1f476f586e80e035c711e0d91820c9714",
    ("gpt-oss-120b", "hierarchy"): "bfec8446f6ddd2244395a5bc304f519444c2a211ac616b8e245a83995549ac7d",
    ("muse-glimmer-30b", "flat"): "9b924301d7b36e16a1adf26ccd49fd750ae17b61037ad0fb8e4ea4db0e3a5121",
    ("muse-glimmer-30b", "hierarchy"): "7d222e4277464acbc3ab3bff418af2f103865a891beaa8cf9825a5927a55bbf3",
    ("gpt-5.6-luna", "flat"): "759b06a2cece7b6ccf22f69b0300e07aad67f8c4d1a9e2b3011067c72b947ca5",
    ("gpt-5.6-luna", "hierarchy"): "577bfffa8e686924700be96097538103057a1aae840091b1267bf1b2540e1c33",
    ("claude-haiku-5-5", "flat"): "f51f4a7794eee17b83ac7e8796f55b5bdc76aeee039969339f12b31721d44112",
    ("claude-haiku-5-5", "hierarchy"): "7507cad3b69ba45d6eed5d0945aff73aeb9c1eb50bc4b334b376025e82329953",
}
#: One row of the summary's rate table, as rendered.
SUMMARY_ROW = ("| gemma4-31b | hierarchy | 0.2324 [0.2016, 0.2640] | 0.2612 [0.2296, 0.2928] | "
               "0.2652 [0.2336, 0.2968] | 0.5900 [0.5536, 0.6268] | 0.6220 [0.5848, 0.6580] | 0.0004 [0.0000, 0.0012] |")


@pytest.fixture(scope="module")
def payload(repo_root):
    return build_ladder.load_golden(repo_root / GOLDEN)


def test_the_golden_is_a_ladder_of_twelve_arms_with_their_pinned_records(payload):
    assert (payload["analysis"], payload["schema_version"], payload["definition"]) == (
        "hierarchy_ladder", hl.SCHEMA_VERSION, hl.DEFINITION)
    assert [(arm["judge"], arm["arm"]) for arm in payload["arms"]] == [(judge, arm) for judge in JUDGES for arm in SIDES]
    assert [rung["levels"] for rung in payload["rungs"]] == [list(LEVEL_ORDER[:k]) for k in hl.RUNGS]
    assert payload["provenance"]["bootstrap"] == {"seed": 16, "n_resamples": 10_000, "ci_level": 0.95}
    assert payload["provenance"]["allow_incomplete"] is False
    assert {arm["arm"]: arm["individual_rule_context"] for arm in payload["arms"]} == {
        "flat": "isolated", "hierarchy": "superiors"}
    for arm in payload["arms"]:
        key = (arm["judge"], arm["arm"])
        assert (arm["n_scenarios"], arm["n_decoder_draws"]) == (500, 5)
        estimates = [rung["rate"]["estimate"] for rung in arm["rungs"]]
        assert estimates == sorted(estimates) and tuple(round(rate, 4) for rate in estimates) == RUNG_RATES[key]
        assert hashlib.sha256(canonical_json(arm).encode("utf-8")).hexdigest() == ARM_SHA256[key], key


def test_each_arm_s_last_rung_is_the_hierarchy_golden_s_rate_and_interval(payload, repo_root):
    hierarchy = json.loads((repo_root / HIERARCHY).read_text(encoding="utf-8"))
    records = {record["judge"]: record for record in hierarchy["judges"]}
    for arm in payload["arms"]:
        record = records[arm["judge"]]
        assert arm["rungs"][-1]["rate"] == record["primary"][f"{arm['arm']}_rate"]
        assert tuple(arm[key] for key in ("root", "condition_id", "grid_digest", "prompt_style",
                                          "individual_rule_context", "label_rule")) == tuple(
            record[arm["arm"]][key] for key in ("root", "condition_id", "grid_digest", "prompt_style",
                                                "individual_rule_context", "label_rule"))
    provenance, comparison = payload["provenance"], hierarchy["provenance"]
    assert provenance["pairs"] == comparison["pairs"]
    for key in ("ruleset", "levels", "approved_prompts_sha256", "campaigns", "scenario_manifest_sha256"):
        assert provenance[key] == comparison[key], key


def test_the_csv_and_summary_render_from_the_golden(repo_root, tmp_path, capsys):
    out = tmp_path / "ladder.csv"
    assert build_ladder.main(["--from_golden", str(repo_root / GOLDEN), "--csv", str(out)]) == 0
    assert SUMMARY_ROW in capsys.readouterr().out.splitlines()
    assert len(out.read_text().splitlines()) == 1 + len(JUDGES) * len(SIDES) * len(hl.RUNGS)


def test_the_hierarchy_experiment_builds_it_and_draws_the_hierarchy_arms_with_a_csv_of_both():
    (row,) = [row for row in table.TABLE if row.name == "hierarchy"]
    (step,) = [step for step in row.builds if step.id == "hierarchy_ladder"]
    assert step.words[2:] == [PROGRAM, "--pairs", f"repo://{PAIRS}", "--golden", f"{{out}}/{GOLDEN}"]
    assert (step.golden.name, step.golden.path, step.golden.rerun_tolerance) == ("hierarchy_ladder", GOLDEN, 0.005)
    plots = [plot for plot in row.plots if f"repo://{GOLDEN}" in plot.words]
    assert all(plot.words[plot.words.index("--arm") + 1] == "hierarchy" for plot in plots)
    assert sorted(word for plot in plots for word in plot.words if word.startswith("{out}/")) == [
        "{out}/hierarchy_ladder.csv", "{out}/hierarchy_ladder.pdf", "{out}/hierarchy_ladder.png"]


@pytest.mark.cluster
def test_the_golden_rebuilds_byte_for_byte_from_the_paper_roots(repo_root, tmp_path):
    if not os.environ.get("RULEFORGE_ARTIFACT_ROOT"):
        pytest.skip("no artifact root is set")
    (row,) = [row for row in table.TABLE if row.name == "hierarchy"]
    if not all((resolve(root) / "aggregate" / "estimates.json").is_file() for root in row.runs):
        pytest.skip("the hierarchy roots are not in this artifact root")
    out = tmp_path / "ladder.json"
    built = subprocess.run([sys.executable, "-m", PROGRAM, "--pairs", PAIRS, "--golden", str(out)], cwd=repo_root,
                           env={**os.environ, "SOURCE_DATE_EPOCH": "0", "OMP_NUM_THREADS": "4",
                                "OPENBLAS_NUM_THREADS": "4"}, capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr[-2000:]
    assert out.read_bytes() == (repo_root / GOLDEN).read_bytes()
