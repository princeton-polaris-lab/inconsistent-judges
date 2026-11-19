from __future__ import annotations

import json
import shutil
from pathlib import Path

from ruleforge.cli import records
from ruleforge.evaluate import api
from ruleforge.scenarios import ScenarioSet


def test_a_created_root_takes_its_conditions_from_campaign_add_and_is_run_and_analysed(cli, inputs):
    scenarios, rules = inputs
    status, created, _ = cli("campaign", "create", "runs/r", "--scenarios", scenarios, "--campaign-id", "r",
                             "--draws", "2", "--attempts", "3", "--retry-scope", "component")
    root = Path.cwd() / "runs" / "r"
    assert status == 0 and created == {"root": str(root), "campaign_id": "r", "scenarios": 3, "draws": 2,
                                       "campaign_mode": "pilot"}
    opened = api.Campaign.open(root)
    assert opened.campaign_id == "r" and opened.config.num_decoder_draws == 2
    assert opened.config.max_semantic_attempts == 3 and opened.config.label_rule == "lenient_last_line"
    [record] = list(records.runs_dir(root).iterdir())
    assert record.name.endswith("-campaign-create")
    assert json.loads((record / records.ARGV_FNAME).read_text())["subcommand"] == "campaign create"
    status, added, _ = cli("campaign", "add", root, "--scenarios", scenarios, "--rules", rules, "--judge", "fake")
    assert status == 0
    status, ran, _ = cli("campaign", "run", root, added["condition_id"], "--judge", "fake", "--rules", rules,
                         "--scenarios", scenarios)
    assert status == 0 and ran["draws"] == {"0": "complete", "1": "complete"}
    status, analysed, _ = cli("campaign", "analyze", root, "--n-resamples", "100")
    assert status == 0 and list(analysed["conditions"]) == [added["condition_id"]]


def test_a_root_created_again_is_rebound_and_one_of_another_sampling_refused(cli, inputs):
    scenarios, _ = inputs
    status, first, _ = cli("campaign", "create", "runs/r", "--scenarios", scenarios, "--campaign-id", "r")
    assert status == 0
    status, again, _ = cli("campaign", "create", "runs/r", "--scenarios", scenarios, "--campaign-id", "r")
    assert status == 0 and again == first
    assert len(list(records.runs_dir(Path(first["root"])).iterdir())) == 2
    status, _, error = cli("campaign", "create", "runs/r", "--scenarios", scenarios, "--campaign-id", "r",
                           "--draws", "3")
    assert status == 2 and "ruleforge campaign: CampaignError" in error


def test_the_default_campaign_id_is_the_digest_of_population_and_sampling(cli, inputs):
    scenarios, _ = inputs
    status, created, _ = cli("campaign", "create", "runs/d", "--scenarios", scenarios, "--draws", "2")
    settings = api.SamplingSettings(draws=2)
    expected = api.default_campaign_id(ScenarioSet.load(scenarios).manifest, settings.config())
    assert status == 0 and created["campaign_id"] == expected and expected.startswith("eval-")


def test_a_final_campaign_of_a_population_of_another_size_is_refused_with_nothing_made(cli, inputs):
    status, _, error = cli("campaign", "create", "runs/f", "--scenarios", inputs[0], "--campaign-id", "f", "--final")
    assert status == 2 and "500" in error
    assert not (Path.cwd() / "runs" / "f").exists() and not records.runs_dir(Path.cwd() / "runs" / "f").exists()


def test_a_root_whose_run_records_are_a_link_is_refused_before_anything_is_made(cli, inputs, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (Path.cwd() / "runs").mkdir()
    records.runs_dir(Path.cwd() / "runs" / "r").symlink_to(elsewhere)
    status, _, error = cli("campaign", "create", "runs/r", "--scenarios", inputs[0], "--campaign-id", "r")
    assert status == 2 and "symlink" in error
    assert not (Path.cwd() / "runs" / "r").exists() and not list(elsewhere.iterdir())


def test_a_root_or_population_spelling_two_places_or_naming_nothing_is_refused_before_anything_is_made(cli, inputs,
                                                                                                      tmp_path):
    scenarios, _ = inputs
    work, other = Path.cwd(), tmp_path / "other"
    (other / "child").mkdir(parents=True)
    for directory in (work, other):
        shutil.copyfile(scenarios, directory / scenarios.name)
    (work / "link").symlink_to(other / "child")
    for argv, message in [(["runs/r", "--scenarios", f"link/../{scenarios.name}"], "names two places"),
                          (["runs/r", "--scenarios", f"missing/../{scenarios.name}"], "is no file there"),
                          (["runs/r", "--scenarios", scenarios.parent], "is no file there"),
                          (["link/../r", "--scenarios", scenarios], "link/../r names two places")]:
        status, _, error = cli("campaign", "create", *argv, "--campaign-id", "r")
        assert status == 2 and "campaign create: " in error and message in error, (argv, error)
    assert not (work / "runs").exists() and not (work / "r").exists() and not (other / "r").exists()
    assert not (work / "missing").exists() and not records.runs_dir(work / "r").exists()
