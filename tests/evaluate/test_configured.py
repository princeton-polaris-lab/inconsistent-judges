from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ruleforge.cli import main
from ruleforge.evaluate import api, configured
from ruleforge.evaluate.configured import MARKER_FNAME, ConfiguredRootError
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.shard_merge import merge_condition_shards
from ruleforge.evaluate.sharding import shards_root
from ruleforge.scenarios.manifest import ScenarioManifest
from tests.test_sampling_runner import write_inputs
from tests.test_sampling_shards import CONDITION, run_shards


@pytest.fixture(autouse=True)
def _roots(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    (tmp_path / "inputs").mkdir()


def _records(root: Path, tmp_path: Path, campaign_id: str = "camp") -> ScenarioManifest:
    manifest = ScenarioManifest.build(write_inputs(tmp_path / "inputs"))
    manifest.write(root / "scenario_manifest.json")
    (root / SAMPLING_MANIFEST_FNAME).write_text(json.dumps({"schema_version": 2, "campaign_id": campaign_id}))
    return manifest


def test_the_library_marker_a_configured_root_is_told_apart_by_is_the_library_s_own():
    # configured keeps its own copy of the name (the api module imports configured), so the two are held equal here
    assert configured.LIBRARY_MARKER_FNAME == api.MARKER_FNAME


def test_the_writer_that_makes_a_root_marks_it_and_no_later_writer_does(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    population = ScenarioManifest.build(write_inputs(tmp_path / "inputs")).manifest_sha256
    assert configured.mark(root, campaign_id="camp", scenario_manifest_sha256=population)
    payload = json.loads((root / MARKER_FNAME).read_text())
    assert payload == {"kind": configured.MARKER_KIND, "schema_version": 1, "campaign_id": "camp",
                       "scenario_manifest_sha256": population, "package_version": payload["package_version"]}
    before = (root / MARKER_FNAME).read_bytes()
    assert not configured.mark(root, campaign_id="camp", scenario_manifest_sha256=population)
    assert (root / MARKER_FNAME).read_bytes() == before
    with pytest.raises(ConfiguredRootError, match="marks campaign 'camp' .* not campaign 'other'"):
        configured.mark(root, campaign_id="other", scenario_manifest_sha256=population)
    for name in (SAMPLING_MANIFEST_FNAME, configured.LIBRARY_MARKER_FNAME):
        made = tmp_path / name.strip(".")
        made.mkdir()
        (made / name).write_text("{}")
        assert not configured.mark(made, campaign_id="camp", scenario_manifest_sha256=population)
        assert not (made / MARKER_FNAME).exists()


@pytest.mark.parametrize(("plant", "message"), [
    (None, f"holds no {MARKER_FNAME} of its own: a root this version's"),
    ("link", "a link in its place marks nothing"),
    ("{not json", "cannot be read as a configured root's marker"),
    ('{"kind": "ruleforge-campaign"}', "is not a ruleforge-configured-campaign marker"),
])
def test_a_marker_that_is_none_a_link_unreadable_or_of_another_kind_is_refused(tmp_path, plant, message):
    root = tmp_path / "root"
    root.mkdir()
    if plant == "link":
        (tmp_path / "elsewhere.json").write_text("{}")
        (root / MARKER_FNAME).symlink_to(tmp_path / "elsewhere.json")
    elif plant is not None:
        (root / MARKER_FNAME).write_text(plant)
    with pytest.raises(ConfiguredRootError, match=message):
        configured.read_marker(root)
    if plant == "link":
        with pytest.raises(ConfiguredRootError, match="a link in its place marks nothing"):
            configured.mark(root, campaign_id="camp", scenario_manifest_sha256="0" * 64)


def test_a_root_is_held_to_the_campaign_and_population_its_marker_names(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    configured.mark(root, campaign_id="camp", scenario_manifest_sha256="0" * 64)
    with pytest.raises(ConfiguredRootError, match="does not hold a whole campaign's records"):
        configured.check_root(root)
    _records(root, tmp_path)
    with pytest.raises(ConfiguredRootError, match="over population 000000000000..., not campaign 'camp' over"):
        configured.check_root(root)
    other = tmp_path / "other"
    other.mkdir()
    population = ScenarioManifest.build(write_inputs(tmp_path / "inputs")).manifest_sha256
    configured.mark(other, campaign_id="camp", scenario_manifest_sha256=population)
    _records(other, tmp_path)
    assert configured.check_root(other)["campaign_id"] == "camp"


def test_which_roots_the_commands_take_as_configured(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert not configured.is_configured(root) and configured.merges_into(root)
    (root / MARKER_FNAME).write_text("{}")
    assert configured.is_configured(root) and configured.merges_into(root)
    (root / configured.LIBRARY_MARKER_FNAME).write_text("{}")
    assert not configured.is_configured(root) and not configured.merges_into(root)
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    (other_root / SAMPLING_MANIFEST_FNAME).write_text("{}")
    assert not configured.is_configured(other_root) and not configured.merges_into(other_root)


def test_a_configured_root_on_a_pinned_root_s_path_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    root = tmp_path / "artifacts" / "sampling" / "pinned"
    root.mkdir(parents=True)
    population = ScenarioManifest.build(write_inputs(tmp_path / "inputs")).manifest_sha256
    configured.mark(root, campaign_id="camp", scenario_manifest_sha256=population)
    _records(root, tmp_path)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    assert configured.checked_root(root) == Path(os.path.realpath(root))
    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://sampling/pinned": ("leniency",)})
    with pytest.raises(api.CampaignError, match="a published run of leniency"):
        configured.checked_root(root)


def test_an_analysis_takes_the_default_permutations_and_refuses_one_there_unless_asked(tmp_path, modelspec):
    from tests.test_sampling_campaign import run_two_conditions

    root, _ = run_two_conditions(tmp_path, modelspec)
    assert configured.analyze(root, n_resamples=100) == Path(os.path.realpath(root))
    estimates = root / "aggregate" / "estimates.json"
    assert json.loads(estimates.read_text())["randomization_test"]["n_permutations"] == 100_000
    with pytest.raises(api.AnalysisExists, match="holds this root's analysis"):
        configured.analyze(root, n_resamples=100)
    assert configured.analyze(root, n_resamples=100, n_permutations=200, replace=True) == Path(os.path.realpath(root))
    assert json.loads(estimates.read_text())["randomization_test"]["n_permutations"] == 200


def test_an_analysis_written_between_the_first_check_and_the_lock_is_never_written_over(tmp_path, modelspec,
                                                                                      monkeypatch):
    from contextlib import contextmanager

    from ruleforge.evaluate import sharding
    from ruleforge.evaluate.aggregate import analyze_campaign
    from tests.test_sampling_campaign import run_two_conditions

    root, _ = run_two_conditions(tmp_path, modelspec)
    held = sharding.campaign_lock

    @contextmanager
    def raced(campaign_root):
        analyze_campaign(root, n_resamples=100, n_permutations=200)
        with held(campaign_root) as path:
            yield path

    monkeypatch.setattr(sharding, "campaign_lock", raced)
    with pytest.raises(api.AnalysisExists, match="holds this root's analysis"):
        configured.analyze(root, n_resamples=100)
    estimates = json.loads((root / "aggregate" / "estimates.json").read_text())
    assert estimates["randomization_test"]["n_permutations"] == 200


def test_a_root_made_between_the_merge_s_first_check_and_its_locks_is_not_merged_into(tmp_path, modelspec,
                                                                                    monkeypatch):
    """:func:`configured.check_merge` runs again under the locks the merge holds through its last write."""
    import shutil
    from contextlib import contextmanager

    from ruleforge.evaluate import sharding

    root = tmp_path / "raced"
    run_shards(tmp_path, modelspec, root, 3)
    shard = sorted((shards_root(root) / CONDITION).iterdir())[0]
    held = sharding.campaign_lock

    @contextmanager
    def raced(campaign_root):
        root.mkdir(exist_ok=True)
        for name in (SAMPLING_MANIFEST_FNAME, "scenario_manifest.json"):
            shutil.copyfile(shard / name, root / name)
        with held(campaign_root) as path:
            yield path

    monkeypatch.setattr(sharding, "campaign_lock", raced)
    with pytest.raises(ConfiguredRootError, match=f"holds no {MARKER_FNAME} of its own"):
        configured.merge_shards(root, CONDITION)
    assert not (root / CONDITION).exists() and not (root / MARKER_FNAME).exists()


def test_a_merge_makes_a_root_marked_as_its_shards_are_and_only_from_this_version_s_shards(tmp_path, modelspec,
                                                                                         capsys):
    """Into a root not made yet the merge writes the first shard root's marker before the manifests."""
    marked = tmp_path / "marked"
    run_shards(tmp_path, modelspec, marked, 3)
    configured.merge_shards(marked, CONDITION)
    shard = next((shards_root(marked) / CONDITION).iterdir())
    assert (marked / MARKER_FNAME).read_bytes() == (shard / MARKER_FNAME).read_bytes()
    assert configured.check_root(marked)["campaign_id"] == json.loads((marked / SAMPLING_MANIFEST_FNAME).read_text())[
        "campaign_id"]
    configured.check_merge(marked, CONDITION)
    unmarked = tmp_path / "unmarked"
    run_shards(tmp_path, modelspec, unmarked, 3)
    for root in (shards_root(unmarked) / CONDITION).iterdir():
        (root / MARKER_FNAME).unlink()
    with pytest.raises(ConfiguredRootError, match=f"holds no {MARKER_FNAME} of its own"):
        configured.check_merge(unmarked, CONDITION)
    merge_condition_shards(unmarked, CONDITION)
    assert not (unmarked / MARKER_FNAME).exists() and (unmarked / SAMPLING_MANIFEST_FNAME).is_file()
    stale = tmp_path / "stale"
    run_shards(tmp_path, modelspec, stale, 3)
    stale.mkdir(exist_ok=True)
    (stale / MARKER_FNAME).write_bytes(json.dumps({"kind": configured.MARKER_KIND, "campaign_id": "another",
                                                   "scenario_manifest_sha256": "0" * 64}).encode())
    with pytest.raises(ConfiguredRootError, match="marks campaign 'another'"):
        configured.check_merge(stale, CONDITION)
    assert main(["campaign", "merge-shards", "--campaign-root", str(stale), "--condition-id", CONDITION]) == 2
    assert "marks campaign 'another'" in capsys.readouterr().err
    assert not (stale / SAMPLING_MANIFEST_FNAME).exists()


def test_a_root_marked_before_its_manifests_is_a_campaign_root_to_every_guard(tmp_path):
    from ruleforge.evaluate.shard_merge import _check_destinations
    from ruleforge.evaluate.sharding import ShardError
    from ruleforge.hub import HubError, refuse_campaign
    from ruleforge.launch.environment import LaunchError
    from ruleforge.launch.guards import refuse_linked_way

    stopped = tmp_path / "stopped" / "root"
    stopped.mkdir(parents=True)
    assert configured.mark(stopped, campaign_id="camp", scenario_manifest_sha256="0" * 64)
    linked = tmp_path / "linked" / "root"
    linked.mkdir(parents=True)
    (linked / MARKER_FNAME).symlink_to(tmp_path / "nowhere")
    for root in (stopped, linked):
        outer = root.parent
        assert not (root / SAMPLING_MANIFEST_FNAME).exists() and api._is_campaign_root(root)
        for campaign_root in (False, True):
            with pytest.raises(api.CampaignError, match=f"is inside the campaign root {root}"):
                api.refuse_protected(root / "child", "an --out", campaign_root=campaign_root)
        with pytest.raises(api.CampaignError, match="is a campaign root, which nothing but its own"):
            api.refuse_protected(root, "an --out")
        with pytest.raises(api.CampaignError, match=f"holds a campaign root of its own \\({root}\\)"):
            api.refuse_protected(outer, "an --out", below=True)
        with pytest.raises(HubError, match=f"is in the campaign root {root}"):
            refuse_campaign(root / "data", "the fetch's root")
        with pytest.raises(ShardError, match=f"{root} is a campaign root of its own inside {outer}"):
            _check_destinations(outer, [Path(root.name) / "draw-000"])
        with pytest.raises(LaunchError, match=f"{root}, on its way, is another campaign's root"):
            refuse_linked_way(outer, root / "hydra", what="a directory the job writes", roots=True)


class _Stopped(Exception):
    """A merge stopped part-way (killed, out of time, out of disk)."""


@pytest.mark.parametrize("stop", [MARKER_FNAME, "scenario_manifest.json", SAMPLING_MANIFEST_FNAME])
def test_a_first_merge_stopped_part_way_is_completed_by_merging_again(tmp_path, modelspec, monkeypatch, stop):
    from ruleforge.evaluate import shard_merge

    root = tmp_path / "root"
    run_shards(tmp_path, modelspec, root, 3)
    copy = shard_merge._copy

    def stopping(source, destination):
        copy(source, destination)
        if destination.name == stop:
            raise _Stopped(destination)

    monkeypatch.setattr(shard_merge, "_copy", stopping)
    with pytest.raises(_Stopped):
        configured.merge_shards(root, CONDITION)
    assert (root / stop).is_file() and not (root / CONDITION).exists()
    monkeypatch.setattr(shard_merge, "_copy", copy)
    assert main(["campaign", "merge-shards", "--campaign-root", str(root), "--condition-id", CONDITION]) == 0
    marker = next((shards_root(root) / CONDITION).iterdir()) / MARKER_FNAME
    assert configured.check_root(root) == json.loads(marker.read_text()) and (root / CONDITION).is_dir()


def _foreign_marker(place: Path) -> None:
    (place / MARKER_FNAME).unlink(missing_ok=True)
    (place / MARKER_FNAME).write_bytes(configured.marker_bytes("another", "0" * 64))


@pytest.mark.parametrize(("plant", "message"), [
    (lambda root, shards: _foreign_marker(root), "marks campaign 'another'"),
    (lambda root, shards: (_foreign_marker(root), (root / "scenario_manifest.json").write_bytes(
        (shards[0] / "scenario_manifest.json").read_bytes())), "marks campaign 'another'"),
    (lambda root, shards: (root / MARKER_FNAME).symlink_to(shards[0] / MARKER_FNAME), "a link in its place"),
    (lambda root, shards: (root / MARKER_FNAME).write_text('{"kind": "other"}'), "is not a .* marker"),
    (lambda root, shards: _foreign_marker(shards[1]), "marks campaign 'another'"),
], ids=["root-alone", "root-with-population", "root-link", "root-other-kind", "shard"])
def test_every_merge_holds_the_root_and_its_shards_to_the_markers_they_hold(tmp_path, modelspec, plant, message):
    root = tmp_path / "root"
    run_shards(tmp_path, modelspec, root, 3)
    shards = sorted((shards_root(root) / CONDITION).iterdir())
    root.mkdir()
    plant(root, shards)
    held = sorted(path.name for path in root.iterdir())
    with pytest.raises(ConfiguredRootError, match=message):
        merge_condition_shards(root, CONDITION)
    assert sorted(path.name for path in root.iterdir()) == held


def test_a_merge_takes_a_root_holding_its_shards_marker_alone(tmp_path, modelspec):
    root = tmp_path / "root"
    run_shards(tmp_path, modelspec, root, 3)
    root.mkdir()
    shard = next((shards_root(root) / CONDITION).iterdir())
    (root / MARKER_FNAME).write_bytes((shard / MARKER_FNAME).read_bytes())
    merge_condition_shards(root, CONDITION)
    assert configured.check_root(root) == json.loads((shard / MARKER_FNAME).read_text()) and (root / CONDITION).is_dir()


def test_a_job_s_preflight_holds_its_roots_to_the_markers_they_hold(tmp_path, modelspec):
    from ruleforge.evaluate.campaign import preflight
    from ruleforge.evaluate.sharding import shard_root
    from tests.test_sampling_shards import compose_cfg, shard_cfg

    root = tmp_path / "root"
    for place in (root, shard_root(root, CONDITION, 0, 2)):
        place.mkdir(parents=True)
        _foreign_marker(place)
        with pytest.raises(ConfiguredRootError, match="marks campaign 'another'"):
            preflight(shard_cfg(tmp_path, root, 0, 2))
        (place / MARKER_FNAME).unlink()
    with pytest.raises(ConfiguredRootError, match="marks campaign 'another'"):
        _foreign_marker(root)
        preflight(compose_cfg(tmp_path, root))
