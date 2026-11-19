from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from ruleforge.cli import main
from ruleforge.core import record_names
from ruleforge.evaluate import ConditionSettings
from ruleforge.evaluate import aggregate as aggregate_module
from ruleforge.evaluate import api
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate import configured
from ruleforge.evaluate import runner as runner_module
from ruleforge.evaluate import shard_merge as merge_module
from ruleforge.evaluate.aggregate import AGGREGATE_DIRNAME, ESTIMATES_FNAME, analyze_campaign, refuse_stripped
from ruleforge.evaluate.api import LOCK_FNAME, Campaign
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    check_condition_id,
    check_new_condition_id,
    check_registered_condition,
    condition_of,
    condition_payload,
    load_conditions,
    preflight,
    read_condition,
    register_condition,
    registers,
)
from ruleforge.evaluate.identity import ConditionSpec, SamplingConfig
from ruleforge.evaluate.runner import CampaignRunError
from ruleforge.evaluate.shard_merge import merge_condition_shards
from ruleforge.evaluate.sharding import ShardError, campaign_lock_path, shard_root
from ruleforge.judges import JudgeSpec
from ruleforge.launch import campaign as launch_campaign
from ruleforge.launch.cli import main as launcher
from ruleforge.launch.guards import CAMPAIGN_MARKER
from ruleforge.launch.records import records_root
from ruleforge.prompts import styles
from ruleforge.rulesets import Ruleset
from ruleforge.scenarios import ScenarioSet
from ruleforge.scenarios.manifest import SAMPLING_SCHEMA_VERSION, ScenarioManifest
from tests.evaluate.helpers import configured_root
from tests.support.trees import tree_files
from tests.test_launch_backends import FakePopen, _executors
from tests.test_launch_campaign import SLURM, jobs_launch, repo  # noqa: F401 - repo: through checkout
from tests.test_launch_cli import _local_record
from tests.test_sampling_campaign import compose_campaign, run_two_conditions
from tests.test_sampling_runner import RULES, write_inputs

KEY = record_names.IDENTITY_KEYS  # public style -> recorded key
RETIRED = sorted(record_names.RETIRED_KEYS)[0]


def spec(**overrides) -> ConditionSpec:
    fields = dict(condition_id="judge/naive-rules-above", model_name="judge", model_type="local", prompt_style="naive-rules-above",
                  individual_rule_context="isolated", judge_type="No_Interpretation", evaluation_mode="both",
                  rule_set_name="rules.jsonl", rule_set_sha256="a" * 64, temperature=1.0, prompt_shape_version=2,
                  forced_json_requirement="complete_rule_map")
    return ConditionSpec(**{**fields, **overrides})


def recorded_payload(condition: ConditionSpec, **changes) -> dict:
    """``condition``'s ``condition.json`` as the published records wrote it."""
    key = KEY[condition.prompt_style]
    payload = condition_payload(condition)
    payload["prompt_style"] = key
    payload["condition_id"] = "/".join(key if segment == condition.prompt_style else segment
                                       for segment in condition.condition_id.split("/"))
    return {**payload, **changes}


def plant(root: Path, payload) -> Path:
    """Write ``payload`` as the registration ``root`` holds under its condition id."""
    path = root / payload["condition_id"] / CONDITION_FNAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def bytes_of(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in tree_files(root)} if root.exists() else {}


def test_a_recorded_registration_is_the_condition_s_own_and_is_never_rewritten(tmp_path):
    """A styleless id (ruleset size's, RuleForge's, training's) is one id under either naming."""
    for public in (spec(condition_id="judge/bank"), spec(condition_id="judge/encouraged", prompt_style="encouraged")):
        path = plant(tmp_path, recorded_payload(public))
        recorded = read_condition(path)
        assert recorded.prompt_style == public.prompt_style and recorded.fingerprint == public.fingerprint
        assert recorded.condition_id == path.parent.relative_to(tmp_path).as_posix()
        assert recorded.identity_id == recorded.condition_id and recorded.public_id == public.condition_id
        before = path.read_bytes()
        assert check_registered_condition(tmp_path, recorded) == path
        assert register_condition(tmp_path, recorded) == path and path.read_bytes() == before
    assert sorted(load_conditions(tmp_path)) == sorted(["judge/bank", f"judge/{KEY['encouraged']}"])


@pytest.mark.parametrize("change, payload_of", [
    ("another rule set", lambda condition: recorded_payload(condition, rule_set_sha256="b" * 64,
                                                            fingerprint=spec(rule_set_sha256="b" * 64,
                                                                             condition_id="judge/bank").fingerprint)),
    ("another schema version", lambda condition: recorded_payload(condition, schema_version=SAMPLING_SCHEMA_VERSION - 1)),
    ("a fingerprint that is not its condition's", lambda condition: recorded_payload(condition, fingerprint="0" * 64)),
    ("a field no condition has", lambda condition: recorded_payload(condition, unknown=1)),
    ("a field of the wrong type", lambda condition: recorded_payload(condition, temperature="warm")),
])
def test_a_registration_of_another_identity_is_refused(tmp_path, change, payload_of):
    condition = spec(condition_id="judge/bank")
    path = plant(tmp_path, payload_of(condition))
    with pytest.raises(CampaignRunError, match=f"{path} already registers condition 'judge/bank' with a different "
                                               "identity"):
        check_registered_condition(tmp_path, condition)


@pytest.mark.parametrize("field, registered, requested", [
    ("enable_thinking", True, 1),
    ("extra", (("axis", 1),), (("axis", True),)),
])
def test_a_registration_equal_in_python_but_hashed_apart_is_another_identity(tmp_path, field, registered, requested):
    """Python equality takes True for 1, which the fingerprint's canonical JSON hashes apart."""
    held, asked = (spec(condition_id="judge/bank", **{field: value}) for value in (registered, requested))
    assert held == asked and held.fingerprint != asked.fingerprint
    assert registers(condition_payload(held), held) and not registers(condition_payload(held), asked)
    path = plant(tmp_path, condition_payload(held))
    with pytest.raises(CampaignRunError, match=f"{path} already registers condition 'judge/bank' with a different "
                                               "identity"):
        check_registered_condition(tmp_path, asked)


def test_registers_compares_a_payload_in_memory_under_a_schema_version(tmp_path):
    condition = spec(condition_id="judge/bank")
    recorded = recorded_payload(condition)
    assert registers(recorded, condition) and registers(condition_payload(condition), condition)
    older = dict(recorded, schema_version="older")
    assert not registers(older, condition) and registers(older, condition, "older")
    assert not registers([recorded], condition) and not registers(None, condition)
    assert not registers(dict(recorded, prompt_style=RETIRED), condition)
    assert condition_of(recorded, "here") == condition
    with pytest.raises(ValueError, match="here: recorded fingerprint '0+' does not match"):
        condition_of(dict(recorded, fingerprint="0" * 64), "here")


@pytest.mark.parametrize("condition_id, prompt_style, held", [
    (f"judge/{KEY['encouraged']}", "encouraged", f"'{KEY['encouraged']}' \\(the records' key of 'encouraged'\\)"),
    (f"judge/{RETIRED}", "naive-rules-above", f"'{RETIRED}' \\(a retired style's key\\)"),
])
def test_a_new_id_holding_a_recorded_or_retired_key_is_refused_before_anything_is_written(
        tmp_path, condition_id, prompt_style, held):
    condition = spec(condition_id=condition_id, prompt_style=prompt_style)
    for check in (check_registered_condition, register_condition, check_condition_id):
        with pytest.raises(CampaignRunError, match=f"{tmp_path}: condition id '{condition_id}' holds {held} as a "
                                                   "segment"):
            check(tmp_path, condition)
    assert bytes_of(tmp_path) == {}


def test_a_condition_beside_its_recorded_twin_is_refused_and_one_path_is_read(tmp_path, monkeypatch):
    public = spec(condition_id="judge/naive-rules-above")
    twin = plant(tmp_path, recorded_payload(public))
    for check in (check_registered_condition, register_condition, check_condition_id):
        with pytest.raises(CampaignRunError, match=f"{twin} registers '{public.identity_id}', the id condition "
                                                   "'judge/naive-rules-above' is hashed over \\(its identity id\\): the two would "
                                                   "share generation seeds, provider request namespaces and draw "
                                                   "grids.* run 'judge/naive-rules-above' in a root of its own"):
            check(tmp_path, public)
    assert not (tmp_path / "judge" / "naive-rules-above").exists()
    read = []
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: read.append(path) or exists(path))
    check_condition_id(tmp_path, spec(condition_id="judge/bank"))
    check_condition_id(tmp_path, spec(condition_id="other/naive-rules-above"))
    assert read == [tmp_path / "judge/bank" / CONDITION_FNAME, tmp_path / "other/naive-rules-above" / CONDITION_FNAME,
                    tmp_path / f"other/{KEY['naive-rules-above']}" / CONDITION_FNAME]


@pytest.fixture
def shards(tmp_path, monkeypatch):
    """test_sampling_shards' environment and helpers: one clock and one recorded stack for every writer."""
    from tests import test_sampling_shards as module

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))  # the configs' interpolation
    for patched in (runner_module, campaign_module, merge_module, aggregate_module):
        monkeypatch.setattr(patched, "utc_now", lambda: module.NOW)
    monkeypatch.setattr(runner_module, "_package_version", module.STACK.get)
    return module


def _shard_condition(root: Path, condition_id: str) -> ConditionSpec:
    return read_condition(shard_root(root, condition_id, 0, 2) / condition_id / CONDITION_FNAME)


def test_a_shard_merge_beside_the_recorded_twin_of_its_condition_is_refused_before_anything_is_written(
        tmp_path, modelspec, shards):
    root = tmp_path / "root"
    shards.run_shards(tmp_path, modelspec, root, 2)
    condition = _shard_condition(root, shards.CONDITION)
    twin = plant(root, recorded_payload(condition))
    before = bytes_of(root)
    with pytest.raises(CampaignRunError, match=f"{twin} registers '{condition.identity_id}', the id condition "
                                               f"'{shards.CONDITION}' is hashed over"):
        merge_condition_shards(root, shards.CONDITION)
    assert bytes_of(root) == before and not [path for path in tmp_path.iterdir() if ".merging" in path.name]


def test_a_shard_merge_into_a_root_registering_its_condition_by_its_recorded_style_compares_in_memory(
        tmp_path, modelspec, shards):
    """A styleless id is one id under either naming."""
    root, condition_id = tmp_path / "root", "gemma-4-31B-it/bank"
    shards.run_shards(tmp_path, modelspec, root, 2, f"condition_id={condition_id}")
    registered = plant(root, recorded_payload(_shard_condition(root, condition_id)))
    before = registered.read_bytes()
    report = merge_condition_shards(root, condition_id)
    assert set(report["condition_state"].values()) == {"complete"}
    assert registered.read_bytes() == before and read_condition(registered) == _shard_condition(root, condition_id)


def test_a_shard_merge_into_a_root_registering_its_condition_equal_in_python_but_hashed_apart_is_refused(
        tmp_path, modelspec, shards):
    """The root's registration holds 1 where the shards' condition holds True."""
    root, condition_id = tmp_path / "root", "gemma-4-31B-it/bank"
    shards.run_shards(tmp_path, modelspec, root, 2, f"condition_id={condition_id}", "condition_extra={axis: true}")
    condition = _shard_condition(root, condition_id)
    held = ConditionSpec.from_dict({**condition.to_dict(), "extra": {"axis": 1}})
    assert held == condition and held.fingerprint != condition.fingerprint
    registered = plant(root, condition_payload(held))
    before = bytes_of(root)
    with pytest.raises(ShardError, match=f"{registered} registers '{condition_id}' with another identity"):
        merge_condition_shards(root, condition_id)
    assert bytes_of(root) == before


NAIVE_KEPT = f"gemma-4-31B-it/{KEY['naive-rules-above']}"
HELD = f"condition id '{NAIVE_KEPT}' holds '{KEY['naive-rules-above']}' \\(the records' key of 'naive-rules-above'\\) as a segment"


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """An artifact root of the test's own, and the model root a composed campaign config reads."""
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    return tmp_path / "artifacts"


@pytest.fixture
def cli(capsys, tmp_path, monkeypatch, artifacts):
    """Run ``ruleforge ARGV`` from a working directory of its own: its exit status, its stdout and its stderr."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    def run(*argv):
        status = main([str(item) for item in argv])
        captured = capsys.readouterr()
        return status, captured.out, captured.err

    return run


def inputs_of(tmp_path: Path) -> tuple[Path, Path]:
    """The three-scenario population and a two-rule ruleset with a meta-rule, in a directory of their own."""
    directory = tmp_path / "inputs"
    directory.mkdir(exist_ok=True)
    rules = directory / "rules.jsonl"
    rules.write_text("".join(json.dumps(rule) + "\n" for rule in RULES), encoding="utf-8")
    return write_inputs(directory), rules


def around(root: Path) -> dict[str, bytes]:
    """Every file in ``root``'s parent directory -- the root's, its lock and run records beside it -- by its bytes."""
    return bytes_of(root.parent)


def test_the_id_check_reads_one_path_and_takes_an_id_the_root_registers(tmp_path):
    check_new_condition_id(tmp_path, "judge/naive-rules-above")
    with pytest.raises(CampaignRunError, match=f"{tmp_path}: condition id 'judge/{KEY['naive-rules-above']}' holds"):
        check_new_condition_id(tmp_path, f"judge/{KEY['naive-rules-above']}")
    plant(tmp_path, recorded_payload(spec()))
    check_new_condition_id(tmp_path, f"judge/{KEY['naive-rules-above']}")


def test_evaluate_refuses_a_new_id_holding_a_recorded_key_before_it_writes_or_submits(tmp_path, cli, monkeypatch):
    from ruleforge.cli import slurm

    submitted = []
    monkeypatch.setattr(slurm, "submit", lambda specs, profile, **kwargs: submitted.append(specs) or 0)
    scenarios, rules = inputs_of(tmp_path)
    root = tmp_path / "out" / "root"
    root.parent.mkdir()
    for extra in ((), ("--slurm", "example"), ("--slurm", "example", "--dry-run")):
        status, _, error = cli("evaluate", "--rules", rules, "--scenarios", scenarios, "--judge", "fake", "--draws", 2,
                               "--out", root, "--prompt", "forced", "--condition-id", f"fake-0/{KEY['forced']}",
                               *extra)
        assert status == 2 and f"holds '{KEY['forced']}' (the records' key of 'forced') as a segment" in error, error
    assert submitted == [] and around(root) == {}


def test_campaign_add_refuses_a_new_id_holding_a_recorded_key_before_its_run_record(tmp_path, cli):
    scenarios, rules = inputs_of(tmp_path)
    root = tmp_path / "out" / "root"
    status, _, error = cli("campaign", "create", root, "--scenarios", scenarios, "--draws", 2)
    assert status == 0, error
    before = around(root)
    status, _, error = cli("campaign", "add", root, "--scenarios", scenarios, "--rules", rules, "--judge", "fake",
                           "--condition-id", f"fake-0/{KEY['encouraged']}", "--prompt", "encouraged")
    assert status == 2 and f"holds '{KEY['encouraged']}'" in error, error
    assert around(root) == before
    status, _, error = cli("campaign", "add", root, "--scenarios", scenarios, "--rules", rules, "--judge", "fake",
                           "--prompt", "encouraged")  # named by its fingerprint
    assert status == 0, error


def test_every_shard_merge_refuses_a_new_id_holding_a_recorded_key_before_it_writes(tmp_path, cli):
    root = tmp_path / "out" / "root"
    root.mkdir(parents=True)
    for merge in (configured.check_merge, configured.merge_shards):
        with pytest.raises(CampaignRunError, match=HELD):
            merge(root, NAIVE_KEPT)
    status, _, error = cli("campaign", "merge-shards", "--campaign-root", root, "--condition-id", NAIVE_KEPT)
    assert status == 2 and f"holds '{KEY['naive-rules-above']}'" in error, error
    assert around(root) == {}
    scenarios, _ = inputs_of(tmp_path)
    library = tmp_path / "library" / "root"
    Campaign.create(library, campaign_id="c", scenarios=ScenarioSet.load(scenarios),
                    sampling=SamplingConfig(campaign_mode="pilot", num_decoder_draws=2))
    before = around(library)
    campaign = Campaign.reopen(library)
    for merge in (campaign.check_shard_merge, campaign.merge_shards):
        with pytest.raises(api.CampaignError, match=HELD):
            merge(NAIVE_KEPT)
    status, _, error = cli("campaign", "merge-shards", "--campaign-root", library, "--condition-id", NAIVE_KEPT)
    assert status == 2 and f"holds '{KEY['naive-rules-above']}'" in error, error
    assert around(library) == before


def test_a_configured_job_s_preflight_refuses_a_new_id_holding_a_recorded_key_where_its_job_registers_it(
        tmp_path, shards):
    root = tmp_path / "root"
    (tmp_path / "job").mkdir()
    with pytest.raises(CampaignRunError, match=HELD):
        preflight(compose_campaign(tmp_path / "job", campaign_root=str(root), condition_id=NAIVE_KEPT))
    assert not root.exists()
    plant(root, recorded_payload(spec(condition_id="gemma-4-31B-it/naive-rules-above")))
    with pytest.raises(CampaignRunError, match=f"{shard_root(root, NAIVE_KEPT, 0, 2)}: {HELD}"):
        preflight(shards.shard_cfg(tmp_path, root, 0, 2, f"condition_id={NAIVE_KEPT}"))
    assert not shard_root(root, NAIVE_KEPT, 0, 2).exists()


@pytest.fixture
def checkout(request):
    """test_launch_campaign's clean checkout holding the repository's configs, its artifact root beside it."""
    return request.getfixturevalue("repo")


@pytest.mark.parametrize("options", [[], ["--shards", "2"], ["--preflight-only"]])
def test_a_campaign_launch_refuses_a_new_id_holding_a_recorded_key_before_its_preflight_runs(
        checkout, artifacts, tmp_path, capsys, monkeypatch, options):
    root = tmp_path / "artifacts" / "sampling" / "c1"
    root.mkdir(parents=True)
    (root / CAMPAIGN_MARKER).write_text("{}")  # a root of a campaign: --campaign-root resumes it
    before = around(root)
    argv = ["campaign", *SLURM, "--campaign-id", "c1", "--campaign-root", "sampling/c1", *options, "--",
            f"condition_id={NAIVE_KEPT}"]
    for entry in (launcher, None):
        log, executor = _executors()
        popen = FakePopen([(0, ["preflight ok"])])
        given = {"repo": checkout, "popen": popen, "executor": executor}
        status = entry(argv, **given) if entry else jobs_launch(monkeypatch, argv, **given)
        error = capsys.readouterr().err
        assert status == 2 and re.search(HELD, error), error
        assert popen.calls == [] and log["jobs"] == [] and not records_root().exists()
    assert around(root) == before


def test_a_launch_checks_the_id_where_each_of_its_jobs_registers_it(tmp_path, artifacts):
    template = launch_campaign.JobSpec("t", "campaign", 0, ("python", launch_campaign.PROGRAM[-1]),
                                       launch_campaign.PREFLIGHT_RESOURCES)
    root = artifacts / "sampling" / "c1"
    with pytest.raises(CampaignRunError, match=f"{root}: {HELD}"):
        launch_campaign.check_roots([launch_campaign.Condition(template, "c1", "sampling/c1", NAIVE_KEPT)])
    plant(root, recorded_payload(spec(condition_id="gemma-4-31B-it/naive-rules-above")))
    launch_campaign.check_roots([launch_campaign.Condition(template, "c1", "sampling/c1", NAIVE_KEPT)])
    with pytest.raises(CampaignRunError, match=f"{shard_root(root, NAIVE_KEPT, 0, 2)}: {HELD}"):
        launch_campaign.check_roots([launch_campaign.Condition(template, "c1", "sampling/c1", NAIVE_KEPT, shards=2)])


def test_a_pull_refuses_to_merge_a_new_id_holding_a_recorded_key_before_anything_is_written(
        checkout, tmp_path, capsys, cli, monkeypatch):
    artifacts = tmp_path / "artifacts"
    root = artifacts / "sampling" / "c"
    root.mkdir(parents=True)
    _local_record(artifacts, "P1", merges=[{"campaign_root": "sampling/c", "condition_id": NAIVE_KEPT}])
    merged = []
    monkeypatch.setattr("ruleforge.evaluate.shard_merge.merge_condition_shards",
                        lambda *args, **kwargs: merged.append(args) or {})
    before = bytes_of(artifacts)
    for argv in (["pull", "P1"], ["wait", "P1", "--pull", "--timeout-min", "1"]):
        assert launcher(argv, repo=checkout) == 2
        error = capsys.readouterr().err
        assert re.search(HELD, error), error
    status, _, error = cli("jobs", "pull", "P1")
    assert status == 2 and re.search(HELD, error), error
    assert merged == [] and bytes_of(artifacts) == before


def absent(condition_ids: list[str]) -> str:
    """The refusal's phrase (a pattern) for a root that does not hold ``condition_ids``."""
    return re.escape(f"does not hold {condition_ids}, which its {AGGREGATE_DIRNAME}/{ESTIMATES_FNAME} names")


def unlocked(root: Path) -> dict[str, bytes]:
    """:func:`around` ``root`` once the locks an earlier write left are gone."""
    campaign_lock_path(root).unlink(missing_ok=True)
    (root / LOCK_FNAME).unlink(missing_ok=True)
    return around(root)


def cover(root: Path, *condition_ids: str) -> None:
    """Have the analysis ``root`` holds cover ``condition_ids`` too."""
    path = root / AGGREGATE_DIRNAME / ESTIMATES_FNAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["conditions"].update({cid: {} for cid in condition_ids})
    path.write_text(json.dumps(payload), encoding="utf-8")


def _numbers(payload: dict) -> dict:
    return {key: payload[key] for key in ("conditions", "style_families", "forced_conditions")}


def test_an_aggregate_takes_a_root_holding_every_condition_its_analysis_covers_and_refuses_a_copy_without_one(
        tmp_path, artifacts, modelspec):
    root, _ = run_two_conditions(tmp_path, modelspec)
    _, second = sorted(load_conditions(root))
    baseline = _numbers(analyze_campaign(root, n_permutations=100))
    assert _numbers(analyze_campaign(root, n_permutations=100)) == baseline
    shutil.rmtree(root / second)
    before = unlocked(root)
    message = re.escape(f"{root} ") + absent([second]) + re.escape(
        " (the analysis it holds estimates them or lists them as excluded): it is a copy that leaves their directories "
        "out")
    for analysis in (analyze_campaign, refuse_stripped):
        with pytest.raises(CampaignRunError, match=message):
            analysis(root)
    assert around(root) == before


@pytest.mark.parametrize("analysis, refused", [
    (None, False),
    ("{not json", False),
    ([], False),
    ({"conditions": []}, False),
    ({"conditions": {"j/a": {}}, "excluded_conditions": []}, False),
    ({"conditions": {"j/a": {}}}, ["j/a"]),
    ({"conditions": {}, "excluded_conditions": {"j/b": {}, "j/c": {}}}, ["j/b"]),
])
def test_the_conditions_an_analysis_covers_are_those_it_estimates_and_those_it_excludes(tmp_path, analysis, refused):
    plant(tmp_path, condition_payload(spec(condition_id="j/c")))
    if analysis is not None:
        path = tmp_path / AGGREGATE_DIRNAME / ESTIMATES_FNAME
        path.parent.mkdir()
        path.write_text(analysis if isinstance(analysis, str) else json.dumps(analysis), encoding="utf-8")
    if not refused:
        refuse_stripped(tmp_path)
        return
    with pytest.raises(CampaignRunError, match=re.escape(f"{tmp_path} ") + absent(refused)):
        refuse_stripped(tmp_path)


def test_a_configured_analysis_refuses_a_copy_without_a_condition_it_covers_before_its_lock_or_run_record(
        tmp_path, cli):
    root = configured_root(tmp_path / "built")
    campaign_id = json.loads((root / "sampling_manifest.json").read_text())["campaign_id"]
    population = ScenarioManifest.load(root / "scenario_manifest.json").manifest_sha256
    (root / configured.MARKER_FNAME).write_bytes(configured.marker_bytes(campaign_id, population))
    recorded = (root / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_bytes()
    cover(root, "j/gone")
    before = unlocked(root)
    with pytest.raises(CampaignRunError, match=absent(["j/gone"])):
        configured.analyze(root, replace=True, n_permutations=100)
    status, _, error = cli("analyze", root, "--replace")
    assert status == 2 and re.search(absent(["j/gone"]), error), error
    assert around(root) == before
    (root / AGGREGATE_DIRNAME / ESTIMATES_FNAME).write_bytes(recorded)
    configured.analyze(root, replace=True, n_permutations=100)


def test_a_library_campaign_s_analysis_refuses_a_copy_without_a_condition_it_covers_before_its_lock_or_run_record(
        tmp_path, cli):
    scenarios, rules = inputs_of(tmp_path)
    library = tmp_path / "out" / "library"
    status, _, error = cli("evaluate", "--rules", rules, "--scenarios", scenarios, "--judge", "fake", "--draws", 2,
                           "--out", library)
    assert status == 0, error
    recorded = (library / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_bytes()
    cover(library, "j/gone")
    before = unlocked(library)
    with pytest.raises(CampaignRunError, match=absent(["j/gone"])):
        Campaign.reopen(library).analyze(replace=True, n_permutations=100)
    status, _, error = cli("analyze", library, "--replace")
    assert status == 2 and re.search(absent(["j/gone"]), error), error
    assert around(library) == before
    (library / AGGREGATE_DIRNAME / ESTIMATES_FNAME).write_bytes(recorded)
    assert Campaign.reopen(library).analyze(replace=True, n_permutations=100).path is not None


def public_names(value, ids: dict):
    """``value`` with each recorded id in ``ids`` named by its public id, joined keys re-joined in public order."""
    if isinstance(value, dict):
        return {public_names(key, ids): public_names(item, ids) for key, item in value.items()}
    if isinstance(value, list):
        return [public_names(item, ids) for item in value]
    if isinstance(value, str):
        for recorded, public in ids.items():
            value = value.replace(recorded, public)
        parts = value.split("+")
        return "+".join(sorted(parts)) if len(parts) > 1 and set(parts) <= set(ids.values()) else value
    return value


def test_a_recorded_and_a_public_naming_of_one_campaign_give_one_analysis(tmp_path, artifacts, monkeypatch):
    scenarios, rules = inputs_of(tmp_path)
    population, ruleset = ScenarioSet.load(scenarios), Ruleset.load(rules)
    styles_of = {f"fake-1/{KEY[style]}": style for style in (styles.NAIVE_RULES_ABOVE, styles.ENCOURAGED, styles.FORCED)}
    ids = {recorded: f"fake-1/{style}" for recorded, style in styles_of.items()}
    assert [ids[cid] for cid in sorted(ids)] != sorted(ids.values())  # the two namings sort the conditions apart

    def analysed(root: Path, named: dict) -> dict:
        campaign = Campaign.create(root, campaign_id="c", scenarios=population,
                                   sampling=SamplingConfig(campaign_mode="pilot", num_decoder_draws=2))
        for cid, style in named.items():
            campaign.condition(cid, rules=ruleset, judge=JudgeSpec.fake(1), settings=ConditionSettings(prompt=style))
            campaign.run(cid)
        payload = json.loads(campaign.analyze(n_permutations=100).path.read_text(encoding="utf-8"))
        payload.pop("created_at")
        return payload

    public = analysed(tmp_path / "public", {ids[cid]: style for cid, style in styles_of.items()})
    with monkeypatch.context() as patch:
        patch.setattr(record_names, "refuse_record_keys", lambda condition_id: None)
        recorded = analysed(tmp_path / "recorded", styles_of)
    assert sorted(recorded["conditions"]) == sorted(styles_of)
    assert public_names(recorded, ids) == public
