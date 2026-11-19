from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import ruleforge.judges.openai_batch as openai_batch
from ruleforge.cli import records
from ruleforge.evaluate.api import Campaign
from ruleforge.judges import JudgeSpec
from ruleforge.judges.base import Capabilities
from ruleforge.judges.batch_collect import main as collect_main
from ruleforge.evaluate.identity import SamplingConfig
from ruleforge.evaluate.sharding import shards_root
from ruleforge.scenarios import ScenarioSet
from tests.cli.test_evaluate_cli import evaluate
from tests.support.equivalence import tree
from tests.test_batch_collect import EXPIRED, EndedOpenAI, submitted
from tests.test_sampling_shards import _environment as _shards_clock  # noqa: F401  (one clock: the bytes)
from tests.test_sampling_shards import CONDITION, compose_cfg, run_shards


def test_collect_writes_the_script_s_files_and_line_and_refuses_as_it_does(cli, tmp_path, monkeypatch, capsys):
    scripted, lines = submitted(tmp_path / "script")
    commanded, _ = submitted(tmp_path / "command")
    client = EndedOpenAI({**lines}, {"batch-0": {"status": EXPIRED, "returned": [0]}})
    monkeypatch.setattr(openai_batch, "default_client", lambda: client)
    assert collect_main([str(scripted)]) == 0
    expected = capsys.readouterr().out
    status, out, _ = cli("campaign", "collect", commanded)
    assert status == 0 and out == expected and "1 of 4 request(s) returned" in out
    assert tree(commanded) == tree(scripted)
    assert not records.runs_dir(commanded).exists()
    running, more = submitted(tmp_path / "running")
    client.endings["batch-0"] = {"status": "in_progress"}
    client.lines = more
    status, _, error = cli("campaign", "collect", running, tmp_path / "nothing")
    assert status == 2 and "have not ended" in error and "holds no state.json" in error


def test_collect_refuses_a_linked_or_pinned_work_directory(cli, tmp_path, monkeypatch):
    """A link would carry its writes out; a pinned root's batch state (a published root's) is never written."""
    import ruleforge.evaluate.api as api

    directory, _ = submitted(tmp_path / "batch")
    link = tmp_path / "link"
    link.symlink_to(directory)
    before = tree(directory)
    status, _, error = cli("campaign", "collect", link)
    assert status == 2 and "is a symlink" in error and tree(directory) == before
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://batch": ("leniency",)})
    status, _, error = cli("campaign", "collect", directory)
    assert status == 2 and "a published run of leniency" in error and tree(directory) == before


def test_collect_refuses_a_link_below_the_work_directory(cli, tmp_path, monkeypatch):
    """A real work directory whose ``outputs/`` is a link elsewhere (a published root's, a campaign root's)."""
    directory, lines = submitted(tmp_path / "batch")
    client = EndedOpenAI({**lines}, {"batch-0": {"status": EXPIRED, "returned": [0]}})
    monkeypatch.setattr(openai_batch, "default_client", lambda: client)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (directory / "outputs").symlink_to(elsewhere)
    before = tree(directory)
    status, _, error = cli("campaign", "collect", directory)
    assert status == 2 and "outputs, below the batch work directory" in error and "is a symlink" in error
    assert list(elsewhere.iterdir()) == [] and tree(directory) == before


def test_collect_checks_the_work_directory_again_as_the_transport_derives_it(cli, tmp_path, monkeypatch):
    import ruleforge.evaluate.api as api

    directory, lines = submitted(tmp_path / "batch")
    client = EndedOpenAI({**lines}, {"batch-0": {"status": EXPIRED, "returned": [0]}})
    monkeypatch.setattr(openai_batch, "default_client", lambda: client)
    checked = []

    def refuse_work_directory(place, what):
        checked.append(Path(place))
        if len(checked) > 1:
            raise api.CampaignError(f"{what} {place} changed since it was checked")

    monkeypatch.setattr(api, "refuse_work_directory", refuse_work_directory)
    before = tree(directory)
    status, _, error = cli("campaign", "collect", directory)
    assert status == 2 and "changed since it was checked" in error and checked == [directory, directory]
    assert tree(directory) == before


@pytest.mark.parametrize("marked", [False, True], ids=["configured", "library"])
def test_collect_holds_a_work_directory_to_the_batch_root_rule(cli, tmp_path, monkeypatch, marked):
    from tests.test_sampling_runner import write_inputs

    if marked:
        root = Campaign.create(tmp_path / "root", campaign_id="c",
                               scenarios=ScenarioSet.load(write_inputs(tmp_path))).root
    else:
        root = tmp_path / "root"
        root.mkdir()
        (root / "sampling_manifest.json").write_text("{}", encoding="utf-8")
    stray, lines = submitted(root / "luna" / "naive-rules-above")
    client = EndedOpenAI({**lines}, {"batch-0": {"status": EXPIRED, "returned": [0]}})
    monkeypatch.setattr(openai_batch, "default_client", lambda: client)
    before = tree(root)
    for spelled in (stray, tmp_path / "via" / stray.name):
        if spelled != stray:
            (tmp_path / "via").symlink_to(stray.parent, target_is_directory=True)
        status, _, error = cli("campaign", "collect", spelled)
        assert status == 2 and f"lies in the campaign root {os.path.realpath(root)} outside its batch/" in error, error
        assert "scripts/" not in error and tree(root) == before
    (tmp_path / "via").unlink()
    shutil.rmtree(root / "luna")
    own, client.lines = submitted(root / "batch")
    status, out, _ = cli("campaign", "collect", own)
    assert status == 0 and "1 of 4 request(s) returned" in out and (own / "partial.json").is_file()
    (tmp_path / "via").symlink_to(root / "batch", target_is_directory=True)
    status, out, _ = cli("campaign", "collect", tmp_path / "via" / own.name)
    assert status == 0 and "1 of 4 request(s) returned" in out


def _own_sharded_root(tmp_path, modelspec, *extra: str) -> Path:
    root = tmp_path / "sharded"
    cfg = compose_cfg(tmp_path, root, *extra)
    Campaign.create(root, campaign_id="camp", scenarios=ScenarioSet.load(str(cfg.scenario_set_path)),
                    sampling=SamplingConfig.from_mapping(OmegaConf.to_container(cfg.sampling, resolve=True)))
    run_shards(tmp_path, modelspec, root, 3, *extra)
    return root


def _digest(root: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


#: A subset condition of a shared-bank root's model, run sharded.
SUBSET = "gemma-4-31B-it/subsets/s00/k02"


def _bank_condition(root: Path, **changes):
    import dataclasses

    from ruleforge.evaluate.campaign import read_condition
    from ruleforge.evaluate.sharding import shard_dirname

    subset = read_condition(shards_root(root) / SUBSET / shard_dirname(0, 3) / SUBSET / "condition.json")
    return dataclasses.replace(subset, condition_id="gemma-4-31B-it/bank", **changes)


def _bank_root(tmp_path, modelspec, *, registered: bool = True, **bank_changes) -> Path:
    from ruleforge.evaluate.bank import SharedBankDesign, write_design
    from ruleforge.rulesets.subsets import WITHOUT_REPLACEMENT
    from ruleforge.evaluate.campaign import register_condition

    root = _own_sharded_root(tmp_path, modelspec, f"condition_id={SUBSET}")
    write_design(root, SharedBankDesign(subset_seeds=(0, 1), subset_sizes=(2, 3), sampling_strategy=WITHOUT_REPLACEMENT,
                                     subset_manifest_sha256="a" * 64, source_ruleset_sha256="b" * 64))
    if registered:
        register_condition(root, _bank_condition(root, **bank_changes))
    return root


def test_a_shard_merge_under_another_judge_configuration_than_its_model_s_is_refused_before_any_write(
        cli, tmp_path, modelspec, capsys):
    from ruleforge.evaluate.shard_merge import MERGING_SUFFIX, merge_condition_shards
    from ruleforge.launch import records as launch_records
    from ruleforge.evaluate.runner import CampaignRunError

    root = _bank_root(tmp_path, modelspec, prompt_style="forced-grpo")
    before = _digest(root)
    refusal = (r"gemma-4-31B-it: .* hold more than one judge configuration \(prompt_style: 'naive-rules-above' "
               r"\(gemma-4-31B-it/subsets/s00/k02\) vs 'forced-grpo' \(gemma-4-31B-it/bank\)\)")
    status, _, error = cli("campaign", "merge-shards", "--campaign-root", root, "--condition-id", SUBSET)
    assert status == 1 and "more than one judge configuration (prompt_style" in error and _digest(root) == before
    with pytest.raises(CampaignRunError, match=refusal):
        Campaign.reopen(root).merge_shards(SUBSET)
    assert _digest(root) == before
    with pytest.raises(CampaignRunError, match=refusal):
        merge_condition_shards(root, SUBSET, num_shards=3)
    assert issubclass(CampaignRunError, launch_records._MERGE_ERRORS)
    assert _digest(root) == before and not (root / SUBSET / "draw-000").exists()
    assert not [entry for entry in root.parent.iterdir() if MERGING_SUFFIX in entry.name]


def test_a_direct_merge_holds_the_root_s_write_lock_from_its_check_to_its_gate(tmp_path, modelspec, monkeypatch,
                                                                                capsys):
    import ruleforge.evaluate.shard_merge as shard_merge
    from ruleforge.evaluate.api import LOCK_FNAME
    from tests.evaluate.test_campaign_ops import _root_locked

    root = _bank_root(tmp_path, modelspec)
    held = []
    check, gate = shard_merge.check_judge_configuration, shard_merge.campaign_completion

    def checked(*args, **kwargs):
        held.append(("check", _root_locked(root)))
        return check(*args, **kwargs)

    def published(*args, **kwargs):
        held.append(("gate", _root_locked(root)))
        return gate(*args, **kwargs)

    monkeypatch.setattr(shard_merge, "check_judge_configuration", checked)
    monkeypatch.setattr(shard_merge, "campaign_completion", published)
    report = shard_merge.merge_condition_shards(root, SUBSET)
    assert held == [("check", True), ("gate", True)] and not _root_locked(root)
    assert report["condition_state"] == {"0": "complete", "1": "complete"}
    foreign = tmp_path / "foreign"
    run_shards(tmp_path, modelspec, foreign, 3)
    report = shard_merge.merge_condition_shards(foreign, CONDITION)
    assert report["campaign_state"] == "complete" and not os.path.lexists(foreign / LOCK_FNAME)


def test_a_registration_made_while_a_direct_merge_runs_waits_for_it_and_is_held_to_what_it_registered(
        tmp_path, modelspec, monkeypatch):
    import threading

    import ruleforge.evaluate.shard_merge as shard_merge
    from ruleforge.evaluate.api import _root_lock
    from ruleforge.evaluate.campaign import register_condition
    from ruleforge.evaluate.runner import CampaignRunError

    root = Path(os.path.realpath(_bank_root(tmp_path, modelspec, registered=False)))
    bank = _bank_condition(root, prompt_style="forced-grpo")
    outcome, trying, waited = [], threading.Event(), []

    def register():
        trying.set()
        try:
            with _root_lock(root):
                register_condition(root, bank)
        except CampaignRunError as error:
            outcome.append(error)
        else:
            outcome.append(None)

    competitor = threading.Thread(target=register, daemon=True)
    write = shard_merge._write_draw

    def staged(*args, **kwargs):
        if not waited:
            competitor.start()
            trying.wait()
            competitor.join(timeout=1.0)
            waited.append(competitor.is_alive())
        return write(*args, **kwargs)

    monkeypatch.setattr(shard_merge, "_write_draw", staged)
    report = shard_merge.merge_condition_shards(root, SUBSET)
    competitor.join(timeout=60)
    assert waited == [True] and report["condition_state"] == {"0": "complete", "1": "complete"}
    assert len(outcome) == 1 and isinstance(outcome[0], CampaignRunError)
    assert "more than one judge configuration (prompt_style: 'forced-grpo'" in str(outcome[0])
    assert not (root / "gemma-4-31B-it" / "bank").exists() and (root / SUBSET / "condition.json").is_file()


def test_a_shard_merge_under_its_model_s_judge_configuration_merges_into_a_shared_bank_root(cli, tmp_path, modelspec):
    root = _bank_root(tmp_path, modelspec)
    status, out, _ = cli("campaign", "merge-shards", "--campaign-root", root, "--condition-id", SUBSET)
    assert status == 0 and out["condition_state"] == {"0": "complete", "1": "complete"}
    assert (root / SUBSET / "condition.json").is_file() and (root / SUBSET / "draw-001").is_dir()


def test_merge_shards_merges_into_a_root_of_its_own_as_the_library_function_does(cli, tmp_path, modelspec):
    from ruleforge.evaluate.shard_merge import merge_condition_shards

    root = _own_sharded_root(tmp_path, modelspec)
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    shards = shards_root(root) / CONDITION
    expected = json.loads(json.dumps(merge_condition_shards(copy, CONDITION, shards_dir=shards)))
    status, out, _ = cli("campaign", "merge-shards", "--campaign-root", root, "--condition-id", CONDITION,
                         "--num-shards", "3")
    assert status == 0 and out["campaign_state"] == expected["campaign_state"] == "complete"
    assert {key: value for key, value in out.items() if key != "campaign_root"} == {
        key: value for key, value in expected.items() if key != "campaign_root"}
    assert tree(root) == tree(copy)
    assert next(records.runs_dir(root).iterdir()).name.endswith("-campaign-merge-shards")
    status, _, error = cli("campaign", "merge-shards", "--campaign-root", root, "--condition-id", "x/y")
    assert status == 1 and "holds no shard roots" in error


def test_merge_shards_writes_under_the_root_s_lock(cli, tmp_path, modelspec, monkeypatch):
    import fcntl
    import os

    import ruleforge.evaluate.shard_merge as shard_merge
    from ruleforge.evaluate.api import LOCK_FNAME

    root = _own_sharded_root(tmp_path, modelspec)
    held = []
    merge = shard_merge.merge_condition_shards

    def merged(*args, **kwargs):
        descriptor = os.open(root / LOCK_FNAME, os.O_WRONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            held.append(True)
        finally:
            os.close(descriptor)
        return merge(*args, **kwargs)

    monkeypatch.setattr(shard_merge, "merge_condition_shards", merged)
    status, out, _ = cli("campaign", "merge-shards", "--campaign-root", root, "--condition-id", CONDITION)
    assert status == 0 and held == [True] and out["campaign_state"] == "complete"


@pytest.mark.parametrize("planted", [CONDITION.split("/")[0], "code_snapshots"])
def test_merge_shards_refuses_a_link_below_the_root_before_anything_is_written(cli, tmp_path, modelspec, planted):
    root = _own_sharded_root(tmp_path, modelspec)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if (root / planted).exists():
        shutil.rmtree(root / planted)
    (root / planted).symlink_to(elsewhere)
    status, _, error = cli("campaign", "merge-shards", "--campaign-root", root, "--condition-id", CONDITION)
    assert status == 2 and f"{planted} is a symlink" in error
    assert list(elsewhere.iterdir()) == [] and not records.runs_dir(root).exists()


def test_merge_shards_refuses_a_root_neither_this_library_nor_this_version_s_jobs_made(cli, tmp_path, modelspec):
    """Shard roots no configured job of this version made are refused into a root not made yet."""
    from ruleforge.evaluate.configured import MARKER_FNAME

    root = tmp_path / "sharded"
    run_shards(tmp_path, modelspec, root, 3)
    for shard in (shards_root(root) / CONDITION).iterdir():
        (shard / MARKER_FNAME).unlink()
    before = tree(tmp_path)
    status, _, error = cli("campaign", "merge-shards", "--campaign-root", root, "--condition-id", CONDITION)
    assert status == 2 and f"holds no {MARKER_FNAME} of its own" in error and tree(tmp_path) == before
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    (other_root / "sampling_manifest.json").write_text("{}")
    before = tree(tmp_path)
    status, _, error = cli("campaign", "merge-shards", "--campaign-root", other_root, "--condition-id", CONDITION,
                           "--shards-dir", shards_root(root) / CONDITION)
    assert status == 2 and "did not create" in error and tree(tmp_path) == before


@pytest.fixture
def own_root(cli, inputs):
    """A root ``evaluate`` made under the fake judge, and a second condition registered beside it and not run."""
    scenarios, rules = inputs
    status, first, _ = evaluate(cli, inputs, "--out", "runs/r", "--campaign-id", "r")
    root = Path(first["root"])
    status, added, _ = cli("campaign", "add", root, "--scenarios", scenarios, "--rules", rules, "--judge", "fake:2")
    return root, first["condition_id"], added["condition_id"]


def test_preflight_only_reports_what_a_run_would_judge_and_writes_nothing(cli, inputs, own_root):
    scenarios, rules = inputs
    root, done, pending = own_root
    before = tree(root.parent)
    status, out, _ = cli("campaign", "run", root, pending, "--judge", "fake:2", "--rules", rules, "--scenarios",
                         scenarios, "--preflight-only")
    registered = json.loads((root / pending / "condition.json").read_text(encoding="utf-8"))
    assert status == 0 and out["pending"] == [0, 1] and out["draws"] == {"0": "in_progress", "1": "in_progress"}
    assert out["fingerprint"] == registered["fingerprint"] and out["identity"]["model_name"] == "fake-2"
    assert out["requests_per_draw"] == len(ScenarioSet.load(scenarios)) * 3 and out["requests"] == 2 * 3 * 3
    status, out, _ = cli("campaign", "run", root, done, "--judge", "fake", "--rules", rules, "--scenarios",
                         scenarios, "--preflight-only", "--draw-ids", "1")
    assert status == 0 and out["pending"] == [] and out["draws"] == {"1": "complete"} and out["requests"] == 0
    assert tree(root.parent) == before


@pytest.mark.parametrize("change, message", [
    (("--judge", "fake:3"), "different identity"),
    (("--", "execution.num_shards=2", "execution.shard_index=0"), "shards a configured campaign's condition"),
])
def test_preflight_only_refuses_what_the_run_would_refuse(cli, inputs, own_root, change, message):
    scenarios, rules = inputs
    root, _, pending = own_root
    judge = ("--judge", "fake:2") if change[0] != "--judge" else change
    rest = change if change[0] == "--" else ()
    before = tree(root.parent)
    status, _, error = cli("campaign", "run", root, pending, *judge, "--rules", rules, "--scenarios", scenarios,
                           "--preflight-only", *rest)
    assert status == 2 and message in error and tree(root.parent) == before


def test_preflight_only_refuses_another_population_and_an_unconfirmed_spend(cli, inputs, own_root, tmp_path,
                                                                           monkeypatch):
    scenarios, rules = inputs
    root, _, pending = own_root
    other = tmp_path / "other.jsonl"
    other.write_text(scenarios.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    status, _, error = cli("campaign", "run", root, pending, "--judge", "fake:2", "--rules", rules, "--scenarios",
                           other, "--preflight-only")
    assert status == 2 and "records campaign 'r' over population" in error
    billed = Capabilities(seed="local_deterministic", batch=False, per_request_errors=True, spends_money=True)
    monkeypatch.setattr(JudgeSpec, "capabilities", lambda self: billed)
    status, _, error = cli("campaign", "run", root, pending, "--judge", "fake:2", "--rules", rules, "--scenarios",
                           scenarios, "--preflight-only")
    assert status == 2 and "spends money on every call" in error
    status, out, _ = cli("campaign", "run", root, pending, "--judge", "fake:2", "--rules", rules, "--scenarios",
                         scenarios, "--preflight-only", "--yes-spend")
    assert status == 0 and out["pending"] == [0, 1]


def test_an_unconfirmed_paid_judge_s_run_is_refused_before_anything_is_written(cli, inputs, own_root, monkeypatch):
    scenarios, rules = inputs
    root, _, pending = own_root
    assert (root / "aggregate" / "estimates.json").is_file()
    for lock in [*root.parent.glob(".*.lock"), *root.rglob("*.lock")]:
        lock.unlink()
    billed = Capabilities(seed="local_deterministic", batch=False, per_request_errors=True, spends_money=True)
    monkeypatch.setattr(JudgeSpec, "capabilities", lambda self: billed)
    before = tree(root.parent)
    status, _, error = cli("campaign", "run", root, pending, "--judge", "fake:2", "--rules", rules, "--scenarios",
                           scenarios)
    assert status == 2 and "spends money on every call" in error and tree(root.parent) == before


def test_an_unconfirmed_paid_judge_s_run_with_nothing_to_judge_returns_its_draws(cli, inputs, own_root, monkeypatch):
    scenarios, rules = inputs
    root, done, _ = own_root
    billed = Capabilities(seed="local_deterministic", batch=False, per_request_errors=True, spends_money=True)
    monkeypatch.setattr(JudgeSpec, "capabilities", lambda self: billed)
    monkeypatch.setattr(JudgeSpec, "request_settings", lambda self: {"reasoning_summary": "detailed"})
    status, out, error = cli("campaign", "run", root, done, "--judge", "fake", "--rules", rules, "--scenarios",
                             scenarios)
    assert status == 0 and out["draws"] == {"0": "complete", "1": "complete"}, error


def test_preflight_only_refuses_a_root_this_library_did_not_create(cli, inputs):
    foreign = Path.cwd() / "published"
    foreign.mkdir()
    (foreign / "sampling_manifest.json").write_text("{}")
    status, _, error = cli("campaign", "run", foreign, "c", "--judge", "fake", "--rules", inputs[1], "--scenarios",
                           inputs[0], "--preflight-only")
    assert status == 2 and "did not create" in error and sorted(path.name for path in foreign.iterdir()) == [
        "sampling_manifest.json"]


def test_a_preflight_of_a_condition_the_root_does_not_register_is_refused(cli, inputs, own_root):
    from ruleforge.evaluate.api import CampaignError, ConditionSettings
    from ruleforge.judges import resolve
    from ruleforge.rulesets import Ruleset

    scenarios, rules = inputs
    root, _, _ = own_root
    draft = Campaign.draft(root, campaign_id="r", scenarios=ScenarioSet.load(scenarios),
                           sampling=Campaign.open(root).sampling)
    with pytest.raises(CampaignError, match="registers no condition 'nope'"):
        draft.preflight("nope", rules=Ruleset.load(rules), judge=resolve("fake"),
                        settings=ConditionSettings())
