from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ruleforge.evaluate.api import MARKER_FNAME
from ruleforge.evaluate import aggregate as aggregate_module
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate import runner as runner_module
from ruleforge.evaluate import shard_merge as merge_module
from ruleforge.evaluate.aggregate import analyze_campaign
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    SCENARIO_MANIFEST_FNAME,
    build_plan,
    draw_runner,
    preflight,
    run_plan,
    write_shard_record,
)
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.identity import RUN_LOG_ROWS_KEY, RUNS_FNAME, request_block_id, run_log_rows
from ruleforge.core.errors import SamplingDesignError
from ruleforge.evaluate.ledger import LEDGER_FNAME, RAW_OUTPUTS_FNAME
from ruleforge.evaluate.packing import ExecutionSettings
from ruleforge.core.provenance import SNAPSHOT_DIRNAME
from ruleforge.evaluate.identity import (
    COMPLETION_FNAME,
    DRAW_PROVENANCE_FNAME,
)
from ruleforge.evaluate.runner import (
    REQUEST_BLOCKS_FNAME,
    STATE_COMPLETE,
    CampaignRunError,
)
from ruleforge.evaluate.shard_merge import MERGING_SUFFIX, discover_shards, merge_condition_shards
from ruleforge.evaluate.sharding import (
    DrawShard,
    ShardAssignment,
    ShardError,
    assign_units,
    block_members,
    blocks_per_draw,
    campaign_lock,
    campaign_lock_path,
    job_root_override,
    parse_shard_dirname,
    shard_block_id,
    shard_dirname,
    shard_root,
    shards_root,
)
from ruleforge.evaluate.stack import CONDITION_STACK_FNAME
from ruleforge.judges.backend import GenerationOutcome
from ruleforge.judges.batch_api import BatchExecutionError
from tests.conftest import requires_git_worktree
from tests.support.hydra_runs import run_program
from tests.support.trees import tree_files
from tests.test_sampling_runner import RULES, FakeBackend, restamp_stacks

REPO_ROOT = Path(__file__).resolve().parents[1]
N_SCENARIOS = 10
PER_BLOCK = 3  # four first-attempt blocks per draw (3, 3, 3, 1): eight units over two draws
CONDITION = "gemma-4-31B-it/naive-rules-above"
NOW = "2026-09-26T00:00:00+00:00"
#: The stack every job here records, whatever this environment installed (CI has no vllm).
STACK = {"vllm": "0.28.0", "torch": "2.13.0", "transformers": "5.17.0", "tokenizers": "0.22.1", "triton": "3.5.0"}


class SeededBackend(FakeBackend):
    """Answers each prompt from its seed alone: a bundle's fate does not depend on the job that judged it."""

    def __init__(self, *, fail_after=None, seen=None, **kwargs):
        super().__init__(**kwargs)
        self.max_new_tokens = 1024
        self.seen: set[int] = set() if seen is None else seen
        self.fail_after = fail_after

    def generate_once(self, prompts, *, temperature, max_new_tokens=None, seeds=None, request_block_id="",
                      truncation=None):
        if self.fail_after is not None and len(self.blocks) >= self.fail_after:
            raise BatchExecutionError("the engine died")
        self.blocks.append((request_block_id, len(prompts), list(seeds)))
        outcomes = []
        for seed in seeds:
            if seed % 17 == 0 and seed not in self.seen:
                outcomes.append(GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="reset"))
            else:
                label = "maybe" if seed % 13 == 0 else ("yes" if seed % 2 else "no")
                outcomes.append(GenerationOutcome(text=label, raw_response=f"<think>{seed}</think>{label}",
                                                  finish_reason="stop"))
            self.seen.add(seed)
        return outcomes


@pytest.fixture(autouse=True)
def _environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    # One clock for every writer -- the analysis's too -- so two runs of the same judgments write the same bytes.
    for module in (runner_module, campaign_module, merge_module, aggregate_module):
        monkeypatch.setattr(module, "utc_now", lambda: NOW)
    # One recorded inference stack, so a stack refusal has two named versions to compare in any environment.
    monkeypatch.setattr(runner_module, "_package_version", STACK.get)


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text("".join(
        json.dumps({"prompt": f"prompt {i}", "response": f"response {i}"}) + "\n" for i in range(N_SCENARIOS)
    ), encoding="utf-8")
    rules = tmp_path / "rules.jsonl"
    rules.write_text("".join(json.dumps(rule) + "\n" for rule in RULES), encoding="utf-8")
    return scenarios, rules


def compose_cfg(tmp_path: Path, root: Path, *extra: str):
    scenarios, rules = write_inputs(tmp_path)
    overrides = [
        "campaign_id=camp",
        f"campaign_root={root}",
        f"scenario_set_path={scenarios}",
        "approved_scenario_sha256=null",
        f"rule_set_path={rules}",
        "sampling.campaign_mode=pilot",
        "sampling.num_decoder_draws=2",
        f"sampling.scenarios_per_request_block={PER_BLOCK}",
        "sampling.bootstrap.n_resamples=150",
        *extra,
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="sampling_campaign", overrides=overrides)


def shard_cfg(tmp_path, root, index, count, *extra):
    return compose_cfg(
        tmp_path, root, f"execution.num_shards={count}", f"execution.shard_index={index}",
        job_root_override(index, count), *extra,
    )


def task_spec_for(modelspec):
    return modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)


def run_cfg(cfg, modelspec, backend=None):
    backend = backend or SeededBackend()
    plan = build_plan(cfg, backend=backend, task_spec=task_spec_for(modelspec))
    return plan, run_plan(plan, backend=backend, task_spec=task_spec_for(modelspec),
                          execution=campaign_module.execution_settings(cfg))


def run_shards(tmp_path, modelspec, root, count, *extra, skip=()):
    for index in range(count):
        if index not in skip:
            run_cfg(shard_cfg(tmp_path, root, index, count, *extra), modelspec)
    return shards_root(root) / CONDITION


def files_of(root: Path) -> dict[str, Path]:
    return {str(path.relative_to(root)): path for path in tree_files(root)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def unsharded(tmp_path, modelspec):
    root = tmp_path / "unsharded"
    run_cfg(compose_cfg(tmp_path, root), modelspec)
    return root


@pytest.fixture
def sharded(tmp_path, modelspec):
    """Three shards of the condition, run and not yet merged."""
    root = tmp_path / "sharded"
    run_shards(tmp_path, modelspec, root, 3)
    return root


def test_the_campaign_lock_is_never_taken_through_a_symlink(tmp_path):
    """A link planted where the lock goes (beside the root, outside it) would have its target created."""
    root = tmp_path / "campaign"
    root.mkdir()
    target = tmp_path / "published" / "aggregate" / "planted.json"
    target.parent.mkdir(parents=True)
    campaign_lock_path(root).symlink_to(target)
    with pytest.raises(SamplingDesignError, match="cannot be opened as a file of its own .*never followed"):
        with campaign_lock(root):
            pass  # pragma: no cover - refused
    assert not target.exists()
    campaign_lock_path(root).unlink()
    with campaign_lock(root) as path:
        assert path.is_file() and not path.is_symlink()


def test_units_are_first_attempt_blocks_cut_into_contiguous_near_equal_runs():
    ids = [f"{i:064x}" for i in range(10)]
    assert blocks_per_draw(10, 3) == 4 and blocks_per_draw(10, None) == 1 and blocks_per_draw(9, 3) == 3
    assert block_members(ids, 3, 3) == (ids[9],) and block_members(ids, None, 0) == tuple(ids)
    assert assign_units([0, 1], 4, 3) == [((0, 0), (0, 1), (0, 2)), ((0, 3), (1, 0), (1, 1)), ((1, 2), (1, 3))]
    assert [len(units) for units in assign_units([0, 1], 4, 7)] == [2, 1, 1, 1, 1, 1, 1]
    with pytest.raises(ShardError, match="9 shards for 8 unit"):
        assign_units([0, 1], 4, 9)


def test_shard_names_and_the_block_ids_a_shard_plans():
    assert shard_dirname(3, 20) == "shard-03-of-20" and shard_dirname(2, 7) == "shard-2-of-7"
    assert parse_shard_dirname("shard-03-of-20") == (3, 20)
    assert parse_shard_dirname("shard-3-of-20") is None and parse_shard_dirname("notes") is None
    root = Path("/a/sampling/camp")
    assert shards_root(root) == Path("/a/sampling/camp.shards")
    assert shard_root(root, "m/s", 1, 3) == Path("/a/sampling/camp.shards/m/s/shard-1-of-3")
    assert job_root_override(1, 3) == "job_root=${campaign_root}.shards/${condition_id}/shard-1-of-3"
    assert shard_block_id("c/x/draw-000/attempt-1/block-0000", 1, 3) == "c/x/draw-000/attempt-1/block-0000@shard-1-of-3"


def test_a_draw_shard_and_an_assignment_check_what_they_hold():
    ids = [f"{i:064x}" for i in range(10)]
    shard = DrawShard(1, 3, (1, 3))
    assert shard.scenario_ids(ids, 3) == (*ids[3:6], ids[9]) and shard.label == "shard-1-of-3"
    assert shard.to_dict() == {"index": 1, "count": 3, "request_blocks": [1, 3]}
    with pytest.raises(ShardError, match="holds 4"):
        DrawShard(0, 3, (4,)).scenario_ids(ids, 3)
    for bad in ((3, 1, 3, 3), ("0", 3), (True, 3), (0, 0)):
        with pytest.raises(ShardError, match="shard index|must be an integer"):
            DrawShard(bad[0], bad[1], (0,))
    for blocks in ((), (2, 1), (1, 1), (-1,)):
        with pytest.raises(ShardError, match="distinct, ascending"):
            DrawShard(0, 1, blocks)
    assignment = ShardAssignment.build(1, 3, [0, 1], 4)
    assert assignment.units == ((0, 3), (1, 0), (1, 1)) and assignment.draws == (0, 1)
    assert assignment.draw_shard(1) == DrawShard(1, 3, (0, 1)) and assignment.label == "shard-1-of-3"
    record = assignment.record(campaign_id="c", condition_id="m/s", condition_fingerprint="f")
    assert ShardAssignment.from_record(record) == assignment
    with pytest.raises(ShardError, match="malformed shard record"):
        ShardAssignment.from_record({**record, "units": [[0]]})
    with pytest.raises(ShardError, match="shard index"):
        ShardAssignment.from_record({**record, "shard_index": 5})


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"num_shards": 3}, "set together"),
        ({"num_shards": 0, "shard_index": 0}, "positive integer"),
        ({"num_shards": 3, "shard_index": 3}, r"integer in \[0, 3\)"),
        ({"num_shards": 3, "shard_index": True}, r"integer in \[0, 3\)"),
    ],
)
def test_the_shard_settings_are_checked(block, message):
    with pytest.raises(CampaignRunError, match=message):
        ExecutionSettings.from_mapping(block)
    assert ExecutionSettings.from_mapping({"num_shards": 3, "shard_index": 2}).sharded


def test_a_shard_writes_its_own_root_and_runs_only_its_units(tmp_path, modelspec):
    root = tmp_path / "root"
    plan, completions = run_cfg(shard_cfg(tmp_path, root, 1, 3), modelspec)
    own = shard_root(root, CONDITION, 1, 3)
    assert plan.campaign_root == own and plan.decoder_draw_ids == (0, 1)
    assert not root.exists(), "a shard writes nothing into the campaign root"
    assert not (own / COMPLETION_FNAME).exists(), "a shard writes no campaign gate"
    record = json.loads((own / "shard.json").read_text())
    assert record["units"] == [[0, 3], [1, 0], [1, 1]] and record["plan_draw_ids"] == [0, 1]
    ids = plan.scenario_manifest.scenario_ids
    gate = completions[0]
    assert gate["state"] == STATE_COMPLETE and gate["accepted"] == 1
    assert gate["shard"] == {"index": 1, "count": 3, "request_blocks": [3], "scenarios": 1}
    assert gate["grid"]["expected_keys"] == 1 and gate["expected_scenarios"] == N_SCENARIOS
    blocks = read_jsonl(own / CONDITION / "draw-001" / REQUEST_BLOCKS_FNAME)
    first = [row for row in blocks if row["semantic_attempt_id"] == 0][:2]
    assert [row["request_block_id"] for row in first] == [
        request_block_id(campaign_id="camp", condition_id=CONDITION, decoder_draw_id=1, semantic_attempt_id=0,
                         block_index=index) for index in (0, 1)
    ]
    assert [row["scenario_ids"] for row in first] == [list(ids[0:3]), list(ids[3:6])]
    runs = read_jsonl(own / CONDITION / "draw-000" / RUNS_FNAME)
    assert runs[0]["shard"] == {"index": 1, "count": 3, "request_blocks": [3]}
    build_plan(shard_cfg(tmp_path, root, 1, 3), backend=SeededBackend(), task_spec=task_spec_for(modelspec))
    with pytest.raises(CampaignRunError, match="records another plan"):
        write_shard_record(own, {**record, "units": [[0, 0]]})
    with pytest.raises(CampaignRunError, match=r"records another plan than this job's \(plan_draw_ids, units differ"):
        preflight(shard_cfg(tmp_path, root, 1, 3, "decoder_draw_ids=[1]"))


def test_a_shard_refuses_a_draw_holding_another_shards_scenarios(tmp_path, modelspec):
    root = tmp_path / "root"
    plan, _ = run_cfg(shard_cfg(tmp_path, root, 0, 3), modelspec)
    other = dataclasses.replace(plan, shard=ShardAssignment.build(1, 3, [0, 1], 4))
    backend = SeededBackend()
    with pytest.raises(CampaignRunError, match="outside shard-1-of-3's request blocks"):
        draw_runner(other, 0, backend=backend, task_spec=task_spec_for(modelspec)).run()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("execution.num_shards=3", "execution.shard_index=0"), "must be that root"),
        ((f"job_root={Path('/elsewhere')}",), "is not the campaign root"),
        (("execution.num_shards=9", "execution.shard_index=0", job_root_override(0, 9)), "9 shards for 8 unit"),
        (("execution.num_shards=2", "execution.shard_index=0", job_root_override(0, 2),
          "sampling.campaign_mode=screening", "sampling.num_decoder_draws=1"), "a screening campaign runs"),
        (("execution.num_shards=2", "execution.shard_index=0", job_root_override(0, 2),
          "model@judge_model=gpt-5.6-luna-batch"), "needs a 'local' judge"),
        (("execution.num_shards=2", "execution.shard_index=0", job_root_override(0, 2),
          "execution.allow_stack_change=true"), "allow_stack_change with execution.num_shards"),
    ],
)
def test_preflight_refuses_a_shard_it_could_not_run_or_merge(tmp_path, extra, message):
    with pytest.raises(CampaignRunError, match=message):
        preflight(compose_cfg(tmp_path, tmp_path / "root", *extra))


def test_the_preflight_script_reports_how_many_units_each_shard_runs(tmp_path, capsys, monkeypatch):
    scenarios, rules = write_inputs(tmp_path)
    script = "ruleforge.launch.programs.preflight_campaign"
    common = [
        str(script), "campaign_id=camp", f"campaign_root={tmp_path / 'root'}", f"scenario_set_path={scenarios}",
        "approved_scenario_sha256=null", f"rule_set_path={rules}", "sampling.campaign_mode=pilot",
        "sampling.num_decoder_draws=2", f"sampling.scenarios_per_request_block={PER_BLOCK}",
    ]
    reports = []
    for extra in ([], ["execution.num_shards=3", "execution.shard_index=0", job_root_override(0, 3)]):
        monkeypatch.setattr(sys, "argv", [*common, *extra, f"hydra.run.dir={tmp_path / 'hydra'}"])
        run_program(script)
        reports.append(json.loads(capsys.readouterr().out)["sharding"])
    assert reports == [None, {"num_shards": 3, "request_blocks_per_draw": 4, "units_per_shard": [3, 3, 2]}]
    assert not (tmp_path / "root").exists()


def reorder_population(path: Path) -> None:
    """Rewrite a frozen scenario manifest over the same scenarios in reverse: another population's digest."""
    payload = json.loads(path.read_text())
    other = payload["scenario_ids"][::-1]
    payload.update(scenario_ids=other, manifest_sha256=merge_module.ScenarioManifest.digest(
        other, payload["source_sha256"], payload["schema_version"]))
    path.write_text(json.dumps(payload))


def test_preflight_checks_a_shard_root_against_the_campaign(tmp_path, modelspec):
    root = tmp_path / "root"
    run_cfg(shard_cfg(tmp_path, root, 0, 2), modelspec)
    reorder_population(shard_root(root, CONDITION, 0, 2) / SCENARIO_MANIFEST_FNAME)
    with pytest.raises(CampaignRunError, match="froze a different population"):
        preflight(shard_cfg(tmp_path, root, 0, 2))


def test_a_shard_plan_over_draws_the_campaign_root_holds_is_refused(tmp_path, modelspec):
    root = tmp_path / "root"
    run_cfg(compose_cfg(tmp_path, root, "decoder_draw_ids=[0]"), modelspec)
    # An unsharded job resumes a draw in place; shards would judge it again, and the merge refuse it.
    with pytest.raises(CampaignRunError, match=r"already holds draw\(s\) \[0\] .*decoder_draw_ids=\[1\]"):
        preflight(shard_cfg(tmp_path, root, 0, 3))
    condition_path = root / CONDITION / CONDITION_FNAME
    original = condition_path.read_text()
    edit_json(condition_path, judge_type="Narrow")
    with pytest.raises(CampaignRunError, match="with a different identity"):
        build_plan(shard_cfg(tmp_path, root, 0, 2, "decoder_draw_ids=[1]"), backend=SeededBackend(),
                   task_spec=task_spec_for(modelspec))
    condition_path.write_text(original)
    (root / CONDITION / "draw-001").mkdir()
    with pytest.raises(CampaignRunError, match="nothing is left to shard"):
        preflight(shard_cfg(tmp_path, root, 0, 3))


def test_a_shard_job_refuses_a_layout_whose_run_log_lost_rows_its_gate_counted(tmp_path, sharded):
    runs = shard_dir(sharded, 1) / CONDITION / "draw-000" / RUNS_FNAME
    runs.write_text("")
    with pytest.raises(ShardError, match=r"draw-000: its runs.jsonl holds 0 complete row\(s\).*the code states its runs "
                                         "recorded cannot be read from it"):
        preflight(shard_cfg(tmp_path, sharded, 0, 3))


def test_a_shard_job_refuses_a_code_state_its_layout_did_not_run(tmp_path, sharded):
    runs = shard_dir(sharded, 0) / CONDITION / "draw-000" / RUNS_FNAME
    runs.write_text(runs.read_text() + json.dumps({"recorded_at": NOW, "stack_change": {}}) + "\n")
    (shard_dir(sharded, 0) / CONDITION / "draw-009").mkdir()
    preflight(shard_cfg(tmp_path, sharded, 1, 3))
    edit_jsonl(shard_dir(sharded, 2) / CONDITION / "draw-001" / RUNS_FNAME, 0, code_revision="b" * 40)
    for index in (0, 2):
        with pytest.raises(CampaignRunError, match=r"shard-2-of-3 \(draw-001\) ran commit bbbbbbbbbbbb.*"
                                                   r"--expect-commit b{40}"):
            preflight(shard_cfg(tmp_path, sharded, index, 3))
    preflight(shard_cfg(tmp_path, sharded, 0, 2))


def assert_same_file(path: Path, other: Path, name: str) -> None:
    """``other`` is ``path`` byte for byte."""
    if path.name == COMPLETION_FNAME and path.parent.name.startswith("draw-"):
        gates = [json.loads(each.read_text()) for each in (path, other)]
        assert [gate.pop(RUN_LOG_ROWS_KEY) for gate in gates] == [run_log_rows(path.parent),
                                                                  run_log_rows(other.parent)], name
        assert list(gates[0].items()) == list(gates[1].items()), name
    else:
        assert path.read_bytes() == other.read_bytes(), name


def assert_same_campaign(unsharded: Path, merged: Path) -> None:
    ours, theirs = files_of(unsharded), files_of(merged)
    assert set(ours) == set(theirs)
    for name, path in ours.items():
        if name.startswith(SNAPSHOT_DIRNAME) or name.endswith(RUNS_FNAME):
            continue  # a tarball's gzip header; a log of the processes that ran
        assert_same_file(path, theirs[name], name)
    for name in ours:
        if name.endswith(RUNS_FNAME):
            shard_rows = read_jsonl(theirs[name])
            assert shard_rows and all("shard" in row for row in shard_rows)
            plain = read_jsonl(ours[name])[0]
            assert {key: value for key, value in shard_rows[0].items() if key not in ("shard", "execution")} == plain


@pytest.mark.parametrize(("count", "extra"), [(3, ()), (7, ()), (3, ("execution.pack_draws=true",))])
def test_merged_shards_are_the_unsharded_campaign_byte_for_byte(tmp_path, modelspec, unsharded, count, extra):
    ledger = [row for draw in ("draw-000", "draw-001") for row in read_jsonl(unsharded / CONDITION / draw / LEDGER_FNAME)]
    statuses = {row["status"] for row in ledger}
    assert {"rejected", "transport_failed", "accepted"} <= statuses, "the fake must exercise both retries"
    root = tmp_path / f"sharded-{count}"
    run_shards(tmp_path, modelspec, root, count, *extra)
    report = merge_condition_shards(root, CONDITION)
    assert report["campaign_state"] == STATE_COMPLETE and report["num_shards"] == count
    assert report["condition_state"] == {"0": STATE_COMPLETE, "1": STATE_COMPLETE}
    assert sum(entry["relabelled_rows"] for entry in report["draws"].values()) > 0
    assert_same_campaign(unsharded, root)
    assert analyze_campaign(unsharded) == analyze_campaign(root)
    gate = root / CONDITION / "draw-000" / COMPLETION_FNAME
    gate.write_text(json.dumps({key: value for key, value in json.loads(gate.read_text()).items()
                                if key != RUN_LOG_ROWS_KEY}))
    again = merge_condition_shards(root, CONDITION)
    assert {entry["state"] for entry in again["draws"].values()} == {"already merged"}


def test_component_scope_shards_merge_to_the_unsharded_campaign(tmp_path, modelspec):
    unsharded = tmp_path / "unsharded-component"
    run_cfg(compose_cfg(tmp_path, unsharded, "sampling.retry_scope=component"), modelspec)
    rows = read_jsonl(unsharded / CONDITION / "draw-000" / LEDGER_FNAME)
    assert any(row["status"] == "rejected" for row in rows) and rows[0]["retry_scope"] == "component"
    root = tmp_path / "sharded-component"
    run_shards(tmp_path, modelspec, root, 3, "sampling.retry_scope=component")
    merge_condition_shards(root, CONDITION)
    assert_same_campaign(unsharded, root)
    assert "component_attempts_histogram" in json.loads((root / CONDITION / "draw-000" / COMPLETION_FNAME).read_text())


def test_a_draw_that_exhausts_merges_as_the_unsharded_exhausted_draw(tmp_path, modelspec):
    unsharded = tmp_path / "unsharded-exhausted"
    run_cfg(compose_cfg(tmp_path, unsharded, "sampling.max_semantic_attempts=1"), modelspec)
    root = tmp_path / "sharded-exhausted"
    run_shards(tmp_path, modelspec, root, 3, "sampling.max_semantic_attempts=1")
    gates = {
        (index, draw): json.loads(draw_file(root, index, draw, COMPLETION_FNAME).read_text())["state"]
        for index in range(3) for draw in ("draw-000", "draw-001") if draw_file(root, index, draw, COMPLETION_FNAME).exists()
    }
    assert {STATE_COMPLETE, "incomplete_exhausted"} <= set(gates.values()), "a draw some shards finished whole"
    report = merge_condition_shards(root, CONDITION)
    assert report["condition_state"] == {"0": "incomplete_exhausted", "1": "incomplete_exhausted"}
    assert not (root / CONDITION / "draw-000" / "evaluations.jsonl").exists()
    assert_same_campaign(unsharded, root)


def test_a_root_holding_the_first_draw_takes_the_sharded_rest(tmp_path, modelspec, unsharded):
    root = tmp_path / "partial"
    run_cfg(compose_cfg(tmp_path, root, "decoder_draw_ids=[0]"), modelspec)
    run_shards(tmp_path, modelspec, root, 2, "decoder_draw_ids=[1]")
    report = merge_condition_shards(root, CONDITION)
    assert set(report["draws"]) == {"1"} and report["campaign_state"] == STATE_COMPLETE
    for name, path in files_of(unsharded).items():
        if "draw-001" in name and not name.endswith(RUNS_FNAME):
            assert_same_file(path, root / name, name)


def test_a_shard_that_died_is_refused_then_resumed_and_merged(tmp_path, modelspec, unsharded):
    root = tmp_path / "resumed"
    run_shards(tmp_path, modelspec, root, 3, skip=(1,))
    seen: set[int] = set()
    with pytest.raises(BatchExecutionError):
        run_cfg(shard_cfg(tmp_path, root, 1, 3), modelspec, SeededBackend(fail_after=2, seen=seen))
    with pytest.raises(ShardError, match="shard-1-of-3 has not finished: draw 0 is 'in_progress'|shard-1-of-3 has not run"):
        merge_condition_shards(root, CONDITION)
    run_cfg(shard_cfg(tmp_path, root, 1, 3), modelspec, SeededBackend(seen=seen))
    merge_condition_shards(root, CONDITION)
    for name, path in files_of(unsharded).items():
        if name.endswith((LEDGER_FNAME, RAW_OUTPUTS_FNAME, REQUEST_BLOCKS_FNAME, COMPLETION_FNAME)):
            assert_same_file(path, root / name, name)


def test_merge_shards_merges_a_configured_condition_and_refuses_with_status_2(tmp_path, sharded, monkeypatch, capsys):
    from ruleforge.cli import main
    from ruleforge.cli.records import runs_dir
    from ruleforge.evaluate.configured import MARKER_FNAME

    pulled = tmp_path / "pulled"
    shutil.move(shards_root(sharded), pulled)
    unmarked = tmp_path / "unmarked"
    shutil.copytree(pulled, unmarked)
    (next((unmarked / CONDITION).iterdir()) / MARKER_FNAME).unlink()
    argv = ["campaign", "merge-shards", "--campaign-root", str(sharded), "--condition-id", CONDITION]
    assert main([*argv, "--shards-dir", str(unmarked / CONDITION)]) == 2
    assert f"holds no {MARKER_FNAME} of its own" in capsys.readouterr().err
    assert not sharded.exists() and not runs_dir(sharded).exists()
    assert main([*argv, "--shards-dir", str(pulled / CONDITION)]) == 0
    assert json.loads(capsys.readouterr().out)["campaign_state"] == STATE_COMPLETE
    assert (sharded / MARKER_FNAME).read_bytes() == (next((pulled / CONDITION).iterdir()) / MARKER_FNAME).read_bytes()
    assert len(list(runs_dir(sharded).iterdir())) == 1
    assert main(["campaign", "merge-shards", "--campaign-root", str(sharded), "--condition-id", "x/y"]) == 2
    assert "holds no shard roots" in capsys.readouterr().err


def shard_dir(root: Path, index: int, count: int = 3) -> Path:
    return shard_root(root, CONDITION, index, count)


def edit_json(path: Path, **changes) -> None:
    payload = json.loads(path.read_text())
    payload.update(changes)
    path.write_text(json.dumps(payload))


def edit_jsonl(path: Path, index: int, **changes) -> None:
    rows = read_jsonl(path)
    rows[index].update(changes)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_the_merge_refuses_missing_or_ambiguous_shard_layouts(tmp_path, modelspec, sharded):
    directory = shards_root(sharded) / CONDITION
    (directory / "notes.txt").write_text("not a shard")
    run_cfg(shard_cfg(tmp_path, sharded, 0, 2), modelspec)
    with pytest.raises(ShardError, match="shards of 2 layouts"):
        merge_condition_shards(sharded, CONDITION)
    with pytest.raises(ShardError, match=r"lacks shard\(s\) \[1\] of 2"):
        merge_condition_shards(sharded, CONDITION, num_shards=2)
    assert len(discover_shards(directory, num_shards=3)) == 3
    (shard_dir(sharded, 1) / "shard.json").unlink()
    with pytest.raises(ShardError, match="has no shard.json"):
        discover_shards(directory, num_shards=3)
    shutil.copy(shard_dir(sharded, 2) / "shard.json", shard_dir(sharded, 1) / "shard.json")
    with pytest.raises(ShardError, match="records shard shard-2-of-3, not its directory's"):
        discover_shards(directory, num_shards=3)


def test_the_merge_refuses_overlapping_units_or_another_plan(sharded):
    record = shard_dir(sharded, 2) / "shard.json"
    original = json.loads(record.read_text())
    edit_json(record, units=[[0, 0], [1, 3]])
    with pytest.raises(ShardError, match=r"1 run by several shards \[\(0, 0\)\], 1 by none \[\(1, 2\)\]"):
        merge_condition_shards(sharded, CONDITION)
    edit_json(record, **{**original, "units": [[1, 2], [1, 3], [2, 0]]})
    with pytest.raises(ShardError, match=r"1 outside the plan \[\(2, 0\)\]"):
        merge_condition_shards(sharded, CONDITION)
    edit_json(record, **{**original, "plan_draw_ids": [0, 1, 2]})
    with pytest.raises(ShardError, match="not one plan's shards"):
        merge_condition_shards(sharded, CONDITION)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda root: edit_json(shard_dir(root, 1) / "shard.json", campaign_id="other"), "ran condition"),
        (lambda root: edit_json(shard_dir(root, 1) / SAMPLING_MANIFEST_FNAME, max_semantic_attempts=9),
         "campaign manifest differs"),
        (lambda root: reorder_population(shard_dir(root, 1) / SCENARIO_MANIFEST_FNAME), "another scenario manifest"),
        (lambda root: [edit_json(shard_dir(root, index) / SAMPLING_MANIFEST_FNAME, scenarios_per_request_block=5)
                       for index in range(3)], r"x 4 request block\(s\), but their campaign has draws 0-1 of 2"),
        (lambda root: edit_json(shard_dir(root, 1) / CONDITION / CONDITION_FNAME, prompt_style="forced-grpo"),
         "another condition identity"),
        (lambda root: edit_json(shard_dir(root, 1) / "shard.json", condition_fingerprint="0" * 64),
         "another condition fingerprint"),
        (lambda root: (shard_dir(root, 1) / CONDITION / CONDITION_STACK_FNAME).unlink(), "never started a draw"),
        (lambda root: edit_jsonl(shard_dir(root, 1) / CONDITION / "draw-000" / RUNS_FNAME, 0,
                                 code_state_sha256="0" * 64), r"2 code states \(.*shard-1-of-3 \(draw-000\)"),
        # A run row behind a malformed one hides its code state from every reader; a torn tail does not.
        (lambda root: append_bytes(shard_dir(root, 1) / CONDITION / "draw-000" / RUNS_FNAME, b'{"torn\n{}\n'),
         r"draw-000: its runs.jsonl holds a malformed row after its 1 complete row\(s\), and 1 line\(s\) after it"),
        (lambda root: judged_elsewhere(shard_dir(root, 2) / CONDITION / "draw-001", vllm_version="0.19.0"),
         "another inference stack"),
    ],
)
def test_the_merge_refuses_shards_of_another_campaign_condition_code_or_stack(sharded, tamper, message):
    tamper(sharded)
    with pytest.raises(ShardError, match=message):
        merge_condition_shards(sharded, CONDITION)


def append_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(path.read_bytes() + data)


def edit_runtime(path: Path, **changes) -> None:
    payload = json.loads(path.read_text())
    edit_json(path, runtime={**payload["runtime"], **changes})


def judged_elsewhere(draw_dir: Path, **changes) -> None:
    """A draw whose judgments came from another runtime: its provenance's, and the stack its rows name."""
    edit_runtime(draw_dir / DRAW_PROVENANCE_FNAME, **changes)
    restamp_stacks(draw_dir)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda root: edit_json(root / SAMPLING_MANIFEST_FNAME, temperature=0.5), "describes another campaign"),
        (lambda root: reorder_population(root / SCENARIO_MANIFEST_FNAME), "another population"),
        (lambda root: edit_json(root / CONDITION / CONDITION_FNAME, judge_type="Narrow"), "with another identity"),
        (lambda root: edit_runtime(root / CONDITION / CONDITION_STACK_FNAME, torch_version="2.10"),
         r"condition_stack.json \(draw-000: first run\) holds judgments of the condition on another stack"),
        # A judged draw holds another stack: by its own record, and in a root older than the condition's record.
        (lambda root: judged_elsewhere(root / CONDITION / "draw-000", vllm_version="0.19.0"),
         r"draw-000 holds judgments of the condition on another stack or engine \(vllm 0.19.0 -> 0.28.0\)"),
        (lambda root: [(root / CONDITION / CONDITION_STACK_FNAME).unlink(),
                       judged_elsewhere(root / CONDITION / "draw-000", vllm_version="0.19.0")],
         r"draw-000/draw_provenance.json\) holds judgments of the condition on another stack"),
    ],
)
def test_the_merge_refuses_a_campaign_root_that_disagrees(sharded, tamper, message):
    sharded.mkdir(parents=True)
    for name in (SAMPLING_MANIFEST_FNAME, SCENARIO_MANIFEST_FNAME):
        shutil.copy(shard_dir(sharded, 0) / name, sharded / name)
    shutil.copytree(shard_dir(sharded, 0) / CONDITION, sharded / CONDITION, ignore=shutil.ignore_patterns("draw-001"))
    tamper(sharded)
    with pytest.raises(ShardError, match=message):
        merge_condition_shards(sharded, CONDITION)


def test_the_merge_compares_populations_by_digest_and_replaces_an_unjudged_stack_record(sharded):
    edit_json(shard_dir(sharded, 1) / SCENARIO_MANIFEST_FNAME, source_name="frozen_copy.jsonl")
    (sharded / CONDITION).mkdir(parents=True)
    shutil.copy(shard_dir(sharded, 0) / SCENARIO_MANIFEST_FNAME, sharded / SCENARIO_MANIFEST_FNAME)
    edit_json(sharded / SCENARIO_MANIFEST_FNAME, source_name="another_copy.jsonl")
    # A stack record no judged draw stands behind names no judgments: the shards' replaces it, as a runner's would.
    stack = sharded / CONDITION / CONDITION_STACK_FNAME
    shutil.copy(shard_dir(sharded, 0) / CONDITION / CONDITION_STACK_FNAME, stack)
    edit_runtime(stack, torch_version="2.10")
    assert merge_condition_shards(sharded, CONDITION)["campaign_state"] == STATE_COMPLETE
    assert stack.read_bytes() == (shard_dir(sharded, 0) / CONDITION / CONDITION_STACK_FNAME).read_bytes()


def append_copy_of_first_row(path: Path, **changes) -> None:
    path.write_text(path.read_text() + json.dumps({**read_jsonl(path)[0], **changes}) + "\n")


def relabel_first_row(path: Path, view: str) -> None:
    """Flip every label of the first row of a view: a row that contradicts what the ledger accepted."""
    rows = read_jsonl(path)
    flip = {"yes": "no", "no": "yes"}
    for components in rows[0][view].values():
        for by_judge in components.values():
            for judge, value in by_judge.items():
                if view == "evaluations":
                    by_judge[judge] = flip.get(value, value)
                else:
                    value["judgment"] = flip.get(value["judgment"], value["judgment"])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def draw_file(root: Path, index: int, draw: str, name: str) -> Path:
    return shard_dir(root, index) / CONDITION / draw / name


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda root: draw_file(root, 1, "draw-001", COMPLETION_FNAME).unlink(), "has not run draw 1"),
        (lambda root: edit_json(draw_file(root, 1, "draw-001", COMPLETION_FNAME), shard=None), "is not shard-1-of-3's gate"),
        (lambda root: edit_json(draw_file(root, 1, "draw-001", COMPLETION_FNAME), state="in_progress", pending=2),
         r"shard-1-of-3 has not finished: draw 1 is 'in_progress' with 2 bundle\(s\) pending; resume it"),
        (lambda root: edit_json(draw_file(root, 1, "draw-001", COMPLETION_FNAME), state="incomplete_exhausted"),
         "its ledger is not its units' as its gate says"),
        (lambda root: edit_json(draw_file(root, 1, "draw-001", COMPLETION_FNAME), campaign_id="other"),
         "gates another campaign"),
        (lambda root: draw_file(root, 1, "draw-001", LEDGER_FNAME).write_text(
            draw_file(root, 1, "draw-001", LEDGER_FNAME).read_text().splitlines(keepends=True)[0]),
         "its ledger is not its units'"),
        (lambda root: draw_file(root, 1, "draw-001", "bundle_outcomes.jsonl").write_text(""), "does not hold its units'"),
        (lambda root: append_copy_of_first_row(draw_file(root, 1, "draw-001", "evaluations.jsonl")), "each once"),
        (lambda root: edit_jsonl(draw_file(root, 1, "draw-001", "bundle_outcomes.jsonl"), 0, semantic_attempt_id=9),
         "row is not the bundle its ledger accepted"),
        (lambda root: edit_jsonl(draw_file(root, 1, "draw-001", "bundle_outcomes.jsonl"), 0, full_set_judgment="maybe"),
         "row is not the bundle its ledger accepted"),
        (lambda root: relabel_first_row(draw_file(root, 1, "draw-001", "evaluations.jsonl"), "evaluations"),
         "row is not the bundle its ledger accepted"),
        (lambda root: relabel_first_row(draw_file(root, 1, "draw-001", "judge_outputs.jsonl"), "judge_outputs"),
         "row is not the bundle its ledger accepted"),
        (lambda root: draw_file(root, 1, "draw-001", RAW_OUTPUTS_FNAME).write_text(""), "are not its ledger row's"),
        (lambda root: append_copy_of_first_row(draw_file(root, 1, "draw-001", RAW_OUTPUTS_FNAME)), "two raw outputs"),
        (lambda root: append_copy_of_first_row(draw_file(root, 1, "draw-001", RAW_OUTPUTS_FNAME),
                                               component_id="Rule 999"), r"no component for \['Rule 999'\]"),
        (lambda root: draw_file(root, 2, "draw-001", "implicated_rules.json").write_text("{}"), "different implicated"),
        (lambda root: edit_json(draw_file(root, 2, "draw-001", DRAW_PROVENANCE_FNAME), judge_type="Narrow"),
         "provenance differs"),
        (lambda root: draw_file(root, 2, "draw-001", REQUEST_BLOCKS_FNAME).write_text(""), "does not record"),
        (lambda root: draw_file(root, 2, "draw-001", RAW_OUTPUTS_FNAME).write_text(
            draw_file(root, 2, "draw-001", RAW_OUTPUTS_FNAME).read_text()
            + json.dumps({"scenario_id": "f" * 64, "semantic_attempt_id": 0, "request_block_id": "x"}) + "\n"),
         "no ledger row vouches for"),
    ],
)
def test_the_merge_refuses_an_incomplete_or_inconsistent_shard_draw(sharded, tamper, message):
    tamper(sharded)
    with pytest.raises(ShardError, match=message):
        merge_condition_shards(sharded, CONDITION)
    assert not sharded.exists(), "a refused merge writes nothing into the campaign root"
    assert not list(sharded.parent.glob(f".*{MERGING_SUFFIX}*")), "nor leaves its scratch directory"


def test_the_merge_carries_an_implicated_rules_sidecar_every_shard_wrote(sharded):
    for index in (1, 2):
        draw_file(sharded, index, "draw-001", "implicated_rules.json").write_text('{"implicated_rules": {}}')
    merge_condition_shards(sharded, CONDITION)
    assert (sharded / CONDITION / "draw-001" / "implicated_rules.json").read_text() == '{"implicated_rules": {}}'


@pytest.mark.parametrize(
    "tamper",
    [
        lambda draw: (draw / "extra.txt").write_text("x"),
        lambda draw: edit_json(draw / COMPLETION_FNAME, accepted=0),
        lambda draw: (draw / LEDGER_FNAME).write_text((draw / LEDGER_FNAME).read_text().replace('"no"', '"yes"', 1)),
    ],
)
def test_the_merge_never_replaces_a_draw_the_root_holds(sharded, tamper):
    merge_condition_shards(sharded, CONDITION)
    draw = sharded / CONDITION / "draw-001"
    tamper(draw)
    before = files_of(draw)
    with pytest.raises(ShardError, match=r"already holds draw\(s\) \['draw-001'\]"):
        merge_condition_shards(sharded, CONDITION)
    assert files_of(draw).keys() == before.keys()
    assert not list(sharded.parent.glob(f".*{MERGING_SUFFIX}*"))


def _link_to(target: Path):
    def plant(place: Path) -> None:
        place.parent.mkdir(parents=True, exist_ok=True)
        place.symlink_to(target, target_is_directory=True)
    return plant


def _nested_root(name: str):
    def plant(place: Path) -> None:
        place.mkdir(parents=True)
        (place / name).write_text("{}")
    return plant


@requires_git_worktree  # the shards hold code snapshots, which need git's file list
@pytest.mark.parametrize(
    ("place", "plant", "message"),
    [
        ("gemma-4-31B-it", "link", "is a symlink: the merge would write through it"),
        (CONDITION, "link", "is a symlink: the merge would write through it"),
        (f"{CONDITION}/draw-001", "link", "is a symlink: the merge would write through it"),
        (SNAPSHOT_DIRNAME, "link", "is a symlink: the merge would write through it"),
        ("gemma-4-31B-it", SAMPLING_MANIFEST_FNAME, "is a campaign root of its own inside"),
        (CONDITION, MARKER_FNAME, "is a campaign root of its own inside"),
    ],
)
def test_the_merge_never_writes_through_a_link_or_into_a_root_below_the_campaign_root(sharded, tmp_path, place, plant,
                                                                                         message):
    assert list((shard_dir(sharded, 0) / SNAPSHOT_DIRNAME).glob("*.tar.gz")), "the shards hold code snapshots"
    elsewhere = tmp_path / "published"
    elsewhere.mkdir()
    (_link_to(elsewhere) if plant == "link" else _nested_root(plant))(sharded / place)
    before = files_of(sharded)
    with pytest.raises(ShardError, match=f"{sharded / place} {message}"):
        merge_condition_shards(sharded, CONDITION)
    assert files_of(sharded) == before and list(elsewhere.iterdir()) == []
    assert not list(sharded.parent.glob(f".*{MERGING_SUFFIX}*"))


def _bytes_of(directory: Path) -> dict[str, bytes]:
    return {name: path.read_bytes() for name, path in files_of(directory).items()}


def test_the_merge_never_writes_a_pinned_root_or_one_inside_another_campaign_root(sharded, tmp_path, monkeypatch,
                                                                                  capsys):
    from ruleforge.evaluate import api

    merge_condition_shards(sharded, CONDITION)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://sharded": ("leniency",)})
    before, beside = _bytes_of(sharded), sorted(path.name for path in tmp_path.iterdir())
    with pytest.raises(api.CampaignError, match=f"{sharded} \\(artifact://sharded\\) is artifact://sharded, or on its "
                                                "path, a published run of leniency"):
        merge_condition_shards(sharded, CONDITION)
    from ruleforge.cli import main

    assert main(["campaign", "merge-shards", "--campaign-root", str(sharded), "--condition-id", CONDITION]) == 2
    assert "is artifact://sharded, or on its path" in capsys.readouterr().err
    assert _bytes_of(sharded) == before and sorted(path.name for path in tmp_path.iterdir()) == beside
    monkeypatch.setattr(api, "pinned_roots", dict)
    inner = sharded / "inner"
    with pytest.raises(api.CampaignError, match=f"the campaign root a shard merge writes {inner} is inside the "
                                                f"campaign root {sharded}"):
        merge_condition_shards(inner, CONDITION, shards_dir=shards_root(sharded) / CONDITION)
    assert _bytes_of(sharded) == before and not inner.exists()
