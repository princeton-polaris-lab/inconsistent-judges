from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path

import pytest

import ruleforge.optimize  # noqa: F401 -- imported (subclassing ModelBackend) before a test stubs the program's backend
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.campaign import (
    build_plan,
    draw_runner,
    execution_settings,
    preflight,
    run_plan,
    run_plans_packed,
)
from ruleforge.evaluate.identity import (
    COMPLETION_FNAME,
    RUN_LOG_ROWS_KEY,
    SamplingConfig,
    differing_manifest_fields,
    run_log_rows,
)
from ruleforge.judges.weights import adapter_identity
from ruleforge.evaluate.packing import (
    DEFAULT_PACK_MAX_PROMPTS,
    ExecutionSettings,
    check_packable,
    pack_blocks,
    run_packed,
)
from ruleforge.evaluate.runner import (
    REQUEST_BLOCKS_FNAME,
    STATE_COMPLETE,
    STATE_IN_PROGRESS,
    CampaignRunError,
    DrawRunner,
    DrawSpec,
    TransportExhausted,
    draw_lock_path,
)
from ruleforge.judges.backend import GenerationOutcome
from tests.support.hydra_runs import run_program
from tests.support.trees import tree_files
from tests.test_sampling_campaign import compose_campaign
from tests.test_sampling_ruleset_size import compose_ruleset_size, write_subsets
from tests.test_sampling_training import write_adapter
from tests.test_sampling_runner import RULES, FakeBackend, _hold, make_spec, read_jsonl, write_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]
_RULESET_RE = re.compile(r"# Rule Set Used for Evaluation\n(.+?)\n\n", re.S)
#: Keys whose values are wall-clock or process facts, never results.
_VOLATILE = {"started_at", "completed_at", "recorded_at", "updated_at", "created_at", "runtime"}


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))


@pytest.fixture(scope="module")
def driver():
    """The shared-bank job's code (``ruleforge.evaluate.bank``, which ``ruleforge bank run`` runs)."""
    import ruleforge.evaluate.bank

    return ruleforge.evaluate.bank


class SeededBackend(FakeBackend):
    """A local judge whose answer is a function of (prompt, seed) alone."""

    def __init__(self, *, failing_calls=0, lost=None, crash_on_call=None, **kwargs):
        super().__init__(**kwargs)
        self.max_new_tokens = 1024
        self.failing_calls = failing_calls
        self.lost = dict(lost or {})
        self.crash_on_call = crash_on_call
        self.prompt_counts: list[int] = []

    @staticmethod
    def answer(entry, seed):
        roll = seed % 8
        label = "maybe" if roll == 0 else ("yes" if roll in (1, 2, 3) else "no")
        prompt = entry["system_prompt"]
        text = label
        if '"full_set_judgment"' in prompt:
            rules = re.findall(r"^(Rule \d+):", _RULESET_RE.search(prompt).group(1), re.M)
            text = json.dumps({
                "rule_judgments": {rule: ("yes" if roll == 1 else "no") for rule in rules},
                "preempted_rules": [],
                "full_set_judgment": label,
            })
        return GenerationOutcome(text=text, raw_response=f"<think>{seed}</think>{text}", finish_reason="stop")

    def generate_once(self, prompts, *, temperature, max_new_tokens=None, seeds=None, request_block_id="", truncation=None):
        assert temperature == 1.0 and seeds is not None and len(seeds) == len(prompts)
        self.blocks.append((request_block_id, len(prompts), list(seeds)))
        self.truncations.append(truncation)
        self.prompt_counts.append(len(prompts))
        if self.crash_on_call is not None and len(self.blocks) == self.crash_on_call:
            raise RuntimeError("engine died")
        if self.failing_calls > 0:
            self.failing_calls -= 1
            return [
                GenerationOutcome(text=None, raw_response=None, finish_reason=None,
                                  error="RuntimeError: engine died", block_failed=True)
                for _ in prompts
            ]
        outcomes = []
        for entry, seed in zip(prompts, seeds):
            if self.lost.get(seed, 0) > 0:
                self.lost[seed] -= 1
                outcomes.append(GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="HTTP 500"))
            else:
                outcomes.append(self.answer(entry, seed))
        return outcomes


def normalized(value):
    """``value`` with every wall-clock and process key removed, recursively."""
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items() if key not in _VOLATILE}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def draw_tree(root: Path, *, skip=("runs.jsonl",)) -> dict[str, object]:
    """Every file under the root's condition directories, parsed and normalised."""
    tree = {}
    for path in tree_files(root):
        relative = path.relative_to(root)
        if relative.parts[0] in ("hydra", "code_snapshots", "aggregate") or len(relative.parts) == 1:
            continue
        if path.name in skip:
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            tree[relative.as_posix()] = [normalized(json.loads(line)) for line in text.splitlines()]
        else:
            tree[relative.as_posix()] = normalized(json.loads(text))
            if path.name == COMPLETION_FNAME and "runs.jsonl" in skip:
                # A gate counts its draw's run log, as the processes that ran it left it: skipped too.
                assert tree[relative.as_posix()].pop(RUN_LOG_ROWS_KEY) == run_log_rows(path.parent)
    return tree


def ruleset_size_plans(tmp_path, modelspec, driver, backend, root_name, **overrides):
    """Every condition of a tiny ruleset-size design (a bank and six full-only subsets), planned."""
    if not (tmp_path / "rules_full.jsonl").exists():
        write_subsets(tmp_path)
    settings = {
        "campaign_root": str(tmp_path / root_name),
        "sampling.max_semantic_attempts": 20,
        "sampling.scenarios_per_request_block": 2,
        **overrides,
    }
    cfg = compose_ruleset_size(tmp_path, tmp_path / "rules_full.jsonl", tmp_path / "subsets", tmp_path / "manifest", **settings)
    design, planned, _ = driver.plan_job(cfg)
    root = Path(str(cfg.campaign_root))
    root.mkdir(parents=True, exist_ok=True)
    driver.write_design(root, design)
    task_spec = modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0)
    plans = [build_plan(driver.condition_cfg(cfg, item), backend=backend, task_spec=task_spec) for item in planned]
    return root, plans, task_spec


def style_runners(tmp_path, modelspec, backend, root_name, styles=("naive", "forced-grpo"), **overrides):
    """Both-shaped draws of one judge under several prompt styles, as runners."""
    runners = []
    for style in styles:
        cfg = compose_campaign(
            tmp_path,
            campaign_root=str(tmp_path / root_name),
            **{
                "eval_cfg.reasoning_prompt_style": style,
                "sampling.max_semantic_attempts": 20,
                "sampling.scenarios_per_request_block": 2,
                **overrides,
            },
        )
        task_spec = modelspec(reasoning_prompt_style=style, evaluation_temperature=1.0)
        plan = build_plan(cfg, backend=backend, task_spec=task_spec)
        runners.extend(draw_runner(plan, draw, backend=backend, task_spec=task_spec) for draw in plan.decoder_draw_ids)
    return runners


def test_blocks_are_packed_in_order_never_split_and_an_oversized_one_goes_alone():
    assert pack_blocks([3, 3, 3], 6) == [[0, 1], [2]]
    assert pack_blocks([2, 9, 1, 1], 4) == [[0], [1], [2, 3]]
    assert pack_blocks([5], 4) == [[0]]
    assert pack_blocks([], 4) == []
    for bad in (0, -1, True, 2.0):
        with pytest.raises(CampaignRunError, match="positive integer"):
            pack_blocks([1], bad)


def test_execution_settings_default_validate_and_refuse_unknown_keys():
    assert ExecutionSettings.from_mapping(None) == ExecutionSettings(False, DEFAULT_PACK_MAX_PROMPTS, None)
    assert ExecutionSettings.from_mapping({"pack_draws": True, "pack_max_draws": 3}).pack_max_draws == 3
    assert ExecutionSettings.from_mapping({"allow_stack_change": True}).allow_stack_change is True
    with pytest.raises(CampaignRunError, match="execution.allow_stack_change must be true or false"):
        ExecutionSettings(allow_stack_change="yes")
    with pytest.raises(CampaignRunError, match="unknown execution keys"):
        ExecutionSettings.from_mapping({"pack_draw": True})
    with pytest.raises(CampaignRunError, match="must be a mapping"):
        ExecutionSettings.from_mapping(["pack_draws"])
    with pytest.raises(CampaignRunError, match="true or false"):
        ExecutionSettings(pack_draws="yes")
    with pytest.raises(CampaignRunError, match="pack_max_prompts"):
        ExecutionSettings(pack_max_prompts=0)
    with pytest.raises(CampaignRunError, match="pack_max_draws"):
        ExecutionSettings(pack_max_draws=True)


def test_the_config_knob_is_read_validated_and_never_identity(tmp_path, modelspec):
    from omegaconf import OmegaConf

    cfg = compose_campaign(tmp_path)
    assert execution_settings(cfg) == ExecutionSettings()
    OmegaConf.set_struct(cfg, False)
    del cfg["execution"]
    assert execution_settings(cfg) == ExecutionSettings()
    cfg.execution = 5
    with pytest.raises(CampaignRunError, match="must be a mapping"):
        execution_settings(cfg)
    with pytest.raises(CampaignRunError, match="unknown execution keys"):
        preflight(compose_campaign(tmp_path, **{"+execution.pack_everything": "true"}))
    with pytest.raises(CampaignRunError, match="needs a 'local' judge"):
        preflight(compose_campaign(tmp_path, **{"model@judge_model": "gpt-5.6-luna-batch", "execution.pack_draws": "true"}))

    task_spec = modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0)
    backend = SeededBackend()
    build_plan(compose_campaign(tmp_path, campaign_root=str(tmp_path / "off")), backend=backend, task_spec=task_spec)
    packed_cfg = compose_campaign(
        tmp_path, campaign_root=str(tmp_path / "on"), **{"execution.pack_draws": "true", "execution.pack_max_prompts": 7}
    )
    assert execution_settings(packed_cfg) == ExecutionSettings(True, 7, None)
    build_plan(packed_cfg, backend=backend, task_spec=task_spec)
    off = json.loads((tmp_path / "off" / SAMPLING_MANIFEST_FNAME).read_text())
    on = json.loads((tmp_path / "on" / SAMPLING_MANIFEST_FNAME).read_text())
    assert differing_manifest_fields(off, on) == [] and "execution" not in on
    assert SamplingConfig.from_mapping(OmegaConf.to_container(packed_cfg.sampling, resolve=True)) == SamplingConfig.from_mapping(
        OmegaConf.to_container(compose_campaign(tmp_path).sampling, resolve=True)
    )
    preflight(compose_campaign(tmp_path, campaign_root=str(tmp_path / "off"), **{"execution.pack_draws": "true"}))


def _runner(tmp_path, inputs, modelspec, backend, *, draw=0, name="d", **condition_overrides):
    manifest, rows = inputs
    spec = make_spec(manifest, draw=draw, seed_capability=backend.seed_capability)
    if condition_overrides:
        spec = DrawSpec(**{**spec.__dict__, "condition": dataclasses.replace(spec.condition, **condition_overrides)})
    return DrawRunner(
        backend=backend, task_spec=modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0),
        spec=spec, draw_dir=tmp_path / name / f"draw-{draw:03d}", scenario_rows=rows, rules=RULES,
    )


@pytest.fixture
def inputs(tmp_path):
    from ruleforge.scenarios.manifest import ScenarioManifest

    path = write_inputs(tmp_path)
    return ScenarioManifest.build(path), [json.loads(line) for line in path.read_text().splitlines()]


def test_runners_one_call_cannot_serve_are_refused(tmp_path, inputs, modelspec):
    backend = SeededBackend()
    first = _runner(tmp_path, inputs, modelspec, backend)
    assert run_packed([]) == []
    with pytest.raises(CampaignRunError, match="positive integer"):
        run_packed([first], max_prompts_per_call=0)
    with pytest.raises(CampaignRunError, match="one backend instance"):
        check_packable([first, _runner(tmp_path, inputs, modelspec, SeededBackend(), draw=1)])
    with pytest.raises(CampaignRunError, match="one truncation"):
        check_packable([first, _runner(tmp_path, inputs, modelspec, backend, draw=1, sampling_truncation={"top_p": 0.95})])
    with pytest.raises(CampaignRunError, match="one ledger has one writer"):
        check_packable([first, _runner(tmp_path, inputs, modelspec, backend)])
    other = _runner(tmp_path, inputs, modelspec, backend, draw=1)
    hot = dataclasses.replace(other.spec.config)
    object.__setattr__(hot, "temperature", 0.7)
    other.spec = DrawSpec(**{**other.spec.__dict__, "config": hot})
    with pytest.raises(CampaignRunError, match="one temperature"):
        check_packable([first, other])

    batch = FakeBackend(model_type="openai_batch", seed_capability="unsupported")
    with pytest.raises(CampaignRunError, match="serves a 'local' judge only"):
        run_packed([_runner(tmp_path, inputs, modelspec, batch, sampling_truncation="provider_default")])
    windowed = SeededBackend()
    windowed.max_enqueued_tokens = 10
    windowed._poll_request_block = lambda *a, **k: None
    with pytest.raises(CampaignRunError, match="windowed"):
        run_packed([_runner(tmp_path, inputs, modelspec, windowed)])
    assert backend.blocks == [] and not (tmp_path / "d").exists()


def test_a_packed_run_holds_every_draw_s_lock_and_is_refused_before_any_draw_is_prepared(tmp_path, inputs, modelspec):
    import os

    backend = SeededBackend()
    runners = [_runner(tmp_path, inputs, modelspec, backend, draw=draw) for draw in (0, 1)]
    held = draw_lock_path(runners[1].draw_dir)
    held.parent.mkdir(parents=True)
    descriptor = _hold(held)
    try:
        with pytest.raises(CampaignRunError, match=r"another process holds .*\.draw-001\.lock"):
            run_packed(runners)
        assert backend.blocks == [] and not any(runner.draw_dir.exists() for runner in runners)
        assert not draw_lock_path(runners[0].draw_dir).exists()
    finally:
        os.close(descriptor)
    assert [result["state"] for result in run_packed(runners)] == [STATE_COMPLETE] * 2
    assert not any(draw_lock_path(runner.draw_dir).exists() for runner in runners)


def test_packed_draws_share_one_adapter_checked_before_anything_is_generated(tmp_path, inputs, modelspec):
    adapter = write_adapter(tmp_path / "adapters", "checkpoint-125")
    identity = adapter_identity(adapter)
    named = {"adapter_path": identity.label, "adapter_sha256": identity.sha256,
             "adapter_config_sha256": identity.config_sha256}
    other = adapter_identity(write_adapter(tmp_path / "adapters", "checkpoint-200"))
    backend = SeededBackend()
    backend.adapter_path = adapter

    mixed = [
        _runner(tmp_path, inputs, modelspec, backend, name="a", **named),
        _runner(tmp_path, inputs, modelspec, backend, name="b", adapter_path=other.label,
                adapter_sha256=other.sha256, adapter_config_sha256=other.config_sha256),
    ]
    with pytest.raises(CampaignRunError, match="one backend serves one adapter") as refused:
        run_packed(mixed)
    # Both digests of both sides: two conditions can differ in either.
    for digest in (identity.sha256, identity.config_sha256, other.sha256, other.config_sha256):
        assert digest in str(refused.value)
    assert backend.blocks == [] and not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    reconfigured = dict(named, adapter_config_sha256="c" * 64)
    with pytest.raises(CampaignRunError, match=r"config SHA-256 c{64}\).*one backend serves one adapter"):
        check_packable([mixed[0], _runner(tmp_path, inputs, modelspec, backend, name="r", **reconfigured)])

    impostor = SeededBackend()
    impostor.adapter_path = write_adapter(tmp_path / "elsewhere", "checkpoint-125", content=b"other weights")
    with pytest.raises(CampaignRunError, match="refusing to add judgments"):
        run_packed([_runner(tmp_path, inputs, modelspec, impostor, draw=d, name="c", **named) for d in (0, 1)])
    assert impostor.blocks == []

    runners = [_runner(tmp_path, inputs, modelspec, backend, draw=d, name="ok", **named) for d in (0, 1)]
    assert [c["state"] for c in run_packed(runners)] == [STATE_COMPLETE, STATE_COMPLETE]
    assert backend.blocks[0][0].startswith("packed:2-blocks:")
    for runner in runners:
        provenance = json.loads((runner.draw_dir / "draw_provenance.json").read_text())
        assert provenance["adapter"]["sha256"] == identity.sha256


def test_a_judge_returning_the_wrong_number_of_outcomes_is_refused(tmp_path, inputs, modelspec):
    class Short(SeededBackend):
        def generate_once(self, prompts, **kwargs):
            return super().generate_once(prompts, **kwargs)[:-1]

    backend = Short()
    runner = _runner(tmp_path, inputs, modelspec, backend)
    with pytest.raises(CampaignRunError, match="outcomes for"):
        run_packed([runner])
    state = runner.ledger.load()
    request = runner.build_request(runner.plan_pass(state, 0)[0])
    with pytest.raises(CampaignRunError, match="outcomes for"):
        runner.record(state, request, [], started_at="t", completed_at="t")


@pytest.mark.parametrize("retry_scope", ["scenario_bundle", "component"])
def test_packed_bank_and_full_draws_match_the_sequential_path(tmp_path, modelspec, driver, retry_scope):
    scope = {"sampling.retry_scope": retry_scope}
    sequential = SeededBackend()
    root_a, plans_a, task_spec = ruleset_size_plans(tmp_path, modelspec, driver, sequential, "seq", **scope)
    for plan in plans_a:
        run_plan(plan, backend=sequential, task_spec=task_spec)
    packed = SeededBackend()
    root_b, plans_b, _ = ruleset_size_plans(tmp_path, modelspec, driver, packed, "packed", **scope)
    execution = ExecutionSettings(pack_draws=True, pack_max_prompts=7)
    completions = run_plans_packed(plans_b, backend=packed, task_spec=task_spec, execution=execution)

    assert set(completions) == {plan.condition.condition_id for plan in plans_b}
    assert all(c["state"] == STATE_COMPLETE for done in completions.values() for c in done.values())
    assert draw_tree(root_a) == draw_tree(root_b)
    def asked(backend):
        return sorted(seed for _, _, seeds in backend.blocks for seed in seeds)

    assert asked(sequential) == asked(packed)
    assert len(packed.blocks) < len(sequential.blocks) / 2
    assert all(n <= 7 or n == 8 for n in packed.prompt_counts) and 8 in packed.prompt_counts
    ledger = read_jsonl(root_b / "gemma-4-31B-it" / "bank" / "draw-000" / "attempts.jsonl")
    assert any(row["status"] == "rejected" for row in ledger) and any(row["semantic_attempt_id"] > 0 for row in ledger)
    retried = [len(row["components"]) for row in ledger if row["semantic_attempt_id"] > 0]
    assert (max(retried) < 4) if retry_scope == "component" else (set(retried) == {4})


@pytest.mark.parametrize("retry_scope", ["scenario_bundle", "component"])
def test_packed_both_shaped_draws_of_two_styles_match_the_sequential_path(tmp_path, modelspec, retry_scope):
    scope = {"sampling.retry_scope": retry_scope}
    sequential = SeededBackend()
    for runner in style_runners(tmp_path, modelspec, sequential, "seq", **scope):
        assert runner.run()["state"] == STATE_COMPLETE
    packed = SeededBackend()
    runners = style_runners(tmp_path, modelspec, packed, "packed", **scope)
    completions = run_packed(runners, max_prompts_per_call=1000)
    assert [c["state"] for c in completions] == [STATE_COMPLETE] * 4
    assert [c["decoder_draw_id"] for c in completions] == [0, 1, 0, 1]
    assert draw_tree(tmp_path / "seq") == draw_tree(tmp_path / "packed")
    assert packed.prompt_counts[0] == 4 * 3 * 3
    assert all(t == sequential.truncations[0] for t in packed.truncations)
    for runner in runners:
        row = json.loads((runner.draw_dir / "runs.jsonl").read_text().splitlines()[-1])
        assert row["execution"] == {"pack_draws": True, "pack_max_prompts": 1000, "packed_draws": 4}
    sequential_rows = [json.loads(line) for path in (tmp_path / "seq").rglob("runs.jsonl") for line in path.read_text().splitlines()]
    assert len(sequential_rows) == 4 and not any("execution" in row for row in sequential_rows)
    again = SeededBackend()
    assert [c["state"] for c in run_packed(style_runners(tmp_path, modelspec, again, "packed", **scope))] == [STATE_COMPLETE] * 4
    assert again.blocks == []
    retried = [
        len(row["components"]) for runner in runners for row in read_jsonl(runner.draw_dir / "attempts.jsonl")
        if row["semantic_attempt_id"] > 0
    ]
    assert retried and ((max(retried) < 3) if retry_scope == "component" else (set(retried) == {3}))


def test_run_plan_packs_a_conditions_draws_in_chunks_and_writes_the_gate(tmp_path, modelspec):
    task_spec = modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0)
    backend = SeededBackend()
    cfg = compose_campaign(tmp_path, **{"sampling.max_semantic_attempts": 20, "execution.pack_draws": "true", "execution.pack_max_draws": 1})
    plan = build_plan(cfg, backend=backend, task_spec=task_spec)
    completions = run_plan(plan, backend=backend, task_spec=task_spec, execution=execution_settings(cfg))
    assert sorted(completions) == [0, 1] and all(c["state"] == STATE_COMPLETE for c in completions.values())
    assert json.loads((plan.campaign_root / COMPLETION_FNAME).read_text())["state"] == STATE_COMPLETE
    draws = [label.split("/draw-")[1][:3] for label, _, _ in backend.blocks]
    assert draws == sorted(draws) and set(draws) == {"000", "001"}
    reference = SeededBackend()
    seq_cfg = compose_campaign(tmp_path, campaign_root=str(tmp_path / "seq"), **{"sampling.max_semantic_attempts": 20})
    seq_plan = build_plan(seq_cfg, backend=reference, task_spec=task_spec)
    run_plan(seq_plan, backend=reference, task_spec=task_spec, execution=ExecutionSettings())
    assert draw_tree(tmp_path / "root") == draw_tree(tmp_path / "seq")
    assert run_plans_packed([], backend=backend, task_spec=task_spec, execution=ExecutionSettings(True)) == {}


def test_every_packed_call_logs_its_label_and_member_blocks(tmp_path, modelspec, caplog):
    backend = SeededBackend()
    runners = style_runners(tmp_path, modelspec, backend, "root", styles=("naive",))
    with caplog.at_level(logging.INFO, logger="ruleforge.evaluate.packing"):
        run_packed(runners)
    logged = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Packed call ")]
    assert len(logged) == len(backend.blocks)
    for message, (label, n_prompts, _) in zip(logged, backend.blocks):
        assert message.startswith(f"Packed call {label}: {n_prompts} prompt(s) from block(s) ")
    first_pass = [
        row["request_block_id"]
        for runner in runners
        for row in map(json.loads, (runner.draw_dir / "request_blocks.jsonl").read_text().splitlines())
        if row["semantic_attempt_id"] == 0
    ]
    assert len(first_pass) == 4 and logged[0].split(" from block(s) ")[1].split(", ") == first_pass


def test_a_call_that_raises_whole_is_retried_with_the_same_packing(tmp_path, modelspec):
    backend = SeededBackend(failing_calls=1)
    runners = style_runners(tmp_path, modelspec, backend, "root", styles=("naive",))
    assert [c["state"] for c in run_packed(runners)] == [STATE_COMPLETE] * 2
    assert backend.blocks[0] == backend.blocks[1]
    rows = [json.loads(line) for line in (runners[0].draw_dir / "attempts.jsonl").read_text().splitlines()]
    assert not any(row["status"] == "transport_failed" for row in rows)


def test_a_judge_that_keeps_raising_exhausts_the_cap_and_every_draw_stays_resumable(tmp_path, modelspec):
    dead = SeededBackend(failing_calls=99)
    runners = style_runners(tmp_path, modelspec, dead, "root", styles=("naive",))
    with pytest.raises(TransportExhausted, match="raised 3 time"):
        run_packed(runners)
    assert len(dead.blocks) == 3
    for runner in runners:
        assert json.loads((runner.draw_dir / COMPLETION_FNAME).read_text())["state"] == STATE_IN_PROGRESS
        assert not runner.ledger.ledger_path.exists()
    recovered = SeededBackend()
    for runner in style_runners(tmp_path, modelspec, recovered, "root", styles=("naive",)):
        assert runner.run()["state"] == STATE_COMPLETE
    clean = SeededBackend()
    for runner in style_runners(tmp_path, modelspec, clean, "clean", styles=("naive",)):
        runner.run()
    assert draw_tree(tmp_path / "root") == draw_tree(tmp_path / "clean")


def test_a_lost_answer_is_a_transport_failure_regenerated_at_the_same_attempt(tmp_path, modelspec):
    probe = style_runners(tmp_path, modelspec, SeededBackend(), "probe", styles=("naive",))[0]
    sid = next(
        sid for sid in probe.spec.scenario_manifest.scenario_ids
        if all(probe.component_seed(sid, 0, c) % 8 for c in ("full_ruleset", "Rule 2"))
    )
    seed = probe.component_seed(sid, 0, "Rule 1")
    backend = SeededBackend(lost={seed: 1})
    runners = style_runners(tmp_path, modelspec, backend, "root", styles=("naive",))
    completions = run_packed(runners)
    assert completions[0]["transport_failed_attempts"] == 1 and completions[0]["state"] == STATE_COMPLETE
    rows = [json.loads(line) for line in (runners[0].draw_dir / "attempts.jsonl").read_text().splitlines()]
    mine = [(row["status"], row["semantic_attempt_id"]) for row in rows if row["scenario_id"] == sid]
    assert mine[0] == ("transport_failed", 0) and mine[1][1] == 0
    blocks = [json.loads(line) for line in (runners[0].draw_dir / REQUEST_BLOCKS_FNAME).read_text().splitlines()]
    assert [b["scenario_ids"] for b in blocks if b["semantic_attempt_id"] == 0][-1] == [sid]
    stubborn = SeededBackend(lost={seed: 99})
    again = style_runners(tmp_path, modelspec, stubborn, "stuck", styles=("naive",))
    with pytest.raises(TransportExhausted, match="failed in transport 3 time"):
        run_packed(again)


def test_a_crash_between_calls_resumes_under_the_sequential_runner(tmp_path, modelspec):
    crashing = SeededBackend(crash_on_call=2)
    runners = style_runners(tmp_path, modelspec, crashing, "root")
    with pytest.raises(RuntimeError, match="engine died"):
        run_packed(runners, max_prompts_per_call=9)
    assert sum(len(r.ledger.load().records) for r in runners) == 3
    assert all((r.draw_dir / REQUEST_BLOCKS_FNAME).exists() for r in runners)
    recovered = SeededBackend()
    for runner in style_runners(tmp_path, modelspec, recovered, "root"):
        assert runner.run()["state"] == STATE_COMPLETE
    clean = SeededBackend()
    for runner in style_runners(tmp_path, modelspec, clean, "clean"):
        runner.run()
    assert draw_tree(tmp_path / "root") == draw_tree(tmp_path / "clean")
    dying = SeededBackend(crash_on_call=2)
    first = style_runners(tmp_path, modelspec, dying, "other")
    with pytest.raises(RuntimeError):
        first[0].run()
    run_packed(style_runners(tmp_path, modelspec, SeededBackend(), "other"))
    assert draw_tree(tmp_path / "other") == draw_tree(tmp_path / "clean")


def test_a_seedless_local_judge_is_packed_without_seeds(tmp_path, inputs, modelspec):
    backend = FakeBackend(seed_capability="unsupported")
    runners = [_runner(tmp_path, inputs, modelspec, backend, draw=draw) for draw in (0, 1)]
    assert [c["state"] for c in run_packed(runners)] == [STATE_COMPLETE] * 2
    assert [(n, seeds) for _, n, seeds in backend.blocks] == [(2 * 3 * 3, None)]
    record = runners[0].ledger.load().accepted[inputs[0].scenario_ids[0]]
    assert all(c.generation_seed is None and c.seed_sent is False for c in record.components)


def test_the_run_script_packs_a_conditions_draws(tmp_path, monkeypatch):
    import sys

    import ruleforge.judges.backend as backend_module

    scenarios = write_inputs(tmp_path)
    rules = tmp_path / "rules.jsonl"
    rules.write_text("".join(json.dumps(r) + "\n" for r in RULES), encoding="utf-8")
    script = "ruleforge.launch.programs.run_campaign"
    backends = []

    def fake_backend(model_cfg):
        backend = SeededBackend()
        backend.llm = None
        backends.append(backend)
        return backend

    monkeypatch.setattr(backend_module, "ModelBackend", fake_backend)
    monkeypatch.setattr(sys, "argv", [
        str(script), "campaign_id=camp", f"campaign_root={tmp_path / 'root'}", f"scenario_set_path={scenarios}",
        "approved_scenario_sha256=null", f"rule_set_path={rules}", "sampling.campaign_mode=pilot",
        "sampling.num_decoder_draws=2", "sampling.max_semantic_attempts=20", "execution.pack_draws=true",
        f"hydra.run.dir={tmp_path / 'hydra'}",
    ])
    run_program(script)
    assert json.loads((tmp_path / "root" / COMPLETION_FNAME).read_text())["state"] == STATE_COMPLETE
    # Both draws' attempt-0 blocks rode the first call.
    assert backends[0].prompt_counts[0] == 2 * 3 * 3


def test_the_stack_override_reaches_every_draw_packed_or_not(tmp_path, modelspec, monkeypatch):
    import ruleforge.evaluate.campaign as campaign_module

    seen = []
    real = campaign_module.draw_runner

    def _spy(*args, **kwargs):
        seen.append(kwargs["allow_stack_change"])
        return real(*args, **kwargs)

    monkeypatch.setattr(campaign_module, "draw_runner", _spy)
    task_spec = modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0)
    for name, execution, expected in [
        ("none", None, False),
        ("seq", ExecutionSettings(allow_stack_change=True), True),
        ("packed", ExecutionSettings(pack_draws=True, allow_stack_change=True), True),
    ]:
        seen.clear()
        backend = SeededBackend()
        cfg = compose_campaign(tmp_path, campaign_root=str(tmp_path / name), **{"sampling.max_semantic_attempts": 20})
        plan = build_plan(cfg, backend=backend, task_spec=task_spec)
        run_plan(plan, backend=backend, task_spec=task_spec, execution=execution)
        assert seen == [expected, expected], name
