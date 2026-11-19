from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import ruleforge.optimize  # noqa: F401 -- imported (subclassing ModelBackend) before a test stubs the program's backend
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate.aggregate import analyze_campaign, style_families
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    SCENARIO_MANIFEST_FNAME,
    build_plan,
    campaign_completion,
    load_conditions,
    register_condition,
    require_complete_campaign,
    run_plan,
    validate_campaign_cfg,
)
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.ledger import LEDGER_FNAME, DrawLedger, LedgerError
from ruleforge.evaluate.identity import COMPLETION_FNAME
from ruleforge.evaluate.runner import (
    STATE_COMPLETE,
    STATE_INCOMPLETE_EXHAUSTED,
    STATE_INVALID_MANIFEST,
    STATE_IN_PROGRESS,
    CampaignRunError,
    DrawRunner,
)
from tests.support.hydra_runs import run_program
from tests.test_sampling_runner import RULES, FakeBackend, forced, restamp_stacks, write_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]


def compose_campaign(tmp_path: Path, **overrides):
    scenarios = write_inputs(tmp_path)
    rules = tmp_path / "rules.jsonl"
    rules.write_text("".join(json.dumps(r) + "\n" for r in RULES), encoding="utf-8")
    base = {
        "campaign_id": "camp",
        "campaign_root": str(tmp_path / "root"),
        "scenario_set_path": str(scenarios),
        "approved_scenario_sha256": "null",
        "rule_set_path": str(rules),
        "sampling.campaign_mode": "pilot",
        "sampling.num_decoder_draws": 2,
        "sampling.scenarios_per_request_block": "null",
        "sampling.bootstrap.n_resamples": 150,
    }
    base.update(overrides)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(
            config_name="sampling_campaign",
            overrides=[f"{key}={value}" for key, value in base.items()],
        )


@pytest.fixture(autouse=True)
def _artifact_root(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))


def backend_for(style: str, **kwargs) -> FakeBackend:
    backend = FakeBackend(style=style, **kwargs)
    backend.max_new_tokens = 1024
    return backend


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"num_tries": 3}, "num_tries=3"),
        ({"eval_cfg.evaluation_temperature": 0.1}, "temperature 1.0"),
        ({"evaluation_mode": "individual"}, "evaluation_mode"),
        ({"eval_cfg.judge_types": "[Narrow]"}, "judge_types"),
        ({"save_raw_outputs": "false"}, "save_raw_outputs"),
        ({"eval_cfg.reasoning_prompt_style": "forced-v2"}, "reasoning_prompt_style='forced-v2' must be one of"),
        # A style its task keeps only to read the records that ran it.
        ({"task_spec": "tests.test_tasks:ArchivedDemoTask", "eval_cfg.reasoning_prompt_style": "json"},
         "reasoning_prompt_style: prompt style 'json' of task 'demo-archived@1.0' is recognised only to read the records"),
        # A paper style has an approved No_Interpretation text only.
        ({"eval_cfg.reasoning_prompt_style": "encouraged", "interpretation": "Narrow", "eval_cfg.judge_types": "[Narrow]"},
         r"interpretation: reasoning_prompt_style 'encouraged' is defined for the No_Interpretation template only; "
         r".*judge type\(s\) \['Narrow'\] would use it"),
    ],
)
def test_the_retired_policies_cannot_be_composed_into_a_campaign(tmp_path, override, message):
    cfg = compose_campaign(tmp_path, **override)
    with pytest.raises(CampaignRunError, match=message):
        validate_campaign_cfg(cfg)


def test_build_plan_freezes_the_population_and_registers_the_condition(tmp_path, modelspec):
    cfg = compose_campaign(tmp_path, **{"condition_extra": "{ruleset_label: initial}"})
    backend = backend_for("naive-rules-above")
    plan = build_plan(cfg, backend=backend, task_spec=modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0))

    root = tmp_path / "root"
    assert (root / SCENARIO_MANIFEST_FNAME).exists() and (root / SAMPLING_MANIFEST_FNAME).exists()
    assert plan.condition.condition_id == "gemma-4-31B-it/naive-rules-above"
    assert plan.condition.model_name == "fake-judge" and plan.condition.seed_capability == "local_deterministic"
    assert plan.condition.extra == (("ruleset_label", "initial"),)
    assert plan.condition.max_new_tokens == 1024 and plan.condition.endpoint is None
    assert plan.condition.model_path == "gemma-4-31B-it"
    assert plan.condition.adapter_path is None and plan.condition.enable_thinking is True
    assert plan.decoder_draw_ids == (0, 1) and plan.scenario_manifest.record_count == 3
    assert len(plan.rules) == 3 and plan.judge_type == "No_Interpretation"
    registered = json.loads((root / "gemma-4-31B-it" / "naive-rules-above" / CONDITION_FNAME).read_text())
    assert registered["fingerprint"] == plan.condition.fingerprint
    manifest = json.loads((root / SAMPLING_MANIFEST_FNAME).read_text())
    assert manifest["expected_decoder_draws"] == 2 and manifest["campaign_mode"] == "pilot"

    again = build_plan(cfg, backend=backend, task_spec=modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0))
    assert again.condition == plan.condition


def test_build_plan_refuses_a_different_population_or_digest(tmp_path, modelspec):
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    cfg = compose_campaign(tmp_path)
    build_plan(cfg, backend=backend_for("naive-rules-above"), task_spec=task_spec)
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"prompt": "prompt 0", "response": "response 0"}) + "\n")
    with pytest.raises(CampaignRunError, match="froze a different population"):
        build_plan(compose_campaign(tmp_path, scenario_set_path=str(other)), backend=backend_for("naive-rules-above"), task_spec=task_spec)
    with pytest.raises(CampaignRunError, match="does not match approved"):
        build_plan(compose_campaign(tmp_path, approved_scenario_sha256="0" * 64), backend=backend_for("naive-rules-above"), task_spec=task_spec)


def _tree(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.mark.parametrize("named", ["public", "recorded"])
def test_a_pinned_root_is_refused_before_the_preflight_or_the_plan_writes_anything(tmp_path, modelspec, monkeypatch,
                                                                                    named):
    from ruleforge.core.record_names import identity_key
    from ruleforge.evaluate import api
    from ruleforge.evaluate.campaign import preflight

    root = tmp_path / "artifacts" / "sampling" / "pub"
    root.mkdir(parents=True)
    (root / SAMPLING_MANIFEST_FNAME).write_text("{}\n", encoding="utf-8")  # a published root's records stand in
    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://sampling/pub": ("leniency",)})
    style = {"public": "naive-rules-above", "recorded": identity_key("naive-rules-above")}[named]
    cfg = compose_campaign(tmp_path, campaign_root=str(root), condition_id=f"gemma-4-31B-it/{style}")
    before = _tree(root)
    refusal = r"\(artifact://sampling/pub\) is artifact://sampling/pub, or on its path, a published run"
    with pytest.raises(api.CampaignError, match=refusal):
        preflight(cfg)
    with pytest.raises(api.CampaignError, match=refusal):
        build_plan(cfg, backend=backend_for("naive-rules-above"),
                   task_spec=modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0))
    assert _tree(root) == before and sorted(path.name for path in root.parent.iterdir()) == ["pub"]


def test_a_shard_whose_own_root_is_on_a_pinned_root_s_path_is_refused_before_anything_is_written(tmp_path, modelspec,
                                                                                                 monkeypatch):
    from ruleforge.evaluate import api
    from ruleforge.evaluate.campaign import preflight
    from ruleforge.evaluate.sharding import job_root_override

    root = tmp_path / "artifacts" / "sampling" / "camp"
    root.parent.mkdir(parents=True)
    monkeypatch.setattr(api, "pinned_roots",
                        lambda: {"artifact://sampling/camp.shards": ("leniency",)})
    key, _, value = job_root_override(0, 2).partition("=")
    cfg = compose_campaign(tmp_path, campaign_root=str(root), **{
        "execution.num_shards": 2, "execution.shard_index": 0, "sampling.scenarios_per_request_block": 1, key: value})
    shard = f"{root}.shards/gemma-4-31B-it/naive-rules-above/shard-0-of-2"
    with pytest.raises(api.CampaignError, match=f"{shard} \\(artifact://sampling/camp.shards/gemma-4-31B-it/naive-rules-above/"
                                                "shard-0-of-2\\) is artifact://sampling/camp.shards, or on its path"):
        preflight(cfg)
    with pytest.raises(api.CampaignError, match="is artifact://sampling/camp.shards, or on its path"):
        build_plan(cfg, backend=backend_for("naive-rules-above"),
                   task_spec=modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0))
    assert list(root.parent.iterdir()) == []


def test_a_refused_registration_writes_nothing_in_the_root(tmp_path, modelspec):
    from tests.support.recorded_root import recorded_payload

    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    made = build_plan(compose_campaign(tmp_path, campaign_root=str(tmp_path / "made")), backend=backend_for("naive-rules-above"),
                      task_spec=task_spec)
    assert made.condition.identity_id != made.condition.condition_id
    recorded = tmp_path / "recorded"
    registration = recorded / made.condition.identity_id / CONDITION_FNAME
    registration.parent.mkdir(parents=True)
    registration.write_text(json.dumps(recorded_payload(made.condition)), encoding="utf-8")
    before = _tree(recorded)
    with pytest.raises(CampaignRunError, match=r"is hashed over \(its identity id\): the two would share"):
        build_plan(compose_campaign(tmp_path, campaign_root=str(recorded)), backend=backend_for("naive-rules-above"),
                   task_spec=task_spec)
    assert _tree(recorded) == before


def test_build_plan_honours_and_checks_a_draw_subset(tmp_path, modelspec):
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    plan = build_plan(compose_campaign(tmp_path, decoder_draw_ids="[1]"), backend=backend_for("naive-rules-above"), task_spec=task_spec)
    assert plan.decoder_draw_ids == (1,)
    with pytest.raises(CampaignRunError, match="distinct ids below 2"):
        build_plan(compose_campaign(tmp_path, decoder_draw_ids="[0,2]"), backend=backend_for("naive-rules-above"), task_spec=task_spec)
    with pytest.raises(CampaignRunError, match="distinct ids below 2"):
        build_plan(compose_campaign(tmp_path, decoder_draw_ids="[1,1]"), backend=backend_for("naive-rules-above"), task_spec=task_spec)


def test_a_batch_condition_records_its_endpoint_and_token_ceiling(tmp_path, modelspec):
    cfg = compose_campaign(
        tmp_path,
        **{"model@judge_model": "gpt-5.6-luna-batch", "sampling.scenarios_per_request_block": "null"},
    )
    backend = backend_for("naive-rules-above", model_type="openai_batch", seed_capability="unsupported")
    plan = build_plan(cfg, backend=backend, task_spec=modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0))
    assert plan.condition.condition_id == "gpt-5.6-luna/naive-rules-above"
    assert plan.condition.endpoint == "responses"
    assert plan.condition.send_max_output_tokens is False
    assert plan.condition.reasoning_effort == "medium"
    assert plan.condition.seed_capability == "unsupported"
    assert plan.condition.model_path is None and plan.condition.enable_thinking is None


def test_register_condition_is_idempotent_and_refuses_a_relabel(tmp_path, modelspec):
    cfg = compose_campaign(tmp_path)
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    plan = build_plan(cfg, backend=backend_for("naive-rules-above"), task_spec=task_spec)
    assert register_condition(plan.campaign_root, plan.condition).exists()
    from dataclasses import replace

    changed = replace(plan.condition, rule_set_sha256="b" * 64)
    with pytest.raises(CampaignRunError, match="different identity"):
        register_condition(plan.campaign_root, changed)
    assert load_conditions(plan.campaign_root) == {plan.condition.condition_id: plan.condition}


def test_load_conditions_refuses_a_forged_or_misplaced_registration(tmp_path, modelspec):
    cfg = compose_campaign(tmp_path)
    plan = build_plan(cfg, backend=backend_for("naive-rules-above"), task_spec=modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0))
    path = plan.condition_root / CONDITION_FNAME
    payload = json.loads(path.read_text())
    payload["fingerprint"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(SamplingDesignError, match="does not match the condition"):
        load_conditions(plan.campaign_root)
    payload["fingerprint"] = plan.condition.fingerprint
    path.write_text(json.dumps(payload))
    misplaced = plan.campaign_root / "elsewhere" / CONDITION_FNAME
    misplaced.parent.mkdir()
    misplaced.write_text(json.dumps(payload))
    with pytest.raises(SamplingDesignError, match="outside its own directory"):
        load_conditions(plan.campaign_root)


def run_two_conditions(tmp_path, modelspec, *, naive_script=None, forced_script=None):
    completions = {}
    for style, script in (("naive-rules-above", naive_script), ("forced-grpo", forced_script)):
        cfg = compose_campaign(tmp_path, **{"eval_cfg.reasoning_prompt_style": style})
        backend = backend_for(style, script=script)
        task_spec = modelspec(reasoning_prompt_style=style, evaluation_temperature=1.0)
        plan = build_plan(cfg, backend=backend, task_spec=task_spec)
        completions[style] = run_plan(plan, backend=backend, task_spec=task_spec)
    return tmp_path / "root", completions


def test_two_conditions_complete_and_the_campaign_gate_agrees(tmp_path, modelspec):
    root, completions = run_two_conditions(tmp_path, modelspec)
    assert {draw: c["state"] for draw, c in completions["naive-rules-above"].items()} == {0: STATE_COMPLETE, 1: STATE_COMPLETE}
    gate = json.loads((root / COMPLETION_FNAME).read_text())
    assert gate["state"] == STATE_COMPLETE and gate["n_conditions"] == 2
    assert set(gate["conditions"]) == {"gemma-4-31B-it/naive-rules-above", "gemma-4-31B-it/forced-grpo"}
    assert gate["conditions"]["gemma-4-31B-it/naive-rules-above"]["draws"]["1"]["accepted"] == 3
    assert require_complete_campaign(root)["state"] == STATE_COMPLETE


def test_the_gate_reports_missing_exhausted_and_foreign_draws(tmp_path, modelspec):
    root, _ = run_two_conditions(tmp_path, modelspec)
    naive = root / "gemma-4-31B-it" / "naive-rules-above"
    (naive / "draw-001" / COMPLETION_FNAME).rename(naive / "stash.json")
    gate = campaign_completion(root)
    assert gate["state"] == STATE_IN_PROGRESS
    assert gate["conditions"]["gemma-4-31B-it/naive-rules-above"]["draws"]["1"] == {"state": STATE_IN_PROGRESS, "recorded": False}
    with pytest.raises(CampaignRunError, match="not complete"):
        require_complete_campaign(root)
    (naive / "stash.json").rename(naive / "draw-001" / COMPLETION_FNAME)
    record = json.loads((naive / "draw-001" / COMPLETION_FNAME).read_text())
    record["state"] = STATE_INCOMPLETE_EXHAUSTED
    (naive / "draw-001" / COMPLETION_FNAME).write_text(json.dumps(record))
    assert campaign_completion(root)["state"] == STATE_INCOMPLETE_EXHAUSTED
    # A draw whose provenance names another condition: invalid manifest wins.
    record["state"] = STATE_COMPLETE
    record["condition_fingerprint"] = "0" * 64
    (naive / "draw-001" / COMPLETION_FNAME).write_text(json.dumps(record))
    assert campaign_completion(root)["state"] == STATE_INVALID_MANIFEST


def test_a_campaign_with_no_registered_condition_is_in_progress(tmp_path, modelspec):
    cfg = compose_campaign(tmp_path)
    plan = build_plan(cfg, backend=backend_for("naive-rules-above"), task_spec=modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0))
    (plan.condition_root / CONDITION_FNAME).unlink()
    assert campaign_completion(plan.campaign_root)["state"] == STATE_IN_PROGRESS


class _Killed(BaseException):
    """A run killed between a block's append and its gate's write (no handler of the runner's catches it)."""


@pytest.mark.parametrize("scope", ["scenario_bundle", "component"])
def test_a_gate_behind_its_ledger_is_refused_when_the_ledgers_are_checked(tmp_path, modelspec, monkeypatch, scope):
    cfg = compose_campaign(tmp_path, **{"sampling.retry_scope": scope})
    backend = backend_for("naive-rules-above", script={(0, "Rule 1"): ["maybe", "no"]})
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    plan = build_plan(cfg, backend=backend, task_spec=task_spec)
    write, writes = DrawRunner._write_completion, []

    def killed(self, state):
        writes.append(len(state.records))
        if len(writes) == 2:  # prepare's write, then the first block's
            raise _Killed
        return write(self, state)

    monkeypatch.setattr(DrawRunner, "_write_completion", killed)
    with pytest.raises(_Killed):
        run_plan(plan, backend=backend, task_spec=task_spec)
    monkeypatch.setattr(DrawRunner, "_write_completion", write)
    root, condition = plan.campaign_root, plan.condition.condition_id
    behind = campaign_completion(root, write=False)["conditions"][condition]["draws"]["0"]
    assert (behind["accepted"], behind["rejected_attempts"]) == (0, 0)
    fields = ["accepted: gate 0, ledger 2", "attempts_histogram: gate {}, ledger {'1': 2}", "rejected_attempts: gate 0, "
              "ledger 1", "invalid_components: gate {}, ledger {'invalid_label': 1}"]
    if scope == "component":
        fields.append("component_attempts_histogram: gate {}, ledger {'1': 6}")
    with pytest.raises(CampaignRunError, match="1 draw gate") as refused:
        campaign_completion(root, write=False, check_ledgers=True)
    assert str(refused.value).endswith(f"{root / condition / 'draw-000' / COMPLETION_FNAME} ({'; '.join(fields)})")
    run_plan(build_plan(cfg, backend=backend, task_spec=task_spec), backend=backend, task_spec=task_spec)
    assert campaign_completion(root, write=False, check_ledgers=True)["state"] == STATE_COMPLETE


_TORN_CUTS = {
    "inside_the_first_row": lambda rows: rows[0][:40],
    "before_the_first_rows_newline": lambda rows: rows[0],
    "before_a_trailing_rows_newline": lambda rows: rows[0] + "\n" + rows[1],
}


@pytest.mark.parametrize("cut", sorted(_TORN_CUTS))
@pytest.mark.parametrize("scope", ["scenario_bundle", "component"])
def test_a_torn_first_ledger_append_is_refused_when_the_ledgers_are_checked(
    tmp_path, modelspec, monkeypatch, scope, cut
):
    cfg = compose_campaign(tmp_path, **{"sampling.retry_scope": scope})
    backend = backend_for("naive-rules-above", script={(0, "Rule 1"): ["maybe", "no"]})
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    plan = build_plan(cfg, backend=backend, task_spec=task_spec)
    append, written = DrawLedger._append_lines, []

    def torn(path, payloads):
        if path.name != LEDGER_FNAME:
            return append(path, payloads)
        written.append(_TORN_CUTS[cut]([json.dumps(payload, ensure_ascii=False) for payload in payloads]))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(written[0])
        raise _Killed

    monkeypatch.setattr(DrawLedger, "_append_lines", staticmethod(torn))
    with pytest.raises(_Killed):
        run_plan(plan, backend=backend, task_spec=task_spec)
    monkeypatch.setattr(DrawLedger, "_append_lines", staticmethod(append))
    root, condition = plan.campaign_root, plan.condition.condition_id
    ledger = root / condition / "draw-000" / LEDGER_FNAME
    size = ledger.stat().st_size
    assert ledger.read_bytes() == written[0].encode("utf-8") and not written[0].endswith("\n")
    if cut != "inside_the_first_row":  # the last line is a whole JSON object: only its newline is missing
        assert json.loads(written[0].split("\n")[-1])["condition_id"] == condition
    behind = campaign_completion(root, write=False)["conditions"][condition]["draws"]["0"]
    assert (behind["state"], behind["accepted"], behind["rejected_attempts"]) == (STATE_IN_PROGRESS, 0, 0)
    with pytest.raises(LedgerError, match="torn trailing record"):
        campaign_completion(root, write=False, check_ledgers=True)
    assert ledger.stat().st_size == size
    run_plan(build_plan(cfg, backend=backend, task_spec=task_spec), backend=backend, task_spec=task_spec)
    assert campaign_completion(root, write=False, check_ledgers=True)["state"] == STATE_COMPLETE


def test_style_families_group_conditions_that_differ_only_in_style(tmp_path, modelspec):
    root, _ = run_two_conditions(tmp_path, modelspec)
    families = style_families(load_conditions(root))
    assert list(families.values()) == [["gemma-4-31B-it/naive-rules-above", "gemma-4-31B-it/forced-grpo"]]


def test_analyze_campaign_writes_replicate_aware_estimates(tmp_path, modelspec):
    forced_script = {
        (1, "full_ruleset"): [forced("no", {"Rule 1": "yes", "Rule 2": "no"})],
    }
    naive_script = {(0, "Rule 2"): ["maybe", "yes"]}
    root, _ = run_two_conditions(tmp_path, modelspec, naive_script=naive_script, forced_script=forced_script)
    payload = analyze_campaign(root, n_permutations=200)

    assert payload["state"] == STATE_COMPLETE and payload["n_decoder_draws"] == 2
    assert payload["inference_method"] == "hierarchical_bootstrap_v2"
    assert payload["excluded_conditions"] == {}
    assert payload["truncation"] == "model_default"
    assert "retry_scope" not in payload
    assert payload["conditions"]["gemma-4-31B-it/naive-rules-above"]["sampling_truncation"] == {
        "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
    }
    assert (root / "aggregate" / "estimates.json").exists()
    naive = payload["conditions"]["gemma-4-31B-it/naive-rules-above"]
    assert naive["estimate"] == pytest.approx(1 / 3) and naive["n_decoder_draws"] == 2
    assert naive["attempt_statistics"]["attempts_histogram"] == {"1": 5, "2": 1}
    assert naive["attempt_statistics"]["rejected_attempts"] == 1
    assert naive["attempt_statistics"]["invalid_components"] == {"invalid_label": 1}
    assert naive["attempt_statistics"]["invalid_attempt_rate"] == pytest.approx(1 / 7)
    forced_rows = payload["forced_conditions"]["gemma-4-31B-it/forced-grpo"]
    assert forced_rows["method_a_vs_b"]["estimate"] == pytest.approx(0 - 1 / 3)
    assert forced_rows["method_a_vs_b"]["inner_resample"] == "joint"
    assert forced_rows["method_b_rate"]["estimate"] == pytest.approx(1 / 3)
    rules = {row["rule_id"]: row for row in forced_rows["rule_context"]}
    assert set(rules) == {"Rule 1", "Rule 2"}
    assert rules["Rule 1"]["estimate"] == pytest.approx(1 / 3)
    assert rules["Rule 2"]["estimate"] == 0.0 and rules["Rule 2"]["p_value"] == 1.0
    assert all(row["test_method"] == "scenario_clustered_paired_randomization_v1" for row in rules.values())
    assert "p_holm" in rules["Rule 1"]
    family = next(iter(payload["style_families"].values()))
    assert family["members"] == ["gemma-4-31B-it/naive-rules-above", "gemma-4-31B-it/forced-grpo"]
    contrast = family["contrasts"][0]
    assert contrast["inner_resample"] == "independent"
    assert contrast["estimate"] == pytest.approx(1 / 3)
    assert "p_holm" in contrast and contrast["n_permutations"] == 200
    assert contrast["family"] == "style_contrasts" and contrast["family_size"] == 1
    assert rules["Rule 1"]["family"] == "rule_context:gemma-4-31B-it/forced-grpo"
    assert rules["Rule 1"]["family_size"] == 2


def test_the_paper_s_three_prompts_run_and_aggregate_as_one_style_family_naive_to_forced(tmp_path, modelspec):
    styles = ("naive-rules-above", "encouraged", "forced")
    cids = {style: f"gemma-4-31B-it/{style}" for style in styles}
    scenario_1 = json.dumps({"rule_judgments": {"Rule 1": "yes", "Rule 2": "no"}, "preempted_rules": ["Rule 1"],
                             "full_set_judgment": "no"})
    scripts = {"naive-rules-above": {(0, "full_ruleset"): ["no", "yes"]}, "encouraged": None,
               "forced": {(1, "full_ruleset"): [scenario_1]}}
    plans = {}
    for style in styles:
        cfg = compose_campaign(tmp_path, **{"eval_cfg.reasoning_prompt_style": style})
        validate_campaign_cfg(cfg)
        backend = backend_for(style, script=scripts[style])
        task_spec = modelspec(reasoning_prompt_style=style, evaluation_temperature=1.0)
        plans[style] = build_plan(cfg, backend=backend, task_spec=task_spec)
        run_plan(plans[style], backend=backend, task_spec=task_spec)
        assert ("provided above?" if style == "naive-rules-above" else "provided below?") in task_spec.build_eval_prompt(
            "Rule 1: x", {"prompt": "p", "response": "r"}, "No_Interpretation")
    validate_campaign_cfg(compose_campaign(tmp_path, **{"eval_cfg.reasoning_prompt_style": "naive"}))
    identities = {style: plans[style].condition.to_dict() for style in styles}
    for style in styles[1:]:
        assert sorted(key for key in identities[style] if identities[style][key] != identities[styles[0]][key]) == [
            "condition_id", "prompt_style"]
    payload = analyze_campaign(tmp_path / "root", n_permutations=100, n_resamples=100)
    assert set(payload["forced_conditions"]) == {cids["forced"]}
    forced_rows = payload["forced_conditions"][cids["forced"]]
    rules = {row["rule_id"]: row for row in forced_rows["rule_context"]}
    assert rules["Rule 1"]["estimate"] == pytest.approx(1 / 3) and forced_rows["method_b_rate"]["estimate"] == (
        pytest.approx(1 / 3))
    [family] = payload["style_families"].values()
    assert family["members"] == [cids[style] for style in styles]
    assert [(row["a"], row["b"]) for row in family["contrasts"]] == [
        (cids["naive-rules-above"], cids["encouraged"]), (cids["naive-rules-above"], cids["forced"]),
        (cids["encouraged"], cids["forced"])]
    assert {row["family_size"] for row in family["contrasts"]} == {3}
    for draw in ("draw-000", "draw-001"):
        records = [json.loads(line) for line in
                   (tmp_path / "root" / cids["forced"] / draw / "attempts.jsonl").read_text().splitlines()]
        full_sets = [record["components"][0] for record in records]
        assert full_sets and all(c["component_id"] == "full_ruleset" and c["preempted_rules"] == [] for c in full_sets)
        labels = [json.loads(line) for line in
                  (tmp_path / "root" / cids["naive-rules-above"] / draw / "attempts.jsonl").read_text().splitlines()]
        assert all(c["rule_judgments"] is None for record in labels for c in record["components"])


def test_analyze_campaign_refuses_an_incomplete_campaign(tmp_path, modelspec):
    root, _ = run_two_conditions(tmp_path, modelspec)
    (root / "gemma-4-31B-it" / "naive-rules-above" / "draw-001" / COMPLETION_FNAME).unlink()
    with pytest.raises(CampaignRunError, match="not complete"):
        analyze_campaign(root, n_permutations=100)


def test_analyze_campaign_can_report_the_complete_conditions_of_a_partial_campaign(tmp_path, modelspec):
    root, _ = run_two_conditions(tmp_path, modelspec, naive_script={(0, "Rule 2"): ["maybe", "yes"]})
    naive_draw = root / "gemma-4-31B-it" / "naive-rules-above" / "draw-001"
    gate = json.loads((naive_draw / COMPLETION_FNAME).read_text())
    gate["state"] = "incomplete_exhausted"
    (naive_draw / COMPLETION_FNAME).write_text(json.dumps(gate))

    payload = analyze_campaign(root, n_permutations=100, n_resamples=100, allow_incomplete_conditions=True)
    assert payload["state"] == "incomplete_exhausted"
    assert set(payload["conditions"]) == {"gemma-4-31B-it/forced-grpo"}
    excluded = payload["excluded_conditions"]["gemma-4-31B-it/naive-rules-above"]
    assert excluded["draw_states"] == {"0": STATE_COMPLETE, "1": "incomplete_exhausted"}
    assert excluded["attempt_statistics"]["invalid_components"] == {"invalid_label": 1}
    assert all(family["contrasts"] == [] for family in payload["style_families"].values())
    assert "gemma-4-31B-it/forced-grpo" in payload["forced_conditions"]

    forced_draw = root / "gemma-4-31B-it" / "forced-grpo" / "draw-000"
    (forced_draw / COMPLETION_FNAME).unlink()
    with pytest.raises(CampaignRunError, match="no complete condition"):
        analyze_campaign(root, n_permutations=100, allow_incomplete_conditions=True)


def test_analyze_campaign_takes_an_explicit_rule_list(tmp_path, modelspec):
    root, _ = run_two_conditions(tmp_path, modelspec)
    payload = analyze_campaign(root, n_permutations=100, rule_ids=["Rule 2"], n_resamples=100, seed=3)
    rows = payload["forced_conditions"]["gemma-4-31B-it/forced-grpo"]["rule_context"]
    assert [row["rule_id"] for row in rows] == ["Rule 2"]
    assert payload["bootstrap"] == {"n_resamples": 100, "ci_level": 0.95, "seed": 3}


def set_recorded_runtime(condition_dir: Path, draw: str = "draw-*", **fields) -> None:
    """Rewrite the matching draws' recorded runtime fields and the stack their ledger rows name."""
    for provenance in condition_dir.glob(f"{draw}/draw_provenance.json"):
        payload = json.loads(provenance.read_text())
        payload["runtime"].update(fields)
        provenance.write_text(json.dumps(payload))
        restamp_stacks(provenance.parent)


def set_recorded_vllm(condition_dir: Path, version: str) -> None:
    """Rewrite every draw's recorded vLLM version, as if its judgments had come from ``version``."""
    set_recorded_runtime(condition_dir, vllm_version=version)


def test_analyze_campaign_refuses_a_contrast_across_inference_stacks(tmp_path, modelspec):
    root, _ = run_two_conditions(tmp_path, modelspec)
    set_recorded_vllm(root / "gemma-4-31B-it" / "naive-rules-above", "0.28.0+cu129")
    set_recorded_vllm(root / "gemma-4-31B-it" / "forced-grpo", "0.19.0")
    with pytest.raises(CampaignRunError, match=r"style_contrasts: 1 contrast\(s\) would compare judgments from different"):
        analyze_campaign(root, n_permutations=100, n_resamples=100)
    payload = analyze_campaign(root, n_permutations=100, n_resamples=100, allow_stack_mismatch=True)
    contrast = next(iter(payload["style_families"].values()))["contrasts"][0]
    assert contrast["stack_mismatch"].startswith("vllm 0.28.0+cu129 / torch")
    assert "against vllm 0.19.0" in contrast["stack_mismatch"]
    stacks = {cid: row["inference_stacks"] for cid, row in payload["conditions"].items()}
    assert [s["vllm"] for s in stacks["gemma-4-31B-it/naive-rules-above"]] == ["0.28.0+cu129"]
    assert [s["vllm"] for s in stacks["gemma-4-31B-it/forced-grpo"]] == ["0.19.0"]
    assert payload["allow_stack_mismatch"] is True
    assert payload["stack_mismatches"] == [{
        "scope": "style_contrasts", "conditions": ["gemma-4-31B-it/naive-rules-above", "gemma-4-31B-it/forced-grpo"],
        "reason": contrast["stack_mismatch"],
    }]
    # another CUDA build of the same release is another stack
    set_recorded_vllm(root / "gemma-4-31B-it" / "forced-grpo", "0.28.0")
    with pytest.raises(CampaignRunError, match=r"vllm 0.28.0\+cu129 / .* against vllm 0.28.0 /"):
        analyze_campaign(root, n_permutations=100, n_resamples=100)
    set_recorded_vllm(root / "gemma-4-31B-it" / "forced-grpo", "0.28.0+cu129")
    clean = analyze_campaign(root, n_permutations=100, n_resamples=100)
    assert "stack_mismatch" not in next(iter(clean["style_families"].values()))["contrasts"][0]
    assert not {"allow_stack_mismatch", "stack_mismatches"} & set(clean)
    assert all("inference_stacks" not in row for row in clean["conditions"].values())
    assert "stack" not in (root / "aggregate" / "estimates.json").read_text()


def test_analyze_campaign_refuses_a_condition_whose_draws_mix_stacks_before_computing_it(tmp_path, modelspec, monkeypatch):
    import ruleforge.evaluate.aggregate as aggregate_module

    root, _ = run_two_conditions(tmp_path, modelspec)
    naive = root / "gemma-4-31B-it" / "naive-rules-above"
    set_recorded_vllm(root / "gemma-4-31B-it" / "forced-grpo", "0.28.0")
    set_recorded_vllm(naive, "0.28.0")
    provenance = naive / "draw-001" / "draw_provenance.json"
    payload = json.loads(provenance.read_text())
    payload["runtime"]["vllm_version"] = "0.19.0"
    provenance.write_text(json.dumps(payload))
    restamp_stacks(provenance.parent)

    real = aggregate_module.load_condition_outcomes
    monkeypatch.setattr(aggregate_module, "load_condition_outcomes", lambda *a, **k: pytest.fail("read before the refusal"))
    with pytest.raises(CampaignRunError, match=r"1 condition\(s\) hold judgments from several inference stacks or "
                                               r"engine configurations \(gemma-4-31B-it/naive-rules-above: its judgments come "
                                               r"from several stacks"):
        analyze_campaign(root, n_permutations=100, n_resamples=100)
    monkeypatch.setattr(aggregate_module, "load_condition_outcomes", real)
    allowed = analyze_campaign(root, n_permutations=100, n_resamples=100, allow_stack_mismatch=True)
    assert [s["vllm"] for s in allowed["conditions"]["gemma-4-31B-it/naive-rules-above"]["inference_stacks"]] == ["0.28.0", "0.19.0"]
    assert [(r["scope"], r["conditions"]) for r in allowed["stack_mismatches"]] == [
        ("condition", ["gemma-4-31B-it/naive-rules-above"]),
        ("style_contrasts", ["gemma-4-31B-it/naive-rules-above", "gemma-4-31B-it/forced-grpo"]),
    ]


def test_analyze_campaign_refuses_a_condition_whose_draws_ran_other_binding_engine_arguments(tmp_path, modelspec):
    from ruleforge.evaluate.stack import recorded_mismatch_refusal

    root, _ = run_two_conditions(tmp_path, modelspec)
    naive, forced_grpo = root / "gemma-4-31B-it" / "naive-rules-above", root / "gemma-4-31B-it" / "forced-grpo"
    engine = {"model": "/a/gemma", "tensor_parallel_size": 1, "max_num_seqs": 256, "max_model_len": 40960,
              "attention_config": {"backend": "TRITON_ATTN"}, "enable_lora": False}
    set_recorded_runtime(naive, engine_args=engine)
    set_recorded_runtime(naive, "draw-001", engine_args={**engine, "tensor_parallel_size": 2, "max_num_seqs": 512})
    set_recorded_runtime(forced_grpo, engine_args={**engine, "model": "/b/policy", "enable_lora": True})
    clean = analyze_campaign(root, n_permutations=100, n_resamples=100)
    assert not {"allow_stack_mismatch", "stack_mismatches"} & set(clean)
    set_recorded_runtime(naive, "draw-001", engine_args={**engine, "attention_config": {"backend": "FLASH_ATTN"}})
    with pytest.raises(CampaignRunError, match=r"gemma-4-31B-it/naive-rules-above: its judgments come from several stacks or "
                                               r"engine configurations \(.*engine_args=\w+; engine_args.attention_config "
                                               r"\{'backend': 'TRITON_ATTN'\} -> \{'backend': 'FLASH_ATTN'\}\)"):
        analyze_campaign(root, n_permutations=100, n_resamples=100)
    allowed = analyze_campaign(root, n_permutations=100, n_resamples=100, allow_stack_mismatch=True)
    assert [(r["scope"], r["conditions"]) for r in allowed["stack_mismatches"]] == [
        ("condition", ["gemma-4-31B-it/naive-rules-above"]),
        ("style_contrasts", ["gemma-4-31B-it/naive-rules-above", "gemma-4-31B-it/forced-grpo"]),
    ]
    assert recorded_mismatch_refusal(allowed, "estimates.json").startswith("estimates.json reports 2 inference-stack")


def test_ruleforge_analyze_writes_a_configured_root_s_analysis_as_the_configured_analysis_does(tmp_path, modelspec,
                                                                                                  capsys):
    import shutil

    from ruleforge.cli import main
    from ruleforge.evaluate.configured import MARKER_FNAME
    from tests.support.equivalence import read

    root, _ = run_two_conditions(tmp_path, modelspec)
    assert (root / MARKER_FNAME).is_file()
    twin = tmp_path / "twin"
    shutil.copytree(root, twin)
    flags = ["--n-permutations", "100", "--n-resamples", "100"]
    assert main(["analyze", str(root), *flags]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed["conditions"]) == {"gemma-4-31B-it/naive-rules-above", "gemma-4-31B-it/forced-grpo"}
    analyze_campaign(twin, n_permutations=100, n_resamples=100)
    estimates = Path("aggregate") / "estimates.json"
    assert read(root / estimates) == read(twin / estimates) and read(root / COMPLETION_FNAME) == read(
        twin / COMPLETION_FNAME)
    assert main(["analyze", str(root), *flags]) == 1 and "holds this root's analysis" in capsys.readouterr().err
    set_recorded_vllm(root / "gemma-4-31B-it" / "naive-rules-above", "0.28.0")
    set_recorded_vllm(root / "gemma-4-31B-it" / "forced-grpo", "0.19.0")
    assert main(["analyze", str(root), *flags, "--replace"]) == 1
    assert "inference stacks" in capsys.readouterr().err
    assert main(["analyze", str(root), *flags, "--replace", "--allow-stack-mismatch"]) == 0


def test_the_run_script_runs_a_condition_and_fails_loudly_when_a_draw_is_incomplete(
    tmp_path, monkeypatch
):
    import ruleforge.judges.backend as backend_module

    scenarios = write_inputs(tmp_path)
    rules = tmp_path / "rules.jsonl"
    rules.write_text("".join(json.dumps(r) + "\n" for r in RULES), encoding="utf-8")
    script = "ruleforge.launch.programs.run_campaign"
    overrides = [
        "campaign_id=camp", f"campaign_root={tmp_path / 'root'}",
        f"scenario_set_path={scenarios}", "approved_scenario_sha256=null",
        f"rule_set_path={rules}", "sampling.campaign_mode=pilot",
        "sampling.num_decoder_draws=2", "sampling.scenarios_per_request_block=null",
        f"hydra.run.dir={tmp_path / 'hydra'}",
    ]
    backends = []

    def fake_backend(model_cfg):
        backend = backend_for("naive-rules-above", script={(0, "Rule 1"): ["???"]} if backends else None)
        backend.llm = None
        backends.append(backend)
        return backend

    monkeypatch.setattr(backend_module, "ModelBackend", fake_backend)
    monkeypatch.setattr(sys, "argv", [str(script), *overrides])
    monkeypatch.chdir(REPO_ROOT)  # the checkout's top, where the launcher runs the job
    run_program(script)
    gate = json.loads((tmp_path / "root" / COMPLETION_FNAME).read_text())
    assert gate["state"] == STATE_COMPLETE

    # A second condition whose judge never answers Rule 1 validly exhausts.
    monkeypatch.setattr(sys, "argv", [str(script), *overrides, "eval_cfg.reasoning_prompt_style=encouraged"])
    with pytest.raises((SystemExit, RuntimeError)) as exit_info:
        run_program(script)
    if isinstance(exit_info.value, SystemExit):
        assert exit_info.value.code == 1
    else:
        assert "did not complete" in str(exit_info.value)
    gate = json.loads((tmp_path / "root" / COMPLETION_FNAME).read_text())
    assert gate["state"] == STATE_INCOMPLETE_EXHAUSTED


def test_ruleforge_shaped_conditions_share_the_manifest_and_count_every_non_meta_rule(tmp_path, modelspec):
    from ruleforge.evaluate.campaign import expected_requests_per_draw

    scenarios = tmp_path / "heldout.jsonl"
    scenarios.write_text(
        "".join(json.dumps({"prompt": f"prompt {i}", "response": f"response {i}"}) + "\n" for i in range(500)),
        encoding="utf-8",
    )
    totals = 0
    fingerprints = set()
    for label, n_rules in (("initial", 57), ("best_seed", 58), ("second", 59)):
        rules = tmp_path / f"{label}.jsonl"
        rules.write_text(
            "".join(json.dumps({"id": f"Rule {n}", "rule": f"rule {n}"}) + "\n" for n in range(1, n_rules + 1))
            + json.dumps({"id": "Meta-Rule 1", "rule": "meta"}) + "\n",
            encoding="utf-8",
        )
        cfg = compose_campaign(
            tmp_path,
            scenario_set_path=str(scenarios),
            rule_set_path=str(rules),
            condition_id=f"gemma/{label}",
            condition_extra=f"{{ruleset_label: {label}}}",
            **{"sampling.campaign_mode": "final", "eval_cfg.individual_rule_context": "implicated"},
        )
        plan = build_plan(cfg, backend=backend_for("naive-rules-above"), task_spec=modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0, individual_rule_context="implicated"))
        assert plan.scenario_manifest.record_count == 500
        assert plan.condition.individual_rule_context == "implicated"
        assert plan.condition.extra == (("ruleset_label", label),)
        fingerprints.add(plan.condition.fingerprint)
        totals += expected_requests_per_draw(plan.scenario_manifest, plan.rules)
    assert totals == 88_500
    assert len(fingerprints) == 3
    manifest = json.loads((tmp_path / "root" / SAMPLING_MANIFEST_FNAME).read_text())
    assert manifest["expected_scenarios"] == 500
    assert len(load_conditions(tmp_path / "root")) == 3


def test_preflight_refuses_before_any_model_is_loaded(tmp_path):
    from ruleforge.evaluate.campaign import preflight

    cfg = compose_campaign(tmp_path)
    config, manifest = preflight(cfg)
    assert config.num_decoder_draws == 2 and manifest.record_count == 3
    with pytest.raises(CampaignRunError, match="num_tries"):
        preflight(compose_campaign(tmp_path, num_tries=2))
    with pytest.raises(CampaignRunError, match="does not exist"):
        preflight(compose_campaign(tmp_path, rule_set_path=str(tmp_path / "absent.jsonl")))


def test_an_adapter_served_under_the_same_label_is_a_different_condition(tmp_path, modelspec):
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    base = build_plan(compose_campaign(tmp_path), backend=backend_for("naive-rules-above"), task_spec=task_spec)
    adapter = tmp_path / "runs" / "grpo" / "final_adapter" / "adapter_model.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"weights")
    (adapter.parent / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    tuned = build_plan(
        compose_campaign(tmp_path, **{"+judge_model.adapter_path": str(adapter), "condition_id": "gemma/tuned"}),
        backend=backend_for("naive-rules-above"), task_spec=task_spec,
    )
    assert tuned.condition.adapter_path == "final_adapter/adapter_model.safetensors"
    assert tuned.condition.adapter_sha256 == sha256_file(adapter)
    assert tuned.condition.fingerprint != base.condition.fingerprint


def test_preflight_refuses_a_root_frozen_for_another_population_or_config(tmp_path, modelspec, monkeypatch):
    from ruleforge.evaluate.campaign import preflight

    calls = []
    monkeypatch.setattr(
        campaign_module, "code_state",
        lambda: {"code_revision": "abc", "code_dirty": True, "code_state_sha256": "f" * 64},
    )
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: calls.append((root, state)))
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    build_plan(compose_campaign(tmp_path), backend=backend_for("naive-rules-above"), task_spec=task_spec)
    manifest = json.loads((tmp_path / "root" / SAMPLING_MANIFEST_FNAME).read_text())
    assert (manifest["code_revision"], manifest["code_dirty"], manifest["code_state_sha256"]) == ("abc", True, "f" * 64)
    assert calls == [(tmp_path / "root", {"code_revision": "abc", "code_dirty": True, "code_state_sha256": "f" * 64})]

    other = tmp_path / "other.jsonl"
    other.write_text(
        "".join(json.dumps({"prompt": f"p{i}", "response": f"r{i}"}) + "\n" for i in range(4)),
        encoding="utf-8",
    )
    with pytest.raises(CampaignRunError, match="froze a different population"):
        preflight(compose_campaign(tmp_path, scenario_set_path=str(other)))
    with pytest.raises(SamplingDesignError, match="different campaign"):
        preflight(compose_campaign(tmp_path, **{"sampling.num_decoder_draws": 3}))
    monkeypatch.setattr(
        campaign_module, "code_state",
        lambda: {"code_revision": None, "code_dirty": True, "code_state_sha256": None},
    )
    preflight(compose_campaign(tmp_path))


def test_build_plan_snapshots_the_real_tree_when_git_can_list_it(tmp_path, modelspec):
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    plan = build_plan(compose_campaign(tmp_path), backend=backend_for("naive-rules-above"), task_spec=task_spec)
    manifest = json.loads((tmp_path / "root" / SAMPLING_MANIFEST_FNAME).read_text())
    assert manifest["code_state_sha256"] == plan.code_state["code_state_sha256"]
    snapshots = list((tmp_path / "root" / "code_snapshots").glob("*.tar.gz"))
    assert (plan.code_state["code_state_sha256"] is None) == (snapshots == [])


def test_a_local_condition_resolves_the_models_shipped_truncation(tmp_path, modelspec, monkeypatch):
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    # The judge's weights directory, with the truncation Gemma ships.
    model_dir = tmp_path / "models" / "gemma-4-31B-it"
    model_dir.mkdir(parents=True)
    (model_dir / "generation_config.json").write_text(
        '{"temperature": 1.0, "top_k": 64, "top_p": 0.95, "do_sample": true}', encoding="utf-8"
    )
    plan = build_plan(compose_campaign(tmp_path), backend=backend_for("naive-rules-above"), task_spec=task_spec)
    shipped = {"top_k": 64, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0}
    assert plan.condition.to_dict()["sampling_truncation"] == shipped
    registered = json.loads((tmp_path / "root" / "gemma-4-31B-it" / "naive-rules-above" / "condition.json").read_text())
    assert registered["sampling_truncation"] == shipped
    manifest = json.loads((tmp_path / "root" / SAMPLING_MANIFEST_FNAME).read_text())
    assert manifest["truncation"] == "model_default"

    with pytest.raises(SamplingDesignError, match="different campaign"):
        build_plan(
            compose_campaign(tmp_path, **{"sampling.truncation": "none"}),
            backend=backend_for("naive-rules-above"), task_spec=task_spec,
        )
    plain = build_plan(
        compose_campaign(tmp_path, campaign_root=str(tmp_path / "root2"), **{"sampling.truncation": "none"}),
        backend=backend_for("naive-rules-above"), task_spec=task_spec,
    )
    assert plain.condition.to_dict()["sampling_truncation"] == {"top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}
    assert plain.condition.fingerprint != plan.condition.fingerprint

    (model_dir / "generation_config.json").unlink()
    bare = build_plan(
        compose_campaign(tmp_path, campaign_root=str(tmp_path / "root3")),
        backend=backend_for("naive-rules-above"), task_spec=task_spec,
    )
    assert bare.condition.to_dict()["sampling_truncation"] == plain.condition.to_dict()["sampling_truncation"]


def test_a_provider_condition_records_provider_default_truncation(tmp_path, modelspec):
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    backend = backend_for("naive-rules-above")
    backend.model_type, backend.seed_capability = "openai_batch", "unsupported"
    cfg = compose_campaign(tmp_path, **{"model@judge_model": "gpt-5.6-luna-batch", "model@agent_model": "gpt-5.6-luna-batch"})
    plan = build_plan(cfg, backend=backend, task_spec=task_spec)
    assert plan.condition.to_dict()["sampling_truncation"] == "provider_default"


CARD = {"top_k": 20, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0}
CARD_DECLARATION = "{top_k: 20, top_p: 0.95, min_p: 0.0}"
#: A local judge whose config declares that default_truncation (Qwen3-8B's config, the key added as a campaign override).
CARD_JUDGE = {"model@judge_model": "qwen3-8b", "+judge_model.default_truncation": CARD_DECLARATION}


def card_backend() -> FakeBackend:
    backend = backend_for("naive-rules-above")
    backend.model_name = "Qwen3-8B"
    return backend


def test_a_models_default_truncation_fills_what_its_weights_do_not_ship(tmp_path, modelspec):
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    weights = tmp_path / "models" / "Qwen3-8B"
    weights.mkdir(parents=True)
    backend = card_backend()
    plan = build_plan(compose_campaign(tmp_path, **CARD_JUDGE), backend=backend, task_spec=task_spec)
    assert plan.condition.condition_id == "Qwen3-8B/naive-rules-above"
    recorded = plan.condition.to_dict()["sampling_truncation"]
    assert recorded == CARD and type(recorded["top_k"]) is int and type(recorded["min_p"]) is float
    registered = json.loads((tmp_path / "root" / "Qwen3-8B" / "naive-rules-above" / CONDITION_FNAME).read_text())
    assert registered["sampling_truncation"] == CARD
    run_plan(plan, backend=backend, task_spec=task_spec)
    assert backend.truncations and all(t == CARD for t in backend.truncations)

    plain = build_plan(
        compose_campaign(tmp_path, campaign_root=str(tmp_path / "none"), **CARD_JUDGE, **{"sampling.truncation": "none"}),
        backend=card_backend(), task_spec=task_spec,
    )
    assert plain.condition.to_dict()["sampling_truncation"] == {"top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}
    assert plain.condition.fingerprint != plan.condition.fingerprint

    (weights / "generation_config.json").write_text('{"top_p": 0.8, "temperature": 0.6}', encoding="utf-8")
    shipped = build_plan(
        compose_campaign(tmp_path, campaign_root=str(tmp_path / "shipped"), **CARD_JUDGE),
        backend=card_backend(), task_spec=task_spec,
    )
    assert shipped.condition.to_dict()["sampling_truncation"] == {**CARD, "top_p": 0.8}


def test_the_preflight_script_reports_the_resolved_truncation(tmp_path, capsys, monkeypatch):
    rules = tmp_path / "rules.jsonl"
    scenarios = write_inputs(tmp_path)
    rules.write_text("".join(json.dumps(r) + "\n" for r in RULES), encoding="utf-8")
    script = "ruleforge.launch.programs.preflight_campaign"
    common = [
        str(script), "campaign_id=camp", f"campaign_root={tmp_path / 'root'}",
        f"scenario_set_path={scenarios}", "approved_scenario_sha256=null", f"rule_set_path={rules}",
        "sampling.campaign_mode=pilot", f"hydra.run.dir={tmp_path / 'hydra'}",
    ]
    reports = {}
    monkeypatch.chdir(REPO_ROOT)  # the checkout's top, where the launcher runs the preflight
    for judge, overrides in (("qwen3-8b", [f"+judge_model.default_truncation={CARD_DECLARATION}"]),
                             ("gpt-5.6-luna-batch", [])):
        monkeypatch.setattr(sys, "argv", [*common, f"model@judge_model={judge}", *overrides])
        run_program(script)
        reports[judge] = json.loads(capsys.readouterr().out)["sampling_truncation"]
    assert reports == {"qwen3-8b": CARD, "gpt-5.6-luna-batch": "provider_default"}
    assert not (tmp_path / "root").exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # The card's presence penalty is not a truncation key: refused, never dropped.
        ({**CARD_JUDGE, "+judge_model.default_truncation.presence_penalty": 1.5},
         r"Qwen3-8B: default_truncation keys \['presence_penalty'\] are not sampler truncation keys"),
        # A malformed config is refused whatever the policy says.
        ({"model@judge_model": "qwen3-8b", "+judge_model.default_truncation.top_k": 20.5,
          "sampling.truncation": "none"}, "top_k must be an integer"),
        # A provider judge would never send it.
        ({"model@judge_model": "gpt-5.6-luna-batch", "+judge_model.default_truncation": "{top_k: 20}"},
         "default_truncation is set for a 'openai_batch' judge"),
    ],
)
def test_preflight_refuses_a_default_truncation_the_sampler_cannot_apply(tmp_path, overrides, message):
    from ruleforge.evaluate.campaign import preflight

    with pytest.raises(CampaignRunError, match=message):
        preflight(compose_campaign(tmp_path, **overrides))


def test_the_campaigns_retry_scope_and_strict_think_enter_the_condition(tmp_path, modelspec):
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    default = build_plan(compose_campaign(tmp_path), backend=backend_for("naive-rules-above"), task_spec=task_spec)
    assert (default.condition.retry_scope, default.condition.strict_think) == ("scenario_bundle", False)
    cfg = compose_campaign(
        tmp_path, campaign_root=str(tmp_path / "qwen"),
        **{"model@judge_model": "qwen3-8b", "sampling.retry_scope": "component", "sampling.strict_think": "true"},
    )
    plan = build_plan(cfg, backend=backend_for("naive-rules-above"), task_spec=task_spec)
    assert (plan.condition.retry_scope, plan.condition.strict_think) == ("component", True)
    assert plan.condition.fingerprint != dataclasses.replace(plan.condition, strict_think=False).fingerprint
    registered = json.loads((plan.condition_root / CONDITION_FNAME).read_text())
    assert (registered["retry_scope"], registered["strict_think"]) == ("component", True)
    assert "retry_scope" not in json.loads((default.condition_root / CONDITION_FNAME).read_text())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # Gemma closes its thinking with <channel|>, not </think>: every answer would be refused.
        ({}, r"judge_model gemma-4-31B-it is 'local' with enable_thinking True and strip_thinking_model 'gemma4'"),
        ({"model@judge_model": "qwen3-8b", "judge_model.enable_thinking": "false"}, "enable_thinking False"),
        ({"model@judge_model": "gpt-5.6-luna-batch"}, "is 'openai_batch'"),
    ],
)
def test_preflight_refuses_strict_think_for_a_judge_that_closes_no_think_tag(tmp_path, overrides, message):
    from ruleforge.evaluate.campaign import preflight

    cfg = compose_campaign(tmp_path, **{"sampling.strict_think": "true", **overrides})
    with pytest.raises(CampaignRunError, match="sampling.strict_think needs a local thinking judge"):
        preflight(cfg)
    with pytest.raises(CampaignRunError, match=message):
        preflight(cfg)
    config, _ = preflight(compose_campaign(tmp_path, **{"sampling.strict_think": "true", "model@judge_model": "olmo3-32b-tp-2"}))
    assert config.strict_think is True


def test_the_preflight_script_reports_the_retry_scope_and_validation_flags(tmp_path, capsys, monkeypatch):
    rules = tmp_path / "rules.jsonl"
    scenarios = write_inputs(tmp_path)
    rules.write_text("".join(json.dumps(r) + "\n" for r in RULES), encoding="utf-8")
    script = "ruleforge.launch.programs.preflight_campaign"
    monkeypatch.setattr(sys, "argv", [
        str(script), "campaign_id=camp", f"campaign_root={tmp_path / 'root'}", f"scenario_set_path={scenarios}",
        "approved_scenario_sha256=null", f"rule_set_path={rules}", "sampling.campaign_mode=pilot",
        "model@judge_model=qwen3-8b", "sampling.retry_scope=component", "sampling.strict_think=true",
        f"hydra.run.dir={tmp_path / 'hydra'}",
    ])
    monkeypatch.chdir(REPO_ROOT)  # the checkout's top, where the launcher runs the preflight
    run_program(script)
    report = json.loads(capsys.readouterr().out)
    assert (report["retry_scope"], report["strict_think"]) == ("component", True)
    assert report["label_rule"] == "lenient_last_line"
    assert not (tmp_path / "root").exists()


@pytest.mark.parametrize(("context", "warned"), [("implicated", 1), ("implicated_ranges", 0)])
def test_a_new_plan_on_the_recorded_context_warns_of_a_range_and_reading_its_records_does_not(
        tmp_path, modelspec, caplog, context, warned):
    rules = tmp_path / "ranged.jsonl"
    rules.write_text("".join(json.dumps(rule) + "\n" for rule in (
        *RULES[:2], {"id": "Meta-Rule 1", "rule": "Rules 1-2 come first"})), encoding="utf-8")
    cfg = compose_campaign(tmp_path, rule_set_path=str(rules), **{"eval_cfg.individual_rule_context": context})
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0,
                          individual_rule_context=context)
    with caplog.at_level("WARNING", logger="ruleforge.rulesets.variants"):
        plan = build_plan(cfg, backend=backend_for("naive-rules-above"), task_spec=task_spec)
    assert sum("'implicated_ranges'" in record.getMessage() for record in caplog.records) == warned
    caplog.clear()
    with caplog.at_level("WARNING", logger="ruleforge.rulesets.variants"):
        assert load_conditions(tmp_path / "root")[plan.condition.condition_id].individual_rule_context == context
    assert caplog.records == []
