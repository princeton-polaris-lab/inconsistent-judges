from __future__ import annotations

import hashlib
import json

import pytest
from omegaconf import OmegaConf

from ruleforge.judges import JudgeSpec
from ruleforge.evaluate.aggregate import analyze_campaign
from ruleforge.evaluate.campaign import build_plan, preflight, run_plan
from ruleforge.evaluate.runner import STATE_COMPLETE, CampaignRunError
from ruleforge.tasks.spec import build_task_spec
from tests.test_sampling_campaign import compose_campaign

PHI = {"yes": 1.0, "unsure": 0.5, "no": 0.0}


def stated_label(fake_seed: int, prompt_digest: str, request_seed) -> str:
    """The fake's stated function, from its statement alone."""
    payload = json.dumps([fake_seed, prompt_digest, request_seed], sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False)
    return ("yes", "no", "unsure")[hashlib.sha256(payload.encode("utf-8")).digest()[0] % 3]


def _fake_campaign(tmp_path, **overrides):
    cfg = compose_campaign(tmp_path, **overrides)
    cfg.judge_model = OmegaConf.create(JudgeSpec.fake(3).options.written())
    return cfg


def test_a_fake_campaign_s_rate_is_the_stated_function_s(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    cfg = _fake_campaign(tmp_path, **{"sampling.num_decoder_draws": 3})
    preflight(cfg)
    judge = JudgeSpec.from_config(cfg.judge_model).start()
    task_spec = build_task_spec(cfg)
    plan = build_plan(cfg, backend=judge, task_spec=task_spec)
    assert plan.condition.condition_id == "fake-3/naive-rules-above" and plan.condition.model_type == "fake"
    assert plan.condition.sampling_truncation == "provider_default" and plan.condition.seed_capability == (
        "local_deterministic")
    completions = run_plan(plan, backend=judge, task_spec=task_spec)
    assert {draw: record["state"] for draw, record in completions.items()} == dict.fromkeys(range(3), STATE_COMPLETE)

    cells = []
    for draw in range(3):
        rows = [json.loads(line) for line in
                (plan.condition_root / f"draw-{draw:03d}" / "attempts.jsonl").read_text().splitlines()]
        for row in rows:
            assert row["status"] == "accepted"
            labels = {component["component_id"]: stated_label(
                3, component["prompt_sha256"], component["generation_seed"] if component["seed_sent"] else None)
                for component in row["components"]}
            assert labels == {component["component_id"]: component["judgment"] for component in row["components"]}
            full = PHI[labels.pop("full_ruleset")]
            cells.append(float(full != max(PHI[label] for label in labels.values())))
    assert len(cells) == 3 * plan.scenario_manifest.record_count and 0 < sum(cells) < len(cells)
    payload = analyze_campaign(plan.campaign_root, n_permutations=100, n_resamples=100)
    assert payload["conditions"]["fake-3/naive-rules-above"]["estimate"] == pytest.approx(sum(cells) / len(cells))


def test_the_fake_judge_is_refused_by_a_final_campaign_s_preflight(tmp_path):
    cfg = _fake_campaign(tmp_path, **{"sampling.campaign_mode": "final"})
    with pytest.raises(CampaignRunError, match="never a 'final' one"):
        preflight(cfg)
