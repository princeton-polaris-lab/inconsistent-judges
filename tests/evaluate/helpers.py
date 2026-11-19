from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from omegaconf import OmegaConf

from ruleforge.judges import JudgeSpec
from ruleforge.rulesets import Ruleset
from ruleforge.evaluate.aggregate import analyze_campaign
from ruleforge.evaluate.campaign import build_plan, preflight, run_plan
from ruleforge.evaluate.configured import MARKER_FNAME as CONFIGURED_MARKER_FNAME
from ruleforge.scenarios import ScenarioSet
from ruleforge.tasks.spec import build_task_spec
from tests.support.trees import tree_files
from tests.test_sampling_runner import RULES, write_inputs

PHI = {"yes": 1.0, "unsure": 0.5, "no": 0.0}


def write_rules(path: Path, rules=RULES) -> Ruleset:
    path.write_text("".join(json.dumps(rule) + "\n" for rule in rules), encoding="utf-8")
    return Ruleset.load(path)


def inputs(tmp_path: Path) -> tuple[ScenarioSet, Ruleset]:
    """Three scenarios and a two-rule ruleset with a meta-rule."""
    return ScenarioSet.load(write_inputs(tmp_path)), write_rules(tmp_path / "rules.jsonl")


def tree(root: Path) -> dict[str, str]:
    """Every file under ``root`` by relative path, with its content's digest."""
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tree_files(root)}


def stated_label(fake_seed: int, prompt_digest: str, request_seed) -> str:
    """The fake judge's stated function, from its statement alone."""
    payload = json.dumps([fake_seed, prompt_digest, request_seed], sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False)
    return ("yes", "no", "unsure")[hashlib.sha256(payload.encode("utf-8")).digest()[0] % 3]


def recorded_cells(condition_root: Path, n_draws: int, fake_seed: int) -> dict[tuple[str, int], float]:
    """Each (scenario, draw)'s inconsistency, recomputed from the ledger's recorded prompts and seeds."""
    cells = {}
    for draw in range(n_draws):
        for line in (condition_root / f"draw-{draw:03d}" / "attempts.jsonl").read_text().splitlines():
            row = json.loads(line)
            labels = {component["component_id"]: stated_label(
                fake_seed, component["prompt_sha256"], component["generation_seed"] if component["seed_sent"] else None)
                for component in row["components"]}
            assert labels == {component["component_id"]: component["judgment"] for component in row["components"]}
            full = PHI[labels.pop("full_ruleset")]
            cells[(row["scenario_id"], draw)] = float(full != max(PHI[label] for label in labels.values()))
    return cells


def cluster_bootstrap(values: np.ndarray, *, n_resamples: int, ci_level: float, seed: int) -> tuple[float, float]:
    """The scenario-cluster percentile bootstrap of an ``(N, S)`` grid, written out."""
    n, s = values.shape
    means = values.mean(axis=1)
    rng = np.random.default_rng(seed)
    replicates = []
    for start in range(0, n_resamples, 500):
        size = min(500, n_resamples - start)
        picks = rng.integers(0, n, size=(size, n))
        rng.integers(0, s, size=(size, n, s))
        replicates.append(means[picks].mean(axis=1))
    values = np.concatenate(replicates)
    alpha = 1.0 - ci_level
    return float(np.percentile(values, alpha / 2 * 100)), float(np.percentile(values, (1 - alpha / 2) * 100))


class Rewritten:
    """A started judge whose answers ``answer(prompt entry, outcome)`` rewrites; everything else is the judge's."""

    def __init__(self, inner: Any, answer: Callable[[dict, Any], Optional[str]]):
        self.inner = inner
        self.answer = answer

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def generate_once(self, prompts, **kwargs):
        outcomes = self.inner.generate_once(prompts, **kwargs)
        rewritten = []
        for prompt, outcome in zip(prompts, outcomes):
            text = self.answer(prompt, outcome)
            rewritten.append(outcome if text is None else dataclasses.replace(outcome, text=text, raw_response=text))
        return rewritten


def rewrite_answers(monkeypatch, answer: Callable[[dict, Any], Optional[str]]) -> list:
    """Every judge this test starts answers through ``answer``; the list collects the started judges."""
    started = []
    real = JudgeSpec.start

    def start(self):
        judge = Rewritten(real(self), answer)
        started.append(judge)
        return judge

    monkeypatch.setattr(JudgeSpec, "start", start)
    return started


def count_starts(monkeypatch) -> list:
    """Record every judge this test starts (the judges themselves start as they would)."""
    started = []
    real = JudgeSpec.start
    monkeypatch.setattr(JudgeSpec, "start", lambda self: started.append(self) or real(self))
    return started


def configured_root(tmp_path: Path, *, fake_seed: int = 3, **overrides) -> Path:
    """A campaign root built and analysed the configured way."""
    from tests.test_sampling_campaign import compose_campaign

    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = compose_campaign(tmp_path, **overrides)
    cfg.judge_model = OmegaConf.create(JudgeSpec.fake(fake_seed).options.written())
    preflight(cfg)
    judge = JudgeSpec.from_config(cfg.judge_model).start()
    task_spec = build_task_spec(cfg)
    plan = build_plan(cfg, backend=judge, task_spec=task_spec)
    run_plan(plan, backend=judge, task_spec=task_spec)
    analyze_campaign(plan.campaign_root, n_permutations=100)
    (plan.campaign_root / CONFIGURED_MARKER_FNAME).unlink()
    return plan.campaign_root
