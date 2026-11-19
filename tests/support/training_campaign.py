from __future__ import annotations

import json
from pathlib import Path

import pytest

from ruleforge.experiments.training.figure import build_figure, panel_values
from ruleforge.grpo import merging as merge_module
from ruleforge.grpo.lora_parity import thresholds_to_json
from tests.support.parity import preset_thresholds
from ruleforge.evaluate import aggregate as aggregate_module
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate.campaign import build_plan, run_plan
from ruleforge.core.digests import sha256_file
from tests.conftest import REPO_ROOT
from tests.test_sampling_campaign import set_recorded_runtime
from tests.test_sampling_runner import forced
from tests.test_sampling_weights import FP32_ENGINE, local_backend, weights_campaign, write_base, write_merged

#: The committed golden of :func:`run_training_campaign`'s root (tests/test_training_campaign_readout.py reproduces it).
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "training" / "synthetic_readout.json"
POLICIES = ("base", "grpo", "final")
N_SCENARIOS = 12
N_DRAWS = 2
#: The one scenario whose per-rule max is no in every arm.
M_NO = 11
#: One recorded stack for every draw, as a real fp32 campaign would record it.
RUNTIME = {
    "vllm_version": "0.28.0+cu129", "torch_version": "2.13.0+cu129", "transformers_version": "5.17.0",
    "tokenizers_version": "0.22.1", "triton_version": "3.7.1",
    "precision_env": {"VLLM_FLOAT32_MATMUL_PRECISION": "highest", "TRITON_F32_DEFAULT": "unset",
                      "VLLM_BATCH_INVARIANT": "0"},
}
QWEN = {"config": "qwen3-8b-fp32", "name": "Qwen3-8B", "prefix": "qwen/figure5", "strict_think": True}
GEMMA = {"config": "gemma4-31b-fp32", "name": "gemma-4-31B-it", "prefix": "gemma-4-31B-it/figure5", "strict_think": False}
PARITY = {"mode": "merged", "segment": "all", "passed": True, "failures": [], "rho_gated": {"all": True},
          "thresholds": thresholds_to_json(preset_thresholds("b")),
          "buckets": {"all": {"rho": 1.0001, "rho_ci": [0.9996, 1.0009], "rho_halfwidth": 0.00065}}}


def source_of(index: int) -> str:
    return "stress" if index % 2 == 0 else "wildchat"


def scripted_answers(policy: str, *, degenerate: bool = False) -> dict[tuple[int, int], dict[str, object]]:
    """``{(scenario, draw): {"F", "Rule 1", "Rule 2", "map"}}``: what the fake judge answers."""
    table = {}
    for i in range(N_SCENARIOS):
        for d in range(N_DRAWS):
            rule1 = "no" if i == M_NO else "yes"
            rule2 = "no"
            if policy == "base":
                full = "unsure" if (i, d) == (0, 1) else "no"
                rule_map = None if (i, d) in {(2, 0), (5, 0)} else {"Rule 1": "no", "Rule 2": "no"}
            elif policy == "grpo":
                if (i, d) == (4, 1):
                    rule1, rule2 = "no", "unsure"
                full = "unsure" if rule2 == "unsure" else rule1
                rule_map = {"Rule 1": full, "Rule 2": "no"}
            else:
                if degenerate:
                    rule1 = "yes"
                elif (i, d) == (6, 1):
                    rule1 = "no"
                full = "yes"
                rule_map = None if i == 8 else {"Rule 1": "no", "Rule 2": "no"}
            table[(i, d)] = {"F": full, "Rule 1": rule1, "Rule 2": rule2, "map": rule_map}
    return table


def _script(policy: str, *, degenerate: bool = False) -> dict[tuple[int, str], list]:
    table = scripted_answers(policy, degenerate=degenerate)
    script: dict[tuple[int, str], list] = {}
    for i in range(N_SCENARIOS):
        full = [
            forced(table[(i, d)]["F"], rules=table[(i, d)]["map"] or {"Rule 1": "no"}) for d in range(N_DRAWS)
        ]
        script[(i, "full_ruleset")] = (["not json"] if (policy, i) == ("grpo", 3) else []) + full
        for rule in ("Rule 1", "Rule 2"):
            script[(i, rule)] = [forced(table[(i, d)][rule]) for d in range(N_DRAWS)]
    return script


def write_population(tmp_path: Path) -> Path:
    path = tmp_path / "training_population.jsonl"
    path.write_text(
        "".join(
            json.dumps({"prompt": f"prompt {i}", "response": f"response {i}", "source": source_of(i)}) + "\n"
            for i in range(N_SCENARIOS)
        ),
        encoding="utf-8",
    )
    return path


def run_training_campaign(tmp_path: Path, modelspec, policies=POLICIES, *, degenerate_final: bool = False,
                         judge=QWEN, family: str | None = None, campaign_id: str = "camp",
                         population: Path | None = None, campaign_root: Path | None = None,
                         seed_base: int | None = None, campaign_mode: str | None = None) -> Path:
    """The synthetic training root, analysed, its manifest then marked final (as a 500-scenario one would be)."""
    approved = None if population is None else sha256_file(population)
    population = population or write_population(tmp_path)
    root = campaign_root or tmp_path / "root"
    base = write_base(tmp_path / "models", judge["name"])
    merged_from = {"base": {"path": str(base), "weights_sha256": merge_module.weights_digest(base)}}
    for policy in policies:
        weights = base if policy == "base" else write_merged(
            tmp_path / "grpo" / "merged", policy, seed=float(POLICIES.index(policy)), manifest=merged_from)
        extra = {"figure": "figure5", "policy": policy, "policy_family": family or "+".join(policies),
                 "policy_reference": "base"}
        cfg = weights_campaign(
            tmp_path, weights, **{
                "campaign_id": campaign_id,
                "model@judge_model": judge["config"],
                "scenario_set_path": str(population),
                "condition_id": f"{judge['prefix']}/{policy}",
                "condition_extra": "{" + ", ".join(f"{k}: {v}" for k, v in extra.items()) + "}",
                "eval_cfg.reasoning_prompt_style": "forced-grpo",
                "sampling.forced_json_requirement": "label_only",
                "sampling.strict_think": str(judge["strict_think"]).lower(),
                "sampling.retry_scope": "component",
                "campaign_root": str(root),
                **({} if approved is None else {"approved_scenario_sha256": approved}),
                **({} if seed_base is None else {"sampling.generation_seed_base": seed_base}),
                **({} if campaign_mode is None else {"sampling.campaign_mode": campaign_mode}),
            },
        )
        backend = local_backend(model_path=weights, dtype="float32", resolved=FP32_ENGINE, style="forced-grpo",
                                script=_script(policy, degenerate=degenerate_final))
        backend.model_name = judge["name"]
        backend.max_new_tokens = 1024
        backend.engine_args.update({"attention_config": {"backend": "TRITON_ATTN"}, "max_model_len": None,
                                    "enable_lora": False})
        backend.attention_backend = "TRITON_ATTN"
        task_spec = modelspec(reasoning_prompt_style="forced-grpo", evaluation_temperature=1.0)
        run_plan(build_plan(cfg, backend=backend, task_spec=task_spec), backend=backend, task_spec=task_spec)
    for policy in policies:
        set_recorded_runtime(root / judge["prefix"] / policy, **RUNTIME)
    return root


def mark_final(root: Path) -> None:
    path = root / "sampling_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["campaign_mode"] = "final"
    path.write_text(json.dumps(manifest))


def write_parity(tmp_path: Path, name: str = "grpo", arm: str = "prefill", **changes) -> Path:
    path = tmp_path / "parity" / name / f"{arm}_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    segment = {"prefill": "all", "decode": "completion"}[arm]
    path.write_text(json.dumps({**PARITY, "segment": segment, **changes}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    # The campaign root sits under the artifact root, so the golden names it by artifact:// URI.
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    for name in RUNTIME["precision_env"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(campaign_module, "code_state", lambda: {})
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)
    monkeypatch.setattr(aggregate_module, "utc_now", lambda: "2026-09-25T00:00:00+00:00")


def assert_panel_a_is_bare(figure) -> None:
    """Panel a draws its bars, their whiskers and their value labels, and nothing else."""
    from matplotlib.collections import LineCollection
    from matplotlib.container import BarContainer, ErrorbarContainer

    figure.canvas.draw()
    (axes,) = figure.axes
    assert axes.get_legend() is None and not figure.legends and not figure.texts
    assert not axes.images and not axes.tables and not axes.artists
    whiskers, bars = axes.containers
    assert isinstance(whiskers, ErrorbarContainer) and isinstance(bars, BarContainer)
    assert list(axes.patches) == list(bars)
    assert [type(collection) for collection in axes.collections] == [LineCollection]
    assert all(dashes is None for _, dashes in axes.collections[0].get_linestyle())
    assert [(line.get_marker(), line.get_linestyle()) for line in axes.lines] == [("_", "None")] * 2
    boxes = [text.get_window_extent() for text in axes.texts]
    assert len(boxes) == len(bars)
    assert all(figure.bbox.x0 <= box.x0 and box.x1 <= figure.bbox.x1 and figure.bbox.y0 <= box.y0
               and box.y1 <= figure.bbox.y1 for box in boxes)
    assert not any(left.overlaps(right) for index, left in enumerate(boxes) for right in boxes[index + 1:])
    assert tuple(figure.get_size_inches()) == (4.6, 3.4) and figure.subplotpars.top == 0.95
    assert axes.get_ylim() == (0.0, 1.12) and list(axes.get_yticks()) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def assert_panel_a_draws_the_flat_rates_alone(golden, value_labels, tick_labels=None, names=None) -> None:
    """Panel a of ``golden``: each arm's ``flat`` row at its estimate, bare, with its value and tick labels."""
    import matplotlib.pyplot as plt

    flat = {golden["arms"][policy]["label"]: golden["arms"][policy]["flat"] for policy in golden["policies"]}
    assert panel_values(golden, "a") == flat
    figure = build_figure(golden, "a", names)
    axes = figure.axes[0]
    assert [patch.get_height() for patch in axes.patches] == pytest.approx([row["estimate"] for row in flat.values()])
    assert_panel_a_is_bare(figure)
    assert [text.get_text() for text in axes.texts] == value_labels
    if tick_labels is not None:
        assert [(text.get_text(), text.get_fontsize()) for text in axes.get_xticklabels()] == tick_labels
    plt.close(figure)


def _edit(path: Path, change) -> None:
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload))

