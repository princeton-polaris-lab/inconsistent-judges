from __future__ import annotations

import shlex
from dataclasses import dataclass

from ruleforge.core.record_names import SHARED_BANK_ESTIMATES_FNAME


@dataclass(frozen=True)
class Golden:
    """A committed result file; ``rerun_tolerance`` is the largest drift a fresh rerun may show (0.0: exact)."""

    name: str
    path: str
    rerun_tolerance: float


@dataclass(frozen=True)
class Step:
    """One command of an experiment; ``title`` names a build's log, ``copied_from`` is the file its golden copies."""

    id: str
    argv: str
    title: str = ""
    golden: Golden | None = None
    copied_from: str = ""
    inputs: tuple[str, ...] = ()
    check_only: bool = False

    @property
    def words(self) -> list[str]:
        return shlex.split(self.argv)


@dataclass(frozen=True)
class Experiment:
    """An experiment's paper figures and tables, its builds, plots and the Hub runs its goldens read, which no code may
    write to."""

    name: str
    paper: tuple[str, ...] = ()
    builds: tuple[Step, ...] = ()
    plots: tuple[Step, ...] = ()
    runs: tuple[str, ...] = ()


def _runs(*names: str) -> tuple[str, ...]:
    return tuple(f"artifact://sampling/{name}" for name in names)


def _estimates(runs: tuple[str, ...], fname: str) -> str:
    return " ".join(f"--estimates {run}/aggregate/{fname}" for run in runs)


def _plots(stem: str, argv: str, pdf_only: str = "") -> tuple[Step, Step]:
    return (Step(f"{stem}_pdf", argv.replace("EXT", "pdf") + pdf_only),
            Step(f"{stem}_png", argv.replace("EXT", "png") + " --dpi 200"))


PY = "python -m ruleforge.experiments"
CURVES = Golden("ruleset_size_curves", "results/ruleset_size/curves.json", 0.005)
REPEATED_DRAW_BASELINE = Golden("ruleset_size_repeated_draw_baseline",
                                "results/ruleset_size/repeated_draw_baseline.json", 0.005)
ABLATION = Golden("prompt_ablation", "results/prompt_ablation/prompt_ablation.json", 0.005)
LENIENCY = Golden("leniency", "results/leniency/leniency.json", 0.0)
RULEFORGE = Golden("ruleforge_final_eval", "results/ruleforge/ruleforge.json", 0.0)
GEMMA = Golden("training_gemma", "results/training/gemma.json", 0.0)
QWEN = Golden("training_qwen", "results/training/qwen.json", 0.0)
CHANGES = Golden("grpo_judgment_changes", "results/training/judgment_changes.json", 0.0)
FROZEN_ANCHORS = Golden("grpo_frozen_anchors", "results/training/frozen_anchors.json", 0.0)
HIERARCHY = Golden("hierarchy", "results/hierarchy/hierarchy.json", 0.005)
LADDER = Golden("hierarchy_ladder", "results/hierarchy/ladder.json", 0.005)

#: Each judge's root of size 57, the full ruleset in an order of each subset seed's own, and the published root its
#: analysis takes the bank and every smaller size from. The builds read the first's estimates and both roots.
RULESET_SIZE_PAIRS = (
    ("k57-shuffled-2026-11", "fig2-2026-09-pop-a22af097-rescored-lenient_last_line"),
    ("k57-shuffled-2026-11-luna", "fig2-2026-09-luna-pop-a22af097-rescored-lenient_last_line"),
    ("k57-shuffled-2026-11-haiku", "fig2-2026-10-haiku-sum-rescored-lenient_last_line"),
    ("k57-shuffled-2026-11-glimmer", "fig2-2026-10-glimmer-pop-a22af097"),
)
RULESET_SIZE_RUNS = _runs(*(size_57 for size_57, _ in RULESET_SIZE_PAIRS))
RULESET_SIZE_BANK_RUNS = _runs(*(bank for _, bank in RULESET_SIZE_PAIRS))
LENIENCY_RUNS = _runs(
    "prompt-ablation-v2-2026-10-pop-a22af097-rescored-lenient_last_line",
    "prompt-ablation-v2-open-2026-10-pop-a22af097-rescored-lenient_last_line",
    "prompt-ablation-v2-2026-10-luna-pop-a22af097-rescored-lenient_last_line",
    "prompt-ablation-v2-2026-10-haiku-sum", "prompt-ablation-v2-2026-10-glimmer-pop-a22af097")
NAIVE_LUNA_RUN = _runs("final-2026-09-luna-pop-a22af097-rescored-lenient_last_line")
#: The roots the prompt-ablation arm map and the hierarchy pairs name; a build reads them from its map, so it declares
#: them as inputs too.
PROMPT_ABLATION_RUNS = (*_runs("final-2026-09-pop-a22af097-rescored-lenient_last_line"), *NAIVE_LUNA_RUN,
                        *LENIENCY_RUNS)
HIERARCHY_RUNS = (*_runs("hierarchy-aware-2026-10-pop-a22af097", "hierarchy-aware-2026-10-glimmer",
                         "hierarchy-aware-2026-10-haiku-sum", "hierarchy-aware-2026-10-luna-sum",
                         "prompt-ablation-v2-2026-10-glimmer-pop-a22af097", "prompt-ablation-v2-2026-10-haiku-sum"),
                  *NAIVE_LUNA_RUN)
RULEFORGE_RUNS = _runs("fig4-2026-09-pop-a22af097-rescored-lenient_last_line")
GEMMA_RUNS = _runs("fig5-gemma-2026-10-07-pop-a22af097", "fig5-gemma-ckpt600-2026-10-09-pop-a22af097")
QWEN_RUNS = _runs("fig5-grpo-v2-fp32-2026-09-26-checkpoint-200-pop-a22af097",
                  "fig5-qwen-mix-2026-10-11-checkpoint-300-pop-a22af097")
GEMMA_SELECTION = "artifact://selection/gemma-grpo-mix-2026-10-04"
QWEN_SELECTION = "artifact://selection/qwen-grpo-mix-2026-10-11"
RULES = "repo://data/rules/modelspec/openai_rules_all.jsonl"

TABLE = (
    Experiment(
        "ruleset-size", ("Figure 3", "Figure 10"), runs=(*RULESET_SIZE_RUNS, *RULESET_SIZE_BANK_RUNS),
        builds=(Step("ruleset_size", f"{PY}.ruleset_size.build {_estimates(RULESET_SIZE_RUNS, SHARED_BANK_ESTIMATES_FNAME)} "
                                     f"--golden {{out}}/{CURVES.path}", "ruleset size", CURVES,
                     inputs=RULESET_SIZE_BANK_RUNS),
                Step("ruleset_size_repeated_draw_baseline",
                     f"{PY}.ruleset_size.build_repeated_draw_baseline "
                     f"{_estimates(RULESET_SIZE_RUNS, SHARED_BANK_ESTIMATES_FNAME)} "
                     f"--golden {{out}}/{REPEATED_DRAW_BASELINE.path}", "ruleset size repeated-draw baseline",
                     REPEATED_DRAW_BASELINE, inputs=RULESET_SIZE_BANK_RUNS),
                Step("ruleset_size_subsets", f"{PY}.ruleset_size.verify_subsets", "rule subsets", check_only=True)),
        plots=(*_plots("ruleset_size_curves", f"{PY}.ruleset_size.build --from_golden repo://{CURVES.path} "
                                              "--figure {out}/ruleset_size_curves.EXT"),
               *_plots("ruleset_size_repeated_draw_baseline",
                       f"{PY}.ruleset_size.build_repeated_draw_baseline --from_golden "
                       f"repo://{REPEATED_DRAW_BASELINE.path} --figure {{out}}/ruleset_size_repeated_draw_baseline.EXT",
                       " --csv {out}/ruleset_size_repeated_draw_baseline.csv"))),
    Experiment(
        "prompt-ablation", ("Figure 4", "Table 3", "Appendix E.1"), runs=PROMPT_ABLATION_RUNS,
        builds=(Step("prompt_ablation", f"{PY}.prompt_ablation.build --arm-map repo://configs/paper/prompt_ablation_arms.yaml "
                                        f"--arm-map-var pop=pop-a22af097 --golden {{out}}/{ABLATION.path}",
                     "prompt ablation", ABLATION, inputs=PROMPT_ABLATION_RUNS),),
        plots=(*_plots("prompt_ablation", f"{PY}.prompt_ablation.build --from_golden repo://{ABLATION.path} "
                                          "--figure_styles {out}/prompt_styles.EXT "
                                          "--figure_methods {out}/method_consistency.EXT",
                       " --csv {out}/prompt_ablation_table.csv --latex {out}/prompt_ablation_table.tex"),
               Step("prompt_templates", f"{PY}.prompt_ablation.templates --output-dir {{out}}/prompt_templates"))),
    Experiment(
        "leniency", ("Figure 1", "Figure 9"), runs=LENIENCY_RUNS,
        builds=(Step("leniency", f"{PY}.leniency.build --prompt-style forced {_estimates(LENIENCY_RUNS, 'estimates.json')} "
                                 f"--golden {{out}}/{LENIENCY.path}", "leniency", LENIENCY),),
        plots=(Step("leniency", f"{PY}.leniency.build --from_golden repo://{LENIENCY.path} --csv {{out}}/leniency.csv "
                                "--figure {out}/leniency.pdf --figure {out}/leniency.png "
                                "--figure_all {out}/leniency_all_rules.pdf --figure_all {out}/leniency_all_rules.png "
                                "--dpi 200"),)),
    Experiment(
        "ruleforge", ("Figure 6",), runs=RULEFORGE_RUNS,
        builds=(Step("ruleforge", f"{PY}.ruleforge.build {_estimates(RULEFORGE_RUNS, 'estimates.json')} "
                                  f"--golden {{out}}/{RULEFORGE.path}", "RuleForge", RULEFORGE),),
        plots=_plots("ruleforge_eval", f"{PY}.ruleforge.plot --from_golden repo://{RULEFORGE.path} "
                                       "--figure_a {out}/ruleforge_eval_inconsistency.EXT "
                                       "--figure_b {out}/ruleforge_eval_changes.EXT")),
    Experiment(
        "training", ("Figure 5", "Figure 11", "Figure 12", "Figure 13"), runs=(*GEMMA_RUNS, *QWEN_RUNS),
        builds=(Step("gemma", f"{PY}.training.build --family gemma --population pop-a22af097 "
                              "--out-dir {out}/gemma", "Gemma family", GEMMA, "{out}/gemma/training_family.json",
                     (*GEMMA_RUNS, f"{GEMMA_SELECTION}/modal_root/finetuning/gemma-grpo-mix-2026-10-04/merged/"
                                   "checkpoint-600/merge_manifest.json", RULES)),
                Step("qwen", f"{PY}.training.build --family qwen --population pop-a22af097 --out-dir {{out}}/qwen",
                     "Qwen family", QWEN, "{out}/qwen/training_family.json",
                     (*QWEN_RUNS, f"{QWEN_SELECTION}/modal_root/finetuning/qwen-grpo-mix-2026-10-11/merged/"
                                  "checkpoint-300/merge_manifest.json",
                      f"{QWEN_SELECTION}/selection/checkpoints/checkpoint-300/parity/prefill_report.json",
                      f"{QWEN_SELECTION}/selection/checkpoints/checkpoint-300/parity/decode_report.json", RULES)),
                Step("judgment_changes", f"{PY}.training.judgment_changes --family gemma --family qwen "
                                         f"--population pop-a22af097 --golden {{out}}/{CHANGES.path}",
                     "judgment changes", CHANGES, inputs=(*GEMMA_RUNS, *QWEN_RUNS)),
                Step("frozen_anchors", f"{PY}.training.frozen_anchors --family gemma --family qwen "
                                       f"--population pop-a22af097 --golden {{out}}/{FROZEN_ANCHORS.path}",
                     "frozen anchors", FROZEN_ANCHORS, inputs=(*GEMMA_RUNS, *QWEN_RUNS))),
        plots=(*_plots("gemma", f"{PY}.training.build --from_golden repo://{GEMMA.path} "
                                "--figure-a {out}/grpo_inconsistency.EXT"),
               *_plots("qwen", f"{PY}.training.build --from_golden repo://{QWEN.path} "
                               "--figure-a {out}/grpo_qwen_inconsistency.EXT"),
               *_plots("judgment_changes", f"{PY}.training.judgment_changes --from_golden repo://{CHANGES.path} "
                                           "--paper gemma --figure-changes {out}/grpo_judgment_changes.EXT "
                                           "--figure-counts {out}/grpo_judgment_counts.EXT"),
               *_plots("judgment_changes_qwen",
                       f"{PY}.training.judgment_changes --from_golden repo://{CHANGES.path} "
                       "--paper qwen --figure-changes {out}/grpo_qwen_judgment_changes.EXT "
                       "--figure-counts {out}/grpo_qwen_judgment_counts.EXT"),
               *_plots("frozen_anchors", f"{PY}.training.frozen_anchors --from_golden repo://{FROZEN_ANCHORS.path} "
                                         "--paper gemma --figure-rates {out}/grpo_frozen_anchors.EXT"),
               *_plots("frozen_anchors_qwen",
                       f"{PY}.training.frozen_anchors --from_golden repo://{FROZEN_ANCHORS.path} "
                       "--paper qwen --figure-rates {out}/grpo_qwen_frozen_anchors.EXT"))),
    Experiment(
        "hierarchy", ("Figure 7", "Figure 8"), runs=HIERARCHY_RUNS,
        builds=(Step("hierarchy", f"{PY}.hierarchy.build --pairs repo://configs/paper/hierarchy_pairs.yaml "
                                  f"--golden {{out}}/{HIERARCHY.path}", "hierarchy appendix", HIERARCHY,
                     inputs=HIERARCHY_RUNS),
                Step("hierarchy_ladder", f"{PY}.hierarchy.build_ladder --pairs repo://configs/paper/hierarchy_pairs.yaml "
                                         f"--golden {{out}}/{LADDER.path}", "hierarchy ladder", LADDER,
                     inputs=HIERARCHY_RUNS)),
        plots=(*_plots("hierarchy", f"{PY}.hierarchy.build --from_golden repo://{HIERARCHY.path} "
                                    "--figure {out}/hierarchy.EXT"),
               *_plots("hierarchy_ladder", f"{PY}.hierarchy.build_ladder --from_golden repo://{LADDER.path} "
                                           "--arm hierarchy --figure {out}/hierarchy_ladder.EXT",
                       " --csv {out}/hierarchy_ladder.csv"))),
)


def goldens() -> list[Golden]:
    """Return every golden in the table's order."""
    return [step.golden for row in TABLE for step in row.builds if step.golden is not None]


def published_runs() -> dict[str, tuple[str, ...]]:
    """Return each run the goldens read with the experiments that read it."""
    found: dict[str, tuple[str, ...]] = {}
    for row in TABLE:
        for run in row.runs:
            found[run] = (*found.get(run, ()), row.name)
    return found
