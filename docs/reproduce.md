# Reproducing the paper

How to reproduce *Aligning Judge Models to Be Logically Consistent Multi-Rule Interpreters*. Find a figure or table in
[the table below](#the-figures-and-tables), then go to its experiment's section for the data and the commands.

| Level | What it does | Needs | Time |
|---|---|---|---|
| **Render** | draws the figures and tables from the committed goldens | nothing | about a minute |
| **Rebuild** | recomputes the goldens from the published campaign roots on the Hub | the dataset's download | a few CPU minutes |
| **Rerun** | judges the scenarios again, and trains again; a new sample, compared within tolerances | GPUs, paid APIs | hours to days |

A **golden** is a committed JSON file under `results/` that holds an experiment's numbers. A **campaign root** is the
directory of one campaign's judgments, its records and its estimates. The paper's roots are read as recorded and never
extended; a rerun ([rerun.md](rerun.md)) writes roots of its own.

- [Setup](#setup), [The figures and tables](#the-figures-and-tables), [The data on the Hub](#the-data-on-the-hub),
  [Rebuild every golden](#rebuild-every-golden)
- Experiments: [Leniency](#leniency), [Ruleset size](#ruleset-size), [Prompt ablation](#prompt-ablation),
  [RuleForge](#ruleforge), [Training](#training), [Hierarchy](#hierarchy)

## Setup

```bash
uv sync                                                      # no GPU required
export RULEFORGE_ARTIFACT_ROOT=/path/to/an/empty/directory   # the data lands here
```

Run everything from the repository root.

## The figures and tables

`ruleforge reproduce <experiment> render|rebuild|verify` runs the row's experiment, and a render or a rebuild
first names the figures and tables it reproduces; `all` runs every one.

| Paper | Experiment | Golden | Files from `render` | Hub folder |
|---|---|---|---|---|
| Figure 1 (14 rules), Figure 9 (all 57) | `leniency` | `results/leniency/leniency.json` | `leniency`, `leniency_all_rules`, `leniency.csv` | `prompt-ablation/` |
| Figure 2 | none: an illustration | | | |
| Figure 3 | `ruleset-size` | `results/ruleset_size/curves.json` | `ruleset_size_curves` | `ruleset-size/` |
| Figure 10: the repeated-draw baseline | `ruleset-size` | `results/ruleset_size/repeated_draw_baseline.json` | `ruleset_size_repeated_draw_baseline`, `ruleset_size_repeated_draw_baseline.csv` | `ruleset-size/` |
| Figure 4a, Figure 4b, Table 3 | `prompt-ablation` | `results/prompt_ablation/prompt_ablation.json` | `prompt_styles`, `method_consistency`, `prompt_ablation_table.{csv,tex}` | `prompt-ablation/` |
| Figure 5a-c (Gemma-4-31B-it), Figure 11 (its frozen-anchor rates) | `training` | `results/training/gemma.json`, `results/training/judgment_changes.json`, `results/training/frozen_anchors.json` | `grpo_inconsistency`, `grpo_judgment_changes`, `grpo_judgment_counts`, `grpo_frozen_anchors` | `training/` |
| Figure 6a, 6b | `ruleforge` | `results/ruleforge/ruleforge.json` | `ruleforge_eval_inconsistency`, `ruleforge_eval_changes` | `ruleforge/` |
| Figure 7 | `hierarchy` | `results/hierarchy/hierarchy.json` | `hierarchy` | `hierarchy/`, plus three `prompt-ablation/` roots |
| Figure 8: inconsistency at hierarchy level h (hierarchy prompt) | `hierarchy` | `results/hierarchy/ladder.json` | `hierarchy_ladder`, `hierarchy_ladder.csv` | `hierarchy/`, plus three `prompt-ablation/` roots |
| Figure 12a-c (Qwen3-8B), Figure 13 (its frozen-anchor rates) | `training` | `results/training/qwen.json`, `results/training/judgment_changes.json`, `results/training/frozen_anchors.json` | `grpo_qwen_inconsistency`, `grpo_qwen_judgment_changes`, `grpo_qwen_judgment_counts`, `grpo_qwen_frozen_anchors` | `training/` |
| Table 2 (the 57 rules) | none | `data/rules/modelspec/openai_rules_all.jsonl` | | |
| Appendix B.1: the hierarchy prompt | none | `tests/fixtures/hierarchy_prompts/` | | |
| Appendix E.1: the judge prompts | `prompt-ablation` | `ruleforge prompts show naive-rules-above` (or `encouraged`, `forced`, `forced-grpo`) | `prompt_templates/*.txt` | |
| Appendix G.2: the revised rulesets | none | `data/rules/modelspec/ruleforge/*.jsonl` | | `ruleforge/` |

```bash
uv run ruleforge reproduce all render --out /tmp/paper-figures
uv run ruleforge reproduce ruleset-size render --out /tmp/ruleset-size
```

Each figure renders as `.pdf` and `.png`. CSV, TeX and text files match byte for byte; PNGs match only under the
locked Matplotlib and its bundled font. The numbers match everywhere.

## The data on the Hub

The dataset `ai-law-society-lab/inconsistent-judges` holds one folder per experiment. Each campaign root is an archive,
`roots/<root>.tar.zst`, with a `.files.sha256` beside it. RuleForge's seed scenarios are in this repository
(`data/scenarios/seeds/`).

```bash
uv run ruleforge data fetch --dry-run                    # what it would place, and where
uv run ruleforge data fetch                              # everything
uv run ruleforge data fetch ruleset-size training        # some experiments, or single roots by name
uv run ruleforge data fetch --model ai-law-society-lab/inconsistent-judges   # the adapters, for training reruns
uv run ruleforge data verify --root "$RULEFORGE_ARTIFACT_ROOT"               # every placed file, again
```

| Fetch rule | |
|---|---|
| Checks | every file against `SHA256SUMS` before anything is placed; no token needed |
| Existing data | nothing is written over a destination; a root already there is refused |
| Download | held in a hidden directory under the root, never in the Hugging Face cache |
| Contents | each root's **compact tier**: the files, rows and fields the rebuilds read, and a `published_copy.json`; the inputs whole ([data/README.md](../data/README.md#the-data-on-the-hugging-face-hub)) |
| Other commands | `campaign cost usage`, `analyze`, `bank analyze` and a cap amendment's disclosure refuse a fetched root and run on a root you ran |
| Names | recorded files keep the names and keys the campaigns ran with; [data/README.md](../data/README.md) maps them |

## Rebuild every golden

```bash
uv run ruleforge reproduce all rebuild --out /tmp/paper-rebuild
uv run ruleforge reproduce all verify --rerun /tmp/paper-rebuild --exact   # ends with ALL 11 EQUAL, or fails
```

`rebuild` writes each golden under `DIR/results/` and each log under `DIR/logs/`; it never writes into the checkout.

| Exactness | |
|---|---|
| Exact on | this lock and OpenBLAS's AVX-512 kernel; the rebuild sets the thread counts and `SOURCE_DATE_EPOCH` it needs |
| Elsewhere | a few bootstrap bounds of the training goldens move in their last digit (about 1e-16) |
| Then | `verify` reports those bounds as `drift` and exits 0, `verify --exact` exits 1; a larger move is a real difference |

## Leniency

Under the `forced` prompt: each rule's judgment inside the whole-ruleset answer against its judgment alone.

```bash
uv run ruleforge reproduce leniency render --out DIR
uv run ruleforge data fetch leniency
uv run ruleforge reproduce leniency rebuild --out DIR
```

| | |
|---|---|
| Golden | `results/leniency/leniency.json` |
| Roots | the prompt-ablation roots' forced conditions (their accepted attempts, `accepted_attempts.json`), so `data fetch leniency` places `prompt-ablation/` |
| Rerun | the prompt ablation's |

## Ruleset size

The inconsistency rate as the ruleset grows from 2 to all 57 rules. Each judge reuses one **shared bank** of
single-rule judgments; each size has ten sampled subsets (subset seeds), five draws each. At 57 each of the ten seeds
judges the full ruleset in an order of its own, so the point averages over ten orders; the bank roots' own size-57
calls use the file's order for every seed.

```bash
uv run ruleforge reproduce ruleset-size render --out DIR
uv run ruleforge data fetch ruleset-size
uv run ruleforge reproduce ruleset-size rebuild --out DIR
```

| | |
|---|---|
| Goldens | `results/ruleset_size/curves.json`: each judge's rate by size. `results/ruleset_size/repeated_draw_baseline.json`: the rate beside its repeated-draw baseline, by size |
| Rule subsets | `ruleforge reproduce ruleset-size subsets` regenerates the 1,400 subsets (207 MB) from `data/rules/subset_manifest/`; `--check` verifies them. Size 57's orders are pinned in `data/rules/subset_manifest_shuffled_full/` |

| Judges | Bank and sizes 2 to 56 | Size 57 |
|---|---|---|
| Gemma-4-31B-it, Olmo-3-32B-Think, gpt-oss-120b | `fig2-2026-09-pop-a22af097-rescored-lenient_last_line` | `k57-shuffled-2026-11` |
| GPT-5.6 Luna | `fig2-2026-09-luna-pop-a22af097-rescored-lenient_last_line` | `k57-shuffled-2026-11-luna` |
| Claude Haiku 5.5 | `fig2-2026-10-haiku-sum-rescored-lenient_last_line` | `k57-shuffled-2026-11-haiku` |
| Muse-Glimmer-30B | `fig2-2026-10-glimmer-pop-a22af097` | `k57-shuffled-2026-11-glimmer` |

The **repeated-draw baseline** reads the same draws: per cell (size, scenario, subset seed), how often two draws of the
whole-ruleset call disagree and how often two draws of the separate-call maximum disagree, averaged. That average is
the mean of the rate expected if the whole-ruleset judgment were one more draw of the separate-call maximum and the
rate expected if the maximum were one more draw of the whole-ruleset judgment.

The **median inconsistency rate**, which the figure also draws, takes each rule's median judgment over its five bank
draws, then the strongest of those medians, and compares it with the whole-ruleset call's median judgment over its
five draws.

Read the gap between the rate and the baseline from the excess's own interval in the CSV, not from whether the two
bands overlap: they are correlated across resamples. The golden defines each statistic, the null and the interval.

## Prompt ablation

Six judges under the three prompts (`naive-rules-above`, `encouraged`, `forced`), five draws each. The golden holds the
separate-call rate (the paper's inconsistency rate) and, for `forced`, the same-call rate
([measure.md](measure.md#what-a-run-writes)).

```bash
uv run ruleforge reproduce prompt-ablation render --out DIR
uv run ruleforge data fetch prompt-ablation
uv run ruleforge reproduce prompt-ablation rebuild --out DIR
```

| | |
|---|---|
| Golden | `results/prompt_ablation/prompt_ablation.json` |
| Arm map | `configs/paper/prompt_ablation_arms.yaml` names each **arm**'s root (an arm is one judge under one prompt); copy it with your roots for a rerun |

| Judges | Prompts | Root |
|---|---|---|
| Olmo-3-32B-Think, Gemma-4-31B-it, gpt-oss-120b | `naive-rules-above` | `final-2026-09-pop-a22af097-rescored-lenient_last_line` |
| Gemma-4-31B-it | `encouraged`, `forced` | `prompt-ablation-v2-2026-10-pop-a22af097-rescored-lenient_last_line` |
| Olmo-3-32B-Think, gpt-oss-120b | `encouraged`, `forced` | `prompt-ablation-v2-open-2026-10-pop-a22af097-rescored-lenient_last_line` |
| GPT-5.6 Luna | `naive-rules-above` | `final-2026-09-luna-pop-a22af097-rescored-lenient_last_line` |
| GPT-5.6 Luna | `encouraged`, `forced` | `prompt-ablation-v2-2026-10-luna-pop-a22af097-rescored-lenient_last_line` |
| Claude Haiku 5.5 | all three | `prompt-ablation-v2-2026-10-haiku-sum` |
| Muse-Glimmer-30B | all three | `prompt-ablation-v2-2026-10-glimmer-pop-a22af097` |

This build, the hierarchy comparison and its ladder refuse any prompt but the paper's, so a `naive` arm, the default
for new runs, stops them.

## RuleForge

The initial 57-rule ruleset against the winner and runner-up of the published ten-seed sweep
([ruleforge.md](ruleforge.md) says how they are picked), each judged by Gemma-4-31B-it under `naive-rules-above`, five
draws.

```bash
uv run ruleforge reproduce ruleforge render --out DIR
uv run ruleforge data fetch ruleforge
uv run ruleforge reproduce ruleforge rebuild --out DIR
uv run ruleforge optimize select \
    --sweep-dir $RULEFORGE_ARTIFACT_ROOT/ruleforge/sweep/modelspec/openai/2026-07-08/23-45-41 \
    --output DIR/selection.json   # job 4 the winner, job 8 the runner-up
```

| | |
|---|---|
| Golden | `results/ruleforge/ruleforge.json`, from the published sweep |
| Root | `fig4-2026-09-pop-a22af097-rescored-lenient_last_line`; the sweep under `ruleforge/sweep/` |
| Revised rulesets | committed under `data/rules/modelspec/ruleforge/` |
| Sweep | the ten-seed launch `2026-07-08/23-45-41`, without its Hydra configs, so `select` names jobs, not seeds: jobs 4 and 8 ran judge seeds 5 and 9 ([the rulesets' README](../data/rules/modelspec/ruleforge/README.md)) |

## Training

Each base model against its GRPO checkpoint, Gemma-4-31B-it's checkpoint-600 and Qwen3-8B's checkpoint-300, on the
same 500 scenarios, five draws each, paired by scenario, under the training prompt `forced-grpo`
([training.md](training.md) merges and serves them).

```bash
uv run ruleforge reproduce training render --out DIR
uv run ruleforge data fetch training
uv run ruleforge reproduce training rebuild --out DIR
```

| Golden | Holds |
|---|---|
| `results/training/gemma.json`, `results/training/qwen.json` | each arm's inconsistency rate |
| `results/training/judgment_changes.json` | judgment changes and label counts |
| `results/training/frozen_anchors.json` | each trained judge's whole-ruleset judgments against its own rule judgments and against the base's (the **frozen anchors**), and the expected label changes (`changes`) |

`judgment_changes.json`: how often a label differs between the base and the trained judge at one draw each, for the
individual-rule, aggregated rule and whole-ruleset judgments, beside two draws of one model (`draw_to_draw`); and the
label counts.

| Model | Base root | Trained root |
|---|---|---|
| Gemma-4-31B-it | `fig5-gemma-2026-10-07-pop-a22af097` | `fig5-gemma-ckpt600-2026-10-09-pop-a22af097` |
| Qwen3-8B | `fig5-grpo-v2-fp32-2026-09-26-checkpoint-200-pop-a22af097` | `fig5-qwen-mix-2026-10-11-checkpoint-300-pop-a22af097` |

The Hub also holds `training/inputs/` (rows, replay rows, anchor conditions, the 300-scenario development set) and
`training/selection/` (merge manifests, parity reports).

## Hierarchy

Per judge, `naive-rules-above` (every rule judged alone) beside `naive-hierarchy` (the Model Spec's levels of authority
explained, each rule judged with the rules that outrank it), five draws.

```bash
uv run ruleforge reproduce hierarchy render --out DIR
uv run ruleforge data fetch hierarchy prompt-ablation-v2-2026-10-glimmer-pop-a22af097 \
    prompt-ablation-v2-2026-10-haiku-sum final-2026-09-luna-pop-a22af097-rescored-lenient_last_line   # skip the roots you have
uv run ruleforge reproduce hierarchy rebuild --out DIR
```

| | |
|---|---|
| Goldens | `results/hierarchy/hierarchy.json`: each judge's two rates and their difference. `results/hierarchy/ladder.json`: each arm's rate by hierarchy level, which the golden defines |
| Pair map | `configs/paper/hierarchy_pairs.yaml` names each judge's two arms; copy it with your roots for a rerun |
| Levels | the `rule_type` of `data/rules/modelspec/openai_rules_all_full.jsonl` |

| Judge | Flat arm | Hierarchy arm |
|---|---|---|
| Olmo-3-32B-Think, Gemma-4-31B-it, gpt-oss-120b | `hierarchy-aware-2026-10-pop-a22af097` | the same root |
| Muse-Glimmer-30B | `prompt-ablation-v2-2026-10-glimmer-pop-a22af097` (its naive arm) | `hierarchy-aware-2026-10-glimmer` |
| Claude Haiku 5.5 | `prompt-ablation-v2-2026-10-haiku-sum` (its naive arm) | `hierarchy-aware-2026-10-haiku-sum` |
| GPT-5.6 Luna | `final-2026-09-luna-pop-a22af097-rescored-lenient_last_line` (its naive arm) | `hierarchy-aware-2026-10-luna-sum` |
