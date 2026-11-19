# Training a judge with GRPO

How to train a judge from the paper's published inputs, then merge and evaluate a checkpoint. Training uses GRPO to
fit a LoRA **adapter**, a small set of extra weights on the **base model**, saved as a **checkpoint** every 100 steps.

Training needs `uv sync --group vllm --group grpo`, Linux and NVIDIA GPUs. Building inputs for a new training pool
also needs protected populations this repository does not hold ([rerun.md](rerun.md#training)).

- [The stages](#the-stages)
- [Train from the published rows](#train-from-the-published-rows)
- [Merge and evaluate a checkpoint](#merge-and-evaluate-a-checkpoint)
- [What training optimizes](#what-training-optimizes)
- [Settings](#settings)
- [Modules](#modules)

## The stages

| Stage | What it makes | Command, or the published copy that skips it |
|---|---|---|
| 1. Anchors | the base model's single-rule judgments and whole-ruleset draws over the training pool, which the reward targets | `ruleforge jobs launch campaign --spec grpo_anchors`, then `ruleforge grpo anchors build`; published: `training/inputs/<model>/anchor_condition.tar.zst` |
| 2. Rows | the GRPO rows (prompt plus reward targets) and the **replay rows** (the base's own single-rule answers) | `ruleforge grpo rows`; published: `grpo_data/<build>/` |
| 3. Train | LoRA checkpoints every 100 steps, up to 600 (500 for Qwen3-8B) | `ruleforge grpo train`; published: the adapters (the model repository) |
| 4. Merge | the adapter merged into its base in fp32 | `ruleforge grpo merge`; published: merge manifests only (`training/selection/`), so merge the adapter yourself |
| 5. Report | each checkpoint's development inconsistency and gate statistics against the base, and the reported checkpoint with the lowest development inconsistency | [training_report.md](training_report.md) |
| 6. Evaluate | base and checkpoint judged on the same 500 scenarios, paired by scenario | `ruleforge evaluate` below, or `ruleforge jobs launch campaign --spec training_arms` |

## Train from the published rows

Gemma-4-31B-it's run: four GPUs, about 25 hours of steps. `jobs launch` submits to SLURM: set
`RULEFORGE_SLURM_PARTITION` or add `--profile NAME` ([rerun.md](rerun.md#before-a-rerun)).

```bash
uv run ruleforge data fetch training
B=$RULEFORGE_ARTIFACT_ROOT/grpo_data/gemma-mix2-2026-10-03-train
uv run python -c "import sys, zstandard; zstandard.ZstdDecompressor().copy_stream(open(sys.argv[1], 'rb'), \
    open(sys.argv[2], 'xb'))" $B/replay_rows.jsonl.zst $B/replay_rows.jsonl      # 13.3 GB for Gemma, 8.3 GB for Qwen
uv run ruleforge jobs launch train --spec grpo_train_gemma_run --run-name my-gemma-run \
    --var rows=grpo_data/gemma-mix2-2026-10-03-train \
    --var condition=sampling/gemma-screen-2026-10-01-confirm/gemma_pool_c.confirm.block-000/gemma-4-31B-it/grpo-v2-anchors/confirm/condition.json \
    --resume latest --preflight-only
```

Drop `--preflight-only` to submit. `--resume latest` continues a preempted run from its newest checkpoint. The
resumed run is not bit-identical to an uninterrupted one: the KL anchor's replay rows are reshuffled from the resumed
step (no sampler state is saved), so they come in another order and some may repeat.

Checkpoints go to `$RULEFORGE_ARTIFACT_ROOT/training/<run name>/`. A `STOP.json` placed there ends the run after the
next checkpoint.

| Before any model loads, the trainer checks | |
|---|---|
| Rows | against their manifest, and their disjointness marker against every protected file the rows build was held to (for the paper's rows, the files `experiments/training/pins.py` pins and the development set) |
| Base model | its directory, tokenizer and weights against the ones the rows were built on (`$RULEFORGE_MODEL_ROOT/gemma-4-31B-it`, `$RULEFORGE_MODEL_ROOT/Qwen3-8B`); the weights' digest, which the published rows predate, also binds a resume |
| Anchor condition | by content, at the `sampling/` path the fetch placed it |

Without the launch layer, run the trainer directly. `--config` reads a relative path from its own directory, so the
ruleset is given as an absolute path:

```bash
uv run ruleforge grpo train --config configs/training/grpo/gemma4_31b.yaml --run-name R -- \
    data.train=<rows> data.replay=<replay rows> data.anchor_condition=<condition.json> reward.truncated=zero \
    data.rules=$PWD/data/rules/modelspec/openai_rules_all.jsonl
```

## Merge and evaluate a checkpoint

A Qwen3-8B example. Merge on CPU, then copy the fp32 judge config, set its `model_path` to the merged directory, and
judge with it:

```bash
uv run ruleforge grpo merge --base $RULEFORGE_MODEL_ROOT/Qwen3-8B \
    --adapter $RULEFORGE_ARTIFACT_ROOT/training/<run>/checkpoint-300 --out <merged dir> --threads 8 \
    --expect-rank 32 --expect-alpha 64 --expect-base-model Qwen3-8B
cp configs/model/qwen3-8b-fp32.yaml my-trained.yaml   # then set model_path: "<merged dir>"
uv run ruleforge evaluate --rules data/rules/modelspec/openai_rules_all.jsonl \
    --scenarios data/populations/frozen_500_r3.jsonl --judge my-trained.yaml \
    --prompt forced-grpo --out $RULEFORGE_ARTIFACT_ROOT/sampling/my-eval
```

A fetched adapter (`ruleforge data fetch --model`) names its base by the Hub id: pass the id's last segment
(`Qwen3-8B`, `gemma-4-31b-it`) as `--expect-base-model`, with the base in a directory of that name. Gemma-4-31B-it's
merge also passes `--expect-scope model.language_model.`, as [its command in rerun.md](rerun.md#training) does.

| Merge | CPU threads | Memory | Time |
|---|---|---|---|
| Qwen3-8B | 8 | about 80 GB | under an hour |
| Gemma-4-31B-it | 16 | about 240 GB | |

The merged directory holds the fp32 weights, the base's tokenizer and generation config, and `merge_manifest.json`
with the weights' digest. The paper served the base and the checkpoint alike on the fp32 engine and judged each on the
same 500 scenarios with five draws under `forced-grpo`; [reproduce.md](reproduce.md#training) lists its goldens.

[`ruleforge grpo report`](training_report.md) compares the checkpoints and selects the reported one with the lowest
development inconsistency; the same page shows how to check that a served adapter computes what it trained.

## What training optimizes

The reward favours a whole-ruleset judgment that agrees with the base model's single-rule judgments; the KL anchor
replays the base's own single-rule answers, so the trained judge's single-rule judgments stay where the reward assumes
they are. `grpo/rewards.py` holds the reward, `grpo/kl_anchor.py` the anchor, and `configs/training/grpo/base.yaml`
their settings (`reward`, `anchor`).

## Settings

`configs/training/grpo/base.yaml` holds the shared settings; `gemma4_31b.yaml`, `qwen3_8b.yaml` and `qwen3_8b_mix.yaml`
add the model and its layout.

| Setting | Value |
|---|---|
| Lengths | prompts capped at 5,000 tokens; completions at 6,144; `vllm_max_model_length` = cap + 5,120 |
| Schedule | 600 steps for Gemma-4-31B-it, 500 for Qwen3-8B (`qwen3_8b_mix.yaml`); a checkpoint every 100, all kept |
| Memory | gradient checkpointing; Gemma under FSDP2 over 4 ranks with vLLM at tensor-parallel 4 |
| Logging | Weights & Biases, one run per training run (`WANDB_MODE=offline`, then `wandb sync`) |
| Weight-sync probe | after every adapter sync into the rollout engine, a fixed set of engine weights is compared with the trained weights; a mismatch stops the run |

## Modules

| Module | Does |
|---|---|
| `grpo/trainer.py` | the anchored GRPO trainer: config checks, the reward set, the KL anchor, the sync probe, resume |
| `grpo/rewards.py`, `grpo/kl_anchor.py` | the three reward functions; the KL anchor loss and replay stream |
| `grpo/anchor_set.py` | anchor sets from the single-rule judgments |
| `grpo/exclusion.py` | the prompt-disjointness check and its marker |
| `grpo/grpo_data.py`, `grpo/merging.py` | the rows build and its manifest; the fp32 merge and its manifest |
| `grpo/lora_targets.py`, `grpo/sync_probe.py`, `grpo/fsdp2_state.py` | adapter targets; the weight-sync probe; sharded checkpoint state |
| `grpo/report.py`, `grpo/report_tables.py`, `grpo/lora_parity.py` | the checkpoint report and its tables; the parity gates |
| `experiments/training/` | the paper's builds: the comparison goldens, the judgment changes and the frozen-anchor rates |
