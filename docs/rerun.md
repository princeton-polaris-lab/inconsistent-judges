# Rerunning the paper's experiments

How to judge the scenarios again and train again. A rerun writes campaign roots of its own and is a new sample, so
compare it with the paper within tolerances. To recompute the goldens from the published roots instead, see
[reproduce.md](reproduce.md).

- [Before a rerun](#before-a-rerun)
- Experiments: [Ruleset size](#ruleset-size), [Prompt ablation](#prompt-ablation), [RuleForge](#ruleforge),
  [Training](#training), [Hierarchy](#hierarchy)
- [Build over your own roots](#build-over-your-own-roots), [Checking a rerun](#checking-a-rerun)

## Before a rerun

```bash
export RULEFORGE_ARTIFACT_ROOT=/path/to/artifacts
mkdir -p "$RULEFORGE_ARTIFACT_ROOT/populations"
cp data/populations/frozen_500_r3.jsonl* "$RULEFORGE_ARTIFACT_ROOT/populations/"
export RULEFORGE_MODEL_ROOT=/path/to/weights RULEFORGE_BATCH_ROOT=$RULEFORGE_ARTIFACT_ROOT/batch
```

Campaigns read the population copied above. Launches go to SLURM: set `RULEFORGE_SLURM_PARTITION`, or add
`--profile NAME`.

| | |
|---|---|
| Environment | `uv sync --group vllm` for local judges and RuleForge; `uv sync --group vllm --group grpo` for training |
| `# paid` | the line spends money, and carries the confirmation it needs, `-- +confirm_spend=true` |
| `--preflight-only`, `--dry-run` | the line checks and submits nothing; drop the flag to submit |
| Launch records | `$RULEFORGE_ARTIFACT_ROOT/.ruleforge-launch/<launch id>/` |
| Following a launch | `ruleforge jobs status\|wait\|pull\|stop <launch id>` |
| Shards | a campaign cut into several jobs (`--shards N`); `jobs wait <launch id> --pull` merges them back |
| QOS | a launch's `--qos`, else its spec's, else its profile's |
| Attempt cap | the paper's specs set 20, as their campaigns ran; Claude Haiku 5.5's ruleset-size root was raised to 30, an amendment its records hold |

The published ruleset-size and first prompt-ablation campaigns of Gemma-4-31B-it, gpt-oss-120b and Olmo-3-32B-Think
ran on vLLM 0.19, Muse-Glimmer-30B's on 0.28, the version this lock has.
`ruleforge analyze` refuses a contrast across the two unless given `--allow-stack-mismatch`, which marks the result.

## Ruleset size

| | |
|---|---|
| Runs | each judge's **bank** (every rule judged alone), then its sampled subsets |
| Cost | Olmo-3-32B-Think about 400 GPU-hours, Muse-Glimmer-30B about 60; GPT-5.6 Luna about $362 at the Batch rates of its run |
| Then | `ruleforge bank analyze <root>` on each root you ran (it refuses a fetched one), then [build over your roots](#build-over-your-own-roots) |

The paper's size 57 judges the full ruleset in an order of each subset seed's own, so its point averages over ten
orders; the bank roots' own size-57 calls use the file's order for every seed. Below 57 the subsets are the same files
either way.
To rerun that design in one root, generate its subsets and add `$SHUFFLED` after `--` on each launch line below:

```bash
uv run python -m ruleforge.experiments.ruleset_size.generate_subsets --manifest-dir data/rules/subset_manifest_shuffled_full
SHUFFLED="ruleset_size.sampling_strategy=with_replacement_shuffled_full \
  ruleset_size.subset_manifest_dir=data/rules/subset_manifest_shuffled_full \
  ruleset_size.approved_subset_manifest_sha256=6d6c5375ca76194ada6973c82656712a3237f1010776629f9b1723941e768461"
```

Each launch names its campaign id (`--var campaign_id=`), which seeds every draw: under a published run's id (the table
below) a rerun draws at that run's seeds. The open judges run as SLURM jobs, one per half of the ten subset seeds:

```bash
uv run ruleforge reproduce ruleset-size subsets
uv run ruleforge bank run --config configs/sampling_ruleset_size.yaml --campaign-id repro-ruleset-size --preflight-only   # checks; loads nothing
uv run ruleforge jobs launch job --spec ruleset_size_open_models --var campaign_id=repro-ruleset-size --var seeds=[0,2,4,6,8]
uv run ruleforge jobs launch job --spec ruleset_size_open_models --var campaign_id=repro-ruleset-size --var seeds=[1,3,5,7,9]
uv run ruleforge jobs launch job --spec ruleset_size_open_models --var campaign_id=repro-ruleset-size-glimmer \
    --var models=muse-glimmer-30b --var seeds=[0,2,4,6,8] -- sampling.retry_scope=component
uv run ruleforge jobs launch job --spec ruleset_size_open_models --var campaign_id=repro-ruleset-size-glimmer \
    --var models=muse-glimmer-30b --var seeds=[1,3,5,7,9] -- sampling.retry_scope=component
```

A paid judge runs on this machine: its bank first, then one process per subset seed, side by side, each waiting on its
batches. These lines are GPT-5.6 Luna's; for Claude Haiku 5.5, add `--var model=claude-haiku-5-5-batch`.

```bash
nohup setsid uv run ruleforge jobs launch job --spec ruleset_size_batch_judge --var campaign_id=repro-ruleset-size-luna \
    --var shares=[] --var run_bank=true -- +confirm_spend=true > bank.log 2>&1 &   # paid
for N in 0 1 2 3 4 5 6 7 8 9; do
  nohup setsid uv run ruleforge jobs launch job --spec ruleset_size_batch_judge \
      --var campaign_id=repro-ruleset-size-luna --var shares=[$N] -- +confirm_spend=true > seed-$N.log 2>&1 &   # paid
done
```

**How the paper's size 57 ran.** Each judge's size 57 has a root of its own, without a bank, judged after the rest of
its curve; its analysis takes the bank and sizes 2 to 56 from the judge's bank root (`--bank-root`), pairing each
draw with the bank's draw of the same index. The two campaigns must agree on every sampling setting in force, and the
retry cap is the one setting a recorded amendment may have raised. The analysis also refuses a bank and a size judged
on two inference stacks, so a local judge's size 57 runs on its bank's stack.

| Size-57 root, and its campaign id | Bank and sizes 2 to 56 from | Their campaign id | Ran on |
|---|---|---|---|
| `k57-shuffled-2026-11` (Gemma-4-31B-it, gpt-oss-120b, Olmo-3-32B-Think) | `fig2-2026-09-pop-a22af097-rescored-lenient_last_line` | `fig2-2026-09` | vllm 0.19.0, torch 2.10.0, transformers 5.5.0, tokenizers 0.22.2, triton 3.6.0 |
| `k57-shuffled-2026-11-glimmer` | `fig2-2026-10-glimmer-pop-a22af097` | `fig2-2026-10-glimmer` | this lock's stack (vllm 0.28.0) |
| `k57-shuffled-2026-11-luna` | `fig2-2026-09-luna-pop-a22af097-rescored-lenient_last_line` | `fig2-2026-09-luna` | the provider |
| `k57-shuffled-2026-11-haiku` | `fig2-2026-10-haiku-sum-rescored-lenient_last_line` | `fig2-2026-10-haiku-sum` | the provider, at the cap of 30 its bank's root was raised to |

`bank analyze` joins a size-57 root to a bank root you ran: a fetched bank root holds only what the rebuild reads,
and is refused. To judge size 57 again, rerun the judge's bank and sizes 2 to 56 first. A size-57 root is named by its
campaign id: under that id, give the new root a place of its own (`-- campaign_root=<root>`). Simpler, and on one
stack throughout: rerun the whole curve in one root with `$SHUFFLED`, as above.

The three September judges need an environment with their bank's stack, apart from this checkout's `.venv`. Launch
them with its Python, not `uv run`, which would select this lock: a launch's jobs run the Python that launched them.

```bash
V19=/path/to/ruleforge-vllm-0.19
uv venv $V19 --python 3.12
echo transformers==5.5.0 > $V19/overrides.txt   # vLLM 0.19 asks for transformers<5; the banks ran on 5.5.0
uv pip install --no-config --python $V19 --overrides $V19/overrides.txt \
    vllm==0.19.0 torch==2.10.0 transformers==5.5.0 tokenizers==0.22.2 triton==3.6.0
uv pip install --no-config --python $V19 -e ".[slurm]"
$V19/bin/python -c "from importlib.metadata import version; print(*map(version, ['vllm', 'torch', 'transformers', 'tokenizers', 'triton']))"
# 0.19.0 2.10.0 5.5.0 0.22.2 3.6.0
```

Muse-Glimmer-30B and the paid judges run from this checkout's environment:

```bash
K57="$SHUFFLED ruleset_size.subset_sizes=[57]"   # the paid judges' spec already sets ruleset_size.run_bank=false
for HALF in [0,2,4,6,8] [1,3,5,7,9]; do
  $V19/bin/python -m ruleforge jobs launch job --spec ruleset_size_open_models --var campaign_id=repro-k57-shuffled \
      --var seeds=$HALF -- $K57 ruleset_size.run_bank=false
  uv run ruleforge jobs launch job --spec ruleset_size_open_models --var campaign_id=repro-k57-shuffled-glimmer \
      --var models=muse-glimmer-30b --var seeds=$HALF -- sampling.retry_scope=component \
      execution.pack_draws=true execution.pack_max_prompts=8192 execution.pack_max_draws=5 $K57 ruleset_size.run_bank=false
done
for N in 0 1 2 3 4 5 6 7 8 9; do
  nohup setsid uv run ruleforge jobs launch job --spec ruleset_size_batch_judge \
      --var campaign_id=repro-k57-shuffled-luna --var shares=[$N] -- $K57 +confirm_spend=true > luna-$N.log 2>&1 &   # paid
  nohup setsid uv run ruleforge jobs launch job --spec ruleset_size_batch_judge \
      --var campaign_id=repro-k57-shuffled-haiku --var model=claude-haiku-5-5-batch --var shares=[$N] -- \
      sampling.retry_scope=component sampling.max_semantic_attempts=30 $K57 +confirm_spend=true > haiku-$N.log 2>&1 &   # paid
done
S=$RULEFORGE_ARTIFACT_ROOT/sampling   # each --bank-root: the bank root you ran for that judge
uv run ruleforge bank analyze $S/repro-k57-shuffled --bank-root $S/<your bank root>
uv run ruleforge bank analyze $S/repro-k57-shuffled-glimmer --bank-root $S/<your Muse-Glimmer-30B bank root>
uv run ruleforge bank analyze $S/repro-k57-shuffled-luna --bank-root $S/<your GPT-5.6 Luna bank root>
uv run ruleforge bank analyze $S/repro-k57-shuffled-haiku --bank-root $S/<your Claude Haiku 5.5 bank root>
```

## Prompt ablation

| | |
|---|---|
| Runs | every **arm** (one judge under one prompt): the naive arms, the other arms, then the paid judges |
| Cost | Muse-Glimmer-30B about 23 H200-hours for three arms; GPT-5.6 Luna about $93 for three arms |
| Then | `ruleforge analyze` on every root; copy the arm map `configs/paper/prompt_ablation_arms.yaml` with your roots, and [build over them](#build-over-your-own-roots) |

```bash
uv run ruleforge jobs launch campaign --spec prompt_ablation_naive --var campaign_id=repro-final --preflight-only
uv run ruleforge jobs launch campaign --spec prompt_ablation_naive --var campaign_id=repro-final
uv run ruleforge jobs launch campaign --spec prompt_ablation --var campaign_id=repro-revised
uv run ruleforge jobs launch campaign --spec prompt_ablation --var campaign_id=repro-revised-open \
    --var models=olmo3-32b-tp-2,gpt-oss-tp-2 --var styles=encouraged,forced
uv run ruleforge jobs launch campaign --spec prompt_ablation --var campaign_id=repro-revised-glimmer \
    --var models=muse-glimmer-30b --var styles=naive-rules-above,encouraged,forced --var retry_scope=component
POP="--var population=populations/frozen_500_r3.jsonl --var population_sha256=c16f66f4ad3fc93fad46bf31efaa8325f30fb750104d1e13a9326b7cd46b10d4"
uv run ruleforge jobs launch campaign --spec batch_judge --var campaign_id=repro-revised-haiku \
    --var model=claude-haiku-5-5-batch $POP --var retry_scope=component \
    --var styles=naive-rules-above,encouraged,forced -- +confirm_spend=true   # paid
uv run ruleforge jobs launch campaign --spec batch_judge --var campaign_id=repro-revised-luna \
    --var model=gpt-5.6-luna-batch $POP -- +confirm_spend=true    # paid
uv run ruleforge jobs wait <launch id> --pull           # each prompt_ablation launch's
uv run ruleforge analyze <root>                         # every root
```

## RuleForge

| | |
|---|---|
| Runs | the ten-seed sweep, then the campaign over the initial ruleset and the two picked revisions |
| Cost | ten seeds × two GPUs × up to 48 hours; the campaign about 60 GPU-hours |
| Then | `ruleforge analyze` on the campaign's root, then [build](#build-over-your-own-roots) with one `--ruleset TAG=LABEL` per ruleset |

To check the setup first, the `ruleforge_smoke` spec runs one seed and one episode, in about 1-2 hours. The sweep:

```bash
export WANDB_MODE=offline
uv run ruleforge jobs launch job --spec ruleforge_modelspec --var run_name=repro_10seed
uv run ruleforge jobs wait <launch id>
uv run ruleforge optimize select --sweep-dir experiments/results/sweep/modelspec/openai/<launch id>
```

Before the campaign, register the two picked revisions in `configs/sampling_ruleforge.yaml`; the campaign refuses a
drifted file, and `ruleforge analyze` refuses a family short of a declared ruleset.

| Where | What |
|---|---|
| `ruleforge.rulesets.<tag>` | each revision's `path` and `sha256` |
| `condition_extra.ruleset_family` | the three tags joined by `+` |
| `--var rulesets=` on the campaign lines | the three tags joined by `,` |

```bash
uv run ruleforge jobs launch campaign --spec ruleforge_rulesets --var campaign_id=repro-ruleforge \
    --var rulesets=initial,<winner>,<runner-up> --preflight-only
uv run ruleforge jobs launch campaign --spec ruleforge_rulesets --var campaign_id=repro-ruleforge \
    --var rulesets=initial,<winner>,<runner-up>
uv run ruleforge jobs wait <launch id>
uv run ruleforge analyze $RULEFORGE_ARTIFACT_ROOT/sampling/repro-ruleforge
```

This code differs from the published sweep's in the judge prompt, what the rule amender is shown, how scenarios and the
panel are seeded, how patches match and how failed calls are retried (the published panel made one attempt per
judgment), so a rerun is a new optimization, not a replay. `--var prompt=naive-rules-above` on any of the three specs
runs the published prompt. Two more differences: the published sweep scored the intent rubric at temperature 1.0, this
code at 0.0 with seed 16, and it ran on vLLM 0.19, this lock on 0.28. A rerun with this code picked two rulesets that
lowered the initial ruleset's inconsistency rate by 0.031 and 0.039, against the published 0.138 and 0.175.

## Training

| | |
|---|---|
| Runs | the stages of [training.md](training.md#the-stages): anchors, references, rows, training, merge, report, evaluation |
| Cost | Gemma: four GPUs, about 25 hours of steps; plus the evaluation arms |
| Then | analyse both arms' roots and [build the comparison over them](#build-over-your-own-roots) |

To train on the published rows instead of building your own, follow
[training.md](training.md#train-from-the-published-rows). Building new rows also needs **protected populations**,
scenario sets the rows build must keep out of training (`ruleforge.experiments.training.pins.PINNED_PROTECTED`), which
neither this repository nor the Hub holds; the 300-scenario development set is on the Hub (`ruleforge data fetch
training` places it at `populations/grpo_v2_dev.jsonl`). `<sha>` below is a clean checkout's commit; wait for each
launch before the step that reads its outputs.

**Gemma 4 31B-it's anchors.** Partition the training pool:

```bash
P=gemma-anchors-<date>; A=$RULEFORGE_ARTIFACT_ROOT; C=gemma-4-31B-it/grpo-v2-anchors
uv run ruleforge scenarios partition --source $A/populations/<pool>.jsonl --expect-source-sha256 <its sha256> \
    --block-size 250 --out-dir $A/populations/<pool>.blocks-250
uv run ruleforge scenarios partition --source $A/populations/<pool>.jsonl --expect-source-sha256 <its sha256> \
    --block-size 1000 --out-dir $A/populations/<pool>.blocks-1000
```

Collect the base's single-rule judgments (`bank`) and whole-ruleset draws (`full`), and wait for both launches:

```bash
uv run ruleforge jobs launch campaign --spec gemma_anchors --var stage=bank --var campaign_prefix=$P \
    --blocks populations/<pool>.blocks-250 --preflight-only
uv run ruleforge jobs launch campaign --spec gemma_anchors --var stage=bank --var campaign_prefix=$P \
    --blocks populations/<pool>.blocks-250 --expect-commit <sha>
uv run ruleforge jobs launch campaign --spec gemma_anchors --var stage=full --var campaign_prefix=$P \
    --var timeout_min=180 --var request_block=250 --blocks populations/<pool>.blocks-1000 --preflight-only
uv run ruleforge jobs launch campaign --spec gemma_anchors --var stage=full --var campaign_prefix=$P \
    --var timeout_min=180 --var request_block=250 --blocks populations/<pool>.blocks-1000 --expect-commit <sha>
uv run ruleforge jobs wait <launch id>   # the bank's, then the full set's
```

Route the rows with `q_yes < 1` to a top-up, run it, and wait for it:

```bash
uv run ruleforge grpo anchors route --bank $A/sampling/$P-bank/<pool>.block-* \
    --blocks $A/populations/<pool>.blocks-250 --population $A/populations/<pool>.jsonl --side train \
    --reference $A/sampling/$P-bank/<pool>.block-000/$C/bank/condition.json \
    --rules data/rules/modelspec/openai_rules_all.jsonl --out $A/populations/$P-top_up.jsonl
uv run ruleforge scenarios partition --source $A/populations/$P-top_up.jsonl --expect-source-sha256 <its sha256> \
    --block-size 125 --out-dir $A/populations/$P-top_up.blocks-125
uv run ruleforge jobs launch campaign --spec gemma_anchors --var stage=top_up --var campaign_prefix=$P \
    --blocks populations/$P-top_up.blocks-125 --expect-commit <sha>
uv run ruleforge jobs wait <launch id>
```

Draw the development references, and wait for them. Their `references_bank` condition is the anchor build's
`--reference`:

```bash
uv run ruleforge jobs launch campaign --spec gemma_references --var stage=references_bank --var campaign_prefix=<prefix> \
    --var population=<dev-population.jsonl> --var population_sha256=<sha256> --preflight-only
uv run ruleforge jobs launch campaign --spec gemma_references --var stage=references_bank --var campaign_prefix=<prefix> \
    --var population=<dev-population.jsonl> --var population_sha256=<sha256>
uv run ruleforge jobs wait <launch id>
```

Build the anchor set the rows read, once the bank, the full set, the top-up and the references have finished:

```bash
uv run ruleforge grpo anchors build --side train --rules data/rules/modelspec/openai_rules_all.jsonl \
    --bank $A/sampling/$P-bank/<pool>.block-* --top-up $A/sampling/$P-top_up/$P-top_up.block-* \
    --full-set $A/sampling/$P-full/<pool>.block-* --population $A/populations/<pool>.jsonl \
    --blocks $A/populations/<pool>.blocks-250 --out $A/anchors/<anchor-set> \
    --reference $A/sampling/<prefix>-references_bank/gemma-4-31B-it/grpo-v2-dev/references_bank/base/condition.json
```

**Qwen3-8B's anchors.** The general spec with its values, over the same blocks. This is its mixed-judgment screen; the
other stages change `stage`, and its per-rule stages take `--var request_block=125`.

```bash
uv run ruleforge jobs launch campaign --spec grpo_anchors --var judge=qwen3-8b --var weights=Qwen3-8B \
    --var strict_think=true --var stage=mixscreen --var campaign_prefix=<prefix> --var request_block=250 \
    --var timeout_min=90 --blocks populations/<pool>.blocks-250 --preflight-only
uv run ruleforge jobs launch campaign --spec grpo_anchors --var judge=qwen3-8b --var weights=Qwen3-8B \
    --var strict_think=true --var stage=mixscreen --var campaign_prefix=<prefix> --var request_block=250 \
    --var timeout_min=90 --blocks populations/<pool>.blocks-250 --expect-commit <sha>
uv run ruleforge jobs wait <launch id>
```

**Build rows.** Gemma's rows from the anchor set; drop `--dry-run` to submit, then wait:

```bash
uv run ruleforge jobs launch job --spec grpo_rows_gemma --var anchors=anchors/<anchor-set> \
    --var scenarios=populations/<training-scenarios>.jsonl --var require_protected=populations/grpo_v2_dev.jsonl \
    --var dev_nonyes=populations/<dev-nonyes>.jsonl --var out=grpo_data/<build> --dry-run
uv run ruleforge jobs wait <launch id>
```

**Train.** On the rows just built, with the anchor condition they were built from: the first bank block's, which the
rows job's log names as `data.anchor_condition`. The trainer refuses any other.

```bash
uv run ruleforge jobs launch train --spec grpo_train_gemma_run --run-name repro-gemma --var rows=grpo_data/<build> \
    --var condition=sampling/$P-bank/<pool>.block-000/$C/bank/condition.json --resume latest --preflight-only
```

This page builds no Qwen3-8B rows. Its training line below trains on its published rows and anchor condition, fetched
and decompressed as [training.md](training.md#train-from-the-published-rows) shows. It trains 500 steps, as the
published run did: that run was configured for 600 and stopped at 500, and its learning rate is constant after the
warmup, so a 500-step run is the same run.

```bash
uv run ruleforge jobs launch train --spec grpo_train_run --var recipe=training/grpo/qwen3_8b_mix --run-name repro-qwen \
    --var rows=grpo_data/qwen-mix-2026-10-11 \
    --var condition=sampling/grpo-v2-anchors-2026-09-26-bank/grpo_v2_train.block-000/Qwen3-8B/grpo-v2-anchors/bank/condition.json \
    --resume latest --preflight-only
```

**Merge.** A checkpoint of the run into `training/repro-gemma/merged/checkpoint-<N>`; Gemma's merge takes 16 threads
and about 240 GB, where the spec's defaults (8, 80 GB) fit Qwen3-8B:

```bash
uv run ruleforge jobs launch job --spec merge_fp32 --var run=repro-gemma --var step=<N> --var base=gemma-4-31B-it \
    --var threads=16 --mem-gb 240 --dry-run -- --expect-scope model.language_model.
uv run ruleforge jobs wait <launch id>
```

**Report.** The development campaigns of [training_report.md](training_report.md#report-a-run), then:

```bash
uv run ruleforge grpo report saves --settings configs/training/report.yaml --base <base tables> --saves <save tables> \
    --rules data/rules/modelspec/openai_rules_all.jsonl --out <report dir>
```

**Evaluate.** The base and the merged checkpoint in one root, with the digest from the merge's `merge_manifest.json`;
Qwen3-8B's evaluation uses `training_arms`. Drop `--preflight-only` to submit, then wait and merge its shards:

```bash
uv run ruleforge jobs launch campaign --spec training_arm_gemma --var campaign_id=<campaign-id> \
    --var checkpoint=training/repro-gemma/merged/checkpoint-<N> --var checkpoint_sha256=<weights-sha256> --preflight-only
uv run ruleforge jobs wait <launch id> --pull
```

## Hierarchy

| | |
|---|---|
| Runs | each judge's hierarchy arm: the local judges, then the paid judges |
| Cost | about 1.3 times the flat arm's GPU-hours per hierarchy arm; the two API judges' hierarchy arms |
| Then | `ruleforge analyze` on every root; copy the pair map `configs/paper/hierarchy_pairs.yaml` with your roots, and [build over them](#build-over-your-own-roots) |

The flat arms of Muse-Glimmer-30B, Claude Haiku 5.5 and GPT-5.6 Luna are the prompt-ablation rerun's naive arms.

```bash
uv run ruleforge jobs launch campaign --spec hierarchy --var campaign_id=repro-hierarchy --var models=gpt-oss-tp-2 --shards 3
uv run ruleforge jobs launch campaign --spec hierarchy --var campaign_id=repro-hierarchy --var models=olmo3-32b-tp-2 --shards 7
uv run ruleforge jobs launch campaign --spec hierarchy --var campaign_id=repro-hierarchy --var models=gemma4-31b-tp-2 --shards 8
uv run ruleforge jobs launch campaign --spec hierarchy --var campaign_id=repro-hierarchy-glimmer \
    --var models=muse-glimmer-30b --var styles=naive-hierarchy --var retry_scope=component
POP="--var population=populations/frozen_500_r3.jsonl --var population_sha256=c16f66f4ad3fc93fad46bf31efaa8325f30fb750104d1e13a9326b7cd46b10d4"
uv run ruleforge jobs launch campaign --spec batch_judge --var campaign_id=repro-hierarchy-haiku \
    --var model=claude-haiku-5-5-batch $POP --var styles=naive-hierarchy --var retry_scope=component \
    -- +hierarchy=model_spec +confirm_spend=true   # paid
uv run ruleforge jobs launch campaign --spec batch_judge --var campaign_id=repro-hierarchy-luna \
    --var model=gpt-5.6-luna-batch $POP --var styles=naive-hierarchy \
    -- +hierarchy=model_spec +confirm_spend=true   # paid
uv run ruleforge jobs wait <launch id> --pull   # each hierarchy launch's
uv run ruleforge analyze <root>                 # every root
```

A hierarchy condition of your own:

```bash
uv run ruleforge evaluate --rules R --scenarios S --judge J --prompt naive-hierarchy \
    --rule-context superiors --rule-levels data/rules/modelspec/openai_rules_all_full.jsonl   # a paid judge adds --yes-spend
```

## Build over your own roots

`ruleforge reproduce <experiment> rebuild` reads the published roots. These lines read yours, and each writes a golden
under `DIR`.

```bash
uv run python -m ruleforge.experiments.ruleset_size.build --estimates <root>/aggregate/figure2_estimates.json \
    --golden DIR/results/ruleset_size/curves.json     # one --estimates per root
uv run python -m ruleforge.experiments.ruleset_size.build_repeated_draw_baseline \
    --estimates <root>/aggregate/figure2_estimates.json \
    --golden DIR/results/ruleset_size/repeated_draw_baseline.json      # the same --estimates; it reads their roots too
uv run python -m ruleforge.experiments.prompt_ablation.build --arm-map <your arm map> \
    --golden DIR/results/prompt_ablation/prompt_ablation.json
uv run python -m ruleforge.experiments.leniency.build --prompt-style forced --estimates <root>/aggregate/estimates.json \
    --golden DIR/results/leniency/leniency.json       # one --estimates per root
uv run python -m ruleforge.experiments.ruleforge.build --estimates <root>/aggregate/estimates.json \
    --ruleset TAG=LABEL --golden DIR/results/ruleforge/ruleforge.json
uv run python -m ruleforge.experiments.training.build --arm base artifact://sampling/<root> <condition> \
    --arm <policy> artifact://sampling/<root> <condition> --reference base \
    --rule-set data/rules/modelspec/openai_rules_all.jsonl --out-dir DIR/training
uv run python -m ruleforge.experiments.hierarchy.build --pairs <your pair map> \
    --golden DIR/results/hierarchy/hierarchy.json
uv run python -m ruleforge.experiments.hierarchy.build_ladder --pairs <your pair map> \
    --golden DIR/results/hierarchy/ladder.json
```

- A training build writes `training_family.json`: copy it to `DIR/results/training/gemma.json` or `qwen.json` to
  verify it. `<policy>` is the arm's policy in its condition id.
- The judgment-changes and frozen-anchors goldens read the paper's roots only, so their checks report `missing` here.
- To draw a rerun's figures, give a program `--from_golden <its golden>` and its figure flags (`--help` lists them;
  RuleForge's figures come from `python -m ruleforge.experiments.ruleforge.plot`).

## Checking a rerun

```bash
uv run ruleforge reproduce all verify --rerun <tree>                  # reads <tree>/results/...
uv run ruleforge reproduce all verify --rerun <tree> --json out.json  # writes every finding
```

A rerun on a new sample moves numbers by chance, so only a structural difference fails it: exit 1 for a missing or
extra golden, judge, condition, size or row, a changed label or design count, NaN or infinity, or a lost value; exit 2
when the check cannot run; else exit 0. Numbers that moved are reported per golden: how many intervals were compared,
how many estimates fell outside the committed interval against how many would by chance at the stored standard
deviations (about one in six), and the largest differences. `--exact` instead requires each golden's parsed values to equal the committed ones (`ALL 11 EQUAL`, else exit 1); a
golden that differs only in formatting is marked "bytes differ" and still passes.

| Finding | Meaning | Fails |
|---|---|---|
| `missing`, `extra` | a value, key or row only one side holds, or null, NaN or infinity where a number stood | yes |
| `design` | a label, an arm, a condition, a retry cap or a design count (sizes, scenarios, draws) differs: the rerun compared something else | yes |
| `containment` | an estimate outside the committed interval | no |
| `overlap` | the committed and rerun intervals do not overlap | no |
| `sign`, `flag` | an estimate or a significance flag on the other side | no |
| `drift` | values moved beyond the golden's tolerance | no |

Where and when a run happened is one notice and never fails: its roots and file paths, campaign ids, dates, digests
and fingerprints, code state and inference-stack versions. Every other value is compared, a retry cap, a cap
amendment's conditions and a level named root included. A stack description is compared with only its package
versions (vllm, torch, transformers, tokenizers, triton) masked, so its dtype, context limit, attention backend,
precision settings, engine-argument digest and provider are compared.

The committed ruleset-size goldens join two roots per judge, and each names its size-57 root's retry scope and cap
(`sizes_from`). A rerun of the whole design in one root has no `sizes_from`, so its check reports that as missing: a
structural difference, not a different result.

The statistics below count rare events, a few cells in 2,500, and some are committed with zero-width intervals, such as
0 [0, 0]. An estimate outside one of them in a fresh rerun is a cell or two moving.

| Golden | Statistic | Cases |
|---|---|---|
| `results/hierarchy/ladder.json` | over-flag | every value |
| `results/hierarchy/ladder.json` | `under_unsure` | some, mostly at the root level and gemma4-31b's; read the larger ones like any other interval |
| `results/ruleset_size/repeated_draw_baseline.json` | strict share | 21 (a judge at a size), each 0 [0, 0] |
| `results/ruleset_size/repeated_draw_baseline.json` | Muse-Glimmer-30B's separate-call repeat disagreement | sizes 53, 56 and 57 |
| `results/ruleset_size/repeated_draw_baseline.json` | Muse-Glimmer-30B's last_step contrast (K57 - K56) | of the strict share and of the separate-call repeat disagreement |

Size 57 judges the one full ruleset, in an order of each subset seed's own; smaller sizes are sampled and shuffled per
seed. So at 57 the seed resample carries the rule order and call-to-call variability, and a last_step contrast compares
ten sampled 56-rule subsets with ten orders of the full ruleset.
