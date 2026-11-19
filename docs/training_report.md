# Reporting a training run's checkpoints

How to compare a training run's checkpoints with its base model on development scenarios, held apart from the
evaluation, and select one to evaluate. The report computes the same numbers for every checkpoint and selects the
reported save with the lowest development inconsistency (`m1`).

Sections: [Report a run](#report-a-run), [What the report gives](#what-the-report-gives), [The published
checkpoints](#the-published-checkpoints), [Parity](#parity-does-a-served-adapter-compute-what-it-trained).

## Report a run

```bash
S=configs/training/report.yaml; R=data/rules/modelspec/openai_rules_all.jsonl; D=$RULEFORGE_ARTIFACT_ROOT/my-report
uv run ruleforge grpo report populations --settings $S --dev <dev.jsonl> --dev-nonyes <nonyes.jsonl> --out-dir $D/base
```

Each `--dev` row needs a `source` field: stage 1 takes its 150 scenarios by source. `populations` writes each
campaign's scenarios to `$D/base/`, with their digests in `populations.json`. A dev set whose sources sit near a
quota's edge is refused, since cutting the table would change stage 1's scenarios: draw it again at random.

Then run one development campaign per row below, merging each checkpoint first (`ruleforge grpo merge`,
[training.md](training.md#merge-and-evaluate-a-checkpoint)):

| Campaign | `grpo_dev.stage` | `training.policy` | `scenario_set_path` | Draws |
|---|---|---|---|---|
| the base's single-rule bank | `references_bank` | `base` | `$D/base/references.jsonl` | 4 a rule |
| the base's whole-ruleset draws | `references_full` | `base` | `$D/base/references.jsonl` | 8 |
| each checkpoint's stage 1 | `stage1` | `checkpoint-N` | `$D/base/stage1.jsonl` | 2 |
| each checkpoint's probe | `probe` | `checkpoint-N` | `$D/base/probe.jsonl` | 1 a rule |

```bash
uv run python -m ruleforge.launch.programs.run_campaign --config-name sampling_training_dev \
    training_serving=merged_fp32 grpo_dev.campaign_prefix=my-dev grpo_dev.stage=stage1 \
    training.policy=checkpoint-600 "+training.policies.checkpoint-600={model_path: <merged dir>, weights_sha256: '<digest>'}" \
    scenario_set_path=$D/base/stage1.jsonl approved_scenario_sha256=<its digest>
```

A base campaign drops the `+training.policies...` override. Each writes to `sampling/my-dev-<stage>/` under the
artifact root.

- `training_serving=merged_fp32` serves Qwen3-8B, `gemma_merged_fp32` Gemma. Use the base the checkpoint was trained
  on: nothing refuses another, which would serve the weights under that model's name, reasoning parser and token cap.
- A campaign exits 1 when a stage 1 or base draw runs out of attempts on a scenario. Its outputs are still readable,
  and `read` names those draws in `exhausted_draws`.

Then read each campaign into a table, and report:

```bash
uv run ruleforge grpo report read --settings $S --scenarios $D/base/scenarios.jsonl --rules $R \
    --root $RULEFORGE_ARTIFACT_ROOT/sampling/my-dev-stage1 --policy checkpoint-600 --kind stage1 \
    --tokenizer $RULEFORGE_MODEL_ROOT/Qwen3-8B --out $D/saves/checkpoint-600/stage1.jsonl
uv run ruleforge grpo report saves --settings $S --base $D/base --saves $D/saves --rules $R --out $D/report
```

| `read` | `--kind` | `--policy` | `--tokenizer` | `--out` |
|---|---|---|---|---|
| the base's bank | `bank` | `base` | no | `$D/base/bank.jsonl` |
| the base's whole-ruleset draws | `full` | `base` | needed | `$D/base/full.jsonl` |
| a checkpoint's stage 1 | `stage1` | `checkpoint-N` | needed | `$D/saves/checkpoint-N/stage1.jsonl` |
| a checkpoint's probe | `probe` | `checkpoint-N` | no | `$D/saves/checkpoint-N/probe.jsonl` |

`saves` writes `tolerances.json`, `report.json` and `report.md`; `*` marks each value outside its tolerance.
`report.json`'s `selected` and the table's second line name the reported save with the lowest `m1` (on a tie, the
earlier save); the table gives each save's gates outside tolerance beside it. Each
checkpoint needs both its tables, on the base's scenarios. `saves` cannot tell whether a checkpoint comes from the
base's model, so keep one run's tables under one `$D`.

## What the report gives

| Term | Meaning |
|---|---|
| `q̂[judgment]` | the probability that the judgment agrees with the max over 57 independent single-rule draws of the **base** model: `q_yes = 1 - Π(1 - p_r(yes))`, `q_no = Π p_r(no)`, `q_unsure = 1 - q_yes - q_no` |
| `p_r` | the share of the base's K draws on rule r that gave each label (K = 4 here, the `references_bank` campaign's draws) |

| Number | What it is |
|---|---|
| `m1` | development inconsistency: one minus the mean, over scenarios and accepted draws, of `q̂[judgment]` from the base's single-rule bank; with a 95% percentile bootstrap interval over scenarios |
| Change from the base | the checkpoint's `m1` minus the base's over the scenarios both answered, with a paired bootstrap interval |
| Gate statistic | the checkpoint's value minus the base's on its reference draws, except G7 |
| Tolerance | how far a gate statistic moves by chance; a checkpoint's statistic is within it or not |

| Gate | Gate statistic | What it measures | Counts against a checkpoint when it |
|---|---|---|---|
| G1 | `validity_per_attempt` | accepted whole-ruleset responses per attempt | falls |
| G1 | `truncation_rate` | truncated whole-ruleset attempts per attempt | rises |
| G2 | `complete_map_rate` | accepted responses with a complete rule map | falls |
| G3 | `kappa_bank` | Cohen's kappa of the judgments against the base bank's `q̂`, chance by source | falls |
| G4 | `mode_agreement_nonyes` | agreement with the bank's mode, where that mode is not yes | falls |
| G5p | `probe_or_yes_share` | probe draws whose isolated rules include a yes | moves either way |
| G5p | `probe_rule_yes_rate` | isolated per-rule yes rate on the probe | moves either way |
| G6 | `many_yes_map_share` | complete maps with at least 29 of 57 yes labels | rises |
| G6 | `flag_precision` | yes labels in a map on rules the bank ever said yes to | falls |
| G6 | `map_yes_count` | yes labels per complete map | rises |
| G7 | `median_think_tokens` | median thinking length, against half the base's | is below |
| G7 | `empty_think_share` | responses with empty thinking, against 2% | is above |

G1-G4, G6 and G7 read stage 1: two whole-ruleset draws of a checkpoint on 150 development scenarios, and the non-yes
rows (`m1`, G1, G2, G6 and G7 on the 150 alone). G5p reads the probe: one draw of each rule on 100 of the 150.

A tolerance comes from the base against itself. Each of 10,000 resamples takes, per scenario, one split of the base's
draws into as many draws as a checkpoint has and the reference draws, and computes the statistic. The tolerance is the
95th percentile of these, in the direction that counts, and at least 0.

On G2, G3 and G4 the tolerance is also at least 3/n, with n the observations on the reference side at a checkpoint's
draw count, so a statistic the base never moves does not get a tolerance of zero. `configs/training/report.yaml` holds
these settings.

## The published checkpoints

`ruleforge grpo report saves` selects the reported save with the lowest `m1` (on stage 1's scenarios of the
300-scenario development set). Each published checkpoint is its selection over its run's saves (every 100 steps).

| Checkpoint | `m1` [95% CI] | Base `m1` |
|---|---|---|
| Gemma-4-31B-it checkpoint-600 | 0.3387 [0.2742, 0.4069] | 0.7168 |
| Qwen3-8B checkpoint-300 | 0.0503 [0.0335, 0.0694] | 0.8366 |

The Hub holds the development set (`training/inputs/development.jsonl`), the merge manifests and Qwen3-8B's parity
reports; it does not hold the other saves or their development campaigns.

## Parity: does a served adapter compute what it trained?

vLLM can serve an adapter wrongly without an error, which makes every adapter-served number wrong while the run looks
healthy. The parity check scores frozen token sequences on Hugging Face Transformers with PEFT (HF) and on vLLM, and
gates them overall and per position bucket, so a failure in long contexts is not averaged away.

These commands are a prefill-only check of an unmerged Qwen3-8B adapter. The decode and greedy-judgment checks need
further commands (`--decode-sequences`, `--greedy-prompts`, and `ruleforge grpo parity greedy-agreement`).

```bash
P=$RULEFORGE_ARTIFACT_ROOT/parity/my-check; C=configs/model/qwen3-8b-fp32.yaml; A=<adapter dir>
uv run ruleforge grpo parity vllm --model-config $C --adapter $A --prompts <prompts.jsonl> \
    --sequences $P/sequences.jsonl --out $P/vllm.json
uv run ruleforge grpo parity hf --model-config $C --adapter $A --sequences $P/sequences.jsonl --out $P/hf.json
uv run ruleforge grpo parity compare --preset b_lora --sequences $P/sequences.jsonl --hf $P/hf.json \
    --vllm $P/vllm.json --report $P/report.json
```

`vllm` samples the sequences from the prompts and writes them, or reads them if the file exists. `--prompts` and
`--greedy-prompts` take JSONL rows with `system_prompt` and `user_prompt`.

| Gate | Checks | What it catches |
|---|---|---|
| Finite and not stuck | every logprob finite; no completion repeating one token | NaN logits from a LoRA kernel |
| Serving gap | the served arm agrees with HF no worse than the base arms agree | an engine serving the adapter wrongly; a dropped module; a wrong weight name |
| Minimum effect | the adapter's HF effect is measurable above the noise | an effect too weak to test, reported as that and not as a wrong delta |
| Paired delta | each token's served delta matches HF's | a wrong sign, an unrelated delta, a dropped module family, a silent no-op |
| Retention `rho` | `sum(dM·dH) / sum(dH²)`, served delta against HF delta, with a bootstrap interval | a no-op (low) and a mis-scaled adapter (high) |
| `rho_hf` | `rho` with the merged directory in HF in place of the engine | a merge that over-applies or drops part of the delta; a `rho` off 1 with `rho_hf` at 1 points at the engine |
| Greedy agreement | teacher-forced argmax agreement between HF and the served arm | a served arm that scores close but picks other tokens |
| Greedy judgment | the whole-ruleset judgments of greedy generations on both engines | a change in what the judge actually answers |
| Effect ratio | mean served effect over HF effect, reported; gated only when a range is asked for | the unmerged no-op signature (near 0) |
| Input checks | refuses vLLM files from differing engines, HF arms not in float32, a decode arm at another batch | comparing unlike things |

| Preset | Used for | `rho` | `rho_hf` |
|---|---|---|---|
| `b` | a real checkpoint, merged | at least 0.99, interval half-width at most 0.01 | 0.99 to 1.01 |
| `m0` | a synthetic adapter, before training | 0.99 to 1.01, per bucket too | |
| `b_lora`, `m0_lora` | an adapter served unmerged | 0.95 to 1.05 | |

The greedy-judgment gate reads an answer after the last `</think>`, so it does not apply to a judge whose thinking
closes otherwise, such as Gemma.

`ruleforge grpo parity plumbing` adds the merge's own checks: the merged directory's manifest and digest, a second
merge's digest, what a bf16 cast would round away, and the campaign identity it changes.
