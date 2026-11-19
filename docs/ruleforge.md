# Rewriting a ruleset with RuleForge

How to run RuleForge on your own ruleset, pick the best revision and measure it. RuleForge works in **episodes**: each
generates scenarios on which a panel of judge readings disagrees, then edits the rules to remove that disagreement
while keeping their intent.

## Run it

```bash
export RULEFORGE_MODEL_ROOT=/path/to/weights
SEEDS=data/scenarios/seeds/modelspec
uv run ruleforge optimize --task modelspec --rules my_rules.jsonl --judge gemma4-31b-tp-2 \
    --seeds-good $SEEDS/wildchat_5k_gpt-oss-120b_threshold_above_90.jsonl \
    --seeds-bad $SEEDS/wildchat_5k_gpt-oss-120b_threshold_below_10.jsonl \
    --out runs/mine/{judge.seed} --sweep judge.seed=1,2,3
```

This runs three random seeds. A local judge needs `uv sync --group vllm` and its GPUs (two for `gemma4-31b-tp-2`).

| Option | What it sets |
|---|---|
| `--task` | `modelspec`, or `legal`, a small second task to copy from (`src/ruleforge/tasks/legal.py`) |
| `--judge`, `--agent` | the panel's model; the two agents' model (default: the judge's engine) |
| `--seeds-good`, `--seeds-bad` | seed scenarios, JSONL rows with a `prompt` and a `response`, shown to the scenario generator as good and bad examples; the paper's are above |
| `--episodes` | at most this many episodes (default 25) |
| `--sweep KEY=a,b` | one run per value; `{KEY}` in `--out` names each |
| `--slurm PROFILE` | submit the runs to SLURM instead of running them here |
| `--wandb` | log each episode to Weights & Biases |
| `--yes-spend` | let a judge that bills its calls run |

A run stops after `--episodes`, or earlier when the rule amender's last 15 rewards all exceed 0.9 or strictly
decrease, or the scenario generator's last 3 rewards all fall below 0.1.

A scenario's ambiguity is the entropy, in bits, of the panel's full-ruleset judgments of it, each "unsure" counted as half
a "yes" and half a "no". The rule amender's reward is 1 minus the mean ambiguity, less the weighted intent penalty; the
scenario generator's is the mean ambiguity.

With a separate `--agent`, the judge's seed seeds the panel alone and the agents run at `agent.seed`, so sweep that
instead (`--out runs/mine/agent.seed.{agent.seed}`). The `fake` judge cannot drive `optimize`: its answers hold no scenarios.

## Pick a winner and measure it

```bash
uv run ruleforge optimize select --sweep-dir runs/mine
WINNER=$(uv run python -c "import json; print(json.load(open('runs/mine/selection.json'))['winner']['optimal_rule_set_path'])")
uv run ruleforge evaluate --rules $WINNER --scenarios my_scenarios.jsonl --judge gemma4-31b-tp-2 --prompt forced \
    --rule-context implicated_ranges
```

A revised ruleset may add meta-rules. `--rule-context implicated_ranges` judges each rule with the meta-rules and the
rules they name, a range such as "Rules 27-43" read as every rule in it; a range whose end is a number it cannot read
as a rule ("Rules 3-50.5", "Rules 5-3") is refused. It does not handle a rule number in prose after a reference and a
comma ("Rule 3, 5 times a day"), a day-first date whose day is a rule number, or a spelled-out number at either end of
a range ("Rules three-5", "Rules 3 through fourteen", read as Rule 3): write such a sentence another way, or use
`implicated`.

`select` writes `runs/mine/selection.json`. Its `winner` is the seed whose last complete episode has the highest
effective reward, and `optimal_rule_set_path` names that episode's ruleset. Each seed scores its rulesets on the
scenarios it generated itself, so seeds are ranked on different scenarios. Runs that never finished an amending
episode are left out. A winner's training reward is not its inconsistency rate: measure it ([measure.md](measure.md)).

## What it writes

| Path under `runs/mine/<seed>/episode_<N>/` | Holds |
|---|---|
| `rule_amendment/optimal_rule_set.jsonl` | the episode's ruleset |
| `rule_amendment/reward_optimal.jsonl` | its rewards and penalties |
| `rule_amendment/sample_<i>/` | a kept candidate's patch |
| `scenario_generation/sample_<B>/` | the scenarios that drove the episode |
| `*_trace.json*`, `judge_outputs.jsonl` beside each output | every model call: raw response, parsed completion, any reasoning a provider showed |

## When a call fails

| What happens | What the run does |
|---|---|
| A patch cannot be applied, or its intent cannot be scored | skips the candidate, with `patch_error.txt` in its `sample_<i>/`; with none left, the previous ruleset stands |
| An answer stops at the output ceiling | asks again, up to the model config's `num_tries`, then keeps the old prior or reads no scenarios; a cut-off candidate patch is skipped |
| An answer holds nothing outside its thought | asks again, up to `num_tries`; then the old prior stands, and a scenario or patch call left with no answer stops the run, as a blank answer does |
| A local model's prompt is over its limit | that call fails, with the prompt's token count in its trace |
| A panel judgment fails | retries up to 30 attempts, for at most an hour per flush (one batch of judgments), then drops it; a scenario whose panel lost a judgment is left out of both rewards, and each reward records its `judgment_count` |

## The paper's runs

New runs use the `naive` prompt. For the paper's prompt and sweep commands, see [rerun.md](rerun.md#ruleforge). The
paper's sweep ran ten seeds, two GPUs a seed, up to 48 hours a seed, with Gemma-4-31B-it as the judge and both agents.
When the paper measured its two revised rulesets, with `implicated`, each single-rule call also carried the meta-rules
and the rules they name, a range such as "Rules 27-43" read as its first rule, Rule 27.
