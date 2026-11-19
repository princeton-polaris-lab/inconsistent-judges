# Measuring inconsistency

How to measure a judge's inconsistency rate on your own ruleset: with a real judge, with several judges or prompts
side by side, or with a model on your own server. Run the README's first run before this page if you have not.

Sections: [Run it](#run-it-on-your-ruleset), [Pick a judge](#pick-a-judge), [Your own server](#use-your-own-server),
[Compare](#compare-judges-or-prompts), [Outputs](#what-a-run-writes), [The rate](#how-the-rate-is-computed),
[Answers](#how-answers-are-read), [Stops and refusals](#when-a-run-stops-or-is-refused).

## Run it on your ruleset

Rules are JSONL, one `{"id": "Rule 1", "rule": "..."}` per line; scenarios one `{"prompt": "...", "response": "..."}`
per line. The default task (`modelspec`) asks whether the response violates the rules.

```bash
export RULEFORGE_ARTIFACT_ROOT=/path/to/artifacts OPENAI_API_KEY=...
uv run ruleforge evaluate --rules my_rules.jsonl --scenarios my_scenarios.jsonl \
    --judge gpt-5.6-luna --prompt forced --out $RULEFORGE_ARTIFACT_ROOT/sampling/my-ruleset --yes-spend
```

The command prints JSON: `estimates.rate.value` is the inconsistency rate and `estimates.rate.ci` its 95% interval.
If the run is killed, launch the same command again: it resumes from its records.

Each **draw** judges every scenario under the whole ruleset and under each rule alone, at temperature 1. The main
options (`--help` lists every one):

| Option | What it sets |
|---|---|
| `--prompt` | `naive` (the default), `encouraged`, or `forced` (JSON: a judgment per rule, then one for the whole ruleset) |
| `--draws` | draws per scenario; default 5, at least 2 |
| `--attempts` | tries per scenario and draw at a well-formed answer; default 30 |
| `--slurm PROFILE` | submit the command to SLURM instead of running it here |
| `--yes-spend` | let a judge that bills its calls run |

`ruleforge prompts show forced` prints a prompt. The paper's results ran `naive-rules-above`, which says the rules
are above, not below ([reproduce.md](reproduce.md#prompt-ablation)).

## Pick a judge

`--judge` (`judge` in Python) takes one of the [judge forms](../README.md#try-it-on-your-own-ruleset).

| Judge | Needs |
|---|---|
| `fake` | nothing: it measures no model and only checks the pipeline, as the command notes on stderr |
| `openai:MODEL`, `anthropic:MODEL` | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, and `--yes-spend` |
| `local:PATH` | `uv sync --group vllm` and a GPU |
| a config name or YAML file (`ruleforge judges list`) | what its type needs; a local config reads its weights under `$RULEFORGE_MODEL_ROOT` |
| an OpenAI-compatible server | [Use your own server](#use-your-own-server) |

Without `--yes-spend` (`confirm_spend=True` in Python) a paid judge is refused before anything is written. A batch
judge (a config ending in `-batch`) keeps the batches it bought in `$RULEFORGE_BATCH_ROOT`, else in the output
directory's `batch/`; keep that state to resume. `ruleforge campaign cost usage --root ROOT` prices them, on a root
you ran: it refuses a published copy (`published_copy.json`), which holds no full attempt ledgers.

## Use your own server

```yaml
# local-qwen.yaml
model_type: openai_compatible
model_name: Qwen/Qwen3-8B
base_url: http://localhost:8000/v1
api_key_env: null   # the server takes no key
```

```bash
vllm serve Qwen/Qwen3-8B --port 8000 --reasoning-parser qwen3
uv run ruleforge evaluate --rules my_rules.jsonl --scenarios my_scenarios.jsonl --judge local-qwen.yaml
```

`base_url` must serve OpenAI's Chat Completions. The URL is part of the judge's identity: two servers are two judges.
A server started with its reasoning parser returns the thinking apart from the answer; `raw_outputs.jsonl` keeps both.

With `api_key_env: null` the judge sends no key and bills no account, so it needs no `--yes-spend`. Otherwise it reads
the key from the variable `api_key_env` names (`OPENAI_API_KEY` by default, and for the shorthand
`--judge openai_compatible:MODEL@URL`) and counts as paid. No price is known for it.

The key is never written or printed verbatim; one under 8 characters, such as a local server's placeholder, is left as
it is. A judge URL carrying a username and password or a query string is refused, without being repeated. This judge
turns off `httpcore`'s debug trace for the whole process, since that trace logs raw responses before they are
scrubbed: do not turn it back on.

## Compare judges or prompts

A **condition** is one judge, prompt, ruleset and setting. A **campaign** compares conditions on the same frozen
scenarios; its directory is the **campaign root**:

```bash
RULES=my_rules.jsonl SCENARIOS=my_scenarios.jsonl ROOT=$RULEFORGE_ARTIFACT_ROOT/sampling/my-comparison CONDITION=gpt-5.6-luna/forced
uv run ruleforge campaign create $ROOT --scenarios $SCENARIOS --campaign-id my-comparison
uv run ruleforge campaign add $ROOT --scenarios $SCENARIOS --rules $RULES --judge gpt-5.6-luna \
    --prompt forced --condition-id $CONDITION
uv run ruleforge campaign run $ROOT $CONDITION --judge gpt-5.6-luna --rules $RULES --scenarios $SCENARIOS \
    --preflight-only --yes-spend   # every check, no judge call
uv run ruleforge campaign run $ROOT $CONDITION --judge gpt-5.6-luna --rules $RULES --scenarios $SCENARIOS \
    --yes-spend
uv run ruleforge analyze $ROOT
```

Repeat `campaign add` and `campaign run` for each condition. Run `analyze` once all have finished (on a root you ran;
it refuses a published copy): it writes each
condition's rate and the paired contrasts between conditions that differ only in prompt or ruleset.

- A root freezes its scenarios by SHA-256 and refuses a changed file; `--final` demands a count, 500 by default, and
  does not check that they are the paper's population.
- `ruleforge bank run` measures many subsets of one ruleset from one **shared bank** of single-rule judgments
  ([rerun.md](rerun.md#ruleset-size)).

## What a run writes

| Path under the root | Holds |
|---|---|
| `scenario_manifest.json`, `sampling_manifest.json` | the frozen scenarios; the sampling design |
| `<judge>/<prompt>-<fingerprint>/condition.json` | the condition's identity (`--condition-id` names this folder instead) |
| `.../draw-NNN/attempts.jsonl` | every attempt, its judgments and why it was rejected |
| `.../draw-NNN/raw_outputs.jsonl` | every returned text, with what an API judge showed of its thinking (`reasoning_summaries`, `reasoning`) |
| `.../draw-NNN/evaluations.jsonl`, `judge_outputs.jsonl` | the accepted judgments; the parsed outputs |
| `.../draw-NNN/smrc_summary.json`, `smrc_per_scenario.jsonl` | one draw's rate and its per-scenario values ([recorded names](../data/README.md#names-in-the-recorded-data)) |
| `aggregate/estimates.json` | `conditions["<condition id>"]`: `estimate`, `ci_low`, `ci_high`; the contrasts in `style_families` and `ruleset_families` |

`estimates.rate` is the **separate-call** rate, the paper's inconsistency rate: the whole-ruleset judgment against the
single-rule calls. A `forced` run also reports `estimates.same_call`, the **same-call** rate: the whole-ruleset
judgment against the per-rule labels inside that same answer.

## How the rate is computed

| Step | Rule |
|---|---|
| One cell | a scenario's draw counts 1 when its whole-ruleset judgment differs from the strongest of its single-rule judgments (yes = 1, unsure = ½, no = 0) |
| Rate | the mean over scenarios of each scenario's mean over draws |
| Interval | scenario-cluster bootstrap, 10,000 resamples, 2.5th and 97.5th percentiles; a two-stage interval that also resamples the draws sits beside it |
| Paired test | scenario-clustered paired randomization, 100,000 permutations, p = `(extreme + 1) / (resamples + 1)`, Holm-adjusted within a family |

## How answers are read

A `forced` or `forced-grpo` answer is a JSON object whose `full_set_judgment` is the judgment. A `naive`,
`naive-rules-above`, `encouraged` or `naive-hierarchy` answer is a label: a `Final Judgment: <label>` line, else a
first line that is a bare label, else either through formatting (control tokens, code fences, quotes, emphasis), else
a last line that is a label or states the ruleset's judgment.

| Read as `no` | Not read (the judge is asked again) |
|---|---|
| `Final Judgment: no` | `Rule 21: no` (one rule's judgment) |
| `no` alone on the last line | `The answer is no.` (a sentence) |
| `**Final answer:** no` | an earlier line states a different judgment |

## When a run stops or is refused

| Exit | Meaning | What to do |
|---|---|---|
| 0 | done | |
| 1 | the run started (its record is in `<output>.runs/`) and then stopped: a refusal, requests that kept failing in transport, a batch that could not be prepared or resumed, or an error such as a full disk | read the error and the record's `log.txt`; fix the cause, then launch the same command again |
| 2 | refused before the run started (no record): bad arguments or inputs | fix what the message names |

A refusal or handled error ends in one line (`ruleforge -v COMMAND ...` shows its traceback); a missing optional
dependency is one, naming the uv group and pip extra to install. An unexpected failure ends in a traceback.

Only a missing response or a malformed answer is retried; a well-formed but surprising judgment is data.
`--retry-scope component` (the default) asks again for the malformed part alone, `scenario_bundle` for every call of
that scenario and draw.

These defaults (30 attempts, component retries) are `ruleforge evaluate`'s and the Python API's. A configured job
(`ruleforge jobs launch campaign`) takes `configs/sampling_campaign.yaml`'s instead: 5 attempts and `scenario_bundle`
retries, unless its spec sets them, as the paper's specs do.
