# RuleForge

[![Website](https://img.shields.io/badge/Website-2563EB)](https://www.polarislab.org/inconsistent-judges/)
[![Dataset](https://img.shields.io/badge/Dataset-181717?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/ai-law-society-lab/inconsistent-judges)
[![Models](https://img.shields.io/badge/Models-181717?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/ai-law-society-lab/inconsistent-judges)

Code of *Aligning Judge Models to Be Logically Consistent Multi-Rule Interpreters*.

Michel Liao<sup>1,2</sup>, Nimra Nadeem<sup>2</sup>, Luxi He<sup>2</sup>, Peter Henderson<sup>1,2</sup>

<sup>1</sup> <a href="https://www.matsprogram.org/"><img src="assets/logos/mats.png" alt="" height="24" align="absmiddle"> MATS</a>
&nbsp;&nbsp;&nbsp;
<sup>2</sup> <a href="https://www.polarislab.org/"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logos/polaris-lab-dark.svg"><img src="assets/logos/polaris-lab-light.svg" alt="" height="24" align="absmiddle"></picture> POLARIS Lab</a>, Princeton University

A judge model is **inconsistent** on a scenario when its judgment under a whole ruleset differs from the strongest of
its judgments on the rules one at a time (yes = 1, unsure = ½, no = 0); the **inconsistency rate** is the share of
inconsistent judgments over repeated samples at temperature 1. The paper measures this rate for several judges and
lowers it two ways: by rewriting the rules (**RuleForge**) and by training the judge (**GRPO**). This repository
reproduces the paper and runs both methods on your own ruleset.

## Install

With [uv](https://docs.astral.sh/uv/) and Python 3.12; `uv.lock` is the paper's environment.

```bash
git clone https://github.com/princeton-polaris-lab/inconsistent-judges && cd inconsistent-judges
uv sync
```

| Command | For | Needs |
|---|---|---|
| `uv sync` | API and `fake` judges; rendering and rebuilding the paper | any machine, no GPU |
| `uv sync --group vllm` | the above, and local vLLM judges; RuleForge with a local judge | Linux x86_64, NVIDIA GPU, CUDA 12.8+ driver |
| `uv sync --group vllm --group grpo` | the above, and GRPO training (TRL, PEFT, DeepSpeed) | the same |

Run every `ruleforge` command below as `uv run ruleforge ...`; `uv run` keeps whatever an earlier `uv sync` installed.
To use your own environment instead, run `pip install -e ".[providers,viz,hub]"` from the clone and drop `uv run`: it
supports the API and `fake` judges and rendering and rebuilding the paper; the test tools come from uv's `dev` group.
The other extras are `openai` and `anthropic` (one provider each), `slurm` and `wandb`; for `vllm` and `grpo`, install
torch and vLLM built for your CUDA first. Exact reproduction is tied to `uv.lock`: pip resolves its own versions.

## Reproduce the paper

A **golden** is a committed JSON file under `results/` that holds an experiment's numbers.

| Level | What it does | Command | Needs | Time |
|---|---|---|---|---|
| Render | draws the figures and tables from the goldens | `ruleforge reproduce all render --out DIR` | nothing | about a minute |
| Rebuild | recomputes the goldens from the downloaded run records | `ruleforge data fetch`, then `ruleforge reproduce all rebuild --out DIR` | the dataset's download | a few CPU minutes |
| Rerun | judges and trains again; a new sample | one command per step, in [docs/rerun.md](docs/rerun.md) | GPUs, paid APIs | hours to days |

```bash
uv run ruleforge reproduce all render --out /tmp/paper-figures
```

[docs/reproduce.md](docs/reproduce.md) maps each figure and table to its experiment, golden and command.

## Try it on your own ruleset

Rules are JSONL, one `{"id": "Rule 1", "rule": "..."}` per line; scenarios are JSONL rows with a `prompt` and a
`response`. Try the paper's 57 rules on 20 scenarios:

```bash
head -n 20 data/populations/frozen_500_r3.jsonl > /tmp/first-run-scenarios.jsonl
```

```python
import ruleforge  # from the repository root, in uv run python

result = ruleforge.evaluate("data/rules/modelspec/openai_rules_all.jsonl", "/tmp/first-run-scenarios.jsonl", "fake",
                            prompt="forced",  # answer in JSON: a judgment per rule, then the whole ruleset's
                            draws=2, out="/tmp/first-run")
print(result.estimates.rate.value, result.estimates.rate.ci)  # 0.625 (0.45, 0.775)
```

```bash
# the same run; prints JSON, whose estimates.rate holds value and ci
uv run ruleforge evaluate --rules data/rules/modelspec/openai_rules_all.jsonl \
    --scenarios /tmp/first-run-scenarios.jsonl --judge fake --prompt forced --draws 2 --out /tmp/first-run
```

`fake` is a free stand-in that measures no model: its numbers only check the pipeline, as the command notes on stderr.
A real judge takes one of these forms, as `judge` in Python or `--judge` on the command line:

| Judge | What it is | Needs |
|---|---|---|
| `fake` | the stand-in | nothing |
| `openai:MODEL`, `anthropic:MODEL` | the provider's API | its key; `confirm_spend=True` or `--yes-spend` |
| `local:PATH` | vLLM over the weights in `PATH` | `--group vllm`, a GPU |
| `openai_compatible:MODEL@URL` | your own OpenAI-compatible server | [docs/measure.md](docs/measure.md#use-your-own-server) |
| a config name or YAML file | the paper's judges (`ruleforge judges list`), e.g. `gpt-5.6-luna`, `gemma4-31b-tp-2` | per its type |

Then [measure](docs/measure.md) judges and prompts side by side, [rewrite a ruleset](docs/ruleforge.md) with RuleForge
(`ruleforge optimize`), [train a judge](docs/training.md) with GRPO (`ruleforge grpo train`), or
[add](docs/extending.md) a task, metric, judge or provider.

## Configuration

Copy `.env.example` to `.env`, uncomment what you set, then `set -a; . ./.env; set +a`.

| Variable | Needed for |
|---|---|
| `RULEFORGE_ARTIFACT_ROOT` | Campaign roots, populations, fetched Hub data, launch records (`.ruleforge-launch/`) |
| `RULEFORGE_MODEL_ROOT` | Local weights: a local judge reads `$RULEFORGE_MODEL_ROOT/<weights directory>` |
| `RULEFORGE_BATCH_ROOT` | Batch API judges: the paid batch state, so a resumed run never buys a batch twice |
| `RULEFORGE_SLURM_PARTITION`, `RULEFORGE_SLURM_ACCOUNT` | `ruleforge jobs launch` on SLURM without a profile (partition required, account optional) |
| `RULEFORGE_CLUSTER_DIR` | Your cluster profiles (`--slurm NAME`, `--profile NAME`); copy `src/ruleforge/configs/cluster/example.yaml` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN` | The API judges and services that use them |
| `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_MODE` | RuleForge's and GRPO's logging; `WANDB_MODE=offline` needs no account |

Paying is a flag, not a variable: `--yes-spend`, `confirm_spend=True`, or `-- +confirm_spend=true` on a `jobs launch`
line.

## Layout

| Path | Holds |
|---|---|
| `src/ruleforge/evaluate/`, `judges/`, `tasks/`, `prompts/` | campaigns, judge backends, tasks, prompt styles and the answer reader |
| `src/ruleforge/rulesets/`, `scenarios/` | rulesets and their subsets; scenario sets |
| `src/ruleforge/optimize/`, `grpo/` | RuleForge; judge training with GRPO |
| `src/ruleforge/experiments/` | the paper's result builds and figures |
| `src/ruleforge/cli/`, `launch/`, `core/` | the `ruleforge` command; SLURM and local launches; paths, digests, recorded names, errors |
| `configs/`, `data/`, `results/` | model, task, campaign, training and launch configs; the population, rules and seeds; the goldens |
| `docs/`, `tests/` | the guides ([docs/development.md](docs/development.md): goldens, identity tests, style); the test suite |

## Data and licenses

Every paper number judges one 500-scenario population, `data/populations/frozen_500_r3.jsonl`: 167 scenarios from
WildChat, 167 from Stress-Testing Model Specs, 166 from WeirdChat. [data/README.md](data/README.md) lists every file
with its source, revision and license, and carries the attributions.

| What | License |
|---|---|
| Code | MIT (`LICENSE`) |
| This project's data outputs (judgments, estimates, rows, adapters) | CC-BY-4.0; the Gemma adapter also under the Gemma Terms of Use |
| WildChat / Stress-Testing Model Specs / WeirdChat / OpenAI Model Spec text | ODC-BY-1.0 / Apache-2.0 / MIT / CC0-1.0 |
| Judge outputs from API providers | the providers' terms |
| The MATS and POLARIS Lab logos (`assets/logos/`) | their owners'; not covered by the licenses above |

**Content warning.** The scenarios include real user messages from WildChat, which can be offensive or explicit,
and WeirdChat transcripts that describe self-harm and suicide.

## Tests

[docs/development.md](docs/development.md#tests) has the test and lint commands.

## Cite

```bibtex
@misc{liao2026aligning,
  title  = {Aligning Judge Models to Be Logically Consistent Multi-Rule Interpreters},
  author = {Liao, Michel and Nadeem, Nimra and He, Luxi and Henderson, Peter},
  year   = {2026}
}
```
