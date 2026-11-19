# Extending the toolkit

How to add your own task, judge config, provider or metric. Read it when the built-in task (`modelspec`: does a
response violate the rules?) or the built-in judges do not fit what you want to measure.

## Add a task

A task, a subclass of `ruleforge.tasks.Task`, decides what a judge is asked and how its answers are read. The
smallest one, in a module of your own:

```python
from ruleforge.tasks import LABEL, Task


class ToneTask(Task):
    name = "tone"
    scenario_fields = ("prompt", "response")
    styles = {"plain": LABEL}  # each style's answer: LABEL (yes / unsure / no) or FORCED_JSON
    default_style = "plain"

    def render(self, rule_text, scenario, settings, *, context_rules=None):
        return (f"Does the response break these rules? Answer yes, no or unsure.\n\nRules:\n{rule_text}\n\n"
                f"User: {scenario['prompt']}\nResponse: {scenario['response']}")
```

Save it as `tone_task.py`; from its directory, `PYTHONPATH=. uv run ruleforge evaluate --task tone_task:ToneTask ...`
runs it.

`styles` maps each prompt style to its answer format, `ruleforge.tasks.LABEL` or `ruleforge.tasks.FORCED_JSON`.
`render` gets one rule's text, or the whole ruleset's, and one scenario; `name` (with an optional `version`) is
recorded in each condition. The inconsistency rate needs `prompt` and `response`; other fields run `--mode full`.

Name a task with `--task package.module:Class`, register it with `ruleforge.tasks.register_task`
(`register_task("tone", ToneTask)`), or install it as an entry point of the `ruleforge.tasks` group.
`ruleforge tasks list` lists every task it finds. The built-in `ruleforge.tasks.LegalTask`
(`src/ruleforge/tasks/legal.py`, with its rules and seed scenarios in `tests/fixtures/legal/`) is a fuller one to copy.

`ruleforge optimize` runs only the built-in tasks, `modelspec` and `legal`; a task of your own runs in campaigns.
RuleForge's interface is `ruleforge.tasks.OptimizableTask`, which `src/ruleforge/tasks/legal.py` implements.

## Add a judge config

A judge that a built-in provider serves is a model config. Copy one, edit it, and check what it describes:

```bash
cp configs/model/qwen3-8b.yaml configs/model/my-judge.yaml   # then edit model_path and the settings
uv run ruleforge judges show my-judge
```

Then `--judge my-judge` uses it. A name is looked up in `model/` beside a `--config` file (for `evaluate`, `optimize`
and `bank run`), then in the checkout's `configs/model/`, then in the package's `src/ruleforge/configs/model/`; a
config kept elsewhere is passed by its path. A server that speaks Chat Completions needs only such a config
([measure.md](measure.md#use-your-own-server)). The configs read some values through the package's resolvers
(`${rf.env:...}`, `${rf.recorded:...}`), which `import ruleforge` registers: a script of your own that composes them
with Hydra or OmegaConf imports `ruleforge` first, or fails with `UnsupportedInterpolationType`.

| Key of a local vLLM config | Value |
|---|---|
| `model_path` | `"${oc.env:RULEFORGE_MODEL_ROOT}/<weights directory>"` |
| `dtype` | part of a campaign condition's identity |
| `strip_thinking_model` | for a thinking model: `qwen3`, `gemma4`, `olmo3`, `openai_gptoss`, `muse_glimmer` |

A batch judge with `batch_root: ${oc.env:RULEFORGE_BATCH_ROOT}` keeps its state there, or in the campaign root's
`batch/` when that variable is unset, whether the config is named or given by its path. A configured launch
(`ruleforge jobs launch`) needs the variable set.

## Add a provider

Another API or engine is a `ruleforge.judges.Provider` serving a `model_type` of its own. Register it with
`ruleforge.judges.register_judge`, or install it as an entry point of the `ruleforge.judges` group.
`EchoProvider` in `tests/judges/test_providers.py` is the smallest one to copy; a real one's `start` returns a judge.

| Member | What it gives |
|---|---|
| `start` | returns a `ruleforge.judges.Judge` |
| `capabilities` | what that judge can do (`ruleforge.judges.Capabilities`) |
| `identity` | every option that can change an answer, in its `ruleforge.judges.extension`; an option left out would let two different judges share one condition |

## Add a metric

RuleForge's scores are registries of named functions under `src/ruleforge/optimize/metrics/`, picked by name in a
run's config. Copy an entry, such as `ENTROPY_METHODS["consensus"]` in `entropy.py` (judgments in, a float out), add
yours to the registry and name it in the config.

| Registry | File | Picked by |
|---|---|---|
| `ENTROPY_METHODS` | `entropy.py` | `rule_amendment.entropy_method` |
| `INTENT_METHODS` | `intent.py` | `rule_amendment.intent_method` |
| `RA_REWARD_METHODS` | `reward.py` | `rule_amendment.ra_reward_method` (the Hydra program only) |
| `SG_REWARD_METHODS` | `reward.py` | `scenario_generation.sg_reward_method` (the Hydra program only) |

The first two are settings of `ruleforge optimize` and of `ruleforge.optimize.RAConfig`. The Hydra program is
`python -m ruleforge.launch.programs.optimize`; `ruleforge optimize` scores both rewards with `standard`.
