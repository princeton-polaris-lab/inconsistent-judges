---
license: other
license_name: cc-by-4.0-and-gemma-terms
license_link: https://ai.google.dev/gemma/terms
library_name: peft
tags: [ruleforge, grpo, lora, multi-rule-consistency]
---

# Inconsistent judges: GRPO LoRA adapters

[![Website](https://img.shields.io/badge/Website-2563EB)](https://www.polarislab.org/inconsistent-judges/)
[![Code](https://img.shields.io/badge/Code-181717?logo=github&logoColor=white)](https://github.com/princeton-polaris-lab/inconsistent-judges)
[![Dataset](https://img.shields.io/badge/Dataset-181717?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/ai-law-society-lab/inconsistent-judges)

Two LoRA adapters from *Aligning Judge Models to Be Logically Consistent Multi-Rule Interpreters*, one for
Gemma-4-31B-it and one for Qwen3-8B, trained with GRPO to make a judge's whole-ruleset judgment agree with its
judgments of the rules one at a time. The paper's data are in the dataset repository (the Dataset button above).

Michel Liao<sup>1,2</sup>, Nimra Nadeem<sup>2</sup>, Luxi He<sup>2</sup>, Peter Henderson<sup>1,2</sup>

<sup>1</sup>MATS &nbsp;&nbsp;&nbsp; <sup>2</sup>POLARIS Lab, Princeton University

| subfolder | base model | checkpoint (step) | license |
|---|---|---|---|
| `gemma4-31b-it/checkpoint-600/` | google/gemma-4-31b-it | 600 | adapter CC-BY-4.0 where the Gemma Terms of Use allow; use with Gemma is governed by the Gemma Terms of Use and Prohibited Use Policy |
| `qwen3-8b/checkpoint-300/` | Qwen/Qwen3-8B | 300 | adapter CC-BY-4.0 (base model Apache-2.0) |

**Training.** Training uses rank-32 LoRA and the Forced Prompt for GRPO (Table 4; Appendix E). Each checkpoint here is
the save `ruleforge grpo report saves` selects in its run: the reported save with the lowest development inconsistency
(`m1`) on a 300-scenario development set (Appendix F.4).

**Files.** Each subfolder holds `adapter_model.safetensors`, `adapter_config.json` and `trainer_state.json`.
`adapter_config.json` names the base model by its Hub id; load the base from the Hub as below. `ruleforge grpo merge`
takes the id's last segment as `--expect-base-model` (`Qwen3-8B`, `gemma-4-31b-it`), the base directory's name too;
Gemma-4-31B-it's merge also takes `--expect-scope model.language_model.`.
A merge through the Python API (`ruleforge.grpo.merge`) records no training-run name unless `training_run` is given;
`ruleforge grpo merge` is unaffected.

**Load.** Load the base model, then the adapter; use the base model's tokenizer and chat template.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = PeftModel.from_pretrained(base, "ai-law-society-lab/inconsistent-judges", subfolder="qwen3-8b/checkpoint-300")
```

The paper evaluated the adapters merged into their bases in fp32; `docs/training.md` in the code repository gives the
merge and the evaluation.

**Use.** For research on rule-following consistency. Training changes how often each judgment is given (the paper's
Figures 5c and 12c count them); after training, Qwen3-8B rarely answers no to a whole ruleset (Appendix F.3).
The adapters are not safety-tuned models.
