---
license: cc-by-4.0
tags: [ruleforge, multi-rule-consistency, llm-evaluation, not-for-all-audiences]
pretty_name: "Inconsistent judges: data behind the paper"
configs:
  - config_name: scenarios
    data_files: populations/frozen_500_r3.jsonl
  - config_name: development
    data_files: training/inputs/development.jsonl
---

# Inconsistent judges: the paper's data

[![Paper](https://img.shields.io/badge/Paper-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.29975)
[![Website](https://img.shields.io/badge/Website-2563EB)](https://www.polarislab.org/inconsistent-judges/)
[![Code](https://img.shields.io/badge/Code-181717?logo=github&logoColor=white)](https://github.com/princeton-polaris-lab/inconsistent-judges)
[![Models](https://img.shields.io/badge/Models-181717?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/ai-law-society-lab/inconsistent-judges)

The judgment records behind the figures and tables of *Aligning Judge Models to Be Logically Consistent Multi-Rule
Interpreters*, the scenarios they judge, and the training inputs. The code repository's commands below download them
and rebuild the paper's numbers from them; the GRPO adapters are in the model repository (the Models button above).

Michel Liao<sup>1,2</sup>, Nimra Nadeem<sup>2</sup>, Luxi He<sup>2</sup>, Peter Henderson<sup>1,2</sup>

<sup>1</sup>MATS &nbsp;&nbsp;&nbsp; <sup>2</sup>POLARIS Lab, Princeton University

> **Content warning.** The scenarios include real user messages from WildChat and self-harm related content from WeirdChat. Tagged `not-for-all-audiences`.

## Use

```bash
uv sync                                                  # in the code repository; no GPU required
export RULEFORGE_ARTIFACT_ROOT=/path/to/an/empty/directory   # where the data lands
uv run ruleforge data fetch                              # everything
uv run ruleforge data fetch ruleset-size training        # or some folders, or single roots by name
uv run ruleforge data verify --root "$RULEFORGE_ARTIFACT_ROOT"
uv run ruleforge reproduce all rebuild --out /path/to/a/new/directory   # every golden, each equal to its copy in results/
```

`docs/reproduce.md` in the code repository has each experiment's commands; the hierarchy comparison also reads three
roots of `prompt-ablation/`.

## What is here

A **campaign root** is one campaign's judgments, records and estimates, on the paper's 500 scenarios with five draws
each. Each is an archive, `roots/<root>.tar.zst`, with the digest of every member in `roots/<root>.files.sha256`.
`SHA256SUMS` lists every file, and `index.json` says where each lands under an artifact root.

| folder | paper | holds |
|---|---|---|
| `populations/` | every experiment | `frozen_500_r3.jsonl`: the 500 evaluation scenarios |
| `ruleset-size/` | Figures 3, 10 | per judge, a bank of single-rule judgments and ten seeded rule subsets at each size; size 57 in a root of its own, each seed's full ruleset in an order of its own |
| `prompt-ablation/` | Figures 1, 4, 9; Table 3 | six judges under the naive, encouraged and forced prompts |
| `ruleforge/` | Figure 6 | the initial and two revised rulesets, judged; `inputs/`: the ten-seed RuleForge run they came from |
| `training/` | Figures 5, 11, 12, 13 | Gemma-4-31B-it and Qwen3-8B, each base and trained |
| `training/selection/` | Figures 5, 12 | the merge manifests, Qwen3-8B's parity reports |
| `training/inputs/` | Appendix F | training rows, replay rows, each anchor condition's `condition.json`, the 300-scenario development set (`development.jsonl`) |
| `hierarchy/` | Figures 7, 8 | the hierarchy prompt's arms of six judges, and the naive arms of three |

Root names are the names the campaigns ran under: their `fig2-`, `fig4-` and `fig5-` prefixes are not the paper's
figure numbers.

| root | judges and arms | paper |
|---|---|---|
| `fig2-2026-09-pop-a22af097-rescored-lenient_last_line` | Gemma-4-31B-it, Olmo-3-32B-Think, gpt-oss-120b: bank, sizes 2 to 56 | Figures 3, 10 |
| `fig2-2026-09-luna-pop-a22af097-rescored-lenient_last_line` | GPT-5.6 Luna: bank, sizes 2 to 56 | Figures 3, 10 |
| `fig2-2026-10-haiku-sum-rescored-lenient_last_line` | Claude Haiku 5.5: bank, sizes 2 to 56 | Figures 3, 10 |
| `fig2-2026-10-glimmer-pop-a22af097` | Muse-Glimmer-30B: bank, sizes 2 to 56 | Figures 3, 10 |
| `k57-shuffled-2026-11` | Gemma-4-31B-it, Olmo-3-32B-Think, gpt-oss-120b: size 57 | Figures 3, 10 |
| `k57-shuffled-2026-11-luna` | GPT-5.6 Luna: size 57 | Figures 3, 10 |
| `k57-shuffled-2026-11-haiku` | Claude Haiku 5.5: size 57 | Figures 3, 10 |
| `k57-shuffled-2026-11-glimmer` | Muse-Glimmer-30B: size 57 | Figures 3, 10 |
| `final-2026-09-pop-a22af097-rescored-lenient_last_line` | Olmo-3-32B-Think, Gemma-4-31B-it, gpt-oss-120b: naive | Figure 4a, Table 3 |
| `final-2026-09-luna-pop-a22af097-rescored-lenient_last_line` | GPT-5.6 Luna: naive | Figure 4a, Table 3; Figures 7, 8 |
| `prompt-ablation-v2-2026-10-pop-a22af097-rescored-lenient_last_line` | Gemma-4-31B-it: encouraged, forced | Figures 1, 4, 9; Table 3 |
| `prompt-ablation-v2-open-2026-10-pop-a22af097-rescored-lenient_last_line` | Olmo-3-32B-Think, gpt-oss-120b: encouraged, forced | Figures 1, 4, 9; Table 3 |
| `prompt-ablation-v2-2026-10-luna-pop-a22af097-rescored-lenient_last_line` | GPT-5.6 Luna: encouraged, forced | Figures 1, 4, 9; Table 3 |
| `prompt-ablation-v2-2026-10-haiku-sum` | Claude Haiku 5.5: all three | Figures 1, 4, 9; Table 3; Figures 7, 8 |
| `prompt-ablation-v2-2026-10-glimmer-pop-a22af097` | Muse-Glimmer-30B: all three | Figures 1, 4, 9; Table 3; Figures 7, 8 |
| `fig4-2026-09-pop-a22af097-rescored-lenient_last_line` | Gemma-4-31B-it: the three rulesets | Figure 6 |
| `fig5-gemma-2026-10-07-pop-a22af097` | Gemma-4-31B-it, base | Figures 5, 11 |
| `fig5-gemma-ckpt600-2026-10-09-pop-a22af097` | Gemma-4-31B-it, trained (checkpoint-600) | Figures 5, 11 |
| `fig5-grpo-v2-fp32-2026-09-26-checkpoint-200-pop-a22af097` | Qwen3-8B, base | Figures 12, 13 |
| `fig5-qwen-mix-2026-10-11-checkpoint-300-pop-a22af097` | Qwen3-8B, trained (checkpoint-300) | Figures 12, 13 |
| `hierarchy-aware-2026-10-pop-a22af097` | Olmo-3-32B-Think, Gemma-4-31B-it, gpt-oss-120b: naive and hierarchy | Figures 7, 8 |
| `hierarchy-aware-2026-10-glimmer` | Muse-Glimmer-30B: hierarchy | Figures 7, 8 |
| `hierarchy-aware-2026-10-haiku-sum` | Claude Haiku 5.5: hierarchy | Figures 7, 8 |
| `hierarchy-aware-2026-10-luna-sum` | GPT-5.6 Luna: hierarchy | Figures 7, 8 |

The figures read size 57 from the `k57-shuffled-2026-11*` roots; the four bank roots keep only the records of their
own size-57 conditions, whose calls judged the file's order.

**What a root holds.** Each archive holds what the paper's rebuilds read of its root, by the readers of its kind:
those files and, of each, only the rows and fields they read, with a `published_copy.json` that says it is a published
copy. A draw's attempt ledger is published as `accepted_attempts.json`: of each accepted scenario's full-ruleset call,
what its root's readers read, without the attempts that were retried. Commands that read more (`campaign cost
usage`, `analyze`, `bank analyze`, a cap amendment's disclosure) refuse it; they run on a root you ran. Recorded files keep their names; `data/README.md` in the code repository maps
them to the names the paper and the code use, and lists what each root leaves out. No published file names a local
path or a host the jobs ran on: run logs and provenance keep no host and no model directory; `rebase.json`, the training manifests and the parity reports name what they read by its `artifact://` path,
model name or repository path; each trained arm's provenance names its merge manifest by the published copy's digest.
The re-scores' `label_replay.json` and the merge manifests' adapter-config digests name those files as recorded.

A few roots leave out condition directories no result reads; their aggregates keep those rows, and `data/README.md`
in the code repository lists them.

## Licenses

| | |
|---|---|
| Our outputs | CC-BY-4.0, POLARIS Lab, Princeton University |
| Scenario texts | their sources' terms: WildChat (ODC-BY-1.0), Stress-Testing Model Specs (Apache-2.0), WeirdChat (MIT), the OpenAI Model Spec (CC0-1.0) |
| API judges' outputs | those providers' terms |

See `data/README.md` in the code repository.

## Citation

```bibtex
@misc{liao2026aligning,
  title         = {Aligning Judge Models to Be Logically Consistent Multi-Rule Interpreters},
  author        = {Liao, Michel and Nadeem, Nimra and He, Luxi and Henderson, Peter},
  year          = {2026},
  eprint        = {2611.29975},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2611.29975}
}
```
