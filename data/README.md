# Data

Where every data file comes from, its license, and whether it holds text written by real users;
[Licenses and attributions](#licenses-and-attributions) has the license terms and the attributions.

> **Content warning.** The scenarios include real user messages from WildChat, which can be offensive or explicit,
> and WeirdChat transcripts that describe self-harm and suicide.

## The 500-scenario population

`data/populations/frozen_500_r3.jsonl` is the population every paper number judges: 500 single-turn prompts and
responses. Each row carries `prompt`, `response`, `source`, `source_key` and upstream metadata in `meta`.

| Source | Rows | Dataset (revision, config, split) | License | Real-user text |
|---|---|---|---|---|
| WildChat | 167 | `allenai/WildChat-4.8M` (`c827c6df8fcf008219ffaffa4d1dd77491099367`, `default`, `train`) | ODC-BY-1.0 | yes, de-identified upstream |
| Stress-Testing Model Specs | 167 | `jifanz/stress_testing_model_spec` (`a9f4320b03a8e08823ea9905271a8256e1729a2f`, `default`, `all_models`) | Apache-2.0 | no: model-generated |
| WeirdChat | 166 | `Transluce/WeirdChat` (`3460e41491308601cd68ee377011c7b7a8f0b4b3`, `transcripts`, `train`) | MIT | no: automatically elicited |

The population's WildChat rows come from one candidate pool or from more rows the same builder drew from the same
revision; its Stress-Testing Model Specs rows come from one candidate pool.

| Digest | Value |
|---|---|
| SHA-256 of the file | `c16f66f4ad3fc93fad46bf31efaa8325f30fb750104d1e13a9326b7cd46b10d4` |
| SHA-256 of its scenario manifest | `a22af097639cfdaa15b4508c66631af1cb6d45780d8ada93a9fb1d5a29390462` |

Every paper campaign pins the file's digest and refuses any other file. Campaigns read it from the artifact root
([docs/rerun.md](../docs/rerun.md)).

**How it was built.** A 600-row candidate pool (200 per source) was screened for validity under every judge and
prompt, and 500 survivors were drawn source-balanced. Three rounds of replacement kept the 167/167/166 split.

| Round | Rows replaced | Which rows |
|---|---|---|
| First draw | — | the 500 survivors of the screen |
| 1 | 1 | row 288, replaced by an unused screening survivor |
| 2 | 5 | scenarios Claude Haiku 5.5's API refuses |
| 3 | 95 | template near-duplicates, replaced from the pool, then from more rows of the same sources |

Of the final 500 rows, 100 differ from the first draw's. The population's record ([Records](#records)) lists every
removed row and replacement, and round 3's rule for a duplicate.

## Rulesets

| File | Rows | Source | License |
|---|---|---|---|
| `data/rules/modelspec/openai_rules_all.jsonl` | 57 | the OpenAI Model Spec, one rule per row, by its title | CC0-1.0 |
| `data/rules/modelspec/openai_rules_all_full.jsonl` | 57 | the same rules with type and explanation; `rule_type` is each rule's level of authority; the ruleset-size subsets are drawn from it | CC0-1.0 |
| `data/rules/modelspec/ruleforge/seed5_winner.jsonl`, `seed9_runner_up.jsonl` | 58, 59 | RuleForge's revisions of the 57 rules, plus one or two meta-rules ([their README](rules/modelspec/ruleforge/README.md)) | derived from CC0-1.0 text |
| `data/rules/subset_manifest/manifest.{json,sha256}` | 1,400 subsets | seeds and SHA-256s of the ruleset-size experiment's random rule subsets; `ruleforge reproduce ruleset-size subsets` regenerates them | this project's |

None holds real-user text.

## Scenario files

| File | Rows | Source | License | Real-user text |
|---|---|---|---|---|
| `data/scenarios/seeds/modelspec/wildchat_5k_gpt-oss-120b_threshold_above_90.jsonl` | 1,832 | RuleForge's seed scenarios: first user turns of `allenai/WildChat-1M` conversations, with gpt-oss-120b judge labels | ODC-BY-1.0 | yes, de-identified upstream |
| `data/scenarios/seeds/modelspec/wildchat_5k_gpt-oss-120b_threshold_below_10.jsonl` | 1,097 | same | ODC-BY-1.0 | yes, de-identified upstream |

## Test fixtures with outside text

| File | Source | License |
|---|---|---|
| `tests/fixtures/rules/openai_subset_prolific.jsonl` | seven rules of the OpenAI Model Spec | CC0-1.0 |
| `tests/fixtures/legal/legal_gms_rules.jsonl`, `legal_gms_seeds.jsonl` | the text of 30 C.F.R. § 100.3(c) and questions written for this repository about *GMS Mine Repair v. FMSHRC* (D.C. Cir. 2023) | public domain; the questions are this project's |

## Licenses and attributions

The code is under the MIT License (`LICENSE`). On the Hugging Face Hub, this project's outputs are under CC-BY-4.0, and
the third-party text inside them under the licenses below.

Where this project changed third-party rows, it added fields: judge labels, source keys and upstream metadata.
The prompt and response text is as published upstream.

The installed package (the ruleforge wheel) holds copies of some of these files inside its own directory, each at
ruleforge/ followed by its path above: data/populations/frozen_500_r3.jsonl and the rulesets under
data/rules/modelspec/. Each copy is under the license listed for its path, as is the original. The wheel carries this
file among its license files.

| Source | License | Attribution |
|---|---|---|
| WildChat | Open Data Commons Attribution License (ODC-BY) v1.0, https://opendatacommons.org/licenses/by/1-0/ | Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi and Yuntian Deng. WildChat: 1M ChatGPT Interaction Logs in the Wild. ICLR 2024. https://openreview.net/forum?id=Bl8u7ZRlbM |
| Stress-Testing Model Specs | Apache License, Version 2.0, https://www.apache.org/licenses/LICENSE-2.0 | Jifan Zhang, Henry Sleight, Andi Peng, John Schulman and Esin Durmus. Stress-Testing Model Specs Reveals Character Differences among Language Models. 2025. |
| WeirdChat | MIT, as the dataset card declares; the project's notice is below | Neil Chowdhury, Cassidy Laidlaw, Kaiying Hou, Daniel Johnson, Sarah Schwettmann and Jacob Steinhardt. WeirdChat: A catalog of unexpected AI behaviors, discovered automatically. 2026. https://transluce.org/weirdchat, doi:10.57967/hf/9670 |
| The OpenAI Model Spec, https://github.com/openai/model_spec | dedicated to the public domain under Creative Commons CC0 1.0, https://creativecommons.org/publicdomain/zero/1.0/ | |
| 30 C.F.R. § 100.3(c) | the regulation's text is a work of the U.S. government and is in the public domain | |

WeirdChat's notice:

```text
Copyright (c) 2026 Transluce

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Records

| File | What it is |
|---|---|
| `data/populations/frozen_500_r3.jsonl.manifest.json` | how the population was built: every removed row with its reason, every replacement with its source row |
| `data/hub_inventory.yaml`, `data/hub_cards/` | what each Hub folder holds; the Hub cards |

## Personal information in the third-party text

WildChat holds real user conversations, de-identified by its maintainers; their dataset card says how to ask for a
removal. A few e-mail-shaped strings remain as published upstream: configs and tests pin these files by SHA-256.

## The data on the Hugging Face Hub

The dataset `ai-law-society-lab/inconsistent-judges` holds the campaigns and the inputs their reruns read, one folder
per experiment ([its card](hub_cards/dataset.md)), and a copy of the 500-scenario population; the model repository of
the same name holds the two GRPO adapters. [docs/reproduce.md](../docs/reproduce.md) has the fetch commands.

The training folder also holds the 300-scenario development set on which the training report selected the published
checkpoints (`training/inputs/development.jsonl`, [docs/training_report.md](../docs/training_report.md)): 100 rows from
each of the population's three sources, under their licenses, with real-user text in its WildChat rows. No row's
prompt is one of the 500 or of the training scenarios.

A root first judged under a stricter label reading (the rule that reads a label answer) is published re-scored as
`<root>-rescored-lenient_last_line`. Its recorded root holds a `label_replay.json` listing the draws the re-score
changed; the published prompt-ablation roots keep it, the others leave it out.

Each published campaign root keeps what the paper's rebuilds (`ruleforge reproduce <experiment> rebuild`) read of
it, by the readers of its kind (below), and a `published_copy.json` naming the conditions whose ledgers it holds, by
which the code knows a published copy. The populations, the development set, the RuleForge run, the training rows, the
merge manifests and the parity reports are published whole; of each training anchor condition the folder holds its
`condition.json`, all the training launch reads of it. `ruleforge campaign cost usage`, `ruleforge analyze`,
`ruleforge bank analyze` and a cap amendment's disclosure read more than a published root holds: they refuse one and
run on a root you ran. `data/hub_inventory.yaml` holds the exact patterns and names each root's kind, and `PROFILES`
in `src/ruleforge/hub.py` the rows and fields.

Three published roots leave out condition directories no result reads; each root's aggregate keeps their rows.

| Root | Left out |
|---|---|
| `final-2026-09-luna-pop-a22af097-rescored-lenient_last_line` | `gpt-5.6-luna/forced-grpo` (under its recorded key), and a retired prompt's condition named below |
| `fig5-gemma-2026-10-07-pop-a22af097` | `gemma-4-31B-it/figure5/grpo` |
| `fig5-grpo-v2-fp32-2026-09-26-checkpoint-200-pop-a22af097` | `Qwen3-8B/figure5/grpo` |

Every root also leaves out its run records (`hydra/`, `code_snapshots/`), `**/smrc_summary.json`,
`**/implicated_rules.json`, its own `completion.json`, `**/condition_stack.json` and its `.ruleforge-*` markers, and
each draw's `judge_outputs.jsonl`, `raw_outputs.jsonl`, `evaluations.jsonl`, per-scenario file and `attempts.jsonl`
but these: the ledger of the forced conditions, and of every condition in the local judges' prompt-ablation and
hierarchy roots, GPT-5.6 Luna's re-scored prompt-ablation roots and the training roots; `raw_outputs.jsonl` and
`draw-000/evaluations.jsonl` of the training roots. A kept ledger is published as `accepted_attempts.json`, written
by the ledger's strict replay: of each accepted scenario's accepted attempt and the full-ruleset call its bundle holds,
and of the attempts' start times and judging stacks, what its kind reads (below). Retried and superseded attempts are
not in it.

| Roots | Also leave out |
|---|---|
| ruleset-size banks (`fig2-*`) | `**/runs.jsonl`, `**/draw_provenance.json`, `**/request_blocks.jsonl`, their own size-57 draws (`*/subsets/*/k57/draw-*/*`), `sampling_manifest.json`, `scenario_manifest.json`, `aggregate/figure2_estimates.json`, and `label_replay.json` or `rebase.json` |
| size 57 (`k57-shuffled-2026-11*`) | `**/runs.jsonl`, `**/draw_provenance.json`, `**/request_blocks.jsonl`, `sampling_manifest.json` |
| prompt ablation and hierarchy | a local judge's `**/request_blocks.jsonl`; GPT-5.6 Luna's and Claude Haiku 5.5's `**/runs.jsonl` and `**/draw_provenance.json`; `rebase.json` of `prompt-ablation-v2-2026-10-glimmer-pop-a22af097` and `hierarchy-aware-2026-10-pop-a22af097` |
| `fig4-2026-09-pop-a22af097-rescored-lenient_last_line` | `**/runs.jsonl`, `**/draw_provenance.json`, `**/request_blocks.jsonl`, `label_replay.json` |
| training | `**/request_blocks.jsonl` |

Each kind keeps, of each draw file it holds, only these rows and fields; the packer refuses a root file of a kind
none of these rules names. Every kind's `bundle_outcomes.jsonl` keeps `scenario_id` and `decoder_draw_id`, and its
`completion.json` `state`, `decoder_draw_id`, `scenario_manifest_sha256` and `condition_fingerprint`. A runtime keeps the
package versions (`vllm`, `torch`, `transformers`, `tokenizers`, `triton`), the three precision settings and the engine
arguments, without the model and the placement arguments: what the stack checks read.

The other root files are kept whole: each `estimates.json` and size 57's `figure2_estimates.json`, whose digests the
goldens record (`estimates_sha256`); each `label_replay.json`, whose digest the prompt-ablation and hierarchy goldens
record; each `scenario_manifest.json`, whose content digest every draw's completion record holds; and the banks'
`figure2_design.json`, every field of which the ruleset-size rebuild reads.

| Kind | Roots |
|---|---|
| ruleset size | `fig2-*`, `k57-shuffled-2026-11*` |
| prompt ablation, local judge | `final-2026-09-pop-a22af097-rescored-lenient_last_line`, `prompt-ablation-v2-2026-10-pop-a22af097-rescored-lenient_last_line`, `prompt-ablation-v2-open-2026-10-pop-a22af097-rescored-lenient_last_line`, `prompt-ablation-v2-2026-10-glimmer-pop-a22af097` |
| prompt ablation, API judge | `final-2026-09-luna-pop-a22af097-rescored-lenient_last_line`, `prompt-ablation-v2-2026-10-luna-pop-a22af097-rescored-lenient_last_line`, `prompt-ablation-v2-2026-10-haiku-sum` |
| hierarchy, local judge | `hierarchy-aware-2026-10-pop-a22af097`, `hierarchy-aware-2026-10-glimmer` |
| hierarchy, API judge | `hierarchy-aware-2026-10-haiku-sum`, `hierarchy-aware-2026-10-luna-sum` |
| RuleForge | `fig4-2026-09-pop-a22af097-rescored-lenient_last_line` |
| training | `fig5-*` |

| File | Kind | Also keeps |
|---|---|---|
| `bundle_outcomes.jsonl` | ruleset size | `full_set_judgment`, `rule_judgments`, `ruleset_rule_ids` |
| | prompt ablation | `contradiction`, `full_set_judgment`, `rule_judgments`, `in_context_rule_judgments`, `semantic_attempt_id` |
| | hierarchy | `contradiction`, `full_set_judgment`, `rule_judgments` |
| | RuleForge | `contradiction` |
| | training | `contradiction`, `full_set_judgment`, `rule_judgments`, `full_set_phi`, `max_rule_phi` |
| `completion.json` | ruleset size; hierarchy, API judge | nothing more |
| | prompt ablation | `materialized`, `campaign_id`, `condition_id`; a local judge's also `run_log_rows` |
| | hierarchy, local judge; training | `run_log_rows` |
| | RuleForge | `materialized`, `campaign_id`, `condition_id`, `accepted` |
| `draw_provenance.json` | local judge | `model_type`, the runtime |
| | training | `model_type`, `model_weights`, `attention_backend`, `tp_size`, the runtime |
| `runs.jsonl` | local judge | `stack_change`, the runtime |
| | training | `code_revision`, `code_dirty`, `code_state_sha256`, `stack_change`, the runtime |
| `request_blocks.jsonl` | API judge | `recorded_at` |
| `draw-000/evaluations.jsonl` | training | `scenario_id` and the scenario's `source` |
| `raw_outputs.jsonl` | training | the rows of the draw's accepted full-ruleset calls: `scenario_id`, `semantic_attempt_id`, `request_block_id`, `component_id`, `raw_response`; Gemma-4-31B-it's also `completion` |
| `condition.json` | every kind | every field but `schema_version`: what its fingerprint covers, its id and fingerprint |
| `sampling_manifest.json` | prompt ablation | `campaign_id`, `campaign_mode`, `schema_version`, `max_semantic_attempts`, `generation_seed_base`, `retry_scope`, `scenarios_per_request_block`, `expected_decoder_draws`, `scenario_manifest_sha256` |
| | hierarchy | `campaign_id`, `campaign_mode`, `schema_version`, `max_semantic_attempts`, `generation_seed_base`, `retry_scope`, `scenarios_per_request_block` |
| | RuleForge | `campaign_id`, `campaign_mode`, `schema_version`, `max_semantic_attempts`, `expected_decoder_draws`, `expected_scenarios`, `forced_json_requirement`, `scenario_manifest_sha256`, `temperature`, `truncation` |
| | training | every field but `conditions` and `created_at`: the family's design check compares the rest |
| `rebase.json` | training | `removed_ids`, `added_ids`, the parent's `campaign_id`, `scenario_manifest_sha256` and `sampling_manifest_sha256`, the target's `scenario_manifest_sha256`; of each link of its `chain`, the parent's identity, the target, `removed_ids` and `added_ids` |
| `accepted_attempts.json` | prompt ablation | each accepted attempt with its full-ruleset judgment and rule judgments; a local judge's also the stacks, an API judge's the first and last start time |
| | hierarchy, local judge | the stacks |
| | training | each accepted full-ruleset call, its judgment and its raw output's digest; the stacks |

<!-- names-note -->
## Names in the recorded data

Published files keep the names their campaigns were recorded under. Runs made with this code write the public names.

The code reads the prompts' recorded keys through `src/ruleforge/core/record_names.json`, and fingerprints hash the
recorded key, so the table changes no identity.

| Recorded key | Public name |
|---|---|
| `legacy` | `naive-rules-above` |
| `good_v2` | `encouraged` |
| `even_better_v2` | `forced` |
| `even_better` | `forced-grpo` |
| `legacy_hierarchy` | `naive-hierarchy` |
| `legacy_v2` | `naive` |

`naive` says the rules are provided below, as `encouraged` and `forced` do, and is the prompt for new runs;
`naive-rules-above` says above, and is what the paper's results ran.

`good` and `even_better_no_preemption` name prompts outside the paper: they have no public name; the code reads no
condition recorded under either, and writes nothing into a root that holds one. `gpt-5.6-luna/good` is the condition the Luna root leaves out above; its aggregate
keeps the row.

A **label reading** is the rule that reads a label answer
([docs/measure.md](../docs/measure.md#how-the-rate-is-computed)); each root records the one it was scored under.

| Label reading | Roots |
|---|---|
| `lenient_last_line`, the one reader | every root but these |
| `lenient` | The four training roots and the two training anchor conditions are the exception: their answers are JSON, which no label reader reads |
| `lenient_markup` | the open judges' hierarchy root, `hierarchy-aware-2026-10-pop-a22af097`, is published as recorded too, under `lenient_markup`; the hierarchy golden reads it so |

Other recorded names kept as they are, in the data and in what new runs write:

| Name | Is |
|---|---|
| `smrc_summary.json`, `smrc_per_scenario.jsonl`, `smrc_provenance.json` | every campaign's per-draw views; the adapters' provenance |
| root prefixes `fig2-`, `fig4-`, `fig5-` | the ruleset-size, RuleForge and training campaigns; their numbers are not the paper's figure numbers |
| `figure2_estimates.json`, `figure2_design.json` | a shared-bank campaign's estimates and design |
| `figure4`, `figure5` in condition ids and in `condition_extra.figure` | a RuleForge and a training condition: `<judge>/figure4/<ruleset>`, `<judge>/figure5/<policy>` |
| the Hub's `.../finetuning/...` destination paths | where the merge manifests were recorded; the training builds read them there |
| the subset manifest's generator fields | fields of record, which nothing reads |

The recorded roots' run configs and launch records, which their published copies leave out, spell the config keys
`ruleset_size`, `ruleforge` and `training` as `figure2`, `figure4` and `figure5`.
<!-- /names-note -->
