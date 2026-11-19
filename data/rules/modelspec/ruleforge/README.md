# RuleForge's revised rulesets

The two amended rulesets the RuleForge experiment compares with the initial one (`../openai_rules_all.jsonl`, 57 rules).
Each is a byte copy of a sweep output.

| File | Source in the sweep | Rows | Figure label | SHA-256 |
|---|---|---|---|---|
| `seed5_winner.jsonl` | job 4 (judge seed 5), `episode_7/rule_amendment/optimal_rule_set.jsonl` | 58 (57 rules + 1 meta-rule) | Revised ruleset 1 | `f2eab3b856b93b49dfa2e71fc168b356d5facc3853b518f4747beab9f6c7058c` |
| `seed9_runner_up.jsonl` | job 8 (judge seed 9), `episode_10/rule_amendment/optimal_rule_set.jsonl` | 59 (57 rules + 2 meta-rules) | Revised ruleset 2 | `d543f4ce67b1def85a23a60ba018c857ce0fb68fce3d11d8ce386c07300b78b5` |

The golden `results/ruleforge/ruleforge.json` keeps the seed labels as its keys. `configs/sampling_ruleforge.yaml` pins
these digests, and a campaign refuses a changed file. A meta-rule is part of the full-ruleset prompt and is never judged
alone, so every condition makes 500 x (1 + 57) = 29,000 requests per draw.
