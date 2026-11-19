from __future__ import annotations

FULL_RULESET_KEY = "full_ruleset"
# At the top of a published copy of a campaign root: commands that read more than the paper's rebuilds do refuse it.
PUBLISHED_COPY_FNAME = "published_copy.json"
# In a published copy's draw, in place of its attempt ledger: what the rebuilds read of the ledger's strict replay.
PUBLISHED_LEDGER_FNAME = "accepted_attempts.json"

OPTIMAL_RULE_SET_FNAME = "optimal_rule_set.jsonl"
OPTIMAL_SCENARIO_SET_FNAME = "optimal_scenario_set.jsonl"
ACCUMULATED_SCENARIO_SET_FNAME = "accumulated_scenario_set.jsonl"
PRIOR_FNAME = "prior.jsonl"
POSTERIOR_SAMPLE_FNAME = "posterior_sample.jsonl"
PATCH_FNAME = "posterior_sample.patch"
PATCH_ERROR_FNAME = "patch_error.txt"
EVALUATIONS_FNAME = "evaluations.jsonl"
ENTROPY_FNAME = "entropy.jsonl"
BATCH_RESULTS_FNAME = "batch_results.jsonl"
REWARD_ALL_FNAME = "reward_all.jsonl"
REWARD_OPTIMAL_FNAME = "reward_optimal.jsonl"

# Per-call traces: each attempt's raw response, completion and reasoning, beside the result it produced.
JUDGE_OUTPUTS_FNAME = "judge_outputs.jsonl"
INTENT_TRACE_FNAME = "intent_penalty_trace.json"
PRIOR_TRACE_FNAME = "prior_trace.jsonl"
POSTERIOR_SAMPLING_TRACE_FNAME = "posterior_sampling_trace.jsonl"