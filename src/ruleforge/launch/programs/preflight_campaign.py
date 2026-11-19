from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from ruleforge.evaluate.campaign import (
    campaign_rule_levels,
    check_job_roots,
    execution_settings,
    expected_requests_per_draw,
    non_meta_rule_count,
    preflight,
    requested_draw_ids,
    resolve_sampling_truncation,
)
from ruleforge.judges.providers import judge_adapter, judge_serving_dtype
from ruleforge.judges.weights import recorded_model_weights
from ruleforge.evaluate.sharding import assign_units, blocks_per_draw
from ruleforge.core.artifacts import load_jsonl_records
from ruleforge.launch.programs import run_program

CONFIG_NAME = "sampling_campaign"


def run(cfg: DictConfig) -> None:
    """Check the condition and print what it would generate."""
    config, manifest = preflight(cfg)
    rules = load_jsonl_records(Path(str(cfg.rule_set_path)))
    non_meta = non_meta_rule_count(rules)
    adapter = judge_adapter(cfg.judge_model)
    local = str(cfg.judge_model.model_type) == "local"
    weights = recorded_model_weights(str(cfg.judge_model.model_path)) if local else None
    settings = execution_settings(cfg)
    levels = campaign_rule_levels(cfg)
    sharding = None
    if settings.sharded:
        n_blocks = blocks_per_draw(manifest.record_count, config.scenarios_per_request_block)
        shards = assign_units(requested_draw_ids(cfg, config), n_blocks, settings.num_shards)
        sharding = {
            "num_shards": settings.num_shards,
            "request_blocks_per_draw": n_blocks,
            "units_per_shard": [len(units) for units in shards],
        }
    report = {
        "campaign_id": str(cfg.campaign_id),
        "condition_id": str(cfg.condition_id),
        "campaign_mode": config.campaign_mode,
        "judge_model": str(cfg.judge_model.model_name),
        "prompt_style": str(cfg.eval_cfg.reasoning_prompt_style),
        "individual_rule_context": str(cfg.eval_cfg.get("individual_rule_context", "isolated")),
        "rule_levels": None if levels is None else {
            **levels.levels_file, "rules_per_level": {level: len(ids) for level, ids in levels.rules_by_level().items()},
        },
        "scenarios": manifest.record_count,
        "scenario_source_sha256": manifest.source_sha256,
        "scenario_manifest_sha256": manifest.manifest_sha256,
        "rules": len(rules),
        "non_meta_rules": non_meta,
        "num_decoder_draws": config.num_decoder_draws,
        "max_semantic_attempts": config.max_semantic_attempts,
        "evaluation_mode": str(cfg.evaluation_mode),
        "requests_per_draw": expected_requests_per_draw(manifest, rules, str(cfg.evaluation_mode)),
        "forced_json_requirement": config.forced_json_requirement,
        "label_rule": config.label_rule,
        "retry_scope": config.retry_scope,
        "strict_think": config.strict_think,
        "sampling_truncation": resolve_sampling_truncation(
            config, str(cfg.judge_model.model_type), cfg.judge_model
        ),
        "adapter": None if adapter is None else {
            "label": adapter.label, "sha256": adapter.sha256, "rank": adapter.rank,
            "base_model": adapter.base_model,
        },
        "serving_dtype": judge_serving_dtype(cfg.judge_model) if local else None,
        "model_weights": None if weights is None else {
            "path": weights.path, "sha256": weights.sha256, "kind": weights.kind, "saved_dtype": weights.saved_dtype,
        },
        "sharding": sharding,
    }
    print(json.dumps(report, indent=2))


def main() -> None:
    """Run `run` under Hydra, every job's roots and places checked first."""
    run_program(run, CONFIG_NAME, check=check_job_roots)


if __name__ == "__main__":
    main()
