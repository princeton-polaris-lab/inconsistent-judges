from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ruleforge.judges.batch_api import config_confirms_spend, spend_confirmed
from ruleforge.evaluate.campaign import build_plan, check_job_roots, execution_settings, preflight, run_plan
from ruleforge.evaluate.identity import CAMPAIGN_MODE_SCREENING
from ruleforge.core.provenance import require_checkout
from ruleforge.evaluate.runner import STATE_COMPLETE, draw_ended
from ruleforge.launch.programs import REPO_ROOT, run_program
from ruleforge.tasks.spec import build_task_spec
from ruleforge.judges.backend import ModelBackend
from ruleforge.optimize.job_logging import ensure_job_logging

logger = logging.getLogger(__name__)

CONFIG_NAME = "sampling_campaign"


def run(cfg: DictConfig) -> None:
    """Run one condition: its preflight, then its plan and draws; a draw left unfinished fails."""
    require_checkout(REPO_ROOT)
    ensure_job_logging(out_dir=Path(HydraConfig.get().runtime.output_dir))
    preflight(cfg)
    confirmed = config_confirms_spend(cfg)
    execution = execution_settings(cfg)
    task_spec = build_task_spec(cfg)
    backend = ModelBackend(cfg.judge_model)
    try:
        plan = build_plan(cfg, backend=backend, task_spec=task_spec)
        with spend_confirmed() if confirmed else nullcontext():
            completions = run_plan(plan, backend=backend, task_spec=task_spec, execution=execution)
    finally:
        backend.close()
    states = {draw: record["state"] for draw, record in completions.items()}
    logger.info("Condition %s draw states: %s", plan.condition.condition_id, json.dumps(states))

    def finished(record: dict) -> bool:
        # a screening draw ends with some scenarios exhausted; a pilot or final draw must be complete
        if plan.config.campaign_mode == CAMPAIGN_MODE_SCREENING:
            return draw_ended(record)
        return record["state"] == STATE_COMPLETE

    incomplete = {draw: record["state"] for draw, record in completions.items() if not finished(record)}
    if incomplete:
        # RuntimeError, not SystemExit: Hydra marks the job failed while a multirun's other jobs continue
        raise RuntimeError(
            f"{len(incomplete)} draw(s) of {plan.condition.condition_id} did not "
            f"complete: {incomplete}. Nothing is aggregated until they do."
        )


def main() -> None:
    """Run `run` under Hydra, every job's roots and places checked first."""
    run_program(run, CONFIG_NAME, check=check_job_roots, opens_log=True)


if __name__ == "__main__":
    main()
