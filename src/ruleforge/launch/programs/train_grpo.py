from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from ruleforge.grpo.trainer import export_submitit_distributed_env, run_grpo
from ruleforge.launch.programs import refuse_outputs, run_program, selected
from ruleforge.optimize.job_logging import ensure_job_logging

logger = logging.getLogger(__name__)

CONFIG_NAME = "training/grpo/qwen3_8b"


def run(cfg: DictConfig) -> None:
    """Run one rank of the run (or its preflight)."""
    ensure_job_logging(out_dir=Path(HydraConfig.get().runtime.output_dir))
    exported = export_submitit_distributed_env(os.environ)
    if exported is not None:
        logger.info("submitit task -> torch.distributed rank %s of %s", exported["RANK"], exported["WORLD_SIZE"])
    run_grpo(OmegaConf.to_container(cfg, resolve=True))


def check_job(cfg: DictConfig, *, workdir: Optional[str] = None) -> None:
    """Refuse a run whose output directory (its checkpoints, logs and W&B files, a tree the run owns) lies where nothing
    may write or holds a link; nothing is written."""
    refuse_outputs([(selected(cfg, "output_dir"), "the run's output directory")], workdir, below=True)


def main() -> None:
    """Run `run` under Hydra from the checkout's top, every job's destinations checked first."""
    run_program(run, CONFIG_NAME, check=check_job, protected=True, opens_log=True)


if __name__ == "__main__":
    main()
