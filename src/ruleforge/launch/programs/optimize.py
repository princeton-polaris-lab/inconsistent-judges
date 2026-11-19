import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from omegaconf import DictConfig, OmegaConf
import hydra

from ruleforge.core.provenance import require_checkout
from ruleforge.launch.programs import REPO_ROOT, refuse_outputs, run_program, selected
from ruleforge.optimize import ValidatedJudge, WandbLogger, seed_paths
from ruleforge.optimize.api import RUN_RECORD_FNAME, _shares, claim_run_dir
from ruleforge.optimize.judging import refuse_unconfirmed
from ruleforge.tasks.spec import build_task_spec
from ruleforge.optimize.trainer import RuleAmendmentTrainer, check_run_inputs
from ruleforge.judges import JudgeSpec
from ruleforge.judges.backend import COMPATIBLE_MODEL_TYPE, ModelBackend
from ruleforge.judges.batch_api import config_confirms_spend, guard_work_directories, spend_confirmed
from ruleforge.optimize.job_logging import ensure_job_logging
from ruleforge.judges.models import cleanup_vllm

logger = logging.getLogger(__name__)

CONFIG_NAME = "config"


def validated(backend: ModelBackend) -> ValidatedJudge:
    """Return `backend` behind the optimizer's validated retries, with its model config's budget."""
    return ValidatedJudge(backend, num_tries=backend.num_tries, max_retry_seconds=backend.max_retry_seconds)


def stop_engines(*backends: "ModelBackend | None") -> None:
    """Stop each distinct backend's vLLM engine; the first failure is raised once every other was tried."""
    distinct: list[ModelBackend] = []
    for backend in backends:
        if backend is not None and all(backend is not seen for seen in distinct):
            distinct.append(backend)
    failure: BaseException | None = None
    for backend in distinct:
        if backend.llm is not None:
            try:
                cleanup_vllm(backend.llm)
            except (Exception, SystemExit) as error:
                failure = failure or error
    if failure is not None:
        raise failure


def _shares_engine(cfg: DictConfig) -> bool:
    """Return whether the agents run on the judge's engine, at the judge's settings: they name its model. Two
    ``openai_compatible`` configs, whose server is part of the judge, are decided as ``optimize`` decides them: an
    equal pair shares, a pair differing in any option (another server) is refused."""
    agent, judge = cfg.agent_model, cfg.judge_model
    if agent.model_type == judge.model_type == COMPATIBLE_MODEL_TYPE:
        return _shares(JudgeSpec.from_config(agent), JudgeSpec.from_config(judge))
    return agent.model_type == judge.model_type and agent.model_name == judge.model_name


def _refuse_unused_agent_seed(cfg: DictConfig, overrides: Sequence[str]) -> None:
    named = any(override.lstrip("+~").startswith("agent_model.seed=") for override in overrides)
    if named and _shares_engine(cfg) and cfg.agent_model.seed != cfg.judge_model.seed:
        raise ValueError(f"agent_model.seed={cfg.agent_model.seed} is not the seed this run would use: the agents share "
                         f"the judge's engine, which runs at judge_model.seed={cfg.judge_model.seed}. Set the judge's "
                         "seed (a seed sweep sweeps judge_model.seed)")


def engine_keys(cfg: DictConfig) -> tuple[str, ...]:
    """Return the model groups the run builds an engine of: the judge's, and the agents' unless they share it."""
    return ("judge_model",) if _shares_engine(cfg) else ("judge_model", "agent_model")


def check_spend(cfg: DictConfig, *, workdir: Optional[str] = None) -> None:
    """Refuse a run whose agents cannot share the judge's engine as they ask, or whose billed judge or own agents run
    with no spend confirmed; nothing is written."""
    keys = engine_keys(cfg)
    if not config_confirms_spend(cfg):
        refuse_unconfirmed(*(JudgeSpec.from_config(cfg[key]) for key in keys))


def batch_roots(cfg: DictConfig) -> list[tuple[Optional[str], str]]:
    """Return each distinct ``batch_root`` an engine of the run keeps a Batch judge's paid state in, with what it is
    (None where it does not resolve: the engine then refuses to start)."""
    found: dict[Optional[str], str] = {}
    for key in engine_keys(cfg):
        spec = JudgeSpec.from_config(cfg[key])
        if spec.capabilities().batch:
            found.setdefault(selected(cfg, f"{key}.batch_root"), f"the batch judge {spec.model_name}'s batch_root")
    return list(found.items())


def _model_record(model: Mapping[str, Any]) -> dict[str, Any]:
    return {"model_type": model["model_type"], "model_name": model["model_name"], "options": dict(model)}


def run_record(config_map: Mapping[str, Any], *, shared: bool) -> dict[str, Any]:
    """Return what a run records of its engines: under a shared engine the judge's settings are the agents' too, and
    `agent_settings_not_used` holds the agent config's values that differ."""
    judge, agent = config_map["judge_model"], config_map["agent_model"]
    not_used = {key: agent.get(key) for key in sorted(set(judge) | set(agent)) if agent.get(key) != judge.get(key)}
    return {"judge": _model_record(judge), "agent": "judge" if shared else _model_record(agent),
            "agent_settings_not_used": not_used if shared else {}}


def run(cfg: DictConfig):
    """Run one seed in Hydra's run directory, every batch work directory its transports derive checked as
    ``ruleforge optimize`` checks it."""
    import wandb

    from ruleforge.evaluate.api import batch_work_guard

    require_checkout(REPO_ROOT)
    config_map = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run_name = cfg.run_name
    job_name = hydra.core.hydra_config.HydraConfig.get().job.name

    match hydra.core.hydra_config.HydraConfig.get().mode.name:
        case "RUN":
            logger.info("Running in regular mode")
            # a launch fan-out member's index is hydra.job.num
            job_num = OmegaConf.select(hydra.core.hydra_config.HydraConfig.get(), "job.num", default=0)
        case "MULTIRUN":
            logger.info("Running in multirun mode")
            job_num = hydra.core.hydra_config.HydraConfig.get().job.num
        case _:
            raise ValueError(
                f"Unknown hydra mode: {hydra.core.hydra_config.HydraConfig.get().mode}"
            )

    judge_backend = agent_backend = None
    try:
        check_run_inputs(cfg, Path(cfg.rule_amendment.initial_rule_set_path))
        confirmed = config_confirms_spend(cfg)
        task_spec = build_task_spec(cfg=cfg)
        task_spec.check_new_run()
        shared = _shares_engine(cfg)
        _refuse_unused_agent_seed(cfg, hydra.core.hydra_config.HydraConfig.get().overrides.task)
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        check_spend(cfg)  # again for a direct run, before the run directory is claimed
        record = run_record(config_map, shared=shared)
        (claim_run_dir(output_dir) / RUN_RECORD_FNAME).write_text(
            json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        if record["agent_settings_not_used"]:
            logger.warning("The agents share the judge's engine (seed %s); agent_model's other values are not used: %s",
                           cfg.judge_model.seed, record["agent_settings_not_used"])
        with guard_work_directories(batch_work_guard()), wandb.init(
            entity=config_map["wandb"]["entity"],
            project=config_map["wandb"]["project"],
            config=config_map,
            group=config_map["wandb"]["group"],
            name=f"{run_name}-{job_num}",
            job_type=job_name,
            dir=config_map["wandb"]["dir"],
            settings=wandb.Settings(init_timeout=180),
        ):
            logger.info("Initializing trainer")
            judge_backend = ModelBackend(cfg.judge_model)
            agent_backend = judge_backend if shared else ModelBackend(cfg.agent_model)
            judge = validated(judge_backend)
            agent = judge if agent_backend is judge_backend else validated(agent_backend)
            trainer = RuleAmendmentTrainer(
                cfg,
                task=task_spec,
                judge=judge,
                agent=agent,
                rules=Path(cfg.rule_amendment.initial_rule_set_path),
                seeds=seed_paths(cfg.scenario_generation),
                output_dir=output_dir,
                logger=WandbLogger(),
            )
            logger.info("Trainer initialized")
            # model stacks can reset logging while they start
            ensure_job_logging(out_dir=output_dir)
            with spend_confirmed() if confirmed else nullcontext():
                trainer.train()
    except Exception as e:
        logger.exception(f"The run failed: {e}")
        raise
    finally:
        # no `return` here: it would discard the exception in flight and the job would read as succeeded
        if judge_backend is None:
            logger.warning("No backend was built, skipping cleanup")
        else:
            stop_engines(judge_backend, agent_backend)


def check_job(cfg: DictConfig, *, workdir: Optional[str] = None) -> None:
    """Refuse a run ``check_spend`` refuses, or whose W&B directory or batch roots lie where nothing may write; nothing
    is written."""
    check_spend(cfg, workdir=workdir)
    refuse_outputs([(selected(cfg, "wandb.dir"), "the run's W&B directory"), *batch_roots(cfg)], workdir)


def main() -> None:
    """Run `run` under Hydra from the checkout's top, every job's spend and destinations checked first."""
    run_program(run, CONFIG_NAME, check=check_job, protected=True, opens_log=True)


if __name__ == "__main__":
    main()
