from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from omegaconf import OmegaConf

from ruleforge.judges import Judge, JudgeError, JudgeSpec
from ruleforge.optimize.config import OptimizeConfig
from ruleforge.optimize.judging import ValidatedJudge, refuse_unconfirmed
from ruleforge.optimize.loggers import RunLogger
from ruleforge.rulesets import Ruleset, RulesetError
from ruleforge.rulesets.rules import duplicate_rule_id_refusal, round_trip_refusal
from ruleforge.tasks.spec import build_task_spec
from ruleforge.tasks import BUILTIN_TASKS, Task, TaskError, load_task
from ruleforge.core.artifacts import jsonl_records_of
from ruleforge.core.constants import REWARD_OPTIMAL_FNAME

#: Created in `out` before any judge starts; holds the run's record and copies of its inputs.
RUN_RECORD_DIR = ".ruleforge-optimize"
RUN_RECORD_FNAME = "run.json"
#: Copies of the inputs the run reads, so a change to the caller's files cannot reach it.
INITIAL_RULES_FNAME = "initial_rule_set.jsonl"
SEED_FNAME = "seed_scenarios_{role}.jsonl"


@dataclass(frozen=True)
class OptimizeRun:
    """What a run did: its output directory, its last episode, what stopped it, and that episode's optimal ruleset and
    reward."""

    out: Path
    last_episode: int
    stopped_by: tuple[str, ...]
    ruleset_path: Path
    reward: Mapping[str, Any]

    @property
    def ruleset(self) -> Ruleset:
        """Return the last episode's optimal ruleset."""
        return Ruleset.load(self.ruleset_path)


def _ruleset(rules: Union[Ruleset, Path, str]) -> Ruleset:
    if isinstance(rules, Ruleset):
        if rules.path is None:
            raise RulesetError(f"ruleset {rules.name} has no file: the optimizer reads its ruleset from one")
        loaded = Ruleset.load(rules.path, name=rules.name, expect_sha256=rules.sha256)
        if loaded.rules != rules.rules:
            raise RulesetError(f"ruleset {rules.name}'s rules are not its file's ({rules.path}), which the "
                               "optimizer reads: save them to a file of their own and load that")
    else:
        loaded = Ruleset.load(rules)
    _check_round_trip(loaded)
    refusal = duplicate_rule_id_refusal([rule.id for rule in loaded.rules], f"ruleset {loaded.name}")
    if refusal is not None:
        raise RulesetError(refusal)
    return loaded


def _check_round_trip(ruleset: Ruleset) -> None:
    refusal = round_trip_refusal(ruleset.rows(), f"ruleset {ruleset.name}", exact_ids=True)
    if refusal is not None:
        raise RulesetError(refusal)


def _task(task: Union[Task, str]) -> Task:
    found = load_task(task)
    if not found.builtin:
        raise TaskError(f"task {found.identifier!r}: optimize runs the built-in tasks {list(BUILTIN_TASKS)} alone "
                        "(their task specs read the scenario generator's seeding and the initial ruleset)")
    return found


def _seed_files(task: Task, seeds: Mapping[str, Path | str]) -> dict[str, tuple[Path, bytes]]:
    files = {}
    for role, given in seeds.items():
        path = Path(given)
        try:
            data = path.read_bytes()
            rows = jsonl_records_of(data, path)
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read the {role!r} seed scenario file {path}: {error}") from error
        if not rows:
            raise ValueError(f"the {role!r} seed scenario file {path} holds no scenarios")
        try:
            task.check_scenarios(rows)
        except TaskError as error:
            raise ValueError(f"the {role!r} seed scenario file {path}: {error}") from error
        files[role] = (path, data)
    return files


def _checked_judge(judge: Any, name: str) -> Union[Judge, JudgeSpec]:
    if not isinstance(judge, (Judge, JudgeSpec)):
        raise TypeError(f"a {name} is a Judge or a JudgeSpec, got {type(judge).__name__}")
    return judge


def _budgeted(judge: Union[Judge, JudgeSpec], config: OptimizeConfig) -> Union[Judge, JudgeSpec]:
    if (isinstance(judge, JudgeSpec) and judge.provider.builtin
            and judge.options.max_retry_seconds != config.max_retry_seconds):
        return judge.with_options(max_retry_seconds=config.max_retry_seconds)
    return judge


def _shares(agent: Union[Judge, JudgeSpec, None], judge: Union[Judge, JudgeSpec]) -> bool:
    if agent is None or agent is judge or agent == judge:
        return True
    if (isinstance(agent, JudgeSpec) and isinstance(judge, JudgeSpec)
            and (agent.model_type, agent.model_name) == (judge.model_type, judge.model_name)):
        ours, theirs = judge.options.written(), agent.options.written()
        differing = sorted(key for key in set(ours) | set(theirs) if ours.get(key) != theirs.get(key))
        raise JudgeError(
            f"the agent names the judge's model ({judge.model_type} {judge.model_name!r}) with other options "
            f"{differing}: one engine would serve both at the judge's settings. Set them on the judge (a seed sweep "
            "sweeps the judge's seed) and pass agent=None, or pass a started judge (spec.start()) as the agent for "
            "an instance of its own")
    return False


def _started(judge: Union[Judge, JudgeSpec], started: list[Judge]) -> Judge:
    if isinstance(judge, JudgeSpec):
        instance = judge.start()
        started.append(instance)
        return instance
    return judge


def _close(started: Sequence[Judge]) -> None:
    failure: Optional[Exception] = None
    for instance in reversed(started):
        try:
            instance.close()
        except Exception as error:
            failure = failure or error
    if failure is not None:
        raise failure


def claim_run_dir(out: Path) -> Path:
    """Make `out`'s run record directory and return it, refused if a run has claimed `out` or written episodes there."""
    if any(out.glob("episode_*")):
        raise FileExistsError(f"{out} holds a run's episodes: a run writes into a directory of its own")
    record_dir = out / RUN_RECORD_DIR
    try:
        record_dir.mkdir(parents=True)
    except FileExistsError:
        raise FileExistsError(f"{out} was claimed by another run ({RUN_RECORD_DIR}): a run writes into a directory "
                              "of its own") from None
    return record_dir


def _claim(out: Path) -> Path:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"{out} exists and is not an empty directory: a run writes into a directory of its own")
    return claim_run_dir(out)


def _judge_record(judge: Union[Judge, JudgeSpec]) -> dict[str, Any]:
    spec = judge if isinstance(judge, JudgeSpec) else judge.spec
    return {"model_type": spec.model_type, "model_name": spec.model_name, "options": spec.options.written(),
            "started_by": "optimize" if isinstance(judge, JudgeSpec) else "caller"}


def optimize(rules: Union[Ruleset, Path, str], *, task: Union[Task, str], judge: Union[Judge, JudgeSpec],
             agent: Union[Judge, JudgeSpec, None] = None, seeds: Mapping[str, Path | str], out: Path | str,
             config: OptimizeConfig = OptimizeConfig(), logger: Optional[RunLogger] = None) -> OptimizeRun:
    """Run RuleForge on `rules` for a built-in `task` with `judge` as the panel and `agent` (the judge when None) as both
    agents, writing every episode under `out`, which must not exist or be empty, and closing after the run a judge
    given as a spec, which it starts."""
    from ruleforge.optimize.trainer import RuleAmendmentTrainer, check_run_inputs  # deferred: it imports this package

    ruleset = _ruleset(rules)
    found = _task(task)
    given_config = config.to_config(task=found.name, rules=ruleset.path, seeds=seeds)
    build_task_spec(cfg=given_config).check_new_run()
    check_run_inputs(given_config, ruleset.path)
    seed_files = _seed_files(found, seeds)
    judge = _budgeted(_checked_judge(judge, "judge"), config)
    agent = None if agent is None else _budgeted(_checked_judge(agent, "agent"), config)
    shared = _shares(agent, judge)
    validated: list[ValidatedJudge] = []

    def validated_for(instance: Judge) -> ValidatedJudge:
        for wrapper in validated:
            if wrapper.judge is instance:
                return wrapper
        validated.append(ValidatedJudge(instance, num_tries=config.num_tries,
                                        max_retry_seconds=config.max_retry_seconds))
        return validated[-1]

    for given in (judge, agent):
        if isinstance(given, Judge):
            validated_for(given)

    refuse_unconfirmed(judge, agent)  # before the output is claimed, so a confirmed retry can claim it
    out = Path(out)
    record_dir = _claim(out)
    initial_rules = record_dir / INITIAL_RULES_FNAME
    shutil.copyfile(ruleset.path, initial_rules)
    Ruleset.load(initial_rules, expect_sha256=ruleset.sha256)
    seed_paths = {}
    for role, (_, data) in seed_files.items():
        seed_paths[role] = record_dir / SEED_FNAME.format(role=role)
        seed_paths[role].write_bytes(data)
    run_config = config.to_config(task=found.name, rules=initial_rules, seeds=seed_paths)
    record = {
        "task": found.identifier,
        "ruleset": {"name": ruleset.name, "sha256": ruleset.sha256, "source": str(ruleset.path)},
        "seeds": {role: {"source": str(path), "sha256": hashlib.sha256(data).hexdigest()}
                  for role, (path, data) in seed_files.items()},
        "judge": _judge_record(judge),
        "agent": "judge" if shared else _judge_record(agent),
        "num_tries": config.num_tries,
        "max_retry_seconds": config.max_retry_seconds,
        "config": OmegaConf.to_container(run_config),
    }
    (record_dir / RUN_RECORD_FNAME).write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n",
                                               encoding="utf-8")

    started: list[Judge] = []
    try:
        judge_instance = _started(judge, started)
        agent_instance = judge_instance if shared else _started(agent, started)
        task_spec = build_task_spec(cfg=run_config)
        trainer = RuleAmendmentTrainer(run_config, task=task_spec, judge=validated_for(judge_instance),
                                       agent=validated_for(agent_instance), rules=initial_rules, seeds=seed_paths,
                                       output_dir=out, logger=logger)
        trainer.train()
    finally:
        _close(started)
    ruleset_path = trainer.optimal_rule_set_path
    reward = json.loads((ruleset_path.parent / REWARD_OPTIMAL_FNAME).read_text(encoding="utf-8"))["reward"]
    return OptimizeRun(out=out, last_episode=trainer.episode_num, stopped_by=trainer.stopped_by,
                       ruleset_path=ruleset_path, reward=reward)


__all__ = ["INITIAL_RULES_FNAME", "OptimizeRun", "RUN_RECORD_DIR", "RUN_RECORD_FNAME", "SEED_FNAME", "optimize"]
