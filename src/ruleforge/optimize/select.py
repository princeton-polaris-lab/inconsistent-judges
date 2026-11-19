from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import yaml

from ruleforge.core.constants import OPTIMAL_RULE_SET_FNAME, REWARD_OPTIMAL_FNAME
from ruleforge.optimize.api import RUN_RECORD_DIR, RUN_RECORD_FNAME, OptimizeRun
from ruleforge.rulesets import Ruleset

logger = logging.getLogger(__name__)

_EPISODE_DIR_RE = re.compile(r"episode_(\d+)")
#: A seed-axis member's default name under `optimize --sweep`.
_SEED_MEMBER_RE = re.compile(r"(judge|agent)\.seed\.(\d+(?:_\d+)*)")
_HYDRA_MODEL_KEYS = {"judge": "judge_model", "agent": "agent_model"}
#: Directories a command writes beside a sweep's runs; passed over.
_RECORDS_SUFFIX = ".runs"
_WANDB_DIRNAME = "wandb"
RUN_NAMING = ("a sweep's run is a directory in it named by its job number (a `ruleforge jobs launch` sweep, a "
              "Hydra multirun), holding its run record (an `optimize --sweep` member: <key>.<value>, or as its "
              "--out template names it) or named as a seed axis's member is (judge.seed.<seed>, "
              "agent.seed.<seed>) and holding its episodes; an `optimize --sweep` over several keys nests one "
              "directory per key, in the order of its --sweep options, so give the directory that holds the seed's "
              "members (the seed's --sweep given last)")
NOT_A_RUN = "not a run"
_NO_SEED = 1 << 30


def extract_metrics(reward_path: Path) -> dict | None:
    """Return a flattened reward_optimal.jsonl record, or None when the file is truncated or corrupt."""
    try:
        with open(reward_path) as f:
            record = json.loads(f.readline())
        reward = record["reward"]
        metrics = {
            "effective_reward": reward["effective_reward"],
            "entropy_reward": reward["entropy_reward"],
            "regularization_penalty": reward["regularization_penalty"],
        }
        for name, entry in reward.get("penalty_breakdown", {}).items():
            metrics[f"{name}_penalty"] = entry["value"]
        return metrics
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def contiguous_episode_rewards(job_dir: Path) -> list[tuple[int, Path]]:
    """Return [(episode, reward path)] for the run's episodes in order, up to the first without a reward file."""
    episode_nums = sorted(
        int(match.group(1))
        for d in job_dir.glob("episode_*")
        if d.is_dir() and (match := _EPISODE_DIR_RE.fullmatch(d.name))
    )
    episodes = []
    for n in episode_nums:
        reward_path = job_dir / f"episode_{n}" / "rule_amendment" / REWARD_OPTIMAL_FNAME
        if not reward_path.is_file():
            break
        episodes.append((n, reward_path))
    return episodes


def _model_seed(models: Iterable[object]) -> Optional[int]:
    for entry in models:
        seed = entry.get("seed") if isinstance(entry, Mapping) else None
        if isinstance(seed, int):
            return seed
    return None


def _record_paths(run: Path) -> tuple[Path, Path]:
    return run / ".hydra" / "config.yaml", run / RUN_RECORD_DIR / RUN_RECORD_FNAME


def _recorded_seeds(run: Path) -> dict[str, Optional[int]]:
    """Return each role's seed: the run record's, written once when the run claims its directory, over Hydra's config,
    which a later start in that directory overwrites. A record and a config that disagree are refused."""
    config_path, record_path = _record_paths(run)
    seeds: dict[str, Optional[int]] = dict.fromkeys(_HYDRA_MODEL_KEYS)
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text()) or {}
        seeds = {role: _model_seed([config.get(key)]) for role, key in _HYDRA_MODEL_KEYS.items()}
    if not record_path.is_file():
        return seeds
    record = json.loads(record_path.read_text(encoding="utf-8"))
    for role in _HYDRA_MODEL_KEYS:
        # agents on the judge's engine run at the judge's seed, whatever their own config says
        shares_engine = record.get(role) == "judge"
        entry = record.get("judge" if shares_engine else role)
        kept = _model_seed([entry.get("options") if isinstance(entry, Mapping) else None])
        if kept is None:
            continue
        if not shares_engine and seeds[role] is not None and seeds[role] != kept:
            raise ValueError(f"{run}: its run record holds {role} seed {kept} and {config_path} holds {seeds[role]}: "
                             "another start overwrote the config in this directory, so its episodes cannot be told "
                             "apart; give each run a directory of its own")
        seeds[role] = kept
    return seeds


def run_seed(run: Path) -> Optional[int]:
    """Return the run's seed: its judge's, else its agent's, from its records, else from a seed-axis member name; None
    when none."""
    seeds = _recorded_seeds(run)
    member = _SEED_MEMBER_RE.fullmatch(run.name)
    if member is not None:
        seed = seeds[member.group(1)]
        return int(member.group(2)) if seed is None else seed
    return seeds["judge"] if seeds["judge"] is not None else seeds["agent"]


def _beside_runs(directory: Path) -> bool:
    if directory.name.startswith("."):
        return True
    written = directory.name.endswith(_RECORDS_SUFFIX) or directory.name == _WANDB_DIRNAME
    return written and not any(path.is_file() for path in _record_paths(directory))


def _no_run(job_dir: Path) -> Optional[str]:
    if job_dir.name.isdigit() or any(path.is_file() for path in _record_paths(job_dir)):
        return None
    if _SEED_MEMBER_RE.fullmatch(job_dir.name) is None:
        return f"{NOT_A_RUN}: no job number for its name, no run record in it"
    if any(path.is_dir() and _EPISODE_DIR_RE.fullmatch(path.name) for path in job_dir.iterdir()):
        return None
    return (f"{NOT_A_RUN}: named as a seed axis's member, with no run record or episode in it (a sweep over several "
            "keys holds its runs a directory further down)")


@dataclass(frozen=True)
class RunSummary:
    """One run as the selection reads it: its last complete episode, that episode's metrics, its peak episode and its
    seed."""

    run: Path
    last_episode: Optional[int] = None
    reward_path: Optional[Path] = None
    metrics: Mapping[str, float] = field(default_factory=dict, hash=False)
    peak_episode: Optional[int] = None
    peak_effective_reward: Optional[float] = None
    unreadable: Optional[Path] = None

    @property
    def ruleset_path(self) -> Optional[Path]:
        """Return the last complete episode's optimal ruleset path, or None."""
        return None if self.reward_path is None else self.reward_path.parent / OPTIMAL_RULE_SET_FNAME

    @cached_property
    def seed(self) -> Optional[int]:
        """Return the run's seed, read from its records on first access."""
        return run_seed(self.run)


def summarize(run: Union[Path, str, OptimizeRun]) -> RunSummary:
    """Return the run at `run` (a directory or an `OptimizeRun`'s output) as the selection reads it."""
    run = run.out if isinstance(run, OptimizeRun) else Path(run)
    complete = []
    unreadable = None
    for n, path in contiguous_episode_rewards(run):
        metrics = extract_metrics(path)
        if metrics is None:
            unreadable = path
            break
        complete.append((n, metrics, path))
    if not complete:
        return RunSummary(run=run, unreadable=unreadable)
    last_n, last_metrics, last_path = complete[-1]
    peak_n, peak_metrics, _ = max(complete, key=lambda e: e[1]["effective_reward"])
    return RunSummary(run=run, last_episode=last_n, reward_path=last_path, metrics=last_metrics,
                      peak_episode=peak_n, peak_effective_reward=peak_metrics["effective_reward"],
                      unreadable=unreadable)


def exclusion(summary: RunSummary, *, min_last_episode: int = 1) -> Optional[str]:
    """Return why the run cannot be selected, or None when it can."""
    if summary.last_episode is None:
        return "no complete episode"
    if summary.last_episode < min_last_episode:
        return f"last complete episode is {summary.last_episode} (< min_last_episode {min_last_episode})"
    if not summary.ruleset_path.is_file():
        return f"episode_{summary.last_episode} missing {OPTIMAL_RULE_SET_FNAME}"
    return None


def rank(summaries: Iterable[RunSummary]) -> list[RunSummary]:
    """Return the selectable runs best first; a tie goes to the lower seed, then to the order given."""
    return sorted(summaries, key=lambda s: (-s.metrics["effective_reward"],
                                            s.seed if s.seed is not None else _NO_SEED))


def select_best(runs: Iterable[Union[Path, str, OptimizeRun]], *, min_last_episode: int = 1) -> Ruleset:
    """Return the last optimal ruleset of the best of `runs`; refused when no run can be selected."""
    summaries = [summarize(run) for run in runs]
    selectable, excluded = [], []
    for summary in summaries:
        if summary.unreadable is not None:
            logger.warning("%s: unreadable (killed mid-write?); the run's complete episodes end before it",
                           summary.unreadable)
        reason = exclusion(summary, min_last_episode=min_last_episode)
        if reason is None:
            selectable.append(summary)
        else:
            logger.warning("skipping %s: %s", summary.run, reason)
            excluded.append(f"{summary.run}: {reason}")
    if not selectable:
        raise ValueError("no run can be selected" + (f" ({'; '.join(excluded)})" if excluded else ": none given"))
    return Ruleset.load(rank(selectable)[0].ruleset_path)


def sweep_table(sweep_dirs: Sequence[Union[Path, str]], *, min_last_episode: int = 1) -> tuple[list[dict], list[str]]:
    """Return the selection table of the sweeps' runs (best first) and the notes on every directory read."""
    selectable, notes = [], []
    for sweep_dir in map(Path, sweep_dirs):
        if not sweep_dir.is_dir():
            raise FileNotFoundError(f"sweep dir not found: {sweep_dir}")
        for job_dir in sorted(sweep_dir.iterdir()):
            # a directory this user may not read is noted, not fatal; a selectable run's torn record still refuses
            try:
                if not job_dir.is_dir() or _beside_runs(job_dir):
                    continue
                no_run = _no_run(job_dir)
                if no_run is not None:
                    notes.append(f"SKIP {job_dir}: {no_run}")
                    continue
                summary = summarize(job_dir)
                if summary.unreadable is not None:
                    notes.append(f"WARN {summary.unreadable}: unparseable (killed mid-write?), truncating here")
                reason = exclusion(summary, min_last_episode=min_last_episode)
                if reason is not None:
                    notes.append(f"SKIP {job_dir}: {reason}")
                    continue
                _ = summary.seed  # read here so rank() reads no file
            except OSError as error:
                notes.append(f"SKIP {job_dir}: cannot be read ({error.strerror})")
                continue
            selectable.append(summary)
    rows = [{
        "sweep_dir": str(summary.run.parent),
        "job_num": int(summary.run.name) if summary.run.name.isdigit() else None,
        "seed": summary.seed,
        "optimal_rule_set_path": str(summary.ruleset_path.resolve()),
        "last_episode": summary.last_episode,
        "peak_episode": summary.peak_episode,
        "peak_effective_reward": summary.peak_effective_reward,
        **summary.metrics,
    } for summary in rank(selectable)]
    return rows, notes


__all__ = ["NOT_A_RUN", "RUN_NAMING", "RunSummary", "contiguous_episode_rewards", "exclusion", "extract_metrics",
           "rank", "run_seed", "select_best", "summarize", "sweep_table"]
