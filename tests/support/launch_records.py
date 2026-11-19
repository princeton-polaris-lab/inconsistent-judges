from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

#: The model configs the checkout commits, by name.
MODEL_CONFIGS = frozenset(path.stem for path in (Path(__file__).resolve().parents[2] / "configs" / "model").glob("*.yaml"))
#: How an override names a model config: ``model@<node>=<name>``.
MODEL_GROUP = "model@"


def model_config_now(recorded: str, site: Optional[str], configs: Iterable[str] = MODEL_CONFIGS) -> str:
    """A recorded launch's model-config name under today's: a committed config's own name, or the committed config it
    names with ``-<site>`` added, ``site`` being the partition the same record names; any other name is left as
    recorded."""
    names = frozenset(configs)
    if recorded in names or not site:
        return recorded
    base = recorded.removesuffix(f"-{site}")
    return base if base != recorded and base in names else recorded


def model_words_now(words: Iterable[str], site: Optional[str], configs: Iterable[str] = MODEL_CONFIGS) -> list[str]:
    """A recorded launch's arguments, each ``model@<node>=<name>`` naming its config by ``model_config_now``."""
    found = []
    for word in words:
        key, assigned, value = word.partition("=")
        found.append(f"{key}={model_config_now(value, site, configs)}"
                     if assigned and key.lstrip("+~").startswith(MODEL_GROUP) else word)
    return found


def launch_site(job: Mapping[str, Any]) -> Optional[str]:
    """The partition a launch record's job asked its scheduler for (``request.submitit.slurm_partition``), or None."""
    return ((job.get("request") or {}).get("submitit") or {}).get("slurm_partition")


def multirun_site(multirun: Mapping[str, Any]) -> Optional[str]:
    """The partition a recorded ``multirun.yaml``'s launcher asked for (``hydra.launcher.partition``), or None."""
    return ((multirun.get("hydra") or {}).get("launcher") or {}).get("partition")
