from __future__ import annotations

from typing import Any, Mapping, Protocol


class RunLogger(Protocol):
    """Takes one episode's metrics (`step` is the episode)."""

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None: ...


class NullLogger:
    """Logs nothing."""

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        return None


class WandbLogger:
    """Logs to the active W&B run, which the caller starts and ends."""

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        import wandb

        wandb.log(metrics, step=step)


__all__ = ["NullLogger", "RunLogger", "WandbLogger"]
