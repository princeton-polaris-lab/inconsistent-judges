from __future__ import annotations

import contextlib
import logging
import runpy
from collections.abc import Iterator

from hydra.core.global_hydra import GlobalHydra


def _loggers() -> dict[str, logging.Logger]:
    """The root logger (as ``""``) and every named logger this process has made so far."""
    named = {name: logger for name, logger in logging.root.manager.loggerDict.items()
             if isinstance(logger, logging.Logger)}
    return {"": logging.root, **named}


@contextlib.contextmanager
def kept_logging() -> Iterator[None]:
    """Put the process's logging back after the block (Hydra's ``main`` rebinds handlers to a stderr pytest closes)."""
    before = {name: (list(logger.handlers), logger.level, logger.disabled, logger.propagate)
              for name, logger in _loggers().items()}
    kept = {handler for handlers, *_ in before.values() for handler in handlers}
    try:
        yield
    finally:
        added = set()
        for name, logger in _loggers().items():
            handlers, level, disabled, propagate = before.get(name, ([], logging.NOTSET, False, True))
            added.update(handler for handler in logger.handlers if handler not in kept)
            logger.handlers = handlers
            logger.setLevel(level)
            logger.disabled, logger.propagate = disabled, propagate
        for handler in added:
            handler.close()


@contextlib.contextmanager
def through_hydra() -> Iterator[None]:
    """Around a program run through Hydra's ``main``: Hydra's state cleared before and after, its logging put back."""
    GlobalHydra.instance().clear()
    try:
        with kept_logging():
            yield
    finally:
        GlobalHydra.instance().clear()


def run_program(module: str) -> None:
    """``python -m <module>`` with the ``sys.argv`` already set, through :func:`through_hydra`."""
    with through_hydra():
        runpy.run_module(module, run_name="__main__")
