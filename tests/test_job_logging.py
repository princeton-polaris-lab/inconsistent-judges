from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

from ruleforge.optimize import job_logging


@pytest.fixture
def root(monkeypatch):
    """The root logger with no handlers, its handlers and level put back after the test."""
    logger = logging.getLogger()
    handlers, level = logger.handlers, logger.level
    logger.handlers = []
    captured = []
    monkeypatch.setattr(logging, "captureWarnings", captured.append)
    yield SimpleNamespace(logger=logger, captured=captured)
    for handler in list(logger.handlers):
        handler.close()
    logger.handlers = handlers
    logger.setLevel(level)


def test_the_banner_logs_its_text_between_two_rules(caplog):
    with caplog.at_level(logging.INFO, logger=job_logging.__name__):
        job_logging.log_with_header("Episode 3")
    assert [record.getMessage() for record in caplog.records] == ["=" * 60, "Episode 3", "=" * 60]


def test_a_root_logger_with_a_console_and_a_file_handler_is_left_alone(root, tmp_path):
    console, file = logging.StreamHandler(sys.stderr), logging.FileHandler(tmp_path / "job.log")
    root.logger.handlers = [console, file]
    job_logging.ensure_job_logging(tmp_path)
    assert root.logger.handlers == [console, file] and root.captured == []


def _manual_setup(root, tmp_path) -> None:
    """The job's own handlers: everything to ``trainer.log`` under the job's directory, INFO and up to stderr."""
    file, console = root.logger.handlers[-2:]
    assert isinstance(file, logging.FileHandler) and file.baseFilename == str(tmp_path / "trainer.log")
    assert (file.level, console.level, console.stream, root.logger.level) == (
        logging.DEBUG, logging.INFO, sys.stderr, logging.DEBUG)
    assert file.formatter._fmt == console.formatter._fmt == "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    assert root.captured == [True]


def test_a_root_logger_left_with_a_file_handler_alone_gets_a_console_back(root, tmp_path):
    """A FileHandler is a StreamHandler, but it writes no console."""
    root.logger.handlers = [logging.FileHandler(tmp_path / "job.log")]
    job_logging.ensure_job_logging(tmp_path)
    assert len(root.logger.handlers) == 2
    _manual_setup(root, tmp_path)


def test_the_handlers_a_model_stack_left_are_closed_and_replaced_by_the_job_s_own(root, tmp_path):
    class Unclosable(logging.Handler):
        def close(self):
            raise OSError("already closed")

    left = [logging.NullHandler(), Unclosable()]
    root.logger.handlers = list(left)
    job_logging.ensure_job_logging(tmp_path)
    assert len(root.logger.handlers) == 2 and not set(left) & set(root.logger.handlers)
    _manual_setup(root, tmp_path)


def test_a_handler_list_that_cannot_be_emptied_is_kept_and_the_jobs_handlers_join_it(root, tmp_path):
    class Unpoppable(list):
        def pop(self, *args):
            raise RuntimeError("held by another thread")

    kept = logging.NullHandler()
    root.logger.handlers = Unpoppable([kept])
    job_logging.ensure_job_logging(tmp_path)
    assert root.logger.handlers[0] is kept and len(root.logger.handlers) == 3
    _manual_setup(root, tmp_path)
