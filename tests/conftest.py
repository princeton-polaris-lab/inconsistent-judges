from __future__ import annotations

import os

# Set before collection imports anything: matplotlib, huggingface_hub and torch read these at import time.
os.environ.update(
    {
        "MPLBACKEND": "Agg",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "OMP_NUM_THREADS": "1",
        **({} if os.environ.get("SUITE_GPUS") == "1" else {"CUDA_VISIBLE_DEVICES": ""}),
    }
)

import socket
from pathlib import Path

import pytest

# peft and transformers import each other; importing them in this order before collection avoids a circular import.
try:  # pragma: no cover - environment-dependent import ordering fix
    import peft  # noqa: F401
    from transformers import trainer_utils  # noqa: F401
except Exception:
    pass

from tests.support import slurm_guard


REPO_ROOT = Path(__file__).resolve().parents[1]

requires_git_worktree = pytest.mark.skipif(
    not (REPO_ROOT / ".git").exists(), reason="needs a git checkout (an unpacked release tree has no .git)"
)


TIER_R_STRICT = "--tier-r-strict"
TIER_R_SKIPS = REPO_ROOT / "tests" / "tier_r_skips.txt"
_TIER_R_LISTED = pytest.StashKey[dict]()


def read_tier_r_skips(path: Path = TIER_R_SKIPS) -> dict[str, str]:
    """The skips file's ``{node id: reason}``; :class:`ValueError` naming the line when it is not in its format."""
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ValueError("it does not end with a newline")
    rows = []
    for number, line in enumerate(text.split("\n")[:-1], 1):
        nodeid, tab, reason = line.partition("\t")
        if not (nodeid.strip() and tab and reason.strip()) or "\t" in reason:
            raise ValueError(f"line {number} is not nodeid<TAB>reason: {line!r}")
        rows.append((nodeid, reason))
    ids = [nodeid for nodeid, _ in rows]
    if ids != sorted(set(ids)):
        raise ValueError("its node ids are not sorted, each once")
    return dict(rows)


def tier_r_refusal(report, *, cluster: bool, listed: dict[str, str]) -> str | None:
    """Why a strict run fails ``report``."""
    if not report.skipped or hasattr(report, "wasxfail") or report.nodeid in listed:
        return None
    if isinstance(report, pytest.TestReport) and not cluster:
        return None
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    return (f"{TIER_R_STRICT}: {report.nodeid} was skipped ({reason}), and {TIER_R_SKIPS.name} does not list it: "
            "a skip a strict Tier R run accepts is listed with its reason")


def _strictly(config, report, *, cluster: bool) -> None:
    listed = config.stash.get(_TIER_R_LISTED, None)
    refusal = None if listed is None else tier_r_refusal(report, cluster=cluster, listed=listed)
    if refusal is not None:
        report.outcome, report.longrepr = "failed", refusal


def pytest_addoption(parser):
    parser.addoption(TIER_R_STRICT, action="store_true", default=False,
                     help="fail a skipped cluster (Tier R) test, and a module skipped as it is collected, unless "
                          "tests/tier_r_skips.txt lists it with its reason; implies --continue-on-collection-errors")


def pytest_configure(config):
    config.addinivalue_line("markers", "allow_slurm_executor: builds submitit's SLURM executor, to render what it "
                                       "would submit (tests/support/slurm_guard.py still refuses every submission)")
    if config.getoption(TIER_R_STRICT):
        try:
            config.stash[_TIER_R_LISTED] = read_tier_r_skips()
        except (OSError, ValueError) as error:
            raise pytest.UsageError(f"{TIER_R_STRICT}: {TIER_R_SKIPS}: {error}") from error
        config.option.continue_on_collection_errors = True


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    _strictly(item.config, outcome.get_result(), cluster=item.get_closest_marker("cluster") is not None)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    _strictly(collector.config, outcome.get_result(), cluster=True)


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config):
    """``-n auto``: one worker per CPU this process may run on."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        int(os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS", ""))
    except ValueError:
        return len(os.sched_getaffinity(0))
    return None


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


class NetworkAccessError(RuntimeError):
    """Raised when a test reaches for the network."""


def _is_local(address: object) -> bool:
    """True for loopback and non-IP (e.g. AF_UNIX) destinations."""
    if not isinstance(address, tuple) or not address:
        return True
    return address[0] in {"127.0.0.1", "::1", "localhost", "", None}


# Captured before patching so the opt-out can restore them.
_REAL_SOCKET_CALLS = {
    "connect": socket.socket.connect,
    "connect_ex": socket.socket.connect_ex,
    "create_connection": socket.create_connection,
}


def _guard(name, real):
    def guarded(*args, **kwargs):
        # socket.socket methods take self first; create_connection does not.
        address = args[1] if name != "create_connection" else args[0]
        if _is_local(address):
            return real(*args, **kwargs)
        raise NetworkAccessError(
            f"blocked outbound connection to {address!r}; mock the client or "
            "mark the test with @pytest.mark.allow_network"
        )

    return guarded


socket.socket.connect = _guard("connect", _REAL_SOCKET_CALLS["connect"])
socket.socket.connect_ex = _guard("connect_ex", _REAL_SOCKET_CALLS["connect_ex"])
socket.create_connection = _guard(
    "create_connection", _REAL_SOCKET_CALLS["create_connection"]
)


@pytest.fixture(autouse=True)
def _network_policy(request, monkeypatch):
    """Restore real sockets for tests marked ``allow_network``."""
    if request.node.get_closest_marker("allow_network"):
        monkeypatch.setattr(socket.socket, "connect", _REAL_SOCKET_CALLS["connect"])
        monkeypatch.setattr(
            socket.socket, "connect_ex", _REAL_SOCKET_CALLS["connect_ex"]
        )
        monkeypatch.setattr(
            socket, "create_connection", _REAL_SOCKET_CALLS["create_connection"]
        )


slurm_guard.install()


@pytest.fixture(autouse=True)
def _slurm_policy(request):
    """Fail a test that reached SLURM."""
    since = slurm_guard.mark()
    slurm_guard.POLICY["executor"] = request.node.get_closest_marker("allow_slurm_executor") is not None
    try:
        yield
    finally:
        slurm_guard.POLICY["executor"] = False
    reached = slurm_guard.reached_since(since)
    if reached:
        pytest.fail("this test reached SLURM, which the suite never does (tests/support/slurm_guard.py): "
                    + "; ".join(reached), pytrace=False)


@pytest.fixture
def stub_backend():
    """Build a ModelBackend with __init__ bypassed and every attribute set."""
    from ruleforge.judges.backend import ModelBackend
    from ruleforge.judges.base import Capabilities

    def _build(batch_generate_fn=None, **overrides):
        backend = object.__new__(ModelBackend)
        defaults = {
            "capabilities": Capabilities(seed="unsupported", per_request_errors=True),
            "model_type": "test",
            "model_name": "test-model",
            "max_new_tokens": 1024,
            "seed": 42,
            "temperature": 0.0,
            "num_tries": 3,
            "fixed_temperature": None,
            "max_retry_seconds": None,
            "_base_delay": 0.0,
            "seed_capability": "unsupported",
            "_generate_request_block": None,
        }
        for name, value in {**defaults, **overrides}.items():
            setattr(backend, name, value)
        if batch_generate_fn is not None:
            backend._batch_generate = batch_generate_fn
        return backend

    return _build


@pytest.fixture
def task_cfg():
    """Minimal but complete config for constructing a TaskSpec."""

    def _build(task_spec="modelspec", **eval_overrides):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "task_spec": task_spec,
                "rule_amendment": {
                    "initial_rule_set_path": (
                        "tests/fixtures/rules/openai_subset_prolific.jsonl"
                    ),
                    "intent_weight": 0.0,
                },
                "scenario_generation": {
                    "initial_scenario_seed_type": "none",
                    "good_seed_scenario_path": "unused.jsonl",
                    "bad_seed_scenario_path": "unused.jsonl",
                    "seed_scenario_path": "unused.jsonl",
                    "num_scenarios_per_batch": 3,
                },
                "agent_model": {},
                "judge_model": {},
                "eval_cfg": {
                    "judge_types": ["No_Interpretation"],
                    "evaluation_temperature": 0.1,
                    "save_batch_size": 10,
                    **eval_overrides,
                },
            }
        )

    return _build


@pytest.fixture
def modelspec(task_cfg):
    """Factory for a ModelSpecTaskSpec with eval-config overrides."""
    from ruleforge.tasks.spec import ModelSpecTaskSpec

    def _build(**eval_overrides):
        return ModelSpecTaskSpec(task_cfg(**eval_overrides))

    return _build
