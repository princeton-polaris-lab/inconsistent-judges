from __future__ import annotations

import atexit
import functools
import inspect
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

COMMANDS = ("sbatch", "srun", "salloc", "scancel", "scontrol")
STUB_EXIT = 97
#: What a test may do that the guard otherwise refuses: build submitit's SLURM executor (``allow_slurm_executor``).
POLICY = {"executor": False}


class SlurmReached(RuntimeError):
    """A test reached SLURM: a real SLURM command, or submitit's SLURM executor."""


_REACHED: list[str] = []
_STATE: dict[str, Any] = {}


def install() -> None:
    """Install the three layers in this process (once)."""
    if _STATE:
        return
    stubs = Path(tempfile.mkdtemp(prefix="suite-slurm-stubs-"))
    atexit.register(shutil.rmtree, stubs, True)
    calls = stubs / "calls.tsv"
    calls.touch()
    for name in COMMANDS:
        stub = stubs / name
        stub.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\t%s\\n' \"${{PYTEST_CURRENT_TEST:-}}\" \"{name} $*\" >> {shlex.quote(str(calls))}\n"
            f"echo '{name}: the suite never reaches SLURM (tests/support/slurm_guard.py); stub it in the test' >&2\n"
            f"exit {STUB_EXIT}\n", encoding="utf-8")
        stub.chmod(0o755)
    _STATE.update(stubs=str(stubs), calls=calls, real=_real_commands(os.environ.get("PATH", "")))
    os.environ["PATH"] = _path_with_stubs(os.environ.get("PATH", os.defpath))
    _guard_popen()
    _guard_submitit()


def stubs() -> str:
    return _STATE["stubs"]


def real_commands() -> frozenset[str]:
    """The cluster's own SLURM commands the guard refuses (their real paths): none where SLURM is not installed."""
    return _STATE["real"]


def mark() -> tuple[int, int]:
    """Where the record of reached SLURM stands, for :func:`reached_since`."""
    return len(_REACHED), _STATE["calls"].stat().st_size


def reached_since(since: tuple[int, int], *, forget: bool = False) -> list[str]:
    """Every refusal and stub call since ``since``, each a line naming the test and the command."""
    refused, size = since
    with _STATE["calls"].open("rb") as record:
        record.seek(size)
        calls = record.read().decode("utf-8", "replace").splitlines()
    found = [*_REACHED[refused:], *(f"stub: {line}" for line in calls)]
    if forget:
        del _REACHED[refused:]
        os.truncate(_STATE["calls"], size)
    return found


def _refuse(what: str) -> None:
    _REACHED.append(f"{os.environ.get('PYTEST_CURRENT_TEST', '(no test)')}: {what}")
    raise SlurmReached(f"{what}: the suite never reaches SLURM (tests/support/slurm_guard.py); stub it in the test")


def _real_commands(path: str) -> frozenset[str]:
    found = set()
    for directory in [*path.split(os.pathsep), *os.defpath.split(os.pathsep)]:
        for name in COMMANDS:
            candidate = os.path.join(directory or ".", name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                found.add(os.path.realpath(candidate))
    return frozenset(found)


@functools.lru_cache(maxsize=None)
def _holds_real(directory: str) -> bool:
    return any(os.path.realpath(os.path.join(directory or ".", name)) in real_commands() for name in COMMANDS)


def _path_with_stubs(path: str) -> str:
    """``path`` with the stubs' directory just before its first directory holding a real SLURM command."""
    entries = path.split(os.pathsep)
    for index, directory in enumerate(entries):
        if directory == stubs():
            return path
        if _holds_real(directory):
            return os.pathsep.join([*entries[:index], stubs(), *entries[index:]])
    return path


def _with_stubs(env: Mapping[Any, Any]) -> tuple[Mapping[Any, Any], str]:
    """A child's environment ``env`` with the stubs on its ``PATH``, and the search path the child will use."""
    binary = any(isinstance(key, bytes) for key in env)
    key = b"PATH" if binary else "PATH"
    given = env.get(key)
    path = os.defpath if given is None else os.fsdecode(given)
    stubbed = _path_with_stubs(path)
    if stubbed == path:
        return env, path
    return {**env, key: os.fsencode(stubbed) if binary else stubbed}, stubbed


def _resolve(name: str, path: str, cwd: Optional[str]) -> Optional[str]:
    """The real path of the program ``name`` a child would run, or None when there is none."""
    if os.sep not in name:
        found = shutil.which(name, path=path)
        return None if found is None else os.path.realpath(found)
    return os.path.realpath(os.path.join(cwd or os.getcwd(), name))


def _slurm_command(call: Mapping[str, Any], path: str) -> Optional[str]:
    """The SLURM command a ``Popen`` call would run, or None."""
    args, cwd = call["args"], call.get("cwd")
    cwd = None if cwd is None else os.fsdecode(cwd)
    if call.get("shell"):
        line = os.fsdecode(args if isinstance(args, (str, bytes, os.PathLike)) else args[0])
        names = [word for word in re.findall(r"[^\s;&|()<>`'\"$]+", line) if os.path.basename(word) in COMMANDS]
    else:
        program = call.get("executable")
        if program is None:
            program = args if isinstance(args, (str, bytes, os.PathLike)) else args[0]
        names = [os.fsdecode(program)]
    return next((name for name in names if _resolve(name, path, cwd) in real_commands()), None)


def _guard_popen() -> None:
    real_init = subprocess.Popen.__init__
    signature = inspect.signature(real_init)

    @functools.wraps(real_init)
    def guarded(self: subprocess.Popen, *args: Any, **kwargs: Any) -> None:
        call = signature.bind(self, *args, **kwargs)
        env = call.arguments.get("env")
        stubbed, path = _with_stubs(os.environ if env is None else env)
        if env is not None or stubbed is not os.environ:  # inherited: given only when a test set PATH without the stubs
            call.arguments["env"] = stubbed
        command = _slurm_command(call.arguments, path)
        if command is not None:
            _refuse(f"a child process running {command}")
        real_init(*call.args, **call.kwargs)

    subprocess.Popen.__init__ = guarded


def _guard_submitit() -> None:
    try:
        from submitit.slurm import slurm
    except ImportError:  # pragma: no cover - the slurm extra is part of the development environment
        return
    real_init = slurm.SlurmExecutor.__init__

    @functools.wraps(real_init)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> None:
        if not POLICY["executor"]:
            _refuse("submitit's SLURM executor (mark the test allow_slurm_executor if it only renders one)")
        real_init(self, *args, **kwargs)

    slurm.SlurmExecutor.__init__ = guarded
