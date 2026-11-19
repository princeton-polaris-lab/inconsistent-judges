from __future__ import annotations

import datetime
import re
import secrets
import shlex
from pathlib import Path, PurePosixPath
from typing import Optional, Sequence

from ruleforge.core.errors import RuleForgeError
from ruleforge.core.paths import ARTIFACT_ROOT_ENV, ARTIFACT_SCHEME, is_artifact_uri, to_uri
from ruleforge.env import PREFIX


class LaunchError(ValueError, RuleForgeError):
    """A launch that cannot be built or run as asked."""


MODEL_ROOT_ENV = "RULEFORGE_MODEL_ROOT"
#: The Hydra launcher config whose `setup` lines are every job's environment.
LAUNCHER_CONFIG = "configs/hydra/launcher/submitit_slurm_gpu.yaml"
#: Names the launcher owns: a job may not export them.
RESERVED_ENV = (MODEL_ROOT_ENV, ARTIFACT_ROOT_ENV, f"{PREFIX}MODEL_ROOT", f"{PREFIX}ARTIFACT_ROOT", "VIRTUAL_ENV",
                "PYTHONPATH", "PYTHONHOME")
RESERVED_ENV_PREFIXES = ("SLURM_",)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: A name with one of these words looks like a credential; exports are recorded, so it is refused.
SECRET_NAME_PARTS = frozenset({"KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "CREDENTIALS"})


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def new_run_id(commit: str, *, now: Optional[datetime.datetime] = None) -> str:
    moment = now or datetime.datetime.now(datetime.timezone.utc)
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{commit[:8]}-{secrets.token_hex(2)}"


def launcher_setup(repo: Path) -> list[str]:
    """Return the SLURM launcher config's `setup` lines, resolved as the launcher resolves them."""
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(repo) / LAUNCHER_CONFIG)
    return [str(line) for line in OmegaConf.to_container(config.setup, resolve=True)]


def check_env_name(name: str, *, flag: str, reserved: Sequence[str] = ()) -> None:
    """Refuse an export of a malformed, launcher-owned or credential-shaped name."""
    if not _ENV_NAME.match(name):
        raise LaunchError(f"{flag} {name!r}: not an environment variable name")
    if name in RESERVED_ENV or name in reserved or name.startswith(RESERVED_ENV_PREFIXES):
        raise LaunchError(f"{flag} {name!r}: the launcher owns that variable")
    if SECRET_NAME_PARTS & set(name.upper().split("_")):
        raise LaunchError(
            f"{flag} {name!r} looks like a credential, and a launch records its environment in its record; "
            "credentials are not passed to jobs"
        )


def env_pairs(env: Sequence[str]) -> list[tuple[str, str]]:
    """Return `NAME=VALUE` items as pairs; a bad name or a name given twice is refused."""
    pairs = []
    for item in env:
        name, separator, value = item.partition("=")
        if not separator:
            raise LaunchError(f"--env {item!r}: expected NAME=VALUE")
        check_env_name(name, flag="--env")
        pairs.append((name, value))
    names = [name for name, _ in pairs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise LaunchError(f"--env names {', '.join(duplicates)} more than once")
    return pairs


def export_lines(env: Sequence[str]) -> list[str]:
    """Return `env` as shell export lines, each value quoted literally."""
    return [f"export {name}={shlex.quote(value)}" for name, value in env_pairs(env)]


def env_script(setup: Sequence[str], env: Sequence[str] = ()) -> str:
    """Return the shell prelude a job's command is exec'd from: the setup lines, then the exports; a failing setup line
    stops the job."""
    return "\n".join(["set -e", *setup, *export_lines(env), 'exec "$@"'])


def artifact_relative(value: str, env: Optional[dict[str, str]] = None) -> str:
    """Return a path inside the artifact tree as the relative POSIX path below the artifact root; `..`, an empty path and
    the root itself are refused."""
    text = str(value).strip()
    if is_artifact_uri(text):
        text = text[len(ARTIFACT_SCHEME):]
    elif text.startswith("/") or text.startswith("~"):
        text = to_uri(text, env)[len(ARTIFACT_SCHEME):]
    return relative_subpath(text, what="artifact path")


def relative_subpath(value: str, *, what: str) -> str:
    """Return `value` as a normalised relative POSIX path below a root."""
    path = PurePosixPath(value)
    parts = [part for part in path.parts if part not in ("", ".")]
    if path.is_absolute() or not parts or ".." in parts:
        raise ValueError(f"{what} {value!r} must name a path below the root (relative, no '..')")
    return "/".join(parts)


def run_workdir(value: str) -> str:
    """Return `--workdir` as a path below the repo (`.` for the repo itself); `ValueError` above it."""
    return "." if value in ("", ".") else relative_subpath(value, what="--workdir")
