from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

from ruleforge.core.errors import ArtifactRootError, RuleForgeError

PREFIX = "RULEFORGE_"


class EnvUnsetError(RuleForgeError):
    """Raised when a config names a variable (``${rf.env:NAME}``, no default) that is unset."""


class PackageResourceError(RuleForgeError):
    """Raised for a packaged file (``${rf.pkg:PATH}``) that neither the package nor its checkout holds."""


def env_value(name: str, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Return the value of ``RULEFORGE_<name>``, or None when it is unset."""
    return (os.environ if env is None else env).get(PREFIX + name) or None


def artifact_root(env: Optional[Mapping[str, str]] = None) -> Path:
    """Return the artifact root (the raw run trees); there is no default, so unset raises."""
    value = env_value("ARTIFACT_ROOT", env)
    if value is None:
        raise ArtifactRootError(f"{PREFIX}ARTIFACT_ROOT is unset: point it at the directory `ruleforge data fetch` "
                                "fills; rendering from results/ needs none")
    root = Path(value).expanduser()
    if not root.is_dir():
        raise ArtifactRootError(
            f"{PREFIX}ARTIFACT_ROOT={value!r} is not a directory. On a cluster "
            "this usually means the filesystem is not mounted on this node."
        )
    return root


def model_root(env: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """Return the weights root as given, or None when unset."""
    value = env_value("MODEL_ROOT", env)
    return None if value is None else Path(value)


def batch_root(env: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """Return the paid batches' state directory as given, or None when unset."""
    value = env_value("BATCH_ROOT", env)
    return None if value is None else Path(value)


PACKAGE_DIR = Path(__file__).resolve().parent
ENV_RESOLVER = "rf.env"
PACKAGE_RESOLVER = "rf.pkg"
RECORDED_RESOLVER = "rf.recorded"


def package_path(relative: str) -> Path:
    """Return the packaged file ``relative`` names, from the package or, in a checkout, the checkout's top."""
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise PackageResourceError(f"{relative!r} is not a path below the package")
    packaged = PACKAGE_DIR / relative
    if packaged.exists():
        return packaged
    from ruleforge.core.paths import repository_root

    try:
        checkout = repository_root() / relative
    except ArtifactRootError:
        checkout = None
    if checkout is not None and checkout.exists():
        return checkout
    raise PackageResourceError(f"no packaged {relative} (looked in {packaged}"
                               f"{'' if checkout is None else f' and {checkout}'})")


_NO_DEFAULT = object()


def _env_resolver(name: str, default: Any = _NO_DEFAULT) -> Any:
    value = env_value(str(name))
    if value is not None:
        return value
    if default is _NO_DEFAULT:
        raise EnvUnsetError(f"{PREFIX}{name} is unset: a config names it (${{{ENV_RESOLVER}:{name}}}); set it, or "
                            "override the key that reads it")
    return default


def _package_resolver(relative: str) -> str:
    return str(package_path(str(relative)))


def _recorded_resolver(block: str) -> str:
    from ruleforge.core.record_names import recorded_block_key

    return recorded_block_key(str(block))


def register_resolvers() -> None:
    """Register the ``rf.env``, ``rf.pkg`` and ``rf.recorded`` resolvers with OmegaConf, uncached, once per process."""
    from omegaconf import OmegaConf

    for name, resolver in ((ENV_RESOLVER, _env_resolver), (PACKAGE_RESOLVER, _package_resolver),
                           (RECORDED_RESOLVER, _recorded_resolver)):
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver, use_cache=False)
