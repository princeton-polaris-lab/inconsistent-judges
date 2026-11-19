from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ruleforge.core.credentials import CREDENTIAL_REFUSAL, carries_credential
from ruleforge.core.errors import RuleForgeError
from ruleforge.core.digests import canonical_json

if TYPE_CHECKING:
    from ruleforge.judges.weights import AdapterIdentity
    from ruleforge.judges.backend import GenerationOutcome
    from ruleforge.judges.token_usage import TokenUsageTracker

_CHECKOUT_MODEL_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "model"
_PACKAGED_MODEL_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "model"
# Where a model config is found by name: a checkout's configs/model/ first, then the package's own.
MODEL_CONFIG_DIRS = (*((_CHECKOUT_MODEL_CONFIGS,) if _CHECKOUT_MODEL_CONFIGS.is_dir() else ()),
                     _PACKAGED_MODEL_CONFIGS)
MODEL_CONFIG_DIR = MODEL_CONFIG_DIRS[0]
# How a config found by name has its roots rewritten, so it reads them by the library's environment rule.
PACKAGED_ROOTS = MappingProxyType({"${oc.env:RULEFORGE_MODEL_ROOT}": "${rf.env:MODEL_ROOT}",
                                   "${oc.env:RULEFORGE_BATCH_ROOT}": "${rf.env:BATCH_ROOT,null}"})
_EMBEDDED_ROOTS = MappingProxyType({"${oc.env:RULEFORGE_MODEL_ROOT}": "${rf.env:MODEL_ROOT}",
                                    "${oc.env:RULEFORGE_BATCH_ROOT}": "${rf.env:BATCH_ROOT}"})


# The judge names ``ruleforge.judges.resolve`` reads beside a model config's name or YAML file, with what each names.
SHORTHANDS = MappingProxyType({
    "fake[:SEED]": "the free deterministic stand-in, to check a pipeline at no cost",
    "openai:MODEL": "OpenAI's Responses API, keyed from OPENAI_API_KEY",
    "anthropic:MODEL": "Anthropic's Messages API, keyed from ANTHROPIC_API_KEY",
    "local:PATH": "a vLLM engine in this process, over the weights at PATH",
    "openai_compatible:MODEL@URL": "any server speaking OpenAI's Chat Completions at base URL URL, keyed from "
                                   "OPENAI_API_KEY",
})


def judge_forms() -> str:
    """Return every way a judge is named, as one phrase."""
    return f"a model config's name (ruleforge judges list), a YAML file, or {', '.join(SHORTHANDS)}"


URL_OPTIONS = ("base_url", "batch_endpoint")


def refuse_credential(value: Any) -> None:
    """Refuse a judge name or URL that is a string carrying a credential, without repeating it."""
    if isinstance(value, str) and carries_credential(value):
        raise JudgeError(CREDENTIAL_REFUSAL)


def model_config_names() -> list[str]:
    """Return every name a model config can be found by, sorted."""
    return sorted({path.stem for directory in MODEL_CONFIG_DIRS for path in directory.glob("*.yaml")})


def model_config_path(name: str) -> Optional[Path]:
    """Return the model config ``<name>.yaml`` in the first of ``MODEL_CONFIG_DIRS`` holding one, or None."""
    for directory in MODEL_CONFIG_DIRS:
        path = directory / f"{name}.yaml"
        if path.is_file():
            return path
    return None


def _renamed(value: Any) -> Any:
    if isinstance(value, str):
        if value in PACKAGED_ROOTS:
            return PACKAGED_ROOTS[value]
        for written, embedded in _EMBEDDED_ROOTS.items():
            value = value.replace(written, embedded)
        return value
    if isinstance(value, dict):
        return {key: _renamed(item) for key, item in value.items()}
    return [_renamed(item) for item in value] if isinstance(value, list) else value


def as_packaged(node: DictConfig) -> DictConfig:
    """Return a model config with its ``${oc.env:RULEFORGE_*}`` roots rewritten to the library's ``${rf.env:*}`` rule."""
    return OmegaConf.create(_renamed(OmegaConf.to_container(node, resolve=False)))


class JudgeError(ValueError, RuleForgeError):
    """Raised for a judge configuration, provider or binding that is refused."""


@dataclass(frozen=True)
class Capabilities:
    """What a judge can do, read from its configuration alone."""

    seed: str  # what a generation seed does (SEED_CAPABILITIES)
    batch: bool = False  # a paid-batch transport addressed by request block
    windowed: bool = False  # it can keep several blocks in flight
    per_request_errors: bool = False  # a transport failure reaches GenerationOutcome.error, never an empty answer
    sampler_truncation: bool = False  # top-p / top-k / min-p reach its sampler
    local_engine: bool = False
    spends_money: bool = False
    fixed_temperature: Optional[float] = None


@dataclass(frozen=True)
class JudgeIdentity:
    """What a condition records of its judge: ``extension`` a registered judge's behaviour-changing options as canonical
    JSON (None when built in), ``truncation_source`` a local judge's ``(model path, default_truncation)``, never
    written."""

    model_name: str
    model_type: str
    seed_capability: str
    max_new_tokens: int
    endpoint: Optional[str] = None
    reasoning_effort: Optional[str] = None
    send_max_output_tokens: Optional[bool] = None
    extension: Optional[str] = None
    model_path: Optional[str] = None
    adapter: Optional["AdapterIdentity"] = None
    model_weights_sha256: Optional[str] = None
    serving_dtype: Optional[str] = None
    enable_thinking: Optional[bool] = None
    truncation_source: Optional[tuple[str, Optional[Mapping[str, Any]]]] = None


def extension(provider: str, version: str, options: Mapping[str, Any]) -> str:
    """Return a registered judge's ``extension``: its provider, version and behaviour-changing options as canonical
    JSON."""
    return canonical_json({"provider": str(provider), "version": str(version), "options": dict(options)})


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, (DictConfig, ListConfig)) else value


def _detached(config: Any) -> DictConfig:
    # A key that cannot be resolved yet (an unset variable) is kept as written and resolved when read.
    if isinstance(config, JudgeOptions):
        return config._config
    node = config if isinstance(config, DictConfig) else OmegaConf.create(dict(config))
    raw = OmegaConf.to_container(node, resolve=False)
    detached: dict[str, Any] = {}
    for key, written in raw.items():
        try:
            detached[key] = _plain(node[key])
        except OmegaConfBaseException:
            detached[key] = written
    result = OmegaConf.create(detached)
    OmegaConf.set_readonly(result, True)
    return result


class JudgeOptions(Mapping):
    """A judge's options as configured, read-only (an absent key raises on item access, reads None as an attribute);
    every judge config becomes one, so a URL option carrying a credential is refused here."""

    __slots__ = ("_config",)

    def __init__(self, config: Any) -> None:
        object.__setattr__(self, "_config", _detached(config))
        written = self.written()
        for key in URL_OPTIONS:
            refuse_credential(written.get(key))

    def __getitem__(self, key: str) -> Any:
        if key not in self._config:
            raise KeyError(key)
        return _plain(self._config[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._config)

    def __len__(self) -> int:
        return len(self._config)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name] if name in self._config else None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("JudgeOptions is read-only; JudgeSpec.with_options makes a changed copy")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JudgeOptions):
            return NotImplemented
        return self.written() == other.written()

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._config.keys(), key=str)))

    def __repr__(self) -> str:
        return f"JudgeOptions({self.written()!r})"

    def written(self) -> dict[str, Any]:
        """Return the options as written, interpolations unresolved."""
        return OmegaConf.to_container(self._config, resolve=False)

    def config(self) -> DictConfig:
        """Return a mutable DictConfig of the options."""
        return OmegaConf.create(self.written())

    def replace(self, **changes: Any) -> "JudgeOptions":
        """Return a copy with ``changes`` set (None sets a key to null)."""
        return JudgeOptions({**self.written(), **{key: _plain(value) for key, value in changes.items()}})


def _load(config: Any) -> Any:
    if isinstance(config, (DictConfig, Mapping)):
        return config
    if not isinstance(config, (str, Path)):
        raise JudgeError(f"a judge is configured by a model config name, a YAML path or a mapping, got {config!r}")
    if isinstance(config, str):
        refuse_credential(config)
    path = Path(config)
    if path.suffix in (".yaml", ".yml") or path.is_file():
        if not path.is_file():
            raise JudgeError(f"no model config at {path}")
        return as_packaged(OmegaConf.load(path))
    named = model_config_path(str(config))
    if named is None:
        raise JudgeError(f"no model config named {config!r}; there are {model_config_names()}. A judge is "
                         f"{judge_forms()}")
    return as_packaged(OmegaConf.load(named))


def _checked_options(options: Mapping[str, Any], allowed: Sequence[str], constructor: str) -> dict[str, Any]:
    unknown = sorted(set(options) - set(allowed))
    if unknown:
        raise TypeError(f"JudgeSpec.{constructor}() takes no option(s) {unknown}; it takes {sorted(allowed)}")
    return {key: str(value) if isinstance(value, Path) else value for key, value in options.items()}


ANTHROPIC_DEFAULT_MAX_TOKENS = 16384
_COMMON_OPTIONS = ("max_new_tokens", "seed", "num_tries", "max_retry_seconds", "temperature", "fixed_temperature")
_LOCAL_OPTIONS = ("model_name", "strip_thinking_model", "reasoning_parser", "gpt_oss_reasoning_level",
                  "speculative_method", "num_speculative_tokens", "max_model_len",
                  "enable_thinking", "attention_backend", "dtype", "adapter_path", "default_truncation",
                  "engine_args", "disable_log_stats", "max_num_seqs", "language_model_only")
# Keys a local model config must carry, null when unset.
_LOCAL_REQUIRED = ("strip_thinking_model", "reasoning_parser", "gpt_oss_reasoning_level", "speculative_method",
                   "num_speculative_tokens", "max_model_len", "enable_thinking", "attention_backend")


@dataclass(frozen=True)
class JudgeSpec:
    """A judge as configured, before any engine or client exists."""

    model_type: str
    model_name: str
    options: JudgeOptions = field(compare=True)

    @classmethod
    def from_config(cls, config: Any, **overrides: Any) -> "JudgeSpec":
        """Return the judge a model config (a name, a YAML path or a mapping) describes, ``overrides`` replacing
        keys."""
        options = JudgeOptions(_load(config))
        if overrides:
            options = options.replace(**overrides)
        if "model_type" not in options:
            raise JudgeError("a model config names its model_type")
        from ruleforge.judges.providers import provider_for

        return provider_for(str(options["model_type"])).spec_from_config(options)

    @classmethod
    def local(cls, model_path: Path | str, *, tp_size: int = 1, **options: Any) -> "JudgeSpec":
        """Return a local vLLM judge over the weights at ``model_path``; ``model_name`` defaults to its directory
        name."""
        given = _checked_options(options, _COMMON_OPTIONS + _LOCAL_OPTIONS, "local")
        path = str(model_path)
        config = {"model_type": "local", "model_name": Path(path).name, "model_path": path, "tp_size": int(tp_size),
                  **dict.fromkeys(_LOCAL_REQUIRED), **given}
        return cls.from_config(config)

    @classmethod
    def openai(cls, model: str, **options: Any) -> "JudgeSpec":
        """Return a judge on OpenAI's synchronous Responses API, asking for a reasoning summary by default."""
        given = _checked_options(options, _COMMON_OPTIONS + (
            "reasoning_level", "reasoning_summary", "send_max_output_tokens", "stream"), "openai")
        return cls.from_config({"model_type": "openai", "model_name": str(model), "reasoning_summary": "auto",
                                **given})

    @classmethod
    def openai_compatible(cls, model: str, base_url: str, **options: Any) -> "JudgeSpec":
        """Return a judge on any server speaking OpenAI's Chat Completions at ``base_url`` (in its identity), keyed from
        ``api_key_env`` (``OPENAI_API_KEY`` by default; None: no key)."""
        given = _checked_options(options, _COMMON_OPTIONS + (
            "api_key_env", "reasoning_level", "send_max_output_tokens"), "openai_compatible")
        return cls.from_config({"model_type": "openai_compatible", "model_name": str(model), "base_url": str(base_url),
                                **given})

    @classmethod
    def anthropic(cls, model: str, **options: Any) -> "JudgeSpec":
        """Return a judge on Anthropic's synchronous Messages API: temperature 1.0, thinking summarized by default."""
        given = _checked_options(options, _COMMON_OPTIONS + ("reasoning_level", "thinking_display"), "anthropic")
        return cls.from_config({"model_type": "anthropic", "model_name": str(model), "send_max_output_tokens": True,
                                "fixed_temperature": 1.0, "temperature": 1.0,
                                "max_new_tokens": ANTHROPIC_DEFAULT_MAX_TOKENS, "thinking_display": "summarized",
                                **given})

    @classmethod
    def fake(cls, seed: int = 1, **options: Any) -> "JudgeSpec":
        """Return the deterministic stand-in judge ``fake-<seed>``; free, but refused in a ``final`` campaign."""
        given = _checked_options(options, ("max_new_tokens", "num_tries"), "fake")
        return cls.from_config({"model_type": "fake", "model_name": f"fake-{int(seed)}", "seed": int(seed), **given})

    def with_options(self, **changes: Any) -> "JudgeSpec":
        """Return this judge with ``changes`` to its options."""
        return type(self).from_config(self.options.replace(**changes))

    @property
    def provider(self) -> Provider:
        from ruleforge.judges.providers import provider_for

        return provider_for(self.model_type)

    def identity(self, *, verify_weights: bool = True) -> JudgeIdentity:
        """Return what a condition records of this judge; ``verify_weights`` recomputes merged weights' digest."""
        found = self.provider.identity(self, verify_weights=verify_weights)
        builtin = getattr(self.provider, "builtin", False)
        if found.model_type != self.model_type or found.model_name != self.model_name:
            raise JudgeError(f"the {self.model_type!r} provider named its judge {found.model_type!r} "
                             f"{found.model_name!r}, not {self.model_name!r}")
        if builtin == (found.extension is not None):
            raise JudgeError(
                f"model_type {self.model_type!r}: a registered judge's identity carries its extension (its "
                "behaviour-changing options) and a built-in one's carries none")
        return found

    def capabilities(self) -> Capabilities:
        """Return what this judge can do, from its configuration."""
        return self.provider.capabilities(self)

    def request_settings(self) -> dict[str, Any]:
        """Return the options that shape its request bytes without being identity (``reasoning_summary``,
        ``thinking_display``)."""
        return dict(self.provider.request_settings(self))

    def start(self) -> "Judge":
        """Build the engine or client."""
        return self.provider.start(self)


class Judge(ABC):
    """A started judge: one attempt per call, no retry and no validation."""

    spec: JudgeSpec
    capabilities: Capabilities

    @abstractmethod
    def generate_once(
        self,
        prompts: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        seeds: Optional[Sequence[Optional[int]]] = None,
        request_block_id: str = "",
        truncation: Optional[Mapping[str, Any]] = None,
    ) -> list["GenerationOutcome"]:
        """Answer every prompt exactly once, a transport exception as ``error``; ``request_block_id`` addresses a batch
        and is never shown in a message."""

    @property
    @abstractmethod
    def usage(self) -> "TokenUsageTracker":
        """Return the token usage this judge has reported so far."""

    def identity(self) -> JudgeIdentity:
        """Return its spec's identity, merged weights' digest read as recorded."""
        return self.spec.identity(verify_weights=False)

    def close(self) -> None:
        """Release what ``start`` built; nothing by default."""


class BatchJudge(Judge):
    """A judge whose blocks can be kept in flight together: each is estimated, submitted, then polled."""

    @abstractmethod
    def estimate_block_tokens(self, prompts: list[dict[str, str]], *, temperature: Optional[float] = None,
                              seeds: Optional[Sequence[Optional[int]]] = None, request_block_id: str = "") -> int:
        """Return the input tokens a block would enqueue."""

    @abstractmethod
    def submit_block(self, prompts: list[dict[str, str]], *, temperature: Optional[float] = None,
                     seeds: Optional[Sequence[Optional[int]]] = None, request_block_id: str = "") -> None:
        """Submit a block without waiting; a block already submitted is not bought again."""

    @abstractmethod
    def poll_block(self, prompts: list[dict[str, str]], *, temperature: Optional[float] = None,
                   seeds: Optional[Sequence[Optional[int]]] = None,
                   request_block_id: str = "") -> Optional[list["GenerationOutcome"]]:
        """Return a block's outcomes once it has landed, else None."""


class Provider(ABC):
    """What serves one ``model_type``: spec, record, capabilities and start; a registered provider's ``identity`` must
    carry every behaviour-changing option in its ``extension``."""

    builtin: ClassVar[bool] = False

    def spec_from_config(self, options: JudgeOptions) -> JudgeSpec:
        """Return the spec a model config describes; refuse one this provider cannot serve."""
        if "model_name" not in options:
            raise JudgeError(f"a {options['model_type']!r} judge's config names its model_name")
        return JudgeSpec(model_type=str(options["model_type"]), model_name=str(options["model_name"]),
                         options=options)

    @abstractmethod
    def identity(self, spec: JudgeSpec, *, verify_weights: bool = True) -> JudgeIdentity:
        """Return what a condition records of ``spec``'s judge."""

    @abstractmethod
    def capabilities(self, spec: JudgeSpec) -> Capabilities:
        """Return what ``spec``'s judge can do."""

    def request_settings(self, spec: JudgeSpec) -> Mapping[str, Any]:
        """Return the request-shaping options that are not identity; none by default."""
        return {}

    @abstractmethod
    def start(self, spec: JudgeSpec) -> Judge:
        """Build ``spec``'s judge."""


def identity_fields(identity: JudgeIdentity) -> dict[str, Any]:
    """Return a ``JudgeIdentity`` as plain data, the adapter as its label and digests."""
    payload = {item.name: getattr(identity, item.name) for item in fields(identity)}
    adapter = identity.adapter
    payload["adapter"] = None if adapter is None else {
        "label": adapter.label, "sha256": adapter.sha256, "config_sha256": adapter.config_sha256}
    if identity.truncation_source is not None:
        payload["truncation_source"] = list(identity.truncation_source)
    return json.loads(json.dumps(payload, default=str))


__all__ = [
    "BatchJudge", "Capabilities", "Judge", "JudgeError", "JudgeIdentity", "JudgeOptions", "JudgeSpec",
    "MODEL_CONFIG_DIR", "MODEL_CONFIG_DIRS", "Provider", "SHORTHANDS", "extension", "identity_fields",
    "judge_forms", "model_config_names", "model_config_path", "refuse_credential",
]
