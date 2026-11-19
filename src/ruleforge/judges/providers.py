from __future__ import annotations

from dataclasses import replace
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

from ruleforge.core.optional import require
from ruleforge.judges.base import Capabilities, Judge, JudgeError, JudgeIdentity, JudgeOptions, JudgeSpec, Provider
from ruleforge.judges.backend import (
    ANOTHER_SERVER,
    BATCH_MODEL_TYPES,
    COMPATIBLE_MODEL_TYPE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SEED,
    REASONING_KEYS,
    SEED_LOCAL_DETERMINISTIC,
    SEED_PROVIDER_BEST_EFFORT,
    SEED_UNSUPPORTED,
    ModelBackend,
    _optional_int,
    _optional_str,
    canonical_serving_dtype,
    logger,
)
from ruleforge.judges.batch_api import (
    CHAT_COMPLETIONS_ENDPOINT,
    DEFAULT_POLL_SECONDS,
    RESPONSES_ENDPOINT,
    resolve_endpoint,
)

if TYPE_CHECKING:
    from ruleforge.judges.weights import AdapterIdentity, ModelWeightsIdentity

ENTRY_POINT_GROUP = "ruleforge.judges"
# Batch transports whose blocks can be kept in flight together when the config sets max_enqueued_tokens.
WINDOWED_MODEL_TYPES = ("openai_batch",)
# The largest max_tokens the anthropic SDK accepts on a non-streaming call with its default timeout.
ANTHROPIC_SYNC_MAX_TOKENS = 128_000 * 600 // 3600


def judge_adapter(model_cfg: Any) -> Optional[AdapterIdentity]:
    """Return the judge's LoRA adapter by content, or None when it serves none."""
    from ruleforge.judges.weights import adapter_identity

    value = model_cfg.get("adapter_path")
    return adapter_identity(str(value)) if value else None


def judge_model_weights(model_cfg: Any, *, verify: bool = True) -> Optional[ModelWeightsIdentity]:
    """Return a local judge's merged weights by content (digest recomputed under ``verify``), or None for upstream
    weights."""
    from ruleforge.judges.weights import model_weights_identity, recorded_model_weights

    return (model_weights_identity if verify else recorded_model_weights)(str(model_cfg.model_path))


def judge_serving_dtype(model_cfg: Any) -> Optional[str]:
    """Return a local judge's canonical serving dtype, or None for vLLM's ``auto``."""
    try:
        return canonical_serving_dtype(model_cfg.get("dtype"), model_cfg.get("attention_backend"))
    except ValueError as error:
        raise JudgeError(f"judge_model {model_cfg.get('model_name')}: {error}") from error


def sends_max_output_tokens(model_type: str, model_cfg: Any) -> Optional[bool]:
    """Return whether a request carries the token ceiling (None for a judge that sends no request)."""
    if model_type in BATCH_MODEL_TYPES or model_type in ("openai", COMPATIBLE_MODEL_TYPE):
        return bool(model_cfg.get("send_max_output_tokens", False))
    if model_type == "anthropic":
        return True
    return None


def config_identity(model_type: str, model_cfg: Any, *, seed_capability: str, verify_weights: bool = True,
                    model_name: Optional[str] = None, max_new_tokens: Optional[int] = None) -> JudgeIdentity:
    """Return what a condition records of a built-in judge, read from its model config."""
    endpoint = recorded_endpoint(model_type, model_cfg)
    local = model_type == "local"
    model_path = model_cfg.get("model_path") if local else None
    adapter = judge_adapter(model_cfg) if local else None
    weights = judge_model_weights(model_cfg, verify=verify_weights) if local else None
    enable_thinking = model_cfg.get("enable_thinking") if local else None
    default_truncation = model_cfg.get("default_truncation") if local else None
    return JudgeIdentity(
        model_name=str(model_cfg.model_name) if model_name is None else model_name,
        model_type=model_type,
        seed_capability=seed_capability,
        max_new_tokens=(int(getattr(model_cfg, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
                        if max_new_tokens is None else max_new_tokens),
        endpoint=endpoint,
        reasoning_effort=ModelBackend._reasoning_effort(model_cfg),
        send_max_output_tokens=sends_max_output_tokens(model_type, model_cfg),
        model_path=None if model_path is None else Path(str(model_path)).name,
        adapter=adapter,
        model_weights_sha256=None if weights is None else weights.sha256,
        serving_dtype=judge_serving_dtype(model_cfg) if local else None,
        enable_thinking=None if enable_thinking is None else bool(enable_thinking),
        truncation_source=(str(model_path), _plain(default_truncation)) if local else None,
    )


def recorded_endpoint(model_type: str, model_cfg: Any) -> Optional[str]:
    """Return the endpoint a condition records: a batch transport's ``batch_endpoint`` as configured, the URL an
    ``openai_compatible`` judge sends to, else None."""
    if model_type == COMPATIBLE_MODEL_TYPE:
        from ruleforge.judges.compatible import endpoint_url

        return endpoint_url(model_cfg)
    endpoint = model_cfg.get("batch_endpoint")
    return None if endpoint is None else str(endpoint)


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, (DictConfig, ListConfig)) else value


def refuse_unsent_reasoning(model_type: str, model_cfg: Any, keys: tuple[str, ...], sent: str) -> None:
    """Refuse a reasoning level this judge's request never carries; ``sent`` says what it sends instead."""
    for key in keys:
        value = _optional_str(model_cfg, key)
        if value is not None:
            raise JudgeError(f"model_type {model_type!r} sends no {key} ({value!r}): {sent}; remove it from its "
                             "config, or the condition would record a reasoning level no request carries")


DISPLAY_MODEL_TYPES = ("anthropic", "anthropic_batch")


def thinking_display(model_type: str, model_cfg: Any) -> Optional[str]:
    """Return a config's ``thinking_display`` (None: none sent); refused on a transport whose request has no such
    field."""
    value = _optional_str(model_cfg, "thinking_display")
    if model_type not in DISPLAY_MODEL_TYPES:
        if value is not None:
            raise JudgeError(f"model_type {model_type!r} sends no thinking_display; remove it from its config")
        return None
    from ruleforge.judges.anthropic_batch import display_setting

    return display_setting(value)


def _fixed_temperature(model_cfg: Any) -> Optional[float]:
    value = getattr(model_cfg, "fixed_temperature", None)
    return None if value is None else float(value)


class BuiltinProvider(Provider):
    """A built-in ``model_type``, started as ``ModelBackend``, which hands it the model config to wire."""

    builtin: ClassVar[bool] = True
    model_type: ClassVar[str]
    seed: ClassVar[str] = SEED_UNSUPPORTED

    def seed_capability(self, model_cfg: Any) -> str:
        return self.seed

    def spec_from_config(self, options: JudgeOptions) -> JudgeSpec:
        spec = super().spec_from_config(options)
        self.validate(options.config())
        return spec

    def validate(self, model_cfg: Any) -> None:
        """Refuse a model config this judge cannot be started from, before any client or engine exists."""
        thinking_display(self.model_type, model_cfg)
        endpoint = _optional_str(model_cfg, "batch_endpoint")
        if endpoint is not None and self.model_type not in BATCH_MODEL_TYPES:
            raise JudgeError(f"model_type {self.model_type!r} sends no batch request; remove "
                             f"batch_endpoint={endpoint!r} from its config")
        self._check_base_url(model_cfg)

    def _check_base_url(self, model_cfg: Any) -> None:
        base_url = _optional_str(model_cfg, "base_url")
        if base_url is not None:
            raise JudgeError(f"model_type {self.model_type!r} takes no base_url, so no request would reach it; remove "
                             f"base_url from its config ({ANOTHER_SERVER})")

    def identity(self, spec: JudgeSpec, *, verify_weights: bool = True) -> JudgeIdentity:
        model_cfg = spec.options.config()
        self.validate(model_cfg)
        return config_identity(self.model_type, model_cfg, seed_capability=self.seed_capability(model_cfg),
                               verify_weights=verify_weights)

    def capabilities(self, spec: JudgeSpec) -> Capabilities:
        model_cfg = spec.options.config()
        return self.capabilities_of(model_cfg)

    def capabilities_of(self, model_cfg: Any) -> Capabilities:
        """Return the capabilities a model config gives."""
        return Capabilities(seed=self.seed_capability(model_cfg), per_request_errors=True, spends_money=True,
                            fixed_temperature=_fixed_temperature(model_cfg))

    def start(self, spec: JudgeSpec) -> Judge:
        return ModelBackend(spec.options.config())

    def configure(self, backend: ModelBackend, model_cfg: Any) -> None:
        raise NotImplementedError


class LocalProvider(BuiltinProvider):
    """A vLLM engine in this process."""

    model_type = "local"
    seed = SEED_LOCAL_DETERMINISTIC

    def validate(self, model_cfg: Any) -> None:
        from ruleforge.judges.models import THINKING_STRIPPERS

        super().validate(model_cfg)
        refuse_unsent_reasoning(self.model_type, model_cfg, ("reasoning_level",),
                                "the chat template is given gpt_oss_reasoning_level alone")
        strip = _optional_str(model_cfg, "strip_thinking_model")
        if strip is not None and strip not in THINKING_STRIPPERS:
            raise JudgeError(f"strip_thinking_model {strip!r} names no thinking-token stripper: one of "
                             f"{list(THINKING_STRIPPERS)}, or null for a judge whose text holds no thought")

    def capabilities_of(self, model_cfg: Any) -> Capabilities:
        return Capabilities(seed=self.seed, per_request_errors=True, sampler_truncation=True, local_engine=True,
                            fixed_temperature=_fixed_temperature(model_cfg))

    def configure(self, backend: ModelBackend, model_cfg: Any) -> None:
        from ruleforge.judges.backend import local_engine_settings
        from ruleforge.judges.engine_state import engine_max_model_len, engine_resolution
        from ruleforge.judges.lora import lora_serving
        from ruleforge.judges.models import (
            call_local_model,
            call_local_model_with_details,
            format_prompt_for_generation,
            harmony_encoding,
            init_local_model,
            local_engine_kwargs,
            prompt_token_ids,
        )

        self.validate(model_cfg)
        require("serving a local judge", "vllm")
        strip_thinking_model: Optional[str] = getattr(model_cfg, "strip_thinking_model", None)
        if strip_thinking_model == "openai_gptoss":
            harmony_encoding()  # fail before the engine is built if the vocab cannot be loaded
        gpt_oss_reasoning_level: Optional[str] = getattr(model_cfg, "gpt_oss_reasoning_level", None)

        serving = lora_serving(backend.adapter_path)
        lora_request, backend.adapter_metadata = serving.request, serving.metadata
        if serving.enable_lora:
            logger.info(
                "Serving LoRA adapter %s (rank ceiling %d)",
                serving.metadata["name"],
                serving.max_lora_rank,
            )

        backend.tp_size = int(model_cfg.tp_size)
        backend.attention_backend = model_cfg.attention_backend
        settings = {**local_engine_settings(model_cfg), **serving.engine_settings()}
        backend.engine_args = local_engine_kwargs(**settings)
        backend.model_path = str(settings["model_path"])
        backend.llm, tokenizer = init_local_model(**settings)
        backend.engine_resolved = engine_resolution(backend.llm)
        backend.prompt_token_limit = engine_max_model_len(backend.llm)
        backend.prompt_token_count = lambda prompt: len(prompt_token_ids(tokenizer, format_prompt_for_generation(
            tokenizer, prompt, gpt_oss_reasoning_level=gpt_oss_reasoning_level)))

        def _local_batch(prompts, temperature, max_new_tokens, num_responses, seed):
            return call_local_model(
                backend.llm,
                tokenizer,
                prompts,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                gpt_oss_reasoning_level=gpt_oss_reasoning_level,
                num_responses=num_responses,
                seed=seed,
                strip_thinking_model=strip_thinking_model,
                lora_request=lora_request,
            )

        def _local_batch_with_details(prompts, temperature, max_new_tokens, num_responses, seed, truncation=None):
            return call_local_model_with_details(
                backend.llm,
                tokenizer,
                prompts,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                gpt_oss_reasoning_level=gpt_oss_reasoning_level,
                num_responses=num_responses,
                seed=seed,
                strip_thinking_model=strip_thinking_model,
                lora_request=lora_request,
                truncation=truncation,
            )

        backend._batch_generate = _local_batch
        backend._batch_generate_with_details = _local_batch_with_details
        backend._base_delay = 0.0
        backend.seed_capability = self.seed


class _SyncProvider(BuiltinProvider):
    """A provider's synchronous API: one call per prompt, each failure carried as its ``error``."""

    def caller(self, backend: ModelBackend, model_cfg: Any) -> Any:
        raise NotImplementedError

    def configure(self, backend: ModelBackend, model_cfg: Any) -> None:
        from ruleforge.judges.models import _api_generate_batch, _api_generate_batch_with_details

        self.validate(model_cfg)
        caller = self.caller(backend, model_cfg)
        backend._batch_generate = partial(_api_generate_batch, caller)
        backend._batch_generate_with_details = partial(_api_generate_batch_with_details, caller)
        backend._base_delay = 1.0
        backend.seed_capability = self.seed_capability(model_cfg)


class OpenAIProvider(_SyncProvider):
    """OpenAI's Responses API; takes no seed."""

    model_type = "openai"
    seed = SEED_UNSUPPORTED

    def request_settings(self, spec: JudgeSpec) -> Mapping[str, Any]:
        model_cfg = spec.options.config()
        summary = _optional_str(model_cfg, "reasoning_summary")
        carried = summary is not None and ModelBackend._reasoning_effort(model_cfg) is not None
        return {"reasoning_summary": summary} if carried else {}

    def caller(self, backend: ModelBackend, model_cfg: Any) -> Any:
        from ruleforge.judges.models import _make_openai_caller, init_openai_client

        return _make_openai_caller(
            init_openai_client(),
            backend.model_name,
            ModelBackend._reasoning_effort(model_cfg),
            _optional_str(model_cfg, "reasoning_summary"),
            tracker=backend.token_tracker,
            stream=bool(getattr(model_cfg, "stream", False)),
            send_max_output_tokens=bool(getattr(model_cfg, "send_max_output_tokens", False)),
        )


class OpenAICompatibleProvider(_SyncProvider):
    """Any server speaking OpenAI's Chat Completions, at the base URL its identity records; it takes no seed."""

    model_type = COMPATIBLE_MODEL_TYPE
    seed = SEED_UNSUPPORTED

    def _check_base_url(self, model_cfg: Any) -> None:
        from ruleforge.judges.compatible import normalised_base_url

        normalised_base_url(_optional_str(model_cfg, "base_url"))

    def validate(self, model_cfg: Any) -> None:
        from ruleforge.judges.compatible import refuse_key_in_url, refuse_unsent_options

        super().validate(model_cfg)
        refuse_unsent_options(model_cfg)
        refuse_key_in_url(model_cfg)

    def capabilities_of(self, model_cfg: Any) -> Capabilities:
        """Return the provider's capabilities, spending money only with a key (``api_key_env: null`` bills no account).
        The key variable is no part of the identity: keyless and keyed configs of one model and URL are one judge."""
        from ruleforge.judges.compatible import api_key_variable

        return replace(super().capabilities_of(model_cfg), spends_money=api_key_variable(model_cfg) is not None)

    def caller(self, backend: ModelBackend, model_cfg: Any) -> Any:
        from ruleforge.judges.compatible import init_client, make_caller

        return make_caller(init_client(model_cfg), backend.model_name, ModelBackend._reasoning_effort(model_cfg),
                           tracker=backend.token_tracker,
                           send_max_output_tokens=bool(getattr(model_cfg, "send_max_output_tokens", False)))


def _refuse_anthropic_keys(model_type: str, model_cfg: Any) -> None:
    if getattr(model_cfg, "reasoning_summary", None):
        raise ValueError(f"model_type {model_type!r} sends no reasoning_summary; remove it from its config")
    if not bool(getattr(model_cfg, "send_max_output_tokens", False)):
        raise ValueError(
            f"model_type {model_type!r} sends max_new_tokens as every request's required "
            "max_tokens; set send_max_output_tokens: true in its config"
        )


class AnthropicProvider(_SyncProvider):
    """Anthropic's Messages API; no seed, no sampling parameter."""

    model_type = "anthropic"
    seed = SEED_UNSUPPORTED

    def validate(self, model_cfg: Any) -> None:
        from ruleforge.judges.anthropic_batch import effort_level

        super().validate(model_cfg)
        _refuse_anthropic_keys(self.model_type, model_cfg)
        effort_level(ModelBackend._reasoning_effort(model_cfg))
        ceiling = int(getattr(model_cfg, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
        if ceiling > ANTHROPIC_SYNC_MAX_TOKENS:
            raise JudgeError(
                f"model_type {self.model_type!r} sends max_new_tokens={ceiling} as max_tokens on a non-streaming "
                f"call, which the anthropic SDK refuses above {ANTHROPIC_SYNC_MAX_TOKENS} before any request (it "
                "would take over ten minutes); set max_new_tokens at most that, or use anthropic_batch")

    def request_settings(self, spec: JudgeSpec) -> Mapping[str, Any]:
        display = thinking_display(self.model_type, spec.options.config())
        return {} if display is None else {"thinking_display": display}

    def caller(self, backend: ModelBackend, model_cfg: Any) -> Any:
        from ruleforge.judges.models import _make_anthropic_caller, init_anthropic_client

        return _make_anthropic_caller(init_anthropic_client(), backend.model_name,
                                      ModelBackend._reasoning_effort(model_cfg), tracker=backend.token_tracker,
                                      thinking_display=thinking_display(self.model_type, model_cfg))


def _fake_seed(model_cfg: Any) -> int:
    return int(getattr(model_cfg, "seed", DEFAULT_SEED))


class FakeProvider(_SyncProvider):
    """The deterministic ``fake-<seed>`` judge."""

    model_type = "fake"
    seed = SEED_LOCAL_DETERMINISTIC

    def validate(self, model_cfg: Any) -> None:
        super().validate(model_cfg)
        refuse_unsent_reasoning(self.model_type, model_cfg, REASONING_KEYS,
                                "its answers are a function of its seed and the prompt alone")
        expected = f"fake-{_fake_seed(model_cfg)}"
        if str(model_cfg.model_name) != expected:
            raise JudgeError(
                f"a fake judge of seed {_fake_seed(model_cfg)} is named {expected!r}, not {model_cfg.model_name!r}: "
                "its name is the only identity field its answers depend on")

    def capabilities_of(self, model_cfg: Any) -> Capabilities:
        return Capabilities(seed=self.seed, per_request_errors=True)

    def caller(self, backend: ModelBackend, model_cfg: Any) -> Any:
        from ruleforge.judges.fake import make_fake_caller

        return make_fake_caller(_fake_seed(model_cfg))

    def configure(self, backend: ModelBackend, model_cfg: Any) -> None:
        super().configure(backend, model_cfg)
        backend._base_delay = 0.0


class BatchProvider(BuiltinProvider):
    """A paid Batch API transport, its state content-addressed under ``batch_root``."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type  # type: ignore[misc]

    @staticmethod
    def _batch_endpoint(model_cfg: Any) -> str:
        return resolve_endpoint(_optional_str(model_cfg, "batch_endpoint") or "responses")

    def seed_capability(self, model_cfg: Any) -> str:
        if self.model_type == "openai_batch":
            endpoint = self._batch_endpoint(model_cfg)
            return SEED_PROVIDER_BEST_EFFORT if endpoint == CHAT_COMPLETIONS_ENDPOINT else SEED_UNSUPPORTED
        return SEED_UNSUPPORTED

    def validate(self, model_cfg: Any) -> None:
        super().validate(model_cfg)
        self._refuse_endpoint(model_cfg)
        if self.model_type == "anthropic_batch":
            _refuse_anthropic_keys(self.model_type, model_cfg)

    def _refuse_endpoint(self, model_cfg: Any) -> None:
        endpoint = _optional_str(model_cfg, "batch_endpoint")
        if self.model_type == "anthropic_batch" and endpoint is not None:
            raise ValueError(
                f"model_type {self.model_type!r} has one endpoint; remove "
                f"batch_endpoint={endpoint!r} from its config"
            )

    def capabilities_of(self, model_cfg: Any) -> Capabilities:
        windowed = self.model_type in WINDOWED_MODEL_TYPES and getattr(model_cfg, "max_enqueued_tokens", None) is not None
        return Capabilities(seed=self.seed_capability(model_cfg), batch=True, windowed=windowed,
                            per_request_errors=True, spends_money=True,
                            fixed_temperature=_fixed_temperature(model_cfg))

    def request_settings(self, spec: JudgeSpec) -> Mapping[str, Any]:
        model_cfg = spec.options.config()
        if self.model_type == "anthropic_batch":
            display = thinking_display(self.model_type, model_cfg)
            return {} if display is None else {"thinking_display": display}
        summary = _optional_str(model_cfg, "reasoning_summary")
        if self.model_type != "openai_batch" or summary is None:
            return {}
        endpoint = self._batch_endpoint(model_cfg)
        carried = endpoint == RESPONSES_ENDPOINT and ModelBackend._reasoning_effort(model_cfg) is not None
        return {"reasoning_summary": summary} if carried else {}

    def configure(self, backend: ModelBackend, model_cfg: Any) -> None:
        # A direct ModelBackend construction skips validate().
        self._check_base_url(model_cfg)
        batch_root = getattr(model_cfg, "batch_root", None)
        if not batch_root:
            raise ValueError(
                f"model_type {self.model_type!r} requires batch_root: the "
                "batch state directory is how a killed run resumes the "
                "batch it already paid for, so it cannot be a default that "
                "changes between invocations."
            )
        timeout = getattr(model_cfg, "batch_timeout_seconds", None)
        adapter_kwargs = dict(
            model_name=backend.model_name,
            root=Path(batch_root),
            reasoning_effort=ModelBackend._reasoning_effort(model_cfg),
            send_max_output_tokens=bool(getattr(model_cfg, "send_max_output_tokens", False)),
            poll_seconds=float(getattr(model_cfg, "batch_poll_seconds", DEFAULT_POLL_SECONDS)),
            timeout_seconds=None if timeout is None else float(timeout),
            tracker=backend.token_tracker,
        )
        self._refuse_endpoint(model_cfg)
        if self.model_type == "anthropic_batch":
            from ruleforge.judges import anthropic_batch as batch_module

            _refuse_anthropic_keys(self.model_type, model_cfg)
            adapter_kwargs.pop("send_max_output_tokens")
            adapter_kwargs["thinking_display"] = thinking_display(self.model_type, model_cfg)
        else:
            from ruleforge.judges import openai_batch as batch_module

            adapter_kwargs.update(
                endpoint=self._batch_endpoint(model_cfg),
                reasoning_summary=_optional_str(model_cfg, "reasoning_summary"),
            )
        backend.seed_capability = self.seed_capability(model_cfg)
        generators = batch_module.make_batch_generate(**adapter_kwargs)
        backend._batch_generate, backend._batch_generate_with_details = generators
        backend._generate_request_block = batch_module.make_block_generate(**adapter_kwargs)
        # max_enqueued_tokens is a submission budget (several blocks in flight), not identity.
        backend.max_enqueued_tokens = _optional_int(model_cfg, "max_enqueued_tokens")
        backend.batch_poll_seconds = float(getattr(model_cfg, "batch_poll_seconds", DEFAULT_POLL_SECONDS))
        backend.batch_timeout_seconds = None if timeout is None else float(timeout)
        if backend.max_enqueued_tokens is not None:
            if hasattr(batch_module, "make_windowed_block_ops"):
                (
                    backend._estimate_request_block,
                    backend._submit_request_block,
                    backend._poll_request_block,
                ) = batch_module.make_windowed_block_ops(**adapter_kwargs)
            else:
                logger.warning(
                    "max_enqueued_tokens is set but %s has no windowed "
                    "submission; running one block at a time",
                    self.model_type,
                )
                backend.max_enqueued_tokens = None
        backend._base_delay = 1.0


PROVIDERS: dict[str, BuiltinProvider] = {
    provider.model_type: provider
    for provider in (LocalProvider(), OpenAIProvider(), AnthropicProvider(), OpenAICompatibleProvider(),
                     FakeProvider(), *(BatchProvider(model_type) for model_type in BATCH_MODEL_TYPES))
}

_REGISTERED: dict[str, Provider] = {}


def _instance(found: Any, where: str) -> Provider:
    provider = found() if isinstance(found, type) else found
    if not isinstance(provider, Provider):
        raise JudgeError(f"{where} names {found!r}, which is not a ruleforge.judges.Provider")
    return provider


def register_judge(model_type: str, provider: Any) -> Provider:
    """Register ``provider`` (a ``Provider`` or its class) for ``model_type`` in this process; a built-in type is
    refused."""
    if not isinstance(model_type, str) or not model_type:
        raise JudgeError(f"a judge provider is registered under a model_type name, got {model_type!r}")
    if model_type in PROVIDERS:
        raise JudgeError(f"{model_type!r} is a built-in model_type; register the provider under a name of its own")
    instance = _instance(provider, f"register_judge({model_type!r}, ...)")
    _REGISTERED[model_type] = instance
    return instance


def _entry_point(model_type: str) -> Optional[Provider]:
    found = None
    for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
        if entry.name in PROVIDERS:
            raise JudgeError(f"the {ENTRY_POINT_GROUP!r} entry point {entry.name!r} ({entry.value}) is named like a "
                             "built-in model_type; an installed provider takes a name of its own")
        if entry.name == model_type and found is None:
            found = _instance(entry.load(), f"the {ENTRY_POINT_GROUP!r} entry point {entry.name!r}")
    return found


def provider_for(model_type: str) -> Provider:
    """Return the provider serving ``model_type``: a built-in, a registered one, else an installed entry point."""
    if model_type in PROVIDERS:
        return PROVIDERS[model_type]
    if model_type not in _REGISTERED:
        found = _entry_point(model_type)
        if found is None:
            raise JudgeError(
                f"no judge provider for model_type {model_type!r}: not built in ({', '.join(PROVIDERS)}), "
                f"registered (register_judge), or installed as a {ENTRY_POINT_GROUP!r} entry point")
        _REGISTERED[model_type] = found
    return _REGISTERED[model_type]


def builtin_provider(model_type: str) -> BuiltinProvider:
    """Return the built-in provider for ``model_type``."""
    if model_type not in PROVIDERS:
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    return PROVIDERS[model_type]


def backend_identity(backend: Any, model_cfg: Any) -> JudgeIdentity:
    """Return what a condition records of a constructed backend, with the backend's own name, ceiling and seed
    capability."""
    model_type = str(backend.model_type)
    if model_type not in PROVIDERS:
        provider = provider_for(model_type)
        found = provider.identity(provider.spec_from_config(JudgeOptions(model_cfg)), verify_weights=True)
        return replace(found, model_name=backend.model_name, max_new_tokens=backend.max_new_tokens,
                       seed_capability=backend.seed_capability)
    return config_identity(model_type, model_cfg, seed_capability=backend.seed_capability,
                           model_name=backend.model_name, max_new_tokens=backend.max_new_tokens)


Judge.register(ModelBackend)
