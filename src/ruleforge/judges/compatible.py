from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

from ruleforge.judges.backend import COMPATIBLE_MODEL_TYPE as MODEL_TYPE
from ruleforge.judges.backend import _optional_str
from ruleforge.judges.base import JudgeError
from ruleforge.judges.batch_api import (
    FATAL_PROVIDER_ERRORS,
    build_chat_body,
    chat_finish_reason,
    chat_reasoning,
    chat_text,
)
from ruleforge.judges.models import (
    EMPTY_USER_TURN,
    CallFailed,
    DetailCallFn,
    _answered,
    _failed,
    _warn_seed_ignored_once,
)
from ruleforge.judges.token_usage import TokenUsage

logger = logging.getLogger(__name__)

# The SDK sends each request to the base URL with this path appended.
CHAT_COMPLETIONS_PATH = "/chat/completions"
# Where the API key is read from when a config names no variable; ``api_key_env: null`` sends none.
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
_DEFAULT_PORTS = {"http": 80, "https": 443}
_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# What a URL carries as written; anything else (a space, a non-ASCII letter, "<>\\^`{|}) the SDK would re-encode.
_URL_TEXT = re.compile(r"[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*")
_TOOL_FINISH_REASONS = ("tool_calls", "function_call")
# A judge name's ``MODEL@URL``: the model, then the server's base URL from the last ``@`` that starts one.
_SHORTHAND = re.compile(r"(?P<model>.+)@(?P<url>https?://.+)", re.IGNORECASE | re.DOTALL)


def split_shorthand(text: str) -> tuple[str, str]:
    """Return the model and base URL a judge name's ``MODEL@URL`` part names."""
    found = _SHORTHAND.fullmatch(text)
    if found is None:
        raise JudgeError(f"{MODEL_TYPE}:MODEL@URL names a model and its server's base URL, e.g. "
                         f"{MODEL_TYPE}:Qwen/Qwen3-8B@http://localhost:8000/v1")
    return found.group("model"), found.group("url")


def normalised_base_url(given: Optional[str]) -> str:
    """Return a base URL as called and recorded (scheme and host lowercased, a default port and trailing slashes
    dropped); no refusal repeats the URL, which may hold a credential."""
    text = (given or "").strip()
    if not text:
        raise JudgeError(f"model_type {MODEL_TYPE!r} needs base_url, the address its server's /chat/completions sits "
                         "under, e.g. http://localhost:8000/v1")
    if not _URL_TEXT.fullmatch(text):
        raise JudgeError("base_url holds a character a URL carries only percent-encoded (a space, a non-ASCII letter "
                         "or one of \"<>\\^`{|}), which the request would re-encode; percent-encode it, or give a "
                         "non-ASCII host in its xn-- form")
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        raise JudgeError("base_url is not a well-formed address: its host or port cannot be read") from None
    scheme = parts.scheme.lower()
    if "@" in parts.netloc:
        raise JudgeError("base_url holds a user or password, which the judge's identity would record; name the "
                         "variable holding the key with api_key_env instead")
    if parts.query or parts.fragment:
        raise JudgeError("base_url has a query or a fragment, which may hold a key and which the judge's identity "
                         "would record; give the server's address alone")
    if scheme not in _DEFAULT_PORTS or not parts.hostname:
        raise JudgeError("base_url is not an http:// or https:// address with a host")
    host = parts.netloc.lower()
    if ":" in host.rpartition("]")[2]:  # the port as written; it is sent as a number, a default one not at all
        host = host.rsplit(":", 1)[0]
    if port not in (None, _DEFAULT_PORTS[scheme]):
        host += f":{port}"
    path = parts.path.rstrip("/")
    if any(segment in (".", "..") for segment in path.split("/")):
        raise JudgeError("base_url's path has a . or .. segment, which the request would resolve away; give the path "
                         "it resolves to")
    if path.endswith(CHAT_COMPLETIONS_PATH):
        raise JudgeError(f"base_url ends with {CHAT_COMPLETIONS_PATH}, which the judge appends itself; give the "
                         "address it sits under, e.g. http://localhost:8000/v1")
    return f"{scheme}://{host}{path}"


def endpoint_url(model_cfg: Any) -> str:
    """Return the URL each request goes to, which the judge's identity records as its endpoint."""
    return normalised_base_url(_optional_str(model_cfg, "base_url")) + CHAT_COMPLETIONS_PATH


def api_key_variable(model_cfg: Any) -> Optional[str]:
    """Return the variable the API key is read from: the config's ``api_key_env``, ``OPENAI_API_KEY`` when it has
    none, None when it is null (a server that takes no key)."""
    if "api_key_env" not in model_cfg:
        return DEFAULT_API_KEY_ENV
    name = model_cfg.get("api_key_env")
    if name is None:
        return None
    if not isinstance(name, str) or not _VARIABLE_NAME.fullmatch(name):
        raise JudgeError("api_key_env names the environment variable that holds the API key (letters, digits and "
                         "_), never the key; set it to that name, or to null for a server that takes no key")
    return name


def refuse_key_in_url(model_cfg: Any) -> None:
    """Refuse a base_url holding the key the variable ``api_key_env`` names (when set, and long enough to be scrubbed
    from replies): the URL is printed and recorded."""
    variable = api_key_variable(model_cfg)
    key = (os.environ.get(variable) or "").strip() if variable is not None else ""
    if len(key) >= MIN_SCRUBBED_KEY_LENGTH and key in (_optional_str(model_cfg, "base_url") or ""):
        raise JudgeError("base_url holds the API key, which must not be part of the URL: the URL is printed and "
                         "recorded; give the server's address alone, the key in the variable api_key_env names")


def refuse_unsent_options(model_cfg: Any) -> None:
    """Refuse what a config may set for another API but this one never sends, and a key written into the config."""
    if "api_key" in model_cfg:
        raise JudgeError("a model config names the variable that holds its key (api_key_env), never the key; remove "
                         "api_key from it")
    for key in ("reasoning_summary", "stream"):
        if model_cfg.get(key):
            raise JudgeError(f"model_type {MODEL_TYPE!r} sends no {key}; remove it from its config")


# A shorter key is not scrubbed: local servers take placeholders such as EMPTY, which answers may contain.
MIN_SCRUBBED_KEY_LENGTH = 8
SCRUBBED_KEY = b"<api key>"


class KeyScrubbingTransport:
    """An httpx transport around the client's own that hands the SDK each response with the key replaced by
    ``<api key>`` in its reason phrase, headers and body, so nothing read from a response holds the key."""

    def __init__(self, inner: Any, key: str) -> None:
        self.inner = inner
        self._key = key.encode()

    def _scrub(self, data: bytes) -> bytes:
        return data.replace(self._key, SCRUBBED_KEY)

    def handle_request(self, request: Any) -> Any:
        response = self.inner.handle_request(request)
        # read() decodes the body by its content-encoding, so a compressed body is scrubbed too; the response handed on
        # carries it decoded, without that header or the length the scrub may have changed.
        body = self._scrub(response.read())
        headers = [(self._scrub(name), self._scrub(value)) for name, value in response.headers.raw
                   if name.lower() not in (b"content-encoding", b"content-length")]
        extensions = dict(response.extensions)
        if "reason_phrase" in extensions:
            extensions["reason_phrase"] = self._scrub(extensions["reason_phrase"])
        # Rebuilt by the response's own class (httpx is no declared dependency); the copy has no elapsed time, which
        # nothing reads.
        return type(response)(response.status_code, headers=headers, content=body, extensions=extensions,
                              request=request)

    def close(self) -> None:
        self.inner.close()


def scrub_responses(http_client: Any, key: str) -> None:
    """Make ``http_client`` hand on every response with ``key`` scrubbed, and keep httpcore's wire-level trace, the one
    log of a response written before the scrub, from being written."""
    if not (hasattr(http_client, "_transport") and hasattr(http_client, "_mounts")):
        raise RuntimeError("this httpx client keeps its transports where the key-scrubbing transport cannot wrap them "
                           "(it wraps an httpx 0.28 client's _transport and _mounts); install an httpx release it "
                           "supports")
    # Wrapped in place, as passing transport= to the client would make httpx ignore the environment's proxies.
    http_client._transport = KeyScrubbingTransport(http_client._transport, key)
    http_client._mounts = {pattern: None if inner is None else KeyScrubbingTransport(inner, key)
                           for pattern, inner in http_client._mounts.items()}
    # httpcore traces reason phrases and headers at DEBUG before the scrub; raised to INFO, it and its child loggers
    # write none.
    wire = logging.getLogger("httpcore")
    wire.setLevel(max(logging.INFO, wire.level))


def init_client(model_cfg: Any) -> Any:
    """Return an openai SDK client for the config's server keyed from ``api_key_env`` (no key when null): an explicit
    base URL, no redirect, no organization or project header, and the key scrubbed from every response."""
    base_url = normalised_base_url(_optional_str(model_cfg, "base_url"))
    variable = api_key_variable(model_cfg)
    api_key = ""
    if variable is not None:
        api_key = (os.environ.get(variable) or "").strip()
        if not api_key:
            raise ValueError("the environment variable the judge's api_key_env names is not set; a server that "
                             "takes no key is configured with api_key_env: null")
    try:
        from openai import DefaultHttpxClient, OpenAI, omit

        # A redirect is not followed: the answer would come from a server the identity does not record.
        http_client = DefaultHttpxClient(follow_redirects=False)
        client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client,
                        default_headers={"OpenAI-Organization": omit, "OpenAI-Project": omit})
    except Exception:
        # Raised ``from None``: an SDK that echoes its arguments into the exception would put the key in a log.
        logger.error("Failed to initialize the OpenAI-compatible client")
        raise RuntimeError("Failed to initialize the OpenAI-compatible client") from None
    if len(api_key) >= MIN_SCRUBBED_KEY_LENGTH:
        scrub_responses(http_client, api_key)
    logger.info("OpenAI-compatible client initialized for %s", base_url)
    return client


def _body(response: Any) -> Mapping[str, Any]:
    dump = getattr(response, "model_dump", None)
    return dump() if callable(dump) else response


def _no_choice(body: Mapping[str, Any]) -> str:
    error = body.get("error")
    message = error.get("message") if isinstance(error, Mapping) else error
    return "the response has no choice" + (f" ({message})" if message else "")


def _not_an_answer(choices: Any) -> Optional[str]:
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
        called = message.get("tool_calls") or message.get("function_call")
        if called or choice.get("finish_reason") in _TOOL_FINISH_REASONS:
            return "tool_calls"
        if message.get("refusal"):
            return "refusal"
        if choice.get("finish_reason") == "content_filter":
            return "content_filter"
    return None


def make_caller(
    client: Any,
    model_name: str,
    reasoning_effort: Optional[str] = None,
    *,
    tracker: Any | None = None,
    send_max_output_tokens: bool = False,
) -> DetailCallFn:
    """Return a DetailCallFn that makes one Chat Completions call (a fixed user turn when the prompt has none, as some
    chat templates drop a lone system turn; never a seed), keeping the reasoning; a tool call, refusal or filtered
    choice is an answer with no text that consumes an attempt, and a response with no choice a failed request."""
    seed_warning: Dict[str, bool] = {}

    def _call(
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_new_tokens: int,
        seed: int | None,
    ) -> Dict[str, Any]:
        _warn_seed_ignored_once(seed_warning, f"The {MODEL_TYPE} judge", seed)
        body = build_chat_body(system_prompt, user_prompt if user_prompt.strip() else EMPTY_USER_TURN,
                               model=model_name, temperature=temperature, reasoning_effort=reasoning_effort)
        if send_max_output_tokens:
            body["max_tokens"] = int(max_new_tokens)
        try:
            response = _body(client.chat.completions.create(**body))
            usage = response.get("usage")
            choices = response.get("choices")
            if not choices:
                raise CallFailed(_no_choice(response), usage=usage)
            if tracker is not None:
                tracker.add(TokenUsage.from_response(usage))
            withheld = _not_an_answer(choices)
            if withheld is not None:
                logger.warning("The %s judge %s gave no answer (%s)", MODEL_TYPE, model_name, withheld)
                return _answered(None, withheld, reasoning=chat_reasoning(response))
            return _answered(chat_text(response), chat_finish_reason(response), reasoning=chat_reasoning(response))
        except FATAL_PROVIDER_ERRORS:
            raise
        except Exception as exc:
            return _failed("OpenAI-compatible", model_name, exc, tracker)

    return _call
