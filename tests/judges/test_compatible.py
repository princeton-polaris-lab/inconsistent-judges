from __future__ import annotations

import gzip
import json
import logging
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpcore
import httpx
import pytest
from omegaconf import OmegaConf

from ruleforge.evaluate import Campaign, CampaignError, SamplingSettings
from ruleforge.evaluate import api
from ruleforge.cli.app import main
from ruleforge.cli.records import LOG_FNAME
from ruleforge.evaluate.runner import TransportExhausted
from ruleforge.core.credentials import CREDENTIAL_REFUSAL
from ruleforge.evaluate.validation import EMPTY_RESPONSE, TRANSPORT_ERROR, ComponentSchema
from ruleforge.judges import SHORTHANDS, JudgeError, JudgeSpec, check, compatible, models, resolve
from ruleforge.judges.backend import BUILTIN_MODEL_TYPES, SEED_UNSUPPORTED, ModelBackend
from ruleforge.judges.batch_api import BatchExecutionError, BatchSpendNotConfirmed, chat_finish_reason, chat_text
from ruleforge.judges.providers import PROVIDERS
import ruleforge.judges.base as judges_base
from tests.evaluate.helpers import inputs
from tests.support.hydra_runs import kept_logging
from tests.support.trees import tree_files

URL = "http://localhost:8000/v1"
# A server under the reserved .test domain: the wire fixture answers for it, and nothing real ever could.
WIRE = "http://judge.test/v1"
SRC = Path(__file__).resolve().parents[2] / "src"
KEY = "SENSITIVE_COMPATIBLE_TEST_KEY"
PROMPT = {"system_prompt": "Rule 1: be brief. Does the response break it? Answer yes, no or unsure.",
          "user_prompt": ""}
SCHEMA = ComponentSchema.from_rules([{"id": "Rule 1", "rule": "be brief"}], prompt_style="naive",
                                    forced_json_requirement="label_only")


@pytest.fixture(autouse=True)
def _kept_logging():
    """A keyed client raises httpcore's logger process-wide (``scrub_responses``); put back after each test."""
    with kept_logging():
        yield


def _config(**extra):
    return OmegaConf.create({"model_type": "openai_compatible", "model_name": "Qwen/Qwen3-8B", "base_url": URL,
                             "num_tries": 1, **extra})


def _answer(content="no", *, finish_reason="stop", usage=None, **message):
    return {"choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": content, **message}}],
            "usage": usage}


class _Tracker:
    def __init__(self):
        self.seen = []

    def add(self, usage):
        self.seen.append(usage)


def _client(*answers):
    client = MagicMock()
    client.sent = []

    def create(**body):
        client.sent.append(body)
        answer = answers[min(len(client.sent), len(answers)) - 1]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    client.chat.completions.create.side_effect = create
    return client


@pytest.fixture
def served(monkeypatch):
    """The judge's client is a recording stand-in; returns a function that sets what it answers."""
    clients = []

    def serve(*answers):
        client = _client(*answers)
        clients.append(client)
        monkeypatch.setattr(compatible, "init_client", lambda model_cfg: client)
        return client

    serve(_answer())
    return serve


@pytest.mark.parametrize("given, recorded", [
    ("http://localhost:8000/v1", "http://localhost:8000/v1"),
    ("  HTTP://LocalHost:8000/v1/  ", "http://localhost:8000/v1"),
    ("https://openrouter.ai:443/api/v1", "https://openrouter.ai/api/v1"),
    ("http://127.0.0.1:80/v1", "http://127.0.0.1/v1"),
    ("http://[::1]:80/v1", "http://[::1]/v1"),
    ("http://[::1]:8000", "http://[::1]:8000"),
    ("https://api.together.xyz/v1//", "https://api.together.xyz/v1"),
    ("http://host:8000/Path/V1", "http://host:8000/Path/V1"),
])
def test_a_base_url_is_recorded_in_one_normal_form(given, recorded):
    assert compatible.normalised_base_url(given) == recorded
    assert compatible.endpoint_url(OmegaConf.create({"base_url": given})) == recorded + "/chat/completions"


@pytest.mark.parametrize("given, match", [
    (None, "needs base_url"),
    ("   ", "needs base_url"),
    ("http://user:" + KEY + "@localhost:8000/v1", "holds a user or password"),
    ("http://localhost:8000/v1?key=" + KEY, "has a query or a fragment"),
    ("http://localhost:8000/v1#" + KEY, "has a query or a fragment"),
    ("localhost:8000/v1", "not an http:// or https:// address"),
    ("ftp://localhost/v1", "not an http:// or https:// address"),
    ("http:///v1", "not an http:// or https:// address"),
    (KEY, "not an http:// or https:// address"),
    ("http://localhost:port" + KEY + "/v1", "not a well-formed address"),
    ("http://[::1/v1", "not a well-formed address"),
    ("http://localhost:8000/v1/chat/completions/", "ends with /chat/completions"),
    ("http://localhost/a/../v1", "has a . or .. segment"),
    ("http://localhost/./v1", "has a . or .. segment"),
    ("http://localhost/v1/.", "has a . or .. segment"),
    ("http://b\u00fccher.example/v1", "carries only percent-encoded"),
    ("http://localhost/a b/v1", "carries only percent-encoded"),
    *((f"http://localhost/v1{character}x", "carries only percent-encoded") for character in '"<>\\^`{|}'),
])
def test_a_base_url_that_cannot_be_recorded_as_called_is_refused_without_repeating_it(given, match):
    with pytest.raises(JudgeError, match=match) as refused:
        compatible.normalised_base_url(given)
    assert KEY not in str(refused.value)


@pytest.mark.parametrize("name, model, url", [
    ("qwen3:8b@http://localhost:11434/v1", "qwen3:8b", "http://localhost:11434/v1"),
    ("@cf/meta/llama-3.1-8b@HTTPS://api.example.com/ai/v1", "@cf/meta/llama-3.1-8b", "HTTPS://api.example.com/ai/v1"),
    ("m@https://host.example/@team/v1", "m", "https://host.example/@team/v1"),
])
def test_the_shorthand_splits_at_the_last_at_sign_that_starts_a_url(name, model, url):
    assert compatible.split_shorthand(name) == (model, url)
    assert resolve(f"openai_compatible:{name}") == JudgeSpec.openai_compatible(model, url)


@pytest.mark.parametrize("name", ["qwen3", "qwen3@localhost:8000/v1", "@http://localhost:8000/v1"])
def test_a_shorthand_without_a_model_and_a_url_is_refused_with_an_example(name):
    with pytest.raises(JudgeError, match=r"openai_compatible:MODEL@URL names a model and its server's base URL"):
        resolve(f"openai_compatible:{name}")


def test_every_listed_shorthand_resolves_to_the_judge_type_it_names(tmp_path):
    filled = {"[:SEED]": ":3", "MODEL@URL": "m@" + URL, "MODEL": "m", "PATH": str(tmp_path / "weights")}
    for form in SHORTHANDS:
        name = form
        for placeholder, value in filled.items():
            name = name.replace(placeholder, value)
        spec = resolve(name)
        assert spec.model_type == form.partition(":")[0].partition("[")[0], form
    assert set(BUILTIN_MODEL_TYPES) - {form.partition(":")[0].partition("[")[0] for form in SHORTHANDS} == {
        "openai_batch", "anthropic_batch"}


def test_the_constructor_writes_the_config_a_yaml_file_would():
    spec = JudgeSpec.openai_compatible("Qwen/Qwen3-8B", URL, api_key_env=None, max_new_tokens=512)
    assert spec == JudgeSpec.from_config({"model_type": "openai_compatible", "model_name": "Qwen/Qwen3-8B",
                                          "base_url": URL, "api_key_env": None, "max_new_tokens": 512})
    with pytest.raises(TypeError, match=r"JudgeSpec.openai_compatible\(\) takes no option\(s\) \['stream'\]"):
        JudgeSpec.openai_compatible("m", URL, stream=True)


def test_the_identity_records_the_url_called_and_no_other_judge_s_identity_moves():
    found = JudgeSpec.openai_compatible("Qwen/Qwen3-8B", "HTTP://LOCALHOST:8000/v1/").identity()
    assert (found.model_type, found.model_name, found.endpoint, found.seed_capability, found.extension) == (
        "openai_compatible", "Qwen/Qwen3-8B", URL + "/chat/completions", SEED_UNSUPPORTED, None)
    assert found.send_max_output_tokens is False and found.reasoning_effort is None
    assert JudgeSpec.openai_compatible("m", URL, send_max_output_tokens=True,
                                       reasoning_level="low").identity().send_max_output_tokens is True
    other = JudgeSpec.openai_compatible("Qwen/Qwen3-8B", "http://localhost:8001/v1").identity()
    assert other != found
    assert JudgeSpec.openai("m").identity().endpoint is None and JudgeSpec.fake(1).identity().endpoint is None


def test_two_servers_are_two_conditions_and_a_root_refuses_the_other(tmp_path):
    scenarios, rules = inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios,
                               sampling=SamplingSettings(draws=2))
    first = campaign.condition("qwen/naive", rules=rules, judge=JudgeSpec.openai_compatible("qwen", URL))
    assert first.spec.endpoint == URL + "/chat/completions" and first.spec.judge_extension is None
    moved = JudgeSpec.openai_compatible("qwen", "http://localhost:8001/v1")
    with pytest.raises(CampaignError, match="already registers condition 'qwen/naive' with a different identity"):
        campaign.condition("qwen/naive", rules=rules, judge=moved)
    second = campaign.condition("qwen-8001/naive", rules=rules, judge=moved)
    assert second.spec.fingerprint != first.spec.fingerprint


def test_a_campaign_judged_by_a_compatible_server_keeps_its_reasoning(tmp_path, served):
    client = served(_answer("no", reasoning_content="Rule 1 is kept.", usage={"prompt_tokens": 7}))
    scenarios, rules = inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios,
                               sampling=SamplingSettings(draws=2))
    campaign.condition("qwen/naive", rules=rules, judge=JudgeSpec.openai_compatible("qwen", URL))
    with pytest.raises(CampaignError, match="spends money on every call"):
        campaign.run()
    assert client.sent == []
    campaign.run(confirm_spend=True)
    raw = [json.loads(line) for path in sorted((campaign.root / "qwen" / "naive").glob("draw-*/raw_outputs.jsonl"))
           for line in path.read_text().splitlines()]
    assert raw and {row["reasoning"] for row in raw} == {"Rule 1 is kept."}
    assert {body["model"] for body in client.sent} == {"qwen"} and all("seed" not in body for body in client.sent)


def test_the_validated_config_refuses_what_the_server_would_never_receive():
    for extra, match in (({"api_key": KEY}, "never the key; remove api_key"),
                         ({"reasoning_summary": "auto"}, "sends no reasoning_summary"),
                         ({"stream": True}, "sends no stream"),
                         ({"thinking_display": "summarized"}, "sends no thinking_display"),
                         ({"batch_endpoint": "chat_completions"}, "sends no batch request"),
                         ({"api_key_env": KEY + "-1"}, "never the key; set it to that name"),
                         ({"api_key_env": 5}, "never the key; set it to that name"),
                         ({"base_url": None}, "needs base_url")):
        with pytest.raises(JudgeError, match=match) as refused:
            JudgeSpec.from_config(_config(**extra))
        assert KEY not in str(refused.value)
    assert JudgeSpec.from_config(_config(stream=False, reasoning_summary=None)).model_type == "openai_compatible"


def test_the_key_variable_is_named_by_the_config_and_null_means_none():
    assert compatible.api_key_variable(_config()) == "OPENAI_API_KEY"
    assert compatible.api_key_variable(_config(api_key_env="OPENROUTER_API_KEY")) == "OPENROUTER_API_KEY"
    assert compatible.api_key_variable(_config(api_key_env=None)) is None


@pytest.fixture
def sdk(monkeypatch):
    """The openai SDK as the client sees it: each construction's arguments recorded."""
    built = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            built.append(kwargs)

    omit = object()

    def http_client(**kwargs):
        return SimpleNamespace(_transport=None, _mounts={})

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI, omit=omit,
                                                               DefaultHttpxClient=http_client))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    return SimpleNamespace(built=built, omit=omit, client=FakeOpenAI)


def test_the_client_is_keyed_from_the_named_variable_at_the_recorded_url_alone(sdk, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", f"  {KEY}  ")
    client = compatible.init_client(_config(api_key_env="OPENROUTER_API_KEY", base_url="HTTPS://Router.example/v1/"))
    assert isinstance(client, sdk.client)
    [built] = sdk.built
    built.pop("http_client")
    assert built == {"api_key": KEY, "base_url": "https://router.example/v1",
                     "default_headers": {"OpenAI-Organization": sdk.omit, "OpenAI-Project": sdk.omit}}


def test_a_server_that_takes_no_key_is_sent_none_even_with_an_openai_key_set(sdk, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    compatible.init_client(_config(api_key_env=None))
    [built] = sdk.built
    assert built["api_key"] == "" and built["base_url"] == URL


def test_an_httpx_whose_transports_cannot_be_wrapped_is_refused_naming_that_and_no_key(sdk, monkeypatch):
    monkeypatch.setitem(sys.modules["openai"].__dict__, "DefaultHttpxClient", lambda **kwargs: SimpleNamespace())
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    with pytest.raises(RuntimeError, match="cannot wrap them .*_transport and _mounts") as refused:
        compatible.init_client(_config())
    assert KEY not in str(refused.value)


@pytest.mark.parametrize("variable", ["OPENAI_API_KEY", "PASTEDKEY0123456789abcdef"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_missing_key_is_refused_before_any_client_without_repeating_the_variable_named(sdk, monkeypatch, variable,
                                                                                        value):
    if value is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match="the environment variable the judge's api_key_env names is not set.*"
                                         "api_key_env: null") as refused:
        compatible.init_client(_config(api_key_env=variable))
    assert variable not in str(refused.value) and sdk.built == []


def test_a_failed_client_never_puts_the_key_in_an_exception_or_a_log(monkeypatch, caplog):
    import openai

    called = []

    class BrokenOpenAI:
        def __init__(self, **kwargs):
            called.append(kwargs)
            raise RuntimeError(f"bad credentials: {kwargs!r}")

    def http_client(**kwargs):
        return openai.DefaultHttpxClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)), **kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=BrokenOpenAI, omit=openai.omit,
                                                               DefaultHttpxClient=http_client))
    with caplog.at_level(logging.DEBUG, logger="ruleforge.judges.compatible"):
        with pytest.raises(RuntimeError, match="Failed to initialize the OpenAI-compatible client") as refused:
            compatible.init_client(_config())
    assert [kwargs["api_key"] for kwargs in called] == [KEY]
    assert KEY not in str(refused.value) and KEY not in caplog.text
    assert refused.value.__cause__ is None and refused.value.__suppress_context__


def test_the_caller_sends_one_chat_request_without_a_seed_and_a_fixed_user_turn_when_there_is_none(caplog):
    client = _client(_answer())
    caller = compatible.make_caller(client, "qwen", "low")
    with caplog.at_level(logging.WARNING, logger="ruleforge.judges.models"):
        caller("system text", "", 0.7, 99, 5)
        caller("system text", "user text", 1.0, 99, 6)
    assert client.sent == [
        {"model": "qwen", "temperature": 0.7, "reasoning_effort": "low",
         "messages": [{"role": "system", "content": "system text"},
                      {"role": "user", "content": models.EMPTY_USER_TURN}]},
        {"model": "qwen", "temperature": 1.0, "reasoning_effort": "low",
         "messages": [{"role": "system", "content": "system text"}, {"role": "user", "content": "user text"}]},
    ]
    warned = [record.getMessage() for record in caplog.records if "no seed parameter" in record.getMessage()]
    assert len(warned) == 1 and "openai_compatible" in warned[0]


def test_the_caller_sends_the_ceiling_as_max_tokens_only_when_configured():
    client = _client(_answer())
    compatible.make_caller(client, "qwen", send_max_output_tokens=True)("s", "u", 1.0, 99, None)
    compatible.make_caller(client, "qwen")("s", "u", 1.0, 99, None)
    assert client.sent[0]["max_tokens"] == 99 and "max_tokens" not in client.sent[1]
    assert "reasoning_effort" not in client.sent[1] and "max_completion_tokens" not in client.sent[0]


@pytest.mark.parametrize("message, reasoning", [
    ({"reasoning_content": "thought"}, "thought"),
    ({"reasoning": "thought"}, "thought"),
    ({"reasoning_details": [{"type": "reasoning.text", "text": "thought"}]}, "thought"),
    ({}, None),
])
def test_the_reasoning_a_server_returns_beside_the_answer_is_kept(message, reasoning):
    tracker = _Tracker()
    usage = {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    detail = compatible.make_caller(_client(_answer(" no ", usage=usage, **message)), "qwen", tracker=tracker)(
        "s", "", 1.0, 99, None)
    assert (detail["completion"], detail["finish_reason"], detail["error"], detail.get("reasoning")) == (
        "no", "stop", None, reasoning)
    assert [(u.prompt_tokens, u.completion_tokens) for u in tracker.seen] == [(11, 5)]


def test_an_sdk_response_object_is_read_like_its_json():
    class Response:
        def model_dump(self):
            return _answer("yes", finish_reason="length", reasoning_content="r")

    detail = compatible.make_caller(_client(Response()), "qwen")("s", "", 1.0, 99, None)
    assert (detail["completion"], detail["finish_reason"], detail["reasoning"]) == ("yes", "length", "r")


@pytest.mark.parametrize("answer, error, counted", [
    ({"choices": [], "usage": {"prompt_tokens": 4}}, "CallFailed: the response has no choice", [4]),
    ({"error": {"message": "model not loaded"}}, "CallFailed: the response has no choice (model not loaded)", [0]),
    ({"error": "overloaded"}, "CallFailed: the response has no choice (overloaded)", [0]),
    (RuntimeError("connection refused"), "RuntimeError: connection refused", []),
])
def test_a_call_that_returned_no_answer_is_this_request_s_error(answer, error, counted):
    tracker = _Tracker()
    detail = compatible.make_caller(_client(answer), "qwen", tracker=tracker)("s", "", 1.0, 99, None)
    assert (detail["completion"], detail["error"]) == ("", error)
    assert [u.prompt_tokens for u in tracker.seen] == counted


@pytest.mark.parametrize("error", [BatchSpendNotConfirmed("confirm"), BatchExecutionError("unreadable")])
def test_a_fatal_provider_error_propagates(error):
    with pytest.raises(type(error)):
        compatible.make_caller(_client(error), "qwen")("s", "", 1.0, 99, None)


def test_the_started_judge_carries_each_failure_on_its_own_request(served):
    client = served(_answer("no"), RuntimeError("server busy"), _answer("yes", reasoning="r"))
    judge = JudgeSpec.openai_compatible("qwen", URL).start()
    assert isinstance(judge, ModelBackend) and judge.capabilities.per_request_errors
    first, second, third = judge.generate_once([dict(PROMPT)] * 3, temperature=0.5, seeds=[1, 2, 3])
    assert (first.text, second.error, third.text, third.reasoning) == ("no", "RuntimeError: server busy", "yes", "r")
    assert not second.block_failed and SCHEMA.classify("full_ruleset", second).failure_reason == TRANSPORT_ERROR
    assert [body["temperature"] for body in client.sent] == [0.5] * 3 and judge.usage.calls == 2
    assert judge._base_delay == 1.0 and judge.seed_capability == SEED_UNSUPPORTED


def test_the_backend_builds_the_caller_from_the_config(monkeypatch):
    seen = {}

    def make_caller(client, model_name, reasoning_effort, **kwargs):
        seen.update(client=client, model_name=model_name, reasoning_effort=reasoning_effort, **kwargs)
        return lambda *args: {"raw_response": "no", "completion": "no", "finish_reason": "stop", "error": None}

    monkeypatch.setattr(compatible, "init_client", lambda model_cfg: ("client", model_cfg.base_url))
    monkeypatch.setattr(compatible, "make_caller", make_caller)
    backend = ModelBackend(_config(reasoning_level="high", send_max_output_tokens=True))
    assert backend.generate([dict(PROMPT)]) == [["no"]]
    assert seen == {"client": ("client", URL), "model_name": "Qwen/Qwen3-8B", "reasoning_effort": "high",
                    "tracker": backend.token_tracker, "send_max_output_tokens": True}


def test_a_keyed_server_is_a_paid_built_in_judge_whose_probe_needs_the_spend_confirmed(monkeypatch):
    assert PROVIDERS["openai_compatible"].builtin and "openai_compatible" in BUILTIN_MODEL_TYPES
    monkeypatch.setattr(compatible, "init_client", lambda model_cfg: pytest.fail("a client was built"))
    report = check(f"openai_compatible:qwen@{URL}")
    assert report.capabilities.spends_money and not report.capabilities.batch and report.probe is None
    with pytest.raises(BatchSpendNotConfirmed):
        check(f"openai_compatible:qwen@{URL}", probe=True)


def _judge_file(directory, key_env):
    """A compatible judge's YAML config, keyed from the variable ``key_env`` names, or keyless when it is None."""
    path = directory / f"{key_env or 'keyless'}.yaml"
    path.write_text(f"model_type: openai_compatible\nmodel_name: qwen\nbase_url: {URL}\n"
                    f"api_key_env: {key_env or 'null'}\n")
    return path


KEYED = {"keyless": None, "keyed": "JUDGE_TEST_KEY"}


def test_a_keyless_server_spends_nothing_and_is_the_same_judge_as_a_keyed_one(tmp_path, served):
    client = served(_answer("no"))
    keyless = JudgeSpec.openai_compatible("qwen", URL, api_key_env=None)
    keyed = JudgeSpec.openai_compatible("qwen", URL, api_key_env="JUDGE_TEST_KEY")
    assert keyed.capabilities().spends_money and JudgeSpec.openai_compatible("qwen", URL).capabilities().spends_money
    assert keyless.capabilities() == replace(keyed.capabilities(), spends_money=False)
    assert keyless.start().capabilities == keyless.capabilities()
    assert keyless.identity() == keyed.identity() == JudgeSpec.openai_compatible("qwen", URL).identity()
    scenarios, rules = inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios,
                               sampling=SamplingSettings(draws=2))
    first = campaign.condition("qwen/naive", rules=rules, judge=keyed)
    with pytest.raises(CampaignError, match="spends money on every call"):
        campaign.run()
    assert client.sent == []
    assert campaign.condition("qwen/naive", rules=rules, judge=keyless).spec.fingerprint == first.spec.fingerprint
    campaign.run()
    records = [path for path in tree_files(campaign.root) if "code_snapshots" not in path.parts]
    assert client.sent and records and not [path for path in records if b"spends_money" in _content(path)]


@pytest.mark.parametrize("key_env", KEYED.values(), ids=KEYED)
def test_evaluate_runs_a_keyless_server_unconfirmed_and_refuses_a_keyed_one_before_writing(tmp_path, served, capsys,
                                                                                           key_env):
    client = served(_answer("no"))
    scenarios, rules = inputs(tmp_path)
    status = main(["evaluate", "--rules", str(rules.path), "--scenarios", str(scenarios.path), "--judge",
                   str(_judge_file(tmp_path, key_env)), "--draws", "2", "--out", str(tmp_path / "root")])
    error = capsys.readouterr().err
    if key_env is None:
        assert status == 0 and client.sent, error
    else:
        assert status == 2 and "spends money on every call; pass confirm_spend=True (--yes-spend)" in error
        assert client.sent == [] and not (tmp_path / "root").exists()


@pytest.mark.parametrize("key_env", KEYED.values(), ids=KEYED)
def test_campaign_run_judges_with_a_keyless_server_unconfirmed_and_refuses_a_keyed_one_before_writing(
        tmp_path, served, capsys, key_env):
    client = served(_answer("no"))
    scenarios, rules = inputs(tmp_path)
    root = tmp_path / "root"
    shared = ["--scenarios", str(scenarios.path), "--rules", str(rules.path), "--judge",
              str(_judge_file(tmp_path, key_env))]
    assert main(["campaign", "create", str(root), "--scenarios", str(scenarios.path), "--draws", "2"]) == 0
    assert main(["campaign", "add", str(root), *shared, "--condition-id", "qwen/naive"]) == 0
    before = {path: path.read_bytes() for path in tree_files(root)}
    status = main(["campaign", "run", str(root), "qwen/naive", *shared])
    error = capsys.readouterr().err
    if key_env is None:
        assert status == 0 and client.sent, error
    else:
        assert status == 2 and "spends money on every call" in error and client.sent == []
        assert {path: path.read_bytes() for path in tree_files(root)} == before


def test_judges_check_probes_a_keyless_server_by_default_and_a_keyed_one_only_when_confirmed(tmp_path, served,
                                                                                               capsys):
    client = served(_answer("no"))

    def checked(*argv):
        status = main(["judges", "check", *map(str, argv)])
        printed = capsys.readouterr()
        return status, json.loads(printed.out) if printed.out else None, printed.err

    status, report, _ = checked(_judge_file(tmp_path, None))
    assert status == 0 and report["probe"]["text"] == "no" and not report["capabilities"]["spends_money"]
    keyed = _judge_file(tmp_path, "JUDGE_TEST_KEY")
    status, report, _ = checked(keyed)
    assert status == 0 and report["probe"] is None and report["capabilities"]["spends_money"]
    status, _, error = checked(keyed, "--probe")
    assert status == 2 and "pass --yes-spend" in error and len(client.sent) == 1
    status, report, _ = checked(keyed, "--probe", "--yes-spend")
    assert status == 0 and report["probe"]["text"] == "no" and len(client.sent) == 2


def _scripted_client(answer):
    """Return scripted Chat Completions answers, using an empty user turn for system-only prompts."""
    client = MagicMock()

    def create(**body):
        turns = {message["role"]: message["content"] for message in body["messages"]}
        user = turns.get("user", "")
        return _answer(answer(turns.get("system", ""), "" if user == models.EMPTY_USER_TURN else user))

    client.chat.completions.create.side_effect = create
    return client


@pytest.mark.parametrize("key_env", KEYED.values(), ids=KEYED)
def test_optimize_runs_a_keyless_server_unconfirmed_and_refuses_a_keyed_one_before_writing(tmp_path, monkeypatch,
                                                                                           capsys, key_env):
    from tests.support.scripted import RuleForgeScript

    script = RuleForgeScript()
    client = _scripted_client(script)
    monkeypatch.setattr(compatible, "init_client", lambda model_cfg: client)
    monkeypatch.setattr("ruleforge.judges.backend.time.sleep", lambda seconds: None)  # the script's retries at once
    monkeypatch.chdir(tmp_path)
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "optimize"
    status = main(["optimize", "--rules", str(fixtures / "rules.jsonl"), "--judge", str(_judge_file(tmp_path, key_env)),
                   "--out", "runs", "--seeds-good", str(fixtures / "good_seeds.jsonl"), "--seeds-bad",
                   str(fixtures / "bad_seeds.jsonl"), "--episodes", "0"])
    error = capsys.readouterr().err
    if key_env is None:
        assert status == 0 and client.chat.completions.create.called and script.unexpected == [], error
    else:
        assert status == 2 and "pass --yes-spend" in error
        assert not client.chat.completions.create.called and not (tmp_path / "runs").exists()


@pytest.mark.parametrize("key_env", KEYED.values(), ids=KEYED)
def test_the_launched_optimize_program_lets_a_keyless_server_run_unconfirmed(key_env):
    from ruleforge.launch.programs.optimize import check_spend

    judge = {"model_type": "openai_compatible", "model_name": "qwen", "base_url": URL, "api_key_env": key_env}
    cfg = OmegaConf.create({"judge_model": judge, "agent_model": judge})
    if key_env is None:
        check_spend(cfg)
    else:
        with pytest.raises(BatchSpendNotConfirmed, match="pass --yes-spend"):
            check_spend(cfg)
    check_spend(OmegaConf.merge(cfg, {"confirm_spend": True}))


@pytest.mark.parametrize("model_type", ["openai", "openai_batch", "anthropic", "anthropic_batch"])
def test_every_other_api_judge_still_refuses_another_server_and_names_this_type(model_type, tmp_path):
    config = {"model_type": model_type, "model_name": "m", "base_url": URL, "send_max_output_tokens": True,
              "batch_root": str(tmp_path)}
    with pytest.raises(JudgeError, match=r"takes no base_url.*model_type 'openai_compatible' and its base_url"):
        JudgeSpec.from_config(config)


def test_an_ambient_redirect_is_still_refused_naming_this_type(monkeypatch):
    from ruleforge.judges import anthropic_batch, openai_batch

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", URL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", URL)
    for build in (models.init_openai_client, openai_batch.default_client, anthropic_batch.default_client):
        with pytest.raises(ValueError, match=r"_BASE_URL set: .*model_type 'openai_compatible' and its base_url"):
            build()


COMPLETION = {"id": "c", "object": "chat.completion", "created": 1, "model": "qwen",
              "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "no"}}]}


@pytest.fixture
def wire(monkeypatch):
    """The real openai SDK over an ``httpx.MockTransport`` replying by ``wire.routes``, else ``wire.fallback``."""
    import openai

    state = SimpleNamespace(routes={}, seen=[],
                            fallback=lambda request: httpx.Response(404, json={"error": {"message": "no route"}}))

    def handle(request):
        state.seen.append(request)
        return state.routes.get(str(request.url), state.fallback)(request)

    real = openai.DefaultHttpxClient
    monkeypatch.setattr(openai, "DefaultHttpxClient",
                        lambda **kwargs: real(transport=httpx.MockTransport(handle), **kwargs))
    return state


def test_a_redirect_is_this_request_s_failure_and_is_never_followed(wire):
    wire.routes[WIRE + "/chat/completions"] = lambda request: httpx.Response(
        307, headers={"location": "http://other.test/v1/chat/completions"})
    wire.routes["http://other.test/v1/chat/completions"] = lambda request: httpx.Response(200, json=COMPLETION)
    judge = JudgeSpec.openai_compatible("qwen", WIRE, api_key_env=None).start()
    [outcome] = judge.generate_once([dict(PROMPT)], temperature=1.0)
    assert outcome.text is None and "307" in outcome.error and not outcome.block_failed
    assert [str(request.url) for request in wire.seen] == [WIRE + "/chat/completions"]


def _reply(choice=None, **fields):
    return {**COMPLETION, "choices": [{"index": 0, "finish_reason": "stop",
                                       "message": {"role": "assistant", "content": "no"}, **(choice or {})}], **fields}


def _message(**message):
    return {"message": {"role": "assistant", "content": "no", **message}}


#: A server echoing the key it was sent, in each place a response can carry it.
ECHOES = {
    "401 body": lambda key: httpx.Response(401, json={"error": {"message": f"bad key {key}"}}),
    "x-echo-key": lambda key: httpx.Response(401, headers={"x-echo-key": key}, json={"error": {"message": "no"}}),
    "WWW-Authenticate": lambda key: httpx.Response(401, headers={"WWW-Authenticate": f'Bearer error="{key}"'},
                                                   json={"error": {"message": "no"}}),
    "reason phrase": lambda key: httpx.Response(401, extensions={"reason_phrase": f"Bad {key}".encode()},
                                                json={"error": {"message": "no"}}),
    "content": lambda key: httpx.Response(200, json=_reply(_message(content=f"{key}\nno"))),
    "reasoning": lambda key: httpx.Response(200, json=_reply(_message(reasoning_content=key))),
    "finish_reason": lambda key: httpx.Response(200, json=_reply({"finish_reason": key})),
    "created": lambda key: httpx.Response(200, json=_reply(created=key)),
    "gzip body": lambda key: httpx.Response(200, headers={"content-encoding": "gzip"}, content=gzip.compress(
        json.dumps(_reply(_message(content=f"{key}\nno"))).encode())),
}


def _content(path):
    """A written file's bytes, a gzip file's decompressed."""
    data = path.read_bytes()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def _run_through_the_cli(tmp_path, base_url, *flags):
    scenarios, rules = inputs(tmp_path)
    judge, root = tmp_path / "judge.yaml", tmp_path / "runs" / "root"
    judge.write_text(f"model_type: openai_compatible\nmodel_name: qwen\nbase_url: {base_url}\n"
                     "api_key_env: JUDGE_TEST_KEY\n")
    shared = ["--scenarios", str(scenarios.path), "--rules", str(rules.path), "--judge", str(judge)]
    assert main(["campaign", "create", str(root), "--scenarios", str(scenarios.path), "--draws", "2"]) == 0
    assert main(["campaign", "add", str(root), *shared, "--condition-id", "qwen/naive"]) == 0
    return main([*flags, "campaign", "run", str(root), "qwen/naive", *shared, "--yes-spend"])


@pytest.mark.parametrize("echo", sorted(ECHOES))
def test_a_key_the_server_echoes_reaches_no_outcome_file_or_output(wire, monkeypatch, tmp_path, capsys, echo):
    monkeypatch.setenv("JUDGE_TEST_KEY", KEY)
    wire.fallback = lambda request: ECHOES[echo](request.headers["authorization"].removeprefix("Bearer "))
    [outcome] = JudgeSpec.openai_compatible("qwen", WIRE, api_key_env="JUDGE_TEST_KEY").start().generate_once(
        [dict(PROMPT)], temperature=1.0)
    _run_through_the_cli(tmp_path, WIRE)
    printed = capsys.readouterr()
    assert wire.seen[0].headers["authorization"] == f"Bearer {KEY}"
    written = {str(path): _content(path) for path in tree_files(tmp_path / "runs")}
    assert len(written) > 10 and not [path for path, data in written.items() if KEY.encode() in data]
    assert KEY not in repr(outcome) + printed.out + printed.err
    seen = repr(outcome).encode() + printed.err.encode() + b"".join(written.values())
    assert compatible.SCRUBBED_KEY in seen or echo == "created"  # created is read into no outcome or record


def test_a_scrubbed_response_keeps_its_status_headers_and_reason_and_a_short_key_is_not_scrubbed(wire, monkeypatch):
    monkeypatch.setenv("JUDGE_TEST_KEY", "EMPTY")
    wire.fallback = lambda request: httpx.Response(200, json=_reply(_message(content="EMPTY\nno")))
    [outcome] = JudgeSpec.openai_compatible("qwen", WIRE, api_key_env="JUDGE_TEST_KEY").start().generate_once(
        [dict(PROMPT)], temperature=1.0)
    assert outcome.text == "EMPTY\nno" and wire.seen[0].headers["authorization"] == "Bearer EMPTY"
    inner = httpx.MockTransport(lambda request: httpx.Response(
        418, headers={"x-kept": "1", KEY: KEY}, extensions={"reason_phrase": b"Teapot"}, content=b"tea"))
    scrubbing = compatible.KeyScrubbingTransport(inner, KEY)
    response = scrubbing.handle_request(httpx.Request("POST", WIRE))
    assert (response.status_code, response.reason_phrase, response.headers["x-kept"], response.read()) == (
        418, "Teapot", "1", b"tea")
    assert response.headers["<api key>"] == "<api key>"
    scrubbing.close()


class _Network(httpcore.MockBackend):
    """httpcore's in-memory network under httpx's transport: each connection reads ``reply``; opened hosts are kept."""

    def __init__(self, reply: bytes) -> None:
        super().__init__([reply])
        self.hosts: list[str] = []

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.hosts.append(f"{host}:{port}")
        return super().connect_tcp(host, port, timeout, local_address, socket_options)


@pytest.fixture
def stack(monkeypatch):
    """httpx's and httpcore's real HTTP/1.1 stack over an in-memory network; ``stack.serve(reply)`` sets the reply."""
    state = SimpleNamespace(network=None)

    def serve(reply: bytes) -> None:
        state.network = _Network(reply)

    for name in ("ConnectionPool", "HTTPProxy"):
        real = getattr(httpcore, name)
        monkeypatch.setattr(httpcore, name,
                            lambda *args, real=real, **kwargs: real(*args, network_backend=state.network, **kwargs))
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    logging.getLogger("httpcore").setLevel(logging.NOTSET)  # its level in a fresh process; put back after the test
    state.serve = serve
    return state


PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy",
                   "no_proxy")


def _http(status: str, *headers: str, body: dict) -> bytes:
    """One HTTP/1.1 response that closes its connection, so the next request reads it again."""
    data = json.dumps(body).encode()
    head = [f"HTTP/1.1 {status}", "Content-Type: application/json", *headers, f"Content-Length: {len(data)}",
            "Connection: close"]
    return ("\r\n".join(head) + "\r\n\r\n").encode() + data


@pytest.mark.parametrize("flags", [(), ("--verbose",)], ids=["run", "verbose run"])
def test_a_key_echoed_on_the_wire_is_in_no_file_or_output_of_a_run_over_the_real_http_stack(stack, monkeypatch,
                                                                                           tmp_path, capsys, flags):
    monkeypatch.setenv("JUDGE_TEST_KEY", KEY)
    stack.serve(_http(f"401 Bad {KEY}", f"x-echo-key: {KEY}", f'WWW-Authenticate: Bearer error="{KEY}"',
                      body={"error": {"message": "unauthorized"}}))
    if flags:
        with pytest.raises(TransportExhausted):
            _run_through_the_cli(tmp_path, WIRE, *flags)
    else:
        assert _run_through_the_cli(tmp_path, WIRE) == 1
    printed = capsys.readouterr()
    written = {str(path): _content(path) for path in tree_files(tmp_path / "runs")}
    assert not [path for path, data in written.items() if KEY.encode() in data] and KEY not in printed.out + printed.err
    logs = [data for path, data in written.items() if path.endswith(LOG_FNAME)]
    assert stack.network.hosts and logs
    # Neither the SDK's trace of the response nor httpcore's is written anywhere.
    assert not any(b"x-echo-key" in data or b"receive_response_headers" in data for data in logs)


#: Each route a request may take: the client's own transport, a proxy's, or its own again for a host NO_PROXY names.
ROUTES = {"direct": ({}, "judge.test:80"), "proxy": ({"HTTP_PROXY": "http://proxy.test:3128"}, "proxy.test:3128"),
          "bypass": ({"HTTP_PROXY": "http://proxy.test:3128", "NO_PROXY": "judge.test"}, "judge.test:80")}


@pytest.mark.parametrize("key, shown", [(KEY, "<api key>"), ("EMPTY", "EMPTY")], ids=["key", "placeholder"])
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_every_route_hands_on_a_scrubbed_reply_and_a_short_placeholder_as_it_came(stack, monkeypatch, route, key,
                                                                                  shown):
    variables, connected = ROUTES[route]
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("JUDGE_TEST_KEY", key)
    stack.serve(_http(f"200 OK {key}", f"x-echo-key: {key}", body=_reply(_message(content=f"{key}\nno"))))
    [outcome] = JudgeSpec.openai_compatible("qwen", WIRE, api_key_env="JUDGE_TEST_KEY").start().generate_once(
        [dict(PROMPT)], temperature=1.0)
    assert outcome.text == f"{shown}\nno" and stack.network.hosts == [connected]


#: A judge's subprocess: the real SDK, with OPENAI_LOG=debug, over a transport answering ``sys.argv[1]``'s echo of the
#: key: a 401 (its header, reason phrase and body), or a 200 whose ``created`` pydantic warns about on stderr.
DEBUG_LOGGED = """
import sys, httpx, openai
real = openai.DefaultHttpxClient
def handle(request):
    key = request.headers["authorization"].removeprefix("Bearer ")
    if sys.argv[1] == "created":
        return httpx.Response(200, json={"id": "c", "object": "chat.completion", "created": key, "model": "qwen",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "no"}}]})
    return httpx.Response(401, headers={"x-echo-key": key}, extensions={"reason_phrase": f"Bad {key}".encode()},
                          json={"error": {"message": f"bad key {key}"}})
openai.DefaultHttpxClient = lambda **kwargs: real(transport=httpx.MockTransport(handle), **kwargs)
from ruleforge.cli import main
sys.exit(main(sys.argv[2:]))
"""


@pytest.mark.parametrize("echo, status, shown", [("401", 1, "HTTP Response"), ("created", 0, "PydanticSerialization")])
def test_a_fresh_process_logging_the_sdk_at_debug_prints_no_echoed_key(tmp_path, echo, status, shown):
    judge = tmp_path / "judge.yaml"
    judge.write_text(f"model_type: openai_compatible\nmodel_name: qwen\nbase_url: {WIRE}\napi_key_env: JUDGE_TEST_KEY\n")
    env = {**os.environ, "OPENAI_LOG": "debug", "JUDGE_TEST_KEY": KEY,
           "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run([sys.executable, "-c", DEBUG_LOGGED, echo, "judges", "check", str(judge), "--probe",
                             "--yes-spend"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == status and shown in result.stderr and "<api key>" in result.stderr
    assert KEY not in result.stdout + result.stderr


@pytest.mark.parametrize("message, finish_reason", [
    ({"tool_calls": [{"id": "t", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}, "tool_calls"),
    ({"refusal": "I will not judge this."}, "stop"),
])
def test_a_choice_calling_a_tool_or_refusing_is_an_answer_with_no_text_whatever_its_content(wire, message,
                                                                                            finish_reason):
    choice = {"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": "yes", **message}}
    wire.fallback = lambda request: httpx.Response(200, json={**COMPLETION, "choices": [choice]})
    [outcome] = JudgeSpec.openai_compatible("qwen", WIRE, api_key_env=None).start().generate_once(
        [dict(PROMPT)], temperature=1.0)
    assert outcome.text is None and outcome.error is None and outcome.returned
    assert outcome.finish_reason == ("tool_calls" if "tool_calls" in message else "refusal")
    # It consumes an attempt, as an OpenAI or Anthropic refusal does.
    assert SCHEMA.classify("full_ruleset", outcome).failure_reason == EMPTY_RESPONSE
    assert (chat_text({"choices": [choice]}), chat_finish_reason({"choices": [choice]})) == ("yes", finish_reason)


@pytest.mark.parametrize("given", [
    "HTTP://Judge.TEST:08000/V1/",
    "http://judge.test:/v1",
    "http://[::FFFF:7f00:1]:80/v1",
    "https://judge.test:443/a%2Fb/@t/v1;x",
    "http://judge.test./v1/..x/~u",
    "http://judge.test/v1[x]!$&'()*+,=",
])
def test_the_recorded_endpoint_is_the_url_the_request_goes_to(wire, given):
    wire.fallback = lambda request: httpx.Response(200, json=COMPLETION)
    spec = JudgeSpec.openai_compatible("qwen", given, api_key_env=None)
    [outcome] = spec.start().generate_once([dict(PROMPT)], temperature=1.0)
    assert outcome.text == "no" and [str(request.url) for request in wire.seen] == [spec.identity().endpoint]


def test_the_resume_command_names_only_a_judge_that_rebuilds_what_is_sent(tmp_path, monkeypatch):
    place = tmp_path / "model"
    place.mkdir()
    (place / "m.yaml").write_text(f"model_type: openai_compatible\nmodel_name: m\nbase_url: {URL}\n")
    monkeypatch.setattr(judges_base, "MODEL_CONFIG_DIRS", (place,))
    keyless = JudgeSpec.openai_compatible("m", URL, api_key_env=None)
    assert keyless.identity() == resolve("m").identity() and keyless != resolve("m")
    assert api._judge_flag(None, keyless) is None
    assert api._judge_flag(None, resolve("m")) == f"openai_compatible:m@{URL}"
    assert api._judge_flag(None, JudgeSpec.openai("m", reasoning_level="high")) is None
    odd = JudgeSpec.openai_compatible("m", "http://127.0.0.1:18001/v1/@http:///")
    assert api._judge_flag(None, odd) is None


def test_evaluate_with_a_judge_no_shorthand_rebuilds_reaches_the_run(tmp_path):
    scenarios, rules = inputs(tmp_path)
    odd = JudgeSpec.openai_compatible("m", "http://127.0.0.1:18001/v1/@http:///")
    with pytest.raises(CampaignError, match="spends money on every call"):
        api.evaluate(rules, scenarios, odd, out=tmp_path / "root", draws=2)


@pytest.mark.parametrize("choices, finish_reason", [
    ([{"message": {"content": "no", "function_call": {"name": "f"}}}], "tool_calls"),
    ([{"finish_reason": "tool_calls", "message": {"content": "no"}}], "tool_calls"),
    ([{"message": {"content": "no"}}, {"message": {"content": "no", "refusal": "not this"}}], "refusal"),
    ([{"finish_reason": "content_filter", "message": {"content": "no"}}], "content_filter"),
])
def test_any_choice_calling_a_tool_or_refusing_is_an_answer_with_no_text(choices, finish_reason):
    tracker = _Tracker()
    answer = {"choices": choices, "usage": {"prompt_tokens": 4}}
    detail = compatible.make_caller(_client(answer), "qwen", tracker=tracker)("s", "", 1.0, 99, None)
    assert (detail["completion"], detail["error"], detail["finish_reason"]) == ("", None, finish_reason)
    assert [u.prompt_tokens for u in tracker.seen] == [4]


@pytest.mark.parametrize("choices", [["no"], [{"message": "no"}]])
def test_a_choice_with_no_message_is_an_empty_answer_not_a_failure(choices):
    detail = compatible.make_caller(_client({"choices": choices}), "qwen")("s", "", 1.0, 99, None)
    assert (detail["completion"], detail["error"]) == ("", None)


@pytest.mark.parametrize("name", [f"http://user:{KEY}@judge.test/judge.yaml",
                                  f"openai-compatible:m@http://user:{KEY}@judge.test/v1",
                                  f"openai_compatible:m@http://judge.test/v1?api_key={KEY}"])
def test_a_judge_named_with_a_credential_is_refused_before_it_is_read_and_not_repeated(name):
    for read in (resolve, JudgeSpec.from_config):
        with pytest.raises(JudgeError) as refused:
            read(name)
        assert str(refused.value) == CREDENTIAL_REFUSAL
