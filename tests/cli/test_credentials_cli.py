from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ruleforge.core.credentials import CREDENTIAL_REFUSAL
from ruleforge.launch import cli as launch_cli
from ruleforge.launch.programs import run_program

SRC = Path(__file__).resolve().parents[2] / "src"

SECRET = "SENSITIVE_CLI_TEST_SECRET"
FORMS = {"userinfo": f"http://user:{SECRET}@judge.test/v1", "query": f"http://judge.test/v1?api_key={SECRET}"}
#: Each command a --judge reaches, the URL standing where the judge's base URL would.
COMMANDS = {
    "judges show": ["judges", "show", "openai_compatible:m@{url}"],
    "judges check": ["judges", "check", "openai_compatible:m@{url}", "--probe"],
    "evaluate": ["evaluate", "--rules", "r.jsonl", "--scenarios", "s.jsonl", "--judge", "openai_compatible:m@{url}"],
    "campaign add": ["campaign", "add", "root", "--scenarios", "s.jsonl", "--rules", "r.jsonl", "--judge",
                     "openai_compatible:m@{url}"],
    "bank run": ["bank", "run", "--config", "c.yaml", "--judge", "openai_compatible:m@{url}"],
    "optimize": ["optimize", "--rules", "r.jsonl", "--out", "o", "--judge", "openai_compatible:m@{url}"],
    "sweep with a duplicate": ["evaluate", "--rules", "r.jsonl", "--scenarios", "s.jsonl", "--sweep",
                               "judge=openai_compatible:m@{url},openai_compatible:m@{url}"],
    "sweep with a trailing comma": ["evaluate", "--rules", "r.jsonl", "--scenarios", "s.jsonl", "--sweep",
                                    "judge=fake,openai_compatible:m@{url},"],
    ".yaml suffix": ["judges", "show", "{url}/judge.yaml"],
    ".yml suffix": ["evaluate", "--rules", "r.jsonl", "--scenarios", "s.jsonl", "--judge", "{url}/judge.yml"],
    "override": ["evaluate", "--rules", "r.jsonl", "--scenarios", "s.jsonl", "--judge", "fake", "--",
                 "judge_model.base_url={url}"],
}


BUILT_IN = {
    "openai": {"model_name": "m"}, "openai_batch": {"model_name": "m"}, "anthropic": {"model_name": "m"},
    "anthropic_batch": {"model_name": "m"}, "fake": {"model_name": "fake-1", "seed": 1},
    "local": {"model_name": "m", "model_path": "weights"},
    "openai_compatible": {"model_name": "m", "base_url": "http://judge.test/v1"},
}


@pytest.mark.parametrize("key", ["base_url", "batch_endpoint"])
@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("model_type", sorted(BUILT_IN))
def test_a_credential_in_a_judge_s_config_is_refused_before_any_provider_reads_it(cli, tmp_path, model_type, form,
                                                                                  key):
    from omegaconf import OmegaConf

    from ruleforge.judges import JudgeError, JudgeSpec, resolve
    from ruleforge.judges.backend import ModelBackend

    config = {"model_type": model_type, **BUILT_IN[model_type], key: FORMS[form]}
    for read in (resolve, JudgeSpec.from_config, lambda given: ModelBackend(OmegaConf.create(given))):
        with pytest.raises(JudgeError) as refused:
            read(config)
        assert str(refused.value) == CREDENTIAL_REFUSAL
    judge = tmp_path / "judge.yaml"
    judge.write_text("".join(f"{name}: {value}\n" for name, value in config.items()))
    status, out, error = cli("judges", "show", str(judge))
    assert status == 2 and CREDENTIAL_REFUSAL in error and SECRET not in error + str(out)


#: Free text holding a URL without a credential, which the check neither covers nor would refuse.
FREE_TEXT = {"chat_template": "Read https://example.org for context. Is the answer correct? {{ messages }}",
             "run_name": "see https://example.org or write me@example.org"}


def test_free_text_holding_a_url_is_taken_as_it_was():
    from ruleforge.core.credentials import carries_credential
    from ruleforge.judges import JudgeSpec

    assert not any(map(carries_credential, FREE_TEXT.values()))
    assert JudgeSpec.local("weights", engine_args=FREE_TEXT).options["engine_args"] == FREE_TEXT


def test_a_credential_inside_a_longer_argument_still_ends_the_command(cli, tmp_path):
    before = _files(tmp_path)
    status, out, error = cli("judges", "show", f"fake --see {FORMS['userinfo']} later")
    assert status == 2 and error == f"ruleforge: {CREDENTIAL_REFUSAL}\n" and out == "" and _files(tmp_path) == before


def _files(tmp_path):
    return sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_a_credential_in_any_argument_ends_the_command_before_anything_reads_or_writes_it(cli, tmp_path, command,
                                                                                         form):
    before = _files(tmp_path)
    status, out, error = cli(*(word.format(url=FORMS[form]) for word in COMMANDS[command]))
    assert status == 2 and error == f"ruleforge: {CREDENTIAL_REFUSAL}\n" and out == ""
    assert _files(tmp_path) == before



#: The entry points a judge reaches without the CLI's main, as ``python -m`` runs them.
ENTRY_POINTS = {
    "launch program": ["-m", "ruleforge.launch.programs.run_campaign", "judge_model.base_url={url}"],
    "launch cli": ["-m", "ruleforge.launch.cli", "job", "--backend", "local", "--dry-run", "--", "ruleforge",
                   "evaluate", "--judge", "openai_compatible:m@{url}"],
}


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
def test_the_launch_entry_points_refuse_a_credential_before_hydra_or_a_record_sees_it(tmp_path, entry, form):
    before = _files(tmp_path)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run([sys.executable, *(word.format(url=FORMS[form]) for word in ENTRY_POINTS[entry])],
                            cwd=Path.cwd(), env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 2 and CREDENTIAL_REFUSAL in result.stderr
    assert SECRET not in result.stdout + result.stderr and _files(tmp_path) == before


def test_the_launch_entry_points_refuse_it_when_called_in_process_too(monkeypatch, capsys):
    assert launch_cli.main(["job", "--", "ruleforge", "evaluate", "--judge", f"m@{FORMS['query']}"]) == 2
    monkeypatch.setattr(sys, "argv", ["run_campaign.py", f"judge_model.base_url={FORMS['userinfo']}"])
    with pytest.raises(SystemExit) as exited:
        run_program(lambda cfg: None, "sampling_campaign", check=lambda cfg, **_: None)
    error = capsys.readouterr().err
    assert exited.value.code == 2 and error.count(CREDENTIAL_REFUSAL) == 2 and SECRET not in error


#: A key long enough to be scrubbed from replies, as a judge's base_url may hold it by mistake.
PATH_KEY = "SENSITIVE_PATH_KEY_0123456789"


def test_a_base_url_holding_the_key_is_refused_and_neither_printed_nor_recorded(cli, inputs, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", PATH_KEY)
    url = f"http://judge.test/{PATH_KEY}/v1"
    judge = tmp_path / "judge.yaml"
    judge.write_text(f"model_type: openai_compatible\nmodel_name: qwen\nbase_url: {url}\n")
    scenarios, rules = inputs
    status, out, error = cli("judges", "show", str(judge))
    assert status == 2 and "must not be part of the URL" in error and PATH_KEY not in error + str(out)
    status, created, _ = cli("campaign", "create", "runs/r", "--scenarios", scenarios, "--campaign-id", "r")
    assert status == 0
    status, out, error = cli("campaign", "add", created["root"], "--scenarios", scenarios, "--rules", rules, "--judge",
                             str(judge))
    assert status == 2 and "must not be part of the URL" in error and PATH_KEY not in error + str(out)
    assert not [path for path in tmp_path.rglob("*")
                if path.is_file() and path != judge and PATH_KEY.encode() in path.read_bytes()]


@pytest.mark.parametrize("variable, key, base_url", [
    ("OPENAI_API_KEY", PATH_KEY, "http://judge.test/v1"),  # the URL does not hold the key
    ("JUDGE_TEST_KEY", "EMPTY", "http://judge.test/EMPTY/v1"),
    (None, PATH_KEY, f"http://judge.test/{PATH_KEY}/v1"),  # a keyless judge reads no key
    ("JUDGE_TEST_KEY", None, f"http://judge.test/{PATH_KEY}/v1"),
], ids=["other-url", "placeholder", "keyless", "unset"])
def test_a_base_url_is_refused_only_for_a_key_that_is_set_and_long_enough_to_scrub(monkeypatch, variable, key,
                                                                                 base_url):
    from ruleforge.judges import JudgeSpec

    monkeypatch.setenv("OPENAI_API_KEY", PATH_KEY)
    if key is None:
        monkeypatch.delenv("JUDGE_TEST_KEY", raising=False)
    else:
        monkeypatch.setenv("JUDGE_TEST_KEY", key)
    spec = JudgeSpec.openai_compatible("qwen", base_url, api_key_env=variable)
    assert spec.identity().endpoint == base_url + "/chat/completions"
