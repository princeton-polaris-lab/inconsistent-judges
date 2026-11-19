from __future__ import annotations

import pytest

from ruleforge.core.credentials import CREDENTIAL_REFUSAL, carries_credential, refused_arguments


@pytest.mark.parametrize("text, carries", [
    ("http://user:pass@judge.test/v1", True),
    ("openai_compatible:m@http://user@judge.test/judge.yaml", True),
    ("http://judge.test/v1?api_key=x", True),
    ("judge_model.base_url=https://judge.test:8000/v1?", True),
    ("openai_compatible:m@http://judge.test/v1", False),
    ("http://judge.test/@team/v1", False),
    ("http://judge.test/v1#a?b", False),
    ("user@judge.test?x", False),
    # One URL token: no @ or ? past the authority's end or a space counts.
    ("Read https://example.org for context. Is the answer correct? {{ messages }}", False),
    ("see https://example.org or write me@example.org", False),
    ("run openai_compatible:m@http://user:pw@judge.test/v1 now", True),
    ("first https://example.org then judge_model.base_url=https://judge.test/v1?api_key=x", True),
])
def test_a_url_carrying_a_user_password_or_query_is_recognised(text, carries):
    assert carries_credential(text) is carries


def test_a_command_line_carrying_one_is_refused_on_stderr_without_repeating_it(capsys):
    assert refused_arguments(["--judge", "fake", "--out", "o"], "prog") is False
    assert capsys.readouterr().err == ""
    assert refused_arguments(["--judge", "openai_compatible:m@http://u:SENSITIVE_ARGUMENT@judge.test/v1"], "prog")
    assert capsys.readouterr().err == f"prog: {CREDENTIAL_REFUSAL}\n"
