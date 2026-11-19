from __future__ import annotations

import json
import subprocess
import sys

import pytest

import ruleforge


def test_importing_the_package_loads_nothing_below_its_core():
    probe = "import json, sys, ruleforge; print(json.dumps(sorted(m for m in sys.modules if m.startswith('ruleforge'))))"
    loaded = json.loads(subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                                       check=True).stdout)
    assert loaded == ["ruleforge", "ruleforge.core", "ruleforge.core.errors", "ruleforge.env"]


def test_the_toolkit_s_names_come_from_their_packages():
    from ruleforge.evaluate import Campaign, SamplingSettings
    from ruleforge.judges import JudgeSpec
    from ruleforge.rulesets import Ruleset

    assert (ruleforge.Ruleset, ruleforge.JudgeSpec, ruleforge.Campaign, ruleforge.SamplingSettings) == (
        Ruleset, JudgeSpec, Campaign, SamplingSettings)
    assert set(ruleforge.__all__) <= set(dir(ruleforge)) and "__version__" in dir(ruleforge)
    assert ruleforge.optimize.optimize and ruleforge.grpo.rows
    with pytest.raises(AttributeError, match="has no attribute 'nothing'"):
        getattr(ruleforge, "nothing")


def test_evaluate_is_the_package_and_calls_the_function(monkeypatch):
    from ruleforge import evaluate
    from ruleforge.evaluate import api

    seen = []
    monkeypatch.setattr(api, "evaluate", lambda *args, **kwargs: seen.append((args, kwargs)) or "result")
    import ruleforge.evaluate as package

    assert evaluate is package and evaluate.Campaign is api.Campaign
    monkeypatch.setattr(package, "evaluate", api.evaluate)
    assert evaluate("rules", "scenarios", "fake", draws=2) == "result"
    assert seen == [(("rules", "scenarios", "fake"), {"draws": 2})]


def test_the_version_is_the_installed_distribution_s():
    from importlib import metadata

    assert ruleforge.__version__ == metadata.version("ruleforge")


def test_a_package_is_exported_before_it_is_imported():
    assert ruleforge.__getattr__("grpo") is ruleforge.grpo


@pytest.mark.parametrize("module", ["ruleforge", "ruleforge.cli"])
def test_running_a_module_runs_the_command(module, monkeypatch, capsys):
    import runpy

    monkeypatch.setattr(sys, "argv", [module, "judges", "list"])
    with pytest.raises(SystemExit) as exited:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    assert exited.value.code == 0 and "qwen3-8b" in capsys.readouterr().out.split()
