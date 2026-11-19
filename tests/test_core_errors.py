from __future__ import annotations

import importlib
import inspect
import json
import os
import pkgutil
import subprocess
import sys

import pytest

import ruleforge.core
from ruleforge.core.errors import RuleForgeError, TracebackError

#: Each error the leaf layer defines, by its qualified name, and its built-in base.
BUILT_IN_BASE = {
    "ruleforge.core.errors.SamplingDesignError": ValueError,
    "ruleforge.core.errors.ArtifactReadError": ValueError,
    "ruleforge.core.errors.ArtifactRootError": RuntimeError,
    "ruleforge.core.artifacts.ArtifactSchemaError": ValueError,
    "ruleforge.core.errors.TracebackError": None,
    "ruleforge.core.optional.MissingDependency": ImportError,
    "ruleforge.core.provenance.CheckoutMismatch": None,  # a SamplingDesignError, and so a RuleForgeError
    "ruleforge.env.EnvUnsetError": None,
    "ruleforge.env.PackageResourceError": None,
}


def leaf_errors() -> dict[str, type]:
    """Every exception class defined in a module of ``ruleforge.core`` or in ``ruleforge.env``."""
    modules = [importlib.import_module(f"ruleforge.core.{info.name}")
               for info in pkgutil.iter_modules(ruleforge.core.__path__)] + [importlib.import_module("ruleforge.env")]
    return {f"{cls.__module__}.{cls.__qualname__}": cls for module in modules
            for cls in vars(module).values()
            if inspect.isclass(cls) and issubclass(cls, BaseException) and cls.__module__ == module.__name__}


def test_the_table_names_every_error_the_leaf_layer_defines():
    assert sorted(leaf_errors()) == sorted([*BUILT_IN_BASE, "ruleforge.core.errors.RuleForgeError"])


@pytest.mark.parametrize("name", sorted(BUILT_IN_BASE))
def test_an_error_is_a_rule_forge_error_with_its_old_base_first(name):
    error = leaf_errors()[name]
    assert issubclass(error, RuleForgeError)
    base = BUILT_IN_BASE[name]
    if base is not None:
        assert error.__bases__ == (base, RuleForgeError)
        assert not [key for key, value in vars(RuleForgeError).items() if callable(value)]
        with pytest.raises(base) as caught:
            raise error("the message")
        assert str(caught.value) == "the message" and caught.value.args == ("the message",)


#: Prints every exception class a module of the package defines, by qualified name, and whether it is a RuleForgeError.
WALK = """
import importlib, inspect, json, pkgutil
import ruleforge
from ruleforge.core.errors import RuleForgeError

found = {}
for info in pkgutil.walk_packages(ruleforge.__path__, "ruleforge."):
    if not info.name.endswith(".__main__"):
        module = importlib.import_module(info.name)
        for value in vars(module).values():
            if inspect.isclass(value) and issubclass(value, BaseException) and value.__module__ == module.__name__:
                found[f"{value.__module__}.{value.__qualname__}"] = issubclass(value, RuleForgeError)
print(json.dumps(found))
"""


def test_every_error_the_package_defines_is_a_rule_forge_error(repo_root):
    run = subprocess.run([sys.executable, "-c", WALK], capture_output=True, text=True, cwd=repo_root, timeout=600,
                         env={**os.environ, "PYTHONPATH": str(repo_root / "src")})
    assert run.returncode == 0, run.stderr
    found = json.loads(run.stdout.splitlines()[-1])
    # the walk reached every layer: a walk that imported nothing would pass vacuously
    assert {"ruleforge.judges.base.JudgeError", "ruleforge.grpo.sync_probe.SyncProbeError",
            "ruleforge.launch.guards._Computed", "ruleforge.experiments.reproduce.rerun.RerunError"} <= set(found)
    assert sorted(name for name, caught in found.items() if not caught) == []


VALUE_ERRORS = (
    "ruleforge.evaluate.estimates.CampaignEstimatesError", "ruleforge.evaluate.inference.RepeatedSamplingError",
    "ruleforge.evaluate.report.ScoringError", "ruleforge.experiments.ruleset_size.build.CampaignCurveError",
    "ruleforge.experiments.ruleset_size.repeated_draw_baseline.RepeatedDrawBaselineError",
    "ruleforge.experiments.training.campaign_readout.TrainingReadoutError",
    "ruleforge.experiments.training.frozen_anchors.FrozenAnchorsError",
    "ruleforge.experiments.training.judgment_changes.JudgmentChangesError", "ruleforge.grpo.report.ReportError",
    "ruleforge.grpo.run_files.RunFileError", "ruleforge.grpo.trainer.TrainerConfigError",
    "ruleforge.grpo.truncation.TruncationError", "ruleforge.judges.base.JudgeError",
    "ruleforge.judges.completion_format.CompletionFormatError", "ruleforge.judges.weights.MergeError",
    "ruleforge.judges.weights.ParityError", "ruleforge.optimize.metrics.intent.IntentScoringError",
    "ruleforge.rulesets.RulesetError", "ruleforge.rulesets.levels.RuleLevelsError", "ruleforge.scenarios.ScenarioSetError",
    "ruleforge.scenarios.subset.SubsetError", "ruleforge.tasks.base.TaskError",
)
TRACEBACK_ERRORS = (
    "ruleforge.grpo.fsdp2_state.Fsdp2StateError", "ruleforge.grpo.sync_probe.SyncProbeError",
    "ruleforge.judges.models.CallFailed", "ruleforge.launch.backends.CommandFailed",
)
RUNTIME_REFUSALS = (
    "ruleforge.experiments.reproduce.rerun.RerunError", "ruleforge.grpo.exclusion.ExclusionError",
    "ruleforge.grpo.leakage.LeakageError", "ruleforge.judges.tokenizer.TokenizerMismatch",
)
UNFINISHED_ERRORS = ("ruleforge.judges.batch_api.BatchExecutionError",)


def _error(name: str) -> type:
    module, _, attribute = name.rpartition(".")
    return getattr(importlib.import_module(module), attribute)


@pytest.mark.parametrize("name", VALUE_ERRORS)
def test_an_error_that_was_a_value_error_still_is_one(name):
    error = _error(name)
    assert issubclass(error, ValueError) and issubclass(error, RuleForgeError) and not issubclass(error, TracebackError)


@pytest.mark.parametrize("name", TRACEBACK_ERRORS)
def test_an_error_a_command_ends_on_with_its_traceback_is_still_a_runtime_error(name):
    error = _error(name)
    assert issubclass(error, RuntimeError) and issubclass(error, TracebackError)


@pytest.mark.parametrize("name", [*RUNTIME_REFUSALS, *UNFINISHED_ERRORS])
def test_a_runtime_error_the_command_line_ends_in_one_line_is_a_rule_forge_error_without_the_traceback_marker(name):
    error = _error(name)
    assert error.__bases__ == (RuntimeError, RuleForgeError) and not issubclass(error, TracebackError)
