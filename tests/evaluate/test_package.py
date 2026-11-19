from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

import ruleforge.evaluate as package


def test_every_public_name_is_the_object_its_module_defines():
    assert sorted(package.__all__) == sorted(package._EXPORTS)
    for name in package.__all__:
        module = importlib.import_module(f"ruleforge.evaluate.{package._EXPORTS[name]}")
        assert getattr(package, name) is getattr(module, name), name
    assert set(package.__all__) <= set(dir(package))


def test_the_shared_bank_module_declares_the_toolkit_s_names_alone():
    from ruleforge.evaluate import bank

    assert bank.__all__ == ["SharedBankDesign", "PlannedCondition", "analyze", "freeze_design", "plan_conditions"]
    assert all(getattr(bank, name).__module__ == bank.__name__ for name in bank.__all__)


def test_a_name_the_package_does_not_export_is_an_attribute_error():
    with pytest.raises(AttributeError, match="module 'ruleforge.evaluate' has no attribute 'nothing'"):
        package.nothing


def test_a_module_of_the_package_imports_without_its_api(repo_root):
    probe = "import sys\nimport ruleforge.evaluate.scoring\nprint(sorted(m for m in ('ruleforge.evaluate.api', " \
            "'ruleforge.judges') if m in sys.modules))\n"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=repo_root,
                            env={**os.environ, "PYTHONPATH": str(repo_root / "src")})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
