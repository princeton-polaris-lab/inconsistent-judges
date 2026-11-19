from __future__ import annotations

import importlib
import re
import sys
import tomllib

import pytest

from ruleforge.core.errors import RuleForgeError
from ruleforge.core.optional import INSTALLED_BY, MissingDependency, require
from tests.conftest import REPO_ROOT


def test_an_installed_dependency_passes_unimported(monkeypatch):
    monkeypatch.delitem(sys.modules, "zstandard", raising=False)
    require("this command", "zstandard")
    assert "zstandard" not in sys.modules


def test_the_missing_dependencies_are_named_in_one_line_with_what_installs_them(monkeypatch):
    for name in ("matplotlib", "submitit"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(MissingDependency) as caught:
        require("this command", "zstandard", "matplotlib", "submitit")
    assert str(caught.value) == ("this command needs matplotlib, submitit, which this environment lacks: add --group dev "
                                 "--group viz to uv sync, or pip install -e '.[slurm,viz]' from the checkout")
    assert isinstance(caught.value, ImportError) and isinstance(caught.value, RuleForgeError)


def test_a_directory_without_a_package_is_missing(monkeypatch, tmp_path):
    (tmp_path / "ruleforge_absent_package").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(INSTALLED_BY, "ruleforge_absent_package", ("viz", "viz"))
    with pytest.raises(MissingDependency, match="^this command needs ruleforge_absent_package, "):
        require("this command", "ruleforge_absent_package")
    assert "ruleforge_absent_package" not in sys.modules


def test_a_directory_without_a_package_is_missing_once_imported_too(monkeypatch, tmp_path):
    (tmp_path / "ruleforge_absent_package").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(INSTALLED_BY, "ruleforge_absent_package", ("viz", "viz"))
    monkeypatch.setitem(sys.modules, "ruleforge_absent_package", None)
    del sys.modules["ruleforge_absent_package"]
    importlib.import_module("ruleforge_absent_package")
    with pytest.raises(MissingDependency, match="^this command needs ruleforge_absent_package, "):
        require("this command", "ruleforge_absent_package")


def _distributions(requirements: list[str]) -> set[str]:
    return {re.split(r"[<>=!~\[ ]", requirement, maxsplit=1)[0].lower().replace("_", "-")
            for requirement in requirements}


def _installs(requirements: list[str], extras: dict[str, list[str]]) -> set[str]:
    found = _distributions(requirements)
    for requirement in requirements:
        if requirement.startswith("ruleforge["):
            for extra in requirement[len("ruleforge["):-1].split(","):
                found |= _installs(extras[extra], extras)
    return found


@pytest.mark.parametrize("module", sorted(INSTALLED_BY))
def test_the_extra_and_the_group_each_dependency_names_install_it(module):
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras, groups = project["project"]["optional-dependencies"], project["dependency-groups"]
    extra, group = INSTALLED_BY[module]
    distribution = module.replace("_", "-")
    assert distribution in _installs(extras[extra], extras)
    assert distribution in _installs(groups[group], extras)
