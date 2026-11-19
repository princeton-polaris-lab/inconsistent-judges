from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
from packaging.requirements import Requirement

from tests.evaluate.helpers import cluster_bootstrap, recorded_cells
from tests.test_sampling_runner import RULES

pytestmark = [pytest.mark.slow, pytest.mark.allow_network]

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]
TARGETS = PYPROJECT["tool"]["hatch"]["build"]["targets"]
#: Packages the core install must not hold: every provider SDK, the engine, training, plotting, SLURM, the Hub, W&B.
NOT_CORE = ("openai", "anthropic", "tiktoken", "transformers", "torch", "vllm", "wandb", "datasets",
            "huggingface_hub", "submitit", "matplotlib", "peft", "trl")
QUICKSTART = """
import json, sys
from ruleforge import Ruleset, ScenarioSet, JudgeSpec, evaluate
r = evaluate(Ruleset.load("my_rules.jsonl"), ScenarioSet.load("five.jsonl"), JudgeSpec.fake(seed=1), draws=2,
             out="runs/quick", campaign_id="quick")
print(json.dumps({"value": r.estimates.rate.value, "ci": list(r.estimates.rate.ci), "root": str(r.root),
                  "condition_id": r.condition_id, "validity": r.estimates.validity.label_rule}))
"""
METADATA = """
import json
from importlib import metadata
found = metadata.metadata("ruleforge")
print(json.dumps({"fields": {key: found.get_all(key) for key in set(found.keys())},
                  "entry_points": sorted([entry.group, entry.name, entry.value]
                                         for entry in metadata.distribution("ruleforge").entry_points)}))
"""
PROVENANCE = """
import json
from ruleforge.core import provenance
state = provenance.code_state()
print(json.dumps({**state, "checkout": provenance.checkout_root(), "version": provenance.package_version(),
                  "snapshot": str(provenance.write_code_snapshot(".", state))}, default=str))
"""


def _run(argv, *, cwd, env=None):
    result = subprocess.run([str(item) for item in argv], cwd=cwd, capture_output=True, text=True, env=env,
                            check=False)
    assert result.returncode == 0, f"{argv}\n{result.stdout}\n{result.stderr}"
    return result.stdout


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    """A fresh environment holding the wheel and its core dependencies alone; its ``python`` and its ``bin``."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("needs uv")
    base = tmp_path_factory.mktemp("wheel")
    _run([uv, "build", "--wheel", "--out-dir", base / "dist", REPO], cwd=base)
    [wheel] = (base / "dist").glob("ruleforge-*.whl")
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    _run([uv, "venv", "--python", version, base / "venv"], cwd=base)
    python = base / "venv" / "bin" / "python"
    _run([uv, "pip", "install", "--python", python, wheel], cwd=base)
    return python, base


def _environment():
    return {name: value for name, value in os.environ.items()
            if not name.startswith(("RULEFORGE_", "PYTHONPATH", "VIRTUAL_ENV"))}


def _inputs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "my_rules.jsonl").write_text("".join(json.dumps(rule) + "\n" for rule in RULES))
    (directory / "five.jsonl").write_text("".join(
        json.dumps({"prompt": f"prompt {i}", "response": f"response {i}"}) + "\n" for i in range(5)))


def test_the_core_install_imports_without_any_extra(installed):
    python, base = installed
    probe = ("import importlib.util, json, ruleforge; "
             "from ruleforge import Ruleset, ScenarioSet, JudgeSpec, Campaign, evaluate, optimize, grpo; "
             f"print(json.dumps([name for name in {NOT_CORE!r} if importlib.util.find_spec(name.split('.')[0])]))")
    assert json.loads(_run([python, "-c", probe], cwd=base, env=_environment())) == []
    named = "from ruleforge import JudgeSpec; print(JudgeSpec.from_config('qwen3-8b').options.model_path)"
    environment = {**_environment(), "RULEFORGE_MODEL_ROOT": "/models"}
    assert _run([python, "-c", named], cwd=base, env=environment).strip() == "/models/Qwen3-8B"


def test_the_installed_library_reads_its_packaged_rate_file(installed):
    python, base = installed
    probe = ("import pathlib, ruleforge; from ruleforge.evaluate.cost import DEFAULT_RATES as rates; "
             "print(rates.is_file() and rates.is_relative_to(pathlib.Path(ruleforge.__file__).parent))")
    assert _run([python, "-c", probe], cwd=base, env=_environment()).strip() == "True"


def test_the_quickstart_from_an_unrelated_directory_gives_the_estimate_computed_outside_the_library(installed):
    python, base = installed
    work = base / "x"
    _inputs(work)
    found = json.loads(_run([python, "-c", QUICKSTART], cwd=work, env=_environment()))
    root = Path(found["root"])
    assert root == work / "runs" / "quick" and found["validity"] == "lenient_last_line"
    cells = recorded_cells(root / found["condition_id"], 2, fake_seed=1)
    ids = sorted({sid for sid, _ in cells}, key=[sid for sid, _ in cells].index)
    values = np.array([[cells[(sid, draw)] for draw in range(2)] for sid in ids])
    assert found["value"] == pytest.approx(values.mean())
    assert found["ci"] == pytest.approx(cluster_bootstrap(values, n_resamples=10_000, ci_level=0.95, seed=16))
    manifest = json.loads((root / "sampling_manifest.json").read_text())
    assert manifest["package_version"] and manifest["code_revision"] is None and manifest["code_dirty"] is True
    assert list((root / "code_snapshots").glob("*.tar.gz"))

    cli = base / "venv" / "bin" / "ruleforge"
    out = json.loads(_run([cli, "evaluate", "--rules", "my_rules.jsonl", "--scenarios", "five.jsonl", "--judge",
                           "fake", "--draws", "2", "--out", "runs/cli", "--campaign-id", "quick"], cwd=work,
                          env=_environment()))
    assert out["estimates"]["rate"]["value"] == found["value"] and out["estimates"]["rate"]["ci"] == found["ci"]
    assert (work / "runs" / "cli.runs").is_dir()
    assert "qwen3-8b" in _run([cli, "judges", "list"], cwd=work, env=_environment()).split()


def test_the_installed_code_state_changes_with_one_byte_of_an_installed_module(installed):
    python, base = installed
    work = base / "provenance"
    work.mkdir()
    before = json.loads(_run([python, "-c", PROVENANCE], cwd=work, env=_environment()))
    assert before["checkout"] is None and before["code_revision"] is None and before["code_dirty"] is True
    site = Path(_run([python, "-c", "import ruleforge, os; print(os.path.dirname(ruleforge.__file__))"], cwd=work,
                     env=_environment()).strip())
    module = site / "rulesets" / "variants.py"
    module.write_bytes(module.read_bytes() + b"#")
    after = json.loads(_run([python, "-c", PROVENANCE], cwd=work, env=_environment()))
    assert after["code_state_sha256"] != before["code_state_sha256"] and after["snapshot"] != before["snapshot"]


def _members(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def _requirement(text: str) -> tuple:
    requirement = Requirement(text)
    return requirement.name, sorted(requirement.extras), str(requirement.specifier), str(requirement.marker)


def _requirements(extra: str) -> list[str]:
    found = []
    for item in PROJECT["optional-dependencies"][extra]:
        requirement = Requirement(item)
        if requirement.name == PROJECT["name"]:
            found += [named for other in sorted(requirement.extras) for named in _requirements(other)]
        else:
            found.append(item)
    return found


def test_the_wheel_holds_the_package_alone_and_installs_the_project_s_metadata(installed):
    python, base = installed
    [wheel] = (base / "dist").glob("ruleforge-*.whl")
    members = _members(wheel)
    info = f"ruleforge-{PROJECT['version']}.dist-info/"
    assert members and [name for name in members if not name.startswith(("ruleforge/", info))] == []
    parts = {part for name in members for part in PurePosixPath(name).parts}
    assert parts & {"tests", "scripts", "tools", "__pycache__"} == set() and not any(
        part.startswith(".env") for part in parts)
    for target in TARGETS["wheel"]["force-include"].values():
        assert any(name == target or name.startswith(f"{target}/") for name in members), target

    found = json.loads(_run([python, "-c", METADATA], cwd=base, env=_environment()))
    fields = found["fields"]
    emailed = [f"{author['name']} <{author['email']}>" for author in PROJECT["authors"] if "email" in author]
    plain = [author["name"] for author in PROJECT["authors"] if "email" not in author]
    assert {key: fields.get(key) for key in ("Name", "Version", "Summary", "License-Expression", "Requires-Python",
                                             "Author", "Author-email", "Classifier", "Project-URL",
                                             "Provides-Extra", "License-File")} == {
        "Name": [PROJECT["name"]], "Version": [PROJECT["version"]], "Summary": [PROJECT["description"]],
        "License-Expression": [PROJECT["license"]], "Requires-Python": [PROJECT["requires-python"]],
        "Author": [", ".join(plain)], "Author-email": [", ".join(emailed)], "Classifier": PROJECT["classifiers"],
        "Project-URL": [f"{key}, {url}" for key, url in PROJECT["urls"].items()],
        "Provides-Extra": sorted(PROJECT["optional-dependencies"]), "License-File": PROJECT["license-files"]}
    expected = [_requirement(item) for item in PROJECT["dependencies"]] + [
        _requirement(f"{item}; extra == '{extra}'") for extra in PROJECT["optional-dependencies"]
        for item in _requirements(extra)]
    assert sorted(_requirement(item) for item in fields["Requires-Dist"]) == sorted(expected)
    assert found["entry_points"] == sorted(
        [["console_scripts", name, value] for name, value in PROJECT["scripts"].items()]
        + [[group, name, value] for group, points in PROJECT["entry-points"].items() for name, value in points.items()])


def test_the_sdist_holds_the_wheel_s_sources_and_builds_the_same_wheel(installed, tmp_path):
    _, base = installed
    uv = shutil.which("uv")
    _run([uv, "build", "--sdist", "--out-dir", tmp_path / "sdist", REPO], cwd=tmp_path)
    [sdist] = (tmp_path / "sdist").glob("ruleforge-*.tar.gz")
    with tarfile.open(sdist) as archive:
        names = [PurePosixPath(member.name) for member in archive.getmembers() if member.isfile()]
    top = f"ruleforge-{PROJECT['version']}"
    assert {name.parts[0] for name in names} == {top}
    included = TARGETS["sdist"]["only-include"]
    relative = [PurePosixPath(*name.parts[1:]).as_posix() for name in names]
    assert [name for name in relative if name not in ("pyproject.toml", "PKG-INFO", ".gitignore") and not any(
        name == path or name.startswith(f"{path}/") for path in included)] == []
    assert all(any(name == path or name.startswith(f"{path}/") for name in relative) for path in included)
    _run([uv, "build", "--wheel", "--out-dir", tmp_path / "wheel", sdist], cwd=tmp_path)
    [from_sdist] = (tmp_path / "wheel").glob("ruleforge-*.whl")
    [from_checkout] = (base / "dist").glob("ruleforge-*.whl")
    record = f"{top}.dist-info/RECORD"
    with zipfile.ZipFile(from_sdist) as one, zipfile.ZipFile(from_checkout) as other:
        assert sorted(one.read(record).decode().splitlines()) == sorted(other.read(record).decode().splitlines())
