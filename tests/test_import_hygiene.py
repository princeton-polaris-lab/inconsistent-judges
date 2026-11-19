from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

import pytest

OPTIONAL_HEAVY_MODULES = {
    "vllm",
    "trl",
    "deepspeed",
    "peft",
    "accelerate",
    "flash_attn",
    "flashinfer",
    "bitsandbytes",
    "matplotlib",
    "statsmodels",
}

MODULE_LEVEL_TORCH_ALLOWED = {
    "src/ruleforge/grpo/kl_anchor.py",
    "src/ruleforge/grpo/sync_probe.py",
    "src/ruleforge/grpo/truncation.py",
}


def _source_files(repo_root):
    files = sorted((repo_root / "src" / "ruleforge").rglob("*.py"))
    assert files, "no module under the scanned root: a stale root would pass vacuously"
    return files


def _module_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_module_level_import_of_an_optional_heavy_dependency(repo_root):
    offenders: list[str] = []
    for path in _source_files(repo_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = _module_level_imports(tree) & OPTIONAL_HEAVY_MODULES
        if hits:
            relative = path.relative_to(repo_root)
            offenders.append(f"{relative}: {sorted(hits)}")
    assert offenders == [], (
        "these modules import an optional dependency at module scope, which "
        "breaks `import ruleforge` without the vllm/grpo groups; move the import "
        "inside the function or under `if TYPE_CHECKING:`\n" + "\n".join(offenders)
    )


def test_module_level_torch_is_confined_to_the_declared_exception(repo_root):
    importers = set()
    for path in _source_files(repo_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if "torch" in _module_level_imports(tree):
            importers.add(str(path.relative_to(repo_root)))
    assert importers <= MODULE_LEVEL_TORCH_ALLOWED, (
        "unexpected module-level torch import in "
        f"{sorted(importers - MODULE_LEVEL_TORCH_ALLOWED)}; add a lazy import "
        "or extend MODULE_LEVEL_TORCH_ALLOWED with a reason"
    )


_IMPORT_PROBE = """
import importlib, json, sys
importlib.import_module({module!r})
print(json.dumps(sorted(m for m in {heavy!r} if m in sys.modules)))
"""


def _imports_pulled_in(module: str, heavy: tuple[str, ...], repo_root) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE.format(module=module, heavy=heavy)],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )
    assert result.returncode == 0, f"importing {module} failed:\n{result.stderr}"
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "module",
    [
        "ruleforge.prompts.context",
        "ruleforge.rulesets.rules",
        "ruleforge.rulesets.variants",
        "ruleforge.optimize",
        "ruleforge.optimize.judging",
        "ruleforge.judges.backend",
        "ruleforge.evaluate.bootstrap",
        "ruleforge.evaluate.report",
        "ruleforge.judges.weights",
        "ruleforge.judges.completion_format",
        # the evaluate package imports its API on first use, so its scoring core and inference helpers stay light
        "ruleforge.evaluate",
        "ruleforge.evaluate.scoring",
        "ruleforge.evaluate.multiplicity",
    ],
)
def test_importing_a_core_module_does_not_pull_in_the_cuda_stack(module, repo_root):
    assert _imports_pulled_in(module, ("vllm", "torch"), repo_root) == [], (
        f"importing {module} loaded the CUDA stack; the coverage gate and CI "
        "run on CPU-only machines where it is not installed"
    )


@pytest.mark.parametrize(
    "module",
    [
        "ruleforge.evaluate.bootstrap",
        "ruleforge.evaluate.report",
        "ruleforge.evaluate.scoring",
        "ruleforge.evaluate.multiplicity",
    ],
)
def test_an_analysis_module_does_not_pull_in_a_plotting_stack(module, repo_root):
    heavy = ("matplotlib", "statsmodels", "transformers")
    assert _imports_pulled_in(module, heavy, repo_root) == []


def test_the_command_line_imports_without_submitit(repo_root):
    assert _imports_pulled_in("ruleforge.cli.app", ("submitit",), repo_root) == []
