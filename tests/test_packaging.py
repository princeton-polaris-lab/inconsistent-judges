from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from ruleforge.cli.app import main
from ruleforge.cli.data import DATASET
from ruleforge.env import package_path
from ruleforge.tasks.registry import BUILTIN_TASK_CLASSES, ENTRY_POINT_GROUP

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]
TARGETS = PYPROJECT["tool"]["hatch"]["build"]["targets"]
README = (REPO / "README.md").read_text(encoding="utf-8")
TITLE = "Aligning Judge Models to Be Logically Consistent Multi-Rule Interpreters"
AUTHORS = ["Michel Liao", "Nimra Nadeem", "Luxi He", "Peter Henderson"]
#: The authors with their affiliation marks, 1 MATS and 2 Princeton, as the paper's title block has them.
BYLINE = ", ".join(f"{name}<sup>{mark}</sup>" for name, mark in zip(AUTHORS, ("1,2", "2", "2", "1,2")))
#: The paper's Hugging Face repositories: one dataset and one model repository of the same name.
HUB = "ai-law-society-lab/inconsistent-judges"
#: The paper on arXiv: its identifier, its address and the badge that links it, first in each badge row.
ARXIV = "2611.29975"
PAPER = f"https://arxiv.org/abs/{ARXIV}"
PAPER_BADGE = f"[![Paper](https://img.shields.io/badge/Paper-B31B1B?logo=arxiv&logoColor=white)]({PAPER})"
#: The citation's fields in their order, as the README's Cite and each Hub card's Citation give them.
CITATION = {"title": TITLE, "author": " and ".join(", ".join(name.rsplit(" ", 1)[::-1]) for name in AUTHORS),
            "year": "2026", "eprint": ARXIV, "archivePrefix": "arXiv", "primaryClass": "cs.AI", "url": PAPER}

#: What ``import ruleforge`` and an evaluation over the fake judge need.
CORE = {"fuzzysearch", "hydra-core", "numpy", "omegaconf", "packaging", "pyyaml", "statsmodels"}
EXTRAS = {
    "openai": {"openai", "tiktoken"},
    "anthropic": {"anthropic"},
    "providers": {"ruleforge"},
    "vllm": {"vllm", "torch", "transformers", "tokenizers"},
    "grpo": {"accelerate", "deepspeed", "kernels", "peft", "pyarrow", "safetensors", "tokenizers", "torch",
             "transformers", "trl"},
    "viz": {"matplotlib"},
    "slurm": {"submitit", "hydra-submitit-launcher"},
    "hub": {"huggingface-hub", "datasets", "zstandard", "pyarrow"},
    "wandb": {"wandb"},
}
IMPORTED = {
    "accelerate": "accelerate", "anthropic": "anthropic", "datasets": "datasets", "fuzzysearch": "fuzzysearch",
    "huggingface_hub": "huggingface-hub", "hydra": "hydra-core",
    "matplotlib": "matplotlib", "numpy": "numpy", "omegaconf": "omegaconf",
    "openai": "openai", "packaging": "packaging", "peft": "peft", "safetensors": "safetensors",
    "statsmodels": "statsmodels", "submitit": "submitit", "tiktoken": "tiktoken", "tokenizers": "tokenizers",
    "torch": "torch", "transformers": "transformers", "trl": "trl", "vllm": "vllm", "wandb": "wandb",
    "yaml": "pyyaml", "zstandard": "zstandard",
}
#: Modules imported only under ``try: ... except ImportError``, the code running without them: vLLM's harmony parser.
GUARDED = {"openai_harmony"}
#: Namespace packages, whose portions are distinct distributions.
NAMESPACES = {"google"}
#: A checkout directory the wheel ships whole, file by file.
SHIPPED_WHOLE = "data/rules/modelspec"
GROUPS = {"dev", "vllm", "grpo", "viz", "population"}
MODAL_ONLY = {"modal", "grpclib", "h2", "hpack", "hyperframe", "synchronicity", "toml", "types-certifi", "types-toml"}


def _names(requirements) -> set[str]:
    return {canonicalize_name(Requirement(item).name) for item in requirements if isinstance(item, str)}


def test_the_project_metadata_is_the_readme_s():
    assert (PROJECT["name"], PROJECT["version"], PROJECT["license"]) == ("ruleforge", "1.0.0", "MIT")
    assert [author["name"] for author in PROJECT["authors"]] == AUTHORS
    assert TITLE in PROJECT["description"]
    # The title, the authors and the affiliations each on a line of their own, each affiliation after its logo.
    mats = '<a href="https://www.matsprogram.org/"><img src="assets/logos/mats.png" alt="" height="24" align="absmiddle"> MATS</a>'
    lab = ('<a href="https://www.polarislab.org/"><picture><source media="(prefers-color-scheme: dark)" '
           'srcset="assets/logos/polaris-lab-dark.svg"><img src="assets/logos/polaris-lab-light.svg" alt="" height="24" '
           'align="absmiddle"></picture> POLARIS Lab</a>, Princeton University')
    assert (f"Code of *{TITLE}*.\n\n{BYLINE}\n\n<sup>1</sup> {mats}\n&nbsp;&nbsp;&nbsp;\n<sup>2</sup> {lab}\n\n"
            in README)
    logos = sorted(re.findall(r'(?:src|srcset)="(assets/[^"]+)"', README))
    assert logos == sorted(path.relative_to(REPO).as_posix() for path in (REPO / "assets" / "logos").iterdir())
    assert cited(README, "## Cite") == CITATION
    assert PROJECT["readme"] == "README.md" and PROJECT["requires-python"] == ">=3.12"
    # The third-party data's attributions ship as a license file, beside the code's license.
    assert PROJECT["license-files"] == ["LICENSE", "data/README.md"]
    assert all((REPO / name).is_file() for name in [PROJECT["readme"], *PROJECT["license-files"]])
    assert "## Licenses and attributions" in (REPO / "data" / "README.md").read_text(encoding="utf-8")


def test_the_urls_name_the_repository_and_the_hub_repositories():
    inventory = yaml.safe_load((REPO / "data" / "hub_inventory.yaml").read_text(encoding="utf-8"))
    assert inventory["dataset"] == inventory["model"] == DATASET == HUB
    assert PROJECT["urls"] == {"Paper": PAPER,
                               "Repository": "https://github.com/princeton-polaris-lab/inconsistent-judges",
                               "Dataset": f"https://huggingface.co/datasets/{HUB}",
                               "Model": f"https://huggingface.co/{HUB}"}
    # The README's badge row links the paper, the website, then the dataset and the model; its install command, the
    # repository.
    badges = re.findall(r"^\[!\[(\w+)\]\(https://img\.shields\.io/badge/[^)\s]+\)\]\((https://\S+)\)$", README, re.M)
    assert [label for label, _ in badges] == ["Paper", "Website", "Dataset", "Models"]
    assert README.splitlines()[2] == PAPER_BADGE
    assert [link for _, link in badges[:1] + badges[2:]] == [PROJECT["urls"][name]
                                                              for name in ("Paper", "Dataset", "Model")]
    assert f"git clone {PROJECT['urls']['Repository']} " in README


WEBSITE = "https://www.polarislab.org/inconsistent-judges/"
CARD_HEADER = re.compile(r"\n---\n\n# [^\n]+\n\n(?P<badges>(?:\[!\[[^\n]+\n)+)\n(?P<about>(?:[^\n]+\n)+)\n"
                         r"(?P<byline>[^\n]+)\n\n(?P<affiliations>[^\n]+)\n\n")
CODE_BADGE = ("[![Code](https://img.shields.io/badge/Code-181717?logo=github&logoColor=white)]"
              f"({PROJECT['urls']['Repository']})")


def cited(text: str, heading: str) -> dict[str, str]:
    """The fields of the one BibTeX entry, the document's last section ``heading`` holds alone, in their order."""
    assert text.count(f"\n{heading}\n") == 1, f"one {heading} section"
    section = text.split(f"\n{heading}\n", 1)[1]
    entry = re.fullmatch(r"\n```bibtex\n@misc\{liao2026aligning,\n(?P<fields>(?: {2}\w+ += \{[^{}\n]+\},?\n)+)"
                         r"\}\n```\n", section)
    assert entry is not None, f"{heading}: one BibTeX entry, liao2026aligning, and nothing after it"
    lines = entry["fields"].splitlines()
    # The fields' equals signs align, one field to a line, a comma after each but the last.
    assert len({line.index("=") for line in lines}) == 1
    assert [line.endswith(",") for line in lines] == [True] * (len(lines) - 1) + [False]
    return dict(re.fullmatch(r" {2}(\w+) += \{(.+)\},?", line).groups() for line in lines)


def test_every_arxiv_address_and_identifier_in_the_readme_the_cards_and_the_metadata_is_the_paper_s():
    texts = {"README.md": README, "pyproject.toml": (REPO / "pyproject.toml").read_text(encoding="utf-8"),
             **{f"data/hub_cards/{card}.md": (REPO / "data" / "hub_cards" / f"{card}.md").read_text(encoding="utf-8")
                for card in ("dataset", "model")}}
    ids = {name: re.findall(r"arxiv\.org/(?:abs|pdf)/([^\s)}\"]+)|eprint += \{([^}]*)\}", text, re.I)
           for name, text in texts.items()}
    # Each document's badge and citation address, its eprint, and the metadata's Paper link: one identifier everywhere.
    assert {name: len(found) for name, found in ids.items()} == {"README.md": 3, "pyproject.toml": 1,
                                                                 "data/hub_cards/dataset.md": 3,
                                                                 "data/hub_cards/model.md": 3}
    assert {value for found in ids.values() for pair in found for value in pair if value} == {ARXIV}
    blocks = {name: re.findall(r"```bibtex\n.*?```\n", text, re.S) for name, text in texts.items()
              if name != "pyproject.toml"}
    assert all(len(found) == 1 for found in blocks.values()) and len({found[0] for found in blocks.values()}) == 1


@pytest.mark.parametrize("card, sibling", [("dataset", "Models"), ("model", "Dataset")])
def test_each_hub_card_s_header_is_the_readme_s_without_logos(card, sibling):
    text = (REPO / "data" / "hub_cards" / f"{card}.md").read_text(encoding="utf-8")
    header = CARD_HEADER.match(text, text.index("\n---\n", 3))
    assert header is not None, "under the title: the badges, what the repository is, the authors, the affiliations"
    # The README's Paper and Website badges, the repository's Code badge, then the README's badge for the other Hub
    # repository.
    readme = {label: line for line, label in re.findall(r"^(\[!\[(\w+)\][^\n]*)$", README, re.M)}
    assert readme["Paper"] == PAPER_BADGE and readme["Website"].endswith(f"]({WEBSITE})")
    assert header["badges"].splitlines() == [readme["Paper"], readme["Website"], CODE_BADGE, readme[sibling]]
    # What the repository is names the paper in italics, and points to the other Hub repository by its badge.
    about = " ".join(header["about"].split())
    assert f"*{TITLE}*" in about and f"(the {sibling} button above)" in about
    assert header["byline"] == BYLINE
    assert header["affiliations"] == "<sup>1</sup>MATS &nbsp;&nbsp;&nbsp; <sup>2</sup>POLARIS Lab, Princeton University"
    # The authors and the affiliations are said once, and a Hub card shows no logo, only the badges.
    assert all(text.count(name) == 1 for name in AUTHORS) and text.count("<sup>1</sup>MATS") == 1
    assert "Paper:" not in text and not re.search(r"<img|<picture|!\[[^]]*\]\((?!https://img\.shields\.io/badge/)", text)
    # The card ends with the README's citation, under a Citation heading.
    assert cited(text, "## Citation") == CITATION


def test_the_core_dependencies_and_the_extras_are_as_stated():
    assert _names(PROJECT["dependencies"]) == CORE
    extras = PROJECT["optional-dependencies"]
    assert {name: _names(items) for name, items in extras.items()} == EXTRAS
    assert extras["providers"] == ["ruleforge[openai,anthropic]"]


def _imports(tree: ast.AST, module: str, *, in_function: bool = False) -> list[tuple[int, bool]]:
    found = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            found += [(node.lineno, in_function) for name in names if name.split(".")[0] == module]
        found += _imports(node, module, in_function=in_function or isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)))
    return found


def test_zstandard_is_the_hub_extra_s_alone_imported_where_it_is_used():
    found = {}
    for path in sorted((REPO / "src" / "ruleforge").rglob("*.py")):
        if imports := _imports(ast.parse(path.read_text(encoding="utf-8")), "zstandard"):
            found[path.relative_to(REPO).as_posix()] = imports
    assert set(found) == {"src/ruleforge/hub.py"}
    assert all(inside for imports in found.values() for _, inside in imports)
    probe = "import zstandard\nclass C:\n    from zstandard import x\ndef f():\n    import zstandard.y\n"
    assert _imports(ast.parse(probe), "zstandard") == [(1, False), (3, False), (5, True)]


def _catches_import_error(node: ast.Try) -> bool:
    return any(handler.type is not None and {"ImportError", "ModuleNotFoundError"} & {
        name.id for name in ast.walk(handler.type) if isinstance(name, ast.Name)} for handler in node.handlers)


def _third_party(node: ast.AST, found: dict[str, set[bool]], guarded: bool = False) -> dict[str, set[bool]]:
    if isinstance(node, (ast.Import, ast.ImportFrom)) and getattr(node, "level", 0) == 0:
        names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                 else [f"{node.module}.{alias.name}" for alias in node.names])
        for parts in (name.split(".") for name in names):
            if parts[0] not in sys.stdlib_module_names and parts[0] not in ("__future__", "ruleforge"):
                found.setdefault(".".join(parts[:2] if parts[0] in NAMESPACES else parts[:1]), set()).add(guarded)
    rest = list(ast.iter_child_nodes(node))
    if isinstance(node, ast.Try) and _catches_import_error(node):
        for statement in node.body:
            _third_party(statement, found, True)
        rest = [*node.handlers, *node.orelse, *node.finalbody]
    for child in rest:
        _third_party(child, found, guarded)
    return found


def test_each_distribution_the_package_imports_is_declared_not_inherited():
    found: dict[str, set[bool]] = {}
    for path in sorted((REPO / "src" / "ruleforge").rglob("*.py")):
        _third_party(ast.parse(path.read_text(encoding="utf-8")), found)
    assert set(found) == set(IMPORTED) | GUARDED
    assert all(found[name] == {True} for name in GUARDED)
    declared = _names(PROJECT["dependencies"]).union(
        *map(_names, PROJECT["optional-dependencies"].values()), *map(_names, PYPROJECT["dependency-groups"].values()))
    assert {module: name for module, name in IMPORTED.items() if name not in declared} == {}
    installed = metadata.packages_distributions()
    assert {module: name for module, name in IMPORTED.items() if module.split(".")[0] in installed and name not in {
        canonicalize_name(found) for found in installed[module.split(".")[0]]}} == {}
    probe = ("import numpy, os.path\nfrom google import genai\nfrom . import x\nfrom google.genai import types\n"
             "try:\n    import a\nexcept (ValueError, ImportError):\n    import b\nelse:\n    import c\n"
             "try:\n    import d\nexcept ValueError:\n    pass\ntry:\n    import e\nexcept:\n    pass\n")
    assert _third_party(ast.parse(probe), {}) == {"numpy": {False}, "google.genai": {False}, "a": {True},
                                                  "b": {False}, "c": {False}, "d": {False}, "e": {False}}


def test_the_dependency_groups_are_as_stated_and_none_is_cloud():
    groups = PYPROJECT["dependency-groups"]
    assert set(groups) == GROUPS and "cloud" not in groups
    assert "ruleforge[providers,hub,wandb,slurm]" in groups["dev"]


def test_the_default_groups_install_no_cuda_stack_and_the_cuda_groups_are_named_as_their_extras():
    groups, extras = PYPROJECT["dependency-groups"], PROJECT["optional-dependencies"]
    default = PYPROJECT["tool"]["uv"]["default-groups"]
    assert default == ["dev", "viz"]
    named = set().union(*(_names(groups[group]) for group in default))
    through = {extra for group in default for item in groups[group] for extra in Requirement(item).extras}
    named |= set().union(*(_names(extras[extra]) for extra in through))
    assert named and not named & {"torch", "vllm", "flashinfer-jit-cache", "deepspeed"}
    assert {"torch", "vllm"} <= _names(groups["vllm"]) and {"torch", "trl", "peft"} <= _names(groups["grpo"])
    assert {"vllm", "grpo"} <= set(extras)


def test_a_distribution_named_in_more_than_one_place_is_pinned_alike():
    pins: dict[str, set[str]] = {}
    for items in [PROJECT["dependencies"], *PROJECT["optional-dependencies"].values(),
                  *PYPROJECT["dependency-groups"].values()]:
        for item in items:
            if isinstance(item, str) and canonicalize_name((requirement := Requirement(item)).name) != "ruleforge":
                pins.setdefault(canonicalize_name(requirement.name), set()).add(str(requirement.specifier))
    assert {name: specifiers for name, specifiers in pins.items() if len(specifiers) > 1} == {}
    assert pins["torch"] == {"==2.13.0"} and pins["vllm"] == {"==0.28.0"}


def test_the_lock_records_these_extras_and_groups_and_no_modal():
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    [entry] = [package for package in lock["package"] if package["name"] == "ruleforge"]
    recorded = entry["metadata"]
    extras = PROJECT["optional-dependencies"]
    assert recorded["provides-extras"] == list(extras)
    by_extra: dict[str | None, set[str]] = {}
    for requirement in recorded["requires-dist"]:
        extra = re.search(r"extra == '([^']+)'", requirement.get("marker", ""))
        by_extra.setdefault(extra and extra.group(1), set()).add(canonicalize_name(requirement["name"]))
    assert by_extra == {None: _names(PROJECT["dependencies"]), **{name: _names(items) for name, items in extras.items()}}
    assert {group: {canonicalize_name(item["name"]) for item in items}
            for group, items in recorded["requires-dev"].items()} == {
        group: _names(items) for group, items in PYPROJECT["dependency-groups"].items()}
    assert set(entry["dev-dependencies"]) == GROUPS
    assert {package["name"] for package in lock["package"]} & MODAL_ONLY == set()


def test_the_console_script_and_the_task_entry_points_resolve():
    assert PROJECT["scripts"] == {"ruleforge": "ruleforge.cli:main"}
    module, _, name = PROJECT["scripts"]["ruleforge"].partition(":")
    assert getattr(importlib.import_module(module), name) is main
    assert set(PROJECT["entry-points"]) == {ENTRY_POINT_GROUP}
    resolved = {}
    for task, value in PROJECT["entry-points"][ENTRY_POINT_GROUP].items():
        module, _, attribute = value.partition(":")
        resolved[task] = getattr(importlib.import_module(module), attribute)
    assert resolved == BUILTIN_TASK_CLASSES


def test_the_wheel_packages_ruleforge_and_the_files_it_force_includes_below_it():
    wheel = TARGETS["wheel"]
    assert set(wheel) == {"packages", "force-include"} and wheel["packages"] == ["src/ruleforge"]
    assert wheel["force-include"]
    for source, target in wheel["force-include"].items():
        assert target == f"ruleforge/{source}" and source.split("/")[0] in ("data", "configs")
        assert (REPO / source).is_file() and not (REPO / source).is_symlink()
        assert package_path(source).resolve() == (REPO / source).resolve()
    package = REPO / "src" / "ruleforge"
    assert [path for path in package.rglob("*") if path.name in ("tests", "scripts", "tools")
            or path.name.startswith(".env")] == []


def test_the_recorded_names_table_ships_below_the_package_and_is_read_as_package_data():
    from importlib import resources

    from ruleforge.core import record_names

    source = REPO / "src" / "ruleforge" / "core" / record_names.TABLE_FNAME
    assert TARGETS["wheel"]["packages"] == ["src/ruleforge"] and source.is_file() and not source.is_symlink()
    assert resources.files("ruleforge.core").joinpath(record_names.TABLE_FNAME).read_bytes() == source.read_bytes()
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout: no list of the files a commit holds")
    relative = source.relative_to(REPO).as_posix()
    assert subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=REPO, capture_output=True).returncode == 0
    assert subprocess.run(["git", "check-ignore", "-q", relative], cwd=REPO).returncode == 1


def test_the_force_included_files_are_those_a_commit_holds_and_every_ruleset():
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout: no list of the files a commit holds")
    sources = list(TARGETS["wheel"]["force-include"])
    listed = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *sources,
                             SHIPPED_WHOLE], cwd=REPO, capture_output=True, check=True).stdout.decode().split("\0")
    listed = sorted(path for path in listed if path and (REPO / path).is_file())
    assert set(sources) <= set(listed)
    assert [path for path in listed if path.startswith(f"{SHIPPED_WHOLE}/") and path not in sources] == []
    assert any(path.startswith(f"{SHIPPED_WHOLE}/") for path in sources)


def test_the_sdist_holds_the_wheel_s_sources_alone():
    sdist = TARGETS["sdist"]
    assert set(sdist) == {"only-include"}
    assert sdist["only-include"] == ["src/ruleforge", *TARGETS["wheel"]["force-include"], PROJECT["readme"],
                                     *PROJECT["license-files"]]
    assert [path for path in sdist["only-include"] if not (REPO / path).exists()] == []


#: The provider SDKs an extra brings: a command that serves one provider must not need another's to import.
PROVIDER_MODULES = ("google", "anthropic", "openai", "httpx")

NO_PROVIDER_SDK = """
import importlib, pkgutil, sys

class Blocked:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {blocked!r}:
            raise ImportError(f"no {{name}} (blocked)")
        return None

sys.meta_path.insert(0, Blocked())
failed = []
import ruleforge

names = [info.name for info in pkgutil.walk_packages(ruleforge.__path__, "ruleforge.")
         if not info.name.endswith(".__main__")]
assert "ruleforge.judges.anthropic_batch" in names and len(names) > 100, names
for name in names:
    try:
        importlib.import_module(name)
    except ImportError as error:
        failed.append(f"{{name}}: {{error}}")
print("\\n".join(failed))
sys.exit(1 if failed else 0)
"""


def test_every_command_module_imports_with_no_provider_sdk_installed():
    """An install with one provider's extra runs every command it advertises."""
    run = subprocess.run([sys.executable, "-c", NO_PROVIDER_SDK.format(blocked=set(PROVIDER_MODULES))],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr


def test_each_provider_sdk_is_bounded_below_its_next_major():
    extras = PYPROJECT["project"]["optional-dependencies"]
    for extra, distribution in (("openai", "openai"), ("anthropic", "anthropic")):
        (requirement,) = [Requirement(spec) for spec in extras[extra] if Requirement(spec).name == distribution]
        clauses = {clause.operator: clause.version for clause in requirement.specifier}
        next_major = Version(clauses[">="]).major + 1
        assert set(clauses) == {">=", "<"} and Version(clauses["<"]) == Version(str(next_major)), requirement
