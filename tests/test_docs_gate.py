from __future__ import annotations

import ast
import importlib
import re
import shlex
import subprocess
import sys
import types
from pathlib import Path
from typing import Optional

import pytest
import yaml

from ruleforge.cli.app import build_parser
from ruleforge.core.paths import ARTIFACT_ROOT_ENV
from ruleforge.experiments import table
from tests.conftest import REPO_ROOT, requires_git_worktree

#: The documents: every tracked Markdown file, and every file under docs/, and the variables' template.
DOC_SUFFIXES = (".md",)
DOC_FILES = (".env.example",)
CONFIG_SUFFIXES = (".yaml", ".yml", ".sh", ".toml")
#: The Python files whose ``ruleforge.*`` imports must resolve; string constants are read under the first alone.
CODE_DIRS = ("src/", "tests/")
#: A string constant that names a module, or an attribute of one, by its import path.
MODULE = re.compile(r"ruleforge(?:\.[A-Za-z_]\w*)+(?::[A-Za-z_]\w*)?")
RECORDED = {"ruleforge.finetuning.evalpipe", "ruleforge.finetuning.lora_parity", "ruleforge.finetuning.selection"}

NAME = re.compile(r"(?<![\w./-])ruleforge(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
SCRIPTS = re.compile(r"(?<![\w./-])" + "scripts" + r"/([A-Za-z0-9_./-]*)")
FENCE = re.compile(r"^\s*```")
HELPER = re.compile(r"^\s*(\w+)\(\)\s*\{\s*uv run [^;]*\bruleforge jobs launch \"\$@\";\s*\}")
VARIABLE = re.compile(r"^\$(\w+)$|^\$\{(\w+)\}$")
PLACEHOLDER = re.compile(r"<[^<>\n]+>")
ASSIGNMENT = re.compile(r"^\w+=")
#: A word that ends a command's arguments: a redirection (``>``, ``2>&1``, ``<in``) or a control operator.
SHELL_END = re.compile(r"^(?:\d*[<>]|&|\||;)")


def _tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.splitlines()


def _documents() -> list[str]:
    return [path for path in _tracked() if (path.endswith(DOC_SUFFIXES) or path.startswith("docs/")
                                            or path in DOC_FILES) and (REPO_ROOT / path).is_file()]


def _documents_and_configs() -> list[str]:
    configs = [path for path in _tracked() if path.startswith("configs/") and path.endswith(CONFIG_SUFFIXES)
               and (REPO_ROOT / path).is_file()]
    return [*_documents(), *configs]


def _blocks(text: str) -> list[list[str]]:
    blocks, current, inside = [], [], False
    for line in text.splitlines():
        if FENCE.match(line):
            if inside:
                blocks.append(current)
            current, inside = [], not inside
            continue
        if inside:
            if current and current[-1].endswith("\\"):
                current[-1] = current[-1][:-1] + " " + line.strip()
            else:
                current.append(line)
    return blocks


def assignments(block: list[str]) -> dict[str, str]:
    """The variables ``block``'s assignment lines (``NAME=value ...`` and nothing else) set."""
    found = {}
    for line in block:
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        if words and all(ASSIGNMENT.match(word) for word in words):
            found.update(word.split("=", 1) for word in words)
    return found


def _word(word: str, variables: dict[str, str]) -> list[str]:
    match = VARIABLE.match(word)
    if match is None:
        return [word]
    value = variables.get(match.group(1) or match.group(2), "1")
    return shlex.split(value) if " " in value.strip() else ["1"]


def command_words(line: str, helpers: set[str], variables: Optional[dict[str, str]] = None) -> list[str] | None:
    """The arguments ``line`` passes to ``ruleforge``, or None when it runs no ``ruleforge`` command."""
    try:
        words = shlex.split(PLACEHOLDER.sub("1", line), comments=True)
    except ValueError as error:
        if _arguments(line.split(), helpers) is not None:
            raise ValueError(f"the shell cannot split it ({error})") from None
        return None
    found = _arguments(words, helpers)
    if found is None:
        return None
    ended = next((index for index, word in enumerate(found) if SHELL_END.match(word)), len(found))
    words = found[:ended]
    words = words[:words.index("--")] if "--" in words else words
    return [item for word in words for item in _word(word, variables or {})]


def _arguments(words: list[str], helpers: set[str]) -> list[str] | None:
    while words and (ASSIGNMENT.match(words[0]) or words[0] in ("nohup", "setsid")):
        words = words[1:]
    if words[:1] and words[0] in helpers:
        words = ["ruleforge", "jobs", "launch", *words[1:]]
    if words[:2] == ["uv", "run"]:
        rest = words[2:]
        while rest and rest[0].startswith("-"):
            rest = rest[2:] if rest[0] in ("--group", "--with", "--extra", "--project") else rest[1:]
        words = rest
    return words[1:] if words[:1] == ["ruleforge"] else None


def documented_commands() -> list[tuple[str, str, list[str] | str]]:
    """Every ``ruleforge`` command the documents' code blocks give."""
    found = []
    for path in _documents():
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        helpers = {match.group(1) for line in text.splitlines() if (match := HELPER.match(line))}
        blocks = _blocks(text)
        variables = {name: value for block in blocks for name, value in assignments(block).items()}
        for block in blocks:
            for line in block:
                try:
                    words = command_words(line, helpers, variables)
                except ValueError as error:
                    found.append((path, line.strip(), str(error)))
                    continue
                if words is not None:
                    found.append((path, line.strip(), words))
    return found


@requires_git_worktree  # reads `git ls-files` through _tracked
def test_every_documented_command_parses(capsys):
    commands = documented_commands()
    assert len(commands) >= 40
    documents = {path for path, _, _ in commands}
    assert {"README.md", "docs/reproduce.md", "data/hub_cards/dataset.md"} <= documents
    parser = build_parser()
    refused = []
    for path, line, words in commands:
        if isinstance(words, str):
            refused.append(f"{path}: {line}: {words}")
            continue
        try:
            parser.parse_args(words)
        except SystemExit:
            refused.append(f"{path}: {line}: {capsys.readouterr().err.strip().splitlines()[-1]}")
    assert refused == []


@pytest.mark.parametrize("line, words", [
    ('uv run --no-sync ruleforge analyze "$ROOT" --replace', ["analyze", "1", "--replace"]),
    ("ruleforge jobs launch job --spec x $H --var y=<a placeholder>", ["jobs", "launch", "job", "--spec", "x",
                                                                       "--var", "a=1", "--var", "y=1"]),
    ("RULEFORGE_X=1 uv run --group grpo ruleforge grpo train --model <dir> -- a=b  # a comment", ["grpo", "train", "--model",
                                                                                          "1"]),
    ('launch job --spec x --var a=$B', ["jobs", "launch", "job", "--spec", "x", "--var", "a=$B"]),
    ("nohup setsid ruleforge version > v.log 2>&1 &", ["version"]),
    ("ruleforge version 2>&1 | tee v.log", ["version"]),
    ("ruleforge version && ruleforge tasks list", ["version"]),
    ("uv run --no-sync python -m ruleforge.hub pack", None),
    ('echo "unclosed', None),
])
def test_a_command_line_is_read_as_the_shell_gives_it(line, words):
    assert command_words(line, {"launch"}, assignments(['H="--var a=1"', "N=2 M=3", "not an assignment"])) == words


@pytest.mark.parametrize("line", ['ruleforge reproduce all verify --rerun "unterminated',
                                  "uv run --no-sync ruleforge version 'open", "X=1 nohup launch job --spec 'x"])
def test_a_command_the_shell_cannot_split_fails_the_gate_rather_than_dropping_out(line):
    with pytest.raises(ValueError, match="the shell cannot split it"):
        command_words(line, {"launch"})


def test_an_unsplittable_command_is_reported_as_refused(monkeypatch, tmp_path, capsys):
    (tmp_path / "doc.md").write_text('```bash\nruleforge version "open\nruleforge version\n```\n', encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "_documents", lambda: ["doc.md"])
    assert documented_commands() == [("doc.md", 'ruleforge version "open', "the shell cannot split it (No closing "
                                      "quotation)"), ("doc.md", "ruleforge version", ["version"])]


def test_a_code_block_s_continuation_lines_are_one_command():
    text = "x\n  ```bash\n  ruleforge reproduce all \\\n      render --out d\n  ```\n```\nruleforge version\n```\n"
    assert _blocks(text) == [["  ruleforge reproduce all  render --out d"], ["ruleforge version"]]


def resolves(name: str) -> bool:
    """Whether the dotted ``name`` is an importable module or an attribute of one."""
    parts = name.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            found = importlib.import_module(".".join(parts[:cut]))
        except ModuleNotFoundError as error:
            if error.name is not None and not ".".join(parts[:cut]).startswith(error.name):
                raise
            continue
        for attribute in parts[cut:]:
            if not hasattr(found, attribute):
                return False
            found = getattr(found, attribute)
        return True
    return False


def _names() -> dict[tuple[str, str], int]:
    """Each (file, dotted name) the Python files import or name in a string constant, with its first line."""
    found: dict[tuple[str, str], int] = {}
    for path in _tracked():
        if not path.endswith(".py") or not path.startswith(CODE_DIRS) or not (REPO_ROOT / path).is_file():
            continue
        for node in ast.walk(ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [f"{node.module}.{alias.name}" for alias in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and path.startswith("src/"):
                names = [node.value.replace(":", ".")] if MODULE.fullmatch(node.value) else []
            else:
                continue
            for name in names:
                if name.startswith("ruleforge.") and name not in RECORDED:
                    found.setdefault((path, name), node.lineno)
    return found


@requires_git_worktree  # reads `git ls-files` through _tracked
def test_every_dotted_ruleforge_name_resolves():
    names = _names()
    assert len(names) >= 100
    unresolved = sorted(key for key in names if not resolves(key[1]))
    assert [f"{path}:{names[(path, name)]}: {name}" for path, name in unresolved] == []


MODULE_PATH = re.compile(r"(?<![\w./:-])((?:src|tests|grpo|evaluate|core|cli|launch|prompts|optimize|tasks|judges|rulesets"
                         r"|scenarios|sampling|experiments)/[A-Za-z0-9_./-]+\.py)\b")


def _stale_module_paths(paths: list[str]) -> list[str]:
    return [f"{path}:{number}: {found}" for path in paths
            for number, line in enumerate((REPO_ROOT / path).read_text(encoding="utf-8").splitlines(), 1)
            for found in MODULE_PATH.findall(line)
            if not ((REPO_ROOT / found).is_file() or (REPO_ROOT / "src/ruleforge" / found).is_file())]


@requires_git_worktree  # reads `git ls-files` through _tracked
def test_every_module_path_a_document_names_exists():
    assert _stale_module_paths(_documents_and_configs()) == []


def _command_lines(page: str, section: Optional[str] = None) -> list[str]:
    text = (REPO_ROOT / "docs" / page).read_text(encoding="utf-8")
    if section is not None:
        text = text.split(f"\n## {section}\n", 1)[1].split("\n## ", 1)[0]
    return [" ".join(line.split()) for block in _blocks(text) for line in block]


#: A stand-in for the artifact root, where the documents' fetched data lands.
FETCHED = Path("/fetched-data")


def _documented(verb: tuple[str, ...]) -> list[tuple[str, list[str]]]:
    found = []
    for path in _documents():
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        helpers = {match.group(1) for line in text.splitlines() if (match := HELPER.match(line))}
        for line in (line for block in _blocks(text) for line in block if " ".join(verb) in line):
            line = re.sub(r"\$\{?PWD\}?", str(REPO_ROOT), line)
            line = re.sub(rf"\$\{{?{ARTIFACT_ROOT_ENV}\}}?", str(FETCHED), line)
            words = _arguments(shlex.split(PLACEHOLDER.sub("1", line), comments=True), helpers) or []
            words = words[:next((k for k, word in enumerate(words) if SHELL_END.match(word)), len(words))]
            if tuple(words[:len(verb)]) == verb:
                found.append((path, words))
    return found


class Reached(Exception):
    """Raised where a driven command has been checked, before it reads or writes anything more."""


def _no_training(*args, **kwargs):
    raise AssertionError("a driven training command reached the trainer")


def _training_ruleset(words: list[str], monkeypatch) -> Optional[Path]:
    from ruleforge.cli import app, grpo

    real, composed = grpo.compose, []

    def compose(*args, **kwargs):
        composed.append(real(*args, **kwargs))
        raise Reached

    trainer = types.ModuleType("ruleforge.grpo.trainer")
    trainer.export_submitit_distributed_env = trainer.run_grpo = _no_training
    with monkeypatch.context() as patch:
        patch.setattr(grpo, "compose", compose)
        patch.setitem(sys.modules, "ruleforge.grpo.trainer", trainer)
        patch.chdir(REPO_ROOT)  # the documents' commands run from the checkout's top
        try:
            app.main(words)
        except Reached:
            return Path(composed[0].data.rules)
    return None


def _selection_destination(words: list[str], monkeypatch) -> Optional[Path]:
    from ruleforge.cli import app, optimize

    destinations = []

    def check_outputs(outputs, **kwargs):
        destinations.extend(outputs)
        raise Reached

    with monkeypatch.context() as patch:
        patch.setattr(optimize.records, "check_outputs", check_outputs)
        patch.chdir(REPO_ROOT)
        try:
            app.main(words)
        except Reached:
            return Path(destinations[0])
    return None


def _fetched(path: Path | str) -> bool:
    return (REPO_ROOT / path).resolve().is_relative_to(FETCHED.resolve())


@requires_git_worktree  # reads `git ls-files` through _tracked
def test_every_documented_training_command_reads_a_ruleset_that_exists(monkeypatch):
    commands = _documented(("grpo", "train"))
    assert commands
    for path, words in commands:
        rules = _training_ruleset(words, monkeypatch)
        assert rules is not None and rules.is_file(), (path, " ".join(words), rules)


@requires_git_worktree  # reads `git ls-files` through _tracked
def test_a_documented_selection_of_fetched_sweeps_is_written_outside_the_fetched_data(monkeypatch):
    commands = [(path, words) for path, words in _documented(("optimize", "select"))
                if all(_fetched(sweep) for sweep in build_parser().parse_args(words).sweep_dirs)]
    assert commands
    for path, words in commands:
        destination = _selection_destination(words, monkeypatch)
        assert destination is not None and not _fetched(destination), (path, " ".join(words), destination)


@requires_git_worktree  # reads `git ls-files` through _tracked
def test_a_training_command_the_cli_refuses_reads_no_ruleset(monkeypatch):
    [(_, words), *_] = _documented(("grpo", "train"))
    assert _training_ruleset(words, monkeypatch) is not None
    assert _training_ruleset([*words, "run_name=OTHER"], monkeypatch) is None


@pytest.mark.parametrize("output, fetched", [
    (["--output", "DIR/selection.json"], False),
    ([], True),
    (["--output", f"/tmp/..{FETCHED}/ruleforge/selection.json"], True),
    (["--output", f"{FETCHED}/../elsewhere/selection.json"], False),
])
def test_a_selection_is_placed_where_the_cli_writes_it_compared_as_resolved_paths(output, fetched, monkeypatch):
    words = ["optimize", "select", "--sweep-dir", f"{FETCHED}/ruleforge/sweep", *output]
    assert _fetched(_selection_destination(words, monkeypatch)) is fetched


def test_the_verify_line_promises_as_many_goldens_as_the_table_holds():
    (verify,) = [line for line in _command_lines("reproduce.md") if "reproduce all verify --rerun" in line]
    assert f"ALL {len(table.goldens())} EQUAL" in verify


TICKED = re.compile(r"`([^`]+)`")


def _paper_table() -> list[list[str]]:
    lines = (REPO_ROOT / "docs" / "reproduce.md").read_text(encoding="utf-8").splitlines()
    start = next(k for k, line in enumerate(lines) if line.startswith("| Paper | Experiment | Golden | Files from"))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def _stem(name: str) -> str:
    return re.split(r"[./]", name, maxsplit=1)[0]


def test_the_paper_table_names_every_golden_and_the_stem_of_every_file_render_writes():
    rows = _paper_table()
    named = {word for row in rows for word in TICKED.findall(row[2])}
    assert [golden.path for golden in table.goldens() if golden.path not in named] == []
    for experiment in table.TABLE:
        listed = {_stem(word) for row in rows if row[1] == f"`{experiment.name}`" for word in TICKED.findall(row[3])}
        written = {_stem(word[len("{out}/"):]) for plot in experiment.plots for word in plot.words
                   if word.startswith("{out}/")}
        assert sorted(written - listed) == [], experiment.name


#: A figure, table or appendix section a cell of the paper table names, without its panels.
PAPER_PART = re.compile(r"\b(Figure \d+|Table \d+|Appendix [A-Z]\.\d+)")


def test_the_paper_table_names_each_experiment_s_figures_and_tables_as_the_experiments_table_does():
    rows = _paper_table()
    for experiment in table.TABLE:
        named = {part for row in rows if row[1] == f"`{experiment.name}`" for part in PAPER_PART.findall(row[0])}
        assert sorted(named) == sorted(experiment.paper), experiment.name


def test_the_prompt_ablation_rerun_runs_every_arm_the_arm_map_reads():
    def spec_vars(name: str) -> dict:
        return yaml.safe_load((REPO_ROOT / "configs" / "launch" / f"{name}.yaml").read_text(encoding="utf-8"))["vars"]

    def judge_of(model: str) -> str:
        path = REPO_ROOT / "configs" / "model" / f"{model}.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))["model_name"]

    ran = set()
    for line in _command_lines("rerun.md", "Prompt ablation"):
        spec = re.search(r"jobs launch campaign --spec (\w+)", line)
        if spec is None or "--preflight-only" in line:
            continue
        settings = {**spec_vars(spec.group(1)), **dict(re.findall(r"--var (\w+)=(\S+)", line))}
        models = settings["models"].split(",") if "models" in settings else [settings["model"]]
        ran |= {(judge_of(model), style) for model in models for style in settings["styles"].split(",")}
    arms = yaml.safe_load((REPO_ROOT / "configs" / "paper" / "prompt_ablation_arms.yaml").read_text(encoding="utf-8"))
    wanted = {(judge, arm["condition"].split("/")[1]) for judge, row in arms["judges"].items()
              for arm in row.values() if isinstance(arm, dict)}
    assert len(wanted) == 18 and wanted - ran == set()


def test_a_name_resolves_to_a_module_or_an_attribute():
    assert resolves("ruleforge.cli.app") and resolves("ruleforge.cli.app.build_parser")
    assert resolves("ruleforge.evaluate.api.Campaign.create")
    assert not resolves("ruleforge.cli.app.no_such_name") and not resolves("ruleforge.no_such_module")


@requires_git_worktree  # reads `git ls-files` through _tracked
def test_no_document_names_scripts():
    named = {}
    for path in _documents_and_configs():
        found = set(SCRIPTS.findall((REPO_ROOT / path).read_text(encoding="utf-8")))
        if found:
            named[path] = found
    assert named == {}
    # git tracks nothing under the directory (a checkout may keep what git does not track there: bytecode, an NFS lock)
    assert [path for path in _tracked() if path.startswith("scripts/")] == []


def test_the_extending_page_names_the_extension_points_the_code_has():
    from ruleforge import judges, tasks

    page = (REPO_ROOT / "docs" / "extending.md").read_text(encoding="utf-8")
    quoted = set(re.findall(r"`([^`\n]+)`", page))
    assert {"ruleforge.tasks.Task", "ruleforge.tasks.OptimizableTask", "ruleforge.tasks.register_task",
            tasks.ENTRY_POINT_GROUP, "tests/fixtures/legal/", "ruleforge.judges.Provider",
            "ruleforge.judges.register_judge", judges.ENTRY_POINT_GROUP, "ruleforge optimize"} <= quoted
    assert all(resolves(name) for name in quoted if NAME.fullmatch(name))
    assert all(name in quoted for name in tasks.BUILTIN_TASKS)
    assert [name for name in quoted if re.search(r"(?:^|\.)_[A-Za-z]", name)] == []
    paths = [name.split("<", 1)[0] for name in quoted if re.match(r"(?:src|tests|configs|data|docs)/", name)]
    assert paths and [path for path in paths if not (REPO_ROOT / path).exists()] == []
