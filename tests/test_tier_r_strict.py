from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from tests import conftest
from tests.test_repro_hub import _write

REPO = Path(__file__).resolve().parents[1]

def test_the_committed_skips_file_is_in_its_format():
    assert conftest.TIER_R_SKIPS == REPO / "tests" / "tier_r_skips.txt"
    for nodeid in conftest.read_tier_r_skips():
        module, _, name = nodeid.partition("::")
        assert (REPO / module).is_file(), nodeid
        function = name.rpartition("::")[2].split("[")[0]
        assert not name or f"def {function}(" in (REPO / module).read_text(encoding="utf-8"), nodeid


@pytest.mark.parametrize(("text", "message"), [
    ("tests/a.py::t\tr", "does not end with a newline"),
    ("tests/a.py::t\n", "line 1 is not nodeid<TAB>reason"),
    ("tests/a.py::t\t \n", "line 1 is not nodeid<TAB>reason"),
    ("\tr\n", "line 1 is not nodeid<TAB>reason"),
    ("tests/a.py::t\tr\textra\n", "line 1 is not nodeid<TAB>reason"),
    ("tests/a.py::t\tr\n\n", "line 2 is not nodeid<TAB>reason"),
    ("tests/b.py::t\tr\ntests/a.py::t\tr\n", "not sorted, each once"),
    ("tests/a.py::t\tr\ntests/a.py::t\ts\n", "not sorted, each once"),
])
def test_a_skips_file_out_of_its_format_is_refused(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        conftest.read_tier_r_skips(_write(tmp_path / "skips.txt", text))


def test_a_skips_file_reads_as_its_node_ids_and_reasons(tmp_path):
    text = "tests/a.py::t[x]\tan environment's skip\ntests/b.py\tno statsmodels\n"
    assert conftest.read_tier_r_skips(_write(tmp_path / "skips.txt", text)) == {
        "tests/a.py::t[x]": "an environment's skip", "tests/b.py": "no statsmodels"}
    assert conftest.read_tier_r_skips(_write(tmp_path / "empty.txt", "")) == {}


def _report(cls, nodeid, outcome, longrepr=None, **extra):
    report = cls(nodeid, ("f", 1, "t"), {}, outcome, longrepr, "call") if cls is pytest.TestReport else \
        cls(nodeid, outcome, longrepr, [])
    for key, value in extra.items():
        setattr(report, key, value)
    return report


def test_a_strict_run_refuses_an_unlisted_skip_of_a_cluster_test_or_a_module():
    skip = ("f.py", 3, "Skipped: no artifact root")
    listed = {"tests/a.py::listed": "attributed", "tests/m.py": "attributed"}
    refuse = conftest.tier_r_refusal
    test = pytest.TestReport
    assert "tests/a.py::t was skipped (Skipped: no artifact root)" in refuse(
        _report(test, "tests/a.py::t", "skipped", skip), cluster=True, listed=listed)
    assert refuse(_report(test, "tests/a.py::listed", "skipped", skip), cluster=True, listed=listed) is None
    assert refuse(_report(test, "tests/a.py::t", "skipped", skip), cluster=False, listed=listed) is None
    assert refuse(_report(test, "tests/a.py::t", "passed"), cluster=True, listed=listed) is None
    assert refuse(_report(test, "tests/a.py::t", "skipped", skip, wasxfail="known"), cluster=True,
                  listed=listed) is None
    collect = pytest.CollectReport
    assert "tests/n.py was skipped" in refuse(_report(collect, "tests/n.py", "skipped", skip), cluster=True,
                                              listed=listed)
    assert refuse(_report(collect, "tests/m.py", "skipped", skip), cluster=True, listed=listed) is None
    assert "tests/n.py was skipped (a reason)" in refuse(_report(collect, "tests/n.py", "skipped", "a reason"),
                                                         cluster=True, listed=listed)


PROBES = '''
import pytest


@pytest.mark.cluster
def test_listed_skip():
    pytest.skip("listed")


@pytest.mark.cluster
def test_unlisted_skip():
    pytest.skip("not listed")


@pytest.mark.cluster
@pytest.mark.skipif(True, reason="a mark's skip, at setup")
def test_unlisted_mark_skip():
    pass


@pytest.mark.cluster
def test_passes():
    pass


@pytest.mark.cluster
@pytest.mark.xfail(reason="known")
def test_xfails():
    raise AssertionError


def test_unmarked_skip():
    pytest.skip("no cluster test")
'''


def _suite_conftest(checkout: Path) -> None:
    for name in ("conftest.py", "support/__init__.py", "support/slurm_guard.py"):
        (checkout / "tests" / name).parent.mkdir(parents=True, exist_ok=True)
        (checkout / "tests" / name).write_bytes((REPO / "tests" / name).read_bytes())


def _probe(tmp_path: Path, *options: str) -> dict[str, str]:
    """The suite's conftest run by a pytest of its own over probe tests and a skips file: each test's outcome."""
    checkout = tmp_path / "checkout"
    _suite_conftest(checkout)
    (checkout / "tests/tier_r_skips.txt").write_text("tests/test_probes.py::test_listed_skip\tattributed\n"
                                                     "tests/test_skipped_listed.py\tattributed\n")
    (checkout / "tests/test_probes.py").write_text(PROBES)
    for name in ("listed", "unlisted"):
        (checkout / f"tests/test_skipped_{name}.py").write_text("import pytest\n\npytest.skip('at collection', "
                                                                "allow_module_level=True)\n")
    (checkout / "pyproject.toml").write_text('[tool.pytest.ini_options]\nmarkers = ["cluster: Tier R"]\n')
    _write(tmp_path / "shadow/peft.py", "raise ImportError('shadowed')\n")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))}
    env["PYTHONPATH"] = str(tmp_path / "shadow")
    result = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:xdist", "-q", "-rA",
                             f"--junitxml={tmp_path / 'j.xml'}", *options, "tests"], cwd=checkout, env=env,
                            capture_output=True, text=True, timeout=600)
    outcomes = {}
    for case in ElementTree.parse(tmp_path / "j.xml").iter("testcase"):
        kinds = [child.tag for child in case if child.tag in ("failure", "error", "skipped")]
        name = ".".join(part for part in (case.get("classname"), case.get("name")) if part)
        outcomes[name] = kinds[0] if kinds else "passed"
    outcomes["exit"] = str(result.returncode)
    return outcomes


def test_tier_r_strict_fails_each_unlisted_skip_and_passes_the_rest(tmp_path):
    outcomes = _probe(tmp_path, "--tier-r-strict")
    probes = "tests.test_probes."
    assert {name: kind for name, kind in outcomes.items() if name.startswith(probes)} == {
        probes + "test_listed_skip": "skipped", probes + "test_unlisted_skip": "failure",
        probes + "test_unlisted_mark_skip": "error", probes + "test_passes": "passed",
        probes + "test_xfails": "skipped", probes + "test_unmarked_skip": "skipped"}
    assert (outcomes["tests.test_skipped_listed"], outcomes["tests.test_skipped_unlisted"]) == ("skipped", "error")
    assert outcomes["exit"] == "1"


def test_without_tier_r_strict_a_skip_is_a_skip(tmp_path):
    outcomes = _probe(tmp_path)
    assert outcomes.pop("exit") == "0"
    assert "failure" not in outcomes.values() and "error" not in outcomes.values()


def test_tier_r_strict_refuses_a_skips_file_out_of_its_format(tmp_path):
    checkout = tmp_path / "checkout"
    _suite_conftest(checkout)
    (checkout / "tests/tier_r_skips.txt").write_text("no tab\n")
    (checkout / "tests/test_x.py").write_text("def test_x():\n    pass\n")
    _write(tmp_path / "shadow/peft.py", "raise ImportError('shadowed')\n")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))}
    env["PYTHONPATH"] = str(tmp_path / "shadow")
    result = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:xdist",
                             "--tier-r-strict", "tests"], cwd=checkout, env=env, capture_output=True, text=True,
                            timeout=600)
    assert result.returncode == 4 and "--tier-r-strict:" in result.stderr and "line 1 is not" in result.stderr
