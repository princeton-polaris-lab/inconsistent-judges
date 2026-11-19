from __future__ import annotations

import json

import pytest

from ruleforge.evaluate import cost as cost_module
from tests.test_sampling_batch_cost import _roots  # noqa: F401  (the batch root, one code state: the fixtures')
from tests.test_sampling_batch_cost import SOL
from tests.test_sampling_batch_cost import roots as batch_roots  # noqa: F401  (the fixture, by this name)


@pytest.mark.parametrize("extra", [
    (),
    ("--group-by", "campaign", "--price", SOL),
    ("--component-kind", "rule", "--price", SOL),
    ("--condition", "gemma-4-31B-it/naive"),
])
def test_usage_prints_and_records_every_condition_priced(cli, batch_roots, tmp_path, extra):  # noqa: F811
    out = tmp_path / "usage.json"
    status, text, _ = cli("campaign", "cost", "usage", "--root", batch_roots["sol"], "--root", batch_roots["local"],
                          *extra, "--out", out)
    assert status == 0 and "unattributed" in text and f"record: {out}" in text
    record = json.loads(out.read_text())
    assert record["kind"] == "usage" and record["rates"]["sha256"]
    priced = {rate for group in record["groups"] for rate in group["cost"]}
    assert ("--price" not in extra) or SOL in priced
    if "--group-by" in extra:
        assert record["group_by"] == "campaign" and {group["label"] for group in record["groups"]} == {"sol", "local"}
    if "--component-kind" in extra:
        assert record["component_kind"] == "rule" and text.startswith("rule requests only")
    if "--condition" in extra:
        assert record["selected_conditions"] == ["gemma-4-31B-it/naive"] and len(record["conditions"]) == 1


def test_usage_over_some_scenarios_reads_each_request_from_the_batch_outputs(cli, batch_roots, tmp_path):  # noqa: F811
    cut = tmp_path / "cut.jsonl"
    cut.write_text((tmp_path / "scenarios.jsonl").read_text().splitlines(keepends=True)[0])
    out = tmp_path / "usage.json"
    status, text, _ = cli("campaign", "cost", "usage", "--root", batch_roots["sol"], "--scenarios", cut, "--price", SOL,
                          "--out", out)
    assert status == 0 and text.startswith(f"over the 1 scenarios of {cut} only")
    record = json.loads(out.read_text())
    assert record["scenarios"]["count"] == 1 and record["scenarios"]["sha256"]


def test_a_missing_batch_exits_1_and_says_so(cli, batch_roots):  # noqa: F811
    key = {condition.label: condition for condition in cost_module.attribute(
        [batch_roots["luna"]], batch_roots["batch"]).conditions}["luna/gpt-5.6-luna/naive"].batches[0]
    (batch_roots["batch"] / key / "usage.json").unlink()
    status, text, _ = cli("campaign", "cost", "usage", "--root", batch_roots["luna"])
    assert status == 1 and "MISSING: 1 batch(es)" in text


def test_usage_refuses_what_it_cannot_price_or_find(cli, batch_roots, monkeypatch, tmp_path):  # noqa: F811
    status, _, error = cli("campaign", "cost", "usage", "--root", batch_roots["sol"], "--price", "gpt-9")
    assert status == 2 and "has no rates for ['gpt-9']" in error
    status, _, error = cli("campaign", "cost", "usage", "--root", batch_roots["sol"], "--condition", "nope")
    assert status == 2 and "--condition ['nope']: no --root campaign holds it" in error
    status, _, error = cli("campaign", "cost", "usage", "--root", batch_roots["sol"], "--scenarios", tmp_path / "nowhere")
    assert status == 2 and "is not a file" in error
    monkeypatch.delenv("RULEFORGE_BATCH_ROOT")
    status, _, error = cli("campaign", "cost", "usage", "--root", batch_roots["sol"], "--root", batch_roots["luna"])
    assert status == 2 and "--batch-root is required with several --root" in error


def test_one_root_s_usage_falls_back_to_its_own_batch_directory(cli, batch_roots, monkeypatch):  # noqa: F811
    monkeypatch.delenv("RULEFORGE_BATCH_ROOT")
    batch_roots["batch"].rename(batch_roots["sol"] / "batch")
    status, text, _ = cli("campaign", "cost", "usage", "--root", batch_roots["sol"], "--price", SOL)
    assert status == 0 and "MISSING" not in text


def test_a_record_inside_the_batch_root_it_reads_is_refused(cli, batch_roots):  # noqa: F811
    """Its run records beside it would be read as one of that root's batches the next time."""
    status, _, error = cli("campaign", "cost", "usage", "--root", batch_roots["sol"], "--out",
                           batch_roots["batch"] / "usage.json")
    assert status == 2 and "is inside the batch root" in error
    assert not (batch_roots["batch"] / "usage.json").exists() and not (batch_roots["batch"] / "usage.json.runs").exists()


def test_the_default_rates_are_the_package_s():
    import tomllib
    from pathlib import Path

    from ruleforge.cli import evaluate
    from ruleforge.env import package_path

    assert evaluate._rates(None) == package_path(evaluate.RATES) == cost_module.DEFAULT_RATES
    assert evaluate._rates(Path("mine.yaml")) == Path("mine.yaml")
    repo = Path(__file__).resolve().parents[2]
    wheel = tomllib.loads((repo / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["force-include"][evaluate.RATES] == f"ruleforge/{evaluate.RATES}"
