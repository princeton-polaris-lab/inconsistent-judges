from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ruleforge.cli import app
from ruleforge.cli.compose import compose
from ruleforge.cli.evaluate import BANK_PATHS, COMPONENT_KINDS, bank_operation
from ruleforge.cli.schema import JUDGE_PATH_OPTIONS, PATH, annotations_of, defaults_of, path_kind


def _paths_of(target) -> set[str]:
    return {name for name, annotation in annotations_of(target).items()
            if path_kind(annotation) is not None and name in defaults_of(target)}


def _function(dotted: str):
    import importlib

    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _parser(words) -> argparse.ArgumentParser:
    parser = app.build_parser()
    for word in words:
        parser = next(action for action in parser._actions
                      if isinstance(action, argparse._SubParsersAction)).choices[word]
    parser.format_help()  # a mounted step's flags are read when it is first formatted or parsed
    return parser


#: Each command, the API function it calls, and which of the function's path parameters each flag (by dest) gives.
COMMANDS = [
    (("campaign", "collect"), "ruleforge.judges.batch_collect.collect", {"directory": "directories"}),
    (("campaign", "cost", "usage"), "ruleforge.evaluate.cost.campaign_usage",
     {"roots": "root", "batch_root": "batch_root", "rates": "rates", "scenarios": "scenarios"}),
    (("campaign", "merge-shards"), "ruleforge.evaluate.shard_merge.merge_condition_shards",
     {"campaign_root": "campaign_root", "shards_dir": "shards_dir"}),
    (("scenarios", "concat"), "ruleforge.scenarios.subset.build_concatenation", {"sources": "source", "output": "out"}),
    (("scenarios", "subtract"), "ruleforge.scenarios.subset.build_difference",
     {"source": "source", "output": "out", "exclude": "exclude"}),
    (("scenarios", "partition"), "ruleforge.scenarios.subset.build_blocks", {"source": "source", "out_dir": "out_dir"}),
    (("optimize", "select"), "ruleforge.optimize.select.sweep_table", {"sweep_dirs": "sweep_dirs"}),
    (("grpo", "exclusion", "attest"), "ruleforge.grpo.exclusion.attest",
     {"train": "dataset", "protected": "protected", "carved": "carved"}),
    (("bank", "analyze"), "ruleforge.evaluate.bank.analyze", {"root": "root", "bank_root": "bank_root"}),
]


@pytest.mark.parametrize("words, function, flags", COMMANDS, ids=[" ".join(words) for words, _, _ in COMMANDS])
def test_every_path_parameter_of_a_command_s_api_is_one_of_its_flags(words, function, flags):
    assert _paths_of(_function(function)) == set(flags)
    dests = {action.dest for action in _parser(words)._actions}
    assert set(flags.values()) <= dests


@pytest.mark.parametrize("spelling", ["blocks", "~/blocks"])
def test_a_path_flag_is_the_working_directory_s(cli, tmp_path, monkeypatch, spelling):
    import ruleforge.scenarios.subset as subset

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    seen = []
    block = {"population_count": 2, "source_sha256": "x", "row_range": [0, 2], "output_name": "b", "output_sha256": "y"}
    monkeypatch.setattr(subset, "build_blocks", lambda source, out_dir, **kwargs: seen.append((source, out_dir)) or [
        block])
    status, _, error = cli("scenarios", "partition", "--source", "p.jsonl", "--block-size", "2",
                           f"--out-dir={spelling}")
    out = tmp_path / "home" / "blocks" if spelling.startswith("~") else Path.cwd() / "blocks"
    assert status == 0 and seen == [(Path("p.jsonl"), out)], error
    assert Path(f"{out}.runs").is_dir()


def test_bank_run_s_path_keys_are_the_subsets_reader_s_and_its_config_s():
    from ruleforge.evaluate.bank import verify_subsets

    operation = bank_operation()
    assert {parameter.name for parameter in operation.parameters} == _paths_of(verify_subsets)
    assert operation.path_keys() == {
        "ruleset_size.subset_root": PATH, "ruleset_size.subset_manifest_dir": PATH, **BANK_PATHS,
        **{f"{node}.{option}": PATH for node in ("judge_model", "agent_model") for option in JUDGE_PATH_OPTIONS}}


def test_a_bank_config_reads_its_relative_paths_from_its_own_directory(tmp_path):
    configs, elsewhere = tmp_path / "configs", tmp_path / "elsewhere"
    configs.mkdir()
    elsewhere.mkdir()
    (configs / "ruleset_size.yaml").write_text("rule_set_path: rules.jsonl\nscenario_set_path: null\nruleset_size:\n"
                                      "  subset_root: subsets\n"
                                      "  subset_manifest_dir: ../manifest\n", encoding="utf-8")
    from ruleforge.cli.compose import parse_overrides

    overrides = parse_overrides(["scenario_set_path=scenarios.jsonl"], bank_operation(), elsewhere)
    cfg = compose(bank_operation(), OmegaConf.create({}), config_file=configs / "ruleset_size.yaml", layers=(overrides,),
                  cwd=elsewhere)
    assert cfg.rule_set_path == str(configs / "rules.jsonl")
    assert (cfg.ruleset_size.subset_root, cfg.ruleset_size.subset_manifest_dir) == (str(configs / "subsets"),
                                                                          str(tmp_path / "manifest"))
    assert cfg.scenario_set_path == str(elsewhere / "scenarios.jsonl")
    # A tree written for another directory holds its files alone: this file is in none, so it reads from its own.
    again = compose(bank_operation(), OmegaConf.create({}), config_file=configs / "ruleset_size.yaml", layers=(overrides,),
                    cwd=elsewhere, config_bases={tmp_path / "checkout" / "configs": tmp_path / "checkout" / "scripts"})
    assert again == cfg


def test_the_checkout_s_configs_are_read_from_its_top_and_none_outside_a_checkout(monkeypatch):
    import ruleforge.core.paths as paths
    from ruleforge.cli.evaluate import checkout_configs
    from ruleforge.core.errors import ArtifactRootError

    assert checkout_configs() == {paths.repository_root() / "configs": paths.repository_root()}

    def no_checkout():
        raise ArtifactRootError("no checkout")

    monkeypatch.setattr(paths, "repository_root", no_checkout)
    assert checkout_configs() == {}


def test_the_choices_a_command_lists_are_the_library_s():
    """Listed by the command line without importing the modules that define them."""
    from ruleforge.evaluate.identity import COMPONENT_KINDS as KINDS

    assert COMPONENT_KINDS == tuple(KINDS)
