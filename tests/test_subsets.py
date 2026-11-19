from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from ruleforge.rulesets.subsets import (
    WITH_REPLACEMENT,
    WITH_REPLACEMENT_SHUFFLED_FULL,
    WITHOUT_REPLACEMENT,
    build_manifest,
    create_subsets,
    diff_manifest,
    file_digest,
    generate_subset_files,
    parse_manifest,
    load_manifest,
    sample_independent,
    sample_nested,
    seed_dir,
    strategy_root,
)
from tests.support.subsets import render_manifest

MANIFEST_DIR = "data/rules/subset_manifest"
SHUFFLED_FULL_MANIFEST_DIR = "data/rules/subset_manifest_shuffled_full"


@pytest.fixture
def ruleset(tmp_path):
    """An eight-rule set, written the way the real ones are."""
    path = tmp_path / "rules.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": f"openai_rules{n}", "rule": f"body {n}"}) + "\n"
            for n in range(1, 9)
        ),
        encoding="utf-8",
    )
    return path


def _rules(count=8):
    return [{"id": f"openai_rules{n}", "rule": f"body {n}"} for n in range(1, count + 1)]


def test_with_replacement_covers_every_size_but_the_full_set():
    subsets = sample_independent(_rules(), seed=0)
    assert [s["size"] for s in subsets] == [2, 3, 4, 5, 6, 7]
    assert all(len(s["rules"]) == s["size"] for s in subsets)


def test_with_replacement_subsets_need_not_be_nested():
    subsets = sample_independent(_rules(20), seed=3)
    nested = all(
        {r["id"] for r in small["rules"]} <= {r["id"] for r in large["rules"]}
        for small, large in zip(subsets, subsets[1:])
    )
    assert not nested


def test_without_replacement_subsets_are_nested():
    subsets = sample_nested(_rules(), seed=0)
    for small, large in zip(subsets, subsets[1:]):
        assert {r["id"] for r in small["rules"]} < {r["id"] for r in large["rules"]}


def test_sampling_draws_without_repeats_inside_a_subset():
    for subset in sample_independent(_rules(), seed=1):
        ids = [rule["id"] for rule in subset["rules"]]
        assert len(ids) == len(set(ids))


def test_the_shuffled_full_strategy_draws_the_full_set_last():
    subsets = create_subsets(WITH_REPLACEMENT_SHUFFLED_FULL, _rules(), seed=0)
    assert [s["size"] for s in subsets] == [2, 3, 4, 5, 6, 7, 8]
    assert sorted(rule["id"] for rule in subsets[-1]["rules"]) == sorted(rule["id"] for rule in _rules())


@pytest.mark.parametrize("seed", [0, 1, 7, 16])
def test_the_shuffled_full_strategy_keeps_every_smaller_subset(seed):
    shuffled = create_subsets(WITH_REPLACEMENT_SHUFFLED_FULL, _rules(20), seed)
    assert shuffled[:-1] == create_subsets(WITH_REPLACEMENT, _rules(20), seed)


def test_each_seed_orders_the_full_set_its_own_way():
    orders = {tuple(rule["id"] for rule in create_subsets(WITH_REPLACEMENT_SHUFFLED_FULL, _rules(20), seed)[-1]["rules"])
              for seed in range(10)}
    assert len(orders) == 10
    assert tuple(rule["id"] for rule in _rules(20)) not in orders


@pytest.mark.parametrize("strategy", [WITH_REPLACEMENT, WITHOUT_REPLACEMENT, WITH_REPLACEMENT_SHUFFLED_FULL])
def test_the_same_seed_gives_the_same_subsets(strategy):
    assert create_subsets(strategy, _rules(), 7) == create_subsets(strategy, _rules(), 7)


def test_different_seeds_give_different_subsets():
    assert create_subsets(WITH_REPLACEMENT, _rules(20), 0) != create_subsets(
        WITH_REPLACEMENT, _rules(20), 1
    )


@pytest.mark.parametrize("strategy", ["", "nested", "WITH_REPLACEMENT"])
def test_an_unknown_strategy_is_rejected(strategy):
    with pytest.raises(ValueError, match="Invalid sampling_strategy"):
        create_subsets(strategy, _rules(), 0)


@pytest.mark.frozen
def test_the_sampler_draws_from_its_own_generator_as_the_published_subsets_did():
    rules = _rules(20)
    rng = random.Random(0)
    assert [subset["rules"] for subset in sample_independent(rules, seed=0)] == [
        rng.sample(list(rules), size) for size in range(2, 20)]
    assert sample_nested(rules, seed=3)[-1]["rules"] == random.Random(3).sample(list(rules), 20)[:19]


@pytest.mark.parametrize("strategy", [WITH_REPLACEMENT, WITHOUT_REPLACEMENT])
def test_sampling_never_seeds_the_module_rng(strategy, monkeypatch):
    monkeypatch.setattr(random, "seed", lambda *args, **kwargs: pytest.fail("seeded the module-global random"))
    assert len(create_subsets(strategy, _rules(20), 0)) == 18


def test_the_manifest_names_the_generator_the_files_are_drawn_with(repo_root):
    meta, _ = load_manifest(repo_root / MANIFEST_DIR)
    assert meta["reproducibility"]["rng"].startswith("random.Random(seed)")


def test_an_n_rule_set_yields_n_minus_one_files(ruleset, tmp_path):
    # Sizes 2..n-1 from the sampler, plus the full set at n.
    written = generate_subset_files(ruleset, tmp_path / "out", WITH_REPLACEMENT, 0)
    assert len(written) == 7
    assert [path.name for path in written][-1] == "subset_size_08.jsonl"


def test_the_baseline_file_is_the_whole_rule_set(ruleset, tmp_path):
    written = generate_subset_files(ruleset, tmp_path / "out", WITH_REPLACEMENT, 0)
    baseline = [p for p in written if p.name == "subset_size_08.jsonl"][0]
    assert baseline.read_text() == ruleset.read_text()


def test_the_shuffled_full_strategy_writes_its_drawn_full_set(ruleset, tmp_path):
    written = generate_subset_files(ruleset, tmp_path / "out", WITH_REPLACEMENT_SHUFFLED_FULL, 0)
    assert [path.name for path in written] == [f"subset_size_{size:02d}.jsonl" for size in range(2, 9)]
    full = [json.loads(line) for line in written[-1].read_text(encoding="utf-8").splitlines()]
    assert full == create_subsets(WITH_REPLACEMENT_SHUFFLED_FULL, _rules(), 0)[-1]["rules"]
    assert written[-1].read_text() != ruleset.read_text()


def test_sizes_are_zero_padded_so_they_sort_lexically(ruleset, tmp_path):
    written = generate_subset_files(ruleset, tmp_path / "out", WITH_REPLACEMENT, 0)
    names = [path.name for path in written]
    assert names == sorted(names)
    assert "subset_size_02.jsonl" in names


def test_generation_is_reproducible_byte_for_byte(ruleset, tmp_path):
    first = generate_subset_files(ruleset, tmp_path / "a", WITH_REPLACEMENT, 5)
    second = generate_subset_files(ruleset, tmp_path / "b", WITH_REPLACEMENT, 5)
    assert [file_digest(p) for p in first] == [file_digest(p) for p in second]


def test_a_rule_set_too_small_to_subset_is_an_error(tmp_path):
    path = tmp_path / "tiny.jsonl"
    path.write_text(json.dumps({"id": "Rule 1", "rule": "only"}) + "\n")
    with pytest.raises(ValueError, match="at least 2 rules"):
        generate_subset_files(path, tmp_path / "out", WITH_REPLACEMENT, 0)


@pytest.mark.frozen
def test_the_serialization_is_part_of_the_checksum_contract(ruleset, tmp_path):
    path = tmp_path / "unicode.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": f"Rule {n}", "rule": "café ünïcode"}) + "\n"
            for n in range(1, 4)
        ),
        encoding="utf-8",
    )
    written = generate_subset_files(path, tmp_path / "out", WITH_REPLACEMENT, 0)
    text = written[-1].read_text(encoding="utf-8")
    assert "café ünïcode" in text
    assert text.endswith("\n")
    assert len(text.splitlines()) == 3


def test_a_manifest_round_trips(ruleset, tmp_path):
    generate_subset_files(ruleset, tmp_path / "out" / "seed_0", WITH_REPLACEMENT, 0)
    entries = build_manifest(tmp_path / "out")
    assert parse_manifest(render_manifest(entries)) == entries


def test_a_malformed_manifest_line_is_rejected():
    with pytest.raises(ValueError, match="Malformed manifest line"):
        parse_manifest("deadbeef seed_0/subset_size_02.jsonl\n")


def test_manifest_parsing_skips_blank_lines():
    assert parse_manifest("\nabc  a.jsonl\n\n") == {"a.jsonl": "abc"}


def test_a_clean_regeneration_reports_no_differences():
    entries = {"a.jsonl": "aaa", "b.jsonl": "bbb"}
    assert diff_manifest(entries, dict(entries)) == {
        "missing": [],
        "unexpected": [],
        "changed": [],
    }


def test_the_three_kinds_of_difference_are_reported_separately():
    expected = {"a.jsonl": "aaa", "b.jsonl": "bbb"}
    actual = {"a.jsonl": "aaa", "b.jsonl": "CHANGED", "c.jsonl": "ccc"}
    assert diff_manifest(expected, actual) == {
        "missing": [],
        "unexpected": ["c.jsonl"],
        "changed": ["b.jsonl"],
    }


def test_a_missing_file_is_reported():
    assert diff_manifest({"a.jsonl": "aaa"}, {})["missing"] == ["a.jsonl"]


@pytest.fixture(scope="module")
def committed_manifest(repo_root):
    meta = json.loads((repo_root / MANIFEST_DIR / "manifest.json").read_text())
    entries = parse_manifest(
        (repo_root / MANIFEST_DIR / meta["files_manifest"]).read_text()
    )
    return meta, entries


def test_the_committed_manifest_covers_every_seed_and_size(committed_manifest):
    meta, entries = committed_manifest
    assert len(entries) == meta["files"]["count"] == 1400
    assert len(meta["seeds"]) == 25
    assert {path.split("/")[0] for path in entries} == {
        f"seed_{seed}" for seed in range(25)
    }


def test_the_source_ruleset_is_committed_and_unchanged(repo_root, committed_manifest):
    meta, _ = committed_manifest
    ruleset = repo_root / meta["source_ruleset"]["path"]
    assert ruleset.exists()
    assert file_digest(ruleset) == meta["source_ruleset"]["sha256"]
    assert meta["source_ruleset"]["n_rules"] == 57


@pytest.mark.frozen
def test_regenerating_one_seed_reproduces_its_committed_digests(
    repo_root, committed_manifest, tmp_path
):
    meta, entries = committed_manifest
    generate_subset_files(
        repo_root / meta["source_ruleset"]["path"],
        tmp_path / "seed_0",
        meta["sampling_strategy"],
        0,
    )
    regenerated = build_manifest(tmp_path)
    expected = {
        path: digest for path, digest in entries.items() if path.startswith("seed_0/")
    }
    assert len(expected) == 56
    assert diff_manifest(expected, regenerated) == {
        "missing": [],
        "unexpected": [],
        "changed": [],
    }


def test_the_shuffled_full_manifest_pins_the_published_subsets_below_the_full_size(repo_root, committed_manifest):
    meta, published = committed_manifest
    shuffled_meta, shuffled = load_manifest(repo_root / SHUFFLED_FULL_MANIFEST_DIR)
    assert shuffled_meta["sampling_strategy"] == WITH_REPLACEMENT_SHUFFLED_FULL
    assert (shuffled_meta["seeds"], shuffled_meta["source_ruleset"]) == (meta["seeds"], meta["source_ruleset"])
    full = f"subset_size_{meta['source_ruleset']['n_rules']}.jsonl"
    assert set(shuffled) == set(published)
    assert ({path: digest for path, digest in shuffled.items() if not path.endswith(full)}
            == {path: digest for path, digest in published.items() if not path.endswith(full)})
    full_digests = {digest for path, digest in shuffled.items() if path.endswith(full)}
    assert len(full_digests) == len(meta["seeds"])
    assert not full_digests & {digest for path, digest in published.items() if path.endswith(full)}


@pytest.mark.frozen
def test_regenerating_one_seed_reproduces_its_shuffled_full_digests(repo_root, tmp_path):
    meta, entries = load_manifest(repo_root / SHUFFLED_FULL_MANIFEST_DIR)
    source = repo_root / meta["source_ruleset"]["path"]
    written = generate_subset_files(source, tmp_path / "seed_3", meta["sampling_strategy"], 3)
    expected = {path: digest for path, digest in entries.items() if path.startswith("seed_3/")}
    assert len(expected) == 56
    assert build_manifest(tmp_path) == expected
    drawn, rules = ([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    for path in (written[-1], source))
    assert drawn != rules and sorted(drawn, key=json.dumps) == sorted(rules, key=json.dumps)


def test_the_seed_layout_matches_the_committed_manifest_keys(repo_root):
    root = Path("/data")
    assert seed_dir(root, WITH_REPLACEMENT, 7) == root / WITH_REPLACEMENT / "seed_7"
    assert seed_dir(root, WITH_REPLACEMENT, 7).parent == strategy_root(
        root, WITH_REPLACEMENT
    )


def test_load_manifest_follows_the_files_manifest_pointer(repo_root):
    meta, entries = load_manifest(repo_root / MANIFEST_DIR)
    assert meta["files_manifest"] == "manifest.sha256"
    assert len(entries) == meta["files"]["count"]


def test_sampling_leaves_the_callers_rng_untouched():
    random.seed(1234)
    expected = [random.random() for _ in range(3)]

    random.seed(1234)
    create_subsets(WITH_REPLACEMENT, _rules(20), 0)
    assert [random.random() for _ in range(3)] == expected

    random.seed(1234)
    create_subsets(WITHOUT_REPLACEMENT, _rules(20), 0)
    assert [random.random() for _ in range(3)] == expected

    random.seed(1234)
    create_subsets(WITH_REPLACEMENT_SHUFFLED_FULL, _rules(20), 0)
    assert [random.random() for _ in range(3)] == expected
