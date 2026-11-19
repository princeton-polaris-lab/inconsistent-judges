from __future__ import annotations

from typing import Iterable

from ruleforge.evaluate import api
from ruleforge.experiments import table
from ruleforge.hub import load_inventory
from tests.identity import published as pub

SAMPLING = "artifact://sampling/"


def sampling_roots(words: Iterable[str]) -> set[str]:
    """The campaign roots ``artifact://sampling/<root>[/<path>]`` words name."""
    return {word[len(SAMPLING):].split("/", 1)[0] for word in words if word.startswith(SAMPLING)}


def readers() -> dict[str, set[str]]:
    """Every campaign root an experiment or one of its builds reads, with what reads it."""
    found: dict[str, set[str]] = {}
    for row in table.TABLE:
        for root in sampling_roots(row.runs):
            found.setdefault(root, set()).add(row.name)
        for build in row.builds:
            for root in sampling_roots([*build.words, *build.inputs]):
                found.setdefault(root, set()).add(f"build {build.id}")
    return found


def test_the_published_roots_are_the_ones_the_inventory_ships():
    inventory = load_inventory(pub.REPO_ROOT / "data" / "hub_inventory.yaml")
    shipped = [root for experiment in inventory["experiments"].values() for root in experiment.get("roots", {})]
    assert sorted(shipped) == list(pub.ROOTS)


def test_every_shipped_root_is_pinned():
    assert set(api.pinned_roots()) == {f"{SAMPLING}{root}" for root in pub.ROOTS}


def test_every_root_a_golden_or_a_build_reads_is_shipped():
    found = readers()
    assert found and all(found.values())
    assert {root: sorted(by) for root, by in found.items() if root not in pub.ROOTS} == {}
    # No golden or build reads a re-score's source: each reads the copy the dataset ships in its place.
    assert not set(found) & set(pub.RESCORED_FROM.values())


def test_sampling_roots_reads_the_root_of_each_sampling_word():
    assert sampling_roots(["--estimates", f"{SAMPLING}r1/aggregate/e.json", f"{SAMPLING}r2", "repo://x",
                           "artifact://selection/r3/m.json"]) == {"r1", "r2"}

