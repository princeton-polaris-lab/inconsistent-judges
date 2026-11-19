from __future__ import annotations

import pytest

from ruleforge.evaluate.chain import CHAIN_KEY, RebaseError, rebase_chain, rebase_hop

RECORD = {"parent": {"root": "parent", "campaign_id": "c"},
          "target": {"scenario_manifest_sha256": "a" * 64, "source_sha256": "b" * 64, "count": 500},
          "removed_ids": ["x"], "added_ids": ["y"]}


def test_a_record_before_the_chain_reads_as_the_one_hop_it_describes():
    hop = rebase_hop(RECORD)
    assert hop == {"parent": RECORD["parent"], "target": {"scenario_manifest_sha256": "a" * 64, "source_sha256": "b" * 64},
                   "removed_ids": ["x"], "added_ids": ["y"]}
    assert rebase_chain(RECORD) == [hop]
    assert rebase_chain({**RECORD, CHAIN_KEY: [hop, hop]}) == [hop, hop]


@pytest.mark.parametrize("chain", ["hops", [{"parent": {}}], [1], {"parent": {}}])
def test_a_chain_that_is_not_a_list_of_hops_is_refused(chain):
    with pytest.raises(RebaseError, match="is not a list of hops"):
        rebase_chain({**RECORD, CHAIN_KEY: chain})
