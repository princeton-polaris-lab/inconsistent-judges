from __future__ import annotations

from pathlib import Path

import pytest

from ruleforge.launch.cli import main
from tests.test_launch_campaign import repo  # noqa: F401 - through checkout


@pytest.fixture
def checkout(request):
    """test_launch_campaign's clean checkout holding the repository's configs, its artifact root beside it."""
    return request.getfixturevalue("repo")


def rebased_root(tmp_path: Path, campaign_id: str) -> tuple[str, Path]:
    """``sampling/<parent>-pop-8f74b719`` holding a campaign (its sampling manifest), as a rebase leaves it."""
    relative = f"sampling/{campaign_id}-pop-8f74b719"
    root = tmp_path / "artifacts" / relative
    root.mkdir(parents=True)
    (root / "sampling_manifest.json").write_text("{}\n")
    return relative, root


def test_no_shards_is_one_invocation_s_choice_and_excludes_a_shard_count(checkout, tmp_path, capsys):
    relative, _ = rebased_root(tmp_path, "c1")
    base = ["campaign", "--backend", "slurm", "--gpu-count", "1", "--campaign-id", "c1", "--campaign-root", relative,
            "--dry-run"]
    for extra in (["--shards", "3"], ["--shards", "3", "--only", "1"]):
        assert main([*base, "--no-shards", *extra], repo=checkout) == 2
        assert "--no-shards runs each condition as one job: not with --shards or --only" in capsys.readouterr().err
    spec = tmp_path / "unsharded.yaml"
    spec.write_text("backend: slurm\ngpu-count: 1\nno-shards: true\n")
    assert main(["campaign", "--spec", str(spec), "--campaign-id", "c1", "--dry-run"], repo=checkout) == 2
    assert "no-shards" in capsys.readouterr().err

