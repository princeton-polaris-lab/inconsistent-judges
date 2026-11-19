from __future__ import annotations

import json
from pathlib import Path

import pytest

from ruleforge.experiments.ruleforge import build, plot

pytest.importorskip("matplotlib")

GOLDEN = Path("results/ruleforge/ruleforge.json")


def _close(figure):
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_a_redraw_takes_only_a_known_golden(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"data_version": 1}))
    with pytest.raises(SystemExit):
        plot.main(["--from_golden", str(golden)])
    assert "data_version 1" in capsys.readouterr().err


def test_the_committed_golden_draws_as_two_panels_and_no_two_panel_figure(repo_root, tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    plot.main(["--from_golden", str(repo_root / GOLDEN), "--figure_a", str(out / "a.pdf"),
               "--figure_b", str(out / "b.pdf")])
    assert sorted(path.name for path in out.iterdir()) == ["a.pdf", "b.pdf"]
    assert f"wrote {out / 'b.pdf'}" in capsys.readouterr().out
    stats = json.loads((repo_root / GOLDEN).read_text())
    for panel in "ab":
        fig = plot.build_figure(stats, panel)
        assert len(fig.axes) == 1 and fig.axes[0].get_title(loc="left") == "" and fig._suptitle is None
        _close(fig)
    with pytest.raises(SystemExit):
        plot.main(["--from_golden", str(repo_root / GOLDEN), "--figure", str(out / "c.pdf")])
    assert "ambiguous option: --figure " in capsys.readouterr().err and not (out / "c.pdf").exists()


def test_the_renderer_refuses_an_unknown_panel_or_a_single_ruleset(repo_root):
    stats = json.loads((repo_root / GOLDEN).read_text())
    with pytest.raises(ValueError, match="unknown RuleForge panel"):
        plot.build_figure(stats, panel="c")
    with pytest.raises(ValueError, match="at least one revision"):
        plot.build_figure({**stats, "labels": stats["labels"][:1]}, "a")


@pytest.mark.parametrize("ruleset", ["no-equals", "=label", "tag=", "a=A --ruleset a=B"])
def test_the_campaign_producer_refuses_a_malformed_or_repeated_ruleset(ruleset, tmp_path, capsys):
    arguments = ["--estimates", str(tmp_path / "estimates.json"), "--golden", str(tmp_path / "golden.json")]
    for item in ruleset.split(" --ruleset "):
        arguments += ["--ruleset", item]
    with pytest.raises(SystemExit) as info:
        build.main(arguments)
    assert info.value.code == 2 and "--ruleset requires unique TAG=LABEL entries" in capsys.readouterr().err
