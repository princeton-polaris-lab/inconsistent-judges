from __future__ import annotations

import contextlib
import importlib
import json

import pytest

from ruleforge.experiments.prompt_ablation.analysis import (
    FORCED_B,
)
from ruleforge.experiments import style
from ruleforge.experiments.labels import MODEL_DISPLAY_LABELS, NAIVE_SLOT, display_order
from ruleforge.prompts.styles import ENCOURAGED, FORCED

PRODUCER = "ruleforge.experiments.prompt_ablation.build"
GOLDEN = "results/prompt_ablation/prompt_ablation.json"
#: The campaign golden's bar set, in drawn order.
BARS = (NAIVE_SLOT, ENCOURAGED, FORCED, FORCED_B)


@pytest.fixture(scope="module")
def figure_module():
    pytest.importorskip("matplotlib")
    from ruleforge.experiments.prompt_ablation import figure as prompt_ablation_figure

    return prompt_ablation_figure


@pytest.fixture(scope="module")
def driver():
    return importlib.import_module(PRODUCER)


@pytest.fixture(scope="module")
def payload(repo_root):
    return json.loads((repo_root / GOLDEN).read_text(encoding="utf-8"))


def _bar(mean, n):
    return {"mean": mean, "ci_low": max(0.0, mean - 0.1), "ci_high": mean + 0.1, "n": n}


def _synthetic_bars():
    """Two models, four bars, one population, a distinct n per bar."""
    return {
        label: {
            "per_bar": {bar: _bar(0.1 * (index + 1), 40 + index) for index, bar in enumerate(BARS)}
        }
        for label in ("modelA", "modelB")
    }


@contextlib.contextmanager
def _panel(figure_module, bar_block, part):
    import matplotlib.pyplot as plt

    figure = figure_module.build_figure(bar_block, "per_bar", part)
    try:
        yield figure
    finally:
        plt.close(figure)


def _texts(figure):
    return [text.get_text() for ax in figure.axes for text in ax.texts]


def _n_labels(figure):
    return [label for label in _texts(figure) if label.startswith("n=")]


def test_both_panels_render_from_the_committed_golden(figure_module, payload, tmp_path):
    for part in figure_module.FIGURE_PARTS:
        out = tmp_path / f"bars_{part}.pdf"
        figure_module.draw(payload["bars"], out, "per_bar", part)
        assert out.exists()
        assert out.stat().st_size > 5_000


def test_both_panels_render_on_synthetic_data(figure_module, tmp_path):
    for part in figure_module.FIGURE_PARTS:
        out = tmp_path / f"bars_{part}.pdf"
        figure_module.draw(_synthetic_bars(), out, "per_bar", part)
        assert out.exists()
        assert out.stat().st_size > 5_000


def test_each_panel_carries_its_legend_in_its_text_sizes(figure_module):
    sizes = (figure_module.TICK_FONTSIZE, figure_module.Y_LABEL_FONTSIZE, figure_module.LEGEND_FONTSIZE,
             figure_module.N_LABEL_FONTSIZE)
    assert sizes == (15, 16, 15, 12)
    legends = {"styles": ["naive", "encouraged", "forced"], "methods": ["Separate judge calls", "Same judge call"]}
    for part, n_labels in (("styles", 6), ("methods", 2)):
        with _panel(figure_module, _synthetic_bars(), part) as figure:
            (axes,) = figure.axes
            assert [text.get_text() for text in axes.get_legend().get_texts()] == legends[part]
            assert axes.get_title(loc="left") == ""
            assert axes.yaxis.label.get_fontsize() == 16
            assert all(tick.get_fontsize() == 15 for tick in axes.get_xticklabels())
            assert all(tick.get_fontsize() == 15 for tick in axes.get_yticklabels())
            assert all(text.get_fontsize() == 15 for text in axes.get_legend().get_texts())
            assert all(text.get_fontsize() == 12 for text in axes.texts if text.get_text().startswith("n="))
            assert len(_n_labels(figure)) == n_labels


def test_the_producer_draws_the_two_panels_and_no_two_panel_figure(driver, repo_root, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "out"
    out.mkdir()
    golden = str(repo_root / GOLDEN)
    assert driver.main(["--from_golden", golden, "--figure_styles", str(out / "a.png"),
                        "--figure_methods", str(out / "b.png"), "--dpi", "72"]) == 0
    assert sorted(path.name for path in out.iterdir()) == ["a.png", "b.png"]
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", golden, "--figure_combined", str(out / "c.png")])
    assert "--figure_combined" in capsys.readouterr().err and not (out / "c.png").exists()


def test_the_panel_contents_are_the_authors(figure_module, driver):
    assert figure_module.FIGURE_PARTS["styles"] == (NAIVE_SLOT, ENCOURAGED, FORCED)
    assert figure_module.FIGURE_PARTS["methods"] == (FORCED, FORCED_B)
    drawn = {bar for bars in figure_module.FIGURE_PARTS.values() for bar in bars}
    assert drawn == set(BARS) == set(driver.DRAWN_BARS)


def test_a_bar_short_of_the_full_population_keeps_its_own_n_label(figure_module):
    with _panel(figure_module, _synthetic_bars(), "styles") as figure:
        n_labels = _n_labels(figure)
        assert len(n_labels) == 6
        expected = {f"n={40 + BARS.index(bar)}" for bar in figure_module.FIGURE_PARTS["styles"]}
        assert set(n_labels) == expected
    with _panel(figure_module, _synthetic_bars(), "methods") as figure:
        assert _n_labels(figure) == ["n=42", "n=42"]


def test_no_panel_carries_a_prose_note_or_an_n_label_on_a_full_population(figure_module):
    uniform = {
        label: {"per_bar": {bar: _bar(0.1 * (index + 1), 500) for index, bar in enumerate(BARS)}}
        for label in ("modelA", "modelB")
    }
    for part in figure_module.FIGURE_PARTS:
        with _panel(figure_module, uniform, part) as figure:
            assert figure.axes[0].get_title(loc="right") == ""
            assert _n_labels(figure) == []


def test_both_panels_name_the_inconsistency_rate_and_never_smri(figure_module):
    headings = {
        "styles": "Inconsistency Rate by Prompt Style",
        "methods": "Inconsistency Rate with Separate Individual Rule Judgments",
    }
    for part, header in headings.items():
        with _panel(figure_module, _synthetic_bars(), part) as figure:
            assert figure.axes[0].get_ylabel() == "Inconsistency rate"
            texts = [text.get_text() for text in figure.texts] + _texts(figure)
            assert header not in texts
            assert figure.axes[0].get_title() == ""
            assert figure._suptitle is None
            for text in texts + [figure.axes[0].get_ylabel()]:
                assert "SMRI" not in text and "ontradiction" not in text, text


def test_whiskers_span_the_interval_not_the_mean(figure_module):
    for part, bars in figure_module.FIGURE_PARTS.items():
        with _panel(figure_module, _synthetic_bars(), part) as figure:
            axes = figure.axes[0]
            # One LineCollection per bar group's errorbars.
            assert len(axes.collections) == len(bars)
            for collection in axes.collections:
                segments = collection.get_segments()
                assert segments, "errorbar collection is empty"
                for (x0, y0), (x1, y1) in segments:
                    assert y1 - y0 == pytest.approx(0.2, abs=1e-6)


def test_the_golden_figure_carries_no_n_label(figure_module, payload):
    """Every bar of both panels is the 500 x 5 grid."""
    assert {
        entry["n"]
        for bars in payload["bars"].values()
        for entry in bars["per_bar"].values()
    } == {500}
    for part in figure_module.FIGURE_PARTS:
        with _panel(figure_module, payload["bars"], part) as figure:
            assert _n_labels(figure) == [], part
            assert not any(label.endswith("*") for label in _texts(figure)), part


PUBLISHED_FOUR = ("Olmo3-32B", "gemma4-31b", "gpt-oss-120b", "gpt-5.6-luna")


def _published_four(payload):
    return {label: payload["bars"][label] for label in PUBLISHED_FOUR}


def test_the_paper_model_order_and_display_names_are_fixed(figure_module, payload):
    with _panel(figure_module, _published_four(payload), "styles") as figure:
        labels = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]
        assert labels == [
            "Olmo-3-32B",
            "Gemma-4-31B",
            "GPT-OSS-120B",
            "GPT-5.6 Luna",
        ]


def test_haiku_is_the_fifth_judge_after_luna_and_a_fifth_group_widens_the_panel(figure_module, payload):
    assert MODEL_DISPLAY_LABELS["claude-haiku-5-5"] == "Claude Haiku 5.5"
    assert display_order(["gpt-6.1-sol", "claude-haiku-5-5", "gpt-5.6-luna", "Olmo3-32B"]) == [
        "Olmo3-32B", "gpt-5.6-luna", "claude-haiku-5-5", "gpt-6.1-sol"]
    bar_block = _published_four(payload)
    with_haiku = {**bar_block, "claude-haiku-5-5": payload["bars"].get("claude-haiku-5-5", bar_block["gpt-5.6-luna"])}
    for part, width in (("styles", 6.8), ("methods", 5.8)):
        with _panel(figure_module, bar_block, part) as figure:
            assert figure.get_size_inches()[0] >= width - 1e-9
            assert figure.get_size_inches()[1] == 3.4
        with _panel(figure_module, with_haiku, part) as figure:
            assert figure.get_size_inches()[0] >= width * 5 / 4 - 1e-9
            assert figure.get_size_inches()[1] == 3.4
            labels = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]
            assert labels == ["Olmo-3-32B", "Gemma-4-31B", "GPT-OSS-120B", "GPT-5.6 Luna", "Claude Haiku 5.5"]


FIVE_JUDGES = "tests/fixtures/prompt_ablation/five_judges.json"
SIX_JUDGES = "tests/fixtures/prompt_ablation/six_judges.json"


def _proportional_sizes(groups):
    """Each panel's proportional size: its published size (four groups), ``groups`` groups wide."""
    return {"styles": (6.8 * groups / 4, 3.4), "methods": (5.8 * groups / 4, 3.4)}


def _five_judges(repo_root):
    return json.loads((repo_root / FIVE_JUDGES).read_text(encoding="utf-8"))["bars"]


def _six_judges(repo_root):
    return json.loads((repo_root / SIX_JUDGES).read_text(encoding="utf-8"))["bars"]


def _figures(figure_module, bar_block):
    return {part: figure_module.build_figure(bar_block, "per_bar", part) for part in figure_module.FIGURE_PARTS}


def _tick_labels(figure_module, figure, groups=5):
    """Per axes, the gaps between neighbouring x tick labels, whether any overlap, and the figure's size."""
    import matplotlib
    import matplotlib.pyplot as plt

    try:
        with matplotlib.rc_context(figure_module.theme.SOFT_STYLE):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            boxes = [[label.get_window_extent(renderer) for label in axes.get_xticklabels()] for axes in figure.axes]
        assert all(len(panel) == groups for panel in boxes)
        overlap = any(left.overlaps(right) for panel in boxes for i, left in enumerate(panel) for right in panel[i + 1:])
        panels = [[(right.x0 - left.x1) * 72 / figure.dpi for left, right in zip(panel, panel[1:])] for panel in boxes]
        return panels, overlap, tuple(figure.get_size_inches())
    finally:
        plt.close(figure)


def test_no_two_tick_labels_meet_in_any_panel_of_a_five_judge_figure(figure_module, repo_root, monkeypatch):
    bar_block = _five_judges(repo_root)
    assert display_order(bar_block)[-1] == "claude-haiku-5-5" and len(bar_block) == 5
    _hold_tick_labels_apart(figure_module, bar_block, monkeypatch)


def test_no_two_tick_labels_meet_in_any_panel_of_the_published_four(figure_module, payload, monkeypatch):
    _hold_tick_labels_apart(figure_module, _published_four(payload), monkeypatch)


def test_no_two_tick_labels_meet_in_any_panel_of_a_six_judge_figure(figure_module, repo_root, payload, monkeypatch):
    bar_block = _six_judges(repo_root)
    assert bar_block == payload["bars"]
    assert display_order(bar_block) == [
        "Olmo3-32B", "gemma4-31b", "gpt-oss-120b", "muse-glimmer-30b", "gpt-5.6-luna", "claude-haiku-5-5"]
    assert MODEL_DISPLAY_LABELS["muse-glimmer-30b"] == "Muse-Glimmer-30B"
    _hold_tick_labels_apart(figure_module, bar_block, monkeypatch)


def _hold_tick_labels_apart(figure_module, bar_block, monkeypatch):
    groups, gap = len(bar_block), style.TICK_LABEL_GAP
    sizes = _proportional_sizes(groups)
    with monkeypatch.context() as patch:
        patch.setattr(figure_module, "_fit_tick_labels", lambda figure: None)
        unfitted = {name: _tick_labels(figure_module, figure, groups)
                    for name, figure in _figures(figure_module, bar_block).items()}
    for name, figure in _figures(figure_module, bar_block).items():
        panels, overlap, size = _tick_labels(figure_module, figure, groups)
        unfitted_panels, _, unfitted_size = unfitted[name]
        assert unfitted_size == pytest.approx(sizes[name]), name
        assert not overlap and min(min(gaps) for gaps in panels) >= gap - 0.05, (name, panels)
        assert size[1] == sizes[name][1] and size[0] >= sizes[name][0] - 1e-9, name
        if min(min(gaps) for gaps in unfitted_panels) < gap:
            assert min(min(gaps) for gaps in panels) == pytest.approx(gap, abs=0.05), (name, panels)
        else:
            assert size == unfitted_size, name


def test_a_figure_whose_labels_need_more_room_widens_by_the_least_factor(figure_module, payload, repo_root,
                                                                          monkeypatch):
    monkeypatch.setattr(style, "TICK_LABEL_GAP", 60.0)
    for bar_block in (_published_four(payload), _five_judges(repo_root)):
        groups = len(bar_block)
        sizes = _proportional_sizes(groups)
        for name, figure in _figures(figure_module, bar_block).items():
            panels, overlap, size = _tick_labels(figure_module, figure, groups)
            assert size[0] > sizes[name][0] and size[1] == sizes[name][1], (groups, name)
            assert not overlap and min(min(gaps) for gaps in panels) == pytest.approx(60.0, abs=0.05), (name, panels)


def test_display_order_is_the_papers_then_alphabetical(figure_module):
    assert display_order(
        ["gpt-oss-120b", "zeta", "gpt-6.1-sol", "Olmo3-32B", "alpha", "gemma4-31b"]
    ) == [
        "Olmo3-32B",
        "gemma4-31b",
        "gpt-oss-120b",
        "gpt-6.1-sol",
        "alpha",
        "zeta",
    ]
    assert MODEL_DISPLAY_LABELS["gpt-6.1-sol"] == "GPT-6.1 Sol"
