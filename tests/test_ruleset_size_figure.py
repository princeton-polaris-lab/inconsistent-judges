from __future__ import annotations

import importlib
import json

import pytest

from ruleforge.experiments.ruleset_size import figure as renderer

PRODUCER = "ruleforge.experiments.ruleset_size.build"
GOLDEN = "results/ruleset_size/curves.json"
PUBLISHED = ("GPT-OSS-120B", "Gemma-4-31B", "Olmo-3-32B", "GPT-5.6 Luna")
HAIKU = "Claude Haiku 5.5"
GLIMMER = "Muse-Glimmer-30B"


@pytest.fixture(scope="module")
def driver():
    pytest.importorskip("matplotlib")
    return importlib.import_module(PRODUCER)


@pytest.fixture(scope="module")
def curves(driver, repo_root):
    return driver.validate_curves(json.loads((repo_root / GOLDEN).read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def labels(curves):
    """The committed golden's curves, in the renderer's order (``SERIES_COLORS``)."""
    found = [curve["label"] for curve in curves]
    assert set(PUBLISHED) <= set(found) <= set(renderer.SERIES_COLORS), found
    return [label for label in renderer.SERIES_COLORS if label in found]


@pytest.fixture
def captured_figure(monkeypatch):
    """Draw without saving and hand back the matplotlib figure."""
    captured = {}
    monkeypatch.setattr(
        renderer.theme,
        "save_soft_figure",
        lambda figure, _path, dpi=None: captured.setdefault("figure", figure),
    )
    return captured


def test_the_figure_renders_from_the_committed_golden(curves, tmp_path):
    out = tmp_path / "ruleset_size.pdf"
    renderer.draw(curves, out)
    assert out.exists()
    # Non-trivial: an empty axes still produces a valid but tiny PDF.
    assert out.stat().st_size > 5_000


def test_the_pdf_keeps_text_as_text(curves, tmp_path):
    """Type 42, not matplotlib's default Type 3."""
    out = tmp_path / "ruleset_size.pdf"
    renderer.draw(curves, out)
    pdf = out.read_bytes()
    assert b"/Type3" not in pdf
    assert b"FontFile2" in pdf


def test_the_lines_on_the_axes_are_the_curves_means_in_order(curves, labels, captured_figure, tmp_path):
    renderer.draw(curves, tmp_path / "unused.png")
    lines = captured_figure["figure"].axes[0].lines
    assert [curve["label"] for curve in curves] == labels
    for line, curve in zip(lines, curves, strict=True):
        assert list(line.get_xdata()) == [point["size"] for point in curve["points"]]
        assert list(line.get_ydata()) == [point["mean"] for point in curve["points"]]


def test_the_committed_golden_holds_every_size(labels, repo_root):
    full = json.loads((repo_root / GOLDEN).read_text(encoding="utf-8"))
    assert [curve["label"] for curve in full] == labels
    assert all([p["size"] for p in curve["points"]] == list(range(2, 58)) for curve in full)


def test_the_series_keep_the_reference_palette():
    theme = renderer.theme
    assert renderer.SERIES_COLORS == {
        "GPT-OSS-120B": theme.PASTEL_BLUE,
        "Gemma-4-31B": theme.PASTEL_CORAL,
        "Olmo-3-32B": theme.PASTEL_GREEN,
        GLIMMER: theme.PASTEL_ROSE,
        "GPT-5.6 Luna": theme.PASTEL_PURPLE,
        "Claude Haiku 5.5": theme.PASTEL_CYAN,
    }
    assert len(set(renderer.SERIES_COLORS.values())) == len(renderer.SERIES_COLORS)
    assert theme.PASTEL_ROSE == "#D68BA2"


@pytest.mark.parametrize(("label", "after", "colour"), [(HAIKU, "GPT-5.6 Luna", "#86DAE0"),
                                                       (GLIMMER, "Olmo-3-32B", "#D68BA2")])
def test_an_added_judge_s_curve_takes_its_place_and_colour_and_the_published_keep_theirs(
        curves, captured_figure, tmp_path, label, after, colour):
    published = [curve for curve in curves if curve["label"] in PUBLISHED]
    added = next((curve for curve in curves if curve["label"] == label), {**published[0], "label": label})
    renderer.draw([added, *reversed(published)], tmp_path / "unused.png")
    lines = captured_figure["figure"].axes[0].lines
    order = [curve["label"] for curve in published]
    assert order == list(PUBLISHED)
    expected = [*order[:order.index(after) + 1], label, *order[order.index(after) + 1:]]
    assert [line.get_label() for line in lines] == expected
    assert [line.get_color() for line in lines] == [renderer.SERIES_COLORS[name] for name in expected]
    assert renderer.SERIES_COLORS[label] == colour


def test_every_campaign_judge_but_those_no_ruleset_size_root_holds_has_a_colour(driver):
    """GPT-6.1 Sol and Qwen3-8B have no ruleset-size root; a curve of either would take a colour no judge drawn has."""
    from ruleforge.experiments.labels import CAMPAIGN_MODEL_LABELS

    named = {driver.judge_label(name) for name in CAMPAIGN_MODEL_LABELS}
    assert named - set(renderer.SERIES_COLORS) == {"GPT-6.1 Sol", "Qwen3-8B"} and set(renderer.SERIES_COLORS) <= named


def test_shuffled_curves_keep_canonical_legend_order_and_colors(curves, labels, captured_figure, tmp_path):
    renderer.draw(list(reversed(curves)), tmp_path / "unused.png")
    lines = captured_figure["figure"].axes[0].lines
    assert [line.get_label() for line in lines] == labels
    assert [line.get_color() for line in lines] == [renderer.SERIES_COLORS[label] for label in labels]


def test_the_legend_puts_labels_without_a_colour_last_in_label_order():
    items = ["Zeta judge", "Gemma-4-31B", "Alpha judge", "GPT-OSS-120B"]
    assert renderer.legend_order(items, str) == ["GPT-OSS-120B", "Gemma-4-31B", "Alpha judge", "Zeta judge"]


def test_a_judge_keeps_its_colour_and_ones_without_take_in_turn_the_colours_no_drawn_judge_has():
    colors = renderer.SERIES_COLORS
    named = [label for label in colors if label != "Claude Haiku 5.5"]
    assert renderer.series_colors(["Olmo-3-32B"]) == [colors["Olmo-3-32B"]]
    # Every named judge but Claude Haiku 5.5 and one of the adopter's: it takes the one colour left, not GPT-OSS-120B's.
    assert renderer.series_colors([*named, "Some judge"]) == [colors[label] for label in named] + [
        colors["Claude Haiku 5.5"]]
    assert len(set(renderer.series_colors(["Some judge", "GPT-OSS-120B", "Other judge"]))) == 3
    assert renderer.series_colors(["Some judge", "GPT-OSS-120B", "Other judge"])[0::2] == list(colors.values())[1:3]
    assert renderer.series_colors([*colors, "Some judge"])[-1] == colors["GPT-OSS-120B"]


def test_the_paper_s_figure_draws_every_curve_in_its_own_colour(curves, labels, captured_figure, tmp_path):
    renderer.draw(curves, tmp_path / "unused.png")
    figure = captured_figure["figure"]
    lines = figure.axes[0].lines
    assert [line.get_label() for line in lines] == labels
    assert [line.get_color() for line in lines] == [renderer.SERIES_COLORS[label] for label in labels]
    legend = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert legend == labels


def test_the_legend_sits_in_two_columns_at_the_lower_right_below_every_curve(curves, captured_figure, tmp_path):
    import matplotlib
    import numpy as np

    drawn = curves
    renderer.draw(drawn, tmp_path / "unused.png")
    figure = captured_figure["figure"]
    axes = figure.axes[0]
    legend = axes.get_legend()
    with matplotlib.rc_context(renderer.theme.SOFT_STYLE):
        figure.canvas.draw()
    renderer_ = figure.canvas.get_renderer()
    frame, plot = legend.get_frame().get_window_extent(renderer_), axes.get_window_extent(renderer_)
    assert plot.x0 <= frame.x0 and frame.x1 <= plot.x1 and plot.y0 <= frame.y0 and frame.y1 <= plot.y1
    assert (frame.x0 + frame.x1) / 2 > (plot.x0 + plot.x1) / 2 and (frame.y0 + frame.y1) / 2 < (plot.y0 + plot.y1) / 2
    keys = [(text.get_window_extent(renderer_), text.get_text()) for text in legend.get_texts()]
    columns = sorted({round(box.x0) for box, _ in keys})
    assert len(columns) == 2
    assert [[text for box, text in keys if round(box.x0) == column] for column in columns] == [
        [line.get_label() for line in axes.lines][:3], [line.get_label() for line in axes.lines][3:]]
    radius = max(line.get_markersize() / 2 + line.get_markeredgewidth() / 2 for line in axes.lines) * figure.dpi / 72
    to_data, to_display = axes.transData.inverted().transform, axes.transData.transform
    sizes = np.linspace(to_data((frame.x0 - radius, 0))[0], to_data((frame.x1 + radius, 0))[0], 400)
    lowest = min(
        min(to_display((size, low))[1], to_display((size, mean))[1] - radius)
        for curve in drawn
        for size, low, mean in zip(sizes, *(np.interp(sizes, [point["size"] for point in curve["points"]],
                                                       [point[key] for point in curve["points"]])
                                            for key in ("ci_low", "mean"))))
    assert lowest > frame.y1


def test_a_curve_with_a_single_model_still_renders(curves, tmp_path):
    out = tmp_path / "one.pdf"
    renderer.draw(curves[:1], out)
    assert out.stat().st_size > 5_000


def test_the_axes_say_what_the_payload_holds(curves, labels, captured_figure, tmp_path):
    renderer.draw(curves, tmp_path / "unused.png")
    figure = captured_figure["figure"]
    axes = figure.axes[0]
    assert axes.get_ylabel() == "Inconsistency rate"
    assert axes.get_xlabel() == "Subset Size"
    assert [text.get_text() for text in figure.texts] == ["Inconsistency Rate Across Ruleset Sizes"]
    assert axes.get_ylim() == (0.0, 1.0)
    assert [text.get_text() for text in axes.get_legend().get_texts()] == labels
    for line in axes.lines:
        ys = line.get_ydata()
        assert ys[0] < ys[-1], f"{line.get_label()} does not rise"


def test_a_graded_penalty_curve_is_refused_under_the_inconsistency_label(curves, tmp_path):
    graded = [dict(curve, metric="regularization", quantity="mean_graded_penalty") for curve in curves]
    with pytest.raises(ValueError, match="inconsistency_rate"):
        renderer.draw(graded, tmp_path / "unused.pdf")
    mixed = [curves[0], dict(curves[1], quantity="mean_graded_penalty")]
    with pytest.raises(ValueError, match="disagree on quantity"):
        renderer.draw(mixed, tmp_path / "unused.pdf")
