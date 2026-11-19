from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ruleforge.experiments import style as theme  # noqa: E402


@pytest.fixture
def figure_axes():
    figure, axes = plt.subplots()
    yield figure, axes
    plt.close(figure)


def test_soft_header_writes_title_and_optional_subtitle(figure_axes):
    figure, _ = figure_axes
    theme.soft_header(figure, "The Title", "the subtitle")
    texts = [t.get_text() for t in figure.texts]
    assert texts == ["The Title", "the subtitle"]

    assert [t.get_position()[0] for t in figure.texts] == [0.5, 0.5]
    assert figure.texts[0].get_fontsize() == 13

    other = plt.figure()
    try:
        theme.soft_header(other, "Alone", None, x=0.62, fontsize=17)
        assert [t.get_text() for t in other.texts] == ["Alone"]
        assert other.texts[0].get_position()[0] == 0.62 and other.texts[0].get_fontsize() == 17
    finally:
        plt.close(other)


def test_style_soft_axes_hides_top_and_right_and_grids_behind(figure_axes):
    _, axes = figure_axes
    theme.style_soft_axes(axes)
    assert not axes.spines["top"].get_visible()
    assert not axes.spines["right"].get_visible()
    assert axes.spines["left"].get_edgecolor()[:3] != (0.0, 0.0, 0.0)
    assert axes.get_axisbelow()


def test_soft_legend_applies_defaults_and_accepts_overrides(figure_axes):
    _, axes = figure_axes
    axes.plot([0, 1], [0, 1], label="series")
    legend = theme.soft_legend(axes, ncol=1)
    assert legend.get_frame().get_linewidth() == pytest.approx(0.9)
    assert legend.handlelength == pytest.approx(1.6)
    wide = theme.soft_legend(axes, handlelength=3.0)
    assert wide.handlelength == pytest.approx(3.0)


def test_soft_style_keeps_pdf_text_as_text():
    assert theme.SOFT_STYLE["pdf.fonttype"] == 42
    assert theme.SOFT_STYLE["ps.fonttype"] == 42


def test_save_soft_figure_writes_type42_text_even_outside_the_context(
    figure_axes, tmp_path
):
    figure, axes = figure_axes
    axes.set_title("text that must stay text")
    out = tmp_path / "nested" / "figure.pdf"
    theme.save_soft_figure(figure, out)
    pdf = out.read_bytes()
    assert b"/Type3" not in pdf
    assert b"FontFile2" in pdf
