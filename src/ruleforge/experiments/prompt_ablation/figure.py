from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from ruleforge.experiments import style as theme
from ruleforge.experiments.labels import MODEL_DISPLAY_LABELS, NAIVE_SLOT, display_order
from ruleforge.experiments.prompt_ablation.analysis import FORCED_B
from ruleforge.prompts.styles import ENCOURAGED, FORCED

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "BAR_STYLE",
    "FIGURE_PARTS",
    "PART_BAR_TITLES",
    "build_figure",
    "draw",
]

#: The two panels: inconsistency across the prompt styles, and the forced responses scored by same-call against
#: separate-call.
FIGURE_PARTS = {
    "styles": (NAIVE_SLOT, ENCOURAGED, FORCED),
    "methods": (FORCED, FORCED_B),
}

#: Legend labels per part: the styles panel names the prompts, the methods panel the measurement.
PART_BAR_TITLES = {
    "styles": {NAIVE_SLOT: "naive", ENCOURAGED: "encouraged", FORCED: "forced"},
    "methods": {
        FORCED: "Separate judge calls",
        FORCED_B: "Same judge call",
    },
}

#: Per-(part, bar) (fill, edge, hatch); matplotlib draws hatches in the edge colour.
BAR_STYLE = {
    "styles": dict(zip(FIGURE_PARTS["styles"], theme.BAR_STYLES)),
    "methods": {
        FORCED: (theme.PASTEL_BLUE, theme.DARK_BLUE, ""),
        FORCED_B: (theme.PASTEL_CORAL, theme.DARK_CORAL, ""),
    },
}

#: Each panel's text sizes, in points.
TICK_FONTSIZE = 15
Y_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 15
N_LABEL_FONTSIZE = 12


def _widening(bar_block: Dict[str, Dict[str, Any]]) -> float:
    return max(1.0, len(bar_block) / theme.PUBLISHED_GROUPS)


def _fit_tick_labels(figure: Figure) -> None:
    """Widen ``figure`` just enough that no two neighbouring x tick labels come closer than :data:`~ruleforge.experiments.style.TICK_LABEL_GAP`
    points."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    gap = theme.TICK_LABEL_GAP * figure.dpi / 72
    scale = 1.0
    for axes in figure.axes:
        boxes = [label.get_window_extent(renderer) for label in axes.get_xticklabels() if label.get_text()]
        for left, right in zip(boxes, boxes[1:]):
            centres = (right.x0 + right.x1 - left.x0 - left.x1) / 2
            scale = max(scale, ((left.width + right.width) / 2 + gap) / centres)
    if scale > 1.0:
        width, height = figure.get_size_inches()
        figure.set_size_inches(width * scale, height)


def _full_population(bar_block: Dict[str, Dict[str, Any]], population: str) -> int:
    return max(
        bar_block[label][population][bar]["n"]
        for label in bar_block
        for bars in FIGURE_PARTS.values()
        for bar in bars
    )


def _draw_panel(
    axes: Any,
    bar_block: Dict[str, Dict[str, Any]],
    population: str,
    part: str,
) -> None:
    """Draw one panel on existing axes, with no title or population note and no n label on a bar at the full population."""
    import numpy as np

    bars = FIGURE_PARTS[part]
    theme.style_soft_axes(axes)
    labels = display_order(bar_block)
    full_n = _full_population(bar_block, population)
    width = 0.24 if len(bars) == 3 else 0.3
    positions = np.arange(len(labels))
    for index, bar in enumerate(bars):
        fill, edge, hatch = BAR_STYLE[part][bar]
        offset = (index - (len(bars) - 1) / 2) * width
        stats = [bar_block[label][population][bar] for label in labels]
        means = [entry["mean"] for entry in stats]
        # max() guards float noise from turning a degenerate interval into a negative bar.
        err_low = [max(0.0, entry["mean"] - entry["ci_low"]) for entry in stats]
        err_high = [max(0.0, entry["ci_high"] - entry["mean"]) for entry in stats]
        axes.bar(
            positions + offset,
            means,
            width * 0.92,
            label=PART_BAR_TITLES[part][bar],
            color=fill,
            edgecolor=edge,
            hatch=hatch,
            linewidth=0.9,
            yerr=[err_low, err_high],
            capsize=2.5,
            error_kw={"linewidth": 0.9, "color": theme.TICK_COLOR},
        )
        for x, entry, high in zip(positions + offset, stats, err_high):
            if entry["n"] == full_n:
                continue
            axes.annotate(
                f"n={entry['n']}",
                xy=(x, max(entry["mean"] + high, entry["ci_high"])),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=N_LABEL_FONTSIZE,
                color=theme.TICK_COLOR,
                annotation_clip=False,
            )
    axes.set_xticks(positions)
    axes.set_xticklabels(
        [MODEL_DISPLAY_LABELS.get(label, label) for label in labels],
        fontsize=TICK_FONTSIZE,
    )
    axes.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    axes.set_ylabel("Inconsistency rate", fontsize=Y_LABEL_FONTSIZE)
    tops = [
        bar_block[label][population][bar]["ci_high"] for label in labels for bar in bars
    ]
    # Headroom above the whiskers for the methods legend and the rotated n labels.
    axes.set_ylim(0, min(max(tops) * 1.35, 1.0) if tops else 1.0)
    if part == "styles":
        theme.soft_legend(
            axes,
            ncol=len(bars),
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            borderaxespad=0.0,
            fontsize=LEGEND_FONTSIZE,
            columnspacing=0.9,
            handletextpad=0.5,
            handlelength=1.2,
            borderpad=0.45,
        )
    else:
        # The methods panel's upper right is empty: same-call bars are near zero.
        theme.soft_legend(axes, ncol=1, loc="upper right", fontsize=LEGEND_FONTSIZE)


def build_figure(
    bar_block: Dict[str, Dict[str, Any]], population: str, part: str
) -> Figure:
    """Return one standalone panel with CI whiskers; ``part`` is ``"styles"`` or ``"methods"`` (:data:`FIGURE_PARTS`)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with matplotlib.rc_context(theme.SOFT_STYLE):
        width = 6.8 if part == "styles" else 5.8
        figure, axes = plt.subplots(figsize=(width * _widening(bar_block), 3.4))
        figure.patch.set_facecolor(theme.CARD_BG)
        # The styles legend sits in a strip above the axes, so that panel's axes start lower.
        figure.subplots_adjust(
            top=0.86 if part == "styles" else 0.96,
            bottom=0.12,
            left=0.11,
            right=0.97,
        )
        _draw_panel(axes, bar_block, population, part)
        _fit_tick_labels(figure)
    return figure


def draw(
    bar_block: Dict[str, Dict[str, Any]],
    out_path: Path,
    population: str,
    part: str,
    dpi: int | None = None,
) -> None:
    """Render one panel for one population to ``out_path``."""
    figure = build_figure(bar_block, population, part)
    # After build_figure, which pins the Agg backend before pyplot loads.
    import matplotlib.pyplot as plt

    theme.save_soft_figure(figure, out_path, dpi=dpi)
    plt.close(figure)
