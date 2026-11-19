from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, TypeVar

from ruleforge.experiments.ruleset_size.curves import INCONSISTENCY_RATE
from ruleforge.experiments import style as theme

__all__ = ["SERIES_COLORS", "draw", "legend_order", "series_colors"]

T = TypeVar("T")

#: Legend order and each model's colour.
SERIES_COLORS = {
    "GPT-OSS-120B": theme.PASTEL_BLUE,
    "Gemma-4-31B": theme.PASTEL_CORAL,
    "Olmo-3-32B": theme.PASTEL_GREEN,
    "Muse-Glimmer-30B": theme.PASTEL_ROSE,
    "GPT-5.6 Luna": theme.PASTEL_PURPLE,
    "Claude Haiku 5.5": theme.PASTEL_CYAN,
}


def legend_order(items: Iterable[T], label: Callable[[T], str]) -> List[T]:
    """Return ``items`` in legend order, those without a colour last and by label."""
    rank = {name: index for index, name in enumerate(SERIES_COLORS)}
    return sorted(items, key=lambda item: (rank.get(label(item), len(rank)), label(item)))


def series_colors(labels: Sequence[str]) -> List[str]:
    """Return the colour of each judge in ``labels``; one without a colour takes, in turn, one no judge drawn has."""
    taken = {SERIES_COLORS[label] for label in labels if label in SERIES_COLORS}
    spare = cycle([color for color in SERIES_COLORS.values() if color not in taken] or list(SERIES_COLORS.values()))
    return [SERIES_COLORS[label] if label in SERIES_COLORS else next(spare) for label in labels]


def draw(curves: List[Dict[str, Any]], out_path: Path, dpi: int | None = None) -> None:
    """Draw the ruleset-size figure of ``curves`` to ``out_path``."""
    for field in ("metric", "strategy", "ci_level", "quantity"):
        values = {curve[field] for curve in curves}
        if len(values) > 1:
            raise ValueError(
                f"curves disagree on {field}: {sorted(map(str, values))} -- "
                "one figure cannot describe both"
            )
    (quantity,) = {curve["quantity"] for curve in curves}
    if quantity != INCONSISTENCY_RATE:
        raise ValueError(
            f"The ruleset-size figure plots the {INCONSISTENCY_RATE!r}; these curves hold "
            f"{quantity!r} (metric {curves[0]['metric']!r})"
        )
    curves = legend_order(curves, lambda curve: curve["label"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with matplotlib.rc_context(theme.SOFT_STYLE):
        figure, axes = plt.subplots(figsize=(7.2, 4.4))
        theme.soft_header(
            figure,
            "Inconsistency Rate Across Ruleset Sizes",
            None,
        )
        figure.subplots_adjust(top=0.87, bottom=0.15, left=0.11, right=0.98)
        theme.style_soft_axes(axes)

        for curve, fill in zip(curves, series_colors([curve["label"] for curve in curves])):
            points = curve["points"]
            sizes = [point["size"] for point in points]
            means = [point["mean"] for point in points]
            axes.fill_between(
                sizes,
                [point["ci_low"] for point in points],
                [point["ci_high"] for point in points],
                facecolor=fill, alpha=0.22, edgecolor="none", linewidth=0.0,
                zorder=1,
            )
            axes.plot(
                sizes, means, color=fill, linestyle="-", linewidth=2.0,
                marker="o", markersize=4.0, markerfacecolor=fill,
                markeredgecolor=theme.CARD_BG, markeredgewidth=0.8,
                label=curve["label"], zorder=3,
            )
        axes.set_xlabel("Subset Size")
        axes.set_ylabel("Inconsistency rate")
        axes.set_ylim(0.0, 1.0)
        max_size = max(
            point["size"] for curve in curves for point in curve["points"]
        )
        axes.set_xlim(1.5, max_size + 0.5)
        axes.set_xticks(list(range(2, max_size + 1, 5)))

        theme.soft_legend(axes, loc="lower right", ncol=2, handlelength=1.8)

    theme.save_soft_figure(figure, out_path, dpi=dpi)
    plt.close(figure)
