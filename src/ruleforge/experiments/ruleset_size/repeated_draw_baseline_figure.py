from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ruleforge.experiments import style as theme
from ruleforge.experiments.ruleset_size.figure import series_colors

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["BASELINE_COLOR", "MEDIAN_COLOR", "X_LABEL", "Y_LABEL", "build_figure", "draw"]

X_LABEL = "Ruleset size"
Y_LABEL = "Inconsistency rate"
BASELINE_COLOR = theme.REFERENCE_GREY
#: The median inconsistency rate: a thin dark dotted line over the rate, with no band (its interval is in the golden
#: and the CSV).
MEDIAN_COLOR = theme.LABEL_COLOR
MEDIAN_WIDTH = 1.1
RATE_NAME, MEDIAN_NAME, BASELINE_NAME = "Inconsistency rate", "Median inconsistency rate", "Repeated-draw baseline"
#: At most this many panels per row, and one panel's size in inches.
MAX_COLUMNS = 3
PANEL_SIZE = (3.0, 2.5)
TITLE_FONTSIZE = 11
LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 10
#: Long enough for the rate's strip of judge colours and two dashes of the baseline.
LEGEND_HANDLE_LENGTH = 3.2
TICK_STEP = 10


def _ticks(sizes: list[int]) -> list[int]:
    inner = [size for size in range(TICK_STEP, sizes[-1], TICK_STEP) if size > sizes[0]]
    return [sizes[0], *inner, sizes[-1]]


def _band(axes: Any, points: list[Mapping[str, Any]], sizes: list[int], name: str, color: str, *, linestyle: str,
          label: str, zorder: int) -> None:
    rows = [point[name] for point in points]
    axes.fill_between(sizes, [row["ci_low"] for row in rows], [row["ci_high"] for row in rows], facecolor=color,
                      alpha=0.25, edgecolor="none", linewidth=0.0, zorder=zorder)
    axes.plot(sizes, [row["estimate"] for row in rows], color=color, linestyle=linestyle, linewidth=1.8, label=label,
              zorder=zorder + 1)


def build_figure(payload: Mapping[str, Any]) -> Figure:
    """Return one panel per judge in the payload's order: its inconsistency rate and repeated-draw baseline, each
    with its interval, and its median inconsistency rate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    judges = payload["judges"]
    rows = math.ceil(len(judges) / MAX_COLUMNS)
    columns = math.ceil(len(judges) / rows)
    colors = series_colors([judge["label"] for judge in judges])
    with matplotlib.rc_context(theme.SOFT_STYLE):
        figure, grid = plt.subplots(rows, columns, figsize=(PANEL_SIZE[0] * columns, PANEL_SIZE[1] * rows + 0.5),
                                    squeeze=False, sharex=True, sharey=True, layout="constrained")
        figure.patch.set_facecolor(theme.CARD_BG)
        for axes, judge, color in zip(grid.flat, judges, colors):
            theme.style_soft_axes(axes)
            points = judge["points"]
            sizes = [point["size"] for point in points]
            _band(axes, points, sizes, "repeated_draw_baseline", BASELINE_COLOR, linestyle="--", label=BASELINE_NAME,
                  zorder=1)
            _band(axes, points, sizes, "inconsistency", color, linestyle="-", label=RATE_NAME, zorder=3)
            axes.plot(sizes, [point["median_inconsistency"]["estimate"] for point in points], color=MEDIAN_COLOR,
                      linestyle=":", linewidth=MEDIAN_WIDTH, label=MEDIAN_NAME, zorder=5)
            axes.set_title(judge["label"], fontsize=TITLE_FONTSIZE, pad=5, color=theme.LABEL_COLOR)
            axes.set_xticks(_ticks(sizes))
            axes.set_xlim(sizes[0] - 1, sizes[-1] + 1)
            axes.set_ylim(0.0, 1.0)
        for axes in list(grid.flat)[len(judges):]:
            axes.set_visible(False)
        # A shared x axis labels only the bottom row's ticks; a panel above an empty slot is a column's bottom too.
        for index, axes in enumerate(list(grid.flat)[:len(judges)]):
            if index + columns >= len(judges):
                axes.xaxis.set_tick_params(labelbottom=True)
        figure.supxlabel(X_LABEL, fontsize=LABEL_FONTSIZE, color=theme.LABEL_COLOR)
        figure.supylabel(Y_LABEL, fontsize=LABEL_FONTSIZE, color=theme.LABEL_COLOR)
        rate = tuple(Line2D([], [], color=color, linestyle="-", linewidth=1.8) for color in colors)
        handles = [rate, Line2D([], [], color=MEDIAN_COLOR, linestyle=":", linewidth=MEDIAN_WIDTH),
                   Line2D([], [], color=BASELINE_COLOR, linestyle="--", linewidth=1.8)]
        theme.soft_legend(figure, handles=handles, labels=[RATE_NAME, MEDIAN_NAME, BASELINE_NAME],
                          loc="outside upper center", ncol=len(handles), fontsize=LEGEND_FONTSIZE,
                          handlelength=LEGEND_HANDLE_LENGTH, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    return figure


def draw(payload: Mapping[str, Any], out_path: Path, *, dpi: int | None = None) -> None:
    """Render the figure to ``out_path`` (``.png`` or ``.pdf``)."""
    figure = build_figure(payload)
    import matplotlib.pyplot as plt

    theme.save_soft_figure(figure, Path(out_path), dpi=dpi)
    plt.close(figure)
