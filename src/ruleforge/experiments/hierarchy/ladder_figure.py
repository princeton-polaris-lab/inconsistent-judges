from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ruleforge.experiments import style as theme
from ruleforge.experiments.hierarchy.figure import ARMS as HIERARCHY_ARMS
from ruleforge.experiments.hierarchy.ladder import rung_label
from ruleforge.experiments.labels import MODEL_DISPLAY_LABELS, display_order

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["ARM_STYLES", "X_LABEL", "Y_LABEL", "build_figure", "draw"]

X_LABEL = "Hierarchy levels considered for inconsistency"
Y_LABEL = "Inconsistency rate"
#: Each arm's line colour (its bars' edge colour in the hierarchy comparison's figure) and marker, so the arms differ
#: without colour.
ARM_STYLES = {name: (style[1], marker) for (name, (_, style)), marker in zip(HIERARCHY_ARMS.items(), ("o", "s"))}
#: At most this many panels per row, one panel's size in inches, and the widest row (in panels) drawn at that size; a
#: wider row narrows its panels so its text prints as a three-panel row's.
MAX_COLUMNS = 4
PANEL_SIZE = (3.4, 2.9)
WIDEST_ROW = 3
TITLE_FONTSIZE = 12
TICK_FONTSIZE = 9.5
LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11
TICK_ROTATION = 30
#: How far apart, in rung units, the arms' points at one rung are drawn.
DODGE = 0.11


def build_figure(payload: Mapping[str, Any]) -> Figure:
    """Return one panel per judge: each arm's rate at every rung with its interval, and a key when two arms are drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    judges = display_order({arm["judge"] for arm in payload["arms"]})
    ticks = [rung_label(rung["levels"]) for rung in payload["rungs"]]
    rows = -(-len(judges) // MAX_COLUMNS)
    columns = -(-len(judges) // rows)
    width, height = PANEL_SIZE[0] * min(columns, WIDEST_ROW) / columns, PANEL_SIZE[1]
    with matplotlib.rc_context(theme.SOFT_STYLE):
        figure, grid = plt.subplots(rows, columns, figsize=(width * columns, height * rows + 0.6), squeeze=False,
                                    sharey=True, layout="constrained")
        figure.patch.set_facecolor(theme.CARD_BG)
        for axes, judge in zip(grid.flat, judges):
            theme.style_soft_axes(axes)
            arms = [arm for arm in payload["arms"] if arm["judge"] == judge]
            for offset, arm in enumerate(arms):
                rates = [rung["rate"] for rung in arm["rungs"]]
                xs = [rung["rung"] + (offset - (len(arms) - 1) / 2) * DODGE for rung in arm["rungs"]]
                means = [rate["estimate"] for rate in rates]
                color, marker = ARM_STYLES[arm["arm"]]
                axes.errorbar(
                    xs, means, yerr=[[max(0.0, m - r["ci_low"]) for m, r in zip(means, rates)],
                                     [max(0.0, r["ci_high"] - m) for m, r in zip(means, rates)]],
                    color=color, marker=marker, markersize=4.5, linewidth=1.4,
                    capsize=2.5, elinewidth=0.9, label=arm["arm"])
            axes.set_title(MODEL_DISPLAY_LABELS.get(judge, judge), fontsize=TITLE_FONTSIZE, pad=5,
                           color=theme.LABEL_COLOR)
            axes.set_xticks(range(1, len(ticks) + 1), ticks, rotation=TICK_ROTATION, ha="right",
                            rotation_mode="anchor")
            axes.tick_params(axis="both", labelsize=TICK_FONTSIZE)
            axes.set_xlim(0.6, len(ticks) + 0.4)
            axes.set_ylim(0.0, 1.0)
        for axes in list(grid.flat)[len(judges):]:
            axes.set_visible(False)
        figure.supxlabel(X_LABEL, fontsize=LABEL_FONTSIZE, color=theme.LABEL_COLOR)
        figure.supylabel(Y_LABEL, fontsize=LABEL_FONTSIZE, color=theme.LABEL_COLOR)
        handles, names = grid.flat[0].get_legend_handles_labels()
        # One arm needs no key: its figure's file and caption name it.
        if len(names) > 1:
            theme.soft_legend(figure, handles=handles, labels=names, loc="outside upper center", ncol=len(names),
                              fontsize=LEGEND_FONTSIZE)
    return figure


def draw(payload: Mapping[str, Any], out_path: Path, *, dpi: int | None = None) -> None:
    """Render the figure to ``out_path`` (``.png`` or ``.pdf``)."""
    figure = build_figure(payload)
    import matplotlib.pyplot as plt

    theme.save_soft_figure(figure, Path(out_path), dpi=dpi)
    plt.close(figure)
