from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ruleforge.evaluate.report import TRANSITION_LABELS
from ruleforge.experiments import style as theme
from ruleforge.experiments.training.family_arms import arm_names
from ruleforge.experiments.training.judgment_changes import COMPARISONS

if TYPE_CHECKING:
    from matplotlib.figure import Figure

PANELS = ("changes", "counts")
#: The kinds of judgment each panel draws: the changes panel all three, the counts panel the individual-rule and
#: full-ruleset judgments.
PANEL_KINDS = {"changes": COMPARISONS, "counts": tuple((kind, label) for kind, label in COMPARISONS
                                                       if kind != "aggregate")}
Y_LABELS = {"changes": "Base-to-GRPO change rate", "counts": "Judgment count (log scale)"}
#: Each panel's size (inches) and its axes' place in it; the changes panel is wide enough for its three tick labels.
SIZES = {"changes": (6.4, 3.6), "counts": (6.2, 3.8)}
POSITIONS = {"changes": (0.16, 0.25, 0.81, 0.71), "counts": (0.13, 0.25, 0.85, 0.72)}
#: Each label's fill and edge in the counts panel.
LABEL_COLORS = {
    "yes": (theme.PASTEL_CORAL, theme.DARK_CORAL),
    "unsure": (theme.PASTEL_GOLD, theme.DARK_GOLD),
    "no": (theme.PASTEL_BLUE, theme.DARK_BLUE),
}
COUNT_LIMITS = (1, 100_000)
CHANGE_WIDTH = 0.62
COUNT_SLOT = 0.23
COUNT_FILL = 0.9


def panel_values(golden: Mapping[str, Any], paper: str, panel: str) -> list[dict[str, Any]]:
    """Return what ``panel`` draws for the family ``paper``: one row per kind of judgment in :data:`PANEL_KINDS`."""
    if panel not in PANELS:
        raise ValueError(f"unknown panel {panel!r}; expected one of {PANELS}")
    family = golden["families"][paper]
    if panel == "changes":
        return [{"label": label, "n": family["changes"][kind]["n"], "rate": family["changes"][kind]["change_rate"]}
                for kind, label in PANEL_KINDS[panel]]
    return [{"label": label, **family["member_counts"][kind]} for kind, label in PANEL_KINDS[panel]]


def build_figure(golden: Mapping[str, Any], paper: str, panel: str) -> Figure:
    """Return one panel as a matplotlib figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = panel_values(golden, paper, panel)
    base, trained = arm_names(golden, paper)
    positions = np.arange(len(rows), dtype=float)
    with matplotlib.rc_context(theme.SOFT_STYLE):
        figure = plt.figure(figsize=SIZES[panel])
        figure.patch.set_facecolor(theme.CARD_BG)
        axes = figure.add_axes(POSITIONS[panel])
        theme.style_soft_axes(axes)
        if panel == "changes":
            rates = [row["rate"] for row in rows]
            bars = axes.bar(positions, rates, CHANGE_WIDTH, color=theme.PASTEL_GREEN, edgecolor=theme.DARK_GREEN,
                            linewidth=1.0, zorder=3, label=f"{base} → {trained}")
            for bar, rate in zip(bars, rates):
                axes.text(bar.get_x() + bar.get_width() / 2, rate + 0.03, f"{rate:.1%}", ha="center", va="bottom",
                          **theme.VALUE_LABEL)
            axes.set_ylim(0.0, 1.0)
            ticks = [f"{row['label']}\n({row['n']:,} judgments)" for row in rows]
            theme.soft_legend(axes, loc="upper left")
        else:
            for index, label in enumerate(TRANSITION_LABELS):
                fill, edge = LABEL_COLORS[label]
                counts = [row[label] for row in rows]
                # Bars start at 1, the log axis's floor.
                bars = axes.bar(positions + (index - 1) * COUNT_SLOT, [max(count - 1, 0) for count in counts],
                                COUNT_SLOT * COUNT_FILL, bottom=1, color=fill, edgecolor=edge, linewidth=1.0,
                                label=label.capitalize(), zorder=3)
                for bar, count in zip(bars, counts):
                    axes.text(bar.get_x() + bar.get_width() / 2, max(count, 1) * 1.18, f"{count:,}", ha="center",
                              va="bottom", **theme.VALUE_LABEL)
            axes.set_yscale("log")
            axes.set_ylim(*COUNT_LIMITS)
            axes.tick_params(axis="y", which="minor", colors=theme.SPINE_COLOR)
            ticks = [f"{row['label']}\n(n = {row['n']:,})" for row in rows]
            legend = theme.soft_legend(axes, title=trained, title_fontsize=9, loc="upper right",
                                       bbox_to_anchor=(0.99, 0.99), ncol=3, borderpad=0.35, columnspacing=0.9,
                                       handlelength=1.2)
            legend.get_title().set_color(theme.LABEL_COLOR)
        axes.set_xticks(positions)
        axes.set_xticklabels(ticks, linespacing=0.95)
        axes.set_ylabel(Y_LABELS[panel])
    return figure


def draw(golden: Mapping[str, Any], paper: str, panel: str, out_path: Path, dpi: int | None = None) -> None:
    """Render one panel of the family ``paper`` to ``out_path``."""
    figure = build_figure(golden, paper, panel)
    import matplotlib.pyplot as plt

    theme.save_soft_figure(figure, Path(out_path), dpi=dpi)
    plt.close(figure)
