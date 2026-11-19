from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping

from ruleforge.experiments.labels import MODEL_DISPLAY_LABELS, PAPER_STYLE_LABELS, display_order
from ruleforge.experiments.leniency.schema import direction
from ruleforge.evaluate.report import TRANSITION_LABELS
from ruleforge.experiments import style as theme
from ruleforge.prompts.styles import FORCED

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "FIGURE_TITLE",
    "FLIPS_FIGURE_TITLE",
    "MAIN_TEXT_BLUE_RULES",
    "build_figure",
    "build_flips_figure",
    "direction_significance_counts",
    "draw",
    "draw_flips",
    "main_text_rule_order",
    "mark_color",
    "rule_order_by_mean_delta",
    "style_note",
    "tick_label",
]

FIGURE_TITLE = "Per-Rule Violation Rates for Full Rule Set versus Individual Rule Judgments"
X_AXIS_LABEL = "Difference in violation rate (full ruleset − individual rule)"
X_TICK_FORMAT = "%.2f"
#: Y tick labels are `R12 · <title>`, the title cut at a word boundary to at most this many characters.
TITLE_CHARS = 33
#: Canvas width (in.) and the panel grid's horizontal span as figure fractions.
FIGURE_WIDTH = 14.5
PANEL_LEFT = 0.255
PANEL_RIGHT = 0.989
PANEL_CENTER = (PANEL_LEFT + PANEL_RIGHT) / 2
HEADER_CENTER = 0.5
#: Print sizes (pt) for the forest plots' text, and their marks to match.
TITLE_FONTSIZE = 17
PANEL_TITLE_FONTSIZE = 13
YTICK_FONTSIZE = 12.5
XTICK_FONTSIZE = 11.5
XLABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 13
MARKER_AREA = 44
MARKER_EDGE = 1.3
LEGEND_MARKER_SIZE = 7.5
WHISKER_WIDTH = 1.2
#: Figure-fraction positions for the legend strip and the panel grid on the compact panel; a taller figure keeps the
#: same bands in inches.
TITLE_Y = 0.965
LEGEND_Y = 0.910
AXES_TOP = 0.775
#: Reserve a clean band between tick labels and the shared x-axis label.
AXES_BOTTOM = 0.150
XLABEL_Y = 0.025
#: Compact panel: the ten most negative cross-model mean effects that are Holm-significant decreases for every judge,
#: plus every rule with a Holm-significant increase for any judge.
MAIN_TEXT_BLUE_RULES = 10
#: Inches per rule row of the forest plots, and the floor of the compact one.
ROW_HEIGHT = 0.45
MIN_HEIGHT = 5.8
ALL_RULES_HEIGHT = 18.5
#: A row on fewer than `few_informative_below` informative scenarios is faded with a dashed whisker: its interval can
#: exclude zero where the sign-flip test cannot reject.
FEW_INFORMATIVE_ALPHA = 0.45
FEW_INFORMATIVE_DASHES = (0, (2.0, 1.4))

FLIPS_FIGURE_TITLE = "Per-Rule Judgments: Judged Alone vs. Inside the Full Ruleset"
FLIPS_FOOTER = (
    "Expected shares of (scenario × rule) cells: every judged-alone draw paired with every "
    "in-full-ruleset draw of the scenario; draws agree = two distinct draws of one side; "
    "flat metric; descriptive."
)


def _effects_by_rule(payload: Mapping[str, Any]) -> Dict[str, Dict[int, Mapping[str, Any]]]:
    return {
        label: {row["rule"]: row for row in model["effects"]}
        for label, model in payload["models"].items()
    }


def rule_order_by_mean_delta(payload: Mapping[str, Any]) -> List[int]:
    """Return rule numbers by cross-model mean delta (exact fractions, so ties do not depend on model order), most
    negative first; ties on number."""
    effects = _effects_by_rule(payload)
    numbers = sorted({number for rows in effects.values() for number in rows})

    def total(number: int) -> Fraction:
        rows = [by_rule[number] for by_rule in effects.values()]
        return sum(
            (Fraction(row["yes_difference"], row["n_scenarios"] * row["n_decoder_draws"]) for row in rows),
            Fraction(0),
        )

    return sorted(numbers, key=lambda number: (total(number), number))


def main_text_rule_order(payload: Mapping[str, Any], *, n_blue: int = MAIN_TEXT_BLUE_RULES) -> List[int]:
    """Return the compact panel's rules, in :func:`rule_order_by_mean_delta` order; every rule if none qualifies."""
    order = rule_order_by_mean_delta(payload)
    effects = list(_effects_by_rule(payload).values())
    blue = [n for n in order if all(direction(rows[n]) < 0 and rows[n]["significant_holm"] for rows in effects)]
    red = {n for n in order if any(direction(rows[n]) > 0 and rows[n]["significant_holm"] for rows in effects)}
    selected = set(blue[:n_blue]) | red
    return [n for n in order if n in selected] or order


def direction_significance_counts(payload: Mapping[str, Any]) -> Dict[str, int]:
    """Return all model-rule estimates by direction and Holm significance, each count with a ``_mc_margin`` companion."""
    names = {-1: "blue", 0: "tie", 1: "red"}
    cells = [f"{name}_{suffix}" for name in ("blue", "red", "tie") for suffix in ("significant", "not_significant")]
    counts = {**{cell: 0 for cell in cells}, **{f"{cell}_mc_margin": 0 for cell in cells}}
    for model in payload["models"].values():
        for row in model["effects"]:
            significance = "significant" if row["significant_holm"] else "not_significant"
            cell = f"{names[direction(row)]}_{significance}"
            counts[cell] += 1
            counts[f"{cell}_mc_margin"] += int(row["holm_mc_margin"])
    return counts


def mark_color(row: Mapping[str, Any]) -> str:
    """Diverging pair from the theme; the neutral carries a tie and a zero-width row."""
    sign = direction(row)
    if row["degenerate"] or sign == 0:
        return theme.TICK_COLOR
    return theme.DARK_BLUE if sign < 0 else theme.DARK_CORAL


def tick_label(number: int, title: str) -> str:
    """``R<n> · <title>``, the title cut at a word boundary to at most :data:`TITLE_CHARS` characters."""
    if len(title) > TITLE_CHARS:
        cut = title[: TITLE_CHARS - 1]
        space = cut.rfind(" ")
        title = (cut[:space] if space > TITLE_CHARS // 2 else cut).rstrip(" ,;:") + "…"
    return f"R{number} · {title}"


def _display(label: str) -> str:
    return MODEL_DISPLAY_LABELS.get(label, label)


def style_note(payload: Mapping[str, Any]) -> str:
    """Return the golden's prompt style as a title names it: ``"<paper name> prompt"`` for the paper's forced style,
    nothing otherwise."""
    style = payload["settings"]["prompt_style"]
    return f"{PAPER_STYLE_LABELS[style]} prompt" if style == FORCED else ""


def build_figure(payload: Mapping[str, Any], *, compact: bool = True) -> Figure:
    """Return the compact or all-rules forest plot; markers are filled when Holm-significant and coloured by sign."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    order = main_text_rule_order(payload) if compact else rule_order_by_mean_delta(payload)
    titles = {rule["number"]: rule["title"] for rule in payload["rules"]}
    labels = display_order(payload["models"])
    effects = _effects_by_rule(payload)
    position = {number: index for index, number in enumerate(order)}
    lowest = min(effects[label][n]["ci_low"] for label in labels for n in order)
    highest = max(effects[label][n]["ci_high"] for label in labels for n in order)
    pad = max(0.015, 0.04 * (highest - lowest))
    height = max(MIN_HEIGHT, ROW_HEIGHT * len(order) + 0.8) if compact else ALL_RULES_HEIGHT
    scale = MIN_HEIGHT / height
    settings = payload["settings"]
    level = f"{100 * settings['bootstrap']['ci_level']:g}%"
    few_below = settings["few_informative_below"]
    drawn = [effects[label][n] for label in labels for n in order]

    with matplotlib.rc_context(theme.SOFT_STYLE):
        figure, axes = plt.subplots(1, len(labels), figsize=(FIGURE_WIDTH, height), sharey=True, squeeze=False)
        axes = list(axes[0])
        theme.soft_header(figure, FIGURE_TITLE, None, x=HEADER_CENTER, fontsize=TITLE_FONTSIZE)
        figure.subplots_adjust(top=TITLE_Y - (TITLE_Y - AXES_TOP) * scale, bottom=AXES_BOTTOM * scale,
                               left=PANEL_LEFT, right=PANEL_RIGHT, wspace=0.10)
        for ax, label in zip(axes, labels):
            theme.style_soft_axes(ax)
            ax.axvline(0, color=theme.SPINE_COLOR, linewidth=WHISKER_WIDTH, zorder=1)
            for number in order:
                row = effects[label][number]
                y = position[number]
                color = mark_color(row)
                few = dict(alpha=FEW_INFORMATIVE_ALPHA) if row["few_informative"] else {}
                ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=theme.TICK_COLOR,
                        linewidth=WHISKER_WIDTH, solid_capstyle="butt", zorder=2,
                        linestyle=FEW_INFORMATIVE_DASHES if few else "-", **few)
                # A hollow marker takes the surface colour, not "none", so it stays legible on a guide line.
                ax.scatter([row["delta"]], [y], s=MARKER_AREA, linewidths=MARKER_EDGE, zorder=3, edgecolors=color,
                           facecolors=color if row["significant_holm"] else theme.CARD_BG, **few)
            ax.set_xlim(lowest - pad, highest + pad)
            ax.xaxis.set_major_formatter(FormatStrFormatter(X_TICK_FORMAT))
            ax.set_ylim(len(order) - 0.5, -0.5)
            ax.set_title(_display(label), fontsize=PANEL_TITLE_FONTSIZE, color=theme.LABEL_COLOR)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=XTICK_FONTSIZE)
        axes[0].set_yticks(range(len(order)))
        axes[0].set_yticklabels([tick_label(number, titles[number]) for number in order],
                                fontsize=YTICK_FONTSIZE)
        figure.supxlabel(X_AXIS_LABEL, x=PANEL_CENTER, y=XLABEL_Y * scale, fontsize=XLABEL_FONTSIZE,
                         color=theme.LABEL_COLOR)

        def marker(filled: bool) -> Line2D:
            return Line2D([], [], marker="o", linestyle="none", markersize=LEGEND_MARKER_SIZE,
                          markerfacecolor=theme.TICK_COLOR if filled else theme.CARD_BG,
                          markeredgecolor=theme.TICK_COLOR, markeredgewidth=MARKER_EDGE)

        # Colour carries direction, fill carries Holm significance.
        handles = [
            Line2D([], [], color=theme.DARK_BLUE, linewidth=3.5),
            Line2D([], [], color=theme.DARK_CORAL, linewidth=3.5),
            Line2D([], [], color=theme.TICK_COLOR, linewidth=3.5),
            marker(True),
            marker(False),
            Line2D([], [], color=theme.TICK_COLOR, linewidth=WHISKER_WIDTH),
        ]
        names = [
            "Lower in full ruleset",
            "Higher in full ruleset",
            "No change",
            "Holm-significant",
            "Not significant",
            f"{level} scenario-cluster bootstrap CI",
        ]
        theme.soft_legend(figure, handles=handles, labels=names, loc="upper center",
                          bbox_to_anchor=(HEADER_CENTER, TITLE_Y - (TITLE_Y - LEGEND_Y) * scale),
                          ncol=len(handles), fontsize=LEGEND_FONTSIZE, columnspacing=1.0, handletextpad=0.4,
                          handlelength=1.2, borderpad=0.5)
        foot = []
        if any(row["few_informative"] for row in drawn):
            key = Line2D([], [], color=theme.TICK_COLOR, linewidth=WHISKER_WIDTH, linestyle=FEW_INFORMATIVE_DASHES,
                         marker="o", markersize=LEGEND_MARKER_SIZE, markerfacecolor=theme.CARD_BG,
                         markeredgecolor=theme.TICK_COLOR, markeredgewidth=MARKER_EDGE, alpha=FEW_INFORMATIVE_ALPHA)
            foot.append((key, f"Fewer than {few_below} informative scenarios"))
        if foot:
            theme.soft_legend(figure, handles=[key for key, _ in foot], labels=[name for _, name in foot],
                              loc="lower left", bbox_to_anchor=(0.01, XLABEL_Y * scale * 0.5), fontsize=LEGEND_FONTSIZE,
                              handletextpad=0.5, handlelength=2.2, borderpad=0.5)
    return figure


def draw(payload: Mapping[str, Any], out_path: Path, dpi: int | None = None, *, compact: bool = True) -> None:
    """Render a forest plot to ``out_path`` (format by extension)."""
    figure = build_figure(payload, compact=compact)
    # After build_figure, which pins the Agg backend before pyplot loads.
    import matplotlib.pyplot as plt

    theme.save_soft_figure(figure, out_path, dpi=dpi)
    plt.close(figure)


def _share_text(share: float) -> str:
    if 0 < share < 0.0005:
        return "<0.1%"
    return f"{100 * share:.1f}%"


def build_flips_figure(payload: Mapping[str, Any]) -> Figure:
    """Return one row-shaded alone -> in-full-ruleset share matrix per model, each with its agreement figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    labels = display_order(payload["models"])
    size = len(TRANSITION_LABELS)
    cmap = LinearSegmentedColormap.from_list("softblue", ["#FFFFFF", theme.PASTEL_BLUE, theme.DARK_BLUE])
    with matplotlib.rc_context(theme.SOFT_STYLE):
        figure, axes = plt.subplots(1, len(labels), figsize=(12.8, 4.5), squeeze=False)
        axes = list(axes[0])
        note = style_note(payload)
        theme.soft_header(figure, FLIPS_FIGURE_TITLE + (f" ({note})" if note else ""))
        figure.subplots_adjust(top=0.66, bottom=0.20, left=0.055, right=0.985, wspace=0.34)
        figure.text(0.5, 0.03, FLIPS_FOOTER, ha="center", va="bottom", fontsize=8, color=theme.LABEL_COLOR)
        for ax, label in zip(axes, labels):
            flips = payload["models"][label]["flips"]
            matrix = np.array([[flips["matrix"][a][b] for b in TRANSITION_LABELS] for a in TRANSITION_LABELS])
            row_fraction = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-12, None)
            ax.imshow(row_fraction, cmap=cmap, vmin=0, vmax=1)
            for i in range(size):
                for j in range(size):
                    ax.text(j, i, _share_text(matrix[i, j]), ha="center", va="center", fontsize=9,
                            color="white" if row_fraction[i, j] > 0.55 else theme.LABEL_COLOR,
                            fontweight="bold" if i == j else "normal")
            references = flips["draw_to_draw"]
            ax.set_title(f"{_display(label)}\n{flips['unchanged_fraction']:.0%} unchanged",
                         fontsize=10, color=theme.LABEL_COLOR, pad=38)
            ax.text(0.5, 1.025,
                    f"floor {flips['chance_agreement']:.2f} · κ {flips['kappa']:.2f}\n"
                    f"draws agree: alone {references['alone']['unchanged_fraction']:.0%} · "
                    f"full {references['in_context']['unchanged_fraction']:.0%}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2,
                    color=theme.TICK_COLOR, linespacing=1.3)
            ax.set_xticks(range(size))
            ax.set_xticklabels(TRANSITION_LABELS)
            ax.set_yticks(range(size))
            ax.set_yticklabels(TRANSITION_LABELS)
            ax.set_xlabel("in full ruleset")
            ax.set_ylabel("judged alone")
            ax.tick_params(colors=theme.SPINE_COLOR, labelcolor=theme.TICK_COLOR, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.xaxis.label.set_color(theme.LABEL_COLOR)
            ax.yaxis.label.set_color(theme.LABEL_COLOR)
    return figure


def draw_flips(payload: Mapping[str, Any], out_path: Path, dpi: int | None = None) -> None:
    """Render the flips matrices to ``out_path`` (format by extension)."""
    figure = build_flips_figure(payload)
    import matplotlib.pyplot as plt

    theme.save_soft_figure(figure, out_path, dpi=dpi)
    plt.close(figure)
