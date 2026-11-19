from __future__ import annotations

from pathlib import Path
from typing import Any

CARD_BG = "#FFFFFF"
GRID_COLOR = "#E4E5E8"
SPINE_COLOR = "#BBBCC2"
TICK_COLOR = "#5E5F66"
LABEL_COLOR = "#363840"
TITLE_COLOR = "#191A1C"
SUBTITLE_COLOR = "#6E6F77"
LEGEND_EDGE = "#D6D7DC"
#: A neutral reference line drawn behind the data, such as a baseline the data are read against.
REFERENCE_GREY = "#8E8F96"

# Cyan and rose are checked to stay distinguishable from the rest under protanopia and deuteranopia.
PASTEL_BLUE = "#7BA7D7"
PASTEL_CORAL = "#F0A07E"
PASTEL_GREEN = "#9AC8A0"
PASTEL_PURPLE = "#B7A4DE"
PASTEL_GOLD = "#EBC56E"
PASTEL_CYAN = "#86DAE0"
PASTEL_ROSE = "#D68BA2"

#: Darker tones for bar edges and markers: matplotlib draws hatches in the edge colour, so the edge needs ink.
DARK_BLUE = "#5B87B7"
DARK_CORAL = "#D0805E"
DARK_GREEN = "#7AA880"
DARK_GOLD = "#C9A254"

#: A bar group's (fill, edge, hatch) for its first, second and third bar: blue, coral hatched, green dotted.
BAR_STYLES = (
    (PASTEL_BLUE, DARK_BLUE, ""),
    (PASTEL_CORAL, DARK_CORAL, "//"),
    (PASTEL_GREEN, DARK_GREEN, ".."),
)
#: The model groups a bar panel is laid out for at its standard width; more groups widen it in proportion.
PUBLISHED_GROUPS = 4
#: The least gap, in points, between two neighbouring x tick labels; a figure whose labels come closer is widened until
#: they clear it.
TICK_LABEL_GAP = 5.5

VALUE_LABEL: dict[str, Any] = {"fontsize": 9, "fontweight": "bold", "color": LABEL_COLOR}

#: rc overrides for ``matplotlib.rc_context``; font type 42 keeps PDF text selectable.
SOFT_STYLE: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Inter", "Helvetica Neue", "Helvetica", "Arial",
        "Liberation Sans", "DejaVu Sans",
    ],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def soft_header(
    figure: Any, title: str, subtitle: str | None = None, *, x: float = 0.5, fontsize: float = 13,
) -> None:
    """Draw the page-band title and optional subtitle, centred on ``x`` (figure fraction)."""
    figure.patch.set_facecolor(CARD_BG)
    figure.text(
        x, 0.965, title, ha="center", va="top",
        color=TITLE_COLOR, fontsize=fontsize, fontweight="bold",
    )
    if subtitle:
        figure.text(
            x, 0.895, subtitle, ha="center", va="top",
            color=SUBTITLE_COLOR, fontsize=9,
        )


def style_soft_axes(axes: Any) -> None:
    """Style ``axes``: left and bottom spines only, muted ticks, a faint grid behind the data."""
    axes.set_facecolor("none")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(SPINE_COLOR)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=SPINE_COLOR, labelcolor=TICK_COLOR, length=3.5, width=1.0)
    axes.xaxis.label.set_color(LABEL_COLOR)
    axes.yaxis.label.set_color(LABEL_COLOR)
    axes.grid(axis="y", color=GRID_COLOR, linewidth=1.0, linestyle="-")
    axes.set_axisbelow(True)


def save_soft_figure(figure: Any, out_path: Path, dpi: int | None = None) -> None:
    """Save ``figure`` to ``out_path`` inside the soft rc context (``pdf.fonttype`` is read at save time)."""
    import matplotlib

    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"dpi": dpi} if dpi is not None else {}
    with matplotlib.rc_context(SOFT_STYLE):
        figure.savefig(
            out_path, bbox_inches="tight", facecolor=figure.get_facecolor(),
            **extra,
        )


def soft_legend(axes: Any, **kwargs: Any) -> Any:
    """Draw a rounded, light-bordered legend; keyword arguments override the soft defaults."""
    defaults: dict[str, Any] = dict(
        frameon=True, framealpha=0.95, edgecolor=LEGEND_EDGE,
        facecolor=CARD_BG, fancybox=True, borderpad=0.6,
        handlelength=1.6, labelcolor=TICK_COLOR,
    )
    defaults.update(kwargs)
    legend = axes.legend(**defaults)
    legend.get_frame().set_linewidth(0.9)
    return legend
