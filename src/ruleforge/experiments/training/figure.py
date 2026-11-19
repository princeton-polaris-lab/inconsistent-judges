from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ruleforge.experiments import style as theme

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

PANELS = ("a",)
#: (fill, edge, hatch) per arm in declared order; hatches tell arms apart beyond colour.
ARM_STYLES = (
    (theme.PASTEL_BLUE, theme.DARK_BLUE, ""),
    (theme.PASTEL_CORAL, theme.DARK_CORAL, "//"),
    (theme.PASTEL_GREEN, theme.DARK_GREEN, ".."),
)
#: Tick-label font sizes tried, largest first, when wrapped labels still overlap.
TICK_FONT_SIZES = (10, 9, 8, 7)
Y_LABELS = {"a": "Inconsistency rate"}
BAR_WIDTH = 0.6
#: The rate panel's size (inches) and margins.
RATE_SIZE = (4.6, 3.4)
RATE_MARGINS = {"top": 0.95, "bottom": 0.14, "left": 0.15, "right": 0.97}
BAR_FIELDS = {"a": "flat"}


def drawn_labels(golden: Mapping[str, Any], names: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the text drawn for each arm, by golden label: ``names``' entry for its policy, else the label itself;
    two arms drawn alike are refused."""
    names = names or {}
    drawn = {golden["arms"][policy]["label"]: names.get(policy, golden["arms"][policy]["label"])
             for policy in golden["policies"]}
    texts = list(drawn.values())
    shared = sorted({text for text in texts if texts.count(text) > 1})
    if shared:
        raise ValueError(f"drawn labels {shared} name more than one arm: a panel draws an arm by its drawn label")
    return drawn


def panel_values(golden: Mapping[str, Any], panel: str) -> dict[str, Any]:
    """Return the golden rows ``panel`` draws, by arm label."""
    if panel not in PANELS:
        raise ValueError(f"unknown training panel {panel!r}; expected one of {PANELS}")
    arms = [golden["arms"][policy] for policy in golden["policies"]]
    if len(arms) > len(ARM_STYLES):
        raise ValueError(f"{len(arms)} arms; the panels draw at most {len(ARM_STYLES)} (base, pick, alternate)")
    labels = [arm["label"] for arm in arms]
    shared = sorted({label for label in labels if labels.count(label) > 1})
    if shared:
        raise ValueError(f"display labels {shared} name more than one arm: a panel draws an arm by its display label")
    return {arm["label"]: dict(arm[BAR_FIELDS[panel]]) for arm in arms}


def _errors(entries) -> list[list[float]]:
    return [
        [max(0.0, e["estimate"] - e["ci_low"]) if e["estimate"] is not None else 0.0 for e in entries],
        [max(0.0, e["ci_high"] - e["estimate"]) if e["estimate"] is not None else 0.0 for e in entries],
    ]


def _bars(axes: Any, positions, entries, styles, width: float, **kwargs: Any) -> None:
    heights = [0.0 if e["estimate"] is None else e["estimate"] for e in entries]
    bars = axes.bar(
        positions, heights, width, color=[style[0] for style in styles], edgecolor=[style[1] for style in styles],
        linewidth=0.9, yerr=_errors(entries), capsize=2.5,
        error_kw={"linewidth": 0.9, "color": theme.TICK_COLOR}, **kwargs,
    )
    for bar, (_, _, hatch) in zip(bars, styles):
        bar.set_hatch(hatch)


def _label(axes: Any, x: float, entry: Mapping[str, Any], text: str) -> None:
    top = entry["ci_high"] if entry["ci_high"] is not None else 0.0
    axes.annotate(
        text, xy=(x, top), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
        annotation_clip=False, **theme.VALUE_LABEL,
    )


def _tick_labels_overlap(figure: Any, axes: Any) -> bool:
    figure.canvas.draw()
    boxes = [text.get_window_extent() for text in axes.get_xticklabels()]
    return any(left.x1 >= right.x0 for left, right in zip(boxes, boxes[1:]))


def _two_lines(label: str) -> str:
    spaces = [index for index, char in enumerate(label) if char == " "]
    if not spaces or "\n" in label:
        return label
    cut = min(spaces, key=lambda index: abs(2 * index - len(label)))
    return f"{label[:cut]}\n{label[cut + 1:]}"


def _fit_tick_labels(figure: Any, axes: Any, labels: Sequence[str]) -> None:
    if not _tick_labels_overlap(figure, axes):
        return
    wrapped = [_two_lines(label) for label in labels]
    for size in TICK_FONT_SIZES:
        axes.set_xticklabels(wrapped, fontsize=size)
        if not _tick_labels_overlap(figure, axes):
            return
    logger.warning("the training panel's tick labels %s overlap even at %s pt; shorten the display labels", labels,
                   TICK_FONT_SIZES[-1])


def rate_axes(figure: Any, axes: Any, entries: Sequence[Mapping[str, Any]], styles: Sequence[tuple[str, str, str]],
              shown: Sequence[str], **tick_style: Any) -> None:
    """Draw one inconsistency-rate bar per entry, with its interval and value, on the 0-1 rate axis."""
    import numpy as np

    positions = np.arange(len(entries))
    _bars(axes, positions, entries, styles, BAR_WIDTH)
    for x, entry in zip(positions, entries):
        _label(axes, x, entry, "undefined" if entry["estimate"] is None else f"{entry['estimate']:.3f}")
    axes.set_xticks(positions)
    axes.set_xticklabels(shown, fontsize=10, **tick_style)
    _fit_tick_labels(figure, axes, shown)
    axes.set_ylim(0, 1.12)
    axes.set_yticks(np.linspace(0.0, 1.0, 6))
    axes.set_ylabel(Y_LABELS["a"], fontsize=10.5)


def build_figure(golden: Mapping[str, Any], panel: str, names: Mapping[str, str] | None = None) -> Figure:
    """Return one panel as a matplotlib figure, its arms drawn under :func:`drawn_labels`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = panel_values(golden, panel)
    drawn = drawn_labels(golden, names)
    with matplotlib.rc_context(theme.SOFT_STYLE):
        figure, axes = plt.subplots(figsize=RATE_SIZE)
        figure.patch.set_facecolor(theme.CARD_BG)
        figure.subplots_adjust(**RATE_MARGINS)
        theme.style_soft_axes(axes)
        rate_axes(figure, axes, list(values.values()), ARM_STYLES[: len(values)], [drawn[label] for label in values])
    return figure


def draw(golden: Mapping[str, Any], out_path: Path, panel: str, dpi: int | None = None,
         names: Mapping[str, str] | None = None) -> None:
    """Render one panel to ``out_path``."""
    figure = build_figure(golden, panel, names)
    import matplotlib.pyplot as plt

    theme.save_soft_figure(figure, Path(out_path), dpi=dpi)
    plt.close(figure)
