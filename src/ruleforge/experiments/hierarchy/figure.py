from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from ruleforge.evaluate.multiplicity import ALPHA
from ruleforge.experiments import style as theme
from ruleforge.experiments.labels import MODEL_DISPLAY_LABELS, display_order

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["ARMS", "FLAT_CONTROLS", "PUBLISHED_SUBTITLE", "PUBLISHED_TITLE", "SIGNIFICANCE_LABEL", "SIGNIFICANCE_MARK",
           "build_figure", "caption_notes", "difference_label", "draw", "drawn_judges", "flat_control", "stack_note"]

#: Each arm's legend label and its bars' (fill, edge, hatch): the first two bar styles, as the prompt-ablation figure
#: draws its naive and encouraged bars.
ARMS = {
    "flat": ("flat", theme.BAR_STYLES[0]),
    "hierarchy": ("hierarchy", theme.BAR_STYLES[1]),
}
#: Appended to the tick of a judge whose paired difference hierarchy - flat is Holm-significant.
SIGNIFICANCE_MARK = "*"
SIGNIFICANCE_LABEL = "Holm-significant (hierarchy − flat)"
#: The kinds of flat control, in caption order, each with its sentence.
FLAT_CONTROLS = {
    "rerun": "Flat arm re-run beside the hierarchy arm, on its inference stack",
    "published": "Flat arm the judge's published naive arm, on the hierarchy arm's inference stack",
    "provider": "Flat arm the API judge's published naive arm, at the provider (a provider change since is not "
                "observable)",
}
#: The title and subtitle the caption carries (the figure draws neither).
PUBLISHED_TITLE = "Inconsistency Rate With the Model Spec's Hierarchy Shown"
PUBLISHED_SUBTITLE = "Flat: rules judged alone. Hierarchy: levels shown, each rule judged with the rules that outrank it"
_RATES = {"flat": "flat_rate", "hierarchy": "hierarchy_rate"}

# Panel size (inches) at style.PUBLISHED_GROUPS judges, widened in proportion beyond; the legend strip sits above the axes.
PANEL_SIZE = (6.8, 3.4)
PANEL_MARGINS = {"top": 0.80, "bottom": 0.12, "left": 0.11, "right": 0.97}
TICK_FONTSIZE = 11
Y_LABEL = "Inconsistency rate"
Y_LABEL_FONTSIZE = 12
#: A two-bar group's bar pitch, each bar drawn at 0.92 of it.
BAR_WIDTH = 0.3
BAR_KWARGS: dict[str, Any] = {"linewidth": 0.9, "capsize": 2.5,
                              "error_kw": {"linewidth": 0.9, "color": theme.TICK_COLOR}}
LEGEND_KWARGS: dict[str, Any] = {"loc": "lower left", "bbox_to_anchor": (0.0, 1.02), "borderaxespad": 0.0,
                                 "fontsize": 11, "columnspacing": 0.9, "handletextpad": 0.5, "handlelength": 1.2,
                                 "borderpad": 0.45}
#: Points between a whisker's top and its rate label.
LABEL_LIFT = 2


def _display(record: Mapping[str, Any]) -> str:
    return MODEL_DISPLAY_LABELS.get(record["judge"], record["judge"])


def _tick(record: Mapping[str, Any]) -> str:
    return _display(record) + (SIGNIFICANCE_MARK if record["primary"]["difference"]["significant_holm"] else "")


def drawn_judges(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the payload's judges in paper order; refuse a payload whose Holm family is not every judge drawn."""
    judges = payload["judges"]
    order = display_order([record["judge"] for record in judges])
    families = sorted({record["primary"]["difference"]["family_size"] for record in judges})
    if families != [len(judges)]:
        raise ValueError(f"the differences were Holm-adjusted over families of {families} judges, the figure draws "
                         f"{len(judges)}: its significance marks must be Holm over every judge drawn")
    return sorted(judges, key=lambda record: order.index(record["judge"]))


def flat_control(record: Mapping[str, Any]) -> str:
    """Return the kind of a judge's flat control (a :data:`FLAT_CONTROLS` key), from its arms' records."""
    if record["flat"]["stack"]["kind"] == "provider":
        return "provider"
    return "rerun" if record["flat"]["root"] == record["hierarchy"]["root"] else "published"


def stack_note(judges: Sequence[Mapping[str, Any]]) -> str:
    """Return the caption's sentences naming the judges under each kind of flat control."""
    kinds = [flat_control(record) for record in judges]
    notes = []
    for kind, words in FLAT_CONTROLS.items():
        named = [_display(record) for record, own in zip(judges, kinds) if own == kind]
        if named:
            notes.append(f"{words}: {', '.join(named)}")
    return ". ".join(notes)


def difference_label(difference: Mapping[str, Any]) -> str:
    """Return a difference as the caption prints it: ``+0.012 [-0.004, +0.029], p(Holm) = 0.034``."""
    return (f"{difference['estimate']:+.3f} [{difference['ci_low']:+.3f}, {difference['ci_high']:+.3f}], "
            f"p(Holm) = {difference['p_holm']:.3g}")


def caption_notes(payload: Mapping[str, Any]) -> str:
    """Return the Markdown section of what the figure leaves to its caption: the arms' definitions, the asterisk's
    meaning, each judge's difference, and the flat controls."""
    judges = drawn_judges(payload)
    level = f"{100 * payload['provenance']['bootstrap']['ci_level']:g}%"
    rows = [f"- {_tick(record)}: {difference_label(record['primary']['difference'])} "
            f"(n = {record['n_scenarios']} scenarios x {record['n_decoder_draws']} draws)" for record in judges]
    return "\n".join([
        "## The figure's caption notes (not drawn)",
        "",
        f"The published versions' title and subtitle: \"{PUBLISHED_TITLE}\". {PUBLISHED_SUBTITLE}.",
        "",
        f"Bars: each arm's inconsistency rate with its {level} scenario-cluster bootstrap interval. An asterisk after "
        "a judge's name marks a paired difference hierarchy - flat that is Holm-significant "
        f"(alpha {ALPHA:g}, Holm over the {len(judges)} judges drawn); a name without one, a difference that is not. "
        f"Per judge, the difference with its {level} scenario-cluster bootstrap interval and Holm-adjusted p "
        "(scenario-clustered paired randomization test):",
        "",
        *rows,
        "",
        f"{stack_note(judges)}.",
        "",
    ])


def _errors(rates: Sequence[Mapping[str, float]]) -> list[list[float]]:
    return [[max(0.0, rate["estimate"] - rate["ci_low"]) for rate in rates],
            [max(0.0, rate["ci_high"] - rate["estimate"]) for rate in rates]]


def _significance_key():
    """Return the legend handle for :data:`SIGNIFICANCE_MARK`: the tick labels' own asterisk glyph, centred."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.path import Path as Outline
    from matplotlib.textpath import TextPath

    glyph = TextPath((0, 0), SIGNIFICANCE_MARK, size=TICK_FONTSIZE, prop=FontProperties())
    box = glyph.get_extents()
    vertices = glyph.vertices - ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
    # matplotlib scales a marker path so its largest |vertex| is half the marker size; a CLOSEPOLY vertex is never
    # drawn, so it is moved to the centre where it cannot set that scale.
    vertices[glyph.codes == Outline.CLOSEPOLY] = 0.0
    return Line2D([], [], marker=Outline(vertices, glyph.codes), linestyle="none", markersize=2 * abs(vertices).max(),
                  markeredgewidth=0.0, color=theme.TICK_COLOR)


def _legend(figure: Figure, axes: Axes, handles: Sequence[Any]) -> None:
    labels = [label for label, _ in ARMS.values()] + [SIGNIFICANCE_LABEL]
    for columns in (len(labels), 2):
        legend = theme.soft_legend(axes, handles=[*handles, _significance_key()], labels=labels, ncol=columns,
                                   **LEGEND_KWARGS)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        if legend.get_window_extent(renderer).x1 <= axes.get_window_extent(renderer).x1:
            return


def _widen(figure: Figure, rows: Iterable[Sequence[Any]]) -> None:
    """Widen ``figure`` just enough that no two neighbouring artists of a row come closer than
    :data:`~ruleforge.experiments.style.TICK_LABEL_GAP` points."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    gap = theme.TICK_LABEL_GAP * figure.dpi / 72
    scale = 1.0
    for row in rows:
        boxes = sorted((artist.get_window_extent(renderer) for artist in row), key=lambda box: box.x0)
        for left, right in zip(boxes, boxes[1:]):
            centres = (right.x0 + right.x1 - left.x0 - left.x1) / 2
            scale = max(scale, ((left.width + right.width) / 2 + gap) / centres)
    if scale > 1.0:
        width, height = figure.get_size_inches()
        figure.set_size_inches(width * scale, height)


def _raise_ceiling(figure: Figure, axes: Axes, labels: Sequence[Any]) -> None:
    """Raise the y axis above 1 only as far as a rate label needs to stay inside the axes; ticks stay at 0..1."""
    import numpy as np

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    frame = axes.get_window_extent(renderer)
    to_display = axes.transData.transform
    ceiling = 1.0
    for label in labels:
        anchor = label.xy[1]
        rise = label.get_window_extent(renderer).y1 - to_display((label.xy[0], anchor))[1]
        ceiling = max(ceiling, anchor * frame.height / (frame.height - rise))
    axes.set_ylim(0.0, ceiling)
    axes.set_yticks(np.linspace(0.0, 1.0, 6))


def build_figure(payload: Mapping[str, Any]) -> Figure:
    """Return the figure: per judge, the flat and hierarchy rates with intervals and value labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    judges = drawn_judges(payload)
    positions = np.arange(len(judges))
    with matplotlib.rc_context(theme.SOFT_STYLE):
        width, height = PANEL_SIZE
        figure, axes = plt.subplots(figsize=(width * max(1.0, len(judges) / theme.PUBLISHED_GROUPS), height))
        figure.patch.set_facecolor(theme.CARD_BG)
        figure.subplots_adjust(**PANEL_MARGINS)
        theme.style_soft_axes(axes)
        handles, rates_drawn = [], []
        for index, (side, (label, (fill, edge, hatch))) in enumerate(ARMS.items()):
            xs = positions + (index - 0.5) * BAR_WIDTH
            rates = [record["primary"][_RATES[side]] for record in judges]
            handles.append(axes.bar(xs, [rate["estimate"] for rate in rates], BAR_WIDTH * 0.92, color=fill,
                                    edgecolor=edge, hatch=hatch, yerr=_errors(rates), label=label, **BAR_KWARGS))
            rates_drawn += [axes.annotate(f"{rate['estimate']:.3f}", xy=(x, rate["ci_high"]), xytext=(0, LABEL_LIFT),
                                          textcoords="offset points", ha="center", va="bottom", annotation_clip=False,
                                          **theme.VALUE_LABEL) for x, rate in zip(xs, rates)]
        axes.set_xticks(positions)
        axes.set_xticklabels([_tick(record) for record in judges], fontsize=TICK_FONTSIZE)
        axes.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        axes.set_ylabel(Y_LABEL, fontsize=Y_LABEL_FONTSIZE)
        axes.set_ylim(0.0, 1.0)
        _widen(figure, [axes.get_xticklabels(), rates_drawn])
        _legend(figure, axes, handles)
        _raise_ceiling(figure, axes, rates_drawn)
    return figure


def draw(payload: Mapping[str, Any], out_path: Path, *, dpi: int | None = None) -> None:
    """Render the figure to ``out_path`` (``.png`` or ``.pdf``)."""
    figure = build_figure(payload)
    import matplotlib.pyplot as plt

    theme.save_soft_figure(figure, Path(out_path), dpi=dpi)
    plt.close(figure)
