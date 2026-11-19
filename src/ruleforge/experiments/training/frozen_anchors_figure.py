from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ruleforge.experiments import style as theme
from ruleforge.experiments.training.family_arms import arm_names
from ruleforge.experiments.training.figure import ARM_STYLES, RATE_MARGINS, RATE_SIZE, rate_axes

if TYPE_CHECKING:
    from matplotlib.figure import Figure

PANELS = ("rates",)
#: Each inconsistency rate's (fill, edge, hatch), with the rule judgments its tick names. The hatch marks a trained
#: arm read against its own rule judgments; plain coral is the trained arm against the frozen base rule judgments.
RATE_BARS = {
    "base": (ARM_STYLES[0], None),
    "trained_updated": (ARM_STYLES[1], "own rule\njudgments"),
    "trained_frozen": ((*ARM_STYLES[1][:2], ""), "frozen base\nrule judgments"),
}


def panel_values(golden: Mapping[str, Any], paper: str, panel: str) -> list[dict[str, Any]]:
    """Return what ``panel`` draws for the family ``paper``: one row per bar, with its tick label."""
    if panel not in PANELS:
        raise ValueError(f"unknown panel {panel!r}; expected one of {PANELS}")
    family = golden["families"][paper]
    base, trained = arm_names(golden, paper)
    return [{"rate": rate, "label": base if anchor is None else f"{trained}\n({anchor})",
             **family["inconsistency"][rate]} for rate, (_, anchor) in RATE_BARS.items()]


def build_figure(golden: Mapping[str, Any], paper: str, panel: str) -> Figure:
    """Return one panel as a matplotlib figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = panel_values(golden, paper, panel)
    with matplotlib.rc_context(theme.SOFT_STYLE):
        figure, axes = plt.subplots(figsize=RATE_SIZE)
        figure.subplots_adjust(**RATE_MARGINS)
        figure.patch.set_facecolor(theme.CARD_BG)
        theme.style_soft_axes(axes)
        rate_axes(figure, axes, rows, [RATE_BARS[row["rate"]][0] for row in rows], [row["label"] for row in rows],
                  linespacing=0.95)
    return figure


def draw(golden: Mapping[str, Any], paper: str, panel: str, out_path: Path, dpi: int | None = None) -> None:
    """Render one panel of the family ``paper`` to ``out_path``."""
    figure = build_figure(golden, paper, panel)
    import matplotlib.pyplot as plt

    theme.save_soft_figure(figure, Path(out_path), dpi=dpi)
    plt.close(figure)
