from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ruleforge.experiments.ruleforge.comparison import DATA_VERSION as CAMPAIGN_DATA_VERSION, TRANSITION_METHOD
from ruleforge.evaluate.stack import recorded_mismatch_refusal
from ruleforge.experiments import driver, style as theme


#: Display names of the golden's labels; a label not listed is drawn as given.
DISPLAY_LABELS = {
    "Seed 5 (winner)": "Revised ruleset 1",
    "Seed 9 (runner-up)": "Revised ruleset 2",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draw the RuleForge comparison panels from a campaign golden.")
    driver.add_from_golden(parser, required=True)
    driver.add_figure(parser, "--figure_a", "where to draw panel A", repeatable=False)
    driver.add_figure(parser, "--figure_b", "where to draw panel B", repeatable=False)
    driver.add_dpi(parser)
    driver.add_allow_stack_mismatch(parser, "draw a campaign golden built from estimates that report "
                                    "inference-stack mismatches (provenance.stack_mismatches)")
    args = driver.parse(parser, argv)

    stats = json.loads(args.from_golden.read_text(encoding="utf-8"))
    version = stats.get("data_version")
    if version != CAMPAIGN_DATA_VERSION:
        parser.error(
            f"{args.from_golden} has data_version {version!r}, expected "
            f"{CAMPAIGN_DATA_VERSION}; regenerate it from its producer"
        )
    refusal = recorded_mismatch_refusal(stats.get("provenance") or {}, str(args.from_golden))
    if refusal is not None and not args.allow_stack_mismatch:
        parser.error(refusal)
    for panel, out_path in (("a", args.figure_a), ("b", args.figure_b)):
        if out_path:
            draw(stats, out_path, panel, dpi=args.dpi, allow_stack_mismatch=args.allow_stack_mismatch)
            print(f"wrote {out_path}")


def build_figure(stats: dict, panel: str, *, allow_stack_mismatch: bool = False):
    """Return one panel (``"a"`` or ``"b"``) from its golden, alone; stack mismatches are refused unless
    ``allow_stack_mismatch``."""
    if panel not in {"a", "b"}:
        raise ValueError(f"unknown RuleForge panel: {panel!r}")
    refusal = recorded_mismatch_refusal(stats.get("provenance") or {}, "the RuleForge golden")
    if refusal is not None and not allow_stack_mismatch:
        raise ValueError(refusal)
    if (
        stats.get("transition_method") != TRANSITION_METHOD
        or stats.get("transition_units") != "share"
        or not stats.get("provenance", {}).get("complete")
    ):
        raise ValueError("campaign figures require complete expected-share provenance")
    labels = stats["labels"]
    if len(labels) < 2:
        raise ValueError("The RuleForge comparison requires an initial and at least one revision")
    # Imported here so the module imports without the viz group.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with matplotlib.rc_context(theme.SOFT_STYLE):
        fig = plt.figure(figsize=(5.5, 4.0))
        ax = fig.add_axes([0.14, 0.18, 0.83, 0.74] if panel == "a" else [0.20, 0.18, 0.77, 0.74])
        theme.style_soft_axes(ax)
        (_draw_rates if panel == "a" else _draw_transitions)(ax, stats, labels)
    return fig


def _draw_rates(ax, stats: dict, labels: list) -> None:
    for index, label in enumerate(labels):
        entry = stats["rulesets"][label]
        mean = entry["mean"]
        fill, edge, hatch = theme.BAR_STYLES[index % len(theme.BAR_STYLES)]
        ax.bar(
            [index],
            [mean],
            0.62,
            yerr=[[mean - entry["ci_low"]], [entry["ci_high"] - mean]],
            capsize=3,
            color=fill,
            edgecolor=edge,
            hatch=hatch,
            linewidth=0.9,
            error_kw={"linewidth": 0.9, "color": theme.TICK_COLOR},
        )
        ax.annotate(
            f"{mean:.3f}",
            (index, entry["ci_high"] + 0.015),
            ha="center",
            va="bottom",
            **theme.VALUE_LABEL,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([DISPLAY_LABELS.get(label, label) for label in labels])
    ax.set_ylabel("Inconsistency rate")
    ax.set_ylim(0, 1)


def _draw_transitions(ax, stats: dict, labels: list) -> None:
    from matplotlib.ticker import FuncFormatter

    revisions = labels[1:]
    transitions = [
        stats["paired_transitions"][f"{label} - {labels[0]}"] for label in revisions
    ]
    y_positions = np.arange(len(revisions), dtype=float)
    fixed_rates = np.array([row["fixed"] for row in transitions])
    regressed_rates = np.array([row["regressed"] for row in transitions])
    ax.barh(
        y_positions,
        -fixed_rates,
        height=0.48,
        color=theme.PASTEL_GREEN,
        edgecolor=theme.DARK_GREEN,
        linewidth=1.0,
        zorder=3,
    )
    ax.barh(
        y_positions,
        regressed_rates,
        height=0.48,
        color=theme.PASTEL_CORAL,
        edgecolor=theme.DARK_CORAL,
        linewidth=1.0,
        zorder=3,
    )
    for y, fixed_rate, regressed_rate in zip(y_positions, fixed_rates, regressed_rates, strict=True):
        ax.text(
            -fixed_rate / 2,
            y,
            f"{fixed_rate:.1%}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=theme.LABEL_COLOR,
        )
        ax.text(
            regressed_rate + 0.008,
            y,
            f"{regressed_rate:.1%}",
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=theme.LABEL_COLOR,
        )
    ax.axvline(0, color=theme.SPINE_COLOR, linewidth=1.0, zorder=2)
    limit = max(float(fixed_rates.max()), float(regressed_rates.max())) + 0.075
    ax.set_xlim(-limit, limit)
    ax.set_ylim(len(revisions) - 0.5, -0.82)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([DISPLAY_LABELS.get(label, label) for label in revisions])
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{abs(value):.0%}")
    )
    ax.grid(False)
    ax.grid(axis="x", color=theme.GRID_COLOR, linewidth=1.0, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_xlabel("Expected share per draw")
    ax.text(
        0.02,
        0.98,
        "Fixed\n(inconsistent → consistent)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=theme.DARK_GREEN,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.98,
        "Regressed\n(consistent → inconsistent)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color=theme.DARK_CORAL,
        fontweight="bold",
    )


def draw(
    stats: dict,
    out_path: Path,
    panel: str,
    dpi: int | None = None,
    allow_stack_mismatch: bool = False,
) -> None:
    """Draw one panel (``"a"`` or ``"b"``) from its golden to ``out_path``."""
    import matplotlib.pyplot as plt

    fig = build_figure(stats, panel=panel, allow_stack_mismatch=allow_stack_mismatch)
    theme.save_soft_figure(fig, out_path, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
