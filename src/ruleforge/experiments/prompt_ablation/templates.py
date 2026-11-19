from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ruleforge.core.errors import RuleForgeError
from ruleforge.experiments.driver import output_dir
from ruleforge.experiments.paper_outputs import paper_output_refusal
from ruleforge.experiments.prompt_ablation.analysis import SLOT_STYLES
from ruleforge.core.paths import repository_root
from ruleforge.tasks.spec import ModelSpecTaskSpec
from ruleforge.prompts.styles import (
    ENCOURAGED,
    FORCED,
    FORCED_GRPO,
    HIERARCHY_STYLES,
    NAIVE,
    NAIVE_RULES_ABOVE,
    VALID_REASONING_PROMPT_STYLES,
)

#: The templates ``--template`` names, by file name, each its style's own name.
TEMPLATES = {style: style for style in (NAIVE, NAIVE_RULES_ABOVE, ENCOURAGED, FORCED, FORCED_GRPO)}
PLACEHOLDER_RULESET = "<ruleset>"
PLACEHOLDER_SCENARIO = {
    "prompt": "<original_prompt>",
    "response": "<ai_response>",
}


class TemplateError(ValueError, RuleForgeError):
    """A template that cannot be rendered or written as asked."""


class _TemplateSpec:
    def __init__(self, style: str) -> None:
        self.reasoning_prompt_style = style


def render_template(style: str) -> str:
    """Render one exact prompt skeleton with semantic placeholders."""
    if style not in VALID_REASONING_PROMPT_STYLES:
        raise TemplateError(f"unknown prompt style: {style}")
    if style in HIERARCHY_STYLES:
        # This prompt tags every rule with its level, which a placeholder ruleset cannot; see
        # tests/fixtures/hierarchy_prompts/.
        raise TemplateError(f"prompt style {style!r} renders only over a levelled ruleset, "
                         "not a placeholder")
    return ModelSpecTaskSpec.build_eval_prompt(
        _TemplateSpec(style),
        PLACEHOLDER_RULESET,
        PLACEHOLDER_SCENARIO,
        "No_Interpretation",
    )


def paper_styles() -> tuple[tuple[str, str], ...]:
    """Return ``(paper name, style)`` of the paper's files, in paper order."""
    return tuple((style, style) for style in (*SLOT_STYLES.values(), FORCED_GRPO))


def selected(names: Sequence[str] | None = None) -> tuple[tuple[str, str], ...]:
    """Return ``(name, style)`` for each template named, or the paper's files when none is."""
    if names is None:
        return paper_styles()
    unknown = [name for name in names if name not in TEMPLATES]
    if unknown or not names or len(set(names)) != len(names):
        raise TemplateError(f"templates {list(names)}: name each of {sorted(TEMPLATES)} at most once")
    return tuple((name, TEMPLATES[name]) for name in names)


def render_templates(names: Sequence[str] | None = None) -> dict[str, str]:
    """Return name -> prompt template: the ``names`` given, or the paper's."""
    return {paper_name: render_template(style) for paper_name, style in selected(names)}


def write_templates(output_dir: Path, names: Sequence[str] | None = None) -> list[Path]:
    """Write the templates and return their paths in order; a stray ``.txt`` in ``output_dir``, or anything but the
    paper's set under the paper's directories, is refused."""
    templates = render_templates(names)
    paper = render_templates()
    refusal = paper_output_refusal([output_dir], templates == paper, repo_root=repository_root())
    if refusal is not None:
        raise TemplateError(refusal)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{paper_name}.txt" for paper_name in templates}
    unexpected_names = sorted(
        path.name
        for path in output_dir.glob("*.txt")
        if path.name not in expected_names
    )
    if unexpected_names:
        listed = ", ".join(unexpected_names)
        raise TemplateError(f"unexpected prompt template file(s): {listed}")

    paths = []
    for paper_name, template in templates.items():
        path = output_dir / f"{paper_name}.txt"
        path.write_text(template, encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the prompt styles as placeholder-filled text files.")
    parser.add_argument("--output-dir", type=output_dir, required=True, help="directory for the templates")
    parser.add_argument(
        "--template",
        action="append",
        choices=sorted(TEMPLATES),
        help="a template to write, repeatable (default: the paper's files)",
    )
    args = parser.parse_args(argv)
    try:
        paths = write_templates(args.output_dir, args.template)
    except (OSError, TemplateError) as error:
        parser.error(str(error))
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
