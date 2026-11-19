from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from ruleforge.core.errors import RuleForgeError
from ruleforge.prompts.styles import HIERARCHY_RULE_CONTEXT
from ruleforge.rulesets.levels import RuleLevels, load_rule_levels
from ruleforge.rulesets.rules import is_meta_rule_id
from ruleforge.rulesets.subsets import generate_subset_files, seed_dir
from ruleforge.rulesets.variants import build_rule_variants, rule_set_text
from ruleforge.core.artifacts import load_jsonl_hashed

Levels = Union[Mapping[str, str], RuleLevels, None]


class RulesetError(ValueError, RuleForgeError):
    """Raised for a ruleset file that cannot be read as one, or is not the one expected."""


@dataclass(frozen=True)
class Rule:
    """One rule: its id, its text (``rule``) and every field of its row."""

    id: str
    text: str
    fields: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_meta(self) -> bool:
        """Return whether this is a meta-rule, which is never judged alone."""
        return is_meta_rule_id(self.id)

    def row(self) -> dict[str, Any]:
        return dict(self.fields)


def _levels(levels: Levels) -> Optional[Mapping[str, str]]:
    return levels.by_id if isinstance(levels, RuleLevels) else levels


@dataclass(frozen=True)
class Ruleset:
    """The rules of one ruleset file, in file order, with the file's identity."""

    rules: tuple[Rule, ...]
    name: str
    sha256: str
    path: Optional[Path] = None

    @classmethod
    def load(cls, path: Path | str, *, name: Optional[str] = None, expect_sha256: Optional[str] = None) -> "Ruleset":
        """Load the ruleset at ``path``, named by its basename unless ``name`` pins another."""
        source = Path(path)
        try:
            rows, digest = load_jsonl_hashed(source)
        except (OSError, ValueError) as error:
            raise RulesetError(f"cannot read ruleset {source}: {error}") from error
        if expect_sha256 is not None and digest != expect_sha256:
            raise RulesetError(f"{source.name} has SHA-256 {digest}, not the expected {expect_sha256}")
        rules = []
        for index, row in enumerate(rows):
            if not (isinstance(row.get("id"), str) and isinstance(row.get("rule"), str)):
                raise RulesetError(f"{source.name} row {index} is not a rule (an object with string id and rule)")
            rules.append(Rule(id=row["id"].strip(), text=row["rule"], fields=dict(row)))
        if not rules:
            raise RulesetError(f"{source.name} holds no rules")
        return cls(rules=tuple(rules), name=name or source.name, sha256=digest, path=source)

    def rows(self) -> list[dict[str, Any]]:
        """Return every rule's row, as the file holds it."""
        return [rule.row() for rule in self.rules]

    def non_meta_ids(self) -> tuple[str, ...]:
        """Return the ids of every rule but the meta-rules, in file order."""
        return tuple(rule.id for rule in self.rules if not rule.is_meta)

    def text(self, levels: Levels = None) -> str:
        """Return the full-set prompt's rule text; with ``levels``, each rule's level tags its line."""
        return rule_set_text(self.rows(), HIERARCHY_RULE_CONTEXT if levels is not None else "isolated", _levels(levels))

    def variants(self, rule_context: str, levels: Levels = None) -> tuple[list[tuple[str, str, Any]], Any]:
        """Return each non-meta rule's ``(id, text, context rules)`` under ``rule_context``, and the implication map."""
        return build_rule_variants(self.rows(), rule_context, _levels(levels))

    def with_levels(self, path: Path | str) -> RuleLevels:
        """Return the levels file at ``path``, aligned with this ruleset's file row by row."""
        return load_rule_levels(self._file(), path)

    def subsets(self, *, seeds: Sequence[int], strategy: str, out: Path | str) -> dict[int, list[Path]]:
        """Write the ruleset-size subset files per seed under ``<out>/<strategy>/seed_<s>`` and return them."""
        return {seed: generate_subset_files(self._file(), seed_dir(Path(out), strategy, seed), strategy, seed)
                for seed in seeds}

    def _file(self) -> Path:
        if self.path is None:
            raise RulesetError(f"ruleset {self.name} has no file")
        return self.path


__all__ = ["Levels", "Rule", "Ruleset", "RulesetError"]
