from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from ruleforge.core.errors import RuleForgeError
from ruleforge.core.artifacts import iter_jsonl_records
from ruleforge.core.digests import file_record

__all__ = [
    "LEVEL_FIELD",
    "LEVEL_ORDER",
    "LEVEL_SOURCE",
    "RuleLevels",
    "RuleLevelsError",
    "load_rule_levels",
    "require_known_levels",
    "superior_ids",
]

# Highest authority first.
LEVEL_ORDER = ("root", "system", "developer", "user", "guideline")
LEVEL_SOURCE = (
    "the OpenAI Model Spec's chain of command (its levels of authority, highest first: root, system, developer, "
    "user, guideline); each rule's level is its rule_type in data/rules/modelspec/openai_rules_all_full.jsonl"
)
LEVEL_FIELD = "rule_type"


class RuleLevelsError(ValueError, RuleForgeError):
    """Raised for a levels file that does not align with its ruleset, or a level outside ``LEVEL_ORDER``."""


@dataclass(frozen=True)
class RuleLevels:
    """The ruleset's rule ids in file order, each rule's level, and both files' records (name, SHA-256)."""

    rule_ids: Tuple[str, ...]
    levels: Tuple[str, ...]
    ruleset: Dict[str, str]
    levels_file: Dict[str, str]

    @property
    def rank(self) -> Dict[str, int]:
        """Return ``{rule_id: index of its level in LEVEL_ORDER}`` (0 is root, the highest)."""
        return {rule_id: LEVEL_ORDER.index(level) for rule_id, level in zip(self.rule_ids, self.levels)}

    @property
    def by_id(self) -> Dict[str, str]:
        """Return ``{rule_id: level}``."""
        return dict(zip(self.rule_ids, self.levels))

    def rules_by_level(self) -> Dict[str, List[str]]:
        return {level: [rule_id for rule_id, own in zip(self.rule_ids, self.levels) if own == level]
                for level in LEVEL_ORDER}


def require_known_levels(rule_ids: Iterable[str], levels: Mapping[str, str]) -> None:
    """Raise unless ``levels`` gives every rule of ``rule_ids`` a level of ``LEVEL_ORDER``."""
    unknown = sorted({rule_id for rule_id in rule_ids if levels.get(rule_id) not in LEVEL_ORDER})
    if unknown:
        raise RuleLevelsError(f"no level, or one not in {list(LEVEL_ORDER)}, for rule(s) {unknown}")


def superior_ids(rule_id: str, levels: Mapping[str, str], order: Sequence[str]) -> List[str]:
    """Return the rules of ``order`` at a strictly higher level than ``rule_id``, in file order."""
    require_known_levels((rule_id, *order), levels)
    own = LEVEL_ORDER.index(levels[rule_id])
    return [other for other in order if LEVEL_ORDER.index(levels[other]) < own]


def load_rule_levels(rules_path: Path | str, levels_path: Path | str) -> RuleLevels:
    """Load each rule's level from ``levels_path``, aligned with ``rules_path`` by position and rule text."""
    rules_path, levels_path = Path(rules_path), Path(levels_path)
    where = f"levels file {levels_path.name} against ruleset {rules_path.name}"
    rules, levelled = list(iter_jsonl_records(rules_path)), list(iter_jsonl_records(levels_path))
    ids = [rule["id"].strip() if isinstance(rule.get("id"), str) else "" for rule in rules]
    _require(
        [
            (bool(rules), "the ruleset holds no rule"),
            (len(rules) == len(levelled), f"{len(rules)} rules but {len(levelled)} levelled rules"),
            (all(ids) and len(set(ids)) == len(ids), "a ruleset row has no id, or two share one"),
        ],
        where,
    )
    differ = [f"{k + 1} ({ids[k]})" for k, (rule, other) in enumerate(zip(rules, levelled))
              if not (isinstance(rule.get("rule"), str) and rule.get("rule") == other.get("rule"))]
    unknown = [f"{k + 1} ({ids[k]}: {other.get(LEVEL_FIELD)!r})" for k, other in enumerate(levelled)
               if other.get(LEVEL_FIELD) not in LEVEL_ORDER]
    _require(
        [
            (not differ, f"the rule text differs at position(s) {differ}: the files do not align one to one"),
            (not unknown, f"no level, or one not in {list(LEVEL_ORDER)}, at position(s) {unknown}"),
        ],
        where,
    )
    return RuleLevels(tuple(ids), tuple(str(other[LEVEL_FIELD]) for other in levelled),
                      file_record(rules_path), file_record(levels_path))


def _require(checks: Sequence[Tuple[bool, str]], where: str) -> None:
    failures = [message for ok, message in checks if not ok]
    if failures:
        raise RuleLevelsError(f"{where}: " + "; ".join(failures))
