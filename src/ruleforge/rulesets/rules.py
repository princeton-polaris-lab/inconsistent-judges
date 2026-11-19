from __future__ import annotations

import itertools
import json
import logging
import re
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple

from ruleforge.core.errors import RuleForgeError
from ruleforge.core.files import process_jsonl_file

logger = logging.getLogger(__name__)


RULE_REFERENCE_TEXT_FIELDS: Tuple[str, ...] = ("rule",)

# "Rule 5", "Rules 3 and 4", "Rules 3, 4, and 5", "Rules 3 & 4", with an optional "Meta-" prefix. Possessive
# quantifiers keep it linear; (?!\.\d) rejects "Rule 12.4".
_NUMERIC_RULE_REFERENCE_RE = re.compile(
    r"(?<![\w-])([Mm]eta-)?[Rr]ules?\s+"
    r"(\d++(?!\.\d)(?:(?:\s*(?:,|and|&))++\s*\d++(?!\.\d))*+)"
)


# The implicated_ranges grammar. A reference list starts at "Rule" or "Rules", whitespace and an item.
_REFERENCE_START_RE = re.compile(r"(?<![\w-])([Mm]eta-)?[Rr]ules?\s+")
# A clean integer: digits not followed by a letter, by "." and a digit, or by "/".
_INTEGER_RE = re.compile(r"\d++(?![^\W\d_]|\.\d|/)")
# A range's last rule is clean, and not followed by a further dash and digit either.
_RANGE_END_RE = re.compile(r"\d++(?![^\W\d_]|\.\d|/|\s*[-\u2013\u2014]\s*\d)")
_RANGE_SEPARATOR_RE = re.compile(r"\s*[-\u2013\u2014]\s*|\s+(?:to|through|thru)\s+")
_LIST_SEPARATOR_RE = re.compile(r"\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+")
# Anything that looks like a range, well formed or not; only ever a reason to warn.
_RANGE_LOOK_RE = re.compile(r"(?<![\w-])(?:[Mm]eta-)?[Rr]ules?\s+\S+\s*(?:[-\u2013\u2014]|\s(?:to|through|thru)\s)",
                            re.IGNORECASE)


class RuleRangeError(ValueError, RuleForgeError):
    """A rule reference that cannot be read whole as rule numbers and ranges of them."""


def _normalize_rule_number(digits: str) -> str:
    return digits.lstrip("0") or "0"


def rule_number_and_kind(rule_id: str) -> Tuple[Optional[str], bool]:
    """Return ``(the id's digits without leading zeros, or None; whether it is a meta-rule)``."""
    digits = re.sub(r"\D+", "", rule_id)
    number = _normalize_rule_number(digits) if digits else None
    return number, is_meta_rule_id(rule_id)


def rule_identity(rule_id: str) -> Tuple[bool, str]:
    """Return the rule an id names as ``(is_meta, number)``, or ``(False, id)`` for an id with no digits."""
    number, is_meta = rule_number_and_kind(rule_id)
    return (is_meta, number) if number is not None else (False, rule_id)


def duplicate_rule_id_refusal(rule_ids: Iterable[Any], holder: str) -> Optional[str]:
    """Return why ``holder`` cannot be scored rule by rule (two ids naming one rule), or None."""
    first: Dict[Tuple[bool, str], str] = {}
    clashes = []
    for raw_id in rule_ids:
        rule_id = str(raw_id).strip()
        identity = rule_identity(rule_id)
        if identity in first:
            clashes.append(f"{rule_id!r} names the rule {first[identity]!r} names")
        else:
            first[identity] = rule_id
    if not clashes:
        return None
    return (f"{holder}: {'; '.join(clashes)}. A rule's judgments, its preemptions and the intent penalty's deletion "
            "check are keyed by the rule its id names, so each rule needs an id of its own")


def is_meta_rule_id(rule_id: Any) -> bool:
    """Return whether a rule id names a meta-rule."""
    return "meta" in str(rule_id).lower()


def extract_rule_references(text: Any) -> List[str]:
    """Return the ``Rule N`` / ``Meta-Rule N`` labels a rule's text references, deduplicated, in order; ranges are not
    expanded."""
    if not isinstance(text, str):
        return []

    references: List[str] = []
    seen: set[str] = set()
    for match in _NUMERIC_RULE_REFERENCE_RE.finditer(text):
        prefix = "Meta-Rule" if match.group(1) else "Rule"
        for number in re.findall(r"\d+", match.group(2)):
            label = f"{prefix} {_normalize_rule_number(number)}"
            if label not in seen:
                seen.add(label)
                references.append(label)
    return references


def looks_like_rule_range(text: Any) -> bool:
    """Return whether a text holds something like a rule range ("Rules 3-5", "Rules 3 through five"), well formed or
    not."""
    return isinstance(text, str) and bool(_RANGE_LOOK_RE.search(text))


def _quoted(text: str, start: int, position: int) -> str:
    return text[start:re.compile(r"\S*").match(text, position).end()]


def read_rule_references(text: Any, rules: Collection[int], meta_rules: Collection[int] = ()
                         ) -> List[Tuple[str, int, int]]:
    """Return each item of each reference list in a text as ``("Rule" or "Meta-Rule", first, last)``, by the
    implicated_ranges grammar over ``rules`` (``meta_rules`` for "Meta-Rules"); a range ending in no rule number at
    least its first is refused, and a list that starts with no item is read as ``extract_rule_references`` reads it."""
    if not isinstance(text, str):
        return []
    found: List[Tuple[str, int, int]] = []
    for start in _REFERENCE_START_RE.finditer(text):
        prefix, numbers = ("Meta-Rule", meta_rules) if start.group(1) else ("Rule", rules)
        position = start.end()
        first = _INTEGER_RE.match(text, position)
        if first is None or int(first.group()) not in numbers:
            old = _NUMERIC_RULE_REFERENCE_RE.match(text, start.start())
            found += [(prefix, int(number), int(number)) for number in re.findall(r"\d+", old.group(2))] if old else []
            continue
        while first is not None and int(first.group()) in numbers:
            low, position = int(first.group()), first.end()
            separator = _RANGE_SEPARATOR_RE.match(text, position)
            if separator is not None and text[separator.end():separator.end() + 1].isdigit():
                last = _RANGE_END_RE.match(text, separator.end())
                if last is None or int(last.group()) not in numbers or int(last.group()) < low:
                    raise RuleRangeError(f"{_quoted(text, start.start(), separator.end())!r} is no range of this "
                                         "ruleset's rules: end it with a rule number at least its first, as 'Rules 27-43'")
                found.append((prefix, low, int(last.group())))
                position = last.end()
            else:
                # "Rule 5 to protect users": a separator with no number after it is prose, and the list ends.
                found.append((prefix, low, low))
                if separator is not None:
                    break
            joined = _LIST_SEPARATOR_RE.match(text, position)
            if joined is None:
                break
            position = joined.end()
            first = _INTEGER_RE.match(text, position)
    return found


def build_rule_implication_map(
    rules: List[Dict[str, Any]],
    *,
    text_fields: Tuple[str, ...] = RULE_REFERENCE_TEXT_FIELDS,
    expand_ranges: bool = False,
) -> Dict[str, List[str]]:
    """Return ``{rule_id: [ids of the rules its text references]}`` by rule number, self-references dropped; a range
    reads as its first rule, or under ``expand_ranges`` as every rule in it (refused when the ruleset lacks an end)."""
    id_by_number: Dict[str, str] = {}
    meta_id_by_number: Dict[str, str] = {}
    for rule in rules:
        rule_id = str(rule.get("id", "")).strip()
        if not rule_id:
            continue
        number, is_meta = rule_number_and_kind(rule_id)
        if number is None:
            continue
        number_map = meta_id_by_number if is_meta else id_by_number
        if number in number_map:
            logger.warning(
                f"Duplicate rule number {number}: keeping "
                f"'{number_map[number]}', ignoring '{rule_id}'"
            )
            continue
        number_map[number] = rule_id

    implication_map: Dict[str, List[str]] = {}
    for rule in rules:
        rule_id = str(rule.get("id", "")).strip()
        if not rule_id:
            continue
        own_number, own_is_meta = rule_number_and_kind(rule_id)
        implicated: List[str] = []
        for field in text_fields:
            labels = (_expanded_labels(rule.get(field), id_by_number, meta_id_by_number) if expand_ranges
                      else extract_rule_references(rule.get(field)))
            for label in labels:
                prefix, _, number = label.rpartition(" ")
                is_meta_reference = prefix == "Meta-Rule"
                if is_meta_reference == own_is_meta and number == own_number:
                    continue
                lookup = (
                    meta_id_by_number if is_meta_reference else id_by_number
                )
                referenced_id = lookup.get(number)
                if referenced_id is None:
                    continue
                if referenced_id not in implicated:
                    implicated.append(referenced_id)
        implication_map[rule_id] = implicated
    return implication_map


def _expanded_labels(text: Any, id_by_number: Dict[str, str], meta_id_by_number: Dict[str, str]) -> List[str]:
    rules, meta_rules = ({int(number) for number in table} for table in (id_by_number, meta_id_by_number))
    return list(dict.fromkeys(f"{prefix} {number}" for prefix, first, last in read_rule_references(text, rules, meta_rules)
                              for number in range(first, last + 1)
                              if number in (meta_rules if prefix == "Meta-Rule" else rules)))


def build_implicated_rules_context(
    rule_id: str,
    implication_map: Dict[str, List[str]],
    rules_by_id: Dict[str, Dict[str, Any]],
) -> List[str] | None:
    """Return the context rules for ``rule_id`` (its references, then each meta-rule with its references), or None."""
    context_ids: List[str] = []

    def _add(candidate_id: str) -> None:
        if candidate_id != rule_id and candidate_id not in context_ids:
            context_ids.append(candidate_id)

    for implicated_id in implication_map.get(rule_id) or []:
        _add(implicated_id)
    for meta_id in rules_by_id:
        if not is_meta_rule_id(meta_id):
            continue
        _add(meta_id)
        for implicated_id in implication_map.get(meta_id) or []:
            _add(implicated_id)

    context_rules = [
        format_rule(rules_by_id[context_id])
        for context_id in context_ids
        if context_id in rules_by_id
    ]
    return context_rules or None


def build_rule_id_index(rules: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map every spelling of a rule id (the id, its number, ``Rule N``, ``Meta-Rule N``) to the id as written."""
    index: Dict[str, str] = {}

    # Literal ids first, so a rule's own id beats another rule's derived spelling.
    for rule in rules:
        raw_id = str(rule.get("id", ""))
        if not raw_id.strip():
            continue
        index.setdefault(raw_id.strip().lower(), raw_id)

    for rule in rules:
        raw_id = str(rule.get("id", ""))
        stripped = raw_id.strip()
        if not stripped:
            continue
        number, is_meta = rule_number_and_kind(stripped)
        if number is None:
            continue
        prefix = "meta-rule" if is_meta else "rule"
        for spelling in (number, f"{prefix} {number}", f"{prefix}{number}"):
            existing = index.setdefault(spelling.lower(), raw_id)
            if existing != raw_id and spelling == number:
                logger.warning(
                    f"Rule number {number} is ambiguous: keeping "
                    f"'{existing}', ignoring '{raw_id}'"
                )
    return index


def normalize_preempting_rule_id(
    raw_id: Any,
    rule_id_index: Dict[str, str],
) -> Optional[str]:
    """Resolve a judge-supplied rule id to the ruleset's own, or None."""
    if not isinstance(raw_id, str):
        return None
    candidate = raw_id.strip().lower()
    if not candidate:
        return None
    if candidate in rule_id_index:
        return rule_id_index[candidate]
    stripped = candidate.strip("().,:;\"'")
    return rule_id_index.get(stripped)


def format_rule(rule: Dict[str, Any], content_key: str = "rule", *, level: Optional[str] = None) -> str:
    """Render one rule as the judge sees it: ``Rule 7: ...``, ``Meta-Rule 2: ...`` or, with a level, ``Rule 43 (user):
    ...``."""
    rule_id = str(rule["id"]).strip()
    number, is_meta = rule_number_and_kind(rule_id)
    label = rule_id if number is None else f"{'Meta-Rule' if is_meta else 'Rule'} {number}"
    if level is not None:
        label = f"{label} ({level})"
    return f"{label}: {rule[content_key]}"


def build_rule_set_string_from_file(rule_set_file: Path) -> str:
    """Render every rule of a JSONL file, one per line."""
    rules = list(process_jsonl_file(rule_set_file))
    return "\n".join(format_rule(rule) for rule in rules)


def _split_rule_headers(text: str) -> str:
    return re.sub(r"(?<=\S)\s+(?=(?:Rule|Meta-Rule)\s*\d+\s*:)", "\n", text)


def parse_rule_set_text(text: str) -> List[Tuple[str, str]]:
    """Return the ``(id, text)`` of each rule in a ruleset's text, a rule starting at each ``Rule <n>:`` header; text
    before the first is a rule ``line_<k>`` of its own."""
    return _parse_rule_lines(_split_rule_headers(text).splitlines())


def _parse_rule_lines(lines: Iterable[str]) -> List[Tuple[str, str]]:
    rule_start_pattern = re.compile(r"^((?:Rule|Meta-Rule)\s*\d+)\s*:")
    rules = []
    current_id = None
    current_lines = []
    preamble_start_line = 1

    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        match = rule_start_pattern.match(line)
        if match:
            if current_id is not None:
                rules.append((current_id, "\n".join(current_lines)))
            elif current_lines:
                rules.append((f"line_{preamble_start_line}", "\n".join(current_lines)))

            current_id = match.group(1).strip()
            rest = line[match.end() :].strip()
            current_lines = [rest] if rest else []
        else:
            if current_id is None and not current_lines:
                preamble_start_line = idx
            current_lines.append(line)

    if current_id is not None:
        rules.append((current_id, "\n".join(current_lines)))
    elif current_lines:
        rules.append((f"line_{preamble_start_line}", "\n".join(current_lines)))
    return [(rule_id, description.strip()) for rule_id, description in rules]


def round_trip_refusal(rules: Iterable[Dict[str, Any]], holder: str, *, exact_ids: bool) -> Optional[str]:
    """Return why ``holder``'s rules would not survive a render-and-parse round trip, or None. Under ``exact_ids`` each
    id must come back as written; otherwise it must name the same rule."""
    rules = list(rules)
    given = [(str(rule["id"]).strip(), rule["rule"]) for rule in rules]
    kept = parse_rule_set_text("\n".join(format_rule(rule) for rule in rules))

    def same(before: Optional[Tuple[str, Any]], after: Optional[Tuple[str, str]]) -> bool:
        if before is None or after is None or before[1] != after[1]:
            return False
        return before[0] == after[0] if exact_ids else rule_identity(before[0]) == rule_identity(after[0])

    for index, pair in enumerate(itertools.zip_longest(given, kept)):
        if not same(*pair):
            before, after = pair
            return (f"{holder}: the optimizer writes a ruleset as 'Rule <n>: <text>' and 'Meta-Rule <n>: <text>' "
                    f"lines and reads it back from them, and its rule {index} {before!r} would come back as "
                    f"{after!r}: give each rule such an id and a text holding no rule header, blank line or "
                    "surrounding whitespace")
    return None


def rule_text_refusal(text: str, holder: str) -> Optional[str]:
    """Return why ``holder``'s text is not a ruleset the optimizer can keep as written, or None: it must start at a rule
    header, and a header must start its line, the checks a seed ruleset gets from ``round_trip_refusal``."""
    rules = _parse_rule_lines(text.splitlines())
    if rules and rules[0][0].startswith("line_"):
        return f"{holder} has text before its first rule header: {rules[0][1]!r}"
    return round_trip_refusal(({"id": rule_id, "rule": rule} for rule_id, rule in rules), holder, exact_ids=False)


def save_rule_set_as_jsonl(
    text: str,
    file_path: Path,
    content_key: str = "rule",
    *,
    on_missing_colon: Optional[str] = "keep",
):
    """Write ``<id>: <text>`` lines as a ruleset JSONL file. ``on_missing_colon="keep"`` joins continuation lines to the
    rule before them; ``"skip"`` drops them."""
    if on_missing_colon == "keep":
        count = 0
        with file_path.open("w", encoding="utf-8") as file:
            for rule_id, description in parse_rule_set_text(text):
                file.write(
                    json.dumps(
                        {"id": rule_id, content_key: description},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
    else:
        count = 0
        with file_path.open("w", encoding="utf-8") as file:
            for raw_line in _split_rule_headers(text).splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split(":", 1)
                if len(parts) == 2:
                    rule_id = parts[0].strip()
                    description = parts[1].strip()
                    file.write(
                        json.dumps(
                            {"id": rule_id, content_key: description},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    count += 1

    print(f"Saved {file_path} ({count} items).")
