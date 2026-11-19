from __future__ import annotations

import re
from typing import Any, Optional

LABELS = ("yes", "no", "unsure")
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# Each thinking style's opening and closing marks: Qwen's and Olmo's tags, and Gemma's thought channel.
THOUGHT_MARKS = ((THINK_OPEN, THINK_CLOSE), ("<|channel>thought", "<channel|>"))


def holds_an_open_thought(text: str) -> bool:
    """Return whether ``text`` opens a thought, in any thinking style, that no closing mark follows: such an output has
    no answer, and the attempt is asked again."""
    return any(text.rfind(opening) > text.rfind(closing) for opening, closing in THOUGHT_MARKS)


def _bare(value: str) -> Optional[str]:
    normalized = value.strip().lower()
    return normalized if normalized in LABELS else None


# "Final Judgment: no", "**Final Judgement**: Unsure.", "Judgment - yes"
_TRAILER = re.compile(
    r"^[^a-z0-9]*(?:final[^a-z0-9]+)?judge?ment[^a-z0-9]*?[:\-][^a-z0-9]*(yes|no|unsure)[^a-z0-9]*$",
    re.IGNORECASE,
)


def _trailer_or_first_line(text: str) -> Optional[str]:
    # Step 1: the last trailer line wins; otherwise the first nonblank line must be a bare label.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = _TRAILER.match(line)
        if match:
            return match.group(1).lower()
    if lines:
        return _bare(lines[0].rstrip(".!:").strip())
    return None


# A chat template's control token leaked into an answer (<|...|>, <...|>, <|...>, </s>, <eos> ...); never a bare <word>.
_CONTROL_TOKEN = re.compile(
    r"<\|[^\s<>|]{1,48}\|?>|<[^\s<>|]{1,48}\|>|</s>|<(?:eos|bos|pad|unk|start_of_turn|end_of_turn)>",
    re.IGNORECASE,
)
_THEMATIC_BREAK = re.compile(r"^(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$")
_FENCE = re.compile(r"^`{3,}([A-Za-z0-9_+.-]*)$")
# Not "~": a strikethrough negates the label it wraps.
_WRAPPERS = "\"'`*_“”‘’"
_LABEL_WORD = re.compile(r"(?<![a-z0-9])(yes|no|unsure)(?![a-z0-9])", re.IGNORECASE)


def _token_break(token: re.Match[str]) -> str:
    return token.group() if _LABEL_WORD.search(token.group()) else "\n"


def _markup_line(line: str) -> bool:
    fence = _FENCE.match(line)
    return bool(_THEMATIC_BREAK.match(line)) or (fence is not None and _LABEL_WORD.search(fence.group(1)) is None)


def _unwrap(line: str) -> str:
    while True:
        stripped = line.strip().strip(_WRAPPERS).rstrip(".!:")
        if stripped == line:
            return line
        line = stripped


def _markup_lines(text: str) -> list[str]:
    lines = []
    for raw in _CONTROL_TOKEN.sub(_token_break, text).splitlines():
        line = raw.strip()
        if line and not _markup_line(line):
            line = _unwrap(line)
            if line:
                lines.append(line)
    return lines


_VERDICT_WORD = re.compile(r"judge?ment|verdict|answer", re.IGNORECASE)
_ALNUM = re.compile(r"[a-z0-9]", re.IGNORECASE)


def _label_words(line: str, start: int = 0) -> set[str]:
    return {match.group(1).lower() for match in _LABEL_WORD.finditer(line, start)}


def _verdict_labels(line: str) -> set[str]:
    verdict = _VERDICT_WORD.search(line)
    return set() if verdict is None else _label_words(line, verdict.end())


def _opening_label(line: str) -> Optional[str]:
    match = _LABEL_WORD.search(line)
    return match.group(1).lower() if match and not _ALNUM.search(line, 0, match.start()) else None


def _through_markup(text: str) -> Optional[str]:
    # Step 2: step 1 again with markup read through; a reading added here is refused when any competing label word
    # differs (every word of every line for a first-line reading; the judgment words, the opening label and the lines
    # after the last trailer for a trailer reading).
    label = _trailer_or_first_line(text)
    if label is not None:
        return label
    lines = _markup_lines(text)
    label = _trailer_or_first_line("\n".join(lines))
    if label is None:
        return None
    trailers = [index for index, line in enumerate(lines) if _TRAILER.match(line)]
    if trailers:
        stated = ({_opening_label(lines[0])} - {None}).union(
            *map(_verdict_labels, lines), *map(_label_words, lines[trailers[-1] + 1:])
        )
    else:
        stated = set().union(*map(_label_words, lines))
    return label if stated <= {label} else None


# Step 3's statements of the rule set's judgment: "final", "overall" or "rule-set" must qualify the noun, since a bare
# "Answer:" may be one rule's. A gap is anything but a letter, a digit or "~" (a strikethrough negates what it wraps).
_GAP = r"[^a-z0-9~]"
_NOUN = r"(?:answer|judge?ment|verdict)"
_RULE_SET = rf"rule{_GAP}?set"
_SET_VERDICT = (
    rf"(?:(?:final|overall)(?:{_GAP}+{_RULE_SET})?|{_RULE_SET}){_GAP}+{_NOUN}"
    rf"(?:{_GAP}+(?:on|for){_GAP}+the{_GAP}+(?:(?:full|whole|entire){_GAP}+)?(?:{_RULE_SET}|set))?"
)
_LABEL_GROUP = r"(yes|no|unsure)"
# "Final answer: no", "Overall judgment -- unsure" and nothing more.
_VERDICT_LINE = re.compile(
    rf"^{_GAP}*{_SET_VERDICT}{_GAP}*?[:=\-\u2013\u2014]{_GAP}*{_LABEL_GROUP}{_GAP}*$", re.IGNORECASE
)
# "The answer is no" and nothing more: an earlier line's judgment only, never a last line's reading.
_ANSWER_IS_LINE = re.compile(
    rf"^{_GAP}*(?:the|my){_GAP}+(?:(?:final|overall){_GAP}+)?answer{_GAP}+is{_GAP}+{_LABEL_GROUP}{_GAP}*$",
    re.IGNORECASE,
)
_STATED_TRAILER = re.compile(_TRAILER.pattern.replace("[^a-z0-9]", _GAP), re.IGNORECASE)


def _line_verdict(line: str) -> Optional[str]:
    statement = _VERDICT_LINE.match(line)
    return statement.group(1).lower() if statement else _bare(line)


def _stated_verdict(line: str) -> Optional[str]:
    label = _line_verdict(line)
    if label is not None:
        return label
    statement = _STATED_TRAILER.match(line) or _ANSWER_IS_LINE.match(line)
    return statement.group(1).lower() if statement else None


def read_label(text: Any) -> Optional[str]:
    """Return the judgment ('yes', 'no' or 'unsure') of a plain-label answer, or None if there is no single clear
    one."""
    if not isinstance(text, str) or holds_an_open_thought(text):
        return None
    label = _through_markup(text)
    if label is not None:
        return label
    lines = _markup_lines(text)
    if not lines:
        return None
    # Step 3: the last line, unless an earlier whole line states another judgment.
    label = _line_verdict(lines[-1])
    if label is None or any(_stated_verdict(line) not in (None, label) for line in lines[:-1]):
        return None
    return label
