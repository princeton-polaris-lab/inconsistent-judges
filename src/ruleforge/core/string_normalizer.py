import functools
import re
import sys
import unicodedata
from typing import List, Optional, Tuple

_ZERO_WIDTHS = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\u180e",  # MONGOLIAN VOWEL SEPARATOR
}

_SOFT_HYPHEN = "\u00ad"

_DASH_MAP = {
    ord("\u2010"): "-",  # hyphen
    ord("\u2011"): "-",  # non-breaking hyphen
    ord("\u2012"): "-",  # figure dash
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
    ord("\u2015"): "-",  # horizontal bar
    ord("\u2212"): "-",  # minus sign
    ord("\ufe58"): "-",  # small em dash
    ord("\u2e3a"): "-",  # two-em dash
    ord("\u2e3b"): "-",  # three-em dash
}

_QUOTE_MAP = {
    ord("\u2018"): "'",  # left single
    ord("\u2019"): "'",  # right single
    ord("\u201b"): "'",  # single high-reversed-9
    ord("\u2032"): "'",  # prime
    ord("\u2035"): "'",  # reversed prime
    ord("\u201c"): '"',  # left double
    ord("\u201d"): '"',  # right double
    ord("\u2033"): '"',  # double prime
    ord("\u2036"): '"',  # reversed double prime
}

_SPACE_TO_NORMAL_SPACE = {
    ord("\u00a0"): " ",  # no-break space
    ord("\u2000"): " ",  # en quad
    ord("\u2001"): " ",  # em quad
    ord("\u2002"): " ",  # en space
    ord("\u2003"): " ",  # em space
    ord("\u2004"): " ",  # three-per-em
    ord("\u2005"): " ",  # four-per-em
    ord("\u2006"): " ",  # six-per-em
    ord("\u2007"): " ",  # figure space
    ord("\u2008"): " ",  # punctuation space
    ord("\u2009"): " ",  # thin space
    ord("\u200a"): " ",  # hair space
    ord("\u202f"): " ",  # narrow no-break space
    ord("\u205f"): " ",  # medium mathematical space
    ord("\u3000"): " ",  # ideographic space
}

# Punctuation whose surrounding spacing is canonicalized: none before, one after.
_PUNCT_CANON = r":;,!?%"

_TRANS_MAP = {**_DASH_MAP, **_QUOTE_MAP, **_SPACE_TO_NORMAL_SPACE}
_PUNCT_SET = set(_PUNCT_CANON)


def _ascii_digit(ch: str) -> str:
    if ch.isdigit() and not ("0" <= ch <= "9"):
        try:
            return str(unicodedata.digit(ch))
        except Exception:
            try:
                val = unicodedata.numeric(ch)
                return str(int(val))
            except Exception:
                return ch
    return ch


def _is_removed_control(ch: str) -> bool:
    return unicodedata.category(ch) in {"Cf", "Cc"} and ch not in {"\n", "\t"}


_ASCII_REMOVED_CONTROLS = bytes(code_point for code_point in range(128) if _is_removed_control(chr(code_point)))

@functools.cache
def _char_tables() -> Tuple[dict, dict]:
    removed: dict = {}
    digits: dict = {}
    for code_point in range(sys.maxunicode + 1):
        ch = chr(code_point)
        if _is_removed_control(ch):
            removed[code_point] = None
        mapped = _ascii_digit(ch)
        if mapped != ch:
            digits[code_point] = mapped
    return removed, digits


def _nfkc_pairs(s: str) -> List[Tuple[str, int]]:
    """NFKC of ``s`` as (character, index) pairs: the index is the base of the combining sequence in ``s`` the character
    came from, the first one's for a character composed from several."""
    bases: List[int] = []
    for i, ch in enumerate(s):
        bases.append(bases[-1] if i and unicodedata.combining(ch) else i)
    # Compatibility decomposition, one character at a time; then canonical order: each run of marks sorted, stably, by
    # combining class.
    decomposed = [(piece, i) for i, ch in enumerate(s) for piece in unicodedata.normalize("NFKD", ch)]
    ordered: List[Tuple[str, int]] = []
    marks: List[Tuple[str, int]] = []
    for pair in decomposed:
        if unicodedata.combining(pair[0]):
            marks.append(pair)
            continue
        ordered += sorted(marks, key=lambda mark: unicodedata.combining(mark[0]))
        marks = []
        ordered.append(pair)
    ordered += sorted(marks, key=lambda mark: unicodedata.combining(mark[0]))
    # Canonical composition: a character joins the last starter unless a character between blocks it.
    composed: List[Tuple[str, int]] = []
    starter, last_class = None, 0
    for ch, i in ordered:
        combining = unicodedata.combining(ch)
        if starter is not None and (starter == len(composed) - 1 or 0 < last_class < combining):
            joined = unicodedata.normalize("NFC", composed[starter][0] + ch)
            if len(joined) == 1:
                composed[starter] = (joined, composed[starter][1])
                continue
        if combining:
            last_class = combining
        else:
            starter, last_class = len(composed), 0
        composed.append((ch, i))
    return [(ch, bases[i]) for ch, i in composed]


def source_end(s: str, start: int, end: int) -> int:
    """Return where the source in ``s`` of a normalized character that begins at ``start`` ends, at ``end`` or later:
    past its combining marks and every character NFKC composes into it."""
    while end < len(s) and (unicodedata.combining(s[end]) or unicodedata.normalize("NFKC", s[start:end + 1]) != (
            unicodedata.normalize("NFKC", s[start:end]) + unicodedata.normalize("NFKC", s[end]))):
        end += 1
    return end


def normalize_for_match(
    s: Optional[str],
    *,
    casefold: bool = True,
    collapse_inline_whitespace: bool = True,
    preserve_newlines: bool = True,
) -> str:
    """Return ``s`` normalized for matching, never to be written back; ``normalize_with_offset_map`` mirrors its
    numbered steps."""
    if not s:
        return ""

    # Step 1: NFKC
    s = unicodedata.normalize("NFKC", s)

    # Step 2: zero-width characters and soft hyphens
    for ch in _ZERO_WIDTHS:
        s = s.replace(ch, "")
    s = s.replace(_SOFT_HYPHEN, "")

    # Steps 3-5: dashes, quotes, spaces (every key is non-ASCII)
    if not s.isascii():
        s = s.translate(_TRANS_MAP)

    # Step 6: line endings, trailing spaces, control characters
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip(" \t") for line in s.split("\n"))

    removed_controls, ascii_digits = _char_tables()
    if s.isascii():
        s = s.encode("ascii").translate(None, _ASCII_REMOVED_CONTROLS).decode("ascii")
    else:
        s = s.translate(removed_controls)

    # Step 7: punctuation spacing
    s = re.sub(rf"\s+([{_PUNCT_CANON}])", r"\1", s)
    s = re.sub(rf"([{_PUNCT_CANON}])\s*", r"\1 ", s)

    # Step 8: whitespace runs
    if preserve_newlines:
        if collapse_inline_whitespace:
            s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        s = s.strip("\n")
    else:
        s = re.sub(r"\s+", " ", s).strip()

    # Step 9: digits
    if not s.isascii():
        s = s.translate(ascii_digits)

    # Step 10: case
    if casefold:
        s = s.casefold()

    return s


def normalize_with_offset_map(
    s: Optional[str],
    *,
    casefold: bool = True,
    collapse_inline_whitespace: bool = True,
    preserve_newlines: bool = True,
) -> Tuple[str, List[int]]:
    """Return ``normalize_for_match``'s output with an offset map: ``offset_map[i]`` is normalized character ``i``'s
    index in ``s``."""
    if not s:
        return "", []

    # Step 1: NFKC
    pairs = _nfkc_pairs(s)

    # Step 2: zero-width characters and soft hyphens
    pairs = [
        (ch, idx)
        for ch, idx in pairs
        if ch not in _ZERO_WIDTHS and ch != _SOFT_HYPHEN
    ]

    # Steps 3-5: dashes, quotes, spaces
    new_pairs = []
    for ch, idx in pairs:
        code = ord(ch)
        new_pairs.append((_TRANS_MAP[code], idx) if code in _TRANS_MAP else (ch, idx))
    pairs = new_pairs

    # Step 6a: line endings
    new_pairs = []
    i = 0
    while i < len(pairs):
        ch, idx = pairs[i]
        if ch == "\r":
            if i + 1 < len(pairs) and pairs[i + 1][0] == "\n":
                i += 1
                new_pairs.append(pairs[i])
            else:
                new_pairs.append(("\n", idx))
        else:
            new_pairs.append((ch, idx))
        i += 1
    pairs = new_pairs

    # Step 6b: trailing spaces per line
    lines: List[List[Tuple[str, int]]] = []
    current_line: List[Tuple[str, int]] = []
    for ch, idx in pairs:
        if ch == "\n":
            while current_line and current_line[-1][0] in (" ", "\t"):
                current_line.pop()
            lines.append(current_line)
            lines.append([("\n", idx)])
            current_line = []
        else:
            current_line.append((ch, idx))
    while current_line and current_line[-1][0] in (" ", "\t"):
        current_line.pop()
    lines.append(current_line)
    pairs = [p for line in lines for p in line]

    # Step 6c: control characters
    pairs = [(ch, idx) for ch, idx in pairs if not _is_removed_control(ch)]

    # Step 7a: no whitespace before punctuation
    new_pairs = []
    for ch, idx in pairs:
        if ch in _PUNCT_SET:
            while new_pairs and new_pairs[-1][0].isspace():
                new_pairs.pop()
        new_pairs.append((ch, idx))
    pairs = new_pairs

    # Step 7b: one space after punctuation, mapped to the punctuation's index
    new_pairs = []
    i = 0
    while i < len(pairs):
        ch, idx = pairs[i]
        if ch in _PUNCT_SET:
            new_pairs.append((ch, idx))
            i += 1
            while i < len(pairs) and pairs[i][0].isspace():
                i += 1
            new_pairs.append((" ", idx))
        else:
            new_pairs.append((ch, idx))
            i += 1
    pairs = new_pairs

    # Step 8: whitespace runs
    if preserve_newlines:
        if collapse_inline_whitespace:
            new_pairs = []
            i = 0
            while i < len(pairs):
                ch, idx = pairs[i]
                if ch in (" ", "\t"):
                    new_pairs.append((" ", idx))
                    i += 1
                    while i < len(pairs) and pairs[i][0] in (" ", "\t"):
                        i += 1
                else:
                    new_pairs.append((ch, idx))
                    i += 1
            pairs = new_pairs

        new_pairs = []
        nl_count = 0
        for ch, idx in pairs:
            if ch == "\n":
                nl_count += 1
                if nl_count <= 2:
                    new_pairs.append((ch, idx))
            else:
                nl_count = 0
                new_pairs.append((ch, idx))
        pairs = new_pairs

        first = 0
        while first < len(pairs) and pairs[first][0] == "\n":
            first += 1
        if first:
            pairs = pairs[first:]
        while pairs and pairs[-1][0] == "\n":
            pairs.pop()
    else:
        new_pairs = []
        i = 0
        while i < len(pairs):
            ch, idx = pairs[i]
            if ch.isspace():
                new_pairs.append((" ", idx))
                i += 1
                while i < len(pairs) and pairs[i][0].isspace():
                    i += 1
            else:
                new_pairs.append((ch, idx))
                i += 1
        pairs = new_pairs
        first = 0
        while first < len(pairs) and pairs[first][0] == " ":
            first += 1
        if first:
            pairs = pairs[first:]
        while pairs and pairs[-1][0] == " ":
            pairs.pop()

    # Step 9: digits
    pairs = [(_ascii_digit(ch), idx) for ch, idx in pairs]

    # Step 10: case
    if casefold:
        new_pairs = []
        for ch, idx in pairs:
            for fc in ch.casefold():
                new_pairs.append((fc, idx))
        pairs = new_pairs

    normalized = "".join(ch for ch, _ in pairs)
    offset_map = [idx for _, idx in pairs]
    return normalized, offset_map
