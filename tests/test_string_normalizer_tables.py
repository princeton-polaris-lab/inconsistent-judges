from __future__ import annotations

import random
import re
import sys
import unicodedata

import pytest

from ruleforge.core import string_normalizer as sn


def _reference(s, *, casefold=True, collapse_inline_whitespace=True, preserve_newlines=True):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for ch in sn._ZERO_WIDTHS:
        s = s.replace(ch, "")
    s = s.replace(sn._SOFT_HYPHEN, "")
    s = s.translate(sn._TRANS_MAP)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip(" \t") for line in s.split("\n"))
    s = "".join(ch for ch in s if (unicodedata.category(ch) not in {"Cf", "Cc"}) or ch in {"\n", "\t"})
    s = re.sub(rf"\s+([{sn._PUNCT_CANON}])", r"\1", s)
    s = re.sub(rf"([{sn._PUNCT_CANON}])\s*", r"\1 ", s)
    if preserve_newlines:
        if collapse_inline_whitespace:
            s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        s = s.strip("\n")
    else:
        s = re.sub(r"\s+", " ", s).strip()
    s = "".join(_reference_digit(ch) for ch in s)
    return s.casefold() if casefold else s


def _reference_digit(ch):
    if ch.isdigit() and not ("0" <= ch <= "9"):
        try:
            return str(unicodedata.digit(ch))
        except Exception:
            try:
                return str(int(unicodedata.numeric(ch)))
            except Exception:
                return ch
    return ch


def test_the_tables_are_the_per_character_steps_at_every_code_point():
    removed, digits = sn._char_tables()
    for code_point in range(sys.maxunicode + 1):
        ch = chr(code_point)
        keep = unicodedata.category(ch) not in {"Cf", "Cc"} or ch in {"\n", "\t"}
        assert (code_point not in removed) == keep, hex(code_point)
        assert ch.translate(digits) == _reference_digit(ch), hex(code_point)
    assert all(code_point >= 128 for code_point in digits)
    assert sn._ASCII_REMOVED_CONTROLS == bytes(cp for cp in range(128) if cp in removed)
    assert sn._char_tables() is sn._char_tables()


ALPHABET = (
    "abcXYZ019 \t\n\r.,:;!?%'\"()-"
    "  ​‍⁠﻿­–—−‘’“”²①"
    "٠१۵１ᵽ9ééक्ष中文\U0001f600‎\u0000\u001f\u007f\u0085"
    "ﬁ 　İß"
)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("flags", [(True, True, True), (True, True, False), (False, False, True), (False, True, False)])
def test_the_function_matches_the_reference_on_adversarial_mixtures(seed, flags):
    rng = random.Random(seed)
    casefold, collapse, newlines = flags
    for _ in range(50):
        text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 80)))
        expected = _reference(text, casefold=casefold, collapse_inline_whitespace=collapse, preserve_newlines=newlines)
        assert sn.normalize_for_match(text, casefold=casefold, collapse_inline_whitespace=collapse, preserve_newlines=newlines) == expected


@pytest.mark.parametrize("text", ["plain ascii text: with punctuation ; and\x01controls\x7f", "café – ١٢", None, ""])
def test_the_ascii_fast_path_and_the_general_path_agree_with_the_reference(text):
    assert sn.normalize_for_match(text, preserve_newlines=False) == _reference(text, preserve_newlines=False)
