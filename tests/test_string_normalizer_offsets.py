from __future__ import annotations

import itertools
import random
import unicodedata

import pytest

from ruleforge.core import string_normalizer as sn

FLAGS = [dict(zip(("casefold", "collapse_inline_whitespace", "preserve_newlines"), values))
         for values in itertools.product((True, False), repeat=3)]

PIECES = ["a", "B", "\r", "\r\n", "\n", "\n\n\n", "  ", "\t", " ,", ";  x", "​", "­", "ﬁ", "é",
          "—", "’", " ", "٣", "ß", "  \n", "\u0007", "!", "%"]


def mixtures(seed: int, count: int = 200) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(PIECES) for _ in range(rng.randint(1, 12))) for _ in range(count)]


@pytest.mark.parametrize("flags", FLAGS, ids=lambda flags: "-".join(k for k, v in flags.items() if v) or "none")
def test_the_offset_mapped_string_is_the_plain_normalization(flags):
    for text in mixtures(sum(flags.values())):
        normalized, offsets = sn.normalize_with_offset_map(text, **flags)
        assert normalized == sn.normalize_for_match(text, **flags), repr(text)
        assert len(offsets) == len(normalized) and all(0 <= index < len(text) for index in offsets), repr(text)


# Characters NFKC composes across combining sequences (Hangul jamo, halfwidth kana and their voiced marks, Indic and
# Tibetan vowel signs), combining marks of several classes, and Unicode whitespace other than space, tab and newline.
UNICODE_PIECES = ["\u1100", "\u1161", "\u11a8", "\uac00", "\uff76", "\uff9e", "\u0301", "\u0323", "\u0304",
                  "\u09c7", "\u09be", "\u0f73", "\u0338", "<", "\u2028", "\u2029", "\u1680", " ", ",", "\n", "x"]


@pytest.mark.parametrize("flags", FLAGS, ids=lambda flags: "-".join(k for k, v in flags.items() if v) or "none")
def test_the_offset_mapped_string_is_the_plain_normalization_beyond_combining_sequences(flags):
    rng = random.Random(sum(flags.values()))
    texts = ["x\u1100\u1161y", "\u1100\u1161\u11a8 x", "\uff76\uff9e", "a\uff9e\u0301", "a\u2028,b", "a\u2029b",
             "Rule 1 :\u2028x", "\u0f71\u0f73\u0301"]
    texts += ["".join(rng.choice(UNICODE_PIECES) for _ in range(rng.randint(1, 12))) for _ in range(500)]
    for text in texts:
        normalized, offsets = sn.normalize_with_offset_map(text, **flags)
        assert normalized == sn.normalize_for_match(text, **flags), repr(text)
        assert len(offsets) == len(normalized) and all(0 <= index < len(text) for index in offsets), repr(text)


def test_a_character_composed_from_several_maps_to_the_first_and_its_source_ends_after_the_last():
    text = "x\u1100\u1161\u11a8\u0301y"
    normalized, offsets = sn.normalize_with_offset_map(text)
    assert (normalized, offsets) == ("x\uac01\u0301y", [0, 1, 3, 5])  # a mark maps to its sequence's base
    assert sn.source_end(text, 1, 2) == sn.source_end(text, 3, 4) == 5
    assert sn.source_end(text, 0, 1) == 1 and sn.source_end(text, 5, 6) == 6


def test_an_offset_points_at_the_character_it_came_from():
    text = "Ab\r\ncd  ,e"
    normalized, offsets = sn.normalize_with_offset_map(text)
    assert normalized == "ab\ncd, e"
    assert [text[index] for index in offsets] == ["A", "b", "\n", "c", "d", ",", ",", "e"]
    assert sn.normalize_with_offset_map("") == ("", []) and sn.normalize_with_offset_map(None) == ("", [])


class _Database:
    """The Unicode database, but for the lookups a test takes away."""

    def __init__(self, *missing: str):
        self.missing = missing

    def __getattr__(self, name):
        if name in self.missing:
            def lookup(ch, *default):
                raise ValueError(f"no {name} for {ch!r}")
            return lookup
        return getattr(unicodedata, name)


@pytest.mark.parametrize(("missing", "expected"), [(("digit",), "3"), (("digit", "numeric"), "٣")])
def test_a_digit_the_database_cannot_value_falls_back_to_its_numeric_value_then_to_itself(monkeypatch, missing,
                                                                                          expected):
    monkeypatch.setattr(sn, "unicodedata", _Database(*missing))
    assert sn._ascii_digit("٣") == expected  # ARABIC-INDIC DIGIT THREE
    assert sn._ascii_digit("7") == "7"
    assert sn.normalize_with_offset_map("x٣", casefold=False) == (f"x{expected}", [0, 1])
