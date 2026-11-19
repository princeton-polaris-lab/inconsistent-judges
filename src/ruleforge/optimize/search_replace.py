# The search/replace block format is adapted from https://github.com/Aider-AI/aider

from typing import Dict, Iterator, List, Optional, Set, Tuple
import re
import unicodedata

from ruleforge.core.string_normalizer import normalize_for_match, normalize_with_offset_map, source_end
from fuzzysearch import find_near_matches

#: A near match may differ from the search text in at most this share of its length.
MAX_NEAR_MATCH_SHARE = 0.1


def _places(text: str, search: str) -> Iterator[int]:
    place = text.find(search)
    while place != -1:
        yield place
        place = text.find(search, place + 1)


def _splits_marks(text: str, start: int, end: int) -> bool:
    """Return whether `text[start:end]` begins or ends inside a character's combining marks."""
    return bool(unicodedata.combining(text[start]) or (end < len(text) and unicodedata.combining(text[end])))


class SearchAndReplace:
    def __init__(self):
        self.START_RE = re.compile(r"^\s*<{3,12}\s*SEARCH\s*$")
        self.SEP_RE = re.compile(r"^\s*={3,12}\s*$")
        self.END_RE = re.compile(r"^\s*>{3,12}\s*REPLACE\s*$")
        self.MARKER_LIKE_RE = re.compile(r"^\s*(?:<{3,}|={3,}|>{3,})")
        self.RULE_HEADER_RE = re.compile(
            r"^\s*(?P<label>Meta[-\s]*Rule|Rule)\s*(?P<number>\d+)\s*:",
            re.IGNORECASE | re.MULTILINE,
        )

    def _is_start(self, line: str) -> bool:
        return self.START_RE.match(line) is not None

    def _is_sep(self, line: str) -> bool:
        return self.SEP_RE.match(line) is not None

    def _is_end(self, line: str) -> bool:
        return self.END_RE.match(line) is not None

    def _refuse_marker_like(self, lines: List[str], index: int) -> None:
        if self.MARKER_LIKE_RE.match(lines[index]):
            raise ValueError(f"Line {index + 1} looks like a block marker but cannot be used here: "
                             f"{lines[index].strip()!r}")

    def extract_multiple_search_replace_blocks(
        self, block_text: str
    ) -> List[Tuple[str, str]]:
        """Return the (search, replace) pairs of every SEARCH/REPLACE block in `block_text`, refusing the whole patch
        for a malformed block or a line that looks like a marker and is not used as one."""
        lines = block_text.splitlines(keepends=True)
        n = len(lines)
        i = 0
        out: List[Tuple[str, str]] = []

        while i < n:
            if self._is_start(lines[i]):
                start_line_no = i + 1
                i += 1

                search_buf: List[str] = []
                while i < n and not self._is_sep(lines[i]):
                    self._refuse_marker_like(lines, i)
                    search_buf.append(lines[i])
                    i += 1
                if i >= n:
                    raise ValueError(
                        "Missing divider (=====) after line {}.".format(start_line_no)
                    )
                i += 1

                if search_buf:
                    search_buf[-1] = search_buf[-1].rstrip()

                replace_buf: List[str] = []
                while i < n and not self._is_end(lines[i]):
                    self._refuse_marker_like(lines, i)
                    replace_buf.append(lines[i])
                    i += 1
                if i >= n:
                    raise ValueError(
                        "Missing end marker (>>>>>> REPLACE) for block starting at line {}.".format(
                            start_line_no
                        )
                    )
                i += 1

                if replace_buf:
                    replace_buf[-1] = replace_buf[-1].rstrip()

                out.append(
                    ("".join(search_buf), "".join(replace_buf) if replace_buf else "")
                )
            else:
                self._refuse_marker_like(lines, i)
                i += 1

        if not out:
            raise ValueError(
                "No SEARCH/REPLACE block found (expected '<{3..12} SEARCH' on its own line)."
            )
        return out

    def _canonical_rule_key(self, label: str, number: str) -> Tuple[str, str]:
        canonical_label = label.lower().replace(" ", "-")
        canonical_label = canonical_label.replace("--", "-")
        normalized_number = str(int(number))
        return canonical_label, normalized_number

    def _extract_rule_chunks(
        self, block_text: str
    ) -> List[Tuple[Tuple[str, str], str]]:
        if not block_text:
            return []

        matches = list(self.RULE_HEADER_RE.finditer(block_text))
        if not matches:
            return []

        chunks: List[Tuple[Tuple[str, str], str]] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = (
                matches[idx + 1].start() if idx + 1 < len(matches) else len(block_text)
            )
            chunk = block_text[start:end].strip("\n")
            if not chunk:  # pragma: no cover -- a header match is never blank, so neither is its chunk
                continue
            key = self._canonical_rule_key(match.group("label"), match.group("number"))
            chunks.append((key, chunk))
        return chunks

    def _rule_keys(self, block_text: str) -> List[Tuple[str, str]]:
        return [key for key, _ in self._extract_rule_chunks(block_text)]

    def _headers_in(self, text: str, start: int = 0, end: Optional[int] = None) -> List[Tuple[str, str]]:
        """Return the rules whose header starts in `text[start:end]`. Each line is normalized on its own: normalizing the
        whole text can join a header to the line above it, and that header would go unseen."""
        end = len(text) if end is None else end
        keys: List[Tuple[str, str]] = []
        offset = 0
        for line in text.splitlines(keepends=True):
            header = self.RULE_HEADER_RE.match(normalize_for_match(line, casefold=False, preserve_newlines=True))
            if header and start <= offset + len(line) - len(line.lstrip()) < end:
                keys.append(self._canonical_rule_key(header.group("label"), header.group("number")))
            offset += len(line)
        return keys

    def _index_rule_lines(
        self, text: str
    ) -> Tuple[List[str], Dict[Tuple[str, str], Tuple[int, int]], bool]:
        lines = text.splitlines()
        header_spans: List[Tuple[Tuple[str, str], int]] = []
        seen_keys: Set[Tuple[str, str]] = set()
        for idx, line in enumerate(lines):
            match = self.RULE_HEADER_RE.match(line)
            if not match:
                continue
            key = self._canonical_rule_key(match.group("label"), match.group("number"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            header_spans.append((key, idx))

        span_map: Dict[Tuple[str, str], Tuple[int, int]] = {}
        for pos, (key, start_idx) in enumerate(header_spans):
            end_idx = (
                header_spans[pos + 1][1] if pos + 1 < len(header_spans) else len(lines)
            )
            span_map[key] = (start_idx, end_idx)

        trailing_newline = text.endswith("\n")
        return lines, span_map, trailing_newline

    def _rule_number_fallback(
        self, text: str, search_text: str, replace_text: str
    ) -> Optional[str]:
        """Return `text` with each rule SEARCH names replaced by REPLACE's rule of that id, or None when a rule is
        missing, SEARCH covers only part of a rule or holds text before its first header; a block it would apply only in
        part is refused."""
        replace_chunks = self._extract_rule_chunks(replace_text)
        search_header = self.RULE_HEADER_RE.search(search_text)
        if not replace_chunks or search_header is None or search_text[:search_header.start()].strip():
            return None

        search_ids, replace_ids = sorted(self._rule_keys(search_text)), sorted(self._rule_keys(replace_text))
        first_header = self.RULE_HEADER_RE.search(replace_text)
        drops_text = replace_text[:first_header.start()].strip()
        if search_ids != replace_ids or len(set(search_ids)) != len(search_ids) or drops_text:
            raise ValueError(
                "SEARCH and REPLACE do not name the same rules once each, so a replacement by rule number would drop "
                f"part of it.\nSearch text: {search_text}\nReplace text: {replace_text}"
            )

        lines, span_map, trailing_newline = self._index_rule_lines(text)
        if any(key not in span_map for key in search_ids):
            return None
        for key, chunk in self._extract_rule_chunks(search_text):
            if not self._covers(chunk, "\n".join(lines[span_map[key][0]:span_map[key][1]])):
                return None

        for start_idx, end_idx, replacement_lines in sorted(
            ((*span_map[key], chunk.rstrip("\n").splitlines()) for key, chunk in replace_chunks),
            key=lambda item: item[0],
            reverse=True,
        ):
            lines[start_idx:end_idx] = replacement_lines

        result = "\n".join(lines)
        if trailing_newline:
            result += "\n"
        return result

    @staticmethod
    def _covers(search_chunk: str, rule: str) -> bool:
        """Return whether a search is as long as `rule`, short of at most `MAX_NEAR_MATCH_SHARE` of it, whatever its
        words."""
        rule_length = len(normalize_for_match(rule, casefold=False, preserve_newlines=True))
        search_length = len(normalize_for_match(search_chunk, casefold=False, preserve_newlines=True))
        return search_length >= rule_length * (1 - MAX_NEAR_MATCH_SHARE)

    def search_replace(self, text: str, search_text: str, replace_text: str) -> str:
        """Return `text` with the first place `search_text` matches (exact, normalized, by rule number, then one near
        match) replaced in the original text, refusing a place without the search's rule headers or one that splits a
        character from its combining marks."""
        normalized_search_text = normalize_for_match(
            search_text, casefold=False, preserve_newlines=True
        )
        if not normalized_search_text.strip():
            raise ValueError("A SEARCH section cannot be empty.")

        named = self._headers_in(search_text)
        for start in _places(text, search_text):
            end = start + len(search_text)
            if self._headers_in(text, start, end) == named and not _splits_marks(text, start, end):
                return text[:start] + replace_text + text[end:]

        normalized_text, offset_map = normalize_with_offset_map(
            text, casefold=False, preserve_newlines=True
        )

        def _original_span(norm_start, norm_end):
            # the offset map points at a character's base: its combining marks, and what composes into it, follow it
            last = offset_map[norm_end - 1]
            return offset_map[norm_start], source_end(text, last, last + 1)

        search_len = len(normalized_search_text)
        for place in _places(normalized_text, normalized_search_text):
            start, end = _original_span(place, place + search_len)
            if self._headers_in(text, start, end) == named and not _splits_marks(normalized_text, place, place + search_len):
                return text[:start] + replace_text + text[end:]

        rule_number_result = self._rule_number_fallback(
            text, search_text, replace_text
        )
        if rule_number_result is not None:
            return rule_number_result

        max_distance = int(search_len * MAX_NEAR_MATCH_SHARE)
        matches = (
            find_near_matches(normalized_search_text, normalized_text, max_l_dist=max_distance)
            if max_distance
            else []
        )
        if len(matches) > 1:
            raise ValueError(
                f"Search text is near {len(matches)} places in the text, so it names none of them.\n"
                f"Search text: {search_text}"
            )
        if matches:
            if _splits_marks(normalized_text, matches[0].start, matches[0].end):
                raise ValueError(f"Search text ends inside a character's combining marks.\nSearch text: {search_text}")
            start, end = _original_span(matches[0].start, matches[0].end)
            if self._headers_in(text, start, end) != named:
                raise ValueError(
                    f"Search text is near a place under other rule headers, so it does not name that place.\n"
                    f"Search text: {search_text}\nPlace: {text[start:end]}"
                )
            return text[:start] + replace_text + text[end:]

        raise ValueError(f"Search text not found in text.\nSearch text: {search_text}")
