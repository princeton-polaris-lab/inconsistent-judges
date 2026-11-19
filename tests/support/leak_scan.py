from __future__ import annotations

import bisect
import dataclasses
import fnmatch
import gzip
import hashlib
import io
import os
import re
import stat
import struct
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, NoReturn

#: Home-directory names that identify no one.
PLACEHOLDER_USERS = frozenset({"user", "me", "runner"})
#: RFC 2606: example domains and reserved top-level domains.
EXAMPLE_DOMAINS = ("example.com", "example.org", "example.net")
RESERVED_TLDS = frozenset({"example", "test", "invalid", "localhost"})

#: Path components, file or directory, that never ship (matched case-insensitively, as globs).
FORBIDDEN_PARTS = (".git", "__pycache__", ".claude", "*.egg-info", ".env*")
#: File names that never ship (matched case-insensitively, as globs): credential stores, keys, caches.
FORBIDDEN_FILES = (
    "*.pem", "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", ".netrc", "_netrc", ".git-credentials",
    ".pypirc", "*.env", "*.pyc", ".coverage", ".coverage.*", "claude.md", ".ds_store",
)
#: The one ``.env*`` name that may ship: a template holding variable names only.
ALLOWED_NAMES = frozenset({".env.example"})

GZIP_MAGIC = b"\x1f\x8b"
OPAQUE_ARCHIVES = {b"\xfd7zXZ\x00": "xz", b"7z\xbc\xaf\x27\x1c": "7z", b"BZh": "bzip2"}
#: A tar's magic, and where it sits: a tar is opaque too (it may hold a zip or gzip the scan never reaches).
TAR_MAGIC, TAR_MAGIC_OFFSET = b"ustar", 257
#: A zip's local-file and end-of-directory signatures, looked for anywhere: a zip still opens after a prefix.
ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06")
UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
#: How many leading bytes decide whether a file without a byte-order mark is ASCII text in UTF-16.
UTF16_PROBE = 64
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_TEXT_CHUNKS = frozenset({b"tEXt", b"zTXt", b"iTXt"})
#: The most any one decompression may produce; a gzip member or PNG chunk over it is unscannable.
MAX_INFLATE = 1 << 28
#: How deep gzip-inside-gzip is followed before the data counts as unscannable.
MAX_DEPTH = 3
#: Characters of text kept on each side of a (non-secret) hit, for the work list.
CONTEXT = 40
#: What each character of a secret becomes in a recorded path or message (one for one, so offsets hold).
SECRET_MASK = "*"
LOOKALIKES = str.maketrans({
    **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u2e3a\u2e3b\ufe58\ufe63\uff0d", "-"),
    **dict.fromkeys("\u00a0\u1680\u202f\u205f\u3000" + "".join(map(chr, range(0x2000, 0x200B))), " "),
    **dict.fromkeys("\u2018\u2019\u201a\u201b\u2032", "'"),
    **dict.fromkeys("\u201c\u201d\u201e\u201f", '"'),
})


class ReleaseError(Exception):
    """The scan cannot run as asked: a refusal, never a leak verdict."""


@dataclass(frozen=True)
class ThirdParty:
    """A data file shipped as published upstream: it skips the broad terms while its bytes are these."""

    path: str
    sha256: str
    license: str


@dataclass(frozen=True)
class Allowance:
    """One ``[[exception]]``: ``term`` may appear in ``path`` while the file's bytes are these."""

    path: str
    sha256: str
    term: str
    reason: str


@dataclass(frozen=True)
class Policy:
    strict: tuple[str, ...]
    broad: tuple[str, ...] = ()
    third_party: tuple[ThirdParty, ...] = ()
    exceptions: tuple[Allowance, ...] = ()


@dataclass(frozen=True)
class _Term:
    text: str
    kind: str  # "strict" | "broad"
    probe: str  # casefolded; found in _probed's text wherever the term matches: a cheap pre-check before the regex
    pattern: re.Pattern[str]


def term_pattern(term: str, *, bounded: bool) -> re.Pattern[str]:
    """The regex a denylisted term is found by."""
    body = r"(?:[\s._-]|%20)+".join(re.escape(word) for word in term.translate(LOOKALIKES).split())
    if term.isdigit():
        body = rf"(?<![0-9]){body}(?![0-9])"
    if bounded:
        body = rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])"
    return re.compile(body, re.IGNORECASE | re.ASCII)


@dataclass(frozen=True)
class _Rule:
    term: str
    pattern: re.Pattern[str]
    probe: str  # casefolded; see _Term.probe
    first_party_only: bool
    secret: bool = False
    accept: Callable[[re.Match[str]], bool] | None = None


def _names_someone(match: re.Match[str]) -> bool:
    return match.group(1).lower() not in PLACEHOLDER_USERS


def _outside_examples(match: re.Match[str]) -> bool:
    domain = match.group(1).lower()
    if domain.rsplit(".", 1)[1] in RESERVED_TLDS:
        return False
    return not any(domain == example or domain.endswith("." + example) for example in EXAMPLE_DOMAINS)


#: Environment variables that hold a credential: a long literal value assigned to one is a secret.
CREDENTIAL_VARIABLES = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "AWS_SECRET_ACCESS_KEY")
#: So does any variable named ``*API_KEY``: a judge's config names the one its key is read from.
API_KEY_VARIABLE = r"[A-Za-z0-9_]*API_KEY"


def _assignment(name: str) -> re.Pattern[str]:
    return re.compile(rf"{name}['\"]?\]?\s*[=:]\s*['\"]?(?=[A-Za-z/+_-]*[0-9])[A-Za-z0-9/+_-]{{20,}}", re.IGNORECASE)


RULES = (
    _Rule("path:/home", re.compile(r"/home/([A-Za-z0-9._-]+)"), "/home/", True, accept=_names_someone),
    _Rule("path:/Users", re.compile(r"/Users/([A-Za-z0-9._-]+)"), "/users/", True, accept=_names_someone),
    _Rule("email", re.compile(r"[A-Za-z0-9._%+-]{1,64}@((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})(?![A-Za-z0-9-])"),
          "@", True, accept=_outside_examples),
    _Rule("secret:openai-style-key", re.compile(r"(?<![A-Za-z0-9])sk-(?=[A-Za-z_-]*[0-9])[A-Za-z0-9_-]{20,}"),
          "sk-", False, secret=True),
    _Rule("secret:google-api-key", re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}"), "aiza", False, secret=True),
    _Rule("secret:huggingface-token", re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{30,}"), "hf_", False,
          secret=True),
    _Rule("secret:aws-access-key-id", re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![0-9A-Z])"), "akia", False,
          secret=True),
    _Rule("secret:github-token", re.compile(r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{36,}"), "gh", False,
          secret=True),
    _Rule("secret:github-token", re.compile(r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{22,}"), "github_pat_",
          False, secret=True),
    _Rule("secret:runpod-api-key", re.compile(r"(?<![A-Za-z0-9_])rpa_[A-Za-z0-9]{30,}"), "rpa_", False,
          secret=True),
    *(_Rule("secret:modal-token", re.compile(rf"(?<![A-Za-z0-9_-]){prefix}[A-Za-z0-9]{{20,}}(?![A-Za-z0-9_-])"),
            prefix, False, secret=True) for prefix in ("ak-", "as-")),
    *(_Rule("secret:credential-assignment", _assignment(name), name.lower().partition("i")[0], False, secret=True)
      for name in CREDENTIAL_VARIABLES),
    _Rule("secret:credential-assignment", _assignment(API_KEY_VARIABLE), "_key", False, secret=True),
    _Rule("secret:private-key",
          re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
                     r"(?:[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----|(?:[A-Za-z0-9+/=\s]|\\[nr/])*)"),
          "private key", False, secret=True),
)
SECRET_TERMS = frozenset(rule.term for rule in RULES if rule.secret)


def _readings(text: str) -> tuple[str, ...]:
    folded = text.translate(LOOKALIKES)
    return (text,) if folded == text else (text, folded)


def _probed(readings: tuple[str, ...]) -> str:
    return readings[-1].casefold()


def _spans(pattern: re.Pattern[str], readings: tuple[str, ...],
           accept: Callable[[re.Match[str]], bool] | None = None) -> set[tuple[int, int]]:
    """``(start, end)`` of each match of ``pattern`` in any of ``readings`` that ``accept`` (if given) takes."""
    return {match.span() for reading in readings for match in pattern.finditer(reading)
            if accept is None or accept(match)}


def _secret_spans(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, rule)`` for every secret-shaped string in ``text``, as written or folded."""
    readings = _readings(text)
    probed = _probed(readings)
    return sorted((start, end, rule.term) for rule in RULES if rule.secret and rule.probe in probed
                  for start, end in _spans(rule.pattern, readings))


def redact_secrets(text: str) -> str:
    """``text`` with every character of each secret-shaped string in it masked (same length, so offsets hold)."""
    for start, end, _ in _secret_spans(text):
        text = text[:start] + SECRET_MASK * (end - start) + text[end:]
    return text


def _forbidden_part(part: str) -> str | None:
    lowered = part.lower()
    if lowered in ALLOWED_NAMES:
        return None
    return next((pattern for pattern in FORBIDDEN_PARTS if fnmatch.fnmatchcase(lowered, pattern)), None)


def forbidden_reason(path: str) -> str | None:
    """Why a file at this repository path must never ship (the glob it matches), or None."""
    parts = path.split("/")
    for part in parts:
        reason = _forbidden_part(part)
        if reason is not None:
            return reason
    base = parts[-1].lower()
    return next((pattern for pattern in FORBIDDEN_FILES if fnmatch.fnmatchcase(base, pattern)), None)


def unsafe_name(name: str) -> bool:
    """A member name that could extract outside its directory, or name nothing."""
    parts = name.rstrip("/").split("/")
    return name.startswith("/") or "\\" in name or "\0" in name or any(part in ("", ".", "..") for part in parts)


def _inflate(data: bytes) -> bytes:
    """A complete zlib stream's contents; raises zlib.error when it is truncated or larger than MAX_INFLATE."""
    inflater = zlib.decompressobj()
    out = inflater.decompress(data, MAX_INFLATE)
    if not inflater.eof:
        raise zlib.error("truncated or oversized deflate stream")
    return out


def _gunzip(data: bytes) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
        out = handle.read(MAX_INFLATE + 1)
    if len(out) > MAX_INFLATE:
        raise OSError("gzip member larger than the scan's inflate limit")
    return out


_PDF_STREAM = re.compile(rb"(?<!end)stream\r?\n")
#: A literal string: escapes, and unescaped parentheses balanced (which PDF allows) to one level of nesting.
_PDF_LITERAL = rb"\((?:\\[\s\S]|[^\\()]|\((?:\\[\s\S]|[^\\()])*\))*\)"
_PDF_HEX = rb"(?<!<)<[0-9A-Fa-f\s]*>"
_PDF_STRING = re.compile(_PDF_LITERAL + rb"|" + _PDF_HEX)
#: A ``TJ`` array: the pieces of one run of text, with kerning numbers between them.
_PDF_TJ = re.compile(rb"\[((?:" + _PDF_LITERAL + rb"|" + _PDF_HEX + rb"|[^\[\]()<>])*)\]\s*TJ")
_PDF_TOKEN = re.compile(rb"(" + _PDF_LITERAL + rb"|" + _PDF_HEX + rb")"
                        rb"|(?<![^\s\0()<>\[\]{}%])(BT|ET)(?![^\s\0()<>\[\]{}/%])")
_PDF_ESCAPE = re.compile(rb"\\(?:([0-7]{1,3})|(\r\n|[\s\S]))")
_PDF_SIMPLE_ESCAPES = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f",
                       b"\r\n": b"", b"\r": b"", b"\n": b""}


def _pdf_unescape(match: re.Match[bytes]) -> bytes:
    if match.group(1) is not None:
        return bytes([int(match.group(1), 8) & 0xFF])
    return _PDF_SIMPLE_ESCAPES.get(match.group(2), match.group(2))


def _pdf_bytes(token: bytes) -> bytes:
    """A literal ``(...)`` string token unescaped, or a hex ``<...>`` one decoded (an odd digit count pads a 0)."""
    if token.startswith(b"("):
        return _PDF_ESCAPE.sub(_pdf_unescape, token[1:-1])
    digits = re.sub(rb"\s", b"", token[1:-1]).decode("ascii")
    return bytes.fromhex(digits + "0" * (len(digits) % 2))


def _pdf_texts(raw: bytes) -> tuple[str, str]:
    """A PDF string's bytes as text, two ways."""
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", "replace"), ""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", "replace"), ""
    return raw.decode("latin-1"), raw.decode("utf-16-be", "replace") if len(raw) % 2 == 0 else ""


def _pdf_strings(buffer: bytes) -> str:
    """Every PDF string in a buffer, one line per reading."""
    lines: list[str] = []
    runs: list[list[tuple[str, str]]] = []  # each run's strings, as their two readings
    inside = False
    for match in _PDF_TOKEN.finditer(buffer):
        if match.group(1) is None:
            inside = match.group(2) == b"BT"
            if inside:
                runs.append([])
            continue
        texts = _pdf_texts(_pdf_bytes(match.group(1)))
        lines.extend(texts)
        if inside:
            runs[-1].append(texts)
    for array in _PDF_TJ.finditer(buffer):
        runs.append([_pdf_texts(_pdf_bytes(match.group())) for match in _PDF_STRING.finditer(array.group(1))])
    lines.extend("".join(texts[reading] for texts in run) for reading in (0, 1) for run in runs)
    return "\n".join(line for line in lines if line)


def _pdf_views(data: bytes) -> Iterator[tuple[str, str | None]]:
    yield "pdf-strings", _pdf_strings(data)
    view = memoryview(data)
    for match in _PDF_STREAM.finditer(data):
        try:
            inflated = zlib.decompressobj().decompress(view[match.end():], MAX_INFLATE)
        except zlib.error:
            continue
        yield f"pdf-stream@{match.end()}", inflated.decode("latin-1")
        yield f"pdf-stream@{match.end()}/strings", _pdf_strings(inflated)


def _png_text(kind: bytes, body: bytes) -> str | None:
    keyword, _, rest = body.partition(b"\0")
    try:
        if kind == b"tEXt":
            return (keyword + b"\n" + rest).decode("latin-1")
        if kind == b"zTXt":
            return (keyword + b"\n" + _inflate(rest[1:])).decode("latin-1")
        compressed = rest[:1] == b"\x01"
        language, _, rest = rest[2:].partition(b"\0")
        translated, _, text = rest.partition(b"\0")
        if compressed:
            text = _inflate(text)
    except zlib.error:
        return None
    return "\n".join(part.decode("utf-8", "replace") for part in (keyword, language, translated, text))


def _png_views(data: bytes) -> Iterator[tuple[str, str | None]]:
    position = len(PNG_SIGNATURE)
    while position + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[position:position + 8])
        body = data[position + 8:position + 8 + length]
        position += 12 + length
        if kind == b"IEND":
            return
        if kind in PNG_TEXT_CHUNKS:
            yield f"png:{kind.decode('ascii')}", _png_text(kind, body)


def _utf16(data: bytes) -> str | None:
    """``data`` as UTF-16 text when it starts with a byte-order mark or alternates NUL and non-NUL; else None."""
    if data.startswith(UTF16_BOMS):
        return data.decode("utf-16", "replace")
    head = data[:UTF16_PROBE]
    if len(head) < 8:
        return None
    for codec, text, nul in (("utf-16-le", head[0::2], head[1::2]), ("utf-16-be", head[1::2], head[0::2])):
        if all(text) and not any(nul):
            return data.decode(codec, "replace")
    return None


def _gunzip_views(data: bytes, depth: int) -> Iterator[tuple[str, str | None]]:
    if depth >= MAX_DEPTH:
        yield "gunzip", None  # nested deeper than the scan follows
        return
    try:
        inner = _gunzip(data)
    except (OSError, EOFError, zlib.error):
        yield "gunzip", None
        return
    for view, text in views(inner, depth + 1):
        yield f"gunzip/{view}", text


def _opaque_archive(data: bytes) -> str | None:
    kind = next((kind for magic, kind in OPAQUE_ARCHIVES.items() if data.startswith(magic)), None)
    if kind is None and data[TAR_MAGIC_OFFSET:TAR_MAGIC_OFFSET + len(TAR_MAGIC)] == TAR_MAGIC:
        kind = "tar"
    if kind is None and any(signature in data for signature in ZIP_SIGNATURES):
        kind = "zip"
    return kind


def views(data: bytes, depth: int = 0) -> Iterator[tuple[str, str | None]]:
    """``(view, text)`` for every text recoverable from ``data``; ``text`` is None where decoding failed."""
    if b"\0" in data:
        yield "latin-1", data.decode("latin-1")
    else:
        yield "text", data.decode("utf-8", "replace")
    utf16 = _utf16(data)
    if utf16 is not None:
        yield "utf-16", utf16
    if data.startswith(GZIP_MAGIC):
        yield from _gunzip_views(data, depth)
    archive = _opaque_archive(data)
    if archive is not None:
        yield f"archive:{archive}", None
    if data.startswith(b"%PDF"):
        yield from _pdf_views(data)
    if data.startswith(PNG_SIGNATURE):
        yield from _png_views(data)


@dataclass(frozen=True)
class Hit:
    """One finding."""

    path: str
    term: str
    kind: str  # "strict" | "broad" | "rule" | "name"
    view: str
    offset: int = 0
    line: int | None = None
    match: str | None = None
    context: str | None = None

    def render(self) -> str:
        where = self.path if self.line is None else f"{self.path}:{self.line}"
        shown = "" if self.match is None else f" {self.match!r}"
        return f"{where} [{self.view}@{self.offset}] {self.kind}:{self.term}{shown}"


@dataclass
class ScanResult:
    hits: list[Hit]
    excepted: list[Hit]
    files: int
    bytes: int
    views: dict[str, int]
    third_party: dict[str, list[str]]  # applied / lapsed / unused
    exceptions_unused: list[dict[str, str]]

    @property
    def ok(self) -> bool:
        return not self.hits

    def summary(self) -> dict[str, dict[str, int]]:
        """Hit counts by path, then by term: the work list."""
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for hit in self.hits:
            grouped[hit.path][hit.term] += 1
        return {path: dict(sorted(terms.items())) for path, terms in sorted(grouped.items())}

    def statistics(self) -> dict[str, Any]:
        return {"files": self.files, "bytes": self.bytes, "views": self.views, "hits": len(self.hits),
                "excepted": len(self.excepted), "third_party": self.third_party,
                "exceptions_unused": self.exceptions_unused}

    def to_json(self) -> dict[str, Any]:
        return {**self.statistics(), "summary": self.summary(),
                "hits": [dataclasses.asdict(hit) for hit in self.hits],
                "excepted_hits": [dataclasses.asdict(hit) for hit in self.excepted]}


class Scanner:
    """Accumulates hits over the files of one tree under one policy."""

    def __init__(self, policy: Policy):
        self.terms = [
            _Term(term, kind, term.translate(LOOKALIKES).split()[0].casefold(),
                  term_pattern(term, bounded=kind == "broad"))
            for kind, terms in (("strict", policy.strict), ("broad", policy.broad)) for term in terms
        ]
        self._third_party = {entry.path: entry for entry in policy.third_party}
        self._exceptions = {(entry.path, entry.term): entry for entry in policy.exceptions}
        self._used_exceptions: set[tuple[str, str]] = set()
        self._third_party_state: dict[str, list[str]] = {"applied": [], "lapsed": []}
        self.hits: list[Hit] = []
        self.excepted: list[Hit] = []
        self.files = 0
        self.bytes = 0
        self.views: Counter[str] = Counter()

    def flag(self, path: str, term: str) -> None:
        """A finding about an entry rather than a file's contents (a link, a forbidden name); never excusable."""
        self.hits.append(Hit(path=redact_secrets(path), term=term, kind="name", view="name"))

    def _term_hits(self, path: str, view: str, text: str, *, first_party: bool) -> list[Hit]:
        readings = _readings(text)
        probed = _probed(readings)
        spans: set[tuple[int, int, str, str, bool]] = set()
        for term in self.terms:
            if (term.kind == "strict" or first_party) and term.probe in probed:
                spans.update((start, end, term.text, term.kind, False) for start, end in _spans(term.pattern, readings))
        for rule in RULES:
            if (first_party or not rule.first_party_only) and rule.probe in probed:
                spans.update((start, end, rule.term, "rule", rule.secret)
                             for start, end in _spans(rule.pattern, readings, rule.accept))
        if not spans:
            return []
        newlines = [m.start() for m in re.finditer("\n", text)]
        return [
            Hit(path=path, term=term, kind=kind, view=view, offset=start,
                line=bisect.bisect_right(newlines, start) + 1,
                match=None if secret else text[start:end],
                context=None if secret else text[max(0, start - CONTEXT):end + CONTEXT].replace("\n", " "))
            for start, end, term, kind, secret in sorted(spans)
        ]

    def _name_hits(self, path: str, name: str) -> list[Hit]:
        found = [Hit(path=path, term=term, kind="rule", view="name", offset=start)
                 for start, _, term in _secret_spans(name)]
        if unsafe_name(name):
            found.append(Hit(path=path, term="unsafe-name", kind="name", view="name", match=name))
        reason = forbidden_reason(name)
        if reason is not None:
            found.append(Hit(path=path, term=f"forbidden-name:{reason}", kind="name", view="name", match=name))
        readings = _readings(name)
        for term in self.terms:
            found.extend(Hit(path=path, term=term.text, kind=term.kind, view="name", offset=start,
                             match=name[start:end]) for start, end in sorted(_spans(term.pattern, readings)))
        return found

    def scan_file(self, path: str, data: bytes, *, name: str | None = None) -> None:
        """Scan one file's name (``name``, default ``path``) and bytes; ``path`` keys the allowances."""
        digest = hashlib.sha256(data).hexdigest()
        self.files += 1
        self.bytes += len(data)
        first_party = True
        third_party = self._third_party.get(path)
        if third_party is not None:
            first_party = third_party.sha256 != digest
            self._third_party_state["lapsed" if first_party else "applied"].append(path)
        shown = redact_secrets(path)
        found = self._name_hits(shown, path if name is None else name)
        for view, text in views(data):
            self.views[re.sub(r"@\d+", "", view)] += 1
            if text is None:
                found.append(Hit(path=shown, term="unscannable", kind="rule", view=view))
            else:
                found.extend(self._term_hits(shown, view, text, first_party=first_party))
        if any(hit.term in SECRET_TERMS for hit in found):
            found = [dataclasses.replace(hit, match=None, context=None) for hit in found]
        for hit in found:
            allowance = self._exceptions.get((path, hit.term))
            if allowance is not None and allowance.sha256 == digest:
                self._used_exceptions.add((path, hit.term))
                self.excepted.append(hit)
            else:
                self.hits.append(hit)

    def result(self) -> ScanResult:
        seen = {path for paths in self._third_party_state.values() for path in paths}
        return ScanResult(
            hits=self.hits,
            excepted=self.excepted,
            files=self.files,
            bytes=self.bytes,
            views=dict(sorted(self.views.items())),
            third_party={**{key: sorted(paths) for key, paths in self._third_party_state.items()},
                         "unused": sorted(set(self._third_party) - seen)},
            exceptions_unused=[dataclasses.asdict(entry) for key, entry in sorted(self._exceptions.items())
                               if key not in self._used_exceptions],
        )


def _unreadable(error: OSError) -> NoReturn:
    raise ReleaseError(f"cannot read {error.filename}: {error.strerror}; the scan does not skip what it cannot "
                       "read") from error


def scan_dir(root: Path | str, policy: Policy, *, allow_empty: bool = False) -> ScanResult:
    """Scan every file under ``root``; paths are relative to it."""
    root = Path(root)
    if not root.is_dir():
        raise ReleaseError(f"{root} is not a directory")
    scanner = Scanner(policy)
    for directory, subdirectories, files in os.walk(root, onerror=_unreadable):
        base = Path(directory)
        for subdirectory in sorted(subdirectories):
            full = base / subdirectory
            reason = _forbidden_part(subdirectory)
            if full.is_symlink():
                term = "symlink"
            elif reason is not None:
                term = f"forbidden-name:{reason}"
            else:
                continue
            scanner.flag(full.relative_to(root).as_posix(), term)
            subdirectories.remove(subdirectory)
        subdirectories.sort()
        for file in sorted(files):
            full = base / file
            relative = full.relative_to(root).as_posix()
            try:
                mode = full.lstat().st_mode
                data = full.read_bytes() if stat.S_ISREG(mode) else None
            except OSError as error:
                _unreadable(error)
            if data is None:
                scanner.flag(relative, "symlink" if stat.S_ISLNK(mode) else "special-file")
            else:
                scanner.scan_file(relative, data)
    if not scanner.files and not scanner.hits and not allow_empty:
        raise ReleaseError(f"{root} holds no file to scan: an empty tree is refused unless it is allowed")
    return scanner.result()


@dataclass(frozen=True)
class TreeAllowance:
    """Findings of ``terms`` (rule ids such as ``email``) the files matching ``glob`` may hold, and why."""

    glob: str
    terms: frozenset[str]
    reason: str


@dataclass(frozen=True)
class TreePolicy:
    """What a public tree may hold beyond nothing; ``forbidden`` names (a site's: its cluster, hosts, account) it may
    hold nowhere, matched as strict terms."""

    identities: Mapping[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
    allowances: tuple[TreeAllowance, ...] = ()
    forbidden: tuple[str, ...] = ()


@dataclass
class TreeResult:
    hits: list[Hit]
    allowed: list[Hit]
    files: int
    unused: list[str]  # the identities and allowances nothing matched: stale entries, reported as such

    @property
    def ok(self) -> bool:
        return not self.hits and not self.unused


def _matches(path: str, globs: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in globs)


def scan_tree(root: Path | str, paths: Iterable[str], policy: TreePolicy) -> TreeResult:
    """Scan each of ``paths`` (relative to ``root``); a path that is not a regular file is refused."""
    root = Path(root)
    scanner = Scanner(Policy(strict=policy.forbidden))
    for relative in sorted(set(paths)):
        full = root / relative
        if full.is_symlink() or not full.is_file():
            raise ReleaseError(f"{relative} is not a regular file under {root}")
        scanner.scan_file(relative, full.read_bytes())
    scanned = scanner.result()
    hits, allowed, used = [], [], set()
    for hit in scanned.hits:
        key = None
        if hit.term not in SECRET_TERMS:
            if hit.term == "email" and hit.match is not None and _matches(hit.path, policy.identities.get(
                    hit.match.lower(), ())):
                key = f"identity {hit.match.lower()}"
            else:
                key = next((f"allowance {entry.glob}" for entry in policy.allowances
                            if hit.term in entry.terms and fnmatch.fnmatchcase(hit.path, entry.glob)), None)
        if key is None:
            hits.append(hit)
        else:
            allowed.append(hit)
            used.add(key)
    declared = [f"identity {address}" for address in policy.identities] + [
        f"allowance {entry.glob}" for entry in policy.allowances]
    return TreeResult(hits=hits, allowed=allowed, files=scanned.files,
                      unused=[key for key in declared if key not in used])
