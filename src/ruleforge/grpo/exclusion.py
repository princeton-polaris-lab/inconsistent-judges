from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence

from ruleforge.core.digests import sha256_file
from ruleforge.core.errors import RuleForgeError
from ruleforge.grpo.leakage import PROMPT_IDENTITY, LeakageError, disjoint_marker_path
from ruleforge.core.paths import artifact_uri_or_path, is_artifact_uri, is_repository_uri, resolve
from ruleforge.core.artifacts import (
    compute_prompt_response_hash,
    extract_prompt_response,
    iter_jsonl_records,
    write_text_atomically,
)
from ruleforge.core.string_normalizer import normalize_for_match

IDENTITY = PROMPT_IDENTITY
#: Bump when the normalizer changes.
NORMALIZER_VERSION = "normalize_for_match(casefold=True,collapse_inline_whitespace=True,preserve_newlines=False)/v1"
SHINGLE_VERSION = "word-ngram-blake2b64/v1"

#: The thresholds of the near-copy screen the published training rows passed (no code here runs it); every policy
#: records them, so its digest is the one the published markers carry.
NEAR_COPY_SCREEN: Mapping[str, Any] = MappingProxyType({
    "shingle_n": 5,
    "jaccard_flag": 0.5,
    "payload_flag": 0.2,
    "payload_strong": 0.5,
    "min_rare_shingles": 3,
    "rare_df_floor": 5,
    "rare_df_fraction": 1e-4,
    "df_units": {"stress": "prompt", "weirdchat": "pattern_id", "wildchat": "hashed_ip"},
    "family_prefix_pattern": r"^(?P<prefix>.+)/pg\d+$",
    "related_fields": {"weirdchat": "behavior_id"},
    "related_payload_flag": 0.03,
    "related_review_floor": 0.01,
})


MARKER_KIND = "prompt_disjoint"
#: The marker's schema: it records its policy's digest. Any other schema is refused.
MARKER_SCHEMA_VERSION = 2
SECOND_PASS_CHECKS = ("identity", "containment", "family")
#: A marker's ``screen``: ``full`` (near-copy tiers, then the second pass; a marker without the key is one) or the second
#: pass alone. Their policy digests differ, so neither passes a check made for the other.
FULL_SCREEN = "full"
UNSCREENED = "containment-only, unscreened"
SCREENS = (FULL_SCREEN, UNSCREENED)

_HEX64 = re.compile(r"[0-9a-f]{64}")


class ExclusionError(RuntimeError, RuleForgeError):
    """The inputs cannot be screened as declared."""


def normalized_prompt(text: str) -> str:
    """Return the normalized text every identity, containment and shingle comparison reads."""
    return normalize_for_match(text, casefold=True, collapse_inline_whitespace=True, preserve_newlines=False)


def key_of_normalized(normalized: str) -> str:
    """Return the SHA-256 of an already normalized prompt."""
    return hashlib.sha256(normalized.encode("utf-8", "surrogatepass")).hexdigest()


def prompt_key(text: str) -> str:
    """Return the exclusion, dedup and split identity of a prompt."""
    return key_of_normalized(normalized_prompt(text))


def _default_family_fields() -> dict[str, str]:
    # a stress source_key names one scenario: its rewordings share it
    return {"stress": "source_key", "weirdchat": "pattern_id"}


@dataclass(frozen=True)
class ExclusionPolicy:
    """The second pass's settings (`family_fields` keyed by a row's `source` value), digested into each marker with
    the near-copy screen's recorded thresholds (`NEAR_COPY_SCREEN`)."""

    min_containment_chars: int = 40
    family_fields: Mapping[str, str] = field(default_factory=_default_family_fields)

    def __post_init__(self) -> None:
        value = self.min_containment_chars
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ExclusionError(f"policy.min_containment_chars must be a positive int, got {value!r}")
        fields = self.family_fields
        if not isinstance(fields, Mapping) or not all(isinstance(k, str) and isinstance(v, str) and k and v for k, v in fields.items()):
            raise ExclusionError("policy.family_fields must map source names to field names")
        object.__setattr__(self, "family_fields", dict(sorted(fields.items())))

    def family_of(self, source: Optional[str], meta: Mapping[str, Any], row: Optional[Mapping[str, Any]] = None) -> Optional[str]:
        """Return a row's family value namespaced by its source, or None."""
        return _field_value(self.family_fields, source, meta, row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": IDENTITY,
            "normalizer_version": NORMALIZER_VERSION,
            "shingle_version": SHINGLE_VERSION,
            "min_containment_chars": self.min_containment_chars,
            **copy.deepcopy(dict(NEAR_COPY_SCREEN)),
            "family_fields": dict(self.family_fields),
        }

    def digest(self, screen: str = FULL_SCREEN) -> str:
        """Return the SHA-256 of `to_dict` as canonical JSON; an unscreened marker's digest also covers the screen's name."""
        if screen not in SCREENS:
            raise ExclusionError(f"a marker's screen is one of {SCREENS}, not {screen!r}")
        payload: Any = self.to_dict() if screen == FULL_SCREEN else {"policy": self.to_dict(), "screen": screen}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "ExclusionPolicy":
        """Return a policy from a config mapping; the version fields must be this module's."""
        data = dict(payload or {})
        pinned = {"identity": IDENTITY, "normalizer_version": NORMALIZER_VERSION, "shingle_version": SHINGLE_VERSION}
        for name, expected in pinned.items():
            if name in data and data.pop(name) != expected:
                raise ExclusionError(f"policy.{name} must be {expected!r}; this code computes nothing else")
        for name, recorded in NEAR_COPY_SCREEN.items():
            if name in data and data.pop(name) != recorded:
                raise ExclusionError(f"policy.{name} must be {recorded!r}, the near-copy screen's recorded value: "
                                     "this code runs no near-copy screen")
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ExclusionError(f"unknown exclusion policy keys: {unknown}")
        return cls(**data)


def recorded_policy(payload: Any) -> ExclusionPolicy:
    """Return the policy a manifest or marker records; a record missing a field is refused."""
    if not isinstance(payload, Mapping):
        raise ExclusionError(f"a recorded exclusion policy must be a mapping, not {type(payload).__name__}")
    policy = ExclusionPolicy.from_mapping(payload)
    missing = sorted(set(policy.to_dict()) - set(payload))
    if missing:
        raise ExclusionError(f"the recorded exclusion policy names no {missing}: it was written under another rule")
    return policy


def _field_value(fields: Mapping[str, str], source: Optional[str], meta: Mapping[str, Any], row: Optional[Mapping[str, Any]]) -> Optional[str]:
    name = fields.get(source) if source is not None else None
    value = meta.get(name, (row or {}).get(name)) if name else None
    return f"{source}:{value}" if isinstance(value, str) and value else None


def _row_fields(record: Mapping[str, Any]) -> tuple[Optional[str], Mapping[str, Any], Mapping[str, Any]]:
    nested = record.get("scenario") if isinstance(record.get("scenario"), Mapping) else {}
    fields = {**nested, **record}
    source, meta = fields.get("source"), fields.get("meta")
    return (source if isinstance(source, str) else None), (meta if isinstance(meta, Mapping) else {}), fields


def resolve_input(path: Path | str) -> Path:
    """Return an `artifact://` or `repo://` URI resolved, a plain path as given."""
    text = str(path)
    return resolve(text) if is_artifact_uri(text) or is_repository_uri(text) else Path(text).expanduser()


def _read_records(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    if not path.is_file():
        raise ExclusionError(f"{path} does not exist")
    data = path.read_bytes()
    return data, list(iter_jsonl_records(path))


def _text_rows(records: Sequence[Mapping[str, Any]], *, name: str, policy: ExclusionPolicy) -> list[tuple[str, str, Optional[str]]]:
    out = []
    for number, record in enumerate(records, start=1):
        pair = extract_prompt_response(record)
        if pair is None:
            raise LeakageError(f"{name} row {number} has no string prompt and response, so its disjointness cannot be checked")
        source, meta, fields = _row_fields(record)
        out.append((compute_prompt_response_hash(*pair), normalized_prompt(pair[0]), policy.family_of(source, meta, fields)))
    return out


def _preview(items: Sequence[str], limit: int = 5) -> str:
    shown = ", ".join(item[:12] for item in items[:limit])
    return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def assert_prompt_disjoint(
    dataset: Sequence[Mapping[str, Any]],
    protected: Sequence[Sequence[Mapping[str, Any]]],
    *,
    policy: ExclusionPolicy,
    allowed_family_pairs: Iterable[tuple[str, str]] = (),
    dataset_name: str = "dataset",
) -> dict[str, int]:
    """Run the second pass (identical prompts, containment, shared family) and raise `LeakageError` on any leak."""
    rows = _text_rows(dataset, name=dataset_name, policy=policy)
    guarded = [entry for index, records in enumerate(protected) for entry in _text_rows(records, name=f"protected file {index + 1}", policy=policy)]
    if not rows or not guarded:
        raise LeakageError(
            f"{dataset_name} has {len(rows)} row(s) and the protected files {len(guarded)}; a disjointness "
            "check against nothing passes vacuously"
        )
    allowed = set(allowed_family_pairs)
    minimum = policy.min_containment_chars
    guarded_texts = {text for _, text, _ in guarded}
    row_texts = {text for _, text, _ in rows}
    identity = sorted(sid for sid, text, _ in rows if text in guarded_texts)
    joined_guarded = "\x00".join(text for _, text, _ in guarded)
    joined_rows = "\x00".join(text for _, text, _ in rows)
    contained = sorted(sid for sid, text, _ in rows if len(text) >= minimum and text not in guarded_texts and text in joined_guarded)
    containing = sorted(pid for pid, text, _ in guarded if len(text) >= minimum and text not in row_texts and text in joined_rows)
    families: dict[str, list[str]] = defaultdict(list)
    for pid, _, family in guarded:
        if family is not None:
            families[family].append(pid)
    family_leaks = sorted(
        sid for sid, _, family in rows if family is not None and any((pid, sid) not in allowed for pid in families.get(family, ()))
    )
    problems = []
    for label, items in (
        ("share a normalized prompt with a protected scenario", identity),
        ("are contained in a protected prompt", contained),
        ("protected prompt(s) are contained in a dataset prompt", containing),
        ("share a family with a protected scenario without an adjudicated keep", family_leaks),
    ):
        if items:
            problems.append(f"{len(items)} {label}: {_preview(items)}")
    if problems:
        raise LeakageError(f"{dataset_name} is not prompt-disjoint from the protected set: " + "; ".join(problems))
    return {"rows": len(rows), "prompt_keys": len({text for _, text, _ in rows}), "protected_rows": len(guarded), "protected_prompt_keys": len(guarded_texts)}


def _load_for_marker(path: Path, role: str) -> tuple[str, list[dict[str, Any]]]:
    data, records = _read_records(path)
    if not records:
        raise LeakageError(f"{role} {path} holds no records")
    return hashlib.sha256(data).hexdigest(), records


def prompt_disjoint_marker(
    dataset: Path | str,
    protected: Sequence[Path | str],
    *,
    policy: ExclusionPolicy,
    allowed_family_pairs: Iterable[tuple[str, str]] = (),
    carved: Sequence[Path | str] = (),
    screen: str,
) -> dict[str, Any]:
    """Re-read `dataset` and every protected and carved file, run the second pass and return the marker describing it;
    `carved` files are read but no policy is attested against them, and `screen` says whether the rows were screened."""
    if screen not in SCREENS:
        raise ExclusionError(f"a marker's screen is one of {SCREENS}, not {screen!r}")
    dataset_path = Path(dataset)
    dataset_sha, records = _load_for_marker(dataset_path, "dataset")

    def read(given: Path | str, role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        path = resolve_input(given)
        digest, rows = _load_for_marker(path, role)
        return {"path": artifact_uri_or_path(path), "sha256": digest, "rows": len(rows)}, rows

    guarded = [read(given, "protected file") for given in protected]
    carve_outs = [read(given, "carved file") for given in carved]
    allowed = sorted(set(allowed_family_pairs))
    stats = assert_prompt_disjoint(
        records, [rows for _, rows in guarded + carve_outs], policy=policy, allowed_family_pairs=allowed, dataset_name=str(dataset_path)
    )
    keys = sorted({prompt_key(pair[0]) for pair in (extract_prompt_response(record) for record in records) if pair is not None})
    marker = {
        "kind": MARKER_KIND,
        "schema_version": MARKER_SCHEMA_VERSION,
        "identity": IDENTITY,
        "normalizer_version": NORMALIZER_VERSION,
        "policy": policy.to_dict(),
        "policy_sha256": policy.digest(screen),
        "checks": list(SECOND_PASS_CHECKS),
        "disjoint": True,
        "dataset": {
            "name": dataset_path.name,
            "sha256": dataset_sha,
            "rows": stats["rows"],
            "prompt_keys": stats["prompt_keys"],
            "prompt_keys_sha256": hashlib.sha256("\n".join(keys).encode("ascii")).hexdigest(),
        },
        "protected": [entry for entry, _ in guarded],
        "protected_rows": stats["protected_rows"],
        "protected_prompt_keys": stats["protected_prompt_keys"],
        "allowed_family_pairs": [list(pair) for pair in allowed],
    }
    if carve_outs:
        marker["carved"] = [{**entry, "checks": list(SECOND_PASS_CHECKS)} for entry, _ in carve_outs]
    if screen != FULL_SCREEN:
        marker["screen"] = screen
    return marker


def marker_screen(marker: Mapping[str, Any]) -> Any:
    """The screen a marker records (:data:`SCREENS`); a marker without the key went through the full screen."""
    return marker.get("screen", FULL_SCREEN)


def assert_prompt_marker(marker: Mapping[str, Any], source: str = "marker") -> None:
    """Refuse a marker that does not positively attest prompt disjointness."""
    problems = []
    if marker.get("kind") != MARKER_KIND or marker.get("schema_version") != MARKER_SCHEMA_VERSION:
        problems.append(f"it is not a {MARKER_KIND} marker of schema {MARKER_SCHEMA_VERSION}")
    if marker.get("identity") != IDENTITY or marker.get("normalizer_version") != NORMALIZER_VERSION:
        problems.append("it was not keyed on this normalized-prompt identity")
    if marker.get("disjoint") is not True:
        problems.append("it does not record disjoint: true")
    if marker_screen(marker) not in SCREENS:
        problems.append(f"it names a screen this code does not know ({marker_screen(marker)!r})")
    elif _marker_policy_digest(marker) is None:
        problems.append("it records no exclusion policy that parses and matches its policy_sha256")
    dataset = marker.get("dataset") if isinstance(marker.get("dataset"), Mapping) else {}
    if not dataset.get("rows") or not dataset.get("prompt_keys") or not _HEX64.fullmatch(str(dataset.get("sha256", ""))):
        problems.append("it names no checked dataset rows or digest")
    if not marker.get("protected") or not marker.get("protected_rows"):
        problems.append("it compared against no protected rows")
    if problems:
        raise LeakageError(f"{source} does not attest prompt disjointness: " + "; ".join(problems))


def _marker_policy_digest(marker: Mapping[str, Any]) -> Optional[str]:
    try:
        digest = recorded_policy(marker.get("policy")).digest(marker_screen(marker))
    except (ExclusionError, TypeError):
        return None
    return digest if marker.get("policy_sha256") == digest else None


def _digests(marker: Mapping[str, Any], key: str) -> set[str]:
    return {str(entry.get("sha256")) for entry in marker.get(key) or () if isinstance(entry, Mapping)}


def _protected_digests(marker: Mapping[str, Any]) -> set[str]:
    return _digests(marker, "protected")


def _read_digests(marker: Mapping[str, Any]) -> set[str]:
    return _digests(marker, "protected") | _digests(marker, "carved")


def write_prompt_disjoint_marker(dataset: Path | str, marker: Mapping[str, Any]) -> Path:
    """Write `marker` beside `dataset`, validated first; a marker there that attests more is never overwritten."""
    assert_prompt_marker(marker, source="marker")
    path = disjoint_marker_path(Path(dataset))
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            existing = None
        if isinstance(existing, Mapping) and isinstance(existing.get("dataset"), Mapping) and (
            existing["dataset"].get("sha256") == marker["dataset"]["sha256"]
        ):
            dropped = sorted(
                (_protected_digests(existing) - _protected_digests(marker)) | (_read_digests(existing) - _read_digests(marker))
            )
            if dropped:
                raise LeakageError(
                    f"{path} already covers {len(dropped)} file(s) of this dataset's that the new marker covers less or "
                    f"not at all ({_preview(dropped)}); refusing to narrow it"
                )
            if marker_screen(existing) == FULL_SCREEN and marker_screen(marker) != FULL_SCREEN:
                raise LeakageError(
                    f"{path} already attests this dataset's bytes under the full screen; refusing to replace it with "
                    f"a marker of the second pass alone ({marker_screen(marker)})"
                )
    write_text_atomically(path, json.dumps(dict(marker), indent=2))
    return path


def check_prompt_marker(
    dataset: Path | str,
    marker_path: Optional[Path | str] = None,
    *,
    required_protected: Iterable[str] = (),
    required_carved: Iterable[str] = (),
    policy: Optional[ExclusionPolicy] = None,
    allow_unscreened: bool = False,
) -> dict[str, Any]:
    """Return the marker beside `dataset`, accepted only for the dataset's current bytes, under `policy` (by digest) and
    attesting against every file `required_protected` names and reading every file `required_carved` names; a marker of
    the second pass alone is refused unless `allow_unscreened`."""
    dataset_path = Path(dataset)
    path = Path(marker_path) if marker_path is not None else disjoint_marker_path(dataset_path)
    if not path.is_file():
        raise LeakageError(f"{dataset_path} has no prompt-disjointness marker at {path}; refusing to use it")
    marker = json.loads(path.read_text(encoding="utf-8"))
    assert_prompt_marker(marker, source=str(path))
    screen = marker_screen(marker)
    required = (policy if policy is not None else ExclusionPolicy()).digest(screen)
    if marker["policy_sha256"] != required:
        raise LeakageError(
            f"{path} was written under exclusion policy {marker['policy_sha256'][:12]}, not the required {required[:12]}; "
            "re-screen the dataset under the required policy"
        )
    if screen != FULL_SCREEN and not allow_unscreened:
        raise LeakageError(
            f"{path} attests the second pass alone ({', '.join(SECOND_PASS_CHECKS)}): its policy digest says "
            f"'{screen}', so no near-copy screen ran over these rows and a prompt differing from a protected one by "
            "a phrase may be among them. Training on them needs an explicit --allow-unscreened-rows "
            "(data.allow_unscreened_rows: true), which the run's provenance records"
        )
    digest = sha256_file(dataset_path)
    if marker["dataset"]["sha256"] != digest:
        raise LeakageError(f"{path} attests a dataset with sha256 {marker['dataset']['sha256']}, not {dataset_path} ({digest})")
    uncovered = sorted(set(required_protected) - _protected_digests(marker))
    if uncovered:
        raise LeakageError(f"{path} was not checked against the required protected file(s) {_preview(uncovered)}")
    unread = sorted(set(required_carved) - _read_digests(marker))
    if unread:
        raise LeakageError(f"{path} was not checked against the required file(s) {_preview(unread)}, not even by the second pass")
    return marker


def attest(train: Path | str, *, protected: Sequence[Path | str], carved: Sequence[Path | str] = ()) -> dict[str, Any]:
    """Attest `train` against `protected` (and read `carved`) by the second pass alone; write and return the marker, which
    a trainer takes only with ``--allow-unscreened-rows``."""
    marker = prompt_disjoint_marker(train, list(protected), policy=ExclusionPolicy(), carved=list(carved),
                                    screen=UNSCREENED)
    write_prompt_disjoint_marker(train, marker)
    return marker
