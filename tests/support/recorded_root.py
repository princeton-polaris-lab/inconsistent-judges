from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ruleforge.core.record_names import identity_key
from ruleforge.evaluate.campaign import CONDITION_FNAME, condition_payload, load_conditions
from ruleforge.evaluate.identity import ConditionSpec

#: The files :func:`record` rewrites, by suffix: one JSON document, or one per line.
JSON_SUFFIXES = (".json",)
JSONL_SUFFIXES = (".jsonl",)
ID_BOUNDARIES = "/:+"


def recorded_style(spec: ConditionSpec) -> str:
    """The key the records name ``spec``'s style by."""
    return identity_key(spec.prompt_style, spec.task)


def recorded_payload(spec: ConditionSpec) -> dict[str, Any]:
    """The ``condition.json`` payload the records hold for ``spec``: identity id, recorded style key, fingerprint."""
    return {**condition_payload(spec), "condition_id": spec.identity_id, "prompt_style": recorded_style(spec)}


def recorded_condition(payload: Mapping[str, Any]) -> ConditionSpec:
    """The condition a recorded ``condition.json`` payload registers, read as the campaign loader reads one."""
    return ConditionSpec.from_dict({key: value for key, value in payload.items()
                                    if key not in ("schema_version", "fingerprint")})


@dataclass(frozen=True)
class Recording:
    """What :func:`record` changed."""

    ids: Mapping[str, str]
    styles: Mapping[str, str]


def record(root: Path) -> Recording:
    """Rewrite the campaign root ``root``, written under the public names, in place into the records' form."""
    root = Path(root)
    conditions = load_conditions(root)
    ids = {cid: spec.identity_id for cid, spec in conditions.items() if spec.identity_id != cid}
    styles: dict[str, str] = {}
    for spec in conditions.values():
        key = recorded_style(spec)
        if styles.get(spec.prompt_style, key) != key:
            raise ValueError(f"{root}: style {spec.prompt_style!r} is recorded both as {styles[spec.prompt_style]!r} "
                             f"and as {key!r}; a recorded root holds one key per style name")
        styles[spec.prompt_style] = key
    styles = {style: key for style, key in styles.items() if style != key}
    mapper = _Mapper(ids, styles)
    rewrites = [(path, mapper.file(path)) for path in sorted(root.rglob("*"))
                if path.is_file() and path.suffix in JSON_SUFFIXES + JSONL_SUFFIXES]
    for path, text in rewrites:
        if text is not None:
            path.write_text(text, encoding="utf-8")
    _rename_directories(root, ids)
    for spec in conditions.values():
        written = json.loads((root / spec.identity_id / CONDITION_FNAME).read_text(encoding="utf-8"))
        assert written == recorded_payload(spec), spec.condition_id
    return Recording(ids=dict(ids), styles=styles)


def _rename_directories(root: Path, ids: Mapping[str, str]) -> None:
    for public, recorded in ids.items():
        tail = tuple(public.split("/"))
        found = [path for path in root.rglob("*") if path.is_dir()
                 and path.relative_to(root).parts[-len(tail):] == tail]
        for path in sorted(found, key=lambda path: len(path.parts), reverse=True):
            base = path
            for _ in tail:
                base = base.parent
            destination = base.joinpath(*recorded.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.rename(destination)
            emptied = path.parent
            while emptied != base and not any(emptied.iterdir()):
                emptied.rmdir()
                emptied = emptied.parent


class _Mapper:
    def __init__(self, ids: Mapping[str, str], styles: Mapping[str, str]) -> None:
        self.ids = dict(ids)
        self.styles = dict(styles)
        self.pattern: Optional[re.Pattern[str]] = None
        if ids:
            alternatives = "|".join(re.escape(cid) for cid in sorted(ids, key=len, reverse=True))
            bound = re.escape(ID_BOUNDARIES)
            self.pattern = re.compile(rf"(?<![^{bound}])(?:{alternatives})(?![^{bound}])")

    def string(self, text: str) -> str:
        """``text`` with each id it holds as whole segments replaced."""
        if self.pattern is None:
            return text
        mapped = self.pattern.sub(lambda match: self.ids[match.group(0)], text)
        if mapped != text and "+" in text:
            parts = text.split("+")
            if parts == sorted(parts):
                mapped = "+".join(sorted(mapped.split("+")))
        return mapped

    def value(self, value: Any, key: Optional[str] = None) -> Any:
        """``value`` rewritten (``key``: the object key it is held under)."""
        if isinstance(value, dict):
            keys = list(value)
            items = [(self.string(name), self.value(item, name)) for name, item in value.items()]
            new_keys = [name for name, _ in items]
            if len(set(new_keys)) != len(new_keys):
                raise ValueError(f"two keys of one object are recorded as one: {keys}")
            if new_keys != keys and keys == sorted(keys):
                items.sort(key=lambda item: item[0])
            return dict(items)
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, str):
            if key == "prompt_style":
                return self.styles.get(value, value)
            return self.string(value)
        return value

    def file(self, path: Path) -> Optional[str]:
        """The rewritten text of the JSON or JSON Lines file ``path``, or None when nothing in it changes."""
        text = path.read_text(encoding="utf-8")
        if path.suffix in JSON_SUFFIXES:
            documents = [text]
        else:
            documents = text.splitlines(keepends=True)
        out = []
        for document in documents:
            body = document.rstrip("\n")
            if not body.strip():
                out.append(document)
                continue
            parsed = json.loads(body)
            encode = _encoder(parsed, body, path)
            out.append(encode(self.value(parsed)) + document[len(body):])
        rewritten = "".join(out)
        return None if rewritten == text else rewritten


_ENCODINGS = tuple(
    {"indent": indent, "sort_keys": sort_keys, "ensure_ascii": ensure_ascii,
     **({"separators": separators} if separators else {})}
    for indent in (2, None) for sort_keys in (False, True) for ensure_ascii in (True, False)
    for separators in (None, (",", ":"))
    if not (indent and separators)
)


def _encoder(parsed: Any, body: str, path: Path) -> Callable[[Any], str]:
    """The ``json.dumps`` encoding that writes ``parsed`` as ``body`` exactly (:data:`_ENCODINGS`)."""
    for arguments in _ENCODINGS:
        if json.dumps(parsed, **arguments) == body:
            return lambda value, arguments=arguments: json.dumps(value, **arguments)
    raise ValueError(f"{path}: no encoding of record()'s writes its text back byte for byte; a synthetic recorded "
                     "root is written by today's writers")
