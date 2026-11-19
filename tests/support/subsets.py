from __future__ import annotations


def render_manifest(entries: dict[str, str]) -> str:
    """Serialize a subset manifest in ``sha256sum`` format, sorted by path: what ``parse_manifest`` reads."""
    return "".join(f"{digest}  {relative_path}\n" for relative_path, digest in sorted(entries.items()))
