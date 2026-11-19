from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from ruleforge.core.artifacts import link_new
from ruleforge.core.provenance import package_version
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME, load_sampling_manifest
from ruleforge.evaluate.runner import CampaignRunError
from ruleforge.scenarios.manifest import ScenarioManifest

MARKER_FNAME = ".ruleforge-configured"
MARKER_KIND = "ruleforge-configured-campaign"
MARKER_SCHEMA_VERSION = 1
LIBRARY_MARKER_FNAME = ".ruleforge-campaign"


class ConfiguredRootError(CampaignRunError):
    """A root refused as a configured root."""


def mark(root: Union[Path, str], *, campaign_id: str, scenario_manifest_sha256: str) -> bool:
    """Mark ``root`` as a configured root if this writer is the first; return whether this call marked it."""
    location = Path(root)
    if any(os.path.lexists(location / name) for name in (SAMPLING_MANIFEST_FNAME, LIBRARY_MARKER_FNAME)):
        return False
    try:
        link_new(location / MARKER_FNAME, marker_bytes(campaign_id, scenario_manifest_sha256))
    except FileExistsError:
        check_marker(location, campaign_id=campaign_id, scenario_manifest_sha256=scenario_manifest_sha256)
        return False
    return True


def marker_bytes(campaign_id: str, scenario_manifest_sha256: str) -> bytes:
    """Return the marker's bytes for a campaign and population."""
    payload = {"kind": MARKER_KIND, "schema_version": MARKER_SCHEMA_VERSION, "campaign_id": campaign_id,
               "scenario_manifest_sha256": scenario_manifest_sha256, "package_version": package_version()}
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def read_marker(root: Union[Path, str]) -> dict[str, Any]:
    """Return ``root``'s marker, refused when absent, a link, unreadable or of another kind."""
    marker = Path(root) / MARKER_FNAME
    if marker.is_symlink() or not marker.is_file():
        raise ConfiguredRootError(
            f"{Path(root)} holds no {MARKER_FNAME} of its own"
            + (" (a link in its place marks nothing)" if marker.is_symlink() else "")
            + ": a root this version's configured campaign job, shard merge or re-score did not make (every "
            "published root among them) is read and never written by these commands")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConfiguredRootError(f"{marker} cannot be read as a configured root's marker ({error})") from error
    if not isinstance(payload, dict) or payload.get("kind") != MARKER_KIND:
        raise ConfiguredRootError(f"{marker} is not a {MARKER_KIND} marker")
    return payload


def check_marker(root: Union[Path, str], *, campaign_id: str, scenario_manifest_sha256: str) -> dict[str, Any]:
    """Return ``root``'s marker, refused unless it names this campaign and population; writers ask this before
    writing."""
    payload = read_marker(root)
    _check_payload(Path(root), payload, campaign_id, scenario_manifest_sha256)
    return payload


def _check_payload(root: Path, payload: Mapping[str, Any], campaign_id: str, scenario_manifest_sha256: str) -> None:
    if (payload.get("campaign_id"), payload.get("scenario_manifest_sha256")) != (campaign_id,
                                                                                scenario_manifest_sha256):
        raise ConfiguredRootError(
            f"{root / MARKER_FNAME} marks campaign {payload.get('campaign_id')!r} over population "
            f"{str(payload.get('scenario_manifest_sha256'))[:12]}..., not campaign {campaign_id!r} over "
            f"{scenario_manifest_sha256[:12]}...: the root is another campaign's")


def check_root(root: Union[Path, str]) -> dict[str, Any]:
    """Refuse ``root`` unless its marker names the campaign and population its own manifests record; return the
    marker."""
    from ruleforge.evaluate.campaign import SCENARIO_MANIFEST_FNAME

    location = Path(root)
    payload = read_marker(location)
    try:
        manifest = load_sampling_manifest(location / SAMPLING_MANIFEST_FNAME)
        population = ScenarioManifest.load(location / SCENARIO_MANIFEST_FNAME)
    except (OSError, ValueError) as error:
        raise ConfiguredRootError(f"{location} does not hold a whole campaign's records ({error})") from error
    _check_payload(location, payload, str(manifest.get("campaign_id")), population.manifest_sha256)
    return payload


def is_configured(root: Union[Path, str]) -> bool:
    """Return whether ``root`` is, by its markers, a configured root."""
    location = Path(root).expanduser()
    return os.path.lexists(location / MARKER_FNAME) and not os.path.lexists(location / LIBRARY_MARKER_FNAME)


def merges_into(root: Union[Path, str]) -> bool:
    """Return whether a shard merge into ``root`` belongs to ``merge_shards``."""
    location = Path(root).expanduser()
    return not os.path.lexists(location / LIBRARY_MARKER_FNAME) and (
        os.path.lexists(location / MARKER_FNAME) or not os.path.lexists(location / SAMPLING_MANIFEST_FNAME))


def checked_root(root: Union[Path, str]) -> Path:
    """Return ``root`` resolved once it is a configured root a command may write; nothing is written."""
    from ruleforge.evaluate.api import refuse_protected

    refuse_protected(root, "the configured root a command writes", campaign_root=True)
    location = Path(os.path.realpath(Path(root).expanduser()))
    check_root(location)
    return location


def analyze(root: Union[Path, str], *, n_resamples: Optional[int] = None, ci_level: Optional[float] = None,
            seed: Optional[int] = None, n_permutations: Optional[int] = None, allow_incomplete: bool = False,
            allow_stack_mismatch: bool = False, rule_set: Optional[Union[Path, str]] = None,
            exclude_exhausted: Optional[Sequence[str]] = None, replace: bool = False) -> Path:
    """Write a configured root's analysis (``aggregate/estimates.json`` and the gate) under the campaign lock and return
    the resolved root; an existing one is refused unless ``replace``."""
    from ruleforge.evaluate.aggregate import _analyze_campaign, refuse_stripped
    from ruleforge.evaluate.api import _refuse_linked_campaign_lock
    from ruleforge.evaluate.inference import DEFAULT_PERMUTATIONS
    from ruleforge.evaluate.sharding import campaign_lock

    location = checked_root(root)
    refuse_stripped(location)
    _check_analysis_destination(location, replace)
    _refuse_linked_campaign_lock(location)
    with campaign_lock(location):
        _check_analysis_destination(location, replace)
        _analyze_campaign(
            location, n_resamples=n_resamples, ci_level=ci_level, seed=seed,
            n_permutations=DEFAULT_PERMUTATIONS if n_permutations is None else n_permutations, rule_ids=None,
            allow_incomplete_conditions=allow_incomplete, allow_stack_mismatch=allow_stack_mismatch,
            rule_set_path=rule_set, exclude_exhausted=None if exclude_exhausted is None else tuple(exclude_exhausted))
    return location


def _check_analysis_destination(location: Path, replace: bool) -> None:
    from ruleforge.evaluate.aggregate import AGGREGATE_DIRNAME, ESTIMATES_FNAME
    from ruleforge.evaluate.api import AnalysisExists, _destination, _refuse_links
    from ruleforge.evaluate.identity import COMPLETION_FNAME

    target = location / AGGREGATE_DIRNAME / ESTIMATES_FNAME
    if target.exists() and not replace:
        raise AnalysisExists(f"{target} holds this root's analysis; a new analysis replaces it only when asked "
                             "(--replace), and changes the bytes anything built from it pinned")
    for relative in (COMPLETION_FNAME, Path(AGGREGATE_DIRNAME) / ESTIMATES_FNAME):
        _destination(location, relative)
    _refuse_links(location / AGGREGATE_DIRNAME, "the analysis's directory")


def check_merge(root: Union[Path, str], condition_id: str, *, shards_dir: Optional[Union[Path, str]] = None,
                num_shards: Optional[int] = None) -> None:
    """Refuse a merge of a condition's shard roots unless ``root`` and each shard root are configured roots; nothing is
    written."""
    from ruleforge.evaluate.campaign import check_new_condition_id, refuse_retired_root
    from ruleforge.evaluate.sharding import shards_root
    from ruleforge.evaluate.shard_merge import discover_shards

    location = Path(root)
    check_new_condition_id(location, condition_id)
    refuse_retired_root(location)
    made = os.path.lexists(location / SAMPLING_MANIFEST_FNAME) or os.path.lexists(location / LIBRARY_MARKER_FNAME)
    if made:
        check_root(location)
    directory = Path(shards_dir) if shards_dir is not None else shards_root(location) / condition_id
    markers = [check_root(shard.root) for shard in discover_shards(directory, num_shards=num_shards)]
    # A marker left by a writer that stopped before its records. Asked here too, so a refused root gets no lock file.
    if not made and os.path.lexists(location / MARKER_FNAME):
        check_marker(location, campaign_id=str(markers[0].get("campaign_id")),
                     scenario_manifest_sha256=str(markers[0].get("scenario_manifest_sha256")))


def merge_shards(root: Union[Path, str], condition_id: str, *, shards_dir: Optional[Union[Path, str]] = None,
                 num_shards: Optional[int] = None) -> dict[str, Any]:
    """Merge a condition's shard roots into ``root`` under its write lock and campaign lock; return what was merged."""
    from ruleforge.evaluate.api import refuse_protected
    from ruleforge.evaluate.sharding import campaign_lock
    from ruleforge.evaluate.shard_merge import _merge_condition_shards, root_writing

    check_merge(root, condition_id, shards_dir=shards_dir, num_shards=num_shards)
    refuse_protected(root, "the campaign root a shard merge writes", campaign_root=True)
    with root_writing(root), campaign_lock(root):
        check_merge(root, condition_id, shards_dir=shards_dir, num_shards=num_shards)
        return _merge_condition_shards(Path(root), condition_id, shards_dir=shards_dir, num_shards=num_shards)
