from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

if __package__ in (None, ""):  # pragma: no cover - run as a file rather than with -m
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.identity import published as pub

DESCRIPTORS = pub.FIXTURES / "judges.json"
#: The judge fields a published condition is matched to its descriptor on.
KEY_FIELDS = ("model_name", "model_type", "model_path", "serving_dtype", "model_weights_sha256")
#: The fields of a merge manifest the recorded weights digest is read from.
MANIFEST_FIELDS = ("weights_digest", "kind")


#: The variable the recorded configs name their weights root by.
RECORDED_ROOT_VARIABLE = "SMRC_MODEL_ROOT"
CONTAINER_PREFIXES = {"/smrc/models/": "RULEFORGE_MODEL_ROOT", "/dev/shm/smrcvol/smrc/models/": "RULEFORGE_MODEL_ROOT"}
#: Where a descriptor taken from a committed model config names its source (relative to the repository).
MODEL_CONFIGS = "configs/model/"
WEIGHTS_DIRS_ENV = "RULEFORGE_WEIGHTS_DIRS"


def engine_args(artifact_root: Path, root: str, condition_id: str) -> Optional[dict[str, Any]]:
    """The engine arguments a published local condition's stack records."""
    stack = artifact_root / "sampling" / root / condition_id / "condition_stack.json"
    if not stack.is_file():
        return None
    return json.loads(stack.read_text(encoding="utf-8")).get("runtime", {}).get("engine_args") or None


def engine_path(artifact_root: Path, root: str, condition_id: str) -> Optional[str]:
    """The weights directory a published local condition's engine started on, as its stack records it, or None."""
    return (engine_args(artifact_root, root, condition_id) or {}).get("model")


def live_weights(artifact_root: Path, root: str, payload: Mapping[str, Any]) -> Optional[Path]:
    """The weights directory a published local condition was judged on, found on this machine, or None."""
    recorded = engine_path(artifact_root, root, payload["condition_id"])
    if recorded is not None:
        return locate_weights(recorded, payload.get("model_weights_sha256"))
    upstream = Path(os.environ.get("RULEFORGE_MODEL_ROOT", "")) / str(payload["model_path"])
    return upstream if payload.get("model_weights_sha256") is None and is_weights_dir(upstream) else None


def is_weights_dir(path: Path) -> bool:
    """Whether ``path`` holds weights an engine starts on (its ``config.json``), not only a record of them."""
    return (path / "config.json").is_file()


def _manifest_digest(directory: Path) -> Optional[str]:
    manifest = directory / "merge_manifest.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text(encoding="utf-8"))["result"]["weights_sha256"]


def locate_weights(recorded: str, digest: Optional[str]) -> Optional[Path]:
    """A recorded weights directory on this machine."""
    path = Path(recorded)
    if is_weights_dir(path):
        return path
    for prefix, variable in CONTAINER_PREFIXES.items():
        if recorded.startswith(prefix) and os.environ.get(variable):
            mapped = Path(os.environ[variable]) / recorded[len(prefix):]
            if is_weights_dir(mapped):
                return mapped
    for entry in os.environ.get(WEIGHTS_DIRS_ENV, "").split(os.pathsep):
        candidate = Path(entry) if entry else None
        if (candidate is not None and candidate.name == path.name and is_weights_dir(candidate)
                and _manifest_digest(candidate) == digest):
            return candidate
    return None


def judge_key(payload: Mapping[str, Any]) -> tuple:
    """The judge a condition payload names, as a descriptor's ``key`` does."""
    return tuple(payload.get(field) for field in KEY_FIELDS)


def reduced_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """What ``recorded_model_weights`` reads of a merge manifest, and nothing that names a machine's path."""
    merge = manifest.get("merge") if isinstance(manifest.get("merge"), Mapping) else {}
    return {**{field: manifest.get(field) for field in MANIFEST_FIELDS},
            "merge": {"saved_dtype": merge.get("saved_dtype")},
            "result": {"weights_sha256": manifest["result"]["weights_sha256"]}}


def run_root(root: str) -> str:
    """The root a published root's runs are recorded in: itself, or the root a re-scored copy was replayed from."""
    return pub.RESCORED_FROM.get(root, root)


def _recorded_configs(artifact_root: Path) -> Iterable[tuple[str, Any]]:
    from omegaconf import OmegaConf

    for root in pub.ROOTS:
        for path in sorted((artifact_root / "sampling" / run_root(root)).glob("hydra/**/.hydra/config.yaml")):
            yield str(path.relative_to(artifact_root)), OmegaConf.load(path)


def _model_configs() -> Iterable[tuple[str, Any]]:
    from omegaconf import OmegaConf

    for path in sorted((pub.REPO_ROOT / MODEL_CONFIGS).glob("*.yaml")):
        yield f"{MODEL_CONFIGS}{path.name}", OmegaConf.create({"judge_model": OmegaConf.load(path)})


def recorded_config(artifact_root: Path, source: str) -> Optional[Any]:
    """A descriptor's source as a run's recorded config holds it."""
    from omegaconf import OmegaConf

    if source.startswith(MODEL_CONFIGS):
        path = pub.REPO_ROOT / source
        return OmegaConf.create({"judge_model": OmegaConf.load(path)}) if path.is_file() else None
    path = artifact_root / source
    return OmegaConf.load(path) if path.is_file() else None


def model_config_problems(artifact_root: Path, descriptor: Mapping[str, Any],
                          conditions: Iterable[Mapping[str, Any]]) -> list[str]:
    """Where a descriptor read from a committed model config departs from its judge's recorded engine arguments."""
    from omegaconf import OmegaConf

    from ruleforge.judges.backend import local_engine_settings
    from ruleforge.judges.models import local_engine_kwargs

    if not descriptor["source"].startswith(MODEL_CONFIGS):
        return []
    cfg = OmegaConf.load(pub.REPO_ROOT / descriptor["source"])
    built = local_engine_kwargs(**local_engine_settings(cfg, engine_fields=()))
    built.pop("model")
    problems, checked = [], 0
    for row in conditions:
        payload = row["payload"]
        if judge_key(payload) != judge_key(descriptor["key"]):
            continue
        checked += 1
        where = f"{row['root']}/{payload['condition_id']}"
        engine = engine_args(artifact_root, row["root"], payload["condition_id"])
        if engine is None:
            problems.append(f"{where}: no engine arguments recorded")
            continue
        recorded = {name: value for name, value in engine.items() if name != "model"}
        differing = sorted(name for name in set(recorded) | set(built) if recorded.get(name) != built.get(name))
        if differing:
            problems.append(f"{where}: engine arguments {differing} are not what {descriptor['source']} builds")
    if not checked:
        problems.append(f"{descriptor['source']}: no published condition names its judge")
    return problems


def _live_weights(artifact_root: Path, source: str, cfg: Any, options: Any, digests: Mapping) -> Optional[Path]:
    from omegaconf.errors import OmegaConfBaseException

    try:
        configured = Path(str(options.model_path))
    except OmegaConfBaseException:  # a weights directory this machine's environment does not name
        configured = None
    if configured is not None and is_weights_dir(configured):
        return configured
    try:
        condition_id = str(cfg.condition_id)
    except OmegaConfBaseException:
        return None
    root = Path(source).parts[1]
    recorded = engine_path(artifact_root, root, condition_id)
    return None if recorded is None else locate_weights(recorded, digests.get((root, condition_id)))


def descriptor_of(artifact_root: Path, source: str, cfg: Any, digests: Mapping) -> Optional[dict[str, Any]]:
    """A descriptor from one recorded campaign config, or None when its judge's weights are not found here."""
    from ruleforge.judges import JudgeOptions, JudgeSpec

    options = JudgeOptions(cfg.judge_model)
    written = options.written()
    files: dict[str, str] = {}
    if written.get("model_type") == "local":
        live = _live_weights(artifact_root, source, cfg, options, digests)
        if live is None:
            return None
        options = options.replace(model_path=str(live))
        written["model_path"] = "${oc.env:RULEFORGE_MODEL_ROOT}/" + live.name
        generation = live / "generation_config.json"
        if generation.is_file():
            files["generation_config.json"] = generation.read_text(encoding="utf-8")
        manifest = live / "merge_manifest.json"
        if manifest.is_file():
            files["merge_manifest.json"] = json.dumps(
                reduced_manifest(json.loads(manifest.read_text(encoding="utf-8"))), indent=2, sort_keys=True) + "\n"
    if "batch_root" in written:
        written["batch_root"] = "${oc.env:RULEFORGE_BATCH_ROOT}"
    identity = JudgeSpec.from_config(options).identity(verify_weights=False)
    key = {"model_name": identity.model_name, "model_type": identity.model_type, "model_path": identity.model_path,
           "serving_dtype": identity.serving_dtype, "model_weights_sha256": identity.model_weights_sha256}
    return {"key": key, "config": written, "files": files, "source": source}


def recorded_digests(conditions: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Optional[str]]:
    """Each published condition's recorded weights digest, by ``(root, condition id)``."""
    return {(run_root(row["root"]), row["payload"]["condition_id"]): row["payload"].get("model_weights_sha256")
            for row in conditions}


def _preferred(descriptor: Mapping[str, Any]) -> bool:
    return f"/hydra/{descriptor['key']['model_name']}/" in descriptor["source"]


def build(artifact_root: Path, committed: Iterable[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """One descriptor per judge the published conditions name."""
    conditions = pub.rows_of(pub.read_fixture(pub.CONDITIONS), "condition")
    wanted = {judge_key(row["payload"]) for row in conditions}
    digests = recorded_digests(conditions)
    found: dict[tuple, dict[str, Any]] = {judge_key(descriptor["key"]): dict(descriptor) for descriptor in committed
                                          if judge_key(descriptor["key"]) in wanted}
    kept = set(found)
    for source, cfg in _recorded_configs(artifact_root):
        descriptor = descriptor_of(artifact_root, source, cfg, digests)
        if descriptor is None:
            continue
        key = judge_key(descriptor["key"])
        if key in wanted and key not in kept and (
                key not in found or (_preferred(descriptor) and not _preferred(found[key]))):
            found[key] = descriptor
    unnamed = {key[0] for key in wanted - set(found)}
    for source, cfg in _model_configs():
        if cfg.judge_model.get("model_name") not in unnamed:
            continue
        descriptor = descriptor_of(artifact_root, source, cfg, digests)
        if descriptor is not None and judge_key(descriptor["key"]) in wanted - set(found):
            problems = model_config_problems(artifact_root, descriptor, conditions)
            if problems:
                raise SystemExit("\n".join(problems))
            found[judge_key(descriptor["key"])] = descriptor
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"no recorded config names the judges {sorted(map(str, missing))}")
    return [found[key] for key in sorted(found, key=str)]


def check(artifact_root: Path) -> list[str]:
    """Every descriptor its source does not give, and every departure of a model config's from the records."""
    problems = []
    conditions = pub.rows_of(pub.read_fixture(pub.CONDITIONS), "condition")
    digests = recorded_digests(conditions)
    for descriptor in json.loads(DESCRIPTORS.read_text(encoding="utf-8")):
        cfg = recorded_config(artifact_root, descriptor["source"])
        again = descriptor_of(artifact_root, descriptor["source"], cfg, digests) if cfg is not None else None
        if again != descriptor:
            problems.append(f"{descriptor['source']}: not the descriptor of {descriptor['key']}")
        problems += model_config_problems(artifact_root, descriptor, conditions)
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    from ruleforge.env import artifact_root

    parser = argparse.ArgumentParser(
        description="Report each judge a published condition names and what its identity is read from.")
    parser.add_argument("--check", action="store_true", help="compare the committed descriptors with their sources")
    args = parser.parse_args(argv)
    if "RULEFORGE_MODEL_ROOT" not in os.environ:
        raise SystemExit("set RULEFORGE_MODEL_ROOT: the recorded configs name their weights under it")
    os.environ.setdefault(RECORDED_ROOT_VARIABLE, os.environ["RULEFORGE_MODEL_ROOT"])
    root = artifact_root()
    if args.check:
        problems = check(root)
        print("\n".join(problems) or "descriptors match their sources")
        return 1 if problems else 0
    committed = json.loads(DESCRIPTORS.read_text(encoding="utf-8")) if DESCRIPTORS.is_file() else []
    descriptors = build(root, committed)
    DESCRIPTORS.write_text(json.dumps(descriptors, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{DESCRIPTORS.name}: {len(descriptors)} judges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
