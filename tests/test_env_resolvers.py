from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from ruleforge import env


@pytest.fixture(autouse=True)
def _unset(monkeypatch):
    monkeypatch.delenv("RULEFORGE_THING", raising=False)
    env.register_resolvers()


def test_rf_env_reads_its_variable_then_its_default(monkeypatch):
    config = OmegaConf.create({"plain": "${rf.env:THING}", "default": "${rf.env:THING,null}",
                               "path": "${rf.env:THING}/x"})
    with pytest.raises(InterpolationResolutionError, match="RULEFORGE_THING is unset"):
        config.plain
    assert config.default is None
    monkeypatch.setenv("RULEFORGE_THING", "/new")
    assert config.path == "/new/x" and config.plain == "/new" and config.default == "/new"


def test_rf_pkg_finds_a_packaged_config_and_the_checkout_s_data():
    config = OmegaConf.create({"conf": "${rf.pkg:configs/evaluate.yaml}",
                               "rules": "${rf.pkg:data/rules/modelspec/openai_rules_all.jsonl}"})
    assert config.conf == str(env.PACKAGE_DIR / "configs" / "evaluate.yaml")
    assert config.rules.endswith("data/rules/modelspec/openai_rules_all.jsonl")


@pytest.mark.parametrize("relative", ["", "/etc/passwd", "../outside", "configs/../../x"])
def test_rf_pkg_refuses_a_path_out_of_the_package(relative):
    with pytest.raises(env.PackageResourceError, match="not a path below the package"):
        env.package_path(relative)


def test_rf_pkg_refuses_what_neither_the_package_nor_its_checkout_holds(monkeypatch):
    with pytest.raises(env.PackageResourceError, match="no packaged nothing/here"):
        env.package_path("nothing/here")
    from ruleforge.core import paths
    from ruleforge.core.errors import ArtifactRootError

    def installed():
        raise ArtifactRootError("no checkout")

    monkeypatch.setattr(paths, "repository_root", installed)
    with pytest.raises(env.PackageResourceError, match=r"looked in .*nothing/here\)$"):
        env.package_path("nothing/here")


def test_registering_twice_changes_nothing():
    env.register_resolvers()
    env.register_resolvers()
    assert OmegaConf.has_resolver(env.ENV_RESOLVER) and OmegaConf.has_resolver(env.PACKAGE_RESOLVER)


def test_rf_recorded_reads_a_block_s_recorded_key_from_the_names_table():
    from ruleforge.core.record_names import RECORDED_BLOCK_KEYS

    config = OmegaConf.create({"training": "${rf.recorded:training}", "other": "${rf.recorded:hierarchy}"})
    assert config.training == RECORDED_BLOCK_KEYS["training"] and OmegaConf.has_resolver(env.RECORDED_RESOLVER)
    with pytest.raises(InterpolationResolutionError, match="no experiment block with a recorded key"):
        config.other


#: A fresh process composing the checkout's training campaign config, with or without ``import ruleforge`` first.
COMPOSE = """
import json, sys
if sys.argv[1] == "imported":
    import ruleforge  # noqa: F401 -- registers the resolvers the configs read
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir=sys.argv[2], version_base=None):
    cfg = compose(config_name="sampling_training")
try:
    print(json.dumps({"condition_id": cfg.condition_id}))
except Exception as error:
    print(json.dumps({"error": type(error).__name__}))
"""


def test_a_script_of_one_s_own_composes_the_campaign_configs_after_importing_ruleforge(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from ruleforge.core.record_names import RECORDED_BLOCK_KEYS

    repo = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "RULEFORGE_MODEL_ROOT": str(tmp_path), "RULEFORGE_ARTIFACT_ROOT": str(tmp_path),
                   "PYTHONPATH": os.pathsep.join([str(repo / "src"), *filter(None, [os.environ.get("PYTHONPATH")])])}

    def composed(imported: str) -> dict:
        ran = subprocess.run([sys.executable, "-c", COMPOSE, imported, str(repo / "configs")], cwd=tmp_path,
                             capture_output=True, text=True, env=environment, check=True)
        return json.loads(ran.stdout.splitlines()[-1])

    assert composed("imported") == {"condition_id": f"Qwen3-8B/{RECORDED_BLOCK_KEYS['training']}/base"}
    assert composed("not imported") == {"error": "UnsupportedInterpolationType"}
