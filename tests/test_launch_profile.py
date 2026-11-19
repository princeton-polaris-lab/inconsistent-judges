from __future__ import annotations

from pathlib import Path

import pytest

from ruleforge.cli import slurm
from ruleforge.env import package_path
from ruleforge.launch import profile as profiles
from ruleforge.launch.environment import LaunchError

PROFILE = """partition: p
account: null
qos: {gpu: g}
gpus_per_node: 2
defaults: {timeout_min: 30, cpus: 2, mem_gb: 8, mem_gb_per_gpu: 16}
setup: []
"""


def test_the_cli_reads_the_launchers_profiles():
    assert slurm.Profile is profiles.Profile and slurm.find_profile is profiles.find_profile
    assert slurm.profile_dirs is profiles.profile_dirs and slurm.CLUSTER_DIR == profiles.CLUSTER_DIR


def test_a_profile_s_exclude_is_optional_and_a_list_of_node_names(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(PROFILE)
    plain = profiles.Profile.load(path)
    assert plain.exclude == () and plain.site() == {"partition": "p", "account": None, "signal_delay_s": None,
                                                    "exclude": []}
    assert (plain.qos(1), plain.qos(0)) == ("g", None)
    plain.check_gpus("j", 2)
    with pytest.raises(LaunchError, match="j asks for 3 GPUs; a node of site has 2"):
        plain.check_gpus("j", 3)
    path.write_text(PROFILE + "exclude: [node-a1, node-b2]\n")
    assert profiles.Profile.load(path).exclude == ("node-a1", "node-b2")
    for bad in ("exclude: node-a1\n", "exclude: [1]\n", "exclude: ['']\n"):
        path.write_text(PROFILE + bad)
        with pytest.raises(LaunchError, match="(?s)not a cluster profile .*exclude is a list of node names"):
            profiles.Profile.load(path)


def test_the_packaged_example_is_a_profile_with_no_excluded_node():
    example = profiles.Profile.load(package_path(profiles.EXAMPLE))
    assert example.partition == "gpu" and example.exclude == () and example.setup == ("export VLLM_NO_USAGE_STATS=1",)


def test_a_name_is_looked_up_in_the_profile_dirs_and_a_file_by_its_path(tmp_path):
    (tmp_path / "site.yaml").write_text(PROFILE)
    assert profiles.find_profile("site", tmp_path).path == tmp_path / "site.yaml"
    assert profiles.find_profile(str(tmp_path / "site.yaml")).partition == "p"
    with pytest.raises(LaunchError, match="no such profile file"):
        profiles.find_profile(str(tmp_path / "absent.yml"))
    assert profiles.profile_dirs(Path("/x"))[0] == Path("/x")


def test_a_profile_named_by_a_relative_path_is_kept_by_its_absolute_one(tmp_path, monkeypatch):
    (tmp_path / "site.yaml").write_text(PROFILE)
    monkeypatch.chdir(tmp_path)
    assert profiles.find_profile("site.yaml").path == tmp_path / "site.yaml"
    assert profiles.find_profile("site", Path(".")).path == tmp_path / "site.yaml"
