from __future__ import annotations

from pathlib import Path

import pytest

from ruleforge import env
from ruleforge.core.errors import ArtifactRootError
from ruleforge.env import artifact_root, batch_root, env_value, model_root

NAMES = ("ARTIFACT_ROOT", "MODEL_ROOT", "BATCH_ROOT", "PROBE")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    """No variable from the calling shell."""
    for name in NAMES:
        monkeypatch.delenv(env.PREFIX + name, raising=False)


def test_a_variable_is_read_under_its_prefix_and_no_other_name():
    assert env_value("PROBE", {"RULEFORGE_PROBE": "a"}) == "a"
    assert env_value("PROBE", {"PROBE": "a", "OTHER_PROBE": "b"}) is None


def test_an_empty_value_is_unset():
    assert env_value("PROBE", {"RULEFORGE_PROBE": ""}) is None


def test_the_process_environment_is_read_when_no_mapping_is_given(monkeypatch):
    monkeypatch.setenv("RULEFORGE_PROBE", "from-the-process")
    assert env_value("PROBE") == "from-the-process"


def test_an_unset_artifact_root_says_what_to_set():
    with pytest.raises(ArtifactRootError) as caught:
        artifact_root({})
    assert str(caught.value) == ("RULEFORGE_ARTIFACT_ROOT is unset: point it at the directory `ruleforge data fetch` "
                                 "fills; rendering from results/ needs none")


def test_an_artifact_root_that_is_no_directory_names_the_variable(tmp_path):
    missing = tmp_path / "not-mounted"
    with pytest.raises(ArtifactRootError) as caught:
        artifact_root({"RULEFORGE_ARTIFACT_ROOT": str(missing)})
    assert str(caught.value) == (f"RULEFORGE_ARTIFACT_ROOT={str(missing)!r} is not a directory. On a cluster this "
                                 "usually means the filesystem is not mounted on this node.")


def test_the_artifact_root_has_its_home_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "runs").mkdir()
    assert artifact_root({"RULEFORGE_ARTIFACT_ROOT": "~/runs"}) == tmp_path / "runs"
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    assert artifact_root() == tmp_path


def test_the_artifact_root_s_name_is_the_one_the_paths_module_names():
    from ruleforge.core import paths

    assert env.PREFIX + "ARTIFACT_ROOT" == paths.ARTIFACT_ROOT_ENV
    assert paths.artifact_root is artifact_root


@pytest.mark.parametrize(("reader", "name"), [(model_root, "MODEL_ROOT"), (batch_root, "BATCH_ROOT")])
def test_a_root_is_the_path_given_or_none(reader, name, monkeypatch):
    assert reader({}) is None
    assert reader({env.PREFIX + name: "~/models/"}) == Path("~/models/")
    monkeypatch.setenv(env.PREFIX + name, "/x")
    assert reader() == Path("/x")


def test_an_artifact_uri_resolves_under_the_root(tmp_path):
    from ruleforge.core.paths import resolve, to_uri

    (tmp_path / "sampling" / "x").mkdir(parents=True)
    environ = {"RULEFORGE_ARTIFACT_ROOT": str(tmp_path)}
    assert resolve("artifact://sampling/x", environ) == (tmp_path / "sampling" / "x").resolve()
    assert to_uri(tmp_path / "sampling" / "x", environ) == "artifact://sampling/x"


def test_a_path_outside_the_root_is_refused_naming_the_variable(tmp_path):
    from ruleforge.core.paths import to_uri

    (tmp_path / "root").mkdir()
    with pytest.raises(ValueError, match="is not under RULEFORGE_ARTIFACT_ROOT \\(") as caught:
        to_uri(tmp_path / "elsewhere", {"RULEFORGE_ARTIFACT_ROOT": str(tmp_path / "root")})
    assert str(caught.value) == (f"{(tmp_path / 'elsewhere').resolve()} is not under RULEFORGE_ARTIFACT_ROOT "
                                 f"({tmp_path / 'root'}), so it has no artifact:// form. A committed file must not "
                                 "record it.")


def test_a_campaign_s_model_directory_is_found_under_the_root(monkeypatch, tmp_path):
    from ruleforge.grpo import grpo_data

    (tmp_path / "judge").mkdir()
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path))
    assert grpo_data.campaign_model_dir("judge") == tmp_path / "judge"
    assert grpo_data.campaign_model_dir("absent") is None
