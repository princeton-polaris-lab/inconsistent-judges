from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest

from ruleforge.launch import guards
from ruleforge.launch.environment import LaunchError
from ruleforge.core.provenance import code_state
from tests.support.git import make_repo

REPO = Path(__file__).resolve().parents[1]

HEAD = "2403d53a491454d063ab2d86a04e650c2a6a8974"


@pytest.mark.parametrize("value, ok", [
    ("20260926T184512Z-e9d26d01-58a2", True), ("smoke_ok.1", True), ("", False), ("-x", False), ("a b", False),
    ("a/b", False), ("a;b", False), ("x" * 129, False),
])
def test_names_that_reach_file_names_and_shells(value, ok):
    if ok:
        assert guards.check_name(value, what="--name") == value
    else:
        with pytest.raises(LaunchError, match="--name"):
            guards.check_name(value, what="--name")


def test_abbreviated_commits_are_one_commit():
    assert guards.same_commit(HEAD, HEAD[:8]) and guards.same_commit(HEAD[:8], HEAD)
    assert not guards.same_commit(HEAD, "bbbbbbbb") and not guards.same_commit("", HEAD)


def test_the_code_record_is_the_checkouts_code_state(tmp_path):
    repo = make_repo(tmp_path)
    state = code_state(repo)
    assert guards.code_record(repo) == {"commit": state["code_revision"], "dirty": False,
                                        "code_state_sha256": state["code_state_sha256"]}
    (repo / "src" / "pkg" / "a.py").write_text("A = 3\n")
    assert guards.code_record(repo)["dirty"] is True
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(LaunchError, match="not a git checkout"):
        guards.code_record(plain)


def test_the_code_is_refused_at_another_commit_or_dirty():
    clean = {"commit": HEAD, "dirty": False}
    guards.check_code(clean)
    guards.check_code(clean, expect_commit=HEAD[:7])
    with pytest.raises(LaunchError, match="7 to 40 hex digits"):
        guards.check_code(clean, expect_commit="HEAD")
    with pytest.raises(LaunchError, match="not --expect-commit bbbbbbb"):
        guards.check_code(clean, expect_commit="bbbbbbb")
    with pytest.raises(LaunchError, match="uncommitted changes"):
        guards.check_code({**clean, "dirty": True})
    guards.check_code({**clean, "dirty": True}, allow_dirty=True)


@pytest.mark.parametrize("argument", [
    "execution.num_shards=3", "+execution.shard_index=0", "job_root=/x", "hydra.run.dir=x", "campaign_id=c",
    "++campaign_root=r", "~campaign_root", "-m", "--multirun",
    # A dict above a reserved key, which Hydra merges into it, and a multirun without -m.
    "execution={num_shards:3,shard_index:2}", "hydra.run={dir:/x}", "hydra={run:{dir:/y}}", "hydra.mode=MULTIRUN",
    "+hydra.mode=MULTIRUN", "hydra.sweep.dir=/s", "hydra/launcher=submitit_slurm", "hydra/sweeper@hydra.sweeper=x",
    "hydra/launcher@=x", "hydra/run=default",
    # A key below a reserved one, which Hydra writes through, replacing the value with a dict.
    "+campaign_id.x=1", "++campaign_root.x=/evil", "++execution.shard_index.x=1", "+execution.num_shards.x=1",
    "+g@execution.num_shards.x=opt", "+g@campaign_id.x=opt", "+g@hydra.run.dir.x=opt",
    "+x_global_/execution@_group_.num_shards.hydra=opt",
    # A sweep's value, which Hydra runs only as a multirun.
    "campaign_idx=1,2", "+seed=range(1,3)", "training.policy=choice(a,b)", "+x=glob(*)",
    "+g@execution=opt", "+g@_global_.execution=opt", "g@hydra.run=opt", "+g@hydra=opt", "+g@hydra.sweeper=opt",
    "+g@_global_=opt", "+g@foo_global_=opt", "+g@foo._global_.execution=opt", "+g@x_global_.execution=opt",
    "+g@_global__execution=opt", "+g@x._global_.hydra=opt", "+run@hydra._group_=opt", "+g@_here_=opt", "+g@=opt",
    # ... or by a _global_ in the group's own path, and a key or group that is the root itself.
    "+_global_/execution=opt", "+x_global_/execution=opt", "_global_=opt", "=x", "+@execution=opt",
    # What Hydra's parser (argparse) also reads as --multirun: an abbreviation, or a bundle of short flags.
    "--m", "--mu", "--multi", "--multiru", "-mr", "-rm", "-hm", "-mcjob",
    # Any other flag: only the config's are taken.
    "--cfg", "-c", "--run", "--config-n=x", "--info", "--", "-",
])
def test_what_the_launcher_sets_per_job_is_refused_in_the_arguments(argument):
    with pytest.raises(LaunchError, match="no -m"):
        guards.check_overrides(["--config-name", "x", argument])


@pytest.mark.parametrize("argument", [
    "hydra.output_subdir=/published/.hydra", "hydra.output_subdir=../../x", "+hydra.output_subdir=x",
    "hydra.output_subdir=null", "hydra.job_logging.handlers.file.filename=/p/x.log", "hydra.job_logging={}",
    "+hydra.job_logging.handlers.evil={class:logging.FileHandler,filename:/p}", "~hydra.job_logging",
    "hydra.hydra_logging.handlers.file.filename=/p", "hydra.job.env_set.RULEFORGE_ARTIFACT_ROOT=/p",
    "+hydra.job.env_set.X=1",
    # the default log handler's file name, ${hydra.runtime.output_dir}/${hydra.job.name}.log, and Hydra's other keys
    "hydra.job.name=../../../../tmp/published/provenance", "+hydra.job.name=x", "~hydra.job.name",
    "hydra.searchpath=[file:///elsewhere]", "+hydra.callbacks.x._target_=os.system", "hydra.job.env_copy=[HOME]",
    "hydra.overrides.task=[x]", "hydra.runtime.output_dir=/p", "hydra.job.override_dirname=x", "hydra.verbosex=1",
    "hydra.job={chdir:false}", "hydra.job={name:x}",
])
@pytest.mark.parametrize("reserved", [guards.RESERVED_OVERRIDES, ("run_name", "hydra.run.dir")])
def test_a_value_given_to_hydra_s_own_keys_but_the_two_it_takes_is_refused(argument, reserved):
    with pytest.raises(LaunchError, match=r"gives Hydra's '\S+' a value: of Hydra's own keys a job's arguments set "
                                          r"hydra\.verbose and hydra\.job\.chdir only"):
        guards.check_overrides(["--config-name", "x", argument], reserved=reserved)


def test_a_composition_reads_only_the_checkout_s_configs_and_hydra_s(tmp_path, monkeypatch):
    """Hydra joins an option's name to its group's directory as given, so a ``..`` climbs out of ``configs/``."""
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    repo = tmp_path / "repo"
    shutil.copytree(REPO / "configs", repo / "configs")
    (tmp_path / "x.yaml").write_text("version: 1\nhandlers:\n  file:\n    class: logging.FileHandler\n"
                                     "    filename: /published/x.log\nroot:\n  handlers: [file]\n")
    (tmp_path / "glob.yaml").write_text("# @package _global_\nhydra:\n  job:\n    name: ../../../published/x\n")
    for arguments, path in ((["hydra/job_logging=../../../../x"], "hydra/job_logging/../../../../x"),
                            (["hydra/job_logging=[disabled,../../../../x]"], "hydra/job_logging/../../../../x"),
                            (["+model@extra=../../../glob"], "model/../../../glob"),
                            (["--config-name", "../../glob", "campaign_id=c"], "../../glob")):
        with pytest.raises(LaunchError, match=rf"the config {re.escape(repr(path))}( \(the primary config\))? is "
                                              "read from outside configs/ and Hydra's own configs"):
            guards.compose(repo, arguments, default_config="sampling_campaign")
    composed = guards.compose(repo, ["hydra/job_logging=disabled", "+eval_rule_set@extra=./default", "campaign_id=c"],
                              default_config="sampling_campaign")
    assert composed.campaign_id == "c" and "extra" in composed


OUTSIDE_NAMES = [["--config-name", "//abs/dir/escape"], ["--config-name=//abs/dir/escape"], ["-cn", "//abs/x"],
                 ["-cn=//abs/x"], ["--config-name", "/sampling_campaign"], ["-cn", "x", "-cn", "//abs/x"]]
CONFIG_PATH_FLAGS = [["--config-dir", "/d"], ["--config-dir=/d"], ["-cd", "d"], ["-cd=d"], ["--config-path", "/p"],
                     ["--config-path=p"], ["-cp", "p"], ["-cp=/p"], ["--config-dir"]]


@pytest.mark.parametrize("arguments", OUTSIDE_NAMES)
def test_a_config_named_by_a_path_is_refused_before_it_is_read(arguments, tmp_path):
    """Hydra reads a ``--config-name`` led by ``/`` as a path."""
    with pytest.raises(LaunchError, match="Hydra reads a name led by '/' as a path"):
        guards.check_overrides([*arguments, "campaign_id=c"])
    with pytest.raises(LaunchError, match="Hydra reads a name led by '/' as a path"):
        guards.compose(tmp_path, [*arguments, "campaign_id=c"], default_config="sampling_campaign")


@pytest.mark.parametrize("arguments", CONFIG_PATH_FLAGS)
def test_a_config_path_flag_is_refused_before_anything_is_composed(arguments):
    with pytest.raises(LaunchError, match=r"a job's config is one of configs/, named by --config-name"):
        guards.check_overrides(["campaign_id=c", *arguments])


def test_a_default_read_from_an_absolute_path_is_refused(tmp_path, monkeypatch):
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    repo = tmp_path / "repo"
    shutil.copytree(REPO / "configs", repo / "configs")
    escape = tmp_path / "out" / "glob"
    escape.parent.mkdir()
    escape.with_suffix(".yaml").write_text("# @package _global_\nhydra:\n  job:\n    name: escaped\n")
    (repo / "configs" / "escape.yaml").write_text(f"defaults:\n  - /{escape}\n  - _self_\na: 1\n")
    with pytest.raises(LaunchError, match=rf"the config {re.escape(repr(str(escape)))} is read from outside"):
        guards.compose(repo, ["--config-name", "escape"], default_config="sampling_campaign")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        with pytest.raises(LaunchError, match=r"\(the primary config\) is read from outside"):
            guards.check_composed_files(GlobalHydra.instance().config_loader(), f"/{escape}", [])
    GlobalHydra.instance().clear()


def test_an_export_of_a_variable_the_job_s_config_reads_is_refused():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"judge_model": {"batch_root": "${oc.env:RULEFORGE_BATCH_ROOT}"},
                            "runs": ["${ rf.env : OUT_ROOT }/x"], "plain": "PLAIN_OTHER"})
    for env in (["RULEFORGE_BATCH_ROOT=/published"], ["A=1", "RULEFORGE_OUT_ROOT=/p"]):
        with pytest.raises(LaunchError, match=r"--env \S+: a campaign job's config reads it"):
            guards.check_exports(cfg, env, what="a campaign job")
    guards.check_exports(cfg, ["PLAIN_OTHER=1", "OUT_ROOT=2", "MAX_JOBS=8"], what="a campaign job")
    computed = OmegaConf.create({"judge_model": {"batch_root": "${oc.env:${judge_model.name}}", "name": "X"}})
    guards.check_exports(computed, [], what="a campaign job")
    with pytest.raises(LaunchError, match=r"--env A, B: a campaign job's config reads a variable whose name it "
                                          r"computes \('\$\{oc.env:\$\{judge_model.name\}\}'\)"):
        guards.check_exports(computed, ["B=1", "A=2"], what="a campaign job")


def test_an_export_a_config_group_s_choice_reads_is_refused_where_it_changes_the_composition(tmp_path, monkeypatch):
    configs = tmp_path / "repo" / "configs"
    (configs / "g").mkdir(parents=True)
    for option in ("a", "b"):
        (configs / "g" / f"{option}.yaml").write_text(f"x: {option}\n")
    (configs / "main.yaml").write_text("defaults:\n  - g: a\n  - _self_\n")
    (configs / "chosen.yaml").write_text("defaults:\n  - g: ${oc.env:PICK}\n  - _self_\n")
    monkeypatch.setenv("PICK", "a")
    monkeypatch.delenv("UNSET_HERE", raising=False)
    repo = tmp_path / "repo"
    for arguments in (["g=${oc.env:PICK}"], ["g='${oc.env:PICK}'"], ["g=${ oc.env : 'PICK' , a }"],
                      ["--config-name", "chosen"]):
        with pytest.raises(LaunchError, match=r"--env PICK, UNSET_HERE: a campaign job's config chooses its files by "
                                              r"the environment .*: under the exports the job would compose g/b at g "
                                              r"where the launch composed, and checked the places of, g/a at g; "
                                              r"export them"):
            guards.compose(repo, arguments, default_config="main", env=["UNSET_HERE=1", "PICK=b"],
                           what="a campaign job")
        assert os.environ["PICK"] == "a" and "UNSET_HERE" not in os.environ
        with pytest.raises(LaunchError, match=r"--env PICK: a job's config does not compose under the exports, as "
                                              r"the job would compose it \(\w+: .*\), where the launch composed it"):
            guards.compose(repo, arguments, default_config="main", env=["PICK=absent"])
        for env in (["PICK=a"], ["OTHER=1"], []):
            assert guards.compose(repo, arguments, default_config="main", env=env).g.x == "a"
    assert guards.compose(repo, ["g=b"], default_config="main", env=["PICK=b"]).g.x == "b"
    assert os.environ["PICK"] == "a"


@pytest.mark.parametrize(("value", "names", "computed"), [
    ('${oc.env:"NAME"}', {"NAME"}, False),  # quoted: a regex over the text missed it
    ("${ oc.env : 'NAME' , default }", {"NAME"}, False),
    ("a/${oc.select:x,${oc.env:'NESTED'}}/b", {"NESTED"}, False),
    ("${oc.decode:'${oc.env:IN_QUOTES}'}", {"IN_QUOTES"}, True),  # found, and oc.decode parses it again: computed
    ("${oc.env:N\\,X}", {"N,X"}, False),  # escaped, as the resolver is handed it
    ("${rf.env:'OUT_ROOT',null}", {"RULEFORGE_OUT_ROOT"}, False),
    ("${oc.env:A}${oc.env:B}", {"A", "B"}, False),
    ("${oc.env:${key}}", set(), True), ('${oc.env:"A${key}"}', set(), True),  # a name it computes
    ("${${resolver}:NAME}", set(), True), ("${oc.${kind}:NAME}", set(), True),  # a resolver's name it computes
    ('${oc.decode:"\\${oc.env:ESCAPED}"}', set(), True),  # oc.decode reads ESCAPED, which no parse sees
    ("${oc.decode:${oc.env:HOLDS_A_READ}}", {"HOLDS_A_READ"}, True),  # and whatever that variable's value reads
    ("${oc.create:'{a: 1}'}", set(), True),  # a string parsed as config when it resolves: a read only then
    ("${oc.env:}", set(), False), ("${oc.env:null}", set(), False), ("\\${oc.env:ESCAPED}", set(), False),
    ("${oc.env:UNCLOSED", set(), False),  # no value OmegaConf reads: resolving it fails
    ("${other:NAME}", set(), False),
    ("${rf.env:null}", {"RULEFORGE_None"}, False),  # rf.env reads str() of a null name
    ("${rf.env:,fallback}", {"RULEFORGE_"}, False),  # an omitted first argument is handed as ""
    ("${oc.env:,}", {""}, False),  # resolves to its default; read, not a crash
])
def test_a_config_value_s_environment_reads_are_found_by_omegaconf_s_own_grammar(value, names, computed):
    assert guards.env_reads(value) == (names, computed)


def test_hydra_resolves_the_package_the_guard_reads():
    assert guards.override_keys("+g@_global_.execution=opt") == ["g", "execution"]
    assert guards.override_keys("+g@x._global_.execution=opt") == ["g", "execution"]
    assert guards.override_keys("+run@hydra._group_=opt") == ["run", "hydra.run"]
    assert guards.override_keys("+g@_global_=opt") == ["g", ""]
    assert guards.override_keys("hydra/launcher=x") == ["hydra.launcher"]
    assert guards.override_keys("~campaign_root") == ["campaign_root"]
    assert guards.override_keys("=x") == [""]
    # Appended by a config in a directory, a group's option is merged below that directory's package.
    assert guards.override_keys("+g=opt", config_dir="sub") == ["g", "sub.g"]
    assert guards.override_keys("+g@_group_=opt", config_dir="sub") == ["g", "sub.g"]
    assert guards.placement("sub") == "sub" and guards.placement("x_global_/hydra") == "hydra"


@pytest.mark.parametrize("arguments", [
    ["--config-name", "hydra/x"], ["-cn=hydra/run/x"], ["--config-name=execution/x"], ["-cn", "a", "-cn", "hydra/x"],
    ["--config-name", "x_global_/hydra/x"], ["--config-name", "campaign_id/x/cfg"],
])
def test_a_config_whose_directory_places_it_at_a_reserved_key_is_refused(arguments):
    with pytest.raises(LaunchError, match="Hydra merges a config in"):
        guards.check_overrides(arguments)


def test_an_argument_hydra_cannot_read_is_refused():
    for argument in ("a b=1", "foo", "/d"):
        with pytest.raises(LaunchError, match="not an override Hydra reads"):
            guards.check_overrides([argument])


def test_other_arguments_pass():
    arguments = ["--config-name", "sampling_training", "campaign_idx=1", "training.policy=grpo", "judge_model.tp_size=2",
                 "hydra.job.chdir=false", "hydra.job.chdir=true", "hydra.verbose=true", "+hydra.verbose=[a,b]",
                 "execution.batch_size=8", "campaign_idx_extra=1", "-cn=y", "x=[1,2]", "hydra/job_logging=disabled",
                 "+g@training=opt", "+g@_group_=opt", "+g@training._group_=opt", "+g@global_x=opt",
                 "--config-name=training/grpo", "+g=opt"]
    assert guards.check_overrides(arguments) == arguments
    # A config flag's value is the next argument only when that is not dash-led: argparse's reading.
    with pytest.raises(LaunchError, match="Hydra flags are"):
        guards.check_overrides(["--config-name", "-m"])


def test_a_root_its_jobs_spell_through_a_symlink_is_refused(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    guards.check_spelled(tmp_path / "real", tmp_path / "real", what="the root")
    guards.check_spelled(f"{tmp_path}/x/../real", tmp_path / "real", what="the root")
    with pytest.raises(LaunchError, match=r"the root \S*/alias, as its jobs spell it, is not \S*/real, where the "
                                          "launch's record and a pull name it: it is reached through a symlink"):
        guards.check_spelled(tmp_path / "alias", tmp_path / "real", what="the root")


def test_the_way_to_a_directory_a_job_makes_holds_no_symlink(tmp_path):
    root = tmp_path / "root"
    (root / "hydra" / "m" / "x").mkdir(parents=True)
    guards.refuse_linked_way(root, root / "hydra" / "m" / "x", what="a Hydra directory")
    guards.refuse_linked_way(root, root / "hydra" / "preflight" / "m", what="a Hydra directory")
    (root / "hydra" / "preflight").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(LaunchError, match=r"a Hydra directory \S*/root/hydra/preflight/m: \S*/hydra/preflight, on its "
                                          r"way, is a symlink: a write would follow it to \S*/elsewhere"):
        guards.refuse_linked_way(root, root / "hydra" / "preflight" / "m", what="a Hydra directory")
    assert not (tmp_path / "elsewhere").exists()


def test_below_a_writer_s_own_subtree_no_symlink_is_left(tmp_path):
    root = tmp_path / "root"
    hydra = root / "hydra" / "m" / "x"
    (hydra / "2026-01-01_00-00-00").mkdir(parents=True)
    (hydra / "2026-01-01_00-00-00" / "run.log").write_text("x")
    guards.refuse_linked_way(root, hydra, what="a Hydra directory", below=True)
    guards.refuse_linked_way(root, root / "hydra" / "n", what="a Hydra directory", below=True)
    for name, directory in ((".hydra", True), ("run.log.1", False)):
        link = hydra / "2026-01-01_00-00-00" / name
        link.symlink_to(tmp_path / "published", target_is_directory=directory)
        guards.refuse_linked_way(root, hydra, what="a Hydra directory")
        with pytest.raises(LaunchError, match=rf"a Hydra directory \S*/hydra/m/x: \S*/2026-01-01_00-00-00/"
                                              rf"{re.escape(name)}, below it, is a symlink: a write there would "
                                              r"follow it to \S*/published"):
            guards.refuse_linked_way(root, hydra, what="a Hydra directory", below=True)
        link.unlink()
    assert not (tmp_path / "published").exists()


def test_below_a_writer_s_own_subtree_an_entry_another_job_removes_mid_walk_is_no_refusal(tmp_path, monkeypatch):
    from tests.fake_scandir import vanishing_scandir

    root = tmp_path / "root"
    snapshots = root / "code_snapshots"
    (snapshots / "gone").mkdir(parents=True)
    (snapshots / "abc.json.x.tmp").write_text("x")
    monkeypatch.setattr("os.scandir", vanishing_scandir({"abc.json.x.tmp"}, {snapshots / "gone"}))
    guards.refuse_linked_way(root, snapshots, what="the code snapshots", below=True, roots=True)
    (snapshots / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(LaunchError, match=r"code_snapshots/linked, below it, is a symlink"):
        guards.refuse_linked_way(root, snapshots, what="the code snapshots", below=True)


def test_below_a_writer_s_own_subtree_a_sibling_job_writing_it_never_makes_the_walk_refuse(tmp_path):
    from tests.fake_scandir import concurrent_writer

    root = tmp_path / "root"
    draw = root / "m" / "base" / "draw-000"
    draw.mkdir(parents=True)
    with concurrent_writer(draw) as rounds:
        for _ in range(300):
            guards.refuse_linked_way(root, root / "m" / "base", what="a directory m/base's jobs write", below=True,
                                     roots=True)
        assert rounds[0] > 0
    (draw / "outputs").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(LaunchError, match=r"draw-000/outputs, below it, is a symlink"):
        guards.refuse_linked_way(root, root / "m" / "base", what="a directory m/base's jobs write", below=True)


def test_below_a_writer_s_own_subtree_a_directory_that_cannot_be_listed_is_refused(tmp_path):
    root = tmp_path / "root"
    hidden = root / "hydra" / "m" / "2026-01-01_00-00-00"
    hidden.mkdir(parents=True)
    (hidden / ".hydra").symlink_to(tmp_path / "published", target_is_directory=True)
    hidden.chmod(0o300)
    try:
        with pytest.raises(LaunchError, match=r"a Hydra directory \S*/hydra/m cannot be walked whole \(\[Errno 13\]"):
            guards.refuse_linked_way(root, root / "hydra" / "m", what="a Hydra directory", below=True)
    finally:
        hidden.chmod(0o700)
    (root / "file").write_text("x")
    guards.refuse_linked_way(root, root / "file", what="a Hydra directory", below=True, roots=True)
    assert not (tmp_path / "published").exists()


def test_the_variable_whose_absence_failed_a_resolution_is_named_by_its_resolver(monkeypatch):
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    from ruleforge.env import EnvUnsetError, register_resolvers

    register_resolvers()
    for name in ("RF_TEST_UNSET_ROOT", "RULEFORGE_TEST_UNSET_ROOT"):
        monkeypatch.delenv(name, raising=False)

    def failure(value: str) -> BaseException:
        cfg = OmegaConf.create({"x": value, "via": "${oc.env:RF_TEST_UNSET_ROOT}"})
        with pytest.raises(OmegaConfBaseException) as raised:
            OmegaConf.select(cfg, "x")
        return raised.value

    for value, named in (("${oc.env:RF_TEST_UNSET_ROOT}", "RF_TEST_UNSET_ROOT"), ("${via}/b", "RF_TEST_UNSET_ROOT"),
                         ("${rf.env:TEST_UNSET_ROOT}", "RULEFORGE_TEST_UNSET_ROOT"),
                         ("${hydra:runtime.cwd}/b", None), ("${nothere}", None)):
        assert guards.unset_variable(failure(value)) == named, value
    first, second = ValueError("a"), ValueError("b")
    first.__cause__, second.__cause__ = second, first
    for error in (KeyError(), KeyError(1), KeyError("another"), EnvUnsetError("another form"), first):
        assert guards.unset_variable(error) is None


@pytest.mark.parametrize(("config", "confirmed"), [({}, False), ({"confirm_spend": False}, False), ({"confirm_spend": True}, True)],
                         ids=["unset", "false", "true"])
def test_a_launched_campaign_confirms_its_spend_by_its_override(monkeypatch, config, confirmed):
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from ruleforge.evaluate.runner import STATE_COMPLETE
    from ruleforge.judges import batch_api
    from ruleforge.launch.programs import run_campaign

    seen = []

    def run_plan(*_args, **_kwargs):
        seen.append(batch_api.spend_is_confirmed())
        return {0: {"state": STATE_COMPLETE, "pending": 0}}

    hydra = SimpleNamespace(get=lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir="out")))
    plan = SimpleNamespace(condition=SimpleNamespace(condition_id="c"), config=SimpleNamespace(campaign_mode="final"))
    for name, value in (("HydraConfig", hydra), ("require_checkout", lambda _repo: None),
                        ("ensure_job_logging", lambda **_kwargs: None), ("preflight", lambda _cfg: None),
                        ("execution_settings", lambda _cfg: None), ("build_task_spec", lambda _cfg: None),
                        ("ModelBackend", lambda _judge: SimpleNamespace(close=lambda: None)),
                        ("build_plan", lambda *_args, **_kwargs: plan), ("run_plan", run_plan)):
        monkeypatch.setattr(run_campaign, name, value)
    run_campaign.run(OmegaConf.create({"judge_model": {}, **config}))
    assert seen == [confirmed] and not batch_api.spend_is_confirmed()
    with pytest.raises(batch_api.BatchSpendNotConfirmed, match="confirm_spend is true or false, got 'false'"):
        run_campaign.run(OmegaConf.create({"judge_model": {}, "confirm_spend": "false"}))
    assert len(seen) == 1
