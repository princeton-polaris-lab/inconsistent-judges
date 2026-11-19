from __future__ import annotations

import json
import logging
import os
import re
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ruleforge.cli import records


def test_a_record_is_made_beside_its_target_and_two_in_one_second_stay_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "_now", lambda: "20261020T000000Z")
    target = tmp_path / "runs" / "root"
    first = records.open_record(target, "campaign run", ["campaign", "run"], None)
    second = records.open_record(target, "campaign run", ["campaign", "run"], OmegaConf.create({"a": 1}))
    assert first == tmp_path / "runs" / "root.runs" / "20261020T000000Z-campaign-run"
    assert second.name == "20261020T000000Z-campaign-run-1" and not target.exists()
    assert json.loads((first / records.ARGV_FNAME).read_text())["argv"] == ["campaign", "run"]
    assert (first / records.CONFIG_FNAME).read_text().startswith("# no config")
    assert (second / records.CONFIG_FNAME).read_text() == "a: 1\n"


def test_a_config_that_does_not_resolve_here_is_kept_as_written():
    text = records.config_text(OmegaConf.create({"a": "${oc.env:SURELY_UNSET_VARIABLE_X}"}))
    assert text.startswith("# unresolved (InterpolationResolutionError") and "${oc.env:SURELY_UNSET_VARIABLE_X}" in text


def test_logging_goes_to_the_record_for_the_block_alone(tmp_path):
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    with records.logging_to(tmp_path, verbose=True):
        logging.getLogger("ruleforge.test").debug("inside")
    logging.getLogger("ruleforge.test").warning("outside")
    assert "inside" in (tmp_path / records.LOG_FNAME).read_text()
    assert "outside" not in (tmp_path / records.LOG_FNAME).read_text()
    assert (list(root.handlers), root.level) == before
    with records.logging_to(None):
        logging.getLogger("ruleforge.test").info("stderr alone")
    assert (list(root.handlers), root.level) == before


def test_the_api_clients_request_bodies_stay_out_of_the_record_s_log(tmp_path, capsys):
    sdk, http = logging.getLogger("openai._base_client"), logging.getLogger("httpx")
    levels = {name: logging.getLogger(name).level for name in ("openai", "anthropic", "httpx", "httpcore")}
    with records.logging_to(tmp_path, verbose=True):
        sdk.debug("Request options: the whole prompt")
        http.info("HTTP Request: POST one request")
        sdk.info("Retrying request in 0.5 seconds")
        logging.getLogger("ruleforge.test").debug("our own detail")
    log = (tmp_path / records.LOG_FNAME).read_text()
    assert "the whole prompt" not in log and "one request" not in log and "the whole prompt" not in capsys.readouterr().err
    assert "Retrying request" in log and "our own detail" in log
    assert {name: logging.getLogger(name).level for name in levels} == levels


#: An SDK imported inside the logging block of a ``ruleforge jobs`` verb with two run records, its debug logging asked
#: for by its variable, over a mock transport.
SDK_INSIDE_THE_BLOCK = """
import sys
from pathlib import Path

import httpx

from ruleforge.cli.jobs import logging_to

provider, places = sys.argv[1], [Path(place) for place in sys.argv[2:]]
assert provider not in sys.modules
reply = {"openai": {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]},
         "anthropic": {"id": "x", "type": "message", "role": "assistant", "model": "m",
                       "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                       "usage": {"input_tokens": 1, "output_tokens": 1}}}[provider]
http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply)))
with logging_to(places):
    if provider == "openai":
        from openai import OpenAI

        OpenAI(api_key="k", max_retries=0, http_client=http).chat.completions.create(
            model="m", messages=[{"role": "user", "content": "THE WHOLE PROMPT"}])
    else:
        from anthropic import Anthropic

        Anthropic(api_key="k", max_retries=0, http_client=http).messages.create(
            model="m", max_tokens=8, messages=[{"role": "user", "content": "THE WHOLE PROMPT"}])
"""


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_an_sdk_asked_for_debug_logging_writes_no_request_body_into_any_record(tmp_path, provider):
    """The SDK sets its logger's level from ``<PROVIDER>_LOG``; neither record's log handler takes the body."""
    import subprocess
    import sys

    places = [tmp_path / "first", tmp_path / "second"]
    for place in places:
        place.mkdir()
    environment = {**os.environ, f"{provider.upper()}_LOG": "debug",
                   "PYTHONPATH": str(Path(records.__file__).resolve().parents[2])}
    result = subprocess.run([sys.executable, "-c", SDK_INSIDE_THE_BLOCK, provider, *map(str, places)],
                            capture_output=True, text=True, env=environment, timeout=300)
    assert result.returncode == 0, result.stderr
    assert ["THE WHOLE PROMPT" in (place / records.LOG_FNAME).read_text() for place in places] == [False, False]


def test_a_record_s_log_handler_drops_a_quieted_line_whatever_its_logger_s_level(tmp_path):
    handler, sdk = records.log_handler(tmp_path), logging.getLogger("openai._base_client")
    for level, text in ((logging.DEBUG, "the whole prompt"), (logging.INFO, "retrying")):
        handler.handle(sdk.makeRecord(sdk.name, level, __file__, 0, text, (), None))
    handler.close()
    log = (tmp_path / records.LOG_FNAME).read_text()
    assert "the whole prompt" not in log and log.rstrip().endswith("retrying")


def test_a_block_ended_by_a_refusal_ends_the_record_s_log_with_it(tmp_path, capsys):
    with pytest.raises(ValueError, match="no such population"):
        with records.logging_to(tmp_path):
            logging.getLogger("ruleforge.test").info("started")
            raise ValueError("no such population")
    log = (tmp_path / records.LOG_FNAME).read_text().splitlines()
    assert "started" in log[0] and log[-1].endswith("the run ended on ValueError: no such population")
    assert "no such population" not in capsys.readouterr().err
    with pytest.raises(ValueError):
        with records.logging_to(None):
            raise ValueError("nothing recorded")


def _campaign_root(path):
    path.mkdir(parents=True)
    (path / "sampling_manifest.json").write_text("{}")
    return path


def test_a_record_is_never_made_in_a_campaign_root_or_through_a_link(tmp_path):
    import pytest

    from ruleforge.evaluate.api import CampaignError

    published = _campaign_root(tmp_path / "published")
    with pytest.raises(CampaignError, match="is inside the campaign root"):
        records.open_record(published / "rows", "grpo rows", [], None)
    with pytest.raises(CampaignError, match="the command's target .* is a campaign root"):
        records.open_record(published, "optimize", [], None)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "out.runs").symlink_to(elsewhere)
    with pytest.raises(CampaignError, match="its run records' directory .* is a symlink"):
        records.open_record(tmp_path / "out", "analyze", [], None, campaign_root=True)
    _campaign_root(tmp_path / "other.runs")
    with pytest.raises(CampaignError, match="its run records' directory .* is a campaign root"):
        records.open_record(tmp_path / "other", "analyze", [], None, campaign_root=True)
    (tmp_path / "link").symlink_to(published)
    with pytest.raises(CampaignError, match="is inside the campaign root"):
        records.open_record(tmp_path / "link" / "x", "optimize", [], None)
    assert sorted(path.name for path in published.iterdir()) == ["sampling_manifest.json"]
    assert not list(elsewhere.iterdir())
    record = records.open_record(published, "analyze", [], None, campaign_root=True)
    assert record.parent == tmp_path / "published.runs"


def test_a_search_only_directory_below_an_output_holding_a_link_is_refused(tmp_path):
    import pytest

    from ruleforge.evaluate.api import CampaignError

    published = _campaign_root(tmp_path / "published")
    hidden = tmp_path / "out" / "campaign" / "gemma" / "naive"
    hidden.mkdir(parents=True)
    (hidden / "draw-000").symlink_to(published, target_is_directory=True)
    hidden.chmod(0o300)
    try:
        for check in (lambda: records.check_outputs([tmp_path / "out"], way_out="x"),
                      lambda: records.open_record(tmp_path / "out", "a command", [], None,
                                                  claimed=[tmp_path / "out" / "campaign"])):
            with pytest.raises(CampaignError, match=f"the command's target {tmp_path / 'out'} cannot be walked whole"):
                check()
    finally:
        hidden.chmod(0o700)
    assert not records.runs_dir(tmp_path / "out").exists()
    assert sorted(path.name for path in published.iterdir()) == ["sampling_manifest.json"]


def test_a_record_is_never_made_on_a_pinned_root_s_path(tmp_path, monkeypatch):
    import pytest

    from ruleforge.evaluate import api

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://sampling/pub": ("leniency",)})
    (tmp_path / "sampling").mkdir()
    for target in (tmp_path / "sampling" / "pub", tmp_path / "sampling" / "pub" / "rows"):
        with pytest.raises(api.CampaignError, match="a published run of leniency"):
            records.open_record(target, "grpo rows", [], None)
    assert not list((tmp_path / "sampling").iterdir())


def test_an_artifact_root_a_command_fills_has_only_its_runs_directory_checked(tmp_path, monkeypatch):
    import pytest

    from ruleforge.evaluate import api

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://sampling/pub": ("leniency",)})
    _campaign_root(tmp_path / "artifacts" / "sampling" / "pub")
    with pytest.raises(api.CampaignError, match="a published run of"):
        records.open_record(tmp_path / "artifacts", "data fetch", [], None)
    record = records.open_record(tmp_path / "artifacts", "data fetch", ["data", "fetch"], None, artifact_root=True)
    assert record.parent == tmp_path / "artifacts.runs"
    _campaign_root(tmp_path / "other.runs")
    (tmp_path / "other").mkdir()
    with pytest.raises(api.CampaignError, match="its run records' directory .* is a campaign root"):
        records.open_record(tmp_path / "other", "data fetch", [], None, artifact_root=True)


def test_a_record_s_place_is_checked_with_the_campaign_roots_its_inventory_left_to_their_campaign(tmp_path):
    """A target holding a campaign root its command's own campaign keeps there (M0's ``<out>/campaign``)."""
    from ruleforge.evaluate.api import CampaignError

    out = tmp_path / "m0"
    _campaign_root(out / "campaign")
    claimed = [out / "campaign"]
    records.check_outputs([out], way_out="x", claimed=claimed)
    with pytest.raises(CampaignError, match="holds a campaign root of its own"):
        records.open_record(out, "a command", [], None)
    record = records.open_record(out, "a command", [], None, claimed=claimed)
    assert record.parent == tmp_path / "m0.runs" and records.check_place(out, claimed=claimed) == record.parent
    _campaign_root(out / "merged")
    with pytest.raises(CampaignError, match=r"holds a campaign root of its own \(.*merged\)"):
        records.open_record(out, "a command", [], None, claimed=claimed)
    assert sorted(path.name for path in out.iterdir()) == ["campaign", "merged"]


@pytest.mark.parametrize("planted", ["draw link", "file link", "root inside"])
def test_a_claimed_campaign_root_is_still_looked_inside_by_the_inventory_and_the_record_alike(tmp_path, planted):
    from ruleforge.evaluate.api import CampaignError

    out = tmp_path / "m0"
    root = _campaign_root(out / "campaign")
    foreign = _campaign_root(tmp_path / "foreign")
    (foreign / "ledger.jsonl").write_text("{}\n")
    condition = root / "model" / "figure5" / "base"
    condition.mkdir(parents=True)
    if planted == "draw link":
        (condition / "draw-000").symlink_to(foreign, target_is_directory=True)
        refusal = "draw-000, below .* is a symlink"
    elif planted == "file link":
        (condition / "condition.json").symlink_to(foreign / "ledger.jsonl")
        refusal = "condition.json, below .* is a symlink"
    else:
        _campaign_root(condition / "nested")
        refusal = r"holds a campaign root of its own \(.*nested\)"
    claimed = [out / "campaign"]
    with pytest.raises(CampaignError, match=refusal):
        records.check_outputs([out], way_out="x", claimed=claimed)
    with pytest.raises(CampaignError, match=refusal):
        records.open_record(out, "a command", [], None, claimed=claimed)
    assert not records.runs_dir(out).exists()
    assert sorted(path.name for path in foreign.iterdir()) == ["ledger.jsonl", "sampling_manifest.json"]
    assert (foreign / "ledger.jsonl").read_text() == "{}\n"


def _two_places(tmp_path):
    """``<safe>/link/../batch``, ``link`` pointing at ``<published>/cache`` in the campaign root ``published``."""
    published = _campaign_root(tmp_path / "published")
    (published / "cache").mkdir()
    (tmp_path / "safe").mkdir()
    (tmp_path / "safe" / "link").symlink_to(published / "cache", target_is_directory=True)
    return Path(f"{tmp_path / 'safe'}/link/../batch"), published


@pytest.mark.parametrize("caller", ["check_place", "check_batch_roots", "campaign_batch_places",
                                    "guarded_batch_writes", "wandb", "launch", "absolute_path", "absolute_path_home"])
def test_a_place_spelled_with_a_dot_dot_after_a_symlink_is_refused_by_every_check(tmp_path, monkeypatch, caller):
    """A check of the spelling written out would pass a write that lands in the campaign root the link points into."""
    from ruleforge.cli import optimize
    from ruleforge.cli.common import CliError
    from ruleforge.evaluate.api import CampaignError, refuse_protected
    from ruleforge.launch.records import check_records_place
    from ruleforge.judges import batch_api

    spelled, published = _two_places(tmp_path)
    assert Path(os.path.realpath(spelled)) == published / "batch" and os.path.abspath(spelled) == str(
        tmp_path / "safe" / "batch")
    refuse_protected(os.path.abspath(spelled), "written out")
    batch_judge = types.SimpleNamespace(capabilities=lambda: types.SimpleNamespace(batch=True), model_name="m",
                                        options=types.SimpleNamespace(batch_root=str(spelled)))
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT", raising=False)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(spelled.parent))  # <safe>/link/..: the published root, to the kernel

    def guarded():
        with records.guarded_batch_writes():
            batch_api.work_directory(spelled, "requests")

    def relative():
        monkeypatch.chdir(tmp_path / "safe")
        records.absolute_path("link/../batch")

    def home():
        monkeypatch.setenv("HOME", str(tmp_path / "safe"))
        records.absolute_path("~/link/../batch")

    checks = {"check_place": lambda: records.check_place(spelled), "check_batch_roots": lambda:
              records.check_batch_roots(batch_judge), "campaign_batch_places": lambda: records.batch_places(
                  None, (batch_judge,), campaign=tmp_path / "root"), "guarded_batch_writes": guarded,
              "wandb": lambda: optimize._check_wandb_dir(spelled), "launch": check_records_place,
              "absolute_path": relative, "absolute_path_home": home}
    with pytest.raises((CampaignError, CliError), match=r"names two places: .* as written out, and .*published"):
        checks[caller]()
    assert sorted(path.name for path in published.iterdir()) == ["cache", "sampling_manifest.json"]
    assert not list((published / "cache").iterdir()) and not list((tmp_path / "safe").glob("[!l]*"))


@pytest.mark.parametrize("caller", ["check_place", "check_batch_roots", "guarded_batch_writes", "wandb"])
def test_a_place_spelled_with_a_tilde_is_refused_by_every_check(tmp_path, monkeypatch, caller):
    from ruleforge.cli import optimize
    from ruleforge.cli.common import CliError
    from ruleforge.evaluate.api import CampaignError
    from ruleforge.judges import batch_api

    published = _campaign_root(tmp_path / "published")
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(published)
    spelled = Path("~/batches")
    batch_judge = types.SimpleNamespace(capabilities=lambda: types.SimpleNamespace(batch=True), model_name="m",
                                        options=types.SimpleNamespace(batch_root=str(spelled)))

    def guarded():
        with records.guarded_batch_writes():
            batch_api.work_directory(spelled, "requests")

    checks = {"check_place": lambda: records.check_place(spelled), "check_batch_roots": lambda:
              records.check_batch_roots(batch_judge), "guarded_batch_writes": guarded,
              "wandb": lambda: optimize._check_wandb_dir(spelled / "run")}
    home, literal = tmp_path / "home" / "batches", published / "~" / "batches"
    with pytest.raises((CampaignError, CliError),
                       match=rf"names two places: {re.escape(str(home))}.* expanded, and {re.escape(str(literal))}"):
        checks[caller]()
    assert sorted(path.name for path in published.iterdir()) == ["sampling_manifest.json"]
    assert not list((tmp_path / "home").iterdir())


def test_a_batch_judge_s_place_is_where_its_transport_writes_checked_here_or_by_its_campaign(tmp_path,
                                                                                               monkeypatch):
    from omegaconf.errors import OmegaConfBaseException

    from ruleforge.judges import JudgeSpec

    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    batch = JudgeSpec.from_config("claude-haiku-5-5-batch", batch_root=str(tmp_path / "b"))
    unset = JudgeSpec.from_config("claude-haiku-5-5-batch", batch_root=None)
    unresolved = JudgeSpec.from_config("claude-haiku-5-5-batch", batch_root="${rf.env:NO_SUCH_BATCH_ROOT_X}")
    assert records.batch_places(None, (None, JudgeSpec.fake(), unset)) == []
    [place] = records.batch_places(None, (batch,), home=tmp_path)
    assert (place.path, place.own, place.home, place.holds, place.kind) == (tmp_path / "b", False, tmp_path, True,
                                                                           records.BATCH_ROOT_KIND)
    assert place.protect is not None and place.protect() is None
    root = tmp_path / "root"
    [place] = records.batch_places(None, (batch, unresolved), campaign=root)
    assert (place.path, place.home, place.protect) == (tmp_path / "b", root, None)
    tilde, relative = (JudgeSpec.from_config("claude-haiku-5-5-batch", batch_root=value) for value in ("~/b", "rel/b"))
    assert [place.path for place in records.batch_places(None, (unset, tilde, relative, JudgeSpec.fake()),
                                                         campaign=root)] == [
        root / "batch", tmp_path / "home" / "b", tmp_path / "rel" / "b"]
    with pytest.raises(OmegaConfBaseException):
        records.batch_places(None, (unresolved,))


def test_places_meet_by_their_kinds(tmp_path):
    from ruleforge.cli.common import CliError
    from ruleforge.cli.sweep import Axis, Member, Place, check_places, refuse_overlapping

    axis = Axis("k", "k", "condition", ("1", "2"))
    first, second = Member(0, ((axis, "1"),)), Member(1, ((axis, "2"),))
    a, b = tmp_path / "a", tmp_path / "b"

    def batch(member, path, home):
        return Place(member, "batch root", path, own=False, home=home, holds=True, kind="batch root")

    def lock(member, path):
        return Place(member, "campaign lock", path, own=False)

    def wandb(member, path):
        return Place(member, "W&B directory", path, own=False, kind="W&B directory")

    def runs(member, path):
        return Place(member, "run records' directory", path, own=False)

    launch = [Place(None, "records lock", tmp_path / "L.lock", own=False),
              Place(None, "records directory", tmp_path / "L", own=False)]
    accepted = [[Place(first, "--out", a), batch(first, a, a)], [Place(first, "--out", a), batch(first, a / "x", a)],
                [Place(first, "--out", a), Place(second, "--out", b), batch(first, tmp_path, a)],
                [Place(first, "--out", a), batch(second, b / "x", b)],
                [batch(first, tmp_path / "s", a), batch(second, tmp_path / "s", b)],
                [wandb(first, tmp_path / "w"), wandb(second, tmp_path / "w")],
                [batch(first, tmp_path, a), lock(first, tmp_path / ".a.lock"), runs(first, tmp_path / "a.runs"),
                 wandb(first, tmp_path / "wandb"), batch(second, tmp_path / "s", b), *launch]]
    for places in accepted:
        refuse_overlapping(places, way_out="")
    refused = [([Place(first, "--out", a), batch(second, a, b)], "are one place"),
               ([Place(first, "--out", a), batch(second, a / "x", b)], "lie one inside the other"),
               ([Place(first, "--out", a), Place(second, "W&B directory", tmp_path, own=False)],
                "lie one inside the other"),
               ([Place(first, "--out", a / "x"), Place(None, "records directory", a, own=False)],
                "its --out .* and the launch's records directory"),
               ([lock(first, tmp_path / ".a.lock"), batch(second, tmp_path / ".a.lock", b)],
                "sweep members k=1 and k=2: campaign lock .* are one place"),
               ([lock(first, tmp_path / ".a.lock"), batch(first, tmp_path / ".a.lock" / "x", a)],
                "sweep member k=1: its campaign lock .* lie one inside the other"),
               ([launch[0], batch(first, tmp_path / "L.lock", a)], "the launch's records lock .* are one place"),
               ([launch[1], batch(first, tmp_path / "L" / "x", a)], "lie one inside the other"),
               ([runs(first, tmp_path / "a.runs"), batch(second, tmp_path / "a.runs", b)], "are one place"),
               ([wandb(first, tmp_path / "w"), batch(first, tmp_path / "w", a)], "are one place"),
               ([wandb(first, tmp_path / "w"), lock(second, tmp_path / "w")], "are one place"),
               ([lock(first, tmp_path / "x"), lock(second, tmp_path / "x")], "are one place"),
               ([Place(None, "records lock", tmp_path / "L", own=False), launch[1]],
                "the launch: the launch's records lock .* and the launch's records directory .* are one place")]
    for places, relation in refused:
        with pytest.raises(CliError, match=relation):
            refuse_overlapping(places, way_out="")
    checked = []
    check_places([Place(first, "--out", a, protect=lambda: checked.append("out")), batch(first, a / "x", a)],
                 way_out="")
    assert checked == ["out"]
