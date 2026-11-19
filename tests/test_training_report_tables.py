from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from ruleforge.grpo import report as rp
from ruleforge.grpo import report_tables as rt
from ruleforge.evaluate.campaign import build_plan, load_conditions, run_plan
from ruleforge.evaluate.identity import FORCED_JSON_LABEL_ONLY
from ruleforge.core.digests import canonical_digest, sha256_file
from ruleforge.grpo.anchor_set import SIDE_DEV, condition_projection
from ruleforge.core.artifacts import compute_scenario_id, load_jsonl_records
from ruleforge.judges.backend import GenerationOutcome
from tests.test_sampling_campaign import compose_campaign
from tests.test_sampling_runner import FakeBackend, forced

RULE_IDS = ("Rule 1", "Rule 2")
RULES_ROWS = [{"id": "Rule 1", "rule": "rule 1 body"}, {"id": "Rule 2", "rule": "rule 2 body"},
              {"id": "Meta-Rule 1", "rule": "meta body"}]
YES, NO = forced("yes"), forced("no")
#: A map naming Rule 1 only: a valid verdict under label_only, an incomplete map.
PARTIAL = json.dumps({"rule_judgments": {"Rule 1": "yes"}, "preempted_rules": [], "full_set_judgment": "yes"})
TRUNCATED = GenerationOutcome(text=None, raw_response="<think>on and on", finish_reason="length")


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    """Roots judged by Qwen3-8B's config (a thinking judge), its weights directory a stand-in."""
    from tests.test_sampling_weights import write_base

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    write_base(tmp_path / "models")


def _population(path: Path, indices, sources="ab") -> Path:
    path.write_text("".join(json.dumps({"prompt": f"prompt {i}", "response": f"response {i}",
                                        "source": sources[i % len(sources)]}) + "\n" for i in indices))
    return path


def run_dev(tmp_path, modelspec, *, root, mode, script, population, draws=2, attempts=3, policy="base",
            condition_id=None, retry="component"):
    """One condition of a dev campaign: the forced prompt, label_only, strict_think, component retry."""
    cfg = compose_campaign(tmp_path, **{
        "sampling.campaign_mode": "pilot" if draws > 1 else "screening",
        "campaign_id": root.name, "campaign_root": str(root), "scenario_set_path": str(population),
        "evaluation_mode": mode, "eval_cfg.reasoning_prompt_style": "forced-grpo",
        "sampling.forced_json_requirement": FORCED_JSON_LABEL_ONLY, "sampling.num_decoder_draws": draws,
        "sampling.max_semantic_attempts": attempts, "sampling.strict_think": "true",
        "sampling.retry_scope": retry, "condition_id": condition_id or f"fake/{mode}/{policy}",
        "model@judge_model": "qwen3-8b", "model@agent_model": "qwen3-8b",
        "condition_extra": f"{{policy: {policy}}}",
    })
    backend = FakeBackend(style="forced-grpo", script=script)
    backend.max_new_tokens = 1024
    task_spec = modelspec(reasoning_prompt_style="forced-grpo", evaluation_temperature=1.0)
    run_plan(build_plan(cfg, backend=backend, task_spec=task_spec), backend=backend, task_spec=task_spec)
    return root


def _ids(population):
    return [compute_scenario_id(row) for row in load_jsonl_records(population)]


def _read(root, kind, population, *, draws=2, condition=None, tokens=len):
    mode = rt.KINDS[kind].evaluation_mode
    condition = condition or rt.find_condition(root, policy="base", evaluation_mode=mode)
    source = rt.open_dev_condition(root, condition, evaluation_mode=mode)
    return source, rt.read_table(source, rt.KINDS[kind], rules=RULES_ROWS, rule_ids=RULE_IDS, draws=draws,
                                 scenario_ids=_ids(population), ruleset_sha256=source.condition.rule_set_sha256,
                                 count_tokens=tokens)


def _row(table, population, index, draw=0):
    sid = _ids(population)[index]
    return next(row for row in table if row["scenario_id"] == sid and row["draw"] == draw)


FULL_SCRIPT = {(0, "full_ruleset"): [TRUNCATED, YES], (1, "full_ruleset"): [PARTIAL], (2, "full_ruleset"): ["no json"],
               (3, "full_ruleset"): [NO]}


@pytest.fixture
def full_campaign(tmp_path, modelspec):
    population = _population(tmp_path / "pop.jsonl", range(4))
    root = run_dev(tmp_path, modelspec, root=tmp_path / "dev-stage1", mode="full", script=FULL_SCRIPT,
                   population=population)
    return root, population


def test_a_full_set_table_reads_attempts_truncation_maps_and_thinking_from_the_ledger(full_campaign):
    root, population = full_campaign
    source, table = _read(root, "stage1", population)
    assert source.draw_states == ("incomplete_exhausted", "incomplete_exhausted") and not source.complete
    assert len(table) == 8 and {row["draw"] for row in table} == {0, 1}
    truncated = _row(table, population, 0)["full"]
    raw = "<think>(0, 'full_ruleset')</think>" + YES  # the fake judge's raw text: its thinking, then the answer
    assert truncated == {"verdict": "yes", "attempts": 2, "truncated_attempts": 1, "map": {"Rule 1": "no", "Rule 2": "no"},
                         "map_complete": True, "map_failure": None,
                         "think_tokens": len("<think>(0, 'full_ruleset')"), "empty_think": False}
    assert raw.startswith("<think>")
    partial = _row(table, population, 1)["full"]
    assert (partial["verdict"], partial["map_complete"], partial["map"]) == ("yes", False, {})
    assert partial["map_failure"]
    assert _row(table, population, 2)["full"] == {"verdict": None, "attempts": 3, "truncated_attempts": 0}
    assert _row(table, population, 3, draw=1)["full"]["verdict"] == "no"
    scenarios = rp.Scenarios(tuple(_ids(population)), ("a",) * 4, ("dev",) * 4)
    path = tmp_path_table(root, table)
    full, _ = rp.read_responses(path, scenarios, rp.np.arange(4), RULE_IDS, 2, full=True, rules=False)
    assert full.verdict[2].tolist() == [rp.MISSING, rp.MISSING] and full.truncated[0, 0] == 1
    assert full.complete[1].tolist() == [False, False] and full.complete[0].tolist() == [True, True]


def tmp_path_table(root: Path, table) -> Path:
    path = root.parent / "table.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in table))
    return path


def test_the_thinking_counts_as_the_reward_counts_it():
    from ruleforge.judges.completion_format import THINKING_FAMILIES

    assert rt.think_stats("<think>\n\n</think>{}", len) == (len("<think>\n\n"), True)
    assert rt.think_stats("<think>a</think>b</think>{}", len) == (len("<think>a</think>b"), False)
    assert rt.think_stats("no thinking at all", len) == (0, True)
    qwen3, gemma4 = THINKING_FAMILIES["qwen3"], THINKING_FAMILIES["gemma4"]
    for raw in ("<think>\n\n</think>{}", "<think>a</think>b</think>{}", "none"):
        assert rt.think_stats(raw, len, qwen3) == rt.think_stats(raw, len)
    weighed = "<|channel>thought\nweigh the rule<channel|>{}"
    assert rt.think_stats(weighed, len, gemma4) == (len("<|channel>thought\nweigh the rule"), False)
    assert rt.think_stats(weighed, len) == (0, True)
    assert rt.think_stats("<|channel>thought\n <channel|>{}", len, gemma4) == (len("<|channel>thought\n "), True)
    assert rt.think_stats("<|channel>thought\ncut short", len, gemma4) == (0, True)


def test_a_draw_still_running_or_a_changed_output_is_refused(full_campaign, tmp_path):
    root, population = full_campaign
    condition = rt.find_condition(root, policy="base", evaluation_mode="full")
    ledger = root / condition / "draw-000" / "attempts.jsonl"
    original = ledger.read_text()
    sid = _ids(population)[3]
    ledger.write_text("".join(line + "\n" for line in original.splitlines() if sid not in line))
    with pytest.raises(rt.TableError, match="neither accepted nor exhausted"):
        _read(root, "stage1", population)
    ledger.write_text(original)
    raw = root / condition / "draw-001" / "raw_outputs.jsonl"
    rows = load_jsonl_records(raw)
    changed = [{**row, "raw_response": row["raw_response"] + " "} if row["scenario_id"] == sid else row for row in rows]
    raw.write_text("".join(json.dumps(row) + "\n" for row in changed))
    with pytest.raises(rt.TableError, match="missing from the raw outputs or is not the text"):
        _read(root, "stage1", population)
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    flipped = original.replace('"judgment": "no"', '"judgment": "yes"')
    ledger.write_text(flipped)
    with pytest.raises(rt.TableError, match="reads as 'no', the ledger says 'yes'"):
        _read(root, "stage1", population)
    ledger.write_text(original)
    source = rt.open_dev_condition(root, condition, evaluation_mode="full")
    for kind, draws, ids, tokens, match in (
        ("bank", 2, _ids(population), len, "is read from an 'individual_bank' condition"),
        ("stage1", 2, _ids(population)[:3], len, "judged 4 scenario"),
        ("stage1", 4, _ids(population), len, "ran 2 draw"),
        ("stage1", 2, _ids(population), None, "needs the model's tokenizer"),
    ):
        with pytest.raises(rt.TableError, match=match):
            rt.read_table(source, rt.KINDS[kind], rules=RULES_ROWS, rule_ids=RULE_IDS, draws=draws, scenario_ids=ids,
                          ruleset_sha256=source.condition.rule_set_sha256, count_tokens=tokens)
    with pytest.raises(rt.TableError, match="judged ruleset .*not the settings'"):
        rt.read_table(source, rt.KINDS["stage1"], rules=RULES_ROWS, rule_ids=RULE_IDS, draws=2,
                      scenario_ids=_ids(population), ruleset_sha256="0" * 64, count_tokens=len)


#: Rule 2 of scenario 0 invalid once (retried alone), then yes; scenario 1's Rule 1 truncated once.
BANK_SCRIPT = {(0, "Rule 2"): ["not json", YES], (1, "Rule 1"): [TRUNCATED, NO], (2, "Rule 1"): [YES]}


def test_a_complete_bank_is_read_per_rule_and_cross_checked_with_the_anchor_reader(tmp_path, modelspec):
    population = _population(tmp_path / "pop.jsonl", range(3))
    root = run_dev(tmp_path, modelspec, root=tmp_path / "dev-refs-bank", mode="individual_bank", script=BANK_SCRIPT,
                   population=population)
    source, table = _read(root, "bank", population)
    assert source.complete
    assert _row(table, population, 0)["rules"] == {
        "Rule 1": {"label": "no", "attempts": 1, "truncated_attempts": 0},
        "Rule 2": {"label": "yes", "attempts": 2, "truncated_attempts": 0}}  # component retry: Rule 2 alone
    assert _row(table, population, 1)["rules"]["Rule 1"] == {"label": "no", "attempts": 2, "truncated_attempts": 1}
    assert "full" not in _row(table, population, 2) and _row(table, population, 2)["rules"]["Rule 1"]["label"] == "yes"
    outcomes = root / source.condition.condition_id / "draw-000" / "bundle_outcomes.jsonl"
    outcomes.write_text(outcomes.read_text().replace('"Rule 1": "yes"', '"Rule 1": "no"', 1))
    with pytest.raises(rt.TableError, match="disagrees with the ledger"):
        _read(root, "bank", population)


def test_an_exhausted_rule_is_null_and_its_settled_siblings_keep_their_labels(tmp_path, modelspec):
    population = _population(tmp_path / "pop.jsonl", range(2))
    root = run_dev(tmp_path, modelspec, root=tmp_path / "dev-probe", mode="individual_bank",
                   script={(1, "Rule 2"): ["never"]}, population=population, draws=1)
    source, table = _read(root, "probe", population, draws=1)
    assert source.draw_states == ("incomplete_exhausted",)
    assert _row(table, population, 1)["rules"] == {"Rule 1": {"label": "no", "attempts": 1, "truncated_attempts": 0},
                                                   "Rule 2": {"label": None, "attempts": 3, "truncated_attempts": 0}}


def test_a_complete_full_set_condition_is_cross_checked_with_read_full_set(tmp_path, modelspec):
    population = _population(tmp_path / "pop.jsonl", range(2))
    root = run_dev(tmp_path, modelspec, root=tmp_path / "dev-refs-full", mode="full", script={(0, "full_ruleset"): [YES]},
                   population=population)
    source, table = _read(root, "full", population)
    assert source.complete and _row(table, population, 0)["full"]["verdict"] == "yes"
    outcomes = root / source.condition.condition_id / "draw-000" / "bundle_outcomes.jsonl"
    outcomes.write_text(outcomes.read_text().replace('"full_set_judgment": "yes"', '"full_set_judgment": "no"', 1))
    with pytest.raises(rt.TableError, match="from the materialised views"):
        _read(root, "full", population)


def test_a_condition_is_opened_only_from_a_finished_pilot_root(full_campaign):
    root, population = full_campaign
    condition = rt.find_condition(root, policy="base", evaluation_mode="full")
    with pytest.raises(rt.TableError, match="registers no condition 'nope'"):
        rt.open_dev_condition(root, "nope", evaluation_mode="full")
    with pytest.raises(rt.TableError, match="registered as 'full', not 'individual_bank'"):
        rt.open_dev_condition(root, condition, evaluation_mode="individual_bank")
    with pytest.raises(rt.TableError, match="0 'individual_bank' condition"):
        rt.find_condition(root, policy="base", evaluation_mode="individual_bank")
    completion = root / condition / "draw-001" / "completion.json"
    record = json.loads(completion.read_text())
    completion.write_text(json.dumps({**record, "state": "in_progress"}))
    with pytest.raises(rt.TableError, match="draw states"):
        rt.open_dev_condition(root, condition, evaluation_mode="full")
    completion.write_text(json.dumps(record))
    manifest = root / "sampling_manifest.json"
    text = manifest.read_text()
    manifest.write_text(text.replace('"campaign_mode": "pilot"', '"campaign_mode": "final"'))
    with pytest.raises(rt.TableError, match="'final' campaign"):
        rt.open_dev_condition(root, condition, evaluation_mode="full")
    frozen = json.loads(text)
    manifest.write_text(json.dumps({**frozen, "scenario_manifest_sha256": "0" * 64}))
    with pytest.raises(rt.TableError, match="not the one the campaign manifest froze"):
        rt.open_dev_condition(root, condition, evaluation_mode="full")


def test_a_table_is_written_with_its_identity_and_never_without_it(full_campaign):
    root, population = full_campaign
    source, table = _read(root, "stage1", population)
    with pytest.raises(rt.TableError, match="declares no serving dtype"):
        rt.write_table(root / "t.jsonl", table, source.condition)
    assert not (root / "t.jsonl").exists()
    served = dataclasses.replace(source.condition, serving_dtype="bfloat16", adapter_path="checkpoint-100/a.safetensors",
                                 adapter_sha256="a" * 64, adapter_config_sha256="c" * 64)
    identity = rt.write_table(root / "t" / "stage1.jsonl", table, served)
    projection = canonical_digest(condition_projection(served, side=SIDE_DEV))
    assert identity == {"model_weights_sha256": None, "adapter_sha256": "a" * 64, "serving_dtype": "bfloat16",
                        "condition_sha256": projection, "table_sha256": sha256_file(root / "t" / "stage1.jsonl")}
    # the condition, less its id, evaluation mode and labels: a bank and a full-set campaign of one judge share it
    assert projection == canonical_digest(condition_projection(
        dataclasses.replace(served, condition_id="other", evaluation_mode="individual_bank"), side=SIDE_DEV))
    assert rp.read_table_identity(root / "t" / "stage1.jsonl", {"adapter_sha256": "a" * 64}) == identity
    assert sorted(path.name for path in (root / "t").iterdir()) == ["stage1.identity.json", "stage1.jsonl"]


def test_two_writers_of_one_table_never_write_each_others_temporary(full_campaign, monkeypatch):
    root, population = full_campaign
    source, table = _read(root, "stage1", population)
    served = dataclasses.replace(source.condition, serving_dtype="float32")
    out = root / "t" / "stage1.jsonl"
    identity, nested = rt.write_table_identity, []

    def identity_beside_another_write(path, **kwargs):
        if not nested:  # the second writer, from start to rename, between the first's rows and its rename
            nested.append(None)
            nested[0] = rt.write_table(out, table, served)
        return identity(path, **kwargs)

    monkeypatch.setattr(rt, "write_table_identity", identity_beside_another_write)
    written = rt.write_table(out, table, served)
    assert written == nested[0] == rp.read_table_identity(out) and written["table_sha256"] == sha256_file(out)
    assert sorted(path.name for path in out.parent.iterdir()) == ["stage1.identity.json", "stage1.jsonl"]

    def failing(path, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(rt, "write_table_identity", failing)
    with pytest.raises(OSError, match="disk full"):
        rt.write_table(root / "u" / "stage1.jsonl", table, served)
    assert not list((root / "u").iterdir())


def test_a_table_beside_another_writes_sidecar_or_a_sidecar_cut_short_is_refused(full_campaign):
    root, population = full_campaign
    source, table = _read(root, "stage1", population)
    out = root / "t" / "stage1.jsonl"
    first = rt.write_table(out, table, dataclasses.replace(source.condition, serving_dtype="float32"))
    sidecar = rp.identity_path(out)
    first_sidecar = sidecar.read_bytes()
    second = rt.write_table(out, table[:-1], dataclasses.replace(source.condition, serving_dtype="bfloat16"))
    assert rp.read_table_identity(out) == second != first
    sidecar.write_bytes(first_sidecar)  # the first writer's sidecar, landing after the second's table
    with pytest.raises(rp.ReportError, match="names another table than .*: the table was rewritten after"):
        rp.read_table_identity(out)
    whole = (json.dumps(second, indent=1) + "\n").encode()  # the second's sidecar, as the report writes it
    sidecar.write_bytes(whole)
    assert rp.read_table_identity(out) == second
    for cut in range(len(whole) - 1):
        sidecar.write_bytes(whole[:cut])
        with pytest.raises(json.JSONDecodeError):
            rp.read_table_identity(out)


def test_the_scenario_table_and_the_stage_populations(tmp_path):
    dev = _population(tmp_path / "dev.jsonl", range(12), sources="abc")
    nonyes = _population(tmp_path / "nonyes.jsonl", range(12, 15), sources="abc")
    rows = rt.scenario_table(dev, nonyes)
    assert [row["subset"] for row in rows] == ["dev"] * 12 + ["dev_nonyes"] * 3
    assert rows[0] == {"scenario_id": compute_scenario_id(load_jsonl_records(dev)[0]), "source": "a", "subset": "dev"}
    assert len(rt.scenario_table(dev)) == 12
    with pytest.raises(rt.TableError, match="twice"):
        rt.scenario_table(dev, dev)
    unsourced = tmp_path / "x.jsonl"
    unsourced.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n")
    with pytest.raises(rt.TableError, match="has no source"):
        rt.scenario_table(unsourced)
    table = tmp_path / "scenarios.jsonl"
    table.write_text("".join(json.dumps(row) + "\n" for row in rows))
    settings = rp.load_settings(_settings(dev_scenarios=6))
    sets = rt.stage_sets(rp.read_scenarios(table), settings)
    assert len(sets["references"]) == 15 and len(sets["probe"]) == 3
    assert len(sets["stage1"]) == 9 and set(sets["probe"]) <= set(sets["stage1"])
    record = rt.write_populations(sets, [dev, nonyes], tmp_path / "pops")
    lines = {compute_scenario_id(json.loads(line)): line for path in (dev, nonyes)
             for line in path.read_text().splitlines(keepends=True)}
    written = (tmp_path / "pops" / "stage1.jsonl").read_text()
    assert written == "".join(line for sid, line in lines.items() if sid in set(sets["stage1"]))
    assert record["stage1"] == {"path": str(tmp_path / "pops" / "stage1.jsonl"),
                                "sha256": sha256_file(tmp_path / "pops" / "stage1.jsonl"), "count": 9}
    assert json.loads((tmp_path / "pops" / rt.POPULATIONS_FNAME).read_text()) == record
    staged, publish = {}, rt.write_text_atomically

    def another_writer_renames_over_it(path, text):
        publish(path, text)
        if Path(path).name == "probe.jsonl":
            staged["probe"] = text
            publish(path, text.split("\n", 1)[1])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rt, "write_text_atomically", another_writer_renames_over_it)
        mixed = rt.write_populations(sets, [dev, nonyes], tmp_path / "mixed")
    assert mixed["probe"] == {"path": str(tmp_path / "mixed" / "probe.jsonl"), "count": 3,
                              "sha256": hashlib.sha256(staged["probe"].encode()).hexdigest()}
    assert sha256_file(tmp_path / "mixed" / "probe.jsonl") != mixed["probe"]["sha256"]
    held = tmp_path / "held.jsonl"
    held.hardlink_to(tmp_path / "pops" / "stage1.jsonl")
    before = held.read_bytes()
    smaller = {"stage1": sets["stage1"][:2]}
    assert rt.write_populations(smaller, [dev, nonyes], tmp_path / "pops")["stage1"]["count"] == 2
    assert held.read_bytes() == before and (tmp_path / "pops" / "stage1.jsonl").read_bytes() != before
    assert not [path for path in (tmp_path / "pops").iterdir() if path.name.startswith(".")]
    with pytest.raises(rt.TableError, match="no population holds"):
        rt.write_populations({"x": ["missing"]}, [dev], tmp_path / "pops")
    ragged = tmp_path / "ragged.jsonl"
    ragged.write_text("\n" + dev.read_text().splitlines()[0])
    rt.write_populations({"one": [rows[0]["scenario_id"]]}, [ragged], tmp_path / "ragged")
    assert (tmp_path / "ragged" / "one.jsonl").read_text() == dev.read_text().splitlines(keepends=True)[0]


def test_a_row_holding_a_raw_line_separator_is_copied_whole(tmp_path):
    record = {"prompt": "first\u2028second", "response": "r", "source": "a"}
    line = json.dumps(record, ensure_ascii=False) + "\n"
    population = tmp_path / "pop.jsonl"
    population.write_text(line, encoding="utf-8")
    rt.write_populations({"one": [compute_scenario_id(record)]}, [population], tmp_path / "pops")
    assert (tmp_path / "pops" / "one.jsonl").read_bytes() == line.encode("utf-8")


def test_populations_are_named_by_absolute_path(tmp_path, monkeypatch):
    dev = _population(tmp_path / "dev.jsonl", range(2), sources="ab")
    monkeypatch.chdir(tmp_path)
    record = rt.write_populations({"one": [rt.scenario_table(dev)[0]["scenario_id"]]}, [dev], "relative")
    assert record["one"]["path"] == str(tmp_path / "relative" / "one.jsonl")


def _settings(**stage1):
    """Small settings over the CLI test's campaigns: a 2-draw bank, 3 base full-set draws, stage 1 at 2."""
    return {
        "ruleset_sha256": "0" * 64,
        "null_model": {"resamples": 10, "seed": 16, "level": 0.95}, "interval": {"resamples": 100, "seed": 16, "level": 0.95},
        "floored_gates": [],
        "stage1": {"draws": 2, "dev_scenarios": 3, "seed": 16, "base_draws": 3, "base_reference_draws": [0], "many_yes": 1,
                   "subsets": {k: ["dev"] for k in ("m1", "G1", "G2", "G3", "G4", "G6", "G7")},
                   "thinking": {"median_ratio": 0.5, "max_empty_share": 0.02}, **stage1},
        "probe": {"scenarios": 3, "seed": 16, "bank_draws": 2, "base_reference_draw": 0},
    }


def test_the_expected_draws_of_each_kind_are_the_settings():
    settings = rp.load_settings(_settings())
    assert {name: rt.expected_draws(kind, settings) for name, kind in rt.KINDS.items()} == {
        "bank": 2, "full": 3, "stage1": 2, "probe": 1}


def test_a_table_cut_to_stage_1_keeps_stage_1_and_the_probe(tmp_path):
    dev = _population(tmp_path / "dev.jsonl", range(9), sources="abc")
    nonyes = _population(tmp_path / "nonyes.jsonl", range(9, 11), sources="abc")
    settings = rp.load_settings(_settings(dev_scenarios=6))
    table = rt.scenario_table(dev, nonyes)
    cut = rt.stage1_table(table, settings)
    whole, part = rt._scenarios_of(table), rt._scenarios_of(cut)
    assert [row["scenario_id"] for row in cut] == [row["scenario_id"] for row in table
                                                  if row["scenario_id"] in set(rp.stage1_scenarios(whole, settings))]
    assert rt.stage_sets(part, settings)["references"] == rp.stage1_scenarios(whole, settings)
    assert rt.stage_sets(part, settings)["probe"] == rt.stage_sets(whole, settings)["probe"]
    assert sum(row["subset"] == "dev_nonyes" for row in cut) == 2 and len(cut) == 8
    # unequal sources: on the cut table the probe's quota would move to another source
    lopsided = rt.scenario_table(_population(tmp_path / "lop.jsonl", range(6), sources="aaaabc"))
    with pytest.raises(rt.TableError, match="would fix other scenarios for the probe than the whole table.*draw --dev and --dev-nonyes again"):
        rt.stage1_table(lopsided, rp.load_settings(_settings(dev_scenarios=4)))


def test_the_commands_prepare_the_campaigns_read_their_tables_and_report(tmp_path, modelspec, monkeypatch, capsys):
    dev = _population(tmp_path / "dev.jsonl", range(6), sources="a")
    nonyes = _population(tmp_path / "nonyes.jsonl", range(6, 8), sources="a")
    rules = tmp_path / "rules.jsonl"
    rules.write_text("".join(json.dumps(row) + "\n" for row in RULES_ROWS))
    config = tmp_path / "settings.yaml"
    config.write_text(json.dumps({**_settings(), "ruleset_sha256": sha256_file(rules)}))
    base = tmp_path / "base"
    assert rt.main(["populations", "--settings", str(config), "--dev", str(dev), "--dev-nonyes", str(nonyes),
                    "--out-dir", str(base)]) == 0
    pops = json.loads(capsys.readouterr().out)
    assert (pops["references"]["count"], pops["stage1"]["count"], pops["probe"]["count"]) == (5, 5, 3)
    assert [row["subset"] for row in load_jsonl_records(base / "scenarios.jsonl")].count("dev_nonyes") == 2
    assert rt.main(["populations", "--settings", str(config), "--dev", str(dev), "--out-dir", str(tmp_path / "dev-only")]) == 0
    assert json.loads(capsys.readouterr().out)["stage1"]["count"] == 3
    references = Path(pops["references"]["path"])
    # odd scenarios: Rule 1 and the verdict say yes; even ones keep the fake judge's no
    answers = {(i, "full_ruleset"): [forced("yes", {"Rule 1": "yes", "Rule 2": "no"})] * 12 for i in range(1, 8, 2)}
    bank_answers = {(i, "Rule 1"): [YES] * 12 for i in range(1, 8, 2)}
    bank_root = run_dev(tmp_path, modelspec, root=tmp_path / "refs-bank", mode="individual_bank", script=bank_answers,
                        population=references)
    full_root = run_dev(tmp_path, modelspec, root=tmp_path / "refs-full", mode="full", script=answers,
                        population=references, draws=3)
    monkeypatch.setattr(rt, "_tokens", lambda path: len)
    from ruleforge.judges.completion_format import CompletionFormatError

    (tmp_path / "bare").mkdir()
    common = ["--settings", str(config), "--scenarios", str(base / "scenarios.jsonl"), "--rules", str(rules)]
    with pytest.raises(CompletionFormatError, match="holds no config.json"):
        rt.main(["read", *common, "--tokenizer", str(tmp_path / "bare"), "--root", str(full_root), "--policy", "base",
                 "--kind", "full", "--out", str(tmp_path / "x.jsonl")])
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    read = ["read", *common, "--tokenizer", str(tmp_path)]
    with pytest.raises(rt.TableError, match="declares no serving dtype"):
        rt.main([*read, "--root", str(bank_root), "--policy", "base", "--kind", "bank", "--out", str(base / "bank.jsonl")])
    identity = {"model_weights_sha256": None, "adapter_sha256": None, "serving_dtype": "bfloat16"}
    monkeypatch.setattr(rt, "table_identity", lambda condition: identity)
    assert rt.main([*read, "--root", str(bank_root), "--policy", "base", "--kind", "bank", "--out",
                    str(base / "bank.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 10
    assert rt.main([*read, "--root", str(full_root), "--policy", "base", "--kind", "full", "--out",
                    str(base / "full.jsonl")]) == 0
    capsys.readouterr()
    assert rp.read_references(base, rp.read_settings(config), RULE_IDS).identity == identity
    stage1_root = run_dev(tmp_path, modelspec, root=tmp_path / "dev-stage1", mode="full", script=answers,
                          population=Path(pops["stage1"]["path"]), policy="checkpoint-100")
    probe_root = run_dev(tmp_path, modelspec, root=tmp_path / "dev-probe", mode="individual_bank", script=bank_answers,
                         population=Path(pops["probe"]["path"]), policy="checkpoint-100", draws=1)
    save = tmp_path / "saves" / "checkpoint-100"
    assert rt.main([*read, "--root", str(stage1_root), "--policy", "checkpoint-100", "--kind", "stage1", "--out",
                    str(save / "stage1.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 2 * 5
    assert rt.main(["read", *common, "--root", str(probe_root), "--policy", "checkpoint-100", "--kind", "probe",
                    "--out", str(save / "probe.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["exhausted_draws"] == []
    assert rp.main(["saves", "--settings", str(config), "--base", str(base), "--saves", str(tmp_path / "saves"),
                     "--rules", str(rules), "--out", str(tmp_path / "out")]) == 0
    assert "| 100 |" in capsys.readouterr().out
    assert list(json.loads((tmp_path / "out" / "report.json").read_text())["saves"]) == ["100"]


def test_every_condition_registered_is_found_by_its_policy(tmp_path, modelspec):
    population = _population(tmp_path / "pop.jsonl", range(2))
    root = tmp_path / "root"
    for policy in ("checkpoint-100", "checkpoint-200"):
        run_dev(tmp_path, modelspec, root=root, mode="full", script={}, population=population, policy=policy)
    assert rt.find_condition(root, policy="checkpoint-200", evaluation_mode="full") == "fake/full/checkpoint-200"
    assert sorted(load_conditions(root)) == ["fake/full/checkpoint-100", "fake/full/checkpoint-200"]


def test_bundle_retry_keeps_only_the_accepted_bundle(tmp_path, modelspec):
    population = _population(tmp_path / "pop.jsonl", range(2))
    root = run_dev(tmp_path, modelspec, root=tmp_path / "bundle", mode="individual_bank", retry="scenario_bundle",
                   script={(1, "Rule 2"): ["never"]}, population=population, draws=1)
    _, table = _read(root, "probe", population, draws=1)
    assert _row(table, population, 1)["rules"] == {"Rule 1": {"label": None, "attempts": 3, "truncated_attempts": 0},
                                                   "Rule 2": {"label": None, "attempts": 3, "truncated_attempts": 0}}
    assert _row(table, population, 0)["rules"]["Rule 1"] == {"label": "no", "attempts": 1, "truncated_attempts": 0}


def test_the_thinking_is_counted_by_the_models_own_tokenizer(monkeypatch, tmp_path):
    import transformers

    seen = []

    def tokenizer(text, add_special_tokens=True):
        seen.append(add_special_tokens)
        return {"input_ids": text.split()}

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path: seen.append(path) or tokenizer)
    count = rt._tokens(tmp_path)
    assert count("<think> a b") == 3 and seen == [str(tmp_path), False]


def test_a_transport_failure_consumes_no_attempt():
    from types import SimpleNamespace

    component = SimpleNamespace(component_id="Rule 1", failure_reason="truncated")
    rows = [SimpleNamespace(consumes_attempt=False, components=[component]),
            SimpleNamespace(consumes_attempt=True, components=[component])]
    assert rt._attempts(rows) == {"Rule 1": (1, 1)}
