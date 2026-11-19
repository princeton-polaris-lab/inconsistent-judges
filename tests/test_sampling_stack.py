from __future__ import annotations

import json
from pathlib import Path

import pytest

from ruleforge.evaluate import stack as st
from ruleforge.evaluate.identity import (
    COMPLETION_FNAME,
    DRAW_PROVENANCE_FNAME,
    RUN_LOG_ROWS_KEY,
    RUNS_FNAME,
    run_log_damage,
    run_log_problem,
    run_log_rows,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.evaluate.ledger import LEDGER_FNAME
from ruleforge.judges.backend import PRECISION_ENV_DEFAULTS, precision_env

OLD = {"vllm_version": "0.19.0", "torch_version": "2.10.0+cu128", "transformers_version": "5.5.0"}
NEW = {"vllm_version": "0.28.0+cu129", "torch_version": "2.13.0+cu129", "transformers_version": "5.17.0"}
UNRECORDED_ENV = {name: None for name in PRECISION_ENV_DEFAULTS}
OLD_SIGNATURE = {"vllm": "0.19.0", "torch": "2.10.0+cu128", "transformers": "5.5.0", "tokenizers": None,
                 "triton": None, **UNRECORDED_ENV}
ONE_GPU = {"model": "/a/Qwen3-8B", "tensor_parallel_size": 1, "max_num_seqs": 256, "disable_log_stats": True,
           "max_model_len": 40960, "attention_config": {"backend": "FLASH_ATTN"}, "enable_lora": False}
TWO_GPUS = {**ONE_GPU, "model": "/b/Qwen3-8B", "tensor_parallel_size": 2, "max_num_seqs": 512,
            "disable_log_stats": False}


def test_signatures_name_exact_versions_local_label_included():
    """torch's and vLLM's local label names the CUDA build (cuBLAS, cuDNN): 0.28.0+cu129 is not 0.28.0+cu130."""
    assert st.stack_signature(OLD) == OLD_SIGNATURE
    assert st.stack_signature({}) == {name: None for name in (*st.STACK_PACKAGES, *PRECISION_ENV_DEFAULTS)}
    assert st.describe_signatures([st.stack_signature(OLD), {"vllm": "0.28.0+cu129"}]) == (
        "vllm 0.19.0 / torch 2.10.0+cu128 / transformers 5.5.0 + vllm 0.28.0+cu129"
    )
    assert st.describe_signature(st.stack_signature({})) == "no stack recorded"


def test_the_precision_settings_are_part_of_the_signature_once_recorded():
    recorded = {**NEW, "precision_env": {"VLLM_FLOAT32_MATMUL_PRECISION": None, "TRITON_F32_DEFAULT": "tf32"}}
    signature = st.stack_signature(recorded)
    assert signature["VLLM_FLOAT32_MATMUL_PRECISION"] == "highest" and signature["TRITON_F32_DEFAULT"] == "tf32"
    assert signature["VLLM_BATCH_INVARIANT"] is None
    assert st.recorded_precision({"precision_env": "garbled"}) == UNRECORDED_ENV
    assert "TRITON_F32_DEFAULT=tf32" in st.describe_signature(signature)
    library_defaults = {**NEW, "precision_env": precision_env({})}
    assert st.stack_changes(recorded, library_defaults) == {"TRITON_F32_DEFAULT": ("tf32", "unset")}
    assert st.stack_changes(NEW, library_defaults) == {}
    spelled = {"precision_env": {"VLLM_FLOAT32_MATMUL_PRECISION": "highest", "TRITON_F32_DEFAULT": "",
                                 "VLLM_BATCH_INVARIANT": "00"}}
    assert st.recorded_precision(spelled) == precision_env({})
    given = {"VLLM_FLOAT32_MATMUL_PRECISION": "HIGHEST", "TRITON_F32_DEFAULT": "unset", "VLLM_BATCH_INVARIANT": "true"}
    assert st.recorded_precision({"precision_env": given}) == given


def test_stack_changes_compare_exact_versions_of_recorded_packages_only():
    assert st.stack_changes({"vllm_version": "0.19.0"}, {"vllm_version": "0.19.0", "torch_version": "2.13.0"}) == {}
    assert st.stack_changes({"vllm_version": "0.19.0"}, {"vllm_version": "0.19.1"}) == {"vllm": ("0.19.0", "0.19.1")}
    assert st.stack_changes({"torch_version": "2.13.0+cu129"}, {"torch_version": "2.13.0+cu130"}) == {
        "torch": ("2.13.0+cu129", "2.13.0+cu130"),
    }
    assert st.stack_changes({"vllm_version": "0.28.0"}, {"vllm_version": "0.28.0+cu129"}) == {
        "vllm": ("0.28.0", "0.28.0+cu129"),
    }
    assert st.stack_changes({"triton_version": "3.6.0"}, {"triton_version": "3.7.1"}) == {"triton": ("3.6.0", "3.7.1")}
    changed = st.stack_changes(OLD, NEW)
    assert list(changed) == ["vllm", "torch", "transformers"]
    assert st.describe_changes({"vllm": ("0.19.0", "0.28.0")}) == "vllm 0.19.0 -> 0.28.0"


def test_engine_arguments_are_compared_placement_aside_when_both_records_name_them():
    assert st.ENGINE_PLACEMENT_ARGS == {
        "model", "tensor_parallel_size", "disable_log_stats", "max_num_seqs",
        "gpu_memory_utilization", "max_num_batched_tokens",
    }
    first, placed = {"engine_args": ONE_GPU}, {"engine_args": TWO_GPUS}
    assert st.engine_args_changes(first, placed) == {}
    assert st.binding_engine_args(placed) == {
        "max_model_len": 40960, "attention_config": {"backend": "FLASH_ATTN"}, "enable_lora": False,
    }
    other = {"engine_args": {**TWO_GPUS, "attention_config": {"backend": "TRITON_ATTN"}, "enable_prefix_caching": False}}
    assert st.engine_args_changes(first, other) == {
        "engine_args.attention_config": ({"backend": "FLASH_ATTN"}, {"backend": "TRITON_ATTN"}),
        "engine_args.enable_prefix_caching": (None, False),
    }
    assert st.engine_args_changes(first, {}) == {} and st.engine_args_changes({}, other) == {}
    assert st.binding_engine_args({"engine_args": "garbled"}) is None
    assert st.runtime_changes({**OLD, **first}, {**NEW, **other}) == {
        **st.stack_changes(OLD, NEW), **st.engine_args_changes(first, other),
    }


#: The contrast arguments of ONE_GPU as a judgment signature names them: canonical JSON, null for unset.
ONE_GPU_CONTRAST = {"engine_args.dtype": "null", "engine_args.attention_config": '{"backend":"FLASH_ATTN"}',
                    "engine_args.max_model_len": "40960"}


def test_a_judgment_signature_adds_the_contrast_arguments_and_a_digest_of_the_binding_engine_arguments():
    signature = st.judgment_signature({**NEW, "engine_args": ONE_GPU})
    assert {name: signature[name] for name in st.stack_signature(NEW)} == st.stack_signature(NEW)
    assert {name: signature[f"engine_args.{name}"] for name in st.CONTRAST_ENGINE_ARGS} == {
        name: ONE_GPU_CONTRAST[f"engine_args.{name}"] for name in ("dtype", "attention_config", "max_model_len")
    }
    assert len(signature["engine_args"]) == 16
    assert list(signature) == [*st.stack_signature(NEW), *ONE_GPU_CONTRAST, "engine_args"]
    assert st.judgment_signature({**NEW, "engine_args": TWO_GPUS}) == signature
    longer = st.judgment_signature({**NEW, "engine_args": {**ONE_GPU, "max_model_len": 8192}})
    assert longer["engine_args"] != signature["engine_args"] and longer["engine_args.max_model_len"] == "8192"
    fp32 = st.judgment_signature({**NEW, "engine_args": {**ONE_GPU, "dtype": "float32"}})
    assert fp32["engine_args.dtype"] == '"float32"'
    unrecorded = st.judgment_signature(NEW)
    assert unrecorded["engine_args"] is None and all(unrecorded[name] is None for name in ONE_GPU_CONTRAST)
    assert st.describe_signature(signature).endswith(
        'engine_args.dtype=null / engine_args.attention_config={"backend":"FLASH_ATTN"} / '
        f"engine_args.max_model_len=40960 / engine_args={signature['engine_args']}"
    )
    assert st.across_conditions(signature) == {**st.stack_signature(NEW), **ONE_GPU_CONTRAST}


def test_signatures_conflict_only_on_a_field_both_name():
    old, new = st.stack_signature(OLD), st.stack_signature(NEW)
    assert st.signatures_conflict(old, new) and not st.signatures_conflict(old, old)
    assert not st.signatures_conflict(old, {"vllm": "0.19.0", "torch": None, "transformers": None})
    assert st.signatures_mixed([old, new]) and not st.signatures_mixed([old])
    assert st.contrast_conflict([old], [old]) is None and st.contrast_conflict([], [new]) is None
    assert st.contrast_conflict([old], [new]).startswith("vllm 0.19.0 / torch 2.10.0+cu128")
    assert st.contrast_conflict([old, new], [old]).startswith("the first member's judgments come from several")
    assert st.contrast_conflict([old], [old, new]).startswith("the second member's")


def test_engine_arguments_mix_a_condition_but_refuse_a_contrast_only_on_the_contrast_arguments():
    base = st.judgment_signature({**NEW, "engine_args": ONE_GPU})
    policy = st.judgment_signature(
        {**NEW, "engine_args": {**ONE_GPU, "model": "/m/merged", "enable_lora": True, "max_lora_rank": 32}}
    )
    assert st.signatures_mixed([base, policy])
    assert st.contrast_conflict([base], [policy]) is None
    assert st.contrast_conflict([base, policy], [base]).startswith(
        "the first member's judgments come from several stacks or engine configurations"
    )
    other_stack = st.judgment_signature({**OLD, "engine_args": ONE_GPU})
    reason = st.contrast_conflict([base], [other_stack])
    assert reason.startswith("vllm 0.28.0+cu129") and f"engine_args={base['engine_args']}" not in reason
    for changed, shown in (
        ({"dtype": "float32", "attention_config": {"backend": "TRITON_ATTN"}}, 'engine_args.dtype="float32"'),
        ({"attention_config": {"backend": "TRITON_ATTN"}}, 'engine_args.attention_config={"backend":"TRITON_ATTN"}'),
        ({"attention_config": None}, "engine_args.attention_config=null"),
        ({"max_model_len": 8192}, "engine_args.max_model_len=8192"),
        ({"max_model_len": None}, "engine_args.max_model_len=null"),
    ):
        other = st.judgment_signature({**NEW, "engine_args": {**ONE_GPU, **changed}})
        reason = st.contrast_conflict([base], [other])
        assert reason is not None and f" against {st.describe_signature(st.across_conditions(other))}" in reason
        assert shown in reason, (changed, reason)
    assert st.contrast_conflict([st.judgment_signature(NEW)], [base]) is None


def _draw(root: Path, draw: int, *, runtime=None, ledger: str | None = '{"scenario_id": "s"}\n', runs=()) -> Path:
    path = root / f"draw-{draw:03d}"
    path.mkdir(parents=True)
    if runtime is not None:
        (path / DRAW_PROVENANCE_FNAME).write_text(json.dumps({"runtime": runtime}))
    if ledger is not None:
        (path / LEDGER_FNAME).write_text(ledger)
    if runs:
        (path / RUNS_FNAME).write_text("".join(json.dumps(row) + "\n" for row in runs if isinstance(row, dict))
                                       + "".join(row for row in runs if isinstance(row, str)))
    return path


@pytest.mark.parametrize("ledger, expected", [
    (None, False),
    ("", False),
    ('{"scenario_id": "s"', False),
    ('["not", "a", "record"]\n', False),
    ("{oops\n", False),
    ('{"scenario_id": "s"}\n', True),
])
def test_a_draw_has_attempts_when_its_first_ledger_line_is_a_complete_record(tmp_path, ledger, expected):
    assert st.draw_has_attempts(_draw(tmp_path, 0, ledger=ledger)) is expected


def test_the_runtimes_a_draws_judgments_came_from(tmp_path):
    allowed = {"runtime": NEW, st.STACK_CHANGE_KEY: {"vllm": ["0.19.0", "0.28.0+cu129"]}}
    draw = _draw(tmp_path, 0, runtime=OLD, runs=[{"runtime": OLD}, allowed, '{"runtime": {"vllm_ver'])
    assert st.judgment_runtimes(draw) == [OLD, NEW]
    assert st.recorded_runtime(draw) == OLD
    assert st.judgment_runtimes(_draw(tmp_path, 1, runtime=NEW, ledger=None)) == []
    assert st.judgment_runtimes(_draw(tmp_path, 2)) == [{}]
    (draw / DRAW_PROVENANCE_FNAME).write_text("[]")
    with pytest.raises(ValueError, match="not a JSON object"):
        st.recorded_runtime(draw)


def test_each_run_s_ledger_rows_are_those_from_its_boundary_to_the_next_run_s():
    change = {"vllm": ["0.19.0", "0.28.0+cu129"]}

    def run(rows, start=3, end=7):
        spans, problem = st.run_spans(rows, start=start, end=end)
        return [(span.appended, span.stack_change) for span in spans], problem

    key = st.LEDGER_ROWS_KEY
    assert run([{key: 3}, {key: 5}, {st.STACK_CHANGE_KEY: change}, {key: 7}]) == (
        [(2, None), (2, change), (0, None)], None)
    assert run([]) == ([], None) and run([], end=3) == ([], None)
    assert run([{key: 3}, {}]) == ([(None, None), (None, None)], None)
    assert run([{key: True}]) == ([(None, None)], None)
    orphan = run([{st.STACK_CHANGE_KEY: change}, {key: 3}])
    assert orphan == ([(None, None)], "a stack change logged before any run")
    for rows in ([{key: 4}], [{key: 3}, {key: 2}], [{key: 3}, {key: 8}]):
        spans, problem = run(rows)
        assert {appended for appended, _ in spans} == {None}
        assert problem == (f"its runs' ledger boundaries {[row[key] for row in rows]} are not its ledger's (rows 3 to "
                           "7): the rows cannot be told to the runs that appended them")


def test_a_run_log_is_held_to_the_rows_its_gate_counted(tmp_path):
    change = {"recorded_at": "t", "runtime": NEW, st.STACK_CHANGE_KEY: {"vllm": ["0.19.0", "0.28.0+cu129"]}}
    draw = _draw(tmp_path, 0, runtime=OLD, runs=[{"started_at": "t"}, change])
    gate = draw / COMPLETION_FNAME
    assert run_log_problem(draw) is None and run_log_problem(draw, required=True) == (
        "its completion.json counts no runs.jsonl rows, so the run log cannot be held to the rows it describes")
    gate.write_text(json.dumps({"state": "complete"}))
    assert run_log_problem(draw) is None and run_log_problem(draw, required=True).startswith("its completion.json")
    for counted in (1, 2):  # one run that logged itself and stopped before writing the gate again appended nothing
        gate.write_text(json.dumps({RUN_LOG_ROWS_KEY: counted}))
        assert run_log_problem(draw, required=True) is None and run_log_rows(draw) == 2
    assert len(st.judgment_runtimes(draw)) == 2
    for counted in (3, "2", None):
        gate.write_text(json.dumps({RUN_LOG_ROWS_KEY: counted}))
        if counted is None:
            assert run_log_problem(draw) is None
            continue
        assert run_log_problem(draw) == (
            f"its runs.jsonl holds 2 complete row(s), and its completion.json counted {counted!r} when it was written: "
            "rows were cut from the run log since, and the ledger rows of a run cut from it would read as the run's "
            "before it, on that run's stack and code state")
        with pytest.raises(SamplingDesignError, match="its judgments' stacks cannot be read from it"):
            st.judgment_runtimes(draw)
    assert st.judgment_runtimes(_draw(tmp_path, 1, ledger=None)) == []


@pytest.mark.parametrize(("log", "expected"), [
    (None, None),
    ('{"a": 1}\n{"b": 2}\n', None),
    ('{"a": 1}\n{"b": 2', ("its runs.jsonl ends in a torn row (7 byte(s) after its 1 complete row(s))", True)),
    ('{"a": 1}\n{"b": 2{"c": 3}\n{"d": 4}\n', (
        "its runs.jsonl holds a malformed row after its 1 complete row(s), and 1 line(s) after it: no reader reads "
        "past it (rows logged behind torn bytes are lost to every reader)", False)),
    ('{"a": 1}\n\n{"d": 4}\n{"e"', (
        "its runs.jsonl holds a malformed row after its 1 complete row(s), and 2 line(s) after it: no reader reads "
        "past it (rows logged behind torn bytes are lost to every reader)", False)),
    ('{"a": 1}\noops\n', (
        "its runs.jsonl holds a malformed row after its 1 complete row(s), and 0 line(s) after it: no reader reads "
        "past it (rows logged behind torn bytes are lost to every reader)", False)),
])
def test_a_run_log_s_bytes_after_its_complete_rows_are_named(tmp_path, log, expected):
    draw = _draw(tmp_path, 0, runs=[log] if log is not None else ())
    assert run_log_damage(draw) == expected


def test_each_row_is_held_to_the_stack_it_names(tmp_path):
    old, new = st.stack_digest(OLD), st.stack_digest(NEW)
    change = {"recorded_at": "t", "runtime": NEW, st.STACK_CHANGE_KEY: {"vllm": ["0.19.0", "0.28.0+cu129"]}}

    def draw(index, rows, runs, model_type="local", provenance=True):
        path = _draw(tmp_path, index, ledger="".join(json.dumps(row) + "\n" for row in rows), runs=runs)
        if provenance:
            (path / DRAW_PROVENANCE_FNAME).write_text(json.dumps({"runtime": OLD, "model_type": model_type}))
        return path

    def named(stack):
        return {"scenario_id": "s", st.JUDGED_ON_KEY: stack}

    unstamped = draw(0, [{"scenario_id": "s"}], [{"runtime": NEW}])
    assert st.unlogged_stacks(unstamped) is None
    assert st.unlogged_stacks(draw(1, [named(old)], [])) is None
    assert st.unlogged_stacks(draw(2, [named(new)] * 2, [{"runtime": OLD}, change, '{"tor'])) is None
    gone = draw(3, [named(old), named(new), named(new)], [{"runtime": OLD}])
    assert st.unlogged_stacks(gone) == (
        f"2 row(s) of its attempts.jsonl were judged on a stack (judged_on {new[:12]}) that neither its "
        "draw_provenance.json nor any complete row of its runs.jsonl records: the run that judged them is gone from "
        "the run log (cut from it, rolled back with the gate that counted it, or logged behind bytes no reader "
        "reads), and they would read as judged on the stacks left")
    with pytest.raises(SamplingDesignError, match=r"judged on a stack \(judged_on .*its judgments' stacks cannot be"):
        st.judgment_runtimes(gone)
    hidden = draw(4, [named(new)], ['{"runtime": {"vllm_ver' + json.dumps({"runtime": NEW}) + "\n"])
    assert st.unlogged_stacks(hidden).startswith("1 row(s) of its attempts.jsonl were judged on a stack")
    unchanged = draw(5, [named(new)], [{"runtime": OLD}, {"runtime": NEW}])
    assert st.unlogged_stacks(unchanged) == (
        "1 row(s) of its attempts.jsonl were judged on another stack than its provenance's (vllm 0.19.0 -> "
        "0.28.0+cu129, torch 2.10.0+cu128 -> 2.13.0+cu129, transformers 5.5.0 -> 5.17.0), and its runs.jsonl logs no "
        "stack change to it: they would read as judged on the provenance's stack")
    assert st.unlogged_stacks(draw(6, [named(new)], [{"runtime": NEW}], model_type="openai_batch")) is None
    partial = {key: value for key, value in NEW.items() if key != "transformers_version"}
    assert st.unlogged_stacks(draw(7, [named(st.stack_digest({**OLD, "tokenizers_version": "0.22"}))],
                                   [{"runtime": {**OLD, "tokenizers_version": "0.22"}}])) is None
    assert st.unlogged_stacks(draw(8, [named(new)], [{"runtime": NEW}, {**change, "runtime": partial}])) is None
    records = [type("Record", (), {"judged_on": stack})() for stack in (None, new)]
    assert st.unlogged_stacks(unstamped, records).startswith("1 row(s) of its attempts.jsonl were judged on another")
    assert st.unlogged_stacks(draw(9, [named(old)], [], provenance=False)).startswith("1 row(s) of its attempts.jsonl "
                                                                                       "were judged on a stack")
    assert st.unlogged_stacks(tmp_path / "draw-404") is None


def test_a_conditions_signatures_are_its_judged_draws(tmp_path):
    _draw(tmp_path, 1, runtime=OLD)
    _draw(tmp_path, 0, runtime=OLD)
    _draw(tmp_path, 2, runtime=NEW, ledger=None)
    (tmp_path / "draw-notes.txt").write_text("a file, not a draw")
    assert [path.name for path in st.draw_dirs(tmp_path)] == ["draw-000", "draw-001", "draw-002"]
    assert st.condition_signatures(tmp_path) == [st.judgment_signature(OLD)]
    _draw(tmp_path, 3, runtime=NEW)
    assert st.condition_signatures(tmp_path) == [st.judgment_signature(OLD), st.judgment_signature(NEW)]


def _two_gpu_draws(condition_dir, changed):
    """draw-000 on ONE_GPU, draw-001 on TWO_GPUS with ``changed``; the condition's signatures and their index."""
    _draw(condition_dir, 0, runtime={**NEW, "engine_args": ONE_GPU})
    _draw(condition_dir, 1, runtime={**NEW, "engine_args": {**TWO_GPUS, **changed}})
    args_by_digest: st.EngineArgsIndex = {}
    return st.condition_signatures(condition_dir, args_by_digest=args_by_digest), args_by_digest


@pytest.mark.parametrize("changed", [
    {},
    # an argument recorded as None is unset
    {"enable_prefix_caching": None},
])
def test_draws_that_differ_in_placement_alone_are_one_engine_configuration(tmp_path, changed):
    assert st.engine_args_changes({"engine_args": ONE_GPU}, {"engine_args": {**TWO_GPUS, **changed}}) == {}
    signatures, args_by_digest = _two_gpu_draws(tmp_path, changed)
    assert len(signatures) == 1
    assert st.mixed_conditions({"c": signatures}, args_by_digest=args_by_digest) == []


@pytest.mark.parametrize("changed, differing", [
    ({"max_model_len": 8192}, {"max_model_len": (40960, 8192)}),
    ({"attention_config": {"backend": "TRITON_ATTN"}},
     {"attention_config": ({"backend": "FLASH_ATTN"}, {"backend": "TRITON_ATTN"})}),
    # canonical JSON, the digest's predicate, as the runner compares too: 40960.0 is not 40960, nor 0 False
    ({"max_model_len": 40960.0}, {"max_model_len": (40960, 40960.0)}),
    ({"enable_lora": 0}, {"enable_lora": (False, 0)}),
])
def test_a_condition_whose_draws_ran_other_binding_engine_arguments_is_mixed(tmp_path, changed, differing):
    expected = {f"engine_args.{name}": pair for name, pair in differing.items()}
    assert st.engine_args_changes({"engine_args": ONE_GPU}, {"engine_args": {**TWO_GPUS, **changed}}) == expected
    signatures, args_by_digest = _two_gpu_draws(tmp_path, changed)
    assert len(signatures) == 2
    records = st.mixed_conditions({"c": signatures}, args_by_digest=args_by_digest)
    assert [(record["scope"], record["conditions"]) for record in records] == [(st.CONDITION_SCOPE, ["c"])]
    digest = signatures[1]["engine_args"]
    assert records[0]["reason"].endswith(f"engine_args={digest}; {st.describe_changes(expected)})")
    # without the arguments behind the digests, the digests alone
    assert st.mixed_conditions({"c": signatures})[0]["reason"].endswith(f"engine_args={digest})")


def test_the_arguments_behind_two_digests_are_named_when_both_are_known():
    one = {"max_model_len": 4096}
    first, second, third = ({**NEW, "engine_args": {**one, "enforce_eager": value}} for value in (False, True, None))
    signatures = [st.judgment_signature(runtime) for runtime in (first, second, third)]
    args_by_digest = {signature["engine_args"]: st.binding_engine_args(runtime)
                      for signature, runtime in zip(signatures, (first, second, third))}
    assert st.describe_engine_args(signatures, args_by_digest=args_by_digest) == (
        "engine_args.enforce_eager False -> True; engine_args.enforce_eager False -> None"
    )
    assert st.describe_engine_args(signatures[:1], args_by_digest=args_by_digest) == ""
    assert st.describe_engine_args(signatures, args_by_digest={}) == ""
    assert st.describe_engine_args([st.stack_signature(NEW), signatures[0]], args_by_digest=args_by_digest) == ""


def test_the_condition_record(tmp_path):
    assert st.read_condition_stack(tmp_path) is None and st.condition_stack(tmp_path) is None
    _draw(tmp_path, 0, runtime=NEW, ledger=None)
    path = st.write_condition_stack(
        tmp_path, runtime={**NEW, "engine_args": {"dtype": "float32"}}, condition_id="c", condition_fingerprint="f",
        source="draw-000: first run", recorded_at="t",
    )
    assert path.name == st.CONDITION_STACK_FNAME
    record = st.read_condition_stack(tmp_path)
    assert record["stack"] == st.stack_signature(NEW) and record["runtime"]["engine_args"] == {"dtype": "float32"}
    assert (record["condition_id"], record["source"], record["recorded_at"]) == ("c", "draw-000: first run", "t")
    assert st.condition_stack(tmp_path) is None
    _draw(tmp_path, 1, runtime=OLD)
    assert st.condition_stack(tmp_path) == record
    path.unlink()
    assert st.condition_stack(tmp_path) == {
        "stack": st.stack_signature(OLD), "runtime": OLD, "source": f"draw-001/{DRAW_PROVENANCE_FNAME}",
    }
    path.write_text("[]")
    with pytest.raises(ValueError, match="not a JSON object"):
        st.read_condition_stack(tmp_path)


def test_mixed_conditions_are_named_one_record_each():
    old, new = st.stack_signature(OLD), st.stack_signature(NEW)
    records = st.mixed_conditions({"a": [old], "b": [old, new], "c": [], "d": [new, old]})
    assert [record["conditions"] for record in records] == [["b"], ["d"]]
    assert records[0]["scope"] == st.CONDITION_SCOPE
    assert records[0]["reason"].startswith(
        "its judgments come from several stacks or engine configurations (vllm 0.19.0 / torch 2.10.0+cu128"
    )


def test_members_share_a_stack_unless_two_of_them_conflict():
    old, new = st.stack_signature(OLD), st.stack_signature(NEW)
    assert st.shared_stack_conflicts({"bank": [old], "s1": [old], "provider": []}, scope="m") == []
    records = st.shared_stack_conflicts({"bank": [old], "s1": [old], "s2": [new], "s3": [new]}, scope="m")
    assert [(r["scope"], r["conditions"]) for r in records] == [("m", ["bank", "s2"]), ("m", ["bank", "s3"])]
    assert records[0]["reason"].startswith(
        "vllm 0.19.0 / torch 2.10.0+cu128 / transformers 5.5.0 against vllm 0.28.0+cu129"
    )
    records = st.shared_stack_conflicts({"bank": [old, new], "s1": [old]}, scope="m")
    assert [(r["scope"], r["conditions"]) for r in records] == [(st.CONDITION_SCOPE, ["bank"])]
    # two members that each agree with the bank (it names no transformers) but not with each other
    partial = {"vllm": "0.28.0+cu129", "torch": "2.13.0+cu129", "transformers": None}
    five = dict(partial, transformers="5.5.0")
    seventeen = dict(partial, transformers="5.17.0")
    records = st.shared_stack_conflicts({"bank": [partial], "s1": [five], "s2": [seventeen]}, scope="m")
    assert [r["conditions"] for r in records] == [["s1", "s2"]]


def test_an_analysis_output_carries_the_stack_keys_only_under_the_override():
    record = st.mismatch_record(st.CONDITION_SCOPE, ["c"], "mixed")
    assert st.override_record(False, []) == {} and st.stacks_record(False, [st.stack_signature(OLD)]) == {}
    assert st.override_record(True, []) == {st.ALLOW_STACK_MISMATCH_KEY: True, st.STACK_MISMATCHES_KEY: []}
    assert st.override_record(True, [record]) == {st.ALLOW_STACK_MISMATCH_KEY: True, st.STACK_MISMATCHES_KEY: [record]}
    assert st.stacks_record(True, [st.stack_signature(OLD)]) == {st.INFERENCE_STACKS_KEY: [OLD_SIGNATURE]}


def test_mismatches_are_described_briefly_and_read_back():
    records = [st.mismatch_record("s", [f"a{i}", f"b{i}"], f"r{i}") for i in range(5)]
    assert st.describe_mismatches(records[:1]) == "a0 vs b0: r0"
    assert st.describe_mismatches(records) == "a0 vs b0: r0; a1 vs b1: r1; a2 vs b2: r2; and 2 more"
    assert st.recorded_mismatches({}) == [] and st.recorded_mismatches({st.STACK_MISMATCHES_KEY: None}) == []
    assert st.recorded_mismatches({st.STACK_MISMATCHES_KEY: records}) == records


def test_a_producer_refuses_estimates_that_report_a_mismatch():
    assert st.recorded_mismatch_refusal({}, "x") is None
    assert st.recorded_mismatch_refusal({st.ALLOW_STACK_MISMATCH_KEY: True, st.STACK_MISMATCHES_KEY: []}, "x") is None
    refusal = st.recorded_mismatch_refusal(
        {st.STACK_MISMATCHES_KEY: [st.mismatch_record(st.CONDITION_SCOPE, ["c"], "mixed")]}, "est.json"
    )
    assert refusal.startswith("est.json reports 1 inference-stack mismatch(es), analysed under allow_stack_mismatch (c: mixed)")
    assert "--allow-stack-mismatch" in refusal
