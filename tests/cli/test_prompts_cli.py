from __future__ import annotations

import pytest

from ruleforge.experiments.prompt_ablation.templates import TEMPLATES


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Every template the export program writes, by its file's name."""
    from ruleforge.experiments.prompt_ablation.templates import main

    out = tmp_path_factory.mktemp("templates")
    argv = [arg for name in TEMPLATES for arg in ("--template", name)]
    assert main([*argv, "--output-dir", str(out)]) == 0
    return {name: (out / f"{name}.txt").read_text(encoding="utf-8") for name in TEMPLATES}


@pytest.mark.parametrize("task", [(), ("--task", "modelspec")], ids=["default-task", "modelspec"])
@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_a_style_s_template_is_the_export_s_file(cli, exported, name, task):
    status, out, _ = cli("prompts", "show", TEMPLATES[name], *task)
    assert status == 0 and out == exported[name]


def test_a_record_era_key_is_no_style_of_the_command(capsys):
    from ruleforge.cli import main
    from ruleforge.core.record_names import RECORD_KEYS, RETIRED_KEYS

    for key in sorted(RECORD_KEYS | RETIRED_KEYS):
        try:
            status = main(["prompts", "show", key])
        except SystemExit as stop:
            status = stop.code
        captured = capsys.readouterr()
        assert (status, captured.out) == (2, ""), key
        assert f"'{key}'" in captured.err, key


def test_an_interpretation_is_rendered_and_a_levelled_refused_record_only_or_unknown_style_refused(cli):
    status, out, _ = cli("prompts", "show", "naive", "--interpretation", "Narrow")
    assert status == 0 and "<ruleset>" in out and "<original_prompt>" in out
    status, _, error = cli("prompts", "show", "naive-hierarchy")
    assert status == 2 and "judges each rule with the rules that outrank it" in error
    assert "has no placeholder template" in error
    status, _, error = cli("prompts", "show", "forced", "--interpretation", "Narrow")
    assert status == 2 and "is defined for the No_Interpretation template only" in error
    status, _, error = cli("prompts", "show", "nope")
    assert status == 2 and "Invalid reasoning_prompt_style: 'nope'" in error and "placeholder" not in error
    assert "valid: ['encouraged', 'forced', 'forced-grpo', 'naive', 'naive-hierarchy', 'naive-rules-above']" in error
    status, _, error = cli("prompts", "show", "json", "--task", "tests.test_tasks:ArchivedDemoTask")
    assert status == 2 and "'json' of task 'demo-archived@1.0' is recognised only to read the records that ran it" in error
    status, out, _ = cli("prompts", "show", "naive-rules-above", "--interpretation", "Narrow")
    assert status == 0 and "<ruleset>" in out


def test_a_task_s_other_fields_are_placeholders_by_their_names(cli, monkeypatch):
    from ruleforge.tasks.modelspec import ModelSpecTask

    monkeypatch.setattr(ModelSpecTask, "scenario_fields", ("prompt", "response", "context"))
    seen = []
    monkeypatch.setattr(ModelSpecTask, "render", lambda self, rules, scenario, settings: seen.append(scenario) or "")
    assert cli("prompts", "show", "naive")[0] == 0
    assert seen == [{"prompt": "<original_prompt>", "response": "<ai_response>", "context": "<context>"}]
