from __future__ import annotations

import hashlib
import importlib

import pytest

FROZEN_TEMPLATE_SHA256 = {
    "naive-rules-above": "ed3e0c4ff31640489014a7169427014ef6f056904019b47890cc6c3b9be29802",
    "encouraged": "50b7ad08885cc15904ee76283f4e30914ca874bd50283079ec7084ff248201d4",
    "forced": "a69711aa21689b26d6c1e6f7fa3cdf6def844b72c3c6afab5c0cf96db28cec38",
    "forced-grpo": "9e7748a0eddecd42d66791a2db025c8a8ed89c6e53870758ce62dfd768e10e86",
}


@pytest.fixture(scope="module")
def driver(repo_root):
    module = importlib.import_module("ruleforge.experiments.prompt_ablation.templates")
    return module


def test_templates_use_paper_names_and_semantic_placeholders(driver):
    templates = driver.render_templates()
    assert list(templates) == ["naive-rules-above", "encouraged", "forced", "forced-grpo"]
    for template in templates.values():
        assert template.count("<ruleset>") == 1
        assert template.count("<original_prompt>") == 1
        assert template.count("<ai_response>") == 1
        assert "None" not in template
        assert template.endswith("\n")


def test_templates_preserve_the_three_experimental_formats(driver):
    templates = driver.render_templates()
    assert driver.render_template(driver.NAIVE_RULES_ABOVE) == templates["naive-rules-above"]
    assert driver.render_template(driver.ENCOURAGED) == templates["encouraged"]
    assert driver.render_template(driver.FORCED) == templates["forced"]
    assert driver.render_template(driver.FORCED_GRPO) == templates["forced-grpo"]
    assert driver.PLACEHOLDER_RULESET in templates["naive-rules-above"]
    assert "go through every individual rule" not in templates["naive-rules-above"]
    assert "go through every individual rule" in templates["encouraged"]
    assert "do not include it in your answer" in templates["encouraged"]
    assert "Respond with ONLY a JSON object" not in templates["encouraged"]
    assert "Respond with ONLY a JSON object" in templates["forced"]
    assert '"<rule_id>": "yes|no|unsure"' in templates["forced"]
    assert "preempted" not in templates["encouraged"] + templates["forced"]
    assert '"<rule_id>": "yes|no|unsure"' in templates["forced-grpo"] and "preempted_rules" in templates["forced-grpo"]


def test_unknown_style_is_rejected(driver):
    with pytest.raises(ValueError, match="unknown prompt style"):
        driver.render_template("unknown")


def test_the_hierarchy_style_has_no_placeholder_template(driver):
    with pytest.raises(ValueError, match="'naive-hierarchy' renders only over a levelled ruleset"):
        driver.render_template("naive-hierarchy")


def test_template_wording_matches_published_frozen_digests(driver):
    actual = {
        paper_name: hashlib.sha256(template.encode()).hexdigest()
        for paper_name, template in driver.render_templates().items()
    }
    assert actual == FROZEN_TEMPLATE_SHA256


def test_writer_emits_only_the_paper_s_templates(driver, tmp_path):
    paths = driver.write_templates(tmp_path)
    assert [path.name for path in paths] == [
        "naive-rules-above.txt",
        "encouraged.txt",
        "forced.txt",
        "forced-grpo.txt",
    ]
    assert {path.name for path in tmp_path.iterdir()} == {
        "naive-rules-above.txt",
        "encouraged.txt",
        "forced.txt",
        "forced-grpo.txt",
    }


def test_writer_rejects_stale_txt_templates(driver, tmp_path):
    stale_path = tmp_path / "deprecated.txt"
    stale_path.write_text("stale", encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"unexpected prompt template file.*deprecated\.txt"
    ):
        driver.write_templates(tmp_path)

    assert stale_path.read_text(encoding="utf-8") == "stale"


def test_a_named_template_is_exported_only_when_named(driver):
    assert driver.TEMPLATES == {name: name for name in ("naive", "naive-rules-above", "encouraged", "forced", "forced-grpo")}
    templates = driver.render_templates(["naive-rules-above", "forced-grpo"])
    assert list(templates) == ["naive-rules-above", "forced-grpo"]
    assert {name: hashlib.sha256(text.encode()).hexdigest() for name, text in templates.items()} == {
        name: FROZEN_TEMPLATE_SHA256[name] for name in templates}
    assert driver.render_templates(["forced"])["forced"] == driver.render_templates()["forced"]
    assert driver.selected() == driver.paper_styles()


@pytest.mark.parametrize("names", [[], ["forced", "forced"], ["forced", "preempted"], ["unknown"],
                                   ["naive-hierarchy"], ["forced_grpo"]])
def test_a_template_selection_names_each_known_template_at_most_once(driver, names):
    with pytest.raises(ValueError, match="name each of .* at most once"):
        driver.selected(names)


def test_main_writes_the_named_templates_beside_a_campaign_and_refuses_the_paper_s_directory(driver, tmp_path, capsys):
    out = tmp_path / "templates"
    assert driver.main(["--output-dir", str(out), "--template", "naive-rules-above", "--template", "forced"]) == 0
    assert sorted(path.name for path in out.iterdir()) == ["forced.txt", "naive-rules-above.txt"]
    assert hashlib.sha256((out / "naive-rules-above.txt").read_bytes()).hexdigest() == FROZEN_TEMPLATE_SHA256["naive-rules-above"]
    assert capsys.readouterr().out.count("wrote ") == 2
    paper = tmp_path / "paper"
    driver.write_templates(paper)
    with pytest.raises(SystemExit):  # the paper's other files are unexpected beside a one-template set
        driver.main(["--output-dir", str(paper), "--template", "forced"])
    assert "unexpected prompt template file(s): encouraged.txt, forced-grpo.txt, naive-rules-above.txt" in capsys.readouterr().err


def test_the_output_directory_is_named_every_time(driver, capsys):
    with pytest.raises(SystemExit):
        driver.main([])
    assert "--output-dir" in capsys.readouterr().err


def test_a_fault_while_rendering_keeps_its_traceback(driver, tmp_path, monkeypatch):
    def fail(names=None):
        raise TypeError("build_eval_prompt() missing 1 required positional argument")

    monkeypatch.setattr(driver, "render_templates", fail)
    with pytest.raises(TypeError, match="missing 1 required"):
        driver.main(["--output-dir", str(tmp_path / "t")])


def test_the_revised_templates_are_the_approved_texts_and_are_exported_only_when_named(driver, tmp_path):
    from tests.test_prompt_styles import APPROVED_REVISED_TEMPLATES

    names = ["encouraged", "forced"]
    templates = driver.render_templates(names)
    for name in names:
        approved = APPROVED_REVISED_TEMPLATES[name]
        assert templates[name] == approved.replace("{rule_text}", driver.PLACEHOLDER_RULESET).replace(
            "{original_prompt}", "<original_prompt>").replace("{response_to_evaluate}", "<ai_response>")
    out = tmp_path / "templates"
    assert driver.main(["--output-dir", str(out), *(f"--template={name}" for name in names)]) == 0
    assert {path.name: path.read_text(encoding="utf-8") for path in out.iterdir()} == {
        f"{name}.txt": templates[name] for name in names}
    assert driver.selected() == driver.paper_styles()


def test_the_paper_s_files_keep_the_published_naive_text_and_take_the_revised_encouraged_and_forced(driver):
    from tests.test_prompt_styles import APPROVED_REVISED_TEMPLATES

    revised = driver.render_templates()
    assert list(revised) == ["naive-rules-above", "encouraged", "forced", "forced-grpo"]
    assert revised["naive-rules-above"] == driver.render_templates(["naive-rules-above"])["naive-rules-above"]
    assert hashlib.sha256(revised["naive-rules-above"].encode()).hexdigest() == FROZEN_TEMPLATE_SHA256["naive-rules-above"]
    for name in ("encouraged", "forced"):
        assert revised[name] == driver.render_templates([name])[name]
        assert revised[name] == APPROVED_REVISED_TEMPLATES[name].replace("{rule_text}", driver.PLACEHOLDER_RULESET).replace(
            "{original_prompt}", "<original_prompt>").replace("{response_to_evaluate}", "<ai_response>")
def test_only_the_paper_s_set_goes_under_the_paper_s_directories(driver, tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    monkeypatch.setattr(driver, "repository_root", lambda: repo)
    for argv in (["--template", "naive-rules-above", "--output-dir", str(repo / "results")],
                 ["--output-dir", str(repo / "results" / "three"), *(f"--template={name}" for name in (
                     "naive-rules-above", "encouraged", "forced"))],
                 ["--template", "forced", "--output-dir", str(repo / "results" / "prompt_ablation" / "t")]):
        with pytest.raises(SystemExit):
            driver.main(argv)
        assert "hold(s) the paper's files; this output is not one of them" in capsys.readouterr().err
    assert not repo.exists()
    assert driver.main(["--output-dir", str(repo / "results" / "b")]) == 0
    assert driver.main(["--output-dir", str(repo / "results" / "c"), *(f"--template={name}" for name in (
        "naive-rules-above", "encouraged", "forced", "forced-grpo"))]) == 0
    assert driver.main(["--template", "forced", "--output-dir", str(repo / "figures")]) == 0
    assert driver.main(["--template", "forced", "--output-dir", str(repo / "scratch" / "t")]) == 0
