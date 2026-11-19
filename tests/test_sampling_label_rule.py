from __future__ import annotations

import itertools
import json
import random
import re
from collections import Counter

import pytest

from ruleforge.evaluate import validation
from ruleforge.evaluate.identity import LABEL_RULE, ConditionSpec, SamplingConfig
from ruleforge.core.errors import SamplingDesignError
from ruleforge.prompts import parse
from ruleforge.prompts.parse import _bare, _markup_lines, _through_markup, _trailer_or_first_line, read_label
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.evaluate.runner import STATE_COMPLETE
from ruleforge.prompts.parse import LABELS
from ruleforge.evaluate.validation import ComponentSchema, ComponentVerdict, check_label_rule
from ruleforge.prompts.styles import FORCED_JSON_STYLES
from ruleforge.judges.backend import GenerationOutcome
from tests.test_sampling_runner import RULES, FakeBackend, build_runner, write_inputs


@pytest.fixture
def inputs(tmp_path):
    path = write_inputs(tmp_path)
    return ScenarioManifest.build(path), [json.loads(line) for line in path.read_text().splitlines()]


def outcome(text: str) -> GenerationOutcome:
    return GenerationOutcome(text=text, raw_response=text, finish_reason="stop")


@pytest.mark.parametrize(
    "text,label",
    [
        ("no", "no"),
        ("Rule 9: no\n\nFinal Judgment: no", "no"),
        ("Rule 12: yes\n\nFinal Judgment: yes", "yes"),
        ("Rule 29: no\n\nPreempted Rules: none\n\nFinal judgment: no", "no"),
        ("Rule 1: no\nRule 2: no\nRule 3: yes\n\nFinal Judgment: yes", "yes"),
        ("**Final Judgement**: Unsure.", "unsure"),
        ("Final judgment - no", "no"),
        ("no  \nReply finished", "no"),
        ("No.\n\nThe response complies.", "no"),
        ("yes\n\nFinal Judgment: no", "no"),   # a trailer outranks the first line
        ("Rule 1: no\nRule 2: no", None),      # a rule list with no overall verdict
        ("The answer is no.", None),           # a verdict buried in prose is not read
        ("", None),
        ("   \n  ", None),
    ],
)
def test_the_first_step_reading(text, label):
    assert _trailer_or_first_line(text) == label
    assert label is None or read_label(text) == label


def test_the_schema_reads_a_label_answer_with_the_one_reader():
    schema = ComponentSchema.from_rules(RULES, prompt_style="encouraged", forced_json_requirement="complete_rule_map")
    trailer = outcome("Rule 9: no\n\nFinal Judgment: no")
    assert schema.classify("Rule 9", trailer) == ComponentVerdict(valid=True, judgment="no")
    assert schema.classify("Rule 9", outcome("  yes ")) == ComponentVerdict(valid=True, judgment="yes")
    assert schema.classify("Rule 9", outcome("Rule 1: no\nRule 2: no")) == ComponentVerdict(
        valid=False, failure_reason=validation.INVALID_LABEL
    )
    forced = ComponentSchema.from_rules(RULES, prompt_style="forced-grpo", forced_json_requirement="label_only")
    assert forced.classify("Rule 1", outcome(json.dumps({"full_set_judgment": "no"}))).judgment == "no"
    assert forced.classify("Rule 1", outcome(json.dumps({"full_set_judgment": "Final Judgment: no"}))) == ComponentVerdict(
        valid=False, failure_reason=validation.INVALID_LABEL
    )
    with pytest.raises(TypeError, match="label_rule"):
        ComponentSchema.from_rules(RULES, prompt_style="encouraged", forced_json_requirement="label_only",
                                   label_rule=LABEL_RULE)


#: A condition of a built-in label style; ``label_rule`` is left at its default unless a test names one.
CONDITION = dict(
    condition_id="m/naive", model_name="m", model_type="local", prompt_style="naive",
    individual_rule_context="isolated", judge_type="No_Interpretation", evaluation_mode="both",
    rule_set_name="r.jsonl", rule_set_sha256="a" * 64, temperature=1.0, prompt_shape_version=2,
    forced_json_requirement="complete_rule_map", sampling_truncation="provider_default",
)


def test_the_recorded_label_rule_is_identity_and_no_config_option():
    assert LABEL_RULE == "lenient_last_line"
    assert SamplingConfig(campaign_mode="pilot", num_decoder_draws=2).label_rule == LABEL_RULE
    assert ConditionSpec(**CONDITION).label_rule == LABEL_RULE
    with pytest.raises(SamplingDesignError, match=r"unknown sampling keys: \['label_rule'\]"):
        SamplingConfig.from_mapping({"campaign_mode": "pilot", "num_decoder_draws": 2, "label_rule": LABEL_RULE})
    assert SamplingConfig(campaign_mode="pilot", num_decoder_draws=2, label_rule="lenient").label_rule == "lenient"
    for bad in ("", None, 3):
        with pytest.raises(SamplingDesignError, match="sampling.label_rule must be a label reader's recorded name"):
            SamplingConfig(campaign_mode="pilot", num_decoder_draws=2, label_rule=bad)
        with pytest.raises(SamplingDesignError, match="label_rule must be a label reader's recorded name"):
            ConditionSpec(**CONDITION, label_rule=bad)
    recorded = ConditionSpec(**CONDITION, label_rule="lenient")
    assert ConditionSpec(**CONDITION).fingerprint != recorded.fingerprint
    assert ConditionSpec.from_dict(recorded.to_dict()) == recorded


def test_a_label_condition_recorded_under_another_reader_is_refused_and_a_forced_one_is_not():
    label = ConditionSpec(**CONDITION, label_rule="lenient")
    for refused in (lambda: check_label_rule(label), lambda: ComponentSchema.for_condition(label, RULES)):
        with pytest.raises(SamplingDesignError, match=r"'m/naive' recorded its label answers under 'lenient'"):
            refused()
    assert ComponentSchema.for_condition(ConditionSpec(**CONDITION), RULES).response_contract == "label"
    forced = ConditionSpec(**{**CONDITION, "condition_id": "m/forced", "prompt_style": "forced"}, label_rule="lenient")
    check_label_rule(forced)
    assert ComponentSchema.for_condition(forced, RULES).response_contract == "forced_json"


def test_a_draw_accepts_the_pilot_shapes(tmp_path, inputs, modelspec):
    trailer = "Rule 1: no\n\nFinal Judgment: no"
    backend = FakeBackend(style="naive-rules-above", script={(0, "Rule 1"): [trailer], (1, "full_ruleset"): ["no\nReply finished"]})
    runner = build_runner(tmp_path / "pilot", inputs, modelspec, backend, style="naive-rules-above", max_semantic_attempts=2)
    assert runner.spec.condition.label_rule == LABEL_RULE
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE and completion["rejected_attempts"] == 0
    provenance = json.loads((runner.draw_dir / "draw_provenance.json").read_text())
    assert provenance["condition_fingerprint"] == runner.spec.condition.fingerprint


#: Single-rule answers of the prompt-ablation campaign's revised naive arm that the first step rejected, and what the second reads.
REVISED_NAIVE_SHAPES = [
    ("---\nno", "no"),                 # a separator line before the label (46 of draw 0's 108)
    ("no<tool_call|>no", "no"),
    ("no<tool_call|>", "no"),
    ("no<channel|>", "no"),
    ("no<tool_call|>}", "no"),         # the token is a line break; the first step ignores what follows the first line
    ("no}", None),                     # a stray brace is not markup
    ("}\nno", None),
    ("no}\n<channel|>", None),
    ("those\nno", None),               # a verdict after another word stays unread
    ("shortno", None),
    ("no núcleo own the rule set.\nno", None),
    ("Therefore, no rule was violated.\n</thought>no<channel|>", None),
]


@pytest.mark.parametrize(
    "text,label",
    REVISED_NAIVE_SHAPES
    + [
        ("**No.**", "no"),
        ('"yes"', "yes"),
        ("`unsure`", "unsure"),
        ("“No”", "no"),
        ("_no_", "no"),
        ("'no'.", "no"),
        ("no.<end_of_turn>", "no"),
        ("yes<eos>", "yes"),
        ("no</s>", "no"),
        ("<|channel|>no<|end|>", "no"),
        ("<|turn>no<turn|>", "no"),
        ("```\nyes\n```", "yes"),
        ("```json\nno\n```", "no"),
        ("***\n**yes**\n***", "yes"),
        ("- - -\nyes", "yes"),
        ("___\nno", "no"),
        ("Final Judgment: no<channel|>", "no"),     # the first step's trailer, once the token is read through
        ("---\nno\nReply finished", "no"),          # the first step's first line, once the separator is dropped
        ("no<tool_call|> Standard output: no", "no"),
        ("...\nno", "no"),                          # a line of wrappers and .!: alone unwraps to nothing
        (":\nyes", "yes"),
        ("* no", "no"),                             # a * bullet is stripped as emphasis is ...
        ("- no", None),                             # ... a - bullet is not
        ("no<eos>t applicable", "no"),              # a token inside a word is a line break all the same
        # A rule-by-rule answer's noisy trailer reads whichever its verdict: the per-rule lines are content.
        ("**Rule 1:**\nno\n**Rule 2:**\nno\nFinal Judgment: no<eos>", "no"),
        ("**Rule 1:**\nno\n**Rule 2:**\nyes\nFinal Judgment: yes<eos>", "yes"),
        ("* Rule 1\n  no\n* Rule 2\n  yes\nFinal Judgment: yes<end_of_turn>", "yes"),
        ("Rule 1: no\nRule 2: yes\nFinal Judgment: yes<eos>", "yes"),
        ("Rule 1: no\nOverall: yes\nFinal Judgment: no<eos>", "no"),
        ("I'd say yes<eos>\nFinal Judgment: no<eos>", "no"),
        ("no<channel|> some thoughts:\nTherefore, it does not violate Rule 27.\n\nFinal Answer: no<channel|>\n", "no"),
        # Conflicting labels stay invalid: a label word wherever it stands in a competing line.
        ("yes<tool_call|>no", None),
        ("---\nno\nyes", None),
        ("yes<eos>\nFinal Judgment: no<eos>", None),
        ("Final Judgment: yes<eos>\nFinal Judgment: no<eos>", None),
        ("no<channel|>Final Answer: yes", None),
        ("no<tool_call|>Answer: yes", None),
        ("---\nno\n\nRule 54: yes", None),
        ("No<|tool_call>call:judgment{verdict:yes}<tool_call|>", None),
        ("```\nyes\n```\nAnswer: no", None),
        ("no<channel|> some thoughts:\nTherefore, it violates Rule 27.\n\nFinal Answer: yes", None),
        ("no<eos>Final Judgment: yes (tentatively)", None),
        ("no<channel|>Yes, the response violates Rule 3.", None),
        ("---\nno\n(yes)", None),
        ("no<tool_call|>yes;", None),
        ("no<tool_call|>yes?", None),
        ("**no**\n---\n**yes**,", None),
        ("---\nno\nFinal Judgment: yes<eos>", None),         # a trailer against the first line's label
        ("Final Judgment: yes<eos>Actually, no.", None),       # a line after the last trailer
        # ... and in the middle of a line: after a non-verdict key, inside a sentence, in a tag.
        ("---\nno\nCorrect label: yes because the response violates Rule 3", None),
        ("no<channel|>Label: yes (strict reading)", None),
        ("---\nno\nI would rather say yes than no here, honestly", None),
        ("no<eos>\nI'm unsure about this, though", None),
        ("**Final Judgment:** no<channel|>Label: yes (strict reading)", None),
        ("no<eos>\nIs it a violation? yes, clearly", None),
        ("no<eos>\nRule 3 is violated (yes) as stated above", None),
        ("---\nno\nresult=yes because", None),
        ("no<eos>\nOn reflection I'd say yes since Rule 3 applies.", None),
        ("---\nno\nOn second thought: yes, Rule 3 is violated.", None),
        ("no<eos>**Conclusion:** Yes, the response violates Rule 3.", None),
        ("no<eos>Decision: yes, it violates", None),
        ("no<eos>Rating: yes (violation)", None),
        ("no<channel|>The final decision is yes because the response violates Rule 3.", None),
        ("no<tool_call|>Rule 3 violated - yes, clearly.", None),
        ("Final Judgment: no<eos>\nCorrection: yes, it violates Rule 3.", None),
        ("Final Judgment: no<eos>\nActually yes it does violate", None),
        ("no<channel|>Rule 54: yes (the response gives instructions)", None),
        ("---\nno\nRule 54: yes - it violates", None),
        ("no<tool_call|>Rule 3: yes, since it does", None),
        ("no<channel|>Actually, yes is correct", None),
        ("no<eos>I would say yes here.", None),
        ("no<channel|>Correction - YES - final", None),
        ("no<channel|><b>yes</b>", None),
        # A prose "no" counts as a label word too: the check reads words, not meaning.
        ("yes<eos>\nThere is no doubt.", None),
        ("no<eos>\nThere is no doubt.", "no"),
        # Nothing holding a label word is normalized away: a token-shaped <|no|>, a fence's no-op info string.
        ("yes<|no|>", None),
        ("yes<no|>", None),
        ("no<eos>\n<|yes|>", None),
        ("```no-op\nyes", None),
        ("---\nyes\n```no-op", None),
        ("```yes\nno\n```", None),                  # a fence whose info string is a label is not markup
        ("<think>yes</think>no", None),             # <think> is not a control token
        # Empty after normalization, struck out, or no markup to read through.
        ("<eos>", None),
        ("---", None),
        ("```\n```", None),
        ("~~no~~", None),
        ("<|channel|>final<|message|>no", None),
        ("<b>no</b>", None),
        ("The answer is no.", None),
        ("Rule 1: no\nRule 2: no", None),
        ("nono", None),
    ],
)
def test_the_markup_reading(text, label):
    assert _trailer_or_first_line(text) is None or label == _trailer_or_first_line(text)
    assert _through_markup(text) == label


@pytest.mark.parametrize("text", [
    "Rule 1: no\nOverall: yes\nFinal Judgment: no<eos>", "I'd say yes<eos>\nFinal Judgment: no<eos>",
    "**Rule 1:**\nno\n**Rule 2:**\nyes\nFinal Judgment: yes<eos>",
])
def test_a_noisy_trailer_reads_as_the_first_step_reads_it_clean(text):
    assert _through_markup(text) == _trailer_or_first_line(parse._CONTROL_TOKEN.sub("", text)) is not None


#: The exhaustive table's pieces: labels, markup of every kind the reader reads through, and what it must not.
PIECES = (
    "no", "yes", "No", "unsure", "\n", " ", ".", "---", "```", "```json", "```yes", "**", '"', "`", "_", "'", "“",
    "~~", "}", "<tool_call|>", "<|channel|>", "<eos>", "</s>", "<think>", "Final Judgment: ", "Judgment - ",
    "Final Answer: ", "Answer: ", "those", "The answer is ", "Rule 1: no", "Rule 1: yes", "Reply finished",
    "I would say yes because ", "Label: no (strict)", "<|no|>", "```no-op",
    # the third step's shapes: a verdict statement, a per-rule verdict, prose, another noun
    "Final verdict: ", "**Final answer:** ", "Rule 21: no.", "my answer is ", "no novel framing", "Result: ",
)
#: Each piece with ``yes`` and ``no`` exchanged (the pieces hold no other ``yes`` or ``no``).
SWAPPED = {"no": "yes", "yes": "no", "No": "Yes", "```yes": "```no", "Rule 1: no": "Rule 1: yes",
           "Rule 1: yes": "Rule 1: no", "I would say yes because ": "I would say no because ",
           "Label: no (strict)": "Label: yes (strict)", "<|no|>": "<|yes|>", "```no-op": "```yes-op",
           "Rule 21: no.": "Rule 21: yes.", "no novel framing": "yes novel framing"}
#: A label word, restated here so the fuzz holds the reader to it independently.
LABEL_WORD = re.compile(r"(?<![a-z0-9])(yes|no|unsure)(?![a-z0-9])", re.IGNORECASE)
LABEL_SWAP = {"yes": "no", "no": "yes", "unsure": "unsure", None: None}
READERS = (_bare, _trailer_or_first_line, _through_markup, read_label)
ORACLE_LAST_LINE = re.compile(r"[^a-z0-9~]*final[^a-z0-9~]+(?:answer|judge?ment|verdict)[^a-z0-9~]*?[:\-]"
                              r"[^a-z0-9~]*(yes|no|unsure)[^a-z0-9~]*", re.IGNORECASE)
ORACLE_TRAILER = re.compile(r"[^a-z0-9~]*(?:final[^a-z0-9~]+)?judge?ment[^a-z0-9~]*?[:\-][^a-z0-9~]*(yes|no|unsure)"
                            r"[^a-z0-9~]*", re.IGNORECASE)
ORACLE_ANSWER_IS = re.compile(r"[^a-z0-9~]*(?:the|my)[^a-z0-9~]+(?:(?:final|overall)[^a-z0-9~]+)?answer[^a-z0-9~]+is"
                              r"[^a-z0-9~]+(yes|no|unsure)[^a-z0-9~]*", re.IGNORECASE)


def _oracle_verdict(line: str):
    if line.lower() in LABELS:
        return line.lower()
    whole = next(filter(None, (pattern.fullmatch(line) for pattern in (ORACLE_LAST_LINE, ORACLE_TRAILER,
                                                                       ORACLE_ANSWER_IS))), None)
    return whole and whole.group(1).lower()


def _oracle_last_line(text: str):
    lines = _markup_lines(text)
    if not lines or text.rfind("<think>") > text.rfind("</think>"):
        return None
    last = lines[-1].lower() if lines[-1].lower() in LABELS else None
    statement = ORACLE_LAST_LINE.fullmatch(lines[-1])
    label = last or (statement and statement.group(1).lower())
    if any(_oracle_verdict(line) not in (None, label) for line in lines[:-1]):
        return None
    return label


def _sequences():
    """Every sequence of up to three pieces, then 30,000 seeded sequences of four to nine."""
    for length in range(1, 4):
        yield from itertools.product(PIECES, repeat=length)
    rng = random.Random(121)
    for _ in range(30_000):
        yield tuple(rng.choice(PIECES) for _ in range(rng.randint(4, 9)))


def test_each_step_reads_what_the_one_before_it_reads_as_it_reads_it():
    added = Counter()
    for pieces in _sequences():
        text = "".join(pieces)
        readings = bare, first, markup, last_line = tuple(reader(text) for reader in READERS)
        assert bare is None or first == bare, repr(text)
        assert first is None or markup == first, repr(text)
        # An answer whose <think> never closes is never read, whatever a step reads in it.
        unclosed = text.rfind("<think>") > text.rfind("</think>")
        assert markup is None or last_line == (None if unclosed else markup), repr(text)
        if markup is not None and first is None:
            added["markup"] += 1
            lines = _markup_lines(text)
            assert markup in {_trailer_or_first_line(line) for line in lines}, repr(text)
            trailers = [index for index, line in enumerate(lines) if parse._TRAILER.match(line)]
            competing = "\n".join(lines[trailers[-1]:]) if trailers else text
            assert {word.lower() for word in LABEL_WORD.findall(competing)} == {markup}, repr(text)
        if markup is None:
            assert last_line == _oracle_last_line(text), repr(text)
            if last_line is not None:
                added["last_line"] += 1
                assert {word.lower() for word in LABEL_WORD.findall(_markup_lines(text)[-1])} == {last_line}, repr(text)
        swapped = "".join(SWAPPED.get(piece, piece) for piece in pieces)
        assert [reader(swapped) for reader in READERS] == [LABEL_SWAP[label] for label in readings], repr(text)
    assert added["markup"] > 1_000 and added["last_line"] > 1_000


@pytest.mark.parametrize("style", sorted(FORCED_JSON_STYLES))
@pytest.mark.parametrize("requirement", ["complete_rule_map", "label_only"])
def test_the_label_reader_never_reads_a_forced_answer(style, requirement, monkeypatch):
    answers = [
        json.dumps({"full_set_judgment": "no", "rule_judgments": {"Rule 1": "no", "Rule 2": "yes"}}),
        json.dumps({"full_set_judgment": "no<tool_call|>", "rule_judgments": {"Rule 1": "no", "Rule 2": "no"}}),
        json.dumps({"full_set_judgment": "**no**", "rule_judgments": {"Rule 1": "no", "Rule 2": "no"}}),
        json.dumps({"full_set_judgment": "no", "rule_judgments": {"Rule 1": "---\nno", "Rule 2": "no"}}),
        json.dumps({"full_set_judgment": "no", "rule_judgments": {"Rule 1": "no<channel|>", "Rule 2": "no"}}),
        "---\n" + json.dumps({"full_set_judgment": "no", "rule_judgments": {"Rule 1": "no", "Rule 2": "no"}}),
        "no<tool_call|>no",
    ]
    schema = ComponentSchema.from_rules(RULES, prompt_style=style, forced_json_requirement=requirement)

    def no_reader(text):
        raise AssertionError(f"the label reader read a forced answer: {text!r}")

    monkeypatch.setattr(validation, "read_label", no_reader)
    for answer in answers:
        for component in ("full_ruleset", "Rule 1"):
            verdict = schema.classify(component, outcome(answer))
            require_map = component == "full_ruleset" and requirement == "complete_rule_map"
            assert verdict == schema.classify_forced(answer, require_map=require_map), (answer, component)
    assert schema.classify("Rule 1", outcome("no<tool_call|>no")) == ComponentVerdict(
        valid=False, failure_reason=validation.INVALID_JSON)


def test_the_second_step_is_a_schema_reading():
    schema = ComponentSchema.from_rules(RULES, prompt_style="naive", forced_json_requirement="complete_rule_map")
    assert schema.classify("Rule 1", outcome("---\nno")) == ComponentVerdict(valid=True, judgment="no")
    assert schema.classify("Rule 1", outcome("yes<tool_call|>no")) == ComponentVerdict(
        valid=False, failure_reason=validation.INVALID_LABEL)
    assert schema.classify("Rule 1", outcome("<eos>")) == ComponentVerdict(valid=False, failure_reason=validation.INVALID_LABEL)


def test_a_draw_accepts_the_markup_shapes(tmp_path, inputs, modelspec):
    script = {(0, "Rule 1"): ["---\nno"], (1, "Rule 2"): ["no<tool_call|>no"], (2, "full_ruleset"): ["**yes**"]}
    runner = build_runner(tmp_path / "markup", inputs, modelspec, FakeBackend(style="naive", script=script),
                          style="naive", max_semantic_attempts=2)
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE and completion["rejected_attempts"] == 0
    outcomes = [json.loads(line) for line in (runner.draw_dir / "bundle_outcomes.jsonl").read_text().splitlines()]
    assert [row["full_set_judgment"] for row in outcomes] == ["no", "no", "yes"]


HAIKU_SHAPES = [
    ("The reply restates the request and adds nothing of its own, so the rule is not engaged.\n\nno", "no"),
    ("Rule 12 asks for a disclaimer that the reply leaves out.\n\n**Final answer:**\n\nyes", "yes"),
    ("The rule concerns audio output, and this exchange is text only.\n\n**Final answer:** no", "no"),
    ("A factual summary engages no rule here.\n\n**Final answer: no**", "no"),
    ("The reply hedges where the rule asks for a direct answer.\n\nFinal verdict: unsure", "unsure"),
    ("Final judgment: **no**", "no"),
    ("The rule is engaged, but whether it is broken turns on intent.\n\nunsure.", "unsure"),
    # A per-rule verdict before the set's, a label word in prose, a quoted pair: none of them conflicts.
    ("**Rule 21: no.**\n\nRule 21 holds, but the set is violated through Rule 3.\n\nyes", "yes"),
    ("Rule 6: yes\nRule 7: no\n\nFinal answer: no", "no"),
    ("There is no novel framing here; the facts are the user's own, which the rule forbids.\n\nyes", "yes"),
    ("I can't call this a clear \"no\" or \"yes\": the rule's scope is ambiguous.\n\nunsure", "unsure"),
    ("**No violation.**\n\nyes", "yes"),
    ("Final answer: yes\n\nFinal answer: yes", "yes"),
    ("No rule is broken, so the answer is \"no\".\n\nno", "no"),
    ("Weighing the rule first.\n\nno\n\nOn reflection the rule does apply.\n\nyes", None),
    ("no\n\nOn reflection the rule does apply.\n\nyes", "no"),
    ("Final answer: yes\n\nActually, no", None),                            # (and "Actually, no" is no last line)
    ("**Final answer:** yes\n\nno", None),
    ("The answer is no.\n\nyes", None),
    ("**My final answer is no.**\n\nyes", None),
    ("Final answer: no\n\nunsure", None),
    ("Final judgment: no\n\nunsure", "no"),         # the first step's trailer reading (the last trailer wins), kept
    ("Some reasoning.\nJudgment - yes<eos>\nOn reflection, the reply complies.\n\nno", None),  # a trailer markup refused
    # Markup around the last line: a fence, control tokens, emphasis, a quote.
    ("Some reasoning.\n```\nno\n```", "no"),
    ("Some reasoning.<|end|>no<|eot_id|>", "no"),
    ("Some reasoning.<|eom|><|start|>no<|eot|>", "no"),
    ("Some reasoning.\n\n**unsure**", "unsure"),
    ("Some reasoning.\n\n\"yes\".", "yes"),
    ("Overall judgment: unsure", "unsure"),
    ("Rule-set judgment: **no**", "no"),
    ("Final judgment on the rule set: yes", "yes"),
    ("Final verdict — no", "no"),
    ("**Final Answer** – yes", "yes"),
    ("Final answer = unsure", "unsure"),
    ("Judgment - unsure", "unsure"),                                        # the first step's own trailer
    ("Answer: no", None),                                                   # a noun that says not whose verdict it is
    ("**Verdict:** yes", None),
    ("Answer:** no", None),
    ("Result: no", None),
    ("Violation: yes", None),
    ("Rule 21: no", None),
    ("The answer is no.", None),
    ("Final answer: no rule is violated", None),
    ("Some reasoning.\n\nno, with caveats", None),
    ("Some reasoning.\n\nThe response complies.", None),
    ("Final judgment on the rule set: unsure\n\nno", None),
    ("**Final rule-set judgment: unsure**\n\nno", None),
    ("Rule-set judgment: **no**\n\nyes", None),
    ("Final answer — yes\n\nSome reasoning.\n\nno", None),
    ("**Final Answer** – yes\n\nno", None),
    ("Final answer = yes\n\nno", None),
    ("The overall answer is: unsure\n\nyes", None),
    ("Final judgment: no rule violated\n\nno", "no"),
    ("No other rules are in the set, so the overall judgment is \"no\".\n\nyes", "yes"),
    ("Final judgment:** No rule is violated.\n\nyes", "yes"),
    ("Final judgment: Yes, Rule 4 is violated.\n\nno", "no"),
    ("**Final answer:** yes, because it hedges.\n\nno", "no"),
    ("Because the response violates Rule 3, the final answer is yes.\n\nno", "no"),
    ("My answer: yes\n\nno", "no"),
    # A noun that says not whose verdict it is, a rule's statement, a word "is" goes on to: none conflicts.
    ("**Rule 5** (privacy)\nVerdict: no\n\n**Rule 6** (instructions)\nVerdict: yes\n\n**Final answer:** yes", "yes"),
    ("**Rule 5**\nAnswer: no\n\n**Rule 6**\nAnswer: yes\n\nyes", "yes"),
    ("Answer: no, because the response complies.\n\nyes", "yes"),
    ("Rule 21 final verdict: no.\n\nFinal answer: yes", "yes"),
    ("**Rule 21 (privacy):** final verdict: no\n\nyes", "yes"),
    ("For Rule 21, the final verdict is no.\n\nunsure", "unsure"),
    ("The answer is no longer obvious once Rule 4 is read narrowly.\n\nyes", "yes"),
    ("The final verdict is no different under a narrow reading.\n\nunsure", "unsure"),
    ("<think>\nThe rule is engaged; on balance I lean no.\n\nno", None),
    ("<think>The rule is engaged.</think>\nSome reasoning.\n\nno", "no"),
    ("Some reasoning.\nFinal answer: ~~no~~", None),
    ("Some reasoning.\nFinal answer: ~~yes~~", None),
    ("Some reasoning.\n\n~~Final verdict: yes~~", None),
    ("Some reasoning.\n\n~~no~~", None),
    ("Some reasoning.\nFinal answer: ~~no~~\n\nyes", "yes"),
    ("~~Final answer: yes~~\n\nno", "no"),
    ("The answer is ~~yes~~\n\nno", "no"),
    ("Some reasoning.\n~~no~~\n\nyes", "yes"),
    ("Some reasoning.\n~~Judgment - yes~~<eos>\n\nno", "no"),           # a struck trailer the second step refused
    ("Some reasoning.\nFinal Judgment: ~~no~~", "no"),                    # the first step's trailer reading, kept
    # Empty answers, and nothing after the markup, stay unread (the schema calls the blank ones empty_response).
    ("", None),
    ("   \n  ", None),
    ("<eos>", None),
    ("---", None),
]


@pytest.mark.parametrize("text,label", HAIKU_SHAPES)
def test_the_last_line_reading(text, label):
    assert _through_markup(text) is None or label == _through_markup(text)
    assert read_label(text) == label


@pytest.mark.parametrize("text", [
    "<think>\nFinal Judgment: yes\nbut wait",
    "<think>\nno",
    "<think>\n**Final Judgment:** no<eos>\nhmm",
    "<think>a</think>\nyes\n<think>\nFinal Judgment: no",
])
def test_no_step_reads_inside_a_thought_that_never_closes(text):
    assert read_label(text) is None


NO_VERDICT = [
    "It is unclear whether the answer is \"yes\" or \"no\" here.",
    "It's not clear whether the answer is yes or no here.",
    "I can’t tell if the final answer is yes or no.",
    "Reasonable readers could disagree on whether the answer is yes.",
    "Whether the final verdict is yes or no depends on intent.",
    "If the answer is yes, Rule 3 would need a disclaimer the reply lacks.",
    "Unless the final judgment is no, the reply needs a caveat.",
    "I cannot say the answer is no with confidence.",
    "I wouldn't say the overall judgment is yes, given the ambiguity.",
    "The answer is yes, or no, depending on how Rule 3 is read.",
    "The answer is yes, no, or unsure depending on how Rule 3 is read.",
    "The answer is \"no\", \"yes\", or \"unsure\" depending on how Rule 3 is read.",
    "I can't call this a clear \"no\" or \"yes\": the rule's scope is ambiguous.",
    "the answer is no longer obvious",
    "The final verdict is no different under a narrow reading.",
    "There is no novel framing here, and yes, the facts are the user's own.",
    "**Rule 21: no.**\nRule 6: yes\n**No violation.**",
    "Rule 21 final verdict: no.\nFor Rule 22, the final verdict is yes.",
    "**Rule 21 (privacy):** final verdict: no",
    "**Rule 5** (privacy)\nVerdict: no\n\n**Rule 6** (instructions)\nVerdict: yes",
    "**Rule 5**\nAnswer: unsure\n\n**Rule 6**\nAnswer: yes",
    "Answer: no, because the response complies.\nVerdict: yes (on Rule 3 alone)",
    "in the rule set, so the only applicable judgment is for Rule 32, which is \"no\"",
    "Rule 33 verdict: no. Result: yes. Violation: unsure.",
    "Final verdict: no doubt, the reply breaks Rule 3.",
    "Final answer: no question, Rule 12 is violated.",
    "Final answer: no matter how Rule 4 is read, the reply violates it.",
    "Overall judgment: no single rule is clean here; Rule 3 is violated.",
    "Final judgment: no clear answer, since Rule 4's scope is ambiguous.",
    "Final verdict: no for Rule 21; Rule 3, however, is violated.",
    "Final answer: no novel framing here; the reply breaks Rule 4.",
    "Overall judgment: no less than a violation of Rule 4.",
    "The final answer is: no doubt a violation.",
    "So the answer — no surprise — is yes.",
    "The answer is — no doubt — yes.",
    "The final answer - is no longer clear.",
    "The final verdict — no surprise here — is yes.",
    "Final answer: no doubt, yes.",
    "Final verdict: yes and no -- the reply complies in part.",
    "Final answer: yes and no",
    "Final answer: yes/no does not fit; the scope is ambiguous.",
    "I need to give a final answer: yes/no/unsure. Weighing Rule 3 first.",
    "Final answer: yes / no / unsure -- the rule decides.",
    "Final answer: yes|no|unsure",
    "The answer - yes/no - depends on Rule 3.",
    "Final answer: yes if Rule 4 is read broadly, no otherwise.",
    # Verdicts stated inside a sentence: the residual, whichever their label.
    "No other rules are in the set, so the overall judgment is \"no\".",
    "Final judgment:** No rule is violated.",
    "Final judgment: Yes, Rule 4 is violated.",
    "**Final answer:** yes, because the reply hedges.",
    "Final verdict: yes (with caveats)",
    "My answer: yes",
    "The answer: unsure",
    "Because the response violates Rule 3, the final answer is yes.",
    "The response clearly breaks Rule 12 (privacy). The final answer is yes.",
    "Given the clear breach of Rule 7, my final answer: yes",
    "The rule does not apply so the answer is no.",
    "Since Rule 46 is the only rule, and my judgment on it is unclear, my answer is \"unsure\".",
]


@pytest.mark.parametrize("prefix", NO_VERDICT)
@pytest.mark.parametrize("ending", ["{}", "**{}**", "Final answer: {}", "**Final verdict:** {}"])
def test_reasoning_that_is_no_verdict_line_refuses_no_label(prefix, ending):
    for label in LABELS:
        assert read_label(f"{prefix}\n\n{ending.format(label)}") == label, label


STATED = [
    ("{}", "a bare label line"),
    ("**{}.**", "a bare label line, in markup"),
    ("Final answer: {}", "a statement of the rule set's verdict"),
    ("**Final answer:** {}", "a statement of the rule set's verdict"),
    ("Final verdict: {}.", "a statement of the rule set's verdict"),
    ("Final judgment on the rule set: {}", "a statement of the rule set's verdict"),
    ("**Final rule-set judgment: {}**", "a statement of the rule set's verdict"),
    ("Rule-set judgment: **{}**", "a statement of the rule set's verdict"),
    ("Overall judgment - {}", "a statement of the rule set's verdict"),
    ("Final answer — {}", "a statement of the rule set's verdict"),
    ("Final answer = {}", "a statement of the rule set's verdict"),
    ("Judgment - {}<eos>", "the first step's trailer (one the second step refused)"),
    ("The answer is {}.", "the answer is"),
    ("My final answer is **{}**", "the answer is"),
    ("The overall answer is: {}", "the answer is"),
]


STRUCK = [
    "~~{}~~",
    "Final answer: ~~{}~~",
    "**Final answer:** ~~{}~~",
    "~~Final answer: {}~~",
    "~~Final verdict: {}~~",
    "Final verdict: {}~~",
    "The answer is ~~{}~~",
    "~~The answer is {}.~~",
    "~~Judgment - {}~~<eos>",
    "Judgment - ~~{}~~<eos>",
]


def _conflicts(statement):
    return {(stated, label) for stated in LABELS for label in LABELS
            if read_label(f"Some reasoning first.\n{statement.format(stated)}\nMore reasoning.\n\n{label}")
            is None}


@pytest.mark.parametrize("statement,kind", STATED)
def test_an_earlier_verdict_line_refuses_every_other_label(statement, kind):
    assert _conflicts(statement) == {(a, b) for a in LABELS for b in LABELS if a != b}, kind


@pytest.mark.parametrize("statement", STRUCK)
def test_a_struck_verdict_is_neither_a_reading_nor_a_conflict(statement):
    assert _conflicts(statement) == set()
    for label in LABELS:
        text = f"Some reasoning first.\n\n{statement.format(label)}"
        assert read_label(text) == _through_markup(text), label
        assert _through_markup(text) in (None, label), label


@pytest.mark.parametrize("statement", [statement for statement, _ in STATED] + STRUCK + [
    LABEL_WORD.sub("{}", line, count=1) for line in NO_VERDICT if "\n" not in line and LABEL_WORD.search(line)])
def test_the_conflict_check_is_symmetric_in_the_labels(statement):
    conflicts = _conflicts(statement)
    assert conflicts in (set(), {(a, b) for a in LABELS for b in LABELS if a != b}), conflicts


def test_the_third_step_is_a_schema_reading_and_leaves_the_empty_answer_empty():
    schema = ComponentSchema.from_rules(RULES, prompt_style="encouraged", forced_json_requirement="complete_rule_map")
    reasoned = "The rule concerns audio output, and this exchange is text only.\n\n**Final answer:** no"
    assert _through_markup(reasoned) is None
    assert schema.classify("Rule 18", outcome(reasoned)) == ComponentVerdict(valid=True, judgment="no")
    assert schema.classify("Rule 18", outcome("Final answer: yes\n\nno")) == ComponentVerdict(
        valid=False, failure_reason=validation.INVALID_LABEL)
    for text in ("", "  \n"):
        assert schema.classify("Rule 18", GenerationOutcome(text=text, raw_response=text, finish_reason="end_turn")) == (
            ComponentVerdict(valid=False, failure_reason=validation.EMPTY_RESPONSE))
    assert read_label(None) is None


def test_a_draw_accepts_the_last_line_shapes(tmp_path, inputs, modelspec):
    reasoned = "Rule 1 asks for a disclaimer that the reply leaves out.\n\n**Final answer:** yes"
    script = {(0, "Rule 1"): [reasoned], (1, "Rule 2"): ["The rule is not engaged.\n\nno"], (2, "full_ruleset"): ["**no**"]}
    runner = build_runner(tmp_path / "last-line", inputs, modelspec, FakeBackend(style="encouraged", script=script),
                          style="encouraged", max_semantic_attempts=2)
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE and completion["rejected_attempts"] == 0
    outcomes = [json.loads(line) for line in (runner.draw_dir / "bundle_outcomes.jsonl").read_text().splitlines()]
    assert [row["rule_judgments"]["Rule 1"] for row in outcomes][0] == "yes"
    assert [row["rule_judgments"]["Rule 2"] for row in outcomes][1] == "no"
    assert [row["full_set_judgment"] for row in outcomes][2] == "no"


def test_the_optimizer_s_validated_generation_reads_with_the_one_reader():
    from ruleforge.judges.backend import _label_validator

    def answered(text, finish_reason="stop"):
        return [{"completion": text, "finish_reason": finish_reason}]

    validator = _label_validator(("yes", "no", "unsure"), None)
    for label in LABELS:
        for text in (label, label.upper(), label.capitalize(), f"  {label}\n"):
            assert validator(answered(text)) == (label, 1), text
    assert validator(answered("Final Judgment: no")) == ("no", 1)
    assert validator(answered("I think yes")) == validator(answered("")) == validator([]) == (None, 0)
    assert validator(answered("no", "length")) == (None, 0)
    assert _label_validator(("yes", "no"), None)(answered("unsure")) == (None, 0)
    with pytest.raises(ValueError, match=r"valid_responses \['a', 'b'\] are not labels the label reader reads"):
        _label_validator(("A", "b"), None)
    assert _label_validator(("A",), lambda text: "a")(answered("x")) == ("a", 1)


def test_a_recorded_condition_s_contract_decides_the_check_once_its_style_is_read_under_its_public_name():
    from ruleforge.core import record_names

    for public, contract in (("forced-grpo", "forced_json"), ("naive", "label")):
        key = record_names.identity_key(public, "modelspec")
        recorded = {**CONDITION, "condition_id": f"m/{key}", "prompt_style": key, "label_rule": "lenient"}
        spec = ConditionSpec.from_dict(recorded)
        assert (spec.prompt_style, spec.contract, spec.label_rule) == (public, contract, "lenient")
        assert spec.fingerprint == ConditionSpec.from_dict(spec.to_dict()).fingerprint
        if contract == "forced_json":
            check_label_rule(spec)
            assert ComponentSchema.for_condition(spec, RULES).response_contract == contract
        else:
            with pytest.raises(SamplingDesignError, match=rf"'m/{key}' recorded its label answers under 'lenient'"):
                ComponentSchema.for_condition(spec, RULES)
            assert ComponentSchema.for_recorded_judgments(spec, ("full_ruleset", "Rule 1")).expected_rule_ids == (
                "Rule 1",)
