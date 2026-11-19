from __future__ import annotations

import hashlib
import json

from ruleforge.experiments import arm_records as ar


class _Condition:
    def __init__(self, **fields):
        self.fields = fields

    def to_dict(self):
        return dict(self.fields)


def test_two_conditions_differ_beyond_their_style_only_in_the_fields_named():
    a = _Condition(condition_id="m/naive", prompt_style="naive", model="m", label_rule="lenient", levels="l.yaml")
    same = _Condition(**{**a.fields, "condition_id": "m/encouraged", "prompt_style": "encouraged"})
    assert ar.differences_beyond_style(a, same) == []
    other = _Condition(**{**a.fields, "prompt_style": "encouraged", "model": "n", "label_rule": "strict"})
    assert ar.differences_beyond_style(a, other) == ["label_rule", "model"]
    # A field one side lacks differs; the label and the style never do.
    assert ar.differences_beyond_style(a, _Condition(condition_id="x", prompt_style="encouraged", model="m",
                                                     label_rule="lenient")) == ["levels"]
    assert ar.differences_beyond_style(a, other, exempt=("model", "label_rule")) == ["prompt_style"]
    assert ar.differences_beyond_style(a, other, exempt=("prompt_style", "model", "label_rule")) == []


def test_the_bootstrap_settings_are_the_campaign_s_as_contrast_row_takes_them():
    payload = {"bootstrap": {"n_resamples": "2000", "ci_level": 0.95, "seed": 7.0, "method": "two_stage"},
               "randomization_test": {"n_permutations": 9999, "seed": 3}}
    settings = ar.bootstrap_settings(payload)
    assert settings == {"n_resamples": 2000, "ci_level": 0.95, "seed": 7, "n_permutations": 9999}
    assert [type(settings[key]) for key in ("n_resamples", "ci_level", "seed", "n_permutations")] == [
        int, float, int, int]


def test_a_root_s_label_replay_is_named_by_the_bytes_it_was_read_from(tmp_path):
    assert ar.replay_provenance(tmp_path) is None
    record = {"tool": "replay", "code_state_sha256": "c" * 64, "source_label_rule": "lenient",
              "label_rule": "lenient_last_line", "sources": ["x"]}
    data = json.dumps(record, indent=1).encode("utf-8")
    (tmp_path / "label_replay.json").write_bytes(data)
    assert ar.replay_provenance(tmp_path) == {
        "name": "label_replay.json", "sha256": hashlib.sha256(data).hexdigest(), "tool": "replay",
        "code_state_sha256": "c" * 64, "source_label_rule": "lenient", "label_rule": "lenient_last_line"}
    (tmp_path / "label_replay.json").write_text(json.dumps({"tool": "replay"}))
    provenance = ar.replay_provenance(tmp_path)
    assert provenance["tool"] == "replay" and provenance["label_rule"] is None
