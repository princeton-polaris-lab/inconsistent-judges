from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Tuple

from ruleforge.experiments.training.family_readout import SCHEMA_VERSION, SOURCE, FamilyArm
from ruleforge.core.digests import file_record
from ruleforge.core.paths import absolute_strings, repository_root

__all__ = [
    "QWEN_FAMILY",
    "QWEN_FIGURES",
    "QWEN_GOLDEN",
    "GEMMA_FAMILY",
    "GEMMA_FIGURES",
    "GEMMA_GOLDEN",
    "PAPER_OUTPUTS",
    "PaperFamily",
    "PaperOutputs",
    "drawn_names",
    "paper_refusal",
]

#: The paper's installed golden and its panels (repository paths without suffix; each written as .pdf and .png).
GEMMA_GOLDEN = "results/training/gemma.json"
GEMMA_FIGURES = {
    "a": "figures/grpo_inconsistency",
}
QWEN_GOLDEN = "results/training/qwen.json"
QWEN_FIGURES = {
    "a": "figures/grpo_qwen_inconsistency",
}


#: A population tag (``pop-<first 8 hex of its scenario manifest>``), and the suffix a rebased root carries.
_POPULATION = re.compile(r"pop-[0-9a-f]{8}")
_ROOT_SUFFIX = re.compile(r"-pop-[0-9a-f]{8}$")


@dataclass(frozen=True)
class PaperFamily:
    """A training family as the paper shows it: its arms (reference first) and the build flags of the same names."""

    arms: Tuple[FamilyArm, ...]
    labels: Mapping[str, str]
    #: The ruleset, as a repository path.
    rule_set: str
    merge_manifests: Mapping[str, str]
    #: Each member's parity reports (``artifact://`` URIs), where parity was built for the judge.
    parity: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    allow_code_mismatch: bool = False

    @property
    def reference(self) -> str:
        return self.arms[0].label

    def on_population(self, population: str) -> "PaperFamily":
        """Return this family with every root renamed for ``population`` (an earlier population suffix replaced)."""
        if not _POPULATION.fullmatch(population):
            raise ValueError(f"population {population!r} is not pop-<first 8 hex of a scenario manifest>")
        return replace(self, arms=tuple(
            replace(arm, root=f"{_ROOT_SUFFIX.sub('', arm.root)}-{population}") for arm in self.arms))

    def arguments(self) -> list[str]:
        """Return the build arguments that read this family (output flags apart)."""
        return [
            *[word for arm in self.arms for word in ("--arm", arm.label, arm.root, arm.condition_id)],
            "--reference", self.reference, "--rule-set", self.rule_set,
            *[word for label, text in self.labels.items() for word in ("--label", f"{label}={text}")],
            *[word for label, uri in self.merge_manifests.items() for word in ("--merge-manifest", f"{label}={uri}")],
            *[word for label, uris in self.parity.items() for uri in uris for word in ("--parity", f"{label}={uri}")],
            *(["--allow-code-mismatch"] if self.allow_code_mismatch else []),
        ]


#: The paper's training family: Gemma 4 31B-it's base against its trained checkpoint, on the paper's population.
GEMMA_FAMILY = PaperFamily(
    arms=(
        FamilyArm("base", "artifact://sampling/fig5-gemma-2026-10-07-pop-a22af097", "gemma-4-31B-it/figure5/base"),
        FamilyArm("final", "artifact://sampling/fig5-gemma-ckpt600-2026-10-09-pop-a22af097",
                  "gemma-4-31B-it/figure5/final"),
    ),
    labels={"base": "Gemma 4 31B-it", "final": "GRPO (checkpoint-600)"},
    rule_set="data/rules/modelspec/openai_rules_all.jsonl",
    merge_manifests={
        "final": "artifact://selection/gemma-grpo-mix-2026-10-04/modal_root/finetuning/gemma-grpo-mix-2026-10-04/"
                 "merged/checkpoint-600/merge_manifest.json",
    },
)

_QWEN_SELECTION = "artifact://selection/qwen-grpo-mix-2026-10-11"
#: The appendix's training family: Qwen3-8B's base against its trained checkpoint, on the paper's population.
QWEN_FAMILY = PaperFamily(
    arms=(
        FamilyArm("base", "artifact://sampling/fig5-grpo-v2-fp32-2026-09-26-checkpoint-200-pop-a22af097",
                  "Qwen3-8B/figure5/base"),
        FamilyArm("pick", "artifact://sampling/fig5-qwen-mix-2026-10-11-checkpoint-300-pop-a22af097",
                  "Qwen3-8B/figure5/pick"),
    ),
    labels={"base": "Qwen3-8B", "pick": "GRPO (checkpoint-300)"},
    rule_set="data/rules/modelspec/openai_rules_all.jsonl",
    merge_manifests={
        "pick": f"{_QWEN_SELECTION}/modal_root/finetuning/qwen-grpo-mix-2026-10-11/merged/checkpoint-300/"
                "merge_manifest.json",
    },
    parity={
        "pick": tuple(f"{_QWEN_SELECTION}/selection/checkpoints/checkpoint-300/parity/{arm}_report.json"
                      for arm in ("prefill", "decode")),
    },
    allow_code_mismatch=True,
)


@dataclass(frozen=True)
class PaperOutputs:
    """One paper family's golden, panels, the module constant naming its family, and the names its panels draw each
    arm under (display only)."""

    whose: str
    title: str
    switch: str
    golden: str
    figures: Mapping[str, str]
    drawn: Mapping[str, str]

    @property
    def family(self) -> PaperFamily:
        return globals()[self.switch]

    @property
    def head(self) -> str:
        """Return how a refusal of a readout as this family's begins."""
        return f"not {self.title} (ruleforge.experiments.training.paper_families.{self.switch}): "


#: Every paper family by its ``--paper`` name.
PAPER_OUTPUTS = {
    "gemma": PaperOutputs("the paper's", "the paper's Gemma family", "GEMMA_FAMILY", GEMMA_GOLDEN,
                          GEMMA_FIGURES, {"base": "Base Gemma", "final": "Trained Gemma"}),
    "qwen": PaperOutputs("the appendix's", "the appendix's Qwen3-8B family", "QWEN_FAMILY",
                         QWEN_GOLDEN, QWEN_FIGURES,
                         {"base": "Base Qwen", "pick": "Trained Qwen"}),
}


def _at(node: Any, *keys: str) -> Any:
    for key in keys:
        node = node.get(key) if isinstance(node, Mapping) else None
    return node


def _keys(node: Any) -> Any:
    return sorted(node) if isinstance(node, Mapping) else node


def _naming(readout: Mapping[str, Any], family: PaperFamily) -> list[tuple[str, Any, Any]]:
    labels = [arm.label for arm in family.arms]
    members = labels[1:]
    named = _at(readout, "provenance", "family")
    return [
        ("policies", readout.get("policies"), labels),
        ("reference_policy", readout.get("reference_policy"), family.reference),
        ("arms", _keys(readout.get("arms")), sorted(labels)),
        ("comparisons", _keys(readout.get("comparisons")), sorted(members)),
        ("yes_shift", _keys(readout.get("yes_shift")), sorted(members)),
        ("provenance.family.labels", _at(named, "labels"), labels),
        ("provenance.family.reference", _at(named, "reference"), family.reference),
        ("provenance.family.members", _at(named, "members"), members),
        ("provenance.family.holm_family_size", _at(named, "holm_family_size"), len(members)),
        ("provenance.arms", _keys(_at(readout, "provenance", "arms")), sorted(labels)),
        ("provenance.base_weights.members", _keys(_at(readout, "provenance", "base_weights", "members")),
         sorted(members)),
    ]


def _differing(name: str, found: Any, expected: Mapping[str, Any], outputs: PaperOutputs) -> list[str]:
    found = found if isinstance(found, Mapping) else {}

    def problem(key: str) -> str:
        if key not in found:
            return f"no {name} for {key} ({outputs.title} has one)"
        if key not in expected:
            return f"{name} given for {key} ({outputs.title} has none)"
        return f"{key}'s {name} is not {outputs.whose}"

    return [problem(key) for key in sorted(set(found) | set(expected)) if found.get(key) != expected.get(key)]


def _arm_problems(readout: Mapping[str, Any], family: PaperFamily, outputs: PaperOutputs) -> list[str]:
    reference, members = family.arms[0], family.arms[1:]
    problems = []
    for arm in family.arms:
        drawn, recorded = readout["arms"][arm.label], readout["provenance"]["arms"][arm.label]
        if (_at(recorded, "root"), _at(recorded, "condition_id")) != (arm.root, arm.condition_id):
            problems.append(f"arm {arm.label} is {_at(recorded, 'condition_id')} of {_at(recorded, 'root')}, not "
                            f"{arm.condition_id} of {arm.root}")
        if _at(drawn, "label") != family.labels.get(arm.label, arm.label):
            problems.append(f"arm {arm.label} is labelled {_at(drawn, 'label')!r}, not "
                            f"{family.labels.get(arm.label, arm.label)!r}")
        copies = {"condition_id": arm.condition_id, "reference": arm is reference,
                  **{key: _at(recorded, key) for key in ("model_path", "model_weights_sha256")}}
        problems += [f"arms.{arm.label}.{key} is {_at(drawn, key)!r}, not {wanted!r}"
                     for key, wanted in copies.items() if _at(drawn, key) != wanted]
    for arm in members:
        for test in ("flat", "kappa"):
            row = _at(readout, "comparisons", arm.label, test)
            if (tuple(_at(row, key) for key in ("policy_a", "policy_b", "a", "b", "family_size"))
                    != (reference.label, arm.label, reference.condition_id, arm.condition_id, len(members))):
                problems.append(f"{arm.label}'s {test} contrast is not {reference.label} against {arm.label} "
                                f"({reference.condition_id} against {arm.condition_id}) in a Holm family of "
                                f"{len(members)}")
    return [
        *problems,
        *_differing("merge manifest", {label: _at(merge, "merge_manifest") for label, merge in
                                       readout["provenance"]["base_weights"]["members"].items()},
                    family.merge_manifests, outputs),
        *_differing("parity report", _parity_reports(_at(readout, "disclosure", "parity_b")),
                    {label: tuple(sorted(uris)) for label, uris in family.parity.items()}, outputs),
    ]


def _parity_reports(parity: Any) -> dict[str, Any]:
    if not isinstance(parity, Mapping):
        return {}
    return {label: tuple(sorted(str(_at(record, "report")) for record in arms.values()))
            if isinstance(arms, Mapping) else arms for label, arms in parity.items()}


def paper_refusal(readout: Mapping[str, Any], family: Optional[PaperFamily] = None, *,
                  paper: str = "gemma") -> Optional[str]:
    """Return why ``readout`` is not exactly the paper family ``paper`` (arms, labels, ruleset, manifests, parity,
    code-state override, no absolute paths), or None when it is."""
    outputs = PAPER_OUTPUTS[paper]
    family = outputs.family if family is None else family
    head = outputs.head
    if readout.get("source") != SOURCE or readout.get("schema_version") != SCHEMA_VERSION:
        return (f"{head}it is no training family readout (source {readout.get('source')!r}, schema_version "
                f"{readout.get('schema_version')!r}; ruleforge.experiments.training.build writes source {SOURCE!r}, "
                f"schema_version "
                f"{SCHEMA_VERSION})")
    wrong = [f"{name} is {found!r}" for name, found, wanted in _naming(readout, family) if found != wanted]
    if wrong:
        return (f"{head}its arms are not {outputs.whose} {[arm.label for arm in family.arms]} (reference "
                f"{family.reference}), which it shows alone and in that order, wherever it names them: "
                + "; ".join(wrong))
    provenance = readout["provenance"]
    rule_set = file_record(repository_root() / family.rule_set)
    checks = (
        (provenance.get("complete") is True, "it is not complete"),
        (provenance.get("rule_set") == rule_set,
         f"the same-call rate was reread against {provenance.get('rule_set')}, not {family.rule_set} "
         f"({rule_set['sha256'][:12]}...)"),
        (provenance.get("allow_code_mismatch") is family.allow_code_mismatch,
         f"it was read {'without' if family.allow_code_mismatch else 'with'} allow_code_mismatch"),
        (not (paths := [key for key, _ in absolute_strings(readout)]),
         f"it holds absolute paths (at {paths[:3]}), which no committed golden may"),
    )
    problems = _arm_problems(readout, family, outputs) + [message for ok, message in checks if not ok]
    return head + "; ".join(problems) if problems else None


def drawn_names(readout: Mapping[str, Any]) -> Mapping[str, str]:
    """Return the panels' names for ``readout``'s arms: a paper family's ``drawn`` when the readout is its, else
    nothing."""
    for name, outputs in PAPER_OUTPUTS.items():
        if paper_refusal(readout, paper=name) is None:
            return outputs.drawn
    return {}
