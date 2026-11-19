from __future__ import annotations

import json
import re
from collections import Counter
from functools import partial
from typing import Any, Callable, Mapping, Optional, Sequence

from omegaconf import OmegaConf

from ruleforge.judges.base import Capabilities, Judge, JudgeOptions, JudgeSpec
from ruleforge.judges.backend import SEED_LOCAL_DETERMINISTIC, GenerationOutcome, ModelBackend
from ruleforge.prompts.interpreters import INTERPRETATION_RULES
from ruleforge.judges.models import _api_generate_batch, _api_generate_batch_with_details
from ruleforge.judges.token_usage import TokenUsageTracker

Answer = Callable[[str, str], str]


class ScriptedFailure(RuntimeError):
    """What a script raises for a request that fails in transport."""


def _failed(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class ScriptedJudge(Judge):
    """A judge answering every prompt with ``answer(system_prompt, user_prompt)``."""

    def __init__(self, answer: Answer, *, model_name: str = "scripted", seed: Optional[int] = None,
                 capabilities: Optional[Capabilities] = None, strip_thinking_model: Optional[str] = None) -> None:
        self.answer = answer
        options = {"model_type": "scripted", "model_name": model_name, **({} if seed is None else {"seed": seed}),
                   **({} if strip_thinking_model is None else {"strip_thinking_model": strip_thinking_model})}
        self.spec = JudgeSpec(model_type="scripted", model_name=model_name, options=JudgeOptions(options))
        self.capabilities = capabilities or Capabilities(seed=SEED_LOCAL_DETERMINISTIC, per_request_errors=True)
        self.requests: list[dict[str, Any]] = []
        self._usage = TokenUsageTracker()

    @property
    def usage(self) -> TokenUsageTracker:
        return self._usage

    def generate_once(self, prompts: list[dict[str, str]], *, temperature: Optional[float] = None,
                      max_new_tokens: Optional[int] = None, seeds: Optional[Sequence[Optional[int]]] = None,
                      request_block_id: str = "", truncation: Optional[Mapping[str, Any]] = None,
                      ) -> list[GenerationOutcome]:
        outcomes = []
        for index, entry in enumerate(prompts):
            system_prompt, user_prompt = entry.get("system_prompt", ""), entry.get("user_prompt", "")
            self.requests.append({"system_prompt": system_prompt, "user_prompt": user_prompt,
                                  "temperature": temperature, "max_new_tokens": max_new_tokens,
                                  "seed": None if seeds is None else seeds[index]})
            try:
                text = self.answer(system_prompt, user_prompt)
            except Exception as exc:
                outcomes.append(GenerationOutcome(text=None, raw_response=None, finish_reason=None,
                                                  error=_failed(exc)))
                continue
            outcomes.append(GenerationOutcome(text=text or None, raw_response=text, finish_reason="stop"))
        return outcomes


def _sync_caller(answer: Answer, requests: Optional[list[dict[str, Any]]]) -> Callable[..., dict[str, Any]]:
    """``answer`` as a provider's synchronous caller (``ruleforge.judges.models``): a raised answer is its error."""

    def call(system_prompt: str, user_prompt: str, temperature: float, max_new_tokens: int,
             seed: Optional[int]) -> dict[str, Any]:
        if requests is not None:
            requests.append({"system_prompt": system_prompt, "user_prompt": user_prompt, "temperature": temperature,
                             "max_new_tokens": max_new_tokens, "seed": seed})
        try:
            text = answer(system_prompt, user_prompt)
        except Exception as exc:
            return {"raw_response": None, "completion": None, "finish_reason": None, "error": _failed(exc)}
        return {"raw_response": text, "completion": text, "finish_reason": "stop", "error": None}

    return call


def scripted_backend(model_cfg: Any, answer: Answer, *,
                     requests: Optional[list[dict[str, Any]]] = None) -> ModelBackend:
    """The ``ModelBackend`` for the model config ``model_cfg``, answering through ``answer``."""
    seed = int(getattr(model_cfg, "seed", 16))
    config: dict[str, Any] = {"model_type": "fake", "model_name": f"fake-{seed}", "seed": seed}
    for key in ("num_tries", "max_retry_seconds", "temperature", "max_new_tokens"):
        value = getattr(model_cfg, key, None)
        if value is not None:
            config[key] = value
    backend = ModelBackend(OmegaConf.create(config))
    backend.model_name = str(model_cfg.model_name)
    caller = _sync_caller(answer, requests)
    backend._batch_generate = partial(_api_generate_batch, caller)
    backend._batch_generate_with_details = partial(_api_generate_batch_with_details, caller)
    return backend


#: What a patch appends to a rule it clarifies; a judge counts these in the rule text it is shown.
CLARIFIED = "(clarified)"
#: The rules the ``j``-th rule-amendment candidate asked for clarifies (candidate 0 fails), then Rule ``j % 3 + 1``.
CANDIDATE_TARGETS = ((), ("Rule 2",), ("Rule 3", "Rule 1"), ("Rule 2",), ("Rule 1",))
JUDGE_TYPES = ("No_Interpretation", *INTERPRETATION_RULES)
_JUDGE_TYPE_OF = {text: name for name, text in INTERPRETATION_RULES.items()}
_REQUEST = re.compile(r"Request (\d+) \[(contested|clear)\]")
_RULE_LINE = re.compile(r"^((?:Meta-)?Rule \d+): (.*)$", re.MULTILINE)
_PER_BATCH = re.compile(r"You must generate (\d+) scenarios\.")


def _between(text: str, start: str, end: str) -> str:
    head = text.index(start) + len(start)
    tail = text.find(end, head)
    return text[head:] if tail < 0 else text[head:tail]


class RuleForgeScript:
    """The answers of a scripted RuleForge run (Model Spec task, ``naive`` prompts), counted per kind of request."""

    def __init__(self, *, failure: str = "empty") -> None:
        if failure not in ("empty", "raise"):
            raise ValueError(f"failure is 'empty' or 'raise', got {failure!r}")
        self.failure = failure
        self.counts: Counter[str] = Counter()
        self.unexpected: list[str] = []
        self._rule3_preemptions = 0

    def fail(self) -> str:
        if self.failure == "raise":
            raise ScriptedFailure("scripted transport failure")
        return ""

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        kind = self.kind(system_prompt)
        index = self.counts[kind]
        self.counts[kind] += 1
        if kind == "unexpected":
            self.unexpected.append(system_prompt[:200])
            return ""
        return getattr(self, f"_{kind}")(index, system_prompt, user_prompt)

    @staticmethod
    def kind(system_prompt: str) -> str:
        if system_prompt.startswith("# Task\nYou are an expert at revising rules"):
            return "ra_sample"
        if system_prompt.startswith("# Task\nYou are an expert at creating scenarios"):
            return "sg_sample"
        if system_prompt.startswith("# Task\nYou are a Bayesian posterior"):
            return "ra_update" if "You are an agent revising a set of rules" in system_prompt else "sg_update"
        if "# Revised Ruleset:" in system_prompt:
            return "intent"
        if '"preempting_rules"' in system_prompt:
            return "preemption"
        if "# Rule Set Used for Evaluation" in system_prompt or "# Rule Set for Evaluation" in system_prompt:
            return "judge"
        return "unexpected"

    def _sg_sample(self, index: int, system_prompt: str, user_prompt: str) -> str:
        per_batch = int(_PER_BATCH.search(system_prompt).group(1))
        blocks = []
        for position in range(per_batch):
            number = index * per_batch + position
            label = "clear" if number % 4 == 2 else "contested"
            blocks.append(f"# Scenario {position + 1}\nSubset of Rule IDs: [R1, R2]\n"
                          f"Possible Ambiguous Topics and Value Tradeoffs: [tradeoff {number}]\n"
                          f"Prompt: Request {number} [{label}]\nResponse: Answer {number}")
        return "\n".join(blocks)

    def _ra_sample(self, index: int, system_prompt: str, user_prompt: str) -> str:
        if index == 0:
            return self.fail()
        previous = _between(user_prompt, "# Previous Rule Set\n", "\n# Agent's Current Knowledge")
        rules = dict(_RULE_LINE.findall(previous))
        targets = CANDIDATE_TARGETS[index] if index < len(CANDIDATE_TARGETS) else (f"Rule {index % 3 + 1}",)
        return "\n".join(f"<<<<<<< SEARCH\n{rule}: {rules[rule]}\n=======\n{rule}: {rules[rule]} {CLARIFIED}\n"
                         ">>>>>>> REPLACE" for rule in targets)

    def _ra_update(self, index: int, system_prompt: str, user_prompt: str) -> str:
        return f"Rule amendment posterior {index}"

    def _sg_update(self, index: int, system_prompt: str, user_prompt: str) -> str:
        return self.fail() if index == 0 else f"Scenario generation posterior {index}"

    def _judge(self, index: int, system_prompt: str, user_prompt: str) -> str:
        header = "# Rule Set Used for Evaluation\n" if "# Rule Set Used for Evaluation" in system_prompt \
            else "# Rule Set for Evaluation\n"
        rule_text = _between(system_prompt, header, "\n# ")
        if "# Interpretation Strategy\n" in system_prompt:
            judge_type = _JUDGE_TYPE_OF[_between(system_prompt, "# Interpretation Strategy\n", "\n\n# ")]
        else:
            judge_type = "No_Interpretation"
        number, label = _REQUEST.search(_between(system_prompt, "# Original Prompt Given to the AI\n",
                                                 "\n# AI Response")).groups()
        number = int(number)
        if judge_type == "Narrow" and number % 5 == 3:
            return self.fail()
        if judge_type == "Ordinary_Meaning" and number % 4 == 1:
            return "maybe"
        if label == "clear":
            return "no"
        rules = _RULE_LINE.findall(rule_text)
        if len(rules) == 1:
            rule, text = rules[0]
            return "yes" if rule == "Rule 1" and CLARIFIED not in text else "no"
        clarity = rule_text.count(CLARIFIED)
        if clarity == 0:
            return "no" if JUDGE_TYPES.index(judge_type) % 2 else "yes"
        if clarity == 1 and judge_type == "Broad":
            return "unsure"
        return "yes"

    def _intent(self, index: int, system_prompt: str, user_prompt: str) -> str:
        if index == 0:
            return "scores to follow"
        revised = system_prompt[system_prompt.index("# Revised Ruleset:\n"):]
        return json.dumps({"rules": [
            {"id": rule, "reasoning": f"{text.count(CLARIFIED)} clarification(s)",
             "score": min(4, text.count(CLARIFIED))}
            for rule, text in _RULE_LINE.findall(revised)]})

    def _preemption(self, index: int, system_prompt: str, user_prompt: str) -> str:
        target = _between(system_prompt, "# Target Rule\n", ":")
        contested = "[contested]" in system_prompt
        if target == "Rule 3":
            self._rule3_preemptions += 1
            if self._rule3_preemptions == 1:
                return "no preemption"
        preempting = [{"id": "Rule 2", "rule": "as written"}] if target == "Rule 1" and contested else []
        return json.dumps({"preempting_rules": preempting})
