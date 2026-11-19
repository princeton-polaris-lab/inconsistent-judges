from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from ruleforge.core.artifacts import prompt_sha256
from ruleforge.core.digests import canonical_json
from ruleforge.prompts.parse import LABELS

if TYPE_CHECKING:
    from ruleforge.judges.models import DetailCallFn

FORCED_JSON_KEY = "full_set_judgment"
_RULE_LINE = re.compile(r"^(Rule \d+)(?: \([^)\n]*\))?:", re.MULTILINE)
_RULE_SET_HEADING = re.compile(r"^#+ [^\n]*Rule Set[^\n]*$", re.MULTILINE)
_HEADING = re.compile(r"^#+ ", re.MULTILINE)
FINISH_REASON = "stop"


def fake_label(fake_seed: int, prompt_digest: str, request_seed: Optional[int], *key: str) -> str:
    """Return the label of ``(fake_seed, prompt_digest, request_seed, *key)``: the first SHA-256 byte of their JSON, mod
    3."""
    digest = hashlib.sha256(canonical_json([fake_seed, prompt_digest, request_seed, *key]).encode("utf-8")).digest()
    return LABELS[digest[0] % len(LABELS)]


def prompt_rule_labels(text: str) -> list[str]:
    """Return each distinct ``Rule N`` that heads a line of ``text``, in order of first appearance."""
    seen: dict[str, None] = {}
    for match in _RULE_LINE.finditer(text):
        seen.setdefault(match.group(1), None)
    return list(seen)


def prompt_rule_set(text: str) -> str:
    """Return the rule-set section of a prompt (from its ``Rule Set`` heading to the next), or the whole text."""
    heading = _RULE_SET_HEADING.search(text)
    if heading is None:
        return text
    following = _HEADING.search(text, heading.end())
    return text[heading.end():None if following is None else following.start()]


def fake_answer(fake_seed: int, entry: Mapping[str, str], request_seed: Optional[int]) -> str:
    """Return the fake's answer to one prompt entry at ``request_seed``."""
    digest = prompt_sha256(entry)
    label = fake_label(fake_seed, digest, request_seed)
    text = entry.get("system_prompt", "") + "\n" + entry.get("user_prompt", "")
    if FORCED_JSON_KEY not in text:
        return label
    rules = {rule: fake_label(fake_seed, digest, request_seed, rule)
             for rule in prompt_rule_labels(prompt_rule_set(text))}
    return json.dumps({"rule_judgments": rules, FORCED_JSON_KEY: label})


def make_fake_caller(fake_seed: int) -> DetailCallFn:
    """Return a sync caller answering with ``fake_answer``; temperature and the token ceiling are ignored."""

    def _call(system_prompt: str, user_prompt: str, temperature: float, max_new_tokens: int,
              seed: Optional[int]) -> Dict[str, Any]:
        text = fake_answer(fake_seed, {"system_prompt": system_prompt, "user_prompt": user_prompt}, seed)
        return {"raw_response": text, "completion": text, "finish_reason": FINISH_REASON, "error": None}

    return _call

