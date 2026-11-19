from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ruleforge.judges.batch_api import is_truncated
from ruleforge.judges.models import holds_a_thought
from ruleforge.optimize.input_output import save_data_to_jsonl

logger = logging.getLogger(__name__)


def unfinished_reason(completion: str, finish_reason: Optional[str], *,
                      strip_thinking_model: Optional[str] = None) -> Optional[str]:
    """Return why an answer cannot be used as text (it stopped at the output ceiling, or holds a thought in its model
    family's framing), or None."""
    if is_truncated(finish_reason):
        return "finished on length"
    if holds_a_thought(completion, strip_thinking_model):
        return "holds an unclosed thought"
    return None


def _sized_call(backend: Any, call: str, prompt: Dict[str, str], trace_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Return the start of a call's trace row, or None when the model cannot take the prompt: the call then fails, and
    the history is never shortened."""
    prompt_tokens, limit = backend.prompt_size(prompt)
    row = {"call": call, "prompt": prompt, "prompt_tokens": prompt_tokens, "prompt_token_limit": limit}
    if prompt_tokens is not None:
        logger.info(f"{call} prompt: {prompt_tokens} tokens of the model's {limit}")
    if prompt_tokens is None or limit is None or prompt_tokens < limit:
        return row
    error = f"the {call} prompt has {prompt_tokens} tokens; the model takes fewer than {limit}"
    logger.warning(f"Failing this call: {error}")
    if trace_path is not None:
        save_data_to_jsonl({**row, "result": None, "selected_attempt": None, "attempts": [], "unfinished": {},
                            "error": error}, trace_path)
    return None


def _unfinished_answers(backend: Any, trace: Dict[str, Any]) -> Dict[int, str]:
    return {
        candidate["result_index"]: reason
        for attempt in trace["attempts"]
        for candidate in attempt["candidates"]
        if candidate["accepted"]
        and (reason := unfinished_reason(candidate["completion"], candidate["finish_reason"],
                                         strip_thinking_model=backend.strip_thinking_model)) is not None
    }


def generate_traced(
    backend: Any,
    call: str,
    prompt_batch: List[Dict[str, str]],
    trace_path: Optional[Path],
    **generation_kwargs: Any,
) -> Tuple[Optional[List[str]], Dict[int, str]]:
    """Return the answers to the one prompt in `prompt_batch` (None when none came) and, by answer index, why an answer
    is unfinished; the call's trace, with the prompt's token count where the judge has one, goes to `trace_path` when
    given."""
    row = _sized_call(backend, call, prompt_batch[0], trace_path)
    if row is None:
        return None, {}
    outputs, traces = backend.generate_with_traces(prompt_batch, **generation_kwargs)
    unfinished = _unfinished_answers(backend, traces[0])
    if trace_path is not None:
        save_data_to_jsonl({**row, **traces[0], "unfinished": unfinished}, trace_path)
    return outputs[0], unfinished


def generate_finished(
    backend: Any,
    call: str,
    prompt_batch: List[Dict[str, str]],
    trace_path: Optional[Path],
    *,
    seed: Optional[int] = None,
    **generation_kwargs: Any,
) -> Tuple[Optional[List[str]], Dict[int, str]]:
    """Return `generate_traced`'s result for a one-answer call, asked again while unfinished up to the judge's
    `num_tries` attempts, each retry with its own seed (`seed` replaces the judge's for a prompt sampled more than
    once); the trace is one row with every attempt, and `retries` holds why each retry was made."""
    row = _sized_call(backend, call, prompt_batch[0], trace_path)
    if row is None:
        return None, {}
    answers: Optional[List[str]] = None
    unfinished: Dict[int, str] = {}
    selected_attempt: Optional[int] = None
    attempts: List[Dict[str, Any]] = []
    retries: List[Dict[str, Any]] = []
    first_seed = backend.seed if seed is None else seed
    while len(attempts) < backend.num_tries:
        call_seed = {"seed": first_seed + len(retries)} if seed is not None or retries else {}
        outputs, traces = backend.generate_with_traces(prompt_batch, num_tries=backend.num_tries - len(attempts),
                                                       **call_seed, **generation_kwargs)
        offset = len(attempts)
        for attempt in traces[0]["attempts"]:
            attempts.append({**attempt, "attempt": attempt["attempt"] + offset})
        if outputs[0] is None:  # no answer in the attempts left: the last unfinished one, if any, stands
            break
        answers, unfinished = outputs[0], _unfinished_answers(backend, traces[0])
        selected_attempt = traces[0]["selected_attempt"] + offset
        if 0 not in unfinished:
            break
        if len(attempts) < backend.num_tries:
            retries.append({"attempt": len(attempts), "reason": unfinished[0]})
            logger.warning(f"Retrying the {call} after attempt {len(attempts)} of {backend.num_tries}: its answer "
                           f"{unfinished[0]}")
    if trace_path is not None:
        save_data_to_jsonl({**row, "result": answers, "selected_attempt": selected_attempt, "attempts": attempts,
                            "unfinished": unfinished, **({"retries": retries} if retries else {})}, trace_path)
    return answers, unfinished


def revised_prior(agent: str, prior: str, answers: Optional[List[str]], unfinished: Dict[int, str]) -> str:
    """Return the prior a posterior update leaves: its answer, or the old prior when it failed."""
    if not answers:
        logger.warning(f"Empty posterior update response in {agent} agent; keeping current prior")
        return prior
    if 0 in unfinished:
        logger.warning(f"The posterior update of the {agent} agent {unfinished[0]}; keeping current prior")
        return prior
    return answers[0]
