from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import random
import shutil
import statistics
from array import array
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Optional, Protocol, Sequence

from ruleforge.env import model_root
from ruleforge.grpo.anchor_set import (
    ANCHOR_CLASSES,
    ANCHORS_FNAME,
    ANCHORS_MANIFEST_FNAME,
    ANCHORS_SCHEMA_VERSIONS,
    CLASS_A,
    CLASS_B,
    CLASS_C,
    MIXED_SELECTION_KIND,
    REPLAY_INDEX_FNAME,
    SIDE_TRAIN,
    SOURCE_BANK,
    SOURCE_TOP_UP,
    TIE_ORDER,
    AnchorError,
    CampaignCondition,
    CampaignInput,
    Ruleset,
    condition_key,
    input_records,
    mode_of,
    one_of,
    open_condition,
    stack_check,
)
from ruleforge.grpo.anchor_set import anchor_class as class_of
from ruleforge.grpo.anchor_set import estimate as estimate_anchor
from ruleforge.judges.completion_format import THINKING_FAMILIES, CompletionFormat, load_training_format
from ruleforge.judges.completion_format import GENERATION_CONFIG_FNAME, GrpoDataError, stop_token_ids
from ruleforge.grpo.leakage import LeakageError, disjoint_marker_path
from ruleforge.grpo.rewards import Q_COLUMNS
from ruleforge.rulesets.variants import build_rule_variants, rule_set_text
from ruleforge.evaluate.campaign import CONDITION_FNAME
from ruleforge.evaluate.identity import (
    EVALUATION_MODE_BANK,
    EVALUATION_MODE_FULL,
    FULL_RULESET_COMPONENT,
    draw_dirname,
)
from ruleforge.core.digests import sha256_file
from ruleforge.core.files import write_json_sidecar
from ruleforge.evaluate.ledger import LEDGER_FNAME, RAW_OUTPUTS_FNAME, DrawLedger, raw_output_sha256
from ruleforge.core.provenance import code_state
from ruleforge.core.record_names import ROWS_DERIVATION
from ruleforge.core.artifacts import prompt_sha256
from ruleforge.evaluate.runner import utc_now
from ruleforge.core.artifacts import (
    compute_scenario_id,
    iter_jsonl_records,
    load_jsonl_records,
    write_jsonl_records,
)
from ruleforge.judges.models import _ThinkingTokenizer, format_prompt_for_generation, prompt_token_ids
from ruleforge.judges.tokenizer import TOKENIZER_JSON_FNAME, check_tokenizer_fidelity

#: 1: every anchor input is one campaign root; 2: one spans several roots.
GRPO_DATA_SCHEMA_VERSION = 1
GRPO_DATA_SCHEMA_VERSION_ROOTS = 2
#: 3: from a staged anchor set (`campaigns` keyed by stage, `q_hat_stages` recorded).
GRPO_DATA_SCHEMA_VERSION_STAGED = 3
#: 4: from several anchor sets (`anchors` a list, `by_anchor_set` recorded).
GRPO_DATA_SCHEMA_VERSION_SETS = 4
GRPO_ROWS_FNAME = "grpo_rows.jsonl"
REPLAY_ROWS_FNAME = "replay_rows.jsonl"
SCENARIOS_FNAME = "scenarios.jsonl"
MANIFEST_FNAME = "grpo_data_manifest.json"
#: Prompt token cap, chat template included, in the base's tokens.
PROMPT_TOKEN_CAP = 5000
#: The run's prompt draws D (steps x prompts per step).
PROMPT_DRAWS = 2400
REPEAT_CAP = 8
CLASS_A_SHARE_CAP = 0.15
DRAW_SEED = 0
#: The judgment mix's strata: the no-favouring rows that take the declared share, then the rest by favoured judgment.
STRATUM_NO = "no"
STRATUM_NO_UNDER_MARGIN = "no_under_margin"
REST_STRATA = (STRATUM_NO_UNDER_MARGIN, "unsure", "yes")
VERDICT_STRATA = (STRATUM_NO, *REST_STRATA)
#: Prompts rendered and compared per batch.
RENDER_BATCH = 256
TASK_SPEC = "modelspec"
#: Same value as `ruleforge.grpo.kl_anchor.IGNORE_INDEX`; not imported, since that module needs torch.
IGNORE_INDEX = -100
EXCLUSION_MODULE = "ruleforge.grpo.exclusion"
NO_PINS: Mapping[str, str] = MappingProxyType({})

TOKENIZER_FILES = (TOKENIZER_JSON_FNAME, "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")


@dataclass(frozen=True)
class CampaignPrompts:
    """The judge prompts of one condition and ruleset, built from the same parts the campaign runner builds them from,
    without a judge backend or a draw."""

    task_spec: Any
    judge_type: str
    rule_set_string: str
    variants: Mapping[str, tuple[str, Optional[list[str]]]]

    @classmethod
    def build(cls, *, task_spec: Any, rules: Sequence[Mapping[str, Any]], judge_type: str,
              levels: Optional[Mapping[str, str]] = None) -> "CampaignPrompts":
        rules = [dict(rule) for rule in rules]
        variants, _ = build_rule_variants(rules, task_spec.individual_rule_context, levels)
        return cls(
            task_spec=task_spec,
            judge_type=judge_type,
            rule_set_string=rule_set_text(rules, task_spec.individual_rule_context, levels),
            variants={rule_id: (text, context) for rule_id, text, context in variants},
        )

    def entry(self, scenario: Mapping[str, Any], component_id: str) -> dict[str, str]:
        """Return one component's prompt entry: the prompt as the system turn, an empty user turn."""
        if component_id == FULL_RULESET_COMPONENT:
            rule_text, context = self.rule_set_string, None
        else:
            rule_text, context = self.variants[component_id]
        system_prompt = self.task_spec.build_eval_prompt(
            rule_text=rule_text, scenario=dict(scenario), judge_type=self.judge_type, context_rules=context
        )
        return {"system_prompt": system_prompt, "user_prompt": ""}


def condition_task_spec(condition: Any) -> Any:
    """Return the task spec a condition's prompts come from; a condition of another task is refused."""
    from omegaconf import OmegaConf

    from ruleforge.tasks.spec import build_task_spec

    task = getattr(condition, "task", TASK_SPEC)
    if task != TASK_SPEC:
        raise GrpoDataError(f"condition {condition.condition_id}: its prompts are task {task!r}'s, and GRPO rows are "
                            f"rendered from the {TASK_SPEC!r} task's prompts alone")
    return build_task_spec(OmegaConf.create({
        "task_spec": TASK_SPEC,
        "rule_amendment": {"initial_rule_set_path": "unused.jsonl"},
        "scenario_generation": {"good_seed_scenario_path": "unused.jsonl", "bad_seed_scenario_path": "unused.jsonl"},
        "agent_model": {},
        "judge_model": {},
        "eval_cfg": {
            "judge_types": [condition.judge_type],
            "reasoning_prompt_style": condition.prompt_style,
            "individual_rule_context": condition.individual_rule_context,
        },
    }))


def conversation(entry: Mapping[str, str]) -> list[dict[str, str]]:
    """Return a prompt entry as the conversation TRL renders: `[system: P, user: ""]`."""
    return [{"role": "system", "content": entry["system_prompt"]}, {"role": "user", "content": entry["user_prompt"]}]


def chat_template_kwargs(enable_thinking: bool) -> dict[str, Any]:
    """Return the chat-template arguments that render as the campaign does: `enable_thinking`, true or false."""
    return {"enable_thinking": bool(enable_thinking)}


def campaign_prompt_ids(tokenizer: Any, entries: Sequence[Mapping[str, str]], *, enable_thinking: bool) -> list[list[int]]:
    """Return each entry's token ids as a campaign sends them to vLLM."""
    if not entries:
        return []
    renderer = _ThinkingTokenizer(tokenizer, enable_thinking)
    return [prompt_token_ids(tokenizer, format_prompt_for_generation(renderer, dict(entry))) for entry in entries]


def trl_prompt_ids(tokenizer: Any, conversations: Sequence[Sequence[Mapping[str, str]]], kwargs: Mapping[str, Any]) -> list[list[int]]:
    """Return each conversation's token ids as TRL's GRPO trainer tokenizes a conversational prompt."""
    if not conversations:
        return []
    encoded = tokenizer.apply_chat_template(
        [list(turns) for turns in conversations], add_generation_prompt=True, tokenize=True, return_dict=True,
        **dict(kwargs),
    )
    return [list(ids) for ids in encoded["input_ids"]]


def tokenizer_files(directory: Path) -> dict[str, str]:
    """Return the digests of the tokenizer files a checkpoint directory has."""
    return {name: sha256_file(directory / name) for name in TOKENIZER_FILES if (directory / name).is_file()}


def campaign_model_dir(name: str) -> Optional[Path]:
    """Return `$RULEFORGE_MODEL_ROOT/<name>` where it exists here, else None."""
    root = model_root()
    candidate = root / name if root else None
    return candidate if candidate is not None and candidate.is_dir() else None


def load_tokenizer(model: Path | str, *, model_path: Optional[str]) -> Any:
    """Return the checkpoint's tokenizer as the trainer loads it, checked against the campaign's model directory by name
    and tokenizer bytes; its eos must be one of the checkpoint's stop ids."""
    from transformers import AutoTokenizer

    directory = Path(model)
    if directory.name != Path(str(model_path)).name:
        raise GrpoDataError(
            f"the tokenizer comes from {directory.name!r}, but the anchors were drawn on {model_path!r}; the prompts "
            "must be tokenized as the base the rows replay"
        )
    canonical = campaign_model_dir(directory.name)
    if canonical is not None and canonical.resolve() != directory.resolve():
        theirs, ours = tokenizer_files(canonical), tokenizer_files(directory)
        differing = sorted(name for name in set(theirs) | set(ours) if theirs.get(name) != ours.get(name))
        if differing:
            raise GrpoDataError(
                f"{directory} and {canonical}, the {directory.name!r} a campaign's model config resolves, differ in "
                f"{differing}; the prompts must be tokenized as the campaign's engine tokenized them"
            )
    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    check_tokenizer_fidelity(tokenizer, directory, where="HF")
    if tokenizer.eos_token_id is None:
        raise GrpoDataError(f"{directory}'s tokenizer has no eos token; a replayed completion ends on it")
    stops = stop_token_ids(directory)
    if int(tokenizer.eos_token_id) not in stops:
        raise GrpoDataError(
            f"{directory}'s tokenizer eos {tokenizer.eos_token_id} is not among the ids its {GENERATION_CONFIG_FNAME} "
            f"stops on ({stops}); a replayed completion would end on a token the base never stops a turn with"
        )
    return tokenizer


def tokenizer_record(model: Path, tokenizer: Any, completion: Optional[CompletionFormat] = None) -> dict[str, Any]:
    """Return what the manifest records about the tokenizer: file digests, template, stop ids and the completion format.
    """
    template = tokenizer.chat_template if isinstance(tokenizer.chat_template, str) else None
    canonical = campaign_model_dir(model.name)
    return {
        "model": str(model),
        "name": model.name,
        "files": tokenizer_files(model),
        "chat_template_sha256": None if template is None else hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "eos_token_id": int(tokenizer.eos_token_id),
        "stop_token_ids": stop_token_ids(model),
        "campaign_model_dir": None if canonical is None else str(canonical),
        **({} if completion is None else {"completion_format": completion.record()}),
    }


#: Weight files a checkpoint loader reads that the weights digest does not.
UNDIGESTED_WEIGHTS = ("pytorch_model*.bin", "*.pt", "*.pth", "*.ckpt", "tf_model.h5", "flax_model.msgpack")


def base_weights_sha256(model: Path | str) -> Optional[str]:
    """Return the weights digest of a base directory (`judges.weights.weights_digest`), None for a tokenizer alone;
    weights in a format other than safetensors are refused, whatever lies beside them, as a loader may read them."""
    from ruleforge.judges.weights import weights_digest

    directory = Path(model)
    if other := sorted(path.name for pattern in UNDIGESTED_WEIGHTS for path in directory.glob(pattern)):
        raise GrpoDataError(f"{directory} holds weights in a format the weights digest does not read ({other}): "
                            "convert them to safetensors, which binds the rows and the run to them by content")
    return weights_digest(directory) if any(directory.glob("*.safetensors")) else None


def check_rows_model(rows: Path | str, model: Path | str, *, sha256: Optional[str] = None,
                     weights_sha256: Optional[str] = None) -> dict[str, Any]:
    """Refuse, reading only, GRPO or replay rows built on another model than `model` (by directory name, tokenizer
    bytes, completion format and, where both are known, the base weights `weights_sha256`); return what was compared,
    the build's anchor condition and the chat-template arguments it rendered the prompts under."""
    rows, model = Path(rows), Path(model)
    manifest_path = rows.parent / MANIFEST_FNAME
    if not manifest_path.is_file():
        raise GrpoDataError(f"{rows} has no {MANIFEST_FNAME} beside it: nothing records the model it was built on")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256 or sha256_file(rows)
    output = [name for name in (GRPO_ROWS_FNAME, REPLAY_ROWS_FNAME) if manifest["outputs"][name]["sha256"] == digest]
    if not output:
        raise GrpoDataError(
            f"{rows} is neither the {GRPO_ROWS_FNAME} nor the {REPLAY_ROWS_FNAME} its {MANIFEST_FNAME} records"
        )
    tokenizer = manifest["tokenizer"]
    built_weights = manifest.get("base_weights_sha256")
    if built_weights is not None and weights_sha256 != built_weights:
        held = "no weights digested here" if weights_sha256 is None else f"{weights_sha256[:12]}..."
        raise GrpoDataError(
            f"{rows} was built on base weights {built_weights[:12]}..., and {model} holds {held}: train on rows built "
            "on the weights being trained"
        )
    if tokenizer["name"] != model.name:
        raise GrpoDataError(
            f"{rows} was built on {tokenizer['model']}, not on {model.name} ({model}): train on rows built on the "
            "model being trained"
        )
    recorded, found = tokenizer["files"], tokenizer_files(model)
    differing = sorted(name for name in set(recorded) | set(found) if recorded.get(name) != found.get(name))
    if differing:
        raise GrpoDataError(
            f"{rows} was built on {tokenizer['model']}'s tokenizer, and {model} holds other {differing}: "
            "train on rows built on the model being trained"
        )
    _, completion = load_training_format(model)
    expected = completion.record()
    built = tokenizer.get("completion_format")
    if built is None and THINKING_FAMILIES[completion.family].end_of_turn is not None:
        raise GrpoDataError(
            f"{rows} records no completion format, and {model.name}'s end of turn is not its tokenizer's eos: its "
            "replay rows end on the tokenizer's eos, not the model's end of turn; rebuild"
        )
    if built is not None and built != expected:
        raise GrpoDataError(
            f"{rows} was built with the completion format {built}, and {model.name}'s training tokenizer gives "
            f"{expected}: train on rows built on the model being trained"
        )
    return {"rows": str(rows), "output": output[0], "manifest_sha256": sha256_file(manifest_path),
            "built_on": tokenizer["model"], "tokenizer_files": found, "completion_format": expected,
            "anchor_condition": manifest.get("anchor_condition"), "base_weights_sha256": built_weights,
            "chat_template_kwargs": (manifest.get("rendering") or {}).get("chat_template_kwargs")}


def recorded_protection(rows: Path | str) -> dict[str, list[str]]:
    """Return what the rows build beside `rows` held its marker to, by digest, as `check_prompt_marker` takes it (the
    pinned and required files under the full screen, the files it added read by the second pass at least), empty
    without a manifest, which the trainer refuses on reading it."""
    path = Path(rows).parent / MANIFEST_FNAME
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))

    def digests(*keys: str) -> list[str]:
        return sorted({str(entry["sha256"]) for key in keys for entry in manifest.get(key) or ()})

    found = {"required_protected": digests("pinned_protected", "required_protected"),
             "required_carved": digests("protected_extra")}
    return {key: values for key, values in found.items() if values}


def special_texts(tokenizer: Any, stop_ids: Sequence[int]) -> tuple[list[str], list[str]]:
    """Return the text of the stop tokens, and of every other special added token."""
    specials = {int(index): token.content for index, token in tokenizer.added_tokens_decoder.items() if token.special}
    return (sorted(text for index, text in specials.items() if index in stop_ids),
            sorted(text for index, text in specials.items() if index not in stop_ids))


def completion_encoder(tokenizer: Any, stop_ids: Sequence[int]) -> Any:
    """Return a copy of the tokenizer's backend that reads a stop token's text as plain text (vLLM leaves the real stop
    token out of the text) and every other special token as the token."""
    from tokenizers import Tokenizer

    spec = json.loads(tokenizer.backend_tokenizer.to_str())
    for token in spec["added_tokens"]:
        token["special"] = int(token["id"]) in stop_ids
    encoder = Tokenizer.from_str(json.dumps(spec))
    encoder.encode_special_tokens = True
    encoder.no_padding()
    encoder.no_truncation()
    return encoder


def completion_ids(encoder: Any, texts: Sequence[str]) -> list[list[int]]:
    """Return each raw completion's ids through `completion_encoder`'s copy."""
    return [list(encoding.ids) for encoding in encoder.encode_batch(list(texts), add_special_tokens=False)]


@dataclass(frozen=True)
class SamplingWeights:
    """The D prompt draws: each scenario's copies (0 when not drawn), the rows' order and the declaration."""

    copies: dict[str, int]
    order: list[str]
    declaration: dict[str, Any]


def proportional_draws(
    total: int, counts: Mapping[str, int], tie_break: Sequence[str]
) -> tuple[dict[str, int], dict[str, Fraction]]:
    """Split `total` draws over the strata in proportion to their counts by largest remainder; return the whole draws and
    each stratum's exact target."""
    pool = sum(counts.values())
    targets = {name: Fraction(total * count, pool) if count else Fraction(0) for name, count in counts.items()}
    draws = {name: math.floor(target) for name, target in targets.items()}
    left = total - sum(draws.values())
    for name in sorted(counts, key=lambda name: (draws[name] - targets[name], list(tie_break).index(name)))[:left]:
        draws[name] += 1
    return draws, targets


def class_draws(
    counts: Mapping[str, int],
    *,
    prompt_draws: int,
    repeat_cap: int,
    class_a_share_cap: float,
    tie_break: Sequence[str] = (CLASS_B, CLASS_C),
) -> tuple[dict[str, int], dict[str, Fraction]]:
    """Return each class's whole draws of the D prompt draws and its exact target: class (a) gets floor(D * min(cap,
    repeat_cap * n_a / D)); (b) and (c) split the rest, each scenario at most once."""
    if not sum(counts.values()):
        raise GrpoDataError("no training scenario is left to weight")
    n_a = counts.get(CLASS_A, 0)
    share_a = min(Fraction(str(class_a_share_cap)), Fraction(repeat_cap * n_a, prompt_draws))
    targets = {CLASS_A: prompt_draws * share_a}
    draws = {CLASS_A: math.floor(targets[CLASS_A])}
    rest = prompt_draws - draws[CLASS_A]
    others = {name: counts.get(name, 0) for name in (CLASS_B, CLASS_C)}
    pool = sum(others.values())
    if rest > pool:
        raise GrpoDataError(
            f"class (a) ({n_a} scenario(s)) takes {draws[CLASS_A]} of the {prompt_draws} prompt draws; the other "
            f"{rest} are drawn from classes (b) and (c), each scenario at most once, but they hold {pool} "
            "scenario(s): fewer prompt draws, or more training scenarios"
        )
    rest_draws, rest_targets = proportional_draws(rest, others, tie_break)
    return {**draws, **rest_draws}, {**targets, **rest_targets}


def interleave(streams: Mapping[str, Sequence[str]]) -> list[str]:
    """Merge the streams so that every prefix holds each class within one row of its share (the trainer keeps this
    order only with grpo.shuffle_dataset false); the order is part of the rows file's bytes."""
    total = sum(len(stream) for stream in streams.values())
    placed = dict.fromkeys(streams, 0)
    order: list[str] = []
    for row in range(1, total + 1):
        behind = {name: row * len(stream) - placed[name] * total for name, stream in streams.items()}
        name = max(behind, key=behind.__getitem__)
        order.append(streams[name][placed[name]])
        placed[name] += 1
    return order


def spread_draws(
    members: Mapping[str, Sequence[str]], draws: Mapping[str, int], rng: random.Random
) -> tuple[dict[str, int], list[str]]:
    """Spread each group's draws evenly over a seeded shuffle of its members and interleave the groups; return each
    scenario's copies and the rows' order."""
    copies: dict[str, int] = {}
    streams: dict[str, list[str]] = {}
    for name, sids in members.items():
        shuffled = list(sids)
        rng.shuffle(shuffled)
        per, extra = divmod(draws[name], len(shuffled)) if shuffled else (0, 0)
        for position, sid in enumerate(shuffled):
            copies[sid] = per + (1 if position < extra else 0)
        streams[name] = [sid for index in range(per + (extra > 0)) for sid in shuffled if copies[sid] > index]
    return copies, interleave(streams)


def copies_summary(values: Sequence[int]) -> dict[str, Any]:
    """Summarize some scenarios' copies: the least, the most and how many scenarios carry each count."""
    return {
        "min": min(values, default=0),
        "max": max(values, default=0),
        "scenarios_by_copies": {str(count): number for count, number in sorted(Counter(values).items())},
    }


def copies_declaration(members: Mapping[str, Sequence[str]], copies: Mapping[str, int]) -> dict[str, Any]:
    """Return the declaration's copies and drawn/not-drawn counts per group."""
    per_group = {name: [copies[sid] for sid in sids] for name, sids in members.items() if sids}
    return {
        "copies": {name: copies_summary(values) for name, values in per_group.items()},
        "scenarios_drawn": {name: sum(1 for value in values if value) for name, values in per_group.items()},
        "scenarios_not_drawn": {name: sum(1 for value in values if not value) for name, values in per_group.items()},
    }


def sampling_weights(
    classes: Mapping[str, str],
    *,
    prompt_draws: int = PROMPT_DRAWS,
    repeat_cap: int = REPEAT_CAP,
    class_a_share_cap: float = CLASS_A_SHARE_CAP,
    seed: int = DRAW_SEED,
) -> SamplingWeights:
    """Return the run's D prompt draws as rows under the anchor-class rule, each scenario drawn at most `repeat_cap`
    times, every prefix as balanced as the whole (`interleave`)."""
    if prompt_draws < 1 or repeat_cap < 1 or not 0 < class_a_share_cap <= 1:
        raise GrpoDataError(
            f"prompt_draws ({prompt_draws}) and repeat_cap ({repeat_cap}) must be positive and "
            f"class_a_share_cap ({class_a_share_cap}) in (0, 1]"
        )
    unknown = sorted(set(classes.values()) - set(ANCHOR_CLASSES))
    if unknown:
        raise GrpoDataError(f"unknown anchor classes {unknown}")
    members = {name: [sid for sid, cls in classes.items() if cls == name] for name in ANCHOR_CLASSES}
    counts = {name: len(sids) for name, sids in members.items()}
    rng = random.Random(seed)
    tie_break = [CLASS_B, CLASS_C]
    rng.shuffle(tie_break)
    draws, targets = class_draws(
        counts, prompt_draws=prompt_draws, repeat_cap=repeat_cap, class_a_share_cap=class_a_share_cap,
        tie_break=tie_break,
    )
    copies, order = spread_draws(members, draws, rng)
    n = len(classes)
    declaration = {
        "rule": "of the D = prompt_draws draws, a: floor(D * min(class_a_share_cap, repeat_cap * n_a / D)), even "
                "copies; b and c: the rest in proportion to n_b : n_c (largest remainder, seeded tie-break), each "
                "scenario drawn at most once",
        "selection": "random.Random(seed): the b/c tie-break order, then each class's scenarios (in the training "
                     "file's order) shuffled, a, b, c; a class's draws go to the first of its shuffle, the extra "
                     "copies of a to the first; the rest are not drawn",
        "order": "the D rows interleaved: row i goes to the class furthest behind i * draws / D (ties a, b, c), so "
                 "every prefix holds each class within one row of its share; a class's own rows are passes over "
                 "its shuffle",
        "seed": seed,
        "tie_break": tie_break,
        "prompt_draws": prompt_draws,
        "repeat_cap": repeat_cap,
        "class_a_share_cap": class_a_share_cap,
        "counts": counts,
        "natural_shares": {name: float(Fraction(counts[name], n)) for name in ANCHOR_CLASSES},
        "target_shares": {name: float(targets[name] / prompt_draws) for name in ANCHOR_CLASSES},
        "target_draws": {name: float(targets[name]) for name in ANCHOR_CLASSES},
        "draws": draws,
        "realized_shares": {name: float(Fraction(draws[name], prompt_draws)) for name in ANCHOR_CLASSES},
        **copies_declaration(members, copies),
        "scenarios_not_drawn_b_c": sum(1 for sid, cls in classes.items() if cls != CLASS_A and not copies[sid]),
        "rows": len(order),
        "max_draws_per_scenario": max(copies.values()),
        "consumption": "the rows are the run's D draws when grpo.max_steps x prompts per step = prompt_draws: TRL's "
                       "RepeatSampler walks one permutation of them (grpo.shuffle_dataset: true, seeded by "
                       "grpo.seed), each row once; the class counts hold exactly, the prefix balance only unshuffled",
    }
    return SamplingWeights(copies=copies, order=order, declaration=declaration)


def favoured_verdict(row: Mapping[str, Any]) -> tuple[str, float]:
    """Return a row's favoured judgment under the reward (its largest q̂, ties by `TIE_ORDER`) and its margin over the
    runner-up, rounded to 12 decimals."""
    q = {verdict: float(row[column]) for verdict, column in Q_COLUMNS.items()}
    verdict = mode_of(q)
    if verdict != row["mode"]:
        raise GrpoDataError(
            f"scenario {str(row.get('scenario_id'))[:12]}: its q-hat columns favour {verdict!r} but its mode is "
            f"{row['mode']!r}; the judgment mix reads the reward's own target"
        )
    return verdict, round(q[verdict] - max(value for name, value in q.items() if name != verdict), 12)


def _margin_summary(margins: Sequence[float]) -> dict[str, float]:
    return {"min": min(margins), "median": statistics.median(margins), "max": max(margins)}


def verdict_stratum(verdict: str, margin: float, no_margin: float) -> str:
    """Return the judgment mix's stratum of a row: `no` when it favours no by at least `no_margin`, `no_under_margin` when
    by less, else the judgment."""
    if verdict != "no":
        return verdict
    return STRATUM_NO if margin >= no_margin else STRATUM_NO_UNDER_MARGIN


#: The margins the judgment mix counts the no-favouring rows at.
MARGIN_STEPS = tuple(step / 10 for step in range(1, 11))


def favouring_no_at_least(margins: Sequence[float]) -> dict[str, int]:
    """Return how many of the no-favouring rows' `margins` reach each of `MARGIN_STEPS`."""
    return {f"{step:.1f}": sum(1 for margin in margins if margin >= step) for step in MARGIN_STEPS}


def check_verdict_mix(*, no_share: float, no_margin: float, prompt_draws: int, repeat_cap: int) -> None:
    """Refuse a judgment mix out of range: positive draws and cap, share and margin in (0, 1]."""
    if prompt_draws < 1 or repeat_cap < 1 or not 0 < no_share <= 1 or not 0 < no_margin <= 1:
        raise GrpoDataError(
            f"prompt_draws ({prompt_draws}) and repeat_cap ({repeat_cap}) must be positive, and no_share "
            f"({no_share}) and no_margin ({no_margin}) in (0, 1] (a margin of 0 would admit the ties, which the "
            "reward pays alike for either tied judgment)"
        )


_NO_STRATUM_RULE = (
    "of the D = prompt_draws draws, the 'no' stratum (favoured judgment no by at least no_margin, which is positive, "
    "so never a tie): ceil(D * no_share), even copies, at most repeat_cap each (refused when it cannot); the rest: "
)


def verdict_weights(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    no_share: float,
    no_margin: float,
    prompt_draws: int = PROMPT_DRAWS,
    repeat_cap: int = REPEAT_CAP,
    seed: int = DRAW_SEED,
) -> SamplingWeights:
    """Return the run's D prompt draws stratified by the reward's own target: the `no` stratum takes at least `no_share`
    of the draws (at most `repeat_cap` per scenario; refused when it cannot), the rest go by favoured judgment, each
    scenario at most once."""
    check_verdict_mix(no_share=no_share, no_margin=no_margin, prompt_draws=prompt_draws, repeat_cap=repeat_cap)
    if not rows:
        raise GrpoDataError("no training scenario is left to weight")
    favoured = {sid: favoured_verdict(row) for sid, row in rows.items()}
    stratum_of = {sid: verdict_stratum(verdict, margin, no_margin) for sid, (verdict, margin) in favoured.items()}
    members = {name: [sid for sid, stratum in stratum_of.items() if stratum == name] for name in VERDICT_STRATA}
    counts = {name: len(sids) for name, sids in members.items()}
    no_margins = [margin for verdict, margin in favoured.values() if verdict == "no"]
    at_least = favouring_no_at_least(no_margins)
    no_exact = prompt_draws * Fraction(str(no_share))
    no_draws = math.ceil(no_exact)
    if counts[STRATUM_NO] * repeat_cap < no_draws:
        table = ", ".join(f"{margin}: {count} ({count * repeat_cap})" for margin, count in at_least.items())
        raise GrpoDataError(
            f"the declared no share {no_share} of the {prompt_draws} prompt draws is {no_draws} draws, but "
            f"{counts[STRATUM_NO]} scenario(s) favour 'no' by at least {no_margin}, and at most {repeat_cap} draws "
            f"each give {counts[STRATUM_NO] * repeat_cap}: more such scenarios (a larger pool), a lower share, a "
            f"lower margin or a higher repeat cap. The scenarios favouring 'no' by at least each margin, and the "
            f"draws they give at a cap of {repeat_cap}: {table}"
        )
    rest = prompt_draws - no_draws
    others = {name: counts[name] for name in REST_STRATA}
    rng = random.Random(seed)
    tie_break = list(REST_STRATA)
    rng.shuffle(tie_break)
    if rest > sum(others.values()):
        raise GrpoDataError(
            f"the 'no' stratum takes {no_draws} of the {prompt_draws} prompt draws; the other {rest} are drawn "
            f"from the rest, each scenario at most once, but it holds {sum(others.values())} scenario(s): fewer "
            "prompt draws, a larger share, or more training scenarios"
        )
    rest_draws, rest_targets = proportional_draws(rest, others, tie_break)
    draws = {STRATUM_NO: no_draws, **rest_draws}
    targets = {STRATUM_NO: no_exact, **rest_targets}
    copies, order = spread_draws(members, draws, rng)
    declaration = {
        "stratification": "favoured_verdict",
        "rule": _NO_STRATUM_RULE + "the other draws over no_under_margin, unsure and yes in proportion to their "
                                   "counts (largest remainder, seeded tie-break), each scenario drawn at most once",
        "favoured_verdict": "the largest q-hat of the row's columns, ties by TIE_ORDER (no, unsure, yes), the "
                            "reward's own target; margin: that q-hat less the next largest, rounded to 12 decimals",
        "estimate": "in sample: the strata are read from the q-hat the reward pays, so selecting on it favours rows "
                    "whose estimate overshoots, and a fresh draw of a 'no' row says all-no less often than its q-hat",
        "selection": "random.Random(seed): the rest's tie-break order, then each stratum's scenarios (in the "
                     "training file's order) shuffled, no, no_under_margin, unsure, yes; a stratum's draws go to "
                     "the first of its shuffle, the extra copies of 'no' to the first; the rest are not drawn",
        "order": "the D rows interleaved: row i goes to the stratum furthest behind i * draws / D (ties in the "
                 "strata's order), so every prefix holds each stratum within one row of its share",
        "seed": seed,
        "tie_break": tie_break,
        "prompt_draws": prompt_draws,
        "repeat_cap": repeat_cap,
        "no_share": no_share,
        "no_margin": no_margin,
        "counts": counts,
        "natural_shares": {name: float(Fraction(counts[name], len(rows))) for name in VERDICT_STRATA},
        "target_shares": {name: float(targets[name] / prompt_draws) for name in VERDICT_STRATA},
        "target_draws": {name: float(targets[name]) for name in VERDICT_STRATA},
        "draws": draws,
        "realized_shares": {name: float(Fraction(draws[name], prompt_draws)) for name in VERDICT_STRATA},
        "draws_by_favoured_verdict": dict(sorted(Counter(favoured[sid][0] for sid in order).items())),
        "draws_by_class": dict(sorted(Counter(rows[sid]["anchor_class"] for sid in order).items())),
        **copies_declaration(members, copies),
        "margins": {
            "drawn_no": _margin_summary([favoured[sid][1] for sid in members[STRATUM_NO] if copies[sid]]),
            "favouring_no": _margin_summary(no_margins),
            "favouring_no_by_a_tie": sum(1 for margin in no_margins if margin == 0),
            "favouring_no_at_least": at_least,
        },
        "rows": len(order),
        "max_draws_per_scenario": max(copies.values()),
        "consumption": "the rows are the run's D draws when grpo.max_steps x prompts per step = prompt_draws: TRL's "
                       "RepeatSampler walks one permutation of them (grpo.shuffle_dataset: true, seeded by "
                       "grpo.seed), each row once; the stratum counts hold exactly, the prefix balance only "
                       "unshuffled",
    }
    return SamplingWeights(copies=copies, order=order, declaration=declaration)


def draw_rule(
    *,
    prompt_draws: int = PROMPT_DRAWS,
    repeat_cap: int = REPEAT_CAP,
    seed: int = DRAW_SEED,
    class_a_share_cap: Optional[float] = None,
    no_share: Optional[float] = None,
    no_margin: Optional[float] = None,
) -> Callable[[Mapping[str, Mapping[str, Any]]], SamplingWeights]:
    """Return the draws a build declares: the anchor-class rule without `no_share`, else the judgment mix; declaring both,
    or half of the mix, is refused."""
    if no_share is None:
        if no_margin is not None:
            raise GrpoDataError(f"no_margin ({no_margin}) belongs to the judgment mix: declare its no_share too")
        cap = CLASS_A_SHARE_CAP if class_a_share_cap is None else class_a_share_cap

        def by_class(rows: Mapping[str, Mapping[str, Any]]) -> SamplingWeights:
            return sampling_weights({sid: row["anchor_class"] for sid, row in rows.items()}, prompt_draws=prompt_draws,
                                    repeat_cap=repeat_cap, class_a_share_cap=cap, seed=seed)

        return by_class
    if class_a_share_cap is not None:
        raise GrpoDataError(
            f"class_a_share_cap ({class_a_share_cap}) belongs to the anchor-class rule, and no_share ({no_share}) "
            "declares the judgment mix: declare one rule"
        )
    if no_margin is None:
        raise GrpoDataError(f"the judgment mix (no_share {no_share}) needs its margin: declare no_margin")
    check_verdict_mix(no_share=no_share, no_margin=no_margin, prompt_draws=prompt_draws, repeat_cap=repeat_cap)

    def by_verdict(rows: Mapping[str, Mapping[str, Any]]) -> SamplingWeights:
        return verdict_weights(rows, no_share=no_share, no_margin=no_margin, prompt_draws=prompt_draws,
                               repeat_cap=repeat_cap, seed=seed)

    return by_verdict


class DisjointnessMarkers(Protocol):
    """What the build needs of the prompt-disjointness marker."""

    def check(self, dataset: Path, required_protected: Sequence[str] = (), required_carved: Sequence[str] = ()
              ) -> Mapping[str, Any]:
        """Return the marker beside `dataset` for its current bytes, checked against every file `required_protected`
        names and reading every file `required_carved` names; raises otherwise."""

    def attest(self, scenarios: Path, source: Mapping[str, Any], extra_protected: Sequence[Path]) -> dict[str, Any]:
        """Return the second pass's marker over `scenarios`: the source's protected files stay protected, and only its
        carved files and `extra_protected` are carved (read by the second pass alone); raises on a leak."""

    def write(self, dataset: Path, marker: Mapping[str, Any]) -> Path:
        """Write `marker` beside `dataset`, validated first."""


class Block2Markers:
    """`DisjointnessMarkers` through `ruleforge.grpo.exclusion`, bound by name."""

    def __init__(self) -> None:
        try:
            self.module = importlib.import_module(EXCLUSION_MODULE)
        except ImportError as error:
            raise LeakageError(
                f"{EXCLUSION_MODULE} is not importable ({error}); the exclusion module must be importable "
                "to check the training scenarios' marker and attest the outputs, and the build refuses without it"
            ) from error

    def check(self, dataset: Path, required_protected: Sequence[str] = (), required_carved: Sequence[str] = ()
              ) -> Mapping[str, Any]:
        # An unscreened source marker's screen goes into the markers the build writes, so the trainer's check decides.
        return self.module.check_prompt_marker(dataset, required_protected=list(required_protected),
                                               required_carved=list(required_carved), allow_unscreened=True)

    def attest(self, scenarios: Path, source: Mapping[str, Any], extra_protected: Sequence[Path]) -> dict[str, Any]:
        policy = self.module.ExclusionPolicy.from_mapping(source["policy"])
        # the build's added files and the source's carved ones have second-pass coverage only, so they stay carved
        carved = [str(entry["path"]) for entry in source.get("carved") or ()] + [str(path) for path in extra_protected]
        marker = self.module.prompt_disjoint_marker(
            scenarios, [str(entry["path"]) for entry in source["protected"]], policy=policy,
            allowed_family_pairs=[tuple(pair) for pair in source.get("allowed_family_pairs") or ()],
            carved=carved, screen=self.module.marker_screen(source),
        )
        before = {str(entry["sha256"]) for key in ("protected", "carved") for entry in source.get(key) or ()}
        now = {str(entry["sha256"]) for key in ("protected", "carved") for entry in marker.get(key) or ()}
        if not before <= now:
            raise LeakageError(
                f"the protected files the training scenarios were attested against have changed since "
                f"({len(before - now)} digest(s) no longer read); re-attest the scenarios first"
            )
        return marker

    def write(self, dataset: Path, marker: Mapping[str, Any]) -> Path:
        return self.module.write_prompt_disjoint_marker(dataset, marker)


def derived_marker(
    scenario_marker: Mapping[str, Any], *, name: str, sha256: str, rows: int, scenarios_marker_sha256: str
) -> dict[str, Any]:
    """Return the marker of an output built from the attested scenario file: that attestation, for the output's bytes,
    naming the scenario file and its marker by digest."""
    marker = copy.deepcopy(dict(scenario_marker))
    scenarios = dict(scenario_marker["dataset"])
    marker["dataset"] = {**scenarios, "name": name, "sha256": sha256, "rows": rows}
    marker["derived_from"] = {
        "by": ROWS_DERIVATION,
        "name": scenarios["name"],
        "sha256": scenarios["sha256"],
        "rows": scenarios["rows"],
        "marker_sha256": scenarios_marker_sha256,
    }
    return marker


@dataclass(frozen=True)
class AnchorInputs:
    """A train-side anchor set, its files checked against its manifest; the replay index stays on disk."""

    directory: Path
    manifest: dict[str, Any]
    rows: dict[str, dict[str, Any]]

    @property
    def replay_index(self) -> Path:
        return self.directory / REPLAY_INDEX_FNAME

    @classmethod
    def open(cls, directory: Path | str) -> "AnchorInputs":
        base = Path(directory)
        manifest = json.loads((base / ANCHORS_MANIFEST_FNAME).read_text(encoding="utf-8"))
        if manifest.get("schema_version") not in ANCHORS_SCHEMA_VERSIONS:
            raise GrpoDataError(f"{base} is not a schema_version {one_of(ANCHORS_SCHEMA_VERSIONS)} anchor set")
        if manifest.get("side") != SIDE_TRAIN:
            raise GrpoDataError(
                f"{base} is a {manifest.get('side')!r}-side anchor set; the GRPO and replay rows come from the train "
                "anchors (the dev references are never trained on)"
            )
        for name, recorded in manifest["outputs"].items():
            if sha256_file(base / name) != recorded["sha256"]:
                raise GrpoDataError(f"{base / name} is not the file its manifest records")
        rows = {row["scenario_id"]: row for row in iter_jsonl_records(base / ANCHORS_FNAME)}
        stages = manifest.get("stages")
        unstaged = sorted(sid for sid, row in rows.items() if stages is not None and row.get("stage") not in stages)
        if unstaged:
            raise GrpoDataError(
                f"{base}: {len(unstaged)} anchor row(s) name no stage of the set (first {unstaged[0][:12]}): a row "
                "with no q-hat"
            )
        return cls(base, manifest, rows)

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.directory),
            "manifest_sha256": sha256_file(self.directory / ANCHORS_MANIFEST_FNAME),
            "outputs": self.manifest["outputs"],
            "estimator": self.manifest.get("estimator"),
            "allow_untopped": self.manifest.get("allow_untopped"),
            "counts": self.manifest.get("counts"),
        }


def estimate_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return what makes anchor sets one estimate: the ruleset, the estimator and the reference condition."""
    reference = manifest.get("reference") or {}
    return {
        "rules_sha256": manifest["rules"]["sha256"],
        "estimator": manifest.get("estimator"),
        "tie_order": manifest.get("tie_order"),
        "projection_excluded": manifest.get("projection_excluded"),
        "reference": {key: reference.get(key) for key in ("sha256", "condition_id", "fingerprint")},
    }


@dataclass(frozen=True)
class AnchorSets:
    """The train-side anchor sets one build reads: one, or several over disjoint scenarios that are one estimate of one
    ruleset; their rows are merged and their replay indexes read in turn."""

    sets: tuple[AnchorInputs, ...]
    rows: dict[str, dict[str, Any]]

    @classmethod
    def open(cls, directories: Path | str | Sequence[Path | str]) -> "AnchorSets":
        paths = [directories] if isinstance(directories, (str, Path)) else list(directories)
        if not paths:
            raise GrpoDataError("no anchor set given")
        if len({Path(path).resolve() for path in paths}) != len(paths):
            raise GrpoDataError(f"an anchor set is named twice among {[str(path) for path in paths]}")
        sets = tuple(AnchorInputs.open(path) for path in paths)
        first = estimate_identity(sets[0].manifest)
        rows: dict[str, dict[str, Any]] = {}
        for anchor_set in sets:
            identity = estimate_identity(anchor_set.manifest)
            differing = sorted(key for key in first if identity[key] != first[key])
            if differing:
                raise GrpoDataError(
                    f"{anchor_set.directory} is not one estimate with {sets[0].directory}: their {differing} differ"
                )
            shared = sorted(set(rows) & set(anchor_set.rows))
            if shared:
                raise GrpoDataError(
                    f"{anchor_set.directory} anchors {len(shared)} scenario(s) an earlier anchor set anchors (first "
                    f"{shared[0][:12]}); a scenario has one q-hat and one set of replay rows"
                )
            rows.update(anchor_set.rows)
        return cls(sets, rows)

    @property
    def directory(self) -> str:
        """Return the sets' directories, for messages."""
        return ", ".join(str(anchor_set.directory) for anchor_set in self.sets)

    @property
    def rules_sha256(self) -> str:
        return str(self.sets[0].manifest["rules"]["sha256"])

    @property
    def staged(self) -> bool:
        """Return whether a set is staged."""
        return any(anchor_set.manifest.get("stages") is not None for anchor_set in self.sets)

    def replay_inputs(self) -> list[str]:
        """Return every set's replayed inputs, each name once, in the sets' order."""
        return list(dict.fromkeys(name for anchor_set in self.sets for name in replay_inputs(anchor_set.manifest)))

    def replay_groups(self) -> Iterator[tuple[str, list[dict[str, Any]]]]:
        """Yield every set's replay index in turn; an index row of another set's scenario is refused."""
        for anchor_set in self.sets:
            for sid, group in replay_groups(anchor_set.replay_index):
                if sid not in anchor_set.rows:
                    raise GrpoDataError(
                        f"{anchor_set.replay_index} holds replay rows of scenario {sid[:12]}, which its anchor set "
                        "does not anchor"
                    )
                yield sid, group

    def sources(self, name: str, *, evaluation_mode: str) -> dict[str, CampaignCondition]:
        """Return every root of input `name` of each set that has it, merged; a root two sets read is refused."""
        merged: dict[str, CampaignCondition] = {}
        for anchor_set in self.sets:
            if name != "full_set" and name not in replay_inputs(anchor_set.manifest):
                continue
            for root, source in open_sources(anchor_set.manifest, name, evaluation_mode=evaluation_mode).items():
                if root in merged:
                    raise GrpoDataError(f"{name} root {root} is an input of two anchor sets")
                merged[root] = source
        return merged

    def record(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Return one set's record, or several sets' as a list in the order given."""
        records = [anchor_set.record() for anchor_set in self.sets]
        return records[0] if len(records) == 1 else records


def open_source(record: Mapping[str, Any], *, evaluation_mode: str) -> CampaignCondition:
    """Open the campaign condition an anchor manifest names, refused unless it is still the one recorded."""
    source = open_condition(Path(record["root"]), record["condition_id"], evaluation_mode=evaluation_mode)
    now = {key: value for key, value in source.record().items() if key != "root"}
    then = {key: value for key, value in record.items() if key != "root"}
    if now != then:
        differing = sorted(key for key in set(now) | set(then) if now.get(key) != then.get(key))
        raise GrpoDataError(f"{record['root']}/{record['condition_id']} is not the campaign the anchors read: {differing}")
    return source


def open_sources(manifest: Mapping[str, Any], name: str, *, evaluation_mode: str) -> dict[str, CampaignCondition]:
    """Return every campaign root the anchor set read for input `name`, keyed by the root string the replay rows name;
    `{}` for none."""
    return {
        record["root"]: open_source(record, evaluation_mode=evaluation_mode)
        for record in input_records(manifest, name)
    }


def replay_inputs(manifest: Mapping[str, Any]) -> list[str]:
    """Return the inputs whose draws define q̂ and are replayed: a staged set's stages, else the bank and the top-up when
    there was one."""
    if manifest.get("stages") is not None:
        return list(manifest["stages"])
    return [SOURCE_BANK, *([SOURCE_TOP_UP] if input_records(manifest, SOURCE_TOP_UP) else [])]


def row_source(anchor: Mapping[str, Any]) -> str:
    """Return the input an anchor row's q̂ and replay rows come from."""
    if "stage" in anchor:
        return str(anchor["stage"])
    return SOURCE_TOP_UP if anchor["topped_up"] else SOURCE_BANK


def judging_roots(sources: Mapping[str, Mapping[str, CampaignCondition]]) -> dict[tuple[str, str], str]:
    """Return `{(input, scenario_id): the root of that input that judged it}`."""
    return {
        (name, sid): root
        for name, roots in sources.items()
        for root, source in roots.items()
        for sid in source.scenario_ids
    }


def campaign_record(source: CampaignCondition, *, raw: bool) -> dict[str, Any]:
    """Return a campaign input's record with each draw's ledger digest (and raw outputs' when `raw`)."""
    draws = {}
    for draw in range(source.n_draws):
        directory = source.root / source.condition_id / draw_dirname(draw)
        entry = {"attempts_sha256": sha256_file(directory / LEDGER_FNAME)}
        if raw:
            entry["raw_outputs_sha256"] = sha256_file(directory / RAW_OUTPUTS_FNAME)
        draws[draw_dirname(draw)] = entry
    return {**source.record(), "draws": draws}


def campaign_records(sources: Mapping[str, CampaignCondition], *, raw: bool) -> dict[str, Any] | list[dict[str, Any]]:
    """Return an input's `campaign_record`: one record for a single root, a list for several."""
    records = [campaign_record(source, raw=raw) for source in sources.values()]
    return records[0] if len(records) == 1 else records


@dataclass(frozen=True)
class TrainingScenario:
    """One training scenario: its row, its line (verbatim) and its anchor row."""

    scenario_id: str
    row: dict[str, Any]
    line: str
    anchor: dict[str, Any]


def read_training_scenarios(path: Path, anchors: AnchorInputs | AnchorSets) -> list[TrainingScenario]:
    """Return the training scenarios in file order, each with its anchor row; refused unless every one was anchored.
    """
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    scenarios: list[TrainingScenario] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        row = json.loads(line)
        sid = compute_scenario_id(row)
        if sid in seen:
            raise GrpoDataError(f"{path} names scenario {sid[:12]} twice (line {number})")
        if sid not in anchors.rows:
            raise GrpoDataError(
                f"{path} line {number} (scenario {sid[:12]}) has no anchor row in {anchors.directory}; every "
                "training scenario needs q-hat and its replay rows"
            )
        seen.add(sid)
        scenarios.append(TrainingScenario(sid, row, line, anchors.rows[sid]))
    if not scenarios:
        raise GrpoDataError(f"{path} holds no training scenario")
    return scenarios


@dataclass(frozen=True)
class ProtectedFile:
    """A `protected` file from one read of its bytes: their digest and each row's scenario id."""

    path: Path
    sha256: str
    scenario_ids: tuple[str, ...]

    @classmethod
    def read(cls, path: Path | str) -> "ProtectedFile":
        path = Path(path)
        data = path.read_bytes()
        ids = []
        for number, line in enumerate(data.decode("utf-8").split("\n"), start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise GrpoDataError(f"{path} line {number} is not JSON ({error.msg})") from error
                if not isinstance(row, dict):
                    raise GrpoDataError(f"{path} line {number} is not a JSON object")
                ids.append(compute_scenario_id(row))
        return cls(path, hashlib.sha256(data).hexdigest(), tuple(ids))

    def anchored(self, anchors: AnchorInputs | AnchorSets) -> int:
        """Return how many of its rows are rows of `anchors`."""
        return sum(sid in anchors.rows for sid in self.scenario_ids)

    def record(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "rows": len(self.scenario_ids)}


def held_out_files(
    files: Sequence[ProtectedFile], anchors: AnchorInputs | AnchorSets, training: Sequence[TrainingScenario]
) -> list[ProtectedFile]:
    """Return the protected files that carve anchor rows out of training, by scenario id; a file naming a training
    scenario is refused."""
    trained = {scenario.scenario_id for scenario in training}
    for protected in files:
        leaked = [sid for sid in protected.scenario_ids if sid in trained]
        if leaked:
            raise LeakageError(
                f"{protected.path}, a protected file, names {len(leaked)} of the {len(trained)} training scenarios "
                f"(first {leaked[0][:12]}); a held-out scenario must not be trained on"
            )
    return [
        protected for protected in files
        if protected.scenario_ids and protected.anchored(anchors) == len(protected.scenario_ids)
    ]


def full_set_prompt_digests(*sources: CampaignCondition) -> dict[str, set[str]]:
    """Return each scenario's full-set prompt digests over every draw of the base's full-set campaign."""
    digests: dict[str, set[str]] = {}
    for source in sources:
        for draw in range(source.n_draws):
            state = DrawLedger(
                source.root / source.condition_id / draw_dirname(draw),
                campaign_id=source.manifest["campaign_id"], condition_id=source.condition_id, decoder_draw_id=draw,
            ).load()
            for sid in state.accepted:
                _, component = state.accepted_components(sid)[FULL_RULESET_COMPONENT]
                digests.setdefault(sid, set()).add(str(component.prompt_sha256))
    return digests


@dataclass(frozen=True)
class GrpoPrompt:
    """One training scenario's GRPO prompt and its length in the base's tokens."""

    scenario: TrainingScenario
    entry: dict[str, str]
    tokens: int


def grpo_prompts(
    scenarios: Sequence[TrainingScenario],
    *,
    prompts: CampaignPrompts,
    tokenizer: Any,
    enable_thinking: bool,
    full_set_digests: Mapping[str, set[str]],
    trl_tokenizer: Any,
) -> list[GrpoPrompt]:
    """Return every scenario's full-set prompt, its digest checked against the campaign's and its TRL token ids checked
    against the campaign rendering's, with its token count."""
    entries = [prompts.entry(scenario.row, FULL_RULESET_COMPONENT) for scenario in scenarios]
    for scenario, entry in zip(scenarios, entries):
        recorded = full_set_digests.get(scenario.scenario_id, set())
        if recorded != {prompt_sha256(entry)}:
            raise GrpoDataError(
                f"scenario {scenario.scenario_id[:12]}: the full-set prompt rebuilt here is not the one the base's "
                f"full-set campaign sent (digests {sorted(recorded)[:2]}); the GRPO prompt must be the campaign's"
            )
    kwargs = chat_template_kwargs(enable_thinking)
    lengths: list[int] = []
    for start in range(0, len(entries), RENDER_BATCH):
        chunk = entries[start : start + RENDER_BATCH]
        trl = trl_prompt_ids(trl_tokenizer, [conversation(entry) for entry in chunk], kwargs)
        campaign = campaign_prompt_ids(tokenizer, chunk, enable_thinking=enable_thinking)
        for scenario, trl_ids, campaign_ids in zip(scenarios[start : start + RENDER_BATCH], trl, campaign):
            if trl_ids != campaign_ids:
                first = next(
                    (i for i, (a, b) in enumerate(zip(trl_ids, campaign_ids)) if a != b),
                    min(len(trl_ids), len(campaign_ids)),
                )
                raise GrpoDataError(
                    f"scenario {scenario.scenario_id[:12]}: TRL renders the conversation as {len(trl_ids)} tokens, "
                    f"the campaign sends {len(campaign_ids)} (first difference at token {first}); training and "
                    "evaluation must see one token sequence"
                )
        lengths.extend(len(ids) for ids in trl)
    return [GrpoPrompt(scenario, entry, length) for scenario, entry, length in zip(scenarios, entries, lengths)]


def grpo_row(prompt: GrpoPrompt, copies: int) -> dict[str, Any]:
    """Return one GRPO row: the conversation, the columns the reward reads and provenance."""
    anchor = prompt.scenario.anchor
    return {
        "prompt": conversation(prompt.entry),
        "scenario_id": prompt.scenario.scenario_id,
        "q_yes": anchor["q_yes"],
        "q_unsure": anchor["q_unsure"],
        "q_no": anchor["q_no"],
        "mode": anchor["mode"],
        "anchor_class": anchor["anchor_class"],
        "topped_up": anchor["topped_up"],
        "k": anchor["k"],
        "base_yes_rules": list(anchor["base_yes_rules"]),
        "sampling_weight": copies,
        "prompt_tokens": prompt.tokens,
    }


def grpo_rows(prompts: Sequence[GrpoPrompt], weights: SamplingWeights) -> list[dict[str, Any]]:
    """Return the D rows, one per draw in the draws' order, each with its scenario's copies."""
    by_id = {prompt.scenario.scenario_id: prompt for prompt in prompts}
    return [grpo_row(by_id[sid], weights.copies[sid]) for sid in weights.order]


@dataclass(frozen=True)
class RawKey:
    """Where one replayed completion's raw text is: the source, its root, the draw and its ledger key."""

    source: str
    root: str
    decoder_draw_id: int
    scenario_id: str
    semantic_attempt_id: int
    request_block_id: str
    rule_id: str

    @classmethod
    def of(cls, entry: Mapping[str, Any]) -> "RawKey":
        return cls(
            str(entry["source"]), str(entry["root"]), int(entry["decoder_draw_id"]), str(entry["scenario_id"]),
            int(entry["semantic_attempt_id"]), str(entry["request_block_id"]), str(entry["rule_id"]),
        )


class RawTexts:
    """The replayed completions, found by one pass over each draw's `raw_outputs.jsonl` and read back by offset; a row
    whose text does not hash to the index's digest refuses the build."""

    def __init__(
        self, sources: Mapping[str, Mapping[str, CampaignCondition]], wanted: Mapping[RawKey, str]
    ) -> None:
        self._handles: dict[tuple[str, str, int], Any] = {}
        self._offsets: dict[RawKey, int] = {}
        self._wanted = dict(wanted)
        by_file: dict[tuple[str, str, int], dict[tuple[str, int, str, str], RawKey]] = {}
        for key in wanted:
            by_file.setdefault((key.source, key.root, key.decoder_draw_id), {})[
                (key.scenario_id, key.semantic_attempt_id, key.request_block_id, key.rule_id)
            ] = key
        try:
            for (name, root, draw), keys in sorted(by_file.items()):
                source = sources[name][root]
                path = source.root / source.condition_id / draw_dirname(draw) / RAW_OUTPUTS_FNAME
                handle = self._handles[(name, root, draw)] = path.open("rb")
                offset = 0
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        key = keys.get((str(row["scenario_id"]), int(row["semantic_attempt_id"]),
                                        str(row["request_block_id"]), str(row["component_id"])))
                        if key is not None and key not in self._offsets and (
                            raw_output_sha256(row.get("raw_response")) == wanted[key] == row.get("raw_output_sha256")
                        ):
                            self._offsets[key] = offset
                    offset += len(line)
        except BaseException:
            self.close()
            raise
        missing = [key for key in wanted if key not in self._offsets]
        if missing:
            self.close()
            raise GrpoDataError(
                f"{len(missing)} replayed completion(s) have no raw output whose text hashes to the digest the replay "
                f"index names (first {missing[0].scenario_id[:12]}/{missing[0].rule_id}, {missing[0].source} draw "
                f"{missing[0].decoder_draw_id} of {missing[0].root})"
            )

    def read(self, key: RawKey) -> dict[str, Any]:
        handle = self._handles[(key.source, key.root, key.decoder_draw_id)]
        handle.seek(self._offsets[key])
        row = json.loads(handle.readline())
        found = (str(row["scenario_id"]), int(row["semantic_attempt_id"]), str(row["request_block_id"]),
                 str(row["component_id"]))
        if found != (key.scenario_id, key.semantic_attempt_id, key.request_block_id, key.rule_id) or (
            raw_output_sha256(row.get("raw_response")) != self._wanted[key]
        ):
            raise GrpoDataError(
                f"{key.source} draw {key.decoder_draw_id}'s raw outputs (of {key.root}) changed during the build: the "
                f"row read back for {key.scenario_id[:12]}/{key.rule_id} is not the one found, with the text the replay "
                "index names"
            )
        return row

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()

    def __enter__(self) -> "RawTexts":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def replay_groups(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Yield the replay index's rows one scenario at a time."""
    current: Optional[str] = None
    group: list[dict[str, Any]] = []
    done: set[str] = set()
    for entry in iter_jsonl_records(path):
        sid = str(entry["scenario_id"])
        if sid != current:
            if current is not None:
                yield current, group
                done.add(current)
            if sid in done:
                raise GrpoDataError(f"{path} does not keep scenario {sid[:12]}'s rows together")
            current, group = sid, []
        group.append(entry)
    if current is not None:
        yield current, group


def check_replay_group(
    scenario: TrainingScenario,
    group: Sequence[Mapping[str, Any]],
    rule_ids: Sequence[str],
    roots: Mapping[tuple[str, str], str],
) -> None:
    """Check a scenario's replay index rows (every rule of every draw once, at 1/K, each naming the root that judged it)
    and that their labels give the anchor row's p̂, q̂, mode and class."""
    anchor = scenario.anchor
    k = int(anchor["k"])
    source = row_source(anchor)
    cells = Counter((str(entry["rule_id"]), int(entry["decoder_draw_id"])) for entry in group)
    expected = {(rule, draw) for rule in rule_ids for draw in range(k)}
    where = f"scenario {scenario.scenario_id[:12]}"
    if set(cells) != expected or any(count != 1 for count in cells.values()):
        raise GrpoDataError(f"{where}: the replay index does not hold each of its {len(rule_ids)} rules x {k} draws once")
    if {entry["source"] for entry in group} != {source} or any(
        not math.isclose(float(entry["weight"]), 1 / k, rel_tol=0, abs_tol=1e-12) for entry in group
    ):
        raise GrpoDataError(f"{where}: its replay rows are not the {source} draws at weight 1/{k}")
    root = roots.get((source, scenario.scenario_id))
    named = {entry.get("root") for entry in group}
    if root is None or named != {root}:
        raise GrpoDataError(
            f"{where}: its replay rows name the root(s) {sorted(map(str, named))}, but the {source} root that judged "
            f"it is {root or 'none the anchor manifest records'}; a replayed completion is read from that root"
        )
    draws: list[dict[str, str]] = [{} for _ in range(k)]
    for entry in group:
        draws[int(entry["decoder_draw_id"])][str(entry["rule_id"])] = str(entry["label"])
    replayed = estimate_anchor(draws, rule_ids)
    for rule in rule_ids:
        mix = {label: float(share) for label, share in replayed.marginals[rule].items()}
        if mix != anchor["rule_marginals"][rule]:
            labels = Counter(draw[rule] for draw in draws)
            raise GrpoDataError(f"{where}, {rule}: the replayed labels {dict(labels)} are not its p-hat")
    columns = {
        **{f"q_{verdict}": float(replayed.q[verdict]) for verdict in TIE_ORDER},
        "mode": mode_of(replayed.q),
        "anchor_class": class_of(replayed.q, anchor["full_set_labels"]),
        "base_yes_rules": [rule for rule, shares in replayed.marginals.items() if shares["yes"] > 0],
    }
    differing = sorted(name for name, value in columns.items() if anchor[name] != value)
    if differing:
        raise GrpoDataError(
            f"{where}: its anchor row's {differing} are not the product form of the replayed labels; the reward's "
            "target and the replayed draws must be one estimate"
        )


@dataclass
class ReplayStats:
    """What the manifest records about the replay rows."""

    rows: int = 0
    pairs: int = 0
    weight_sum: float = 0.0
    opening_think: int = 0
    #: Completions holding a stop token's text (kept as text).
    stop_text: int = 0
    #: Completions holding another special token's text (read as the token).
    special_text: int = 0
    prompt_max: int = 0
    completion_sum: int = 0
    labels: Counter = field(default_factory=Counter)
    lengths: array = field(default_factory=lambda: array("q"))

    def add(
        self, *, prompt: int, completion: int, weight: float, label: str, opens_think: bool, stop_text: bool,
        special_text: bool,
    ) -> None:
        self.rows += 1
        self.weight_sum += weight
        self.labels[label] += 1
        self.opening_think += opens_think
        self.stop_text += stop_text
        self.special_text += special_text
        self.prompt_max = max(self.prompt_max, prompt)
        self.completion_sum += completion
        self.lengths.append(prompt + completion)

    def summary(self) -> dict[str, Any]:
        lengths = sorted(self.lengths)
        return {
            "rows": self.rows,
            "pairs": self.pairs,
            "weight_mean": self.weight_sum / self.rows,
            "labels": {label: self.labels.get(label, 0) for label in TIE_ORDER},
            "completions_opening_think": self.opening_think,
            "completions_quoting_stop_tokens": self.stop_text,
            "completions_with_other_special_tokens": self.special_text,
            "prompt_tokens_max": self.prompt_max,
            "completion_tokens_mean": self.completion_sum / self.rows,
            "row_tokens_mean": statistics.fmean(lengths),
            "row_tokens_p95": lengths[math.ceil(0.95 * len(lengths)) - 1],
            "row_tokens_max": lengths[-1],
        }


def write_replay_rows(
    path: Path,
    *,
    anchors: AnchorSets,
    kept: Mapping[str, TrainingScenario],
    rule_ids: Sequence[str],
    prompts: CampaignPrompts,
    tokenizer: Any,
    stop_ids: Sequence[int],
    enable_thinking: bool,
    raw: RawTexts,
    end_of_turn: int,
    think_open: str,
) -> ReplayStats:
    """Stream the replay rows of the kept scenarios to `path` in the index's order: each completion's ids closed by the
    model's end of turn, labels -100 over the prompt, weights scaled to a mean of 1."""
    mean_k = Fraction(sum(int(scenario.anchor["k"]) for scenario in kept.values()), len(kept))
    eos = int(end_of_turn)
    encoder = completion_encoder(tokenizer, stop_ids)
    stop_texts, other_texts = special_texts(tokenizer, stop_ids)
    stats = ReplayStats()
    with path.open("w", encoding="utf-8") as handle:
        for sid, group in anchors.replay_groups():
            scenario = kept.get(sid)
            if scenario is None:
                continue
            weight = float(mean_k / int(scenario.anchor["k"]))
            entries = {rule: prompts.entry(scenario.row, rule) for rule in rule_ids}
            digests = {rule: prompt_sha256(entry) for rule, entry in entries.items()}
            stale = sorted({entry["rule_id"] for entry in group if digests[entry["rule_id"]] != entry["prompt_sha256"]})
            if stale:
                raise GrpoDataError(
                    f"scenario {sid[:12]}: the per-rule prompts rebuilt here for {stale[:3]} are not the ones the base "
                    "answered (their ledger digests)"
                )
            prompt_ids = dict(zip(rule_ids, campaign_prompt_ids(
                tokenizer, [entries[rule] for rule in rule_ids], enable_thinking=enable_thinking
            )))
            texts = []
            for entry in group:
                row = raw.read(RawKey.of(entry))
                if row.get("finish_reason") != "stop":
                    raise GrpoDataError(
                        f"scenario {sid[:12]}, {entry['rule_id']}: its accepted output finished by "
                        f"{row.get('finish_reason')!r}, not by stopping; a replayed trace is a finished one"
                    )
                texts.append(row["raw_response"])
            for entry, text, ids in zip(group, texts, completion_ids(encoder, texts)):
                prompt = prompt_ids[entry["rule_id"]]
                completion = [*ids, eos]
                record = {
                    "input_ids": prompt + completion,
                    "labels": [IGNORE_INDEX] * len(prompt) + completion,
                    "weight": weight,
                    "scenario_id": sid,
                    "rule_id": entry["rule_id"],
                    "decoder_draw_id": entry["decoder_draw_id"],
                    "source": entry["source"],
                    "label": entry["label"],
                    "raw_output_sha256": entry["raw_output_sha256"],
                    "prompt_sha256": entry["prompt_sha256"],
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats.add(prompt=len(prompt), completion=len(completion), weight=weight, label=entry["label"],
                          opens_think=text.startswith(think_open), stop_text=any(s in text for s in stop_texts),
                          special_text=any(s in text for s in other_texts))
            stats.pairs += len(rule_ids)
    return stats


def replay_plan(
    anchors: AnchorSets,
    kept: Mapping[str, TrainingScenario],
    rule_ids: Sequence[str],
    roots: Mapping[tuple[str, str], str],
) -> dict[RawKey, str]:
    """Return the raw outputs the kept scenarios replay, each with the digest the index names, after one checking pass
    over the replay index."""
    wanted: dict[RawKey, str] = {}
    seen: set[str] = set()
    for sid, group in anchors.replay_groups():
        if sid not in kept:
            continue
        check_replay_group(kept[sid], group, rule_ids, roots)
        seen.add(sid)
        wanted.update((RawKey.of(entry), str(entry["raw_output_sha256"])) for entry in group)
    missing = sorted(set(kept) - seen)
    if missing:
        raise GrpoDataError(f"{len(missing)} training scenario(s) have no replay rows in the index (first {missing[0][:12]})")
    return wanted


def _refuse_non_local(condition: Any) -> None:
    if condition.model_type != "local" or condition.reasoning_effort is not None:
        raise GrpoDataError(
            f"the anchors were drawn by a {condition.model_type!r} judge (reasoning effort "
            f"{condition.reasoning_effort!r}); the GRPO data renders a local judge's prompts only"
        )


def build_grpo_data(
    *,
    anchors: Path | str | Sequence[Path | str],
    scenarios: Path | str,
    rules: Path | str,
    model: Path | str,
    out: Path | str,
    protected: Sequence[Path | str] = (),
    required_protected: Sequence[Path | str] = (),
    allow_no_held_out: bool = False,
    pinned_protected: Mapping[str, str] = NO_PINS,
    allow_no_pinned_protected: bool = False,
    prompt_token_cap: int = PROMPT_TOKEN_CAP,
    prompt_draws: int = PROMPT_DRAWS,
    repeat_cap: int = REPEAT_CAP,
    class_a_share_cap: Optional[float] = None,
    draw_seed: int = DRAW_SEED,
    markers: Optional[DisjointnessMarkers] = None,
    no_share: Optional[float] = None,
    no_margin: Optional[float] = None,
    task_spec_for: Optional[Callable[[Any, Path], Any]] = None,
) -> dict[str, Any]:
    """Build the GRPO rows, the replay rows and their markers under `out` and return the manifest."""
    out_dir = Path(out)
    partial = out_dir.with_name(out_dir.name + ".partial")
    for path in (out_dir, partial):
        if path.exists():
            raise GrpoDataError(f"{path} exists; GRPO data is written once, to a new directory")
    if prompt_token_cap < 1:
        raise GrpoDataError(f"prompt_token_cap must be positive, got {prompt_token_cap}")
    weigh = draw_rule(prompt_draws=prompt_draws, repeat_cap=repeat_cap, seed=draw_seed,
                      class_a_share_cap=class_a_share_cap, no_share=no_share, no_margin=no_margin)
    markers = markers if markers is not None else Block2Markers()
    anchor_set = AnchorSets.open(anchors)
    ruleset = Ruleset.load(rules)
    if ruleset.sha256 != anchor_set.rules_sha256:
        raise GrpoDataError(f"{ruleset.path} is not the ruleset the anchors judged")
    sources = {name: anchor_set.sources(name, evaluation_mode=EVALUATION_MODE_BANK)
               for name in anchor_set.replay_inputs()}
    full_set = anchor_set.sources("full_set", evaluation_mode=EVALUATION_MODE_FULL)
    # the replayed roots must be one condition; the first renders
    replayed = [source.condition for roots in sources.values() for source in roots.values()]
    if len({condition_key(condition) for condition in replayed}) != 1:
        fingerprints = sorted({condition.fingerprint for condition in replayed})
        raise GrpoDataError(
            f"the anchor set's replayed roots (its bank and top-up, or its stages) are not one condition (fingerprints "
            f"{fingerprints}); their replayed prompts are rendered under one"
        )
    if len({condition_key(source.condition) for source in full_set.values()}) != 1:
        full_fingerprints = sorted({source.condition.fingerprint for source in full_set.values()})
        raise GrpoDataError(
            f"the anchor sets' full-set roots are not one condition (fingerprints {full_fingerprints}); every "
            "scenario's class is read from the same full-set draws"
        )
    # every input of every set on one inference stack: the anchors' own guard over the merged inputs
    if len(anchor_set.sets) > 1:
        merged = {name: CampaignInput(name, tuple(roots.values()))
                  for name, roots in {**sources, "full_set": full_set}.items()}
        try:
            stack_check(merged)
        except AnchorError as error:
            raise GrpoDataError(f"the anchor sets' inputs are not on one inference stack: {error}") from error
    first_input = next(iter(sources.values()))
    first_root = next(iter(first_input.values()))
    condition = first_root.condition
    _refuse_non_local(condition)

    scenarios_path = Path(scenarios)
    required = {str(path): sha256_file(Path(path)) for path in required_protected}
    pinned = {} if allow_no_pinned_protected else dict(pinned_protected)
    must_cover = set(required.values()) | set(pinned.values())
    protected_files = [ProtectedFile.read(path) for path in protected]
    extra = {str(file.path): file.sha256 for file in protected_files}
    source_marker = markers.check(scenarios_path, sorted(set(required.values())))
    try:
        markers.check(scenarios_path, sorted(set(pinned.values())))
    except LeakageError as error:
        pins = ", ".join(f"{name} ({digest[:12]})" for name, digest in pinned.items())
        raise LeakageError(
            f"{error}. The training scenarios' marker must have been checked against the pinned protected files "
            f"({pins}), whichever files it names: re-attest the scenarios with each of them among the protected "
            "files, or declare a build whose marker was not (--allow-no-pinned-protected)"
        ) from error
    training = read_training_scenarios(scenarios_path, anchor_set)
    check_training_by_set(anchor_set, training)
    held_out = held_out_files(protected_files, anchor_set, training)
    if not held_out and not allow_no_held_out:
        named = "; ".join(
            f"{file.path}: {file.anchored(anchor_set)} of its {len(file.scenario_ids)} rows anchored"
            for file in protected_files
        )
        raise LeakageError(
            f"no protected file carves anchor rows out of {scenarios_path} ({named or 'none named'}): "
            "dev_nonyes (at least 30 clusters) comes out of this anchored pool, so dev_nonyes, every row one of this "
            "anchor set's and none a training scenario, must be named (--protected), and the training scenarios must "
            "be the pool less dev_nonyes and its dropped group mates. A build that holds nothing out must say so "
            "(--allow-no-held-out)"
        )
    model_dir = Path(model)
    tokenizer = load_tokenizer(model_dir, model_path=condition.model_path)
    trl_tokenizer, completion = load_training_format(model_dir)
    thinking = bool(condition.enable_thinking)
    task_spec = condition_task_spec(condition) if task_spec_for is None else task_spec_for(condition, first_root.root)
    prompts = CampaignPrompts.build(
        task_spec=task_spec, rules=load_jsonl_records(ruleset.path), judge_type=condition.judge_type
    )
    candidates = grpo_prompts(
        training, prompts=prompts, tokenizer=tokenizer, enable_thinking=thinking,
        full_set_digests=full_set_prompt_digests(*full_set.values()), trl_tokenizer=trl_tokenizer,
    )
    kept_prompts = [prompt for prompt in candidates if prompt.tokens <= prompt_token_cap]
    dropped = [prompt for prompt in candidates if prompt.tokens > prompt_token_cap]
    if not kept_prompts:
        raise GrpoDataError(f"every training prompt exceeds the {prompt_token_cap}-token cap")
    kept = {prompt.scenario.scenario_id: prompt.scenario for prompt in kept_prompts}
    weights = weigh({sid: scenario.anchor for sid, scenario in kept.items()})

    # every replayed completion found and checked before anything is written
    with RawTexts(sources, replay_plan(anchor_set, kept, ruleset.rule_ids, judging_roots(sources))) as raw:
        partial.mkdir(parents=True)
        try:
            # the scenarios attested first: a leak refuses before the long write
            scenario_file = partial / SCENARIOS_FNAME
            scenario_file.write_text("".join(scenario.line + "\n" for scenario in kept.values()), encoding="utf-8")
            scenario_marker = markers.attest(scenario_file, source_marker, [file.path for file in protected_files])
            marker_paths = {SCENARIOS_FNAME: markers.write(scenario_file, scenario_marker)}
            markers.check(scenario_file, sorted(must_cover), sorted(set(extra.values())))
            rows = grpo_rows(kept_prompts, weights)
            write_jsonl_records(partial / GRPO_ROWS_FNAME, rows)
            replay = write_replay_rows(
                partial / REPLAY_ROWS_FNAME, anchors=anchor_set, kept=kept, rule_ids=ruleset.rule_ids,
                prompts=prompts, tokenizer=tokenizer, stop_ids=stop_token_ids(model_dir), enable_thinking=thinking,
                raw=raw, end_of_turn=trl_tokenizer.eos_token_id,
                think_open=THINKING_FAMILIES[completion.family].raw_think_open,
            )
            outputs = {
                name: {"sha256": sha256_file(partial / name), "rows": count}
                for name, count in ((SCENARIOS_FNAME, len(kept)), (GRPO_ROWS_FNAME, len(rows)),
                                    (REPLAY_ROWS_FNAME, replay.rows))
            }
            scenario_marker_sha256 = sha256_file(marker_paths[SCENARIOS_FNAME])
            for name in (GRPO_ROWS_FNAME, REPLAY_ROWS_FNAME):
                marker_paths[name] = markers.write(partial / name, derived_marker(
                    scenario_marker, name=name, sha256=outputs[name]["sha256"], rows=outputs[name]["rows"],
                    scenarios_marker_sha256=scenario_marker_sha256,
                ))
            campaigns = {
                **{name: campaign_records(roots, raw=True) for name, roots in sources.items()},
                "full_set": campaign_records(full_set, raw=False),
            }
            several = any(len(roots) > 1 for roots in (*sources.values(), full_set))
            staged = anchor_set.staged
            if len(anchor_set.sets) > 1:
                schema_version = GRPO_DATA_SCHEMA_VERSION_SETS
            elif staged:
                schema_version = GRPO_DATA_SCHEMA_VERSION_STAGED
            else:
                schema_version = GRPO_DATA_SCHEMA_VERSION_ROOTS if several else GRPO_DATA_SCHEMA_VERSION
            manifest = {
                "schema_version": schema_version,
                "anchors": anchor_set.record(),
                "scenarios": {
                    "path": str(scenarios_path),
                    "sha256": sha256_file(scenarios_path),
                    "rows": len(training),
                    "marker_sha256": sha256_file(disjoint_marker_path(scenarios_path)),
                },
                "rules": ruleset.record(),
                "campaigns": campaigns,
                "anchor_condition": str(first_root.root / condition.condition_id / CONDITION_FNAME),
                "tokenizer": tokenizer_record(model_dir, tokenizer, completion),
                # the replay rows are this base's outputs
                "base_weights_sha256": base_weights_sha256(model_dir),
                "rendering": {
                    **({"task_spec": TASK_SPEC} if task_spec_for is None else task_spec.rendering()),
                    "prompt_style": condition.prompt_style,
                    "judge_type": condition.judge_type,
                    "individual_rule_context": condition.individual_rule_context,
                    "conversation": "[system: P, user: '']",
                    "chat_template_kwargs": chat_template_kwargs(thinking),
                },
                "protected_extra": [{"path": path, "sha256": digest} for path, digest in extra.items()],
                "required_protected": [{"path": path, "sha256": digest} for path, digest in required.items()],
                "pinned_protected": [{"name": name, "sha256": digest} for name, digest in pinned.items()],
                "allow_no_pinned_protected": allow_no_pinned_protected,
                "held_out": [file.record() for file in held_out],
                "allow_no_held_out": allow_no_held_out,
                "prompt_token_cap": prompt_token_cap,
                "counts": {
                    "anchor_rows": len(anchor_set.rows),
                    "anchor_rows_not_training": len(anchor_set.rows) - len(training),
                    "held_out_scenarios": len({sid for file in held_out for sid in file.scenario_ids}),
                    "training_scenarios": len(training),
                    "dropped_by_prompt_cap": len(dropped),
                    "kept": len(kept),
                    "kept_by_source": dict(sorted(Counter(str(s.row.get("source")) for s in kept.values()).items())),
                    "prompt_tokens_max": max(prompt.tokens for prompt in kept_prompts),
                },
                "dropped_by_prompt_cap": [
                    {"scenario_id": prompt.scenario.scenario_id, "prompt_tokens": prompt.tokens,
                     "anchor_class": prompt.scenario.anchor["anchor_class"],
                     **(dict(zip(("favoured_verdict", "margin"), favoured_verdict(prompt.scenario.anchor)))
                        if no_share is not None else {})}
                    for prompt in dropped
                ],
                "sampling_weights": weights.declaration,
                **({"q_hat_stages": q_hat_stages(kept, weights)} if staged else {}),
                **({"by_anchor_set": by_anchor_set(anchor_set, kept, weights)} if len(anchor_set.sets) > 1 else {}),
                "replay": replay.summary(),
                "outputs": outputs,
                "markers": {name: sha256_file(path) for name, path in marker_paths.items()},
                **code_state(),
                "created_at": utc_now(),
            }
            write_json_sidecar(partial / MANIFEST_FNAME, manifest)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
    partial.rename(out_dir)
    return manifest


def check_training_by_set(anchor_set: AnchorSets, training: Sequence[TrainingScenario]) -> None:
    """Refuse, of several anchor sets, one anchoring no training scenario, or a mixed-selection set anchoring a row whose
    q̂ favours another judgment than yes."""
    trained = {scenario.scenario_id for scenario in training}
    idle = [str(each.directory) for each in anchor_set.sets if trained.isdisjoint(each.rows)]
    if len(anchor_set.sets) > 1 and idle:
        raise GrpoDataError(f"the anchor set(s) {', '.join(idle)} anchor no training scenario: a set read beside "
                            "another gives the union's rows (the training scenarios are the sets' rows, concatenated)")
    for each in anchor_set.sets:
        if (each.manifest.get("selection") or {}).get("kind") != MIXED_SELECTION_KIND:
            continue
        others = Counter(favoured_verdict(scenario.anchor)[0] for scenario in training
                         if scenario.scenario_id in each.rows)
        others.pop("yes", None)
        if others:
            raise GrpoDataError(
                f"{sum(others.values())} training scenario(s) anchored by {each.directory}, a set built on a mixed "
                f"selection, favour another judgment than yes ({dict(sorted(others.items()))}): such a set gives its "
                "yes-favoured rows alone, its full-set labels being the screen draws that selected them"
            )


def _strata(kept: Mapping[str, TrainingScenario], weights: SamplingWeights) -> Optional[dict[str, str]]:
    if weights.declaration.get("stratification") != "favoured_verdict":
        return None
    no_margin = weights.declaration["no_margin"]
    return {sid: verdict_stratum(*favoured_verdict(scenario.anchor), no_margin) for sid, scenario in kept.items()}


def _by_stratum(sids: Sequence[str], stratum_of: Mapping[str, str], weights: SamplingWeights) -> dict[str, Any]:
    return {stratum: {"rows": sum(1 for sid in sids if stratum_of[sid] == stratum),
                      "draws": sum(weights.copies[sid] for sid in sids if stratum_of[sid] == stratum)}
            for stratum in VERDICT_STRATA}


def by_anchor_set(anchor_set: AnchorSets, kept: Mapping[str, TrainingScenario],
                  weights: SamplingWeights) -> list[dict[str, Any]]:
    """Return what a build over several anchor sets records of each: its path, manifest digest, rows and draws (by
    judgment stratum under the mix)."""
    stratum_of = _strata(kept, weights)
    records = []
    for each in anchor_set.sets:
        mine = [sid for sid in kept if sid in each.rows]
        record = {"path": str(each.directory), "manifest_sha256": each.record()["manifest_sha256"],
                  "anchor_rows": len(each.rows), "kept": len(mine), "draws": sum(weights.copies[sid] for sid in mine)}
        if stratum_of is not None:
            record["strata"] = _by_stratum(mine, stratum_of, weights)
        records.append(record)
    return records


def q_hat_stages(kept: Mapping[str, TrainingScenario], weights: SamplingWeights) -> dict[str, Any]:
    """Return what a staged build's manifest records of each kept row's stage and the rows and draws per stage."""
    stage_of = {sid: row_source(scenario.anchor) for sid, scenario in kept.items()}
    names = list(dict.fromkeys(stage_of.values()))
    record: dict[str, Any] = {
        "by_scenario": stage_of,
        "rows": {name: sum(1 for stage in stage_of.values() if stage == name) for name in names},
        "k": {name: sorted({int(kept[sid].anchor["k"]) for sid, stage in stage_of.items() if stage == name})
              for name in names},
        "draws": {name: sum(weights.copies[sid] for sid, stage in stage_of.items() if stage == name) for name in names},
    }
    stratum_of = _strata(kept, weights)
    if stratum_of is not None:
        record["strata"] = {name: _by_stratum([sid for sid in stage_of if stage_of[sid] == name], stratum_of, weights)
                            for name in names}
    return record


def _build_parser(pins: Mapping[str, str] = NO_PINS) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_grpo_data",
        description="Build GRPO training data: the GRPO rows, the KL anchor's replay rows, their markers and a manifest.")
    parser.add_argument(
        "--anchors", type=Path, action="append", required=True,
        help="a train-side anchor set (python -m ruleforge.grpo.anchor_set build); repeatable: the sets must be one "
             "estimate (same ruleset, estimator and reference condition) over disjoint scenarios on one inference "
             "stack, and are read as one, the first set's first replayed root being the anchor condition",
    )
    parser.add_argument(
        "--scenarios", type=Path, required=True,
        help="the training scenarios (JSONL, each anchored): the anchored pool less the held-out dev rows whose "
             "per-rule judgment is not yes and their dropped group mates, not the pool file, with its disjointness "
             "marker for this file's bytes beside it",
    )
    parser.add_argument("--rules", type=Path, required=True, help="the ruleset the anchors judged")
    parser.add_argument("--model", type=Path, required=True, help="the base checkpoint directory (its tokenizer)")
    parser.add_argument("--out", type=Path, required=True, help="the directory to create")
    parser.add_argument(
        "--protected", type=Path, action="append", default=[],
        help="a protected file, such as the held-out dev rows whose per-rule judgment is not yes, that the second "
             "pass adds to those the scenarios' marker names; repeatable; "
             "unless --allow-no-held-out, at least one must be a carve-out of the anchor set: every row an anchor "
             "row, none a training scenario",
    )
    parser.add_argument(
        "--require-protected", type=Path, action="append", default=[],
        help="a protected file (dev) the scenarios' marker must already cover, beyond the pinned ones; repeatable",
    )
    parser.add_argument(
        "--allow-no-pinned-protected", action="store_true",
        help=f"declare a build whose scenarios' marker was not checked against the pinned protected files "
             f"({', '.join(pins) or 'none pinned'}), which are otherwise required by digest (recorded in the manifest)",
    )
    parser.add_argument(
        "--allow-no-held-out", action="store_true",
        help="declare a build that names no carve-out of the anchor set as --protected (recorded in the manifest)",
    )
    parser.add_argument("--prompt-token-cap", type=int, default=PROMPT_TOKEN_CAP, help="the longest prompt, in tokens (default %(default)s)")
    parser.add_argument(
        "--prompt-draws", type=int, default=PROMPT_DRAWS,
        help="the run's prompt draws, max_steps x prompts per step (default %(default)s)",
    )
    parser.add_argument(
        "--repeat-cap", type=int, default=REPEAT_CAP,
        help="the most draws of one scenario in anchor class a (per-rule judgment not mostly yes) or in the judgment "
             "mix's 'no' stratum (default %(default)s)",
    )
    parser.add_argument(
        "--class-a-share-cap", type=float, default=None,
        help="the most share of the draws anchor class a (per-rule judgment not mostly yes) takes under the "
             f"anchor-class rule (default {CLASS_A_SHARE_CAP})",
    )
    parser.add_argument(
        "--no-share", type=float, default=None,
        help="the judgment mix: the share of the draws, at least, on scenarios whose q-hat favours "
             "'no' by --no-margin, at most --repeat-cap draws each (refused when it cannot); in place of the "
             "anchor-class rule",
    )
    parser.add_argument(
        "--no-margin", type=float, default=None,
        help="with --no-share: the least margin of q-hat(no) over the next largest judgment's, in (0, 1] (a "
             "refused build lists the rows each margin gives)",
    )
    parser.add_argument(
        "--draw-seed", type=int, default=DRAW_SEED,
        help="the seed of the draws' selection and order (default %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, pins: Mapping[str, str] = NO_PINS) -> int:
    """Run the build's command line; `pins` are the protected files the scenarios' marker must cover."""
    args = _build_parser(pins).parse_args(argv)
    manifest = build_grpo_data(
        anchors=args.anchors, scenarios=args.scenarios, rules=args.rules, model=args.model, out=args.out,
        protected=args.protected, required_protected=args.require_protected, allow_no_held_out=args.allow_no_held_out,
        pinned_protected=pins, allow_no_pinned_protected=args.allow_no_pinned_protected,
        prompt_token_cap=args.prompt_token_cap,
        prompt_draws=args.prompt_draws, repeat_cap=args.repeat_cap, class_a_share_cap=args.class_a_share_cap,
        draw_seed=args.draw_seed, no_share=args.no_share, no_margin=args.no_margin,
    )
    summary = {
        "out": str(args.out),
        **manifest["counts"],
        "grpo_rows": manifest["outputs"][GRPO_ROWS_FNAME]["rows"],
        "replay_rows": manifest["outputs"][REPLAY_ROWS_FNAME]["rows"],
        # the anchor micro-step's peak memory follows the longest replay row
        "replay_row_tokens_p95": manifest["replay"]["row_tokens_p95"],
        "replay_row_tokens_max": manifest["replay"]["row_tokens_max"],
        "draws": manifest["sampling_weights"]["draws"],
        "realized_shares": manifest["sampling_weights"]["realized_shares"],
        "scenarios_not_drawn": manifest["sampling_weights"]["scenarios_not_drawn"],
        "prompt_draws": manifest["sampling_weights"]["prompt_draws"],
        "trainer": {
            "data.train": str(args.out / GRPO_ROWS_FNAME),
            "data.replay": str(args.out / REPLAY_ROWS_FNAME),
            "data.anchor_condition": manifest["anchor_condition"],
            "data.prompt_token_cap": manifest["prompt_token_cap"],
            "grpo.chat_template_kwargs": manifest["rendering"]["chat_template_kwargs"],
            "grpo.shuffle_dataset": True,
        },
    }
    print(json.dumps(summary, indent=1))
    return 0
