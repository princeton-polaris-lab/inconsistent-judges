from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

ROOTS = (
    "fig2-2026-09-luna-pop-a22af097-rescored-lenient_last_line",
    "fig2-2026-09-pop-a22af097-rescored-lenient_last_line",
    "fig2-2026-10-glimmer-pop-a22af097",
    "fig2-2026-10-haiku-sum-rescored-lenient_last_line",
    "fig4-2026-09-pop-a22af097-rescored-lenient_last_line",
    "fig5-gemma-2026-10-07-pop-a22af097",
    "fig5-gemma-ckpt600-2026-10-09-pop-a22af097",
    "fig5-grpo-v2-fp32-2026-09-26-checkpoint-200-pop-a22af097",
    "fig5-qwen-mix-2026-10-11-checkpoint-300-pop-a22af097",
    "final-2026-09-luna-pop-a22af097-rescored-lenient_last_line",
    "final-2026-09-pop-a22af097-rescored-lenient_last_line",
    "hierarchy-aware-2026-10-glimmer",
    "hierarchy-aware-2026-10-haiku-sum",
    "hierarchy-aware-2026-10-luna-sum",
    "hierarchy-aware-2026-10-pop-a22af097",
    "k57-shuffled-2026-11",
    "k57-shuffled-2026-11-glimmer",
    "k57-shuffled-2026-11-haiku",
    "k57-shuffled-2026-11-luna",
    "prompt-ablation-v2-2026-10-glimmer-pop-a22af097",
    "prompt-ablation-v2-2026-10-haiku-sum",
    "prompt-ablation-v2-2026-10-luna-pop-a22af097-rescored-lenient_last_line",
    "prompt-ablation-v2-2026-10-pop-a22af097-rescored-lenient_last_line",
    "prompt-ablation-v2-open-2026-10-pop-a22af097-rescored-lenient_last_line",
)
#: The roots whose size 57 judged the full ruleset in an order of each subset seed's own.
SHUFFLED_FULL_ROOTS = ("k57-shuffled-2026-11", "k57-shuffled-2026-11-glimmer", "k57-shuffled-2026-11-haiku",
                       "k57-shuffled-2026-11-luna")
#: The roots whose condition files name each prompt style by its public name; the other roots hold its recorded key.
PUBLIC_STYLE_ROOTS = SHUFFLED_FULL_ROOTS
RESCORED_FROM = {
    f"{original}-rescored-lenient_last_line": original for original in (
        "fig2-2026-09-luna-pop-a22af097",
        "fig2-2026-09-pop-a22af097",
        "fig2-2026-10-haiku-sum",
        "fig4-2026-09-pop-a22af097",
    )
}
EXTERNAL_ROOTS = {
    "final-2026-09": "the open judges' published arms as they ran: today's builder reproduces the prompt digests "
                     "they recorded (tests/test_frozen_ablation_prompts.py), the revised arms' specs compose to their "
                     "conditions but for the style (tests/test_launch_specs.py), and its Hydra multirun record holds "
                     "the overrides prompt_ablation_naive renders (tests/test_launch_campaign.py)",
    "final-2026-09-luna": "GPT-5.6 Luna's published arms as they ran: the paper-prompt arms' specs compose to their recorded "
                          "runs and fingerprints but for the style and the label rule (tests/test_launch_specs.py), "
                          "and the pricing tool refuses to bill it without its retired style's arm "
                          "(tests/test_sampling_batch_cost.py)",
    "final-2026-09-pop-a22af097": "the published campaign rebased onto the paper's population, before its re-score: prompt_ablation_naive's "
                                  "conditions preflight against it as they ran (tests/test_launch_campaign.py)",
    "final-2026-09-pop-6f528dcb": "the open judges' GRPO-prompt arms rebased onto the previous population, before their re-score: "
                                  "their condition files keep their fingerprints (tests/test_sampling_training.py)",
    "final-2026-09-luna-pop-6f528dcb": "GPT-5.6 Luna's GRPO-prompt arms rebased onto the previous population, before their re-score: "
                                       "their condition files keep their fingerprints (tests/test_sampling_training.py)",
    "fig2-2026-09-luna": "GPT-5.6 Luna's ruleset-size root as it ran: the pricing plan's two ruleset-size designs are its "
                         "two request shapes, its bank's and size-57 subsets' prompts the prompt ablation's naive ones "
                         "(tests/test_sampling_batch_cost.py)",
    **dict.fromkeys(("final-2026-09-luna-pop-6f528dcb-rescored-lenient_last_line",
                     "prompt-ablation-v2-2026-10-luna-pop-6f528dcb-rescored-lenient_last_line",
                     "fig2-2026-09-luna-pop-6f528dcb"), (
        "GPT-5.6 Luna's three roots on the previous population as the release kept them: the committed pricing plan prices every design "
        "from them (tests/test_sampling_batch_cost.py); the ruleset-size one also holds the run records GPT-5.6 Luna's frozen "
        "judge descriptor was written from (tests/identity/judge_descriptors.py)")),
    **dict.fromkeys(("fig2-2026-09-pop-6f528dcb", "fig4-2026-09-pop-6f528dcb",
                     "fig5-gemma-2026-10-07-pop-6f528dcb", "fig5-gemma-ckpt600-2026-10-09-pop-6f528dcb",
                     "fig5-grpo-v2-fp32-2026-09-26-checkpoint-200-pop-6f528dcb",
                     "fig5-qwen-mix-2026-10-11-checkpoint-300-pop-6f528dcb",
                     "prompt-ablation-v2-2026-10-haiku-pop-6f528dcb"), (
        "a root on the previous population whose run records a frozen judge descriptor was written from (tests/identity/fixtures/"
        "judges.json, each descriptor's source): the descriptors' check reads them again "
        "(tests/identity/judge_descriptors.py)")),
    "fig4-2026-09": "the RuleForge campaign as it ran: its Hydra multirun record holds the overrides ruleforge_rulesets "
                    "renders against its rebase onto the paper's population (tests/test_launch_campaign.py)",
    "fig5-gemma-2026-10-07": "Gemma's training root as it ran: its estimates file predates the recorded retry scope, "
                             "which the read paths take from its sampling manifest "
                             "(tests/evaluate/test_campaign_ops.py)",
    "hierarchy-aware-2026-10-qwen3-8b": "Qwen3-8B's hierarchy arms as they ran, left out of the paper and the Hub: "
                                        "its launch recipe composes to their conditions (tests/test_launch_specs.py)",
    "pilot-sol-v2-2026-10": "the GPT-6.1 Sol pilot of the revised prompts: the committed pricing plan's revised designs are "
                            "priced from it (tests/test_sampling_batch_cost.py)",
    **dict.fromkeys(RESCORED_FROM.values(), (
        "a published root as recorded, under lenient, before its one-label-rule re-score: the copy the dataset "
        "ships was replayed from it, the draws the replay changed recorded in that copy's label_replay.json, the "
        "judge descriptors read its run records (tests/identity/judge_descriptors.py); no golden reads it, each "
        "reading the copy (tests/identity/test_shipped_roots.py)")),
}
VALIDATION_ROOTS = ROOTS + tuple(EXTERNAL_ROOTS)
EXTERNAL_INPUTS = {
    "batch": "the Batch judges' work directories: the request lines they sent and the usage they billed (the "
             "identity fixtures' batch table, tests/test_sampling_batch_cost.py)",
    "populations": "the frozen populations and their derivations (tests/test_sampling_subset.py, "
                   "tests/test_frozen_ablation_prompts.py)",
    "selection": "the GRPO selection runs' records and merged checkpoints' manifests (the training goldens, "
                 "tests/test_launch_specs.py)",
    "hf_cache": "the Hugging Face splits the populations were drawn from, as cached",
    "sampling/subsets": "the population cuts the pilots judged, with their sidecars (tests/test_sampling_subset.py, "
                        "tests/test_sampling_batch_cost.py)",
}

CONDITIONS = "conditions.jsonl.gz"
SEEDS = "seeds.jsonl.gz"
PROMPTS = "prompts.jsonl.gz"
SETTLEMENT = "settlement.jsonl.gz"
BATCH = "batch.jsonl.gz"
FIXTURE_NAMES = (CONDITIONS, SEEDS, PROMPTS, SETTLEMENT, BATCH)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture_text(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """The JSON Lines bytes of ``rows`` before compression: sorted keys, compact separators, UTF-8."""
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")


def fixture_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """The gzipped JSON Lines bytes of ``rows``, no timestamp or file name, so equal rows give equal bytes."""
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as handle:
        handle.write(fixture_text(rows))
    return buffer.getvalue()


def committed_text(name: str) -> bytes:
    """A committed fixture's JSON Lines bytes, decompressed."""
    return gzip.decompress((FIXTURES / name).read_bytes())


def read_fixture(name: str) -> list[dict[str, Any]]:
    """Every row of a committed fixture."""
    with gzip.open(FIXTURES / name, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def rows_of(rows: Iterable[Mapping[str, Any]], kind: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["kind"] == kind]


def repo_rule_files() -> dict[tuple[str, str], Path]:
    """Every ruleset file the repository commits, by ``(basename, sha256)``: how a condition names its ruleset."""
    found: dict[tuple[str, str], Path] = {}
    for path in sorted((REPO_ROOT / "data" / "rules").rglob("*.jsonl")):
        found[(path.name, sha256_bytes(path.read_bytes()))] = path
    return found


def numbered_lines(path: Path) -> Iterator[tuple[int, bytes]]:
    """``(line number, line bytes)`` of a JSON Lines file, the newline kept, numbered from 0."""
    with Path(path).open("rb") as handle:
        yield from enumerate(handle)

#: The eleven goldens under ``results/`` the published roots rebuild.
GOLDENS = (
    "ruleset_size/curves.json", "ruleset_size/repeated_draw_baseline.json", "ruleforge/ruleforge.json",
    "training/gemma.json", "training/qwen.json", "training/judgment_changes.json", "training/frozen_anchors.json",
    "hierarchy/hierarchy.json", "hierarchy/ladder.json",
    "leniency/leniency.json", "prompt_ablation/prompt_ablation.json",
)

GLIMMER_ROOTS = {"prompt-ablation-v2-2026-10-glimmer-pop-a22af097": 3, "fig2-2026-10-glimmer-pop-a22af097": 24,
                 "hierarchy-aware-2026-10-glimmer": 1}
GLIMMER_SHARDS = tuple((root, index) for root, shards in GLIMMER_ROOTS.items() for index in range(shards))
GLIMMER_SHARD_IDS = tuple(f"{root}:{index}" for root, index in GLIMMER_SHARDS)


def shard_draws(base: Path, index: int, shards: int) -> Iterator[tuple[Any, Path]]:
    """``(condition, draw directory)`` for every draw of every ``shards``-th condition of ``base``."""
    from ruleforge.evaluate.campaign import load_conditions

    conditions = load_conditions(base)
    taken = sorted(conditions)[index::shards]
    assert taken, f"{base.name}: shard {index} of {shards} holds no condition"
    for condition_id in taken:
        draws = sorted(path for path in (base / condition_id).glob("draw-*") if path.is_dir())
        assert draws, f"{base.name}/{condition_id}: no draw"
        for draw in draws:
            yield conditions[condition_id], draw


def ledger_rows(path: Path) -> Iterator[dict[str, Any]]:
    """The rows of a JSON Lines ledger (a draw's ``attempts.jsonl`` or ``raw_outputs.jsonl``), parsed."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)
