from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ruleforge.core.errors import RuleForgeError
from ruleforge.core.artifacts import load_jsonl_records
from ruleforge.core.digests import canonical_digest, sha256_file
from ruleforge.evaluate.bootstrap import bootstrap_ci, paired_bootstrap_diff_ci
from ruleforge.grpo.run_files import checkpoint_name, checkpoint_steps

LABELS = ("no", "unsure", "yes")
#: Label codes: the index in `LABELS`, so phi = code / 2.
CODE = {label: code for code, label in enumerate(LABELS)}
YES = CODE["yes"]
MISSING = -1
DEV, DEV_NONYES = "dev", "dev_nonyes"
SUBSETS = (DEV, DEV_NONYES)
#: The dev campaign stages a statistic reads; each stage's null draws from a stream named after it.
STAGE1, PROBE = "stage1", "probe"
HIGHER, LOWER, TWO_SIDED = "higher", "lower", "two_sided"
GATES = ("G1", "G2", "G3", "G4", "G5p", "G6", "G7")
STAGE1_FNAME, PROBE_FNAME = "stage1.jsonl", "probe.jsonl"
SCENARIOS_FNAME, BANK_FNAME, FULL_FNAME = "scenarios.jsonl", "bank.jsonl", "full.jsonl"
TOLERANCES_FNAME, REPORT_FNAME, TABLE_FNAME = "tolerances.json", "report.json", "report.md"
TOLERANCES_KIND, REPORT_KIND = "ruleforge-grpo-tolerances", "ruleforge-grpo-report"


class ReportError(ValueError, RuleForgeError):
    """Inputs the report cannot be computed from."""


@dataclass(frozen=True)
class Scenarios:
    """The scenarios the base and every save are judged on, in one fixed order."""

    ids: tuple[str, ...]
    sources: tuple[str, ...]
    subsets: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.ids or not len(self.ids) == len(self.sources) == len(self.subsets):
            raise ReportError("a scenario table needs one source and one subset per scenario, and a scenario")
        if len(set(self.ids)) != len(self.ids):
            raise ReportError("a scenario id appears twice in the scenario table")
        unknown = sorted(set(self.subsets) - set(SUBSETS))
        if unknown:
            raise ReportError(f"subsets {unknown} are not among {SUBSETS}")

    def rows(self, subsets: Iterable[str]) -> np.ndarray:
        """Return the row indices of the scenarios in `subsets`, in table order."""
        wanted = set(subsets)
        return np.array([i for i, subset in enumerate(self.subsets) if subset in wanted], dtype=int)

    def rows_of(self, ids: Sequence[str]) -> np.ndarray:
        index = {sid: i for i, sid in enumerate(self.ids)}
        missing = [sid for sid in ids if sid not in index]
        if missing:
            raise ReportError(f"scenario(s) {missing[:3]} are not in the scenario table")
        return np.array([index[sid] for sid in ids], dtype=int)


def read_scenarios(path: Path | str) -> Scenarios:
    """Read a scenario table: JSONL rows with `scenario_id`, `source` and `subset`."""
    rows = load_jsonl_records(path)
    return Scenarios(
        tuple(str(row["scenario_id"]) for row in rows),
        tuple(str(row["source"]) for row in rows),
        tuple(str(row["subset"]) for row in rows),
    )


def _take(array: np.ndarray, rows: np.ndarray | None, draws: Sequence[int] | None) -> np.ndarray:
    if rows is not None:
        array = array[rows]
    if draws is not None:
        array = array[:, list(draws)]
    return array


@dataclass(frozen=True)
class FullSet:
    """Full-set responses, scenarios x draws; `map_labels` holds each accepted response's parsed map."""

    verdict: np.ndarray  # [N, D] label code, MISSING without an accepted response
    attempts: np.ndarray  # [N, D]
    truncated: np.ndarray  # [N, D] truncated attempts
    complete: np.ndarray  # [N, D] bool: the map names every rule once
    map_labels: np.ndarray  # [N, D, R] label code, MISSING where the map lacks the rule
    think_tokens: np.ndarray  # [N, D], MISSING where not accepted
    empty_think: np.ndarray  # [N, D] bool

    def take(self, rows: np.ndarray | None = None, draws: Sequence[int] | None = None) -> "FullSet":
        """Return these scenario rows and draw columns."""
        return FullSet(*(_take(getattr(self, name), rows, draws) for name in FullSet.__dataclass_fields__))

    @property
    def accepted(self) -> np.ndarray:
        return self.verdict != MISSING


@dataclass(frozen=True)
class RuleDraws:
    """Isolated per-rule labels, scenarios x draws x rules; MISSING where exhausted."""

    label: np.ndarray

    def take(self, rows: np.ndarray | None = None, draws: Sequence[int] | None = None) -> "RuleDraws":
        return RuleDraws(_take(self.label, rows, draws))

    @property
    def present(self) -> np.ndarray:
        return self.label != MISSING


def _label(value: Any, where: str) -> int:
    if value is None:
        return MISSING
    if value not in CODE:
        raise ReportError(f"{where}: label {value!r} is not one of {LABELS}")
    return CODE[value]


def _count(value: Any, where: str, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ReportError(f"{where}: {name} must be a non-negative integer, got {value!r}")
    return value


def read_responses(
    path: Path | str,
    scenarios: Scenarios,
    rows: np.ndarray,
    rule_ids: Sequence[str],
    draws: int,
    *,
    full: bool,
    rules: bool,
) -> tuple[FullSet | None, RuleDraws | None]:
    """Read a response table over `scenarios.ids[rows]` x draws `0..draws-1`, exactly one row for each; a repeated,
    missing or malformed row is refused."""
    n, n_rules = len(rows), len(rule_ids)
    position = {scenarios.ids[row]: i for i, row in enumerate(rows)}
    rule_index = {rule: r for r, rule in enumerate(rule_ids)}
    seen = np.zeros((n, draws), dtype=bool)
    fs = {
        "verdict": np.full((n, draws), MISSING, dtype=np.int8), "attempts": np.zeros((n, draws), dtype=np.int64),
        "truncated": np.zeros((n, draws), dtype=np.int64), "complete": np.zeros((n, draws), dtype=bool),
        "map_labels": np.full((n, draws, n_rules), MISSING, dtype=np.int8),
        "think_tokens": np.full((n, draws), MISSING, dtype=np.int64), "empty_think": np.zeros((n, draws), dtype=bool),
    }
    labels = np.full((n, draws, n_rules), MISSING, dtype=np.int8)
    for row in load_jsonl_records(path):
        sid, draw = str(row.get("scenario_id")), row.get("draw")
        where = f"{path}: scenario {sid!r} draw {draw!r}"
        if sid not in position or isinstance(draw, bool) or not isinstance(draw, int) or not 0 <= draw < draws:
            raise ReportError(f"{where} is not one of the table's {n} scenarios x {draws} draws")
        i = position[sid]
        if seen[i, draw]:
            raise ReportError(f"{where} appears twice")
        seen[i, draw] = True
        if full:
            _read_full(row.get("full"), where, fs, i, draw, rule_index)
        if rules:
            _read_rules(row.get("rules"), where, labels, i, draw, rule_index)
    if not seen.all():
        i, draw = map(int, np.argwhere(~seen)[0])
        raise ReportError(f"{path} has no row for scenario {scenarios.ids[rows[i]]!r} draw {draw}")
    return (FullSet(**fs) if full else None), (RuleDraws(labels) if rules else None)


def _read_full(obj: Any, where: str, fs: dict[str, np.ndarray], i: int, d: int, rule_index: Mapping[str, int]) -> None:
    if not isinstance(obj, Mapping):
        raise ReportError(f"{where} has no full-set object")
    verdict = _label(obj.get("verdict"), where)
    attempts = _count(obj.get("attempts"), where, "attempts")
    truncated = _count(obj.get("truncated_attempts"), where, "truncated_attempts")
    if truncated > attempts - (verdict != MISSING) or (verdict != MISSING and attempts < 1):
        raise ReportError(f"{where}: {truncated} truncated of {attempts} attempts for judgment {obj.get('verdict')!r}")
    fs["verdict"][i, d], fs["attempts"][i, d], fs["truncated"][i, d] = verdict, attempts, truncated
    if verdict == MISSING:
        return
    labels = obj.get("map") or {}
    unknown = sorted(set(labels) - set(rule_index))
    complete = bool(obj.get("map_complete"))
    if unknown or (complete and len(labels) != len(rule_index)):
        raise ReportError(f"{where}: the map names rules outside the ruleset {unknown[:3]} or is called "
                          "complete without every rule")
    for rule, label in labels.items():
        fs["map_labels"][i, d, rule_index[rule]] = _label(label, where)
    fs["complete"][i, d] = complete
    fs["think_tokens"][i, d] = _count(obj.get("think_tokens"), where, "think_tokens")
    fs["empty_think"][i, d] = bool(obj.get("empty_think"))


def _read_rules(obj: Any, where: str, labels: np.ndarray, i: int, d: int, rule_index: Mapping[str, int]) -> None:
    if not isinstance(obj, Mapping) or set(obj) != set(rule_index):
        raise ReportError(f"{where}: the rules object must name every rule of the ruleset exactly")
    for rule, entry in obj.items():
        label = _label(entry.get("label"), where)
        attempts = _count(entry.get("attempts"), where, "attempts")
        truncated = _count(entry.get("truncated_attempts"), where, "truncated_attempts")
        if truncated > attempts - (label != MISSING) or (label != MISSING and attempts < 1):
            raise ReportError(f"{where} {rule}: {truncated} truncated of {attempts} attempts")
        labels[i, d, rule_index[rule]] = label


#: A response table's identity sidecar: `stage1.jsonl` -> `stage1.identity.json`.
IDENTITY_SUFFIX = ".identity.json"
IDENTITY_KEYS = ("model_weights_sha256", "adapter_sha256", "serving_dtype", "table_sha256")
#: The digest of the condition a table's campaign ran under (`report_tables.table_identity`); a recorded sidecar may
#: lack it.
CONDITION_KEY = "condition_sha256"


def identity_path(table: Path | str) -> Path:
    table = Path(table)
    return table.with_name(table.stem + IDENTITY_SUFFIX)


def write_table_identity(
    table: Path | str, *, model_weights_sha256: str | None, serving_dtype: str, adapter_sha256: str | None = None,
    condition_sha256: str | None = None, contents: Path | str | None = None,
) -> dict[str, Any]:
    """Write a table's identity sidecar: the campaign condition that produced it and the table's SHA-256."""
    identity = {"model_weights_sha256": model_weights_sha256, "adapter_sha256": adapter_sha256,
                "serving_dtype": serving_dtype, **({} if condition_sha256 is None else {CONDITION_KEY: condition_sha256}),
                "table_sha256": sha256_file(contents or table)}
    identity_path(table).write_text(json.dumps(identity, indent=1) + "\n", encoding="utf-8")
    return identity


def read_table_identity(table: Path | str, expect: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read a table's identity sidecar, refused when missing, naming other bytes than the table's, or differing from
    `expect`."""
    path = identity_path(table)
    if not path.is_file():
        raise ReportError(f"{table} has no identity sidecar {path.name}: the weights and dtype that produced it are "
                          "unknown")
    identity = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(identity, dict) or sorted(set(identity) - {CONDITION_KEY}) != sorted(IDENTITY_KEYS):
        raise ReportError(f"{path} must hold exactly {list(IDENTITY_KEYS)}, and {CONDITION_KEY} where recorded")
    digest = sha256_file(table)
    if identity["table_sha256"] != digest:
        raise ReportError(f"{path} names another table than {table} (SHA-256 {digest}): the table was rewritten after "
                          "its identity")
    wrong = {key: identity[key] for key, value in (expect or {}).items() if identity[key] != value}
    if wrong:
        raise ReportError(f"{table} was produced under {wrong}, not {dict(expect or {})}")
    return identity


@dataclass(frozen=True)
class Context:
    """What the statistics read of the scenarios and the base bank, row-aligned with the data they get."""

    q: np.ndarray  # [N, 3] q-hat(no, unsure, yes)
    mode: np.ndarray  # [N] the label code with the largest q-hat
    flagged: np.ndarray  # [N, R] bool: the bank's p-hat_r(yes) > 0
    source: np.ndarray  # [N] source code
    n_sources: int
    many_yes: int  # 'maps with >= this many yes'

    def take(self, rows: np.ndarray) -> "Context":
        return Context(self.q[rows], self.mode[rows], self.flagged[rows], self.source[rows], self.n_sources,
                       self.many_yes)


def bank_context(scenarios: Scenarios, bank: RuleDraws, *, many_yes: int) -> Context:
    """Return the `Context` of every scenario from its bank, q-hat and its mode as the anchors estimate them; a bank draw
    short of a label is refused."""
    # imported here, so that importing the report loads no model backend
    from ruleforge.grpo.anchor_set import estimate, mode_of

    if not bank.present.all():
        i = int(np.argwhere(~bank.present)[0][0])
        raise ReportError(f"the bank of scenario {scenarios.ids[i]!r} misses a label: q-hat needs every rule in every "
                          "draw")
    rules = [str(rule) for rule in range(bank.label.shape[2])]
    exact = [estimate([dict(zip(rules, (LABELS[code] for code in draw))) for draw in draws], rules).q
             for draws in bank.label]
    q = np.array([[float(row[label]) for label in LABELS] for row in exact])
    mode = np.array([CODE[mode_of(row)] for row in exact], dtype=np.int8)
    sources = sorted(set(scenarios.sources))
    source = np.array([sources.index(s) for s in scenarios.sources], dtype=int)
    return Context(q, mode, (bank.label == YES).any(axis=1), source, len(sources), many_yes)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _validity(data: FullSet, ctx: Context) -> np.ndarray:
    return np.stack([data.accepted.sum(1), data.attempts.sum(1)], 1)


def _truncation(data: FullSet, ctx: Context) -> np.ndarray:
    return np.stack([data.truncated.sum(1), data.attempts.sum(1)], 1)


def _complete_maps(data: FullSet, ctx: Context) -> np.ndarray:
    return np.stack([(data.accepted & data.complete).sum(1), data.accepted.sum(1)], 1)


def _map_yes(data: FullSet) -> tuple[np.ndarray, np.ndarray]:
    complete = data.accepted & data.complete
    return complete, (data.map_labels == YES).sum(-1) * complete


def _many_yes(data: FullSet, ctx: Context) -> np.ndarray:
    complete, yes = _map_yes(data)
    return np.stack([(complete & (yes >= ctx.many_yes)).sum(1), complete.sum(1)], 1)


def _yes_count(data: FullSet, ctx: Context) -> np.ndarray:
    complete, yes = _map_yes(data)
    return np.stack([yes.sum(1), complete.sum(1)], 1)


def _flag_precision(data: FullSet, ctx: Context) -> np.ndarray:
    complete, yes = _map_yes(data)
    flagged = ((data.map_labels == YES) & ctx.flagged[:, None, :]).sum(-1) * complete
    return np.stack([flagged.sum(1), yes.sum(1)], 1)


def _mode_agreement(data: FullSet, ctx: Context) -> np.ndarray:
    nonyes = (ctx.mode != YES)[:, None] & data.accepted
    return np.stack([(nonyes & (data.verdict == ctx.mode[:, None])).sum(1), nonyes.sum(1)], 1)


def _kappa_sums(data: FullSet, ctx: Context) -> np.ndarray:
    accepted = data.accepted
    n = accepted.sum(1)
    safe = np.where(accepted, data.verdict, 0).astype(int)
    po = (np.take_along_axis(ctx.q, safe, axis=1) * accepted).sum(1)
    counts = np.stack([((data.verdict == c) & accepted).sum(1) for c in range(3)], 1)
    columns = [po, n]
    for g in range(ctx.n_sources):
        member = ctx.source == g
        columns += [n * member, *(counts[:, c] * member for c in range(3)),
                    *(n * ctx.q[:, c] * member for c in range(3))]
    return np.stack(columns, 1).astype(float)


def _kappa_value(sums: np.ndarray, n_sources: int) -> np.ndarray:
    n = sums[..., 1]
    p_o = _ratio(sums[..., 0], n)
    p_e = np.zeros_like(n, dtype=float)
    for g in range(n_sources):
        base = 2 + 7 * g
        n_g = sums[..., base]
        for c in range(3):
            p_e = p_e + np.nan_to_num(_ratio(sums[..., base + 1 + c] * sums[..., base + 4 + c], n * n_g))
    return _ratio(p_o - p_e, 1 - p_e)


def _or_yes(data: RuleDraws, ctx: Context) -> np.ndarray:
    # a draw missing a rule's label has no known OR: only complete draws count
    has = data.present.all(-1)
    return np.stack([((data.label == YES).any(-1) & has).sum(1), has.sum(1)], 1)


def _rule_yes(data: RuleDraws, ctx: Context) -> np.ndarray:
    return np.stack([(data.label == YES).sum((1, 2)), data.present.sum((1, 2))], 1)


def _m1_side(data: FullSet, ctx: Context) -> np.ndarray:
    accepted = data.accepted
    safe = np.where(accepted, data.verdict, 0).astype(int)
    q = (np.take_along_axis(ctx.q, safe, axis=1) * accepted).sum(1)
    mean = _ratio(q.astype(float), accepted.sum(1).astype(float))
    has = accepted.any(1)
    return np.stack([np.nan_to_num(1 - mean) * has, has.astype(float)], 1)


def _ratio_value(sums: np.ndarray) -> np.ndarray:
    return _ratio(sums[..., 0], sums[..., 1])


@dataclass(frozen=True)
class Statistic:
    """One gate statistic: the difference between a save's draws and the base's reference draws, computed from
    per-scenario sums; `direction` says which side is outside the tolerance."""

    name: str
    gate: str
    stage: str
    direction: str
    definition: str
    components: Callable[[Any, Any, Context], np.ndarray]
    value: Callable[[np.ndarray, Context], np.ndarray]
    count: Callable[[np.ndarray, Context], np.ndarray]

    def delta(self, cand: Any, ref: Any, ctx: Context) -> float:
        return float(self.value(self.components(cand, ref, ctx).sum(0), ctx))

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "gate": self.gate, "stage": self.stage, "direction": self.direction,
                "definition": self.definition}


def _side_difference(side: Callable[[Any, Context], np.ndarray]):

    def components(cand: Any, ref: Any, ctx: Context) -> np.ndarray:
        return np.concatenate([side(cand, ctx), side(ref, ctx)], 1)

    def value(sums: np.ndarray, ctx: Context) -> np.ndarray:
        return _ratio_value(sums[..., :2]) - _ratio_value(sums[..., 2:])

    def count(sums: np.ndarray, ctx: Context) -> np.ndarray:
        return sums[..., 3]

    return components, value, count


def _kappa_difference():
    def components(cand: FullSet, ref: FullSet, ctx: Context) -> np.ndarray:
        return np.concatenate([_kappa_sums(cand, ctx), _kappa_sums(ref, ctx)], 1)

    def value(sums: np.ndarray, ctx: Context) -> np.ndarray:
        half = sums.shape[-1] // 2
        return _kappa_value(sums[..., :half], ctx.n_sources) - _kappa_value(sums[..., half:], ctx.n_sources)

    def count(sums: np.ndarray, ctx: Context) -> np.ndarray:
        return sums[..., sums.shape[-1] // 2 + 1]

    return components, value, count


def _statistic(name: str, gate: str, stage: str, direction: str, definition: str, parts) -> Statistic:
    return Statistic(name, gate, stage, direction, definition, *parts)


#: Every gate statistic with a tolerance, in gate order. The thinking gate, G7, has fixed thresholds instead.
STATISTICS: tuple[Statistic, ...] = (
    _statistic("validity_per_attempt", "G1", STAGE1, HIGHER,
               "accepted full-set responses / full-set attempts", _side_difference(_validity)),
    _statistic("truncation_rate", "G1", STAGE1, LOWER,
               "truncated full-set attempts / full-set attempts", _side_difference(_truncation)),
    _statistic("complete_map_rate", "G2", STAGE1, HIGHER,
               "accepted responses with a complete rule map / accepted responses", _side_difference(_complete_maps)),
    _statistic("kappa_bank", "G3", STAGE1, HIGHER,
               "(P_o - P_e) / (1 - P_e); P_o = mean over accepted draws of q-hat_s(phi_sd); P_e = sum over sources "
               "of w_source sum_c P(phi = c | source) * draw-weighted mean q-hat_s(c)", _kappa_difference()),
    _statistic("mode_agreement_nonyes", "G4", STAGE1, HIGHER,
               "P(judgment = the bank's mode | mode != yes), over accepted draws", _side_difference(_mode_agreement)),
    _statistic("probe_or_yes_share", "G5p", PROBE, TWO_SIDED,
               "share of probe draws whose isolated per-rule labels include a yes", _side_difference(_or_yes)),
    _statistic("probe_rule_yes_rate", "G5p", PROBE, TWO_SIDED,
               "pooled isolated per-rule yes rate on the probe", _side_difference(_rule_yes)),
    _statistic("many_yes_map_share", "G6", STAGE1, LOWER,
               "share of complete maps with at least many_yes yes labels", _side_difference(_many_yes)),
    _statistic("flag_precision", "G6", STAGE1, HIGHER,
               "share of complete maps' yes rules the bank ever said yes to (p-hat_r(yes) > 0)",
               _side_difference(_flag_precision)),
    _statistic("map_yes_count", "G6", STAGE1, LOWER,
               "mean yes labels per complete map", _side_difference(_yes_count)),
)


def m1_value(data: FullSet, ctx: Context) -> float:
    return float(_ratio_value(_m1_side(data, ctx).sum(0)))


@dataclass(frozen=True)
class Settings:
    """The report's settings: `configs/training/report.yaml`."""

    raw: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def subsets(self, gate: str) -> tuple[str, ...]:
        """Return the scenario subsets a stage-1 gate (or `m1`) is computed on."""
        return tuple(self.raw["stage1"]["subsets"][gate])

    def floored(self, gate: str) -> bool:
        """Return whether `gate`'s tolerances are floored at 3 / n."""
        return gate in self.raw["floored_gates"]


REQUIRED_KEYS = ("ruleset_sha256", "null_model", "interval", "floored_gates", "stage1", "probe")


def load_settings(raw: Mapping[str, Any]) -> Settings:
    """Return `Settings` from their mapping; a missing section, a gate or subset not known here, or base draws that a
    null cannot be split from is refused."""
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ReportError(f"the report's settings have no {missing}")
    stage1, probe = raw["stage1"], raw["probe"]
    for name, pool, needed in (("stage1", stage1["base_draws"], stage1["draws"] + len(stage1["base_reference_draws"])),
                               ("probe", probe["bank_draws"], 2)):
        if pool < needed:
            raise ReportError(f"{name}: the base's {pool} draws cannot be split into a save's draws and the reference "
                              f"draws ({needed}): the null would have no split")
    subsets = stage1.get("subsets") or {}
    absent = [gate for gate in ("m1", "G1", "G2", "G3", "G4", "G6", "G7") if gate not in subsets]
    if absent:
        raise ReportError(f"the report's settings name no scenario subsets for {absent}")
    unknown = sorted({s for values in subsets.values() for s in values} - set(SUBSETS))
    if unknown:
        raise ReportError(f"the report's settings name unknown subsets {unknown}")
    if set(raw["floored_gates"]) - set(GATES):
        raise ReportError(f"floored_gates {raw['floored_gates']} must name gates of {list(GATES)}")
    return Settings(dict(raw))


def read_settings(path: Path | str) -> Settings:
    """Return the settings in a YAML file."""
    import yaml

    return load_settings(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def ordered_splits(pool: Sequence[int], candidate: int, reference: int
                   ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return every (save draws, reference draws) of disjoint parts of `pool` with these sizes, in a fixed order."""
    splits = []
    for ref in itertools.combinations(pool, reference):
        rest = [d for d in pool if d not in ref]
        splits += [(cand, ref) for cand in itertools.combinations(rest, candidate)]
    return splits


def split_choices(n_scenarios: int, n_splits: int, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Return `[resamples, N]`: the split each scenario takes in each resample."""
    return rng.integers(0, n_splits, size=(resamples, n_scenarios))


def permutation_sums(parts: np.ndarray, choices: np.ndarray) -> np.ndarray:
    """Return `[resamples, m]`: the per-scenario sums of the split each scenario took, summed over scenarios."""
    sums = np.zeros((choices.shape[0], parts.shape[2]))
    for k in range(parts.shape[0]):
        sums += (choices == k).astype(float) @ parts[k]
    return sums


def _rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(name.encode("utf-8"))])


def null_values(
    parts: Sequence[np.ndarray],
    values: Sequence[Callable[[np.ndarray], np.ndarray]],
    *,
    resamples: int,
    seed: int,
    name: str,
) -> list[np.ndarray]:
    """Return each statistic's null under one set of per-scenario splits; every statistic of the group sees the same
    resamples, so their nulls are joint."""
    widths = [part.shape[2] for part in parts]
    stacked = np.concatenate(parts, axis=2).astype(float)
    choices = split_choices(stacked.shape[1], stacked.shape[0], resamples, _rng(seed, name))
    sums = permutation_sums(stacked, choices)
    edges = np.cumsum([0, *widths])
    return [value(sums[:, a:b]) for value, a, b in zip(values, edges[:-1], edges[1:])]


#: A delta this close to its tolerance is within it: a tie in exact arithmetic can round either way.
TIE_SLACK = 1e-12


def outside(values: np.ndarray, tol: float, direction: str) -> np.ndarray:
    """Return where a delta is outside a tolerance `tol` in `direction`; within `TIE_SLACK` it is inside, and a
    non-finite value is outside."""
    values = np.asarray(values, dtype=float)
    edge = tol + TIE_SLACK
    with np.errstate(invalid="ignore"):
        inside = {HIGHER: values >= -edge, LOWER: values <= edge, TWO_SIDED: np.abs(values) <= edge}[direction]
    return ~(np.isfinite(values) & inside)


def null_quantile(values: np.ndarray, direction: str, level: float) -> float:
    """Return the null's `level` quantile in the outside direction (of `|.|` when two-sided), NaNs dropped."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ReportError("the null has no finite value: the statistic is not defined on the base's draws")
    oriented = {HIGHER: -finite, LOWER: finite, TWO_SIDED: np.abs(finite)}[direction]
    return float(np.quantile(oriented, level))


def _summary(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    q05, q50, q95 = (float(x) for x in np.quantile(finite, [0.05, 0.5, 0.95]))
    return {"n": int(values.size), "nonfinite": int(values.size - finite.size), "mean": float(finite.mean()),
            "sd": float(finite.std()), "q05": q05, "q50": q50, "q95": q95}


@dataclass(frozen=True)
class References:
    """The base's draws: the scenario table, the rule ids, the bank (K draws) and the full-set draws (S)."""

    scenarios: Scenarios
    rule_ids: tuple[str, ...]
    bank: RuleDraws
    full: FullSet
    context: Context
    inputs: Mapping[str, str] = field(default_factory=dict)
    #: The campaign condition the bank and full-set tables come from.
    identity: Mapping[str, Any] = field(default_factory=dict)


def read_references(directory: Path | str, settings: Settings, rule_ids: Sequence[str]) -> References:
    """Read the base's draws from their tables; the bank and the full-set draws must come from one condition of the
    base."""
    directory = Path(directory)
    scenarios_path, bank_path, full_path = (directory / name for name in (SCENARIOS_FNAME, BANK_FNAME, FULL_FNAME))
    scenarios = read_scenarios(scenarios_path)
    everything = np.arange(len(scenarios.ids))
    _, bank = read_responses(bank_path, scenarios, everything, rule_ids, settings["probe"]["bank_draws"],
                             full=False, rules=True)
    full, _ = read_responses(full_path, scenarios, everything, rule_ids, settings["stage1"]["base_draws"],
                             full=True, rules=False)
    base = {"model_weights_sha256": None, "adapter_sha256": None}
    identities = [read_table_identity(path, base) for path in (bank_path, full_path)]
    dtypes = sorted({identity["serving_dtype"] for identity in identities})
    if len(dtypes) != 1:
        raise ReportError(f"the bank and the full-set draws were served in {dtypes}: one engine serves both")
    conditions = [identity.get(CONDITION_KEY) for identity in identities]
    if conditions[0] != conditions[1]:
        raise ReportError(f"the bank and the full-set draws come from other conditions ({conditions[0]} and "
                          f"{conditions[1]}): one condition of the base judges both")
    context = bank_context(scenarios, bank, many_yes=settings["stage1"]["many_yes"])
    inputs = {name: sha256_file(path) for name, path in
              (("scenarios", scenarios_path), ("bank", bank_path), ("full", full_path))}
    return References(scenarios, tuple(rule_ids), bank, full, context, inputs, {**base, "serving_dtype": dtypes[0]})


def probe_scenarios(scenarios: Scenarios, count: int, seed: int) -> list[str]:
    """Return a fixed set of `count` dev scenarios, split over sources by largest remainder, each source's taken in the
    order of SHA-256(seed:id)."""
    dev = [i for i in scenarios.rows([DEV])]
    if count > len(dev):
        raise ReportError(f"{count} fixed dev scenarios are wanted, dev has {len(dev)}")
    by_source: dict[str, list[str]] = {}
    for i in dev:
        by_source.setdefault(scenarios.sources[i], []).append(scenarios.ids[i])
    shares = {source: count * len(ids) / len(dev) for source, ids in by_source.items()}
    quota = {source: int(math.floor(share)) for source, share in shares.items()}
    for source in sorted(shares, key=lambda s: (-(shares[s] - quota[s]), s))[: count - sum(quota.values())]:
        quota[source] += 1
    chosen = []
    for source in sorted(by_source):
        ranked = sorted(by_source[source], key=lambda sid: hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest())
        chosen += ranked[: quota[source]]
    return sorted(chosen)


def stage1_scenarios(scenarios: Scenarios, settings: Settings) -> list[str]:
    """Return stage 1's fixed scenarios: `stage1.dev_scenarios` dev scenarios fixed as `probe_scenarios` fixes them, and
    every dev_nonyes one, sorted."""
    stage = settings["stage1"]
    chosen = probe_scenarios(scenarios, stage["dev_scenarios"], stage["seed"])
    return sorted([*chosen, *(scenarios.ids[i] for i in scenarios.rows([DEV_NONYES]))])


def scenario_sets(scenarios: Scenarios, settings: Settings) -> dict[str, list[str]]:
    """Return the fixed scenarios of stage 1 and of the probe."""
    probe = settings["probe"]
    return {STAGE1: stage1_scenarios(scenarios, settings),
            PROBE: probe_scenarios(scenarios, probe["scenarios"], probe["seed"])}


def _stage1_rows(refs: References, sets: Mapping[str, Sequence[str]]) -> np.ndarray:
    return np.sort(refs.scenarios.rows_of(sets[STAGE1]))


def _in_subsets(refs: References, rows: np.ndarray, subsets: Sequence[str]) -> np.ndarray:
    wanted = set(subsets)
    return np.array([row for row in rows if refs.scenarios.subsets[row] in wanted], dtype=int)


def _base_draws(stage: str, refs: References, draws: Sequence[int], rows: np.ndarray | None) -> Any:
    return (refs.full if stage == STAGE1 else refs.bank).take(rows, draws)


def _stage_rows(statistic: Statistic, refs: References, settings: Settings, sets: Mapping[str, Any]) -> np.ndarray:
    if statistic.stage == STAGE1:
        return _in_subsets(refs, _stage1_rows(refs, sets), settings.subsets(statistic.gate))
    return refs.scenarios.rows_of(sets[PROBE])


def _pool(statistic: Statistic, settings: Settings) -> tuple[list[int], int, list[int]]:
    if statistic.stage == STAGE1:
        stage = settings["stage1"]
        return list(range(stage["base_draws"])), stage["draws"], list(stage["base_reference_draws"])
    probe = settings["probe"]
    return list(range(probe["bank_draws"])), 1, [probe["base_reference_draw"]]


def _null_parts(statistic: Statistic, refs: References, rows: np.ndarray,
                splits: Sequence[tuple[tuple[int, ...], tuple[int, ...]]]) -> np.ndarray:
    ctx = refs.context.take(rows)
    rowed = replace(refs, bank=refs.bank.take(rows), full=refs.full.take(rows))
    parts = [statistic.components(_base_draws(statistic.stage, rowed, cand, None),
                                  _base_draws(statistic.stage, rowed, ref, None), ctx) for cand, ref in splits]
    out = np.zeros((len(parts), len(refs.scenarios.ids), parts[0].shape[1]))
    out[:, rows] = np.stack(parts)
    return out


def base_count(statistic: Statistic, refs: References, settings: Settings, rows: np.ndarray) -> float:
    """Return the n of the 3 / n floor: the observations a statistic's reference side is computed over, on as many base
    draws as a save contributes."""
    pool, candidate, reference = _pool(statistic, settings)
    draws = [*reference, *(draw for draw in pool if draw not in reference)][:candidate]
    ctx = refs.context.take(rows)
    sums = statistic.components(_base_draws(statistic.stage, refs, draws, rows),
                                _base_draws(statistic.stage, refs, draws, rows), ctx).sum(0)
    return float(statistic.count(sums, ctx))


def tolerance_floor(statistic: Statistic, refs: References, settings: Settings, rows: np.ndarray
                    ) -> dict[str, Any] | None:
    """Return the 3 / n floor under a statistic's tolerance, or None when its gate is not floored."""
    if not settings.floored(statistic.gate):
        return None
    n = base_count(statistic, refs, settings, rows)
    if n <= 0:
        raise ReportError(f"{statistic.name} is floored, and the base's reference draws hold no observation of it: "
                          "3 / n has no n")
    return {"n": int(n), "value": 3.0 / n}


def _median_think(data: FullSet) -> float:
    tokens = data.think_tokens[data.accepted]
    return float(np.median(tokens)) if tokens.size else math.nan


def _empty_think_share(data: FullSet) -> float:
    accepted = data.accepted
    return float(data.empty_think[accepted].mean()) if accepted.any() else math.nan


def compute_tolerances(refs: References, settings: Settings) -> dict[str, Any]:
    """Return every statistic's tolerance (the null's quantile, floored), the base's thinking and m1, and the fixed
    scenarios."""
    null = settings["null_model"]
    sets = scenario_sets(refs.scenarios, settings)
    statistics: dict[str, Any] = {}
    for stage in (STAGE1, PROBE):
        group = [statistic for statistic in STATISTICS if statistic.stage == stage]
        pool, candidate, reference = _pool(group[0], settings)
        splits = ordered_splits(pool, candidate, len(reference))
        rows = [_stage_rows(statistic, refs, settings, sets) for statistic in group]
        contexts = [refs.context.take(r) for r in rows]
        nulls = null_values(
            [_null_parts(statistic, refs, r, splits) for statistic, r in zip(group, rows)],
            [lambda sums, s=statistic, ctx=ctx: s.value(sums, ctx) for statistic, ctx in zip(group, contexts)],
            resamples=null["resamples"], seed=null["seed"], name=stage,
        )
        for statistic, statistic_rows, ctx, values in zip(group, rows, contexts, nulls):
            quantile = null_quantile(values, statistic.direction, null["level"])
            floor = tolerance_floor(statistic, refs, settings, statistic_rows)
            statistics[statistic.name] = {
                **statistic.describe(),
                "subsets": list(settings.subsets(statistic.gate)) if stage == STAGE1 else [DEV],
                "scenarios": int(len(ctx.q)), "splits": len(splits), "resamples": int(null["resamples"]),
                "quantile": quantile, "floor": floor, "tolerance": max(quantile, 0.0, floor["value"] if floor else 0.0),
                "floored": bool(floor and floor["value"] > max(quantile, 0.0)), "null": _summary(values),
            }
    rows1 = _stage1_rows(refs, sets)
    think = refs.full.take(_in_subsets(refs, rows1, settings.subsets("G7")))
    m1_rows = _in_subsets(refs, rows1, settings.subsets("m1"))
    base_m1 = m1_value(refs.full.take(m1_rows, settings["stage1"]["base_reference_draws"]), refs.context.take(m1_rows))
    return json_ready({
        "kind": TOLERANCES_KIND,
        "inputs": dict(refs.inputs),
        "reference_identity": dict(refs.identity),
        "scenarios": {"count": len(refs.scenarios.ids), "sha256": canonical_digest(list(zip(
            refs.scenarios.ids, refs.scenarios.sources, refs.scenarios.subsets)))},
        "stage1_scenarios": sets[STAGE1],
        "probe_scenarios": sets[PROBE],
        "statistics": statistics,
        "thinking": {"base_median_think_tokens": _median_think(think),
                     "base_empty_think_share": _empty_think_share(think)},
        "base_m1": base_m1,
    })


def _interval(side: np.ndarray, base_side: np.ndarray, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return percentile scenario bootstraps of a save's per-scenario m1 and of that m1 minus the base's."""
    has = side[:, 1] > 0
    both = has & (base_side[:, 1] > 0)
    options = {"n_resamples": spec["resamples"], "ci_level": spec["level"], "seed": spec["seed"]}
    point, low, high = bootstrap_ci(side[has, 0], **options) if has.any() else (None, None, None)
    paired, paired_low, paired_high = (paired_bootstrap_diff_ci(side[both, 0], base_side[both, 0], **options)
                                       if both.any() else (None, None, None))
    return {"m1": {"bootstrap_mean": point, "ci95": [low, high], "scenarios": int(has.sum())},
            "delta": {"paired_mean": paired, "ci95": [paired_low, paired_high], "scenarios": int(both.sum())}}


def save_report(
    refs: References, settings: Settings, tolerances: Mapping[str, Any], full: FullSet, probe: RuleDraws,
) -> dict[str, Any]:
    """Return one save's report from its full-set draws over stage 1's scenarios and its probe draw: m1 with its
    interval, its change from the base paired over the scenarios both answered, and each gate statistic with whether
    it is within its tolerance."""
    stage = settings["stage1"]
    sets = {STAGE1: tolerances["stage1_scenarios"], PROBE: tolerances["probe_scenarios"]}
    rows1 = _stage1_rows(refs, sets)
    if full.verdict.shape != (len(rows1), stage["draws"]):
        raise ReportError(f"stage 1 needs {stage['draws']} full-set draws of each of its {len(rows1)} scenarios")
    position = {int(row): i for i, row in enumerate(rows1)}

    def among(rows: np.ndarray) -> np.ndarray:
        return np.array([position[int(row)] for row in rows], dtype=int)

    results: dict[str, Any] = {}
    for statistic in STATISTICS:
        rows = _stage_rows(statistic, refs, settings, sets)
        _, _, reference = _pool(statistic, settings)
        cand = full.take(among(rows)) if statistic.stage == STAGE1 else probe
        delta = statistic.delta(cand, _base_draws(statistic.stage, refs, reference, rows), refs.context.take(rows))
        tol = tolerances["statistics"][statistic.name]["tolerance"]
        within = not bool(outside(np.array([delta]), tol, statistic.direction)[0])
        results[statistic.name] = {"gate": statistic.gate, "delta": delta, "tolerance": tol,
                                   "direction": statistic.direction, "within_tolerance": within}
    thinking = stage["thinking"]
    think = full.take(among(_in_subsets(refs, rows1, settings.subsets("G7"))))
    median, empty = _median_think(think), _empty_think_share(think)
    threshold = thinking["median_ratio"] * tolerances["thinking"]["base_median_think_tokens"]
    results["median_think_tokens"] = {"gate": "G7", "value": median, "threshold": threshold,
                                      "within_tolerance": bool(median >= threshold)}
    results["empty_think_share"] = {"gate": "G7", "value": empty, "threshold": thinking["max_empty_share"],
                                    "within_tolerance": bool(empty <= thinking["max_empty_share"])}
    m1_rows = _in_subsets(refs, rows1, settings.subsets("m1"))
    ctx = refs.context.take(m1_rows)
    data = full.take(among(m1_rows))
    m1, base_m1 = m1_value(data, ctx), tolerances["base_m1"]
    interval = _interval(_m1_side(data, ctx), _m1_side(refs.full.take(m1_rows, stage["base_reference_draws"]), ctx),
                         settings["interval"])
    gates = {gate: all(entry["within_tolerance"] for entry in results.values() if entry["gate"] == gate)
             for gate in GATES}
    return {"m1": {"value": m1, **interval["m1"]}, "base_m1": base_m1,
            "delta_m1_vs_base": {"value": interval["delta"]["paired_mean"], **interval["delta"]},
            "gates": gates, "outside_tolerance": [gate for gate in GATES if not gates[gate]], "statistics": results}


def read_save(directory: Path | str, refs: References, settings: Settings, tolerances: Mapping[str, Any]
              ) -> dict[str, Any]:
    """Return a save's report from its directory's tables; both must name one condition, served as the base was."""
    directory = Path(directory)
    stage1 = read_table_identity(directory / STAGE1_FNAME)
    condition = {key: stage1[key] for key in ("model_weights_sha256", "adapter_sha256", "serving_dtype")}
    probe = read_table_identity(directory / PROBE_FNAME, condition)
    if condition["serving_dtype"] != refs.identity["serving_dtype"]:
        raise ReportError(f"{directory} was served in {condition['serving_dtype']}, and the base in "
                          f"{refs.identity['serving_dtype']}: a statistic would compare two engines")
    rows1 = np.sort(refs.scenarios.rows_of(tolerances["stage1_scenarios"]))
    probe_rows = refs.scenarios.rows_of(tolerances["probe_scenarios"])
    full, _ = read_responses(directory / STAGE1_FNAME, refs.scenarios, rows1, refs.rule_ids,
                             settings["stage1"]["draws"], full=True, rules=False)
    _, draws = read_responses(directory / PROBE_FNAME, refs.scenarios, probe_rows, refs.rule_ids, 1,
                              full=False, rules=True)
    return {**condition, "tables": {STAGE1: stage1["table_sha256"], PROBE: probe["table_sha256"]},
            **save_report(refs, settings, tolerances, full, draws)}


def rule_ids_of(ruleset: Path | str, expect_sha256: str) -> tuple[str, ...]:
    """Return the non-meta rule ids of a ruleset JSONL in file order; its SHA-256 must be `expect_sha256`."""
    from ruleforge.rulesets.rules import is_meta_rule_id

    if sha256_file(ruleset) != expect_sha256:
        raise ReportError(f"{ruleset} is not the settings' ruleset (SHA-256 {expect_sha256})")
    return tuple(str(row["id"]).strip() for row in load_jsonl_records(ruleset) if not is_meta_rule_id(str(row["id"])))


def json_ready(value: Any) -> Any:
    """Return a payload as strict JSON holds it: numpy scalars as Python ones, tuples as lists, NaN and infinities as
    null."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _number(value: Any, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _interval_text(value: Any, entry: Mapping[str, Any], spec: str) -> str:
    return f"{_number(value, spec)} [{_number(entry['ci95'][0], spec)}, {_number(entry['ci95'][1], spec)}]"


def selected_save(saves: Mapping[str, Mapping[str, Any]]) -> int | None:
    """Return the save of ``saves`` with the lowest finite m1, the earlier save on a tie; None when no save has one."""
    ranked = [(value, int(step)) for step, save in saves.items()
              if (value := save["m1"]["value"]) is not None and math.isfinite(value)]
    return min(ranked)[1] if ranked else None


def report_table(saves: Mapping[str, Mapping[str, Any]], tolerances: Mapping[str, Any]) -> str:
    """Return the report as Markdown: the selected save, each save's m1 and its paired change over the scenarios the
    save and the base both answered, then each gate statistic by save, `*` where outside tolerance."""
    steps = sorted(saves, key=int)
    selected = selected_save(saves)
    lines = [f"Base m1: {_number(tolerances['base_m1'], '.4f')}",
             f"Selected: {selected}, the reported save with the lowest m1 (on a tie, the earlier save)" if selected is not None
             else "Selected: none, no save has an m1", "",
             "| Save | m1 [95% CI] | Change from the base [95% CI] | Gates outside tolerance |", "|---|---|---|---|"]
    for step in steps:
        save = saves[step]
        change = save["delta_m1_vs_base"]
        lines.append(f"| {step} | {_interval_text(save['m1']['value'], save['m1'], '.4f')} | "
                     f"{_interval_text(change['paired_mean'], change, '+.4f')} | "
                     f"{', '.join(save['outside_tolerance']) or 'none'} |")
    lines += ["", f"| Gate | Statistic | Tolerance | {' | '.join(steps)} |", f"|---|---|---|{'---|' * len(steps)}"]
    for statistic in STATISTICS:
        cells = [saves[step]["statistics"][statistic.name] for step in steps]
        shown = [f"{_number(cell['delta'], '+.4f')}{'' if cell['within_tolerance'] else '*'}" for cell in cells]
        tol = tolerances["statistics"][statistic.name]["tolerance"]
        lines.append(f"| {statistic.gate} | {statistic.name} | {tol:.4f} | {' | '.join(shown)} |")
    for name in ("median_think_tokens", "empty_think_share"):
        cells = [saves[step]["statistics"][name] for step in steps]
        shown = [f"{_number(cell['value'], '.4f')}{'' if cell['within_tolerance'] else '*'}" for cell in cells]
        lines.append(f"| G7 | {name} | {cells[0]['threshold']:.4f} | {' | '.join(shown)} |")
    return "\n".join(lines) + "\n"


def saves_with_tables(saves: Path | str) -> dict[int, Path]:
    """Return every `checkpoint-N` under `saves` with its two tables, refusing a save with one and not the other rather
    than letting it vanish from the report unannounced."""
    directories = {}
    for step in checkpoint_steps(saves):
        directory = Path(saves) / checkpoint_name(step)
        held = [name for name in (STAGE1_FNAME, PROBE_FNAME) if (directory / name).is_file()]
        if len(held) == 1:
            absent = PROBE_FNAME if held[0] == STAGE1_FNAME else STAGE1_FNAME
            raise ReportError(f"{directory} holds {held[0]} and no {absent}: read its campaign or move the save away")
        if held:
            directories[step] = directory
    if not directories:
        raise ReportError(f"{saves} holds no checkpoint-N directory with {STAGE1_FNAME} and {PROBE_FNAME}")
    return directories


def report(settings: Path | str, *, base: Path | str, saves: Path | str, rules: Path | str, out: Path | str
           ) -> dict[str, Any]:
    """Compute the tolerances from `base` and the report of every `checkpoint-N` under `saves` that has tables, select
    the reported save with the lowest m1, write `tolerances.json`, `report.json` and `report.md` under `out` and
    return the report."""
    loaded = read_settings(settings)
    directories = saves_with_tables(saves)
    refs = read_references(base, loaded, rule_ids_of(rules, loaded["ruleset_sha256"]))
    tolerances = compute_tolerances(refs, loaded)
    found = {str(step): json_ready(read_save(directory, refs, loaded, tolerances)) for step, directory in directories.items()}
    document = {"kind": REPORT_KIND, "saves": found, "selected": selected_save(found)}
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    (target / TOLERANCES_FNAME).write_text(json.dumps(tolerances, indent=1) + "\n", encoding="utf-8")
    (target / REPORT_FNAME).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    (target / TABLE_FNAME).write_text(report_table(found, tolerances), encoding="utf-8")
    return document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grpo_report", description="Report a training run's saves against the base.")
    sub = parser.add_subparsers(dest="command", required=True)
    saves = sub.add_parser("saves", help="the tolerances, every save's report and the reported save with the lowest "
                                         "m1: JSON and a table (CPU)")
    saves.add_argument("--settings", type=Path, required=True,
                       help="the report's settings (configs/training/report.yaml)")
    saves.add_argument("--base", type=Path, required=True,
                       help=f"a directory with the base's {SCENARIOS_FNAME}, {BANK_FNAME}, {FULL_FNAME}")
    saves.add_argument("--saves", type=Path, required=True,
                       help=f"checkpoint-N/ directories, each with {STAGE1_FNAME} and {PROBE_FNAME}")
    saves.add_argument("--rules", type=Path, required=True, help="the settings' ruleset")
    saves.add_argument("--out", type=Path, required=True, help="the directory to write the report in")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report(args.settings, base=args.base, saves=args.saves, rules=args.rules, out=args.out)
    print((args.out / TABLE_FNAME).read_text(encoding="utf-8"), end="")
    return 0
