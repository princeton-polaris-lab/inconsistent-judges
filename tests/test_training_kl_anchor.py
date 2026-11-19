from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="needs the grpo group (CPU torch is enough)")
import torch.nn.functional as F  # noqa: E402

from ruleforge.grpo.kl_anchor import (  # noqa: E402
    AnchorAuxConfig,
    AnchorReplayManager,
    ReplayRows,
    anchor_aux_loss,
    anchor_ce_loss,
    anchor_kl_loss,
    answer_segment_labels,
    collate_replay_rows,
    replay_seed,
    validate_anchor_num_workers,
    weighted_anchor_loss,
)

pytestmark = pytest.mark.torch


@pytest.fixture(autouse=True)
def _seeded():
    torch.manual_seed(0)


def _labels(batch: int, length: int, supervised: dict[int, int]):
    labels = torch.full((batch, length), -100)
    for position, token in supervised.items():
        labels[:, position] = token
    return labels


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"loss_weight": 1.0, "kind": "bad"}, "kind"),
        ({"loss_weight": -1.0}, "loss_weight"),
        ({"loss_weight": 1.0, "mix_every_n_steps": 0}, "mix_every"),
        ({"loss_weight": 1.0, "per_device_batch_size": 0}, "batch_size"),
        ({"loss_weight": 1.0, "dataloader_num_workers": -1}, "num_workers"),
        ({"loss_weight": 1.0, "dataloader_num_workers": 1}, "num_workers=0"),
    ],
)
def test_anchor_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AnchorAuxConfig(**kwargs)


def test_anchor_config_accepts_ce():
    assert AnchorAuxConfig(loss_weight=0.0, kind="ce").kind == "ce"


@pytest.mark.parametrize(
    "count,match",
    [
        (-1, "must be non-negative"),
        (1, "dataloader_num_workers=0"),
    ],
)
def test_the_num_workers_guard_rejects_counts_that_can_fork_after_cuda(count, match):
    """Positive is as wrong as negative: a worker forked after CUDA init hangs."""
    with pytest.raises(ValueError, match=match):
        validate_anchor_num_workers(count)


def test_the_num_workers_guard_returns_the_only_safe_count():
    assert validate_anchor_num_workers(0) == 0


def test_the_config_rejects_positive_workers_through_the_canonical_guard():
    with pytest.raises(ValueError, match="fork-after-CUDA"):
        AnchorAuxConfig(loss_weight=1.0, dataloader_num_workers=2)


@pytest.mark.parametrize(
    "logits,labels,match",
    [
        (torch.randn(2, 3), torch.ones(2, 3), "logits must"),
        (torch.randn(2, 3, 4), torch.ones(2, 3, 1), "labels must"),
        (torch.randn(2, 3, 4), torch.ones(1, 3), "shape mismatch"),
    ],
)
def test_anchor_ce_validates_shapes(logits, labels, match):
    with pytest.raises(ValueError, match=match):
        anchor_ce_loss(logits, labels)


def test_anchor_kl_zero_when_distributions_match():
    torch.manual_seed(0)
    policy = torch.randn(2, 4, 5)
    labels = _labels(2, 4, {2: 3, 3: 1})

    loss_sum, token_count = anchor_kl_loss(policy, policy.clone(), labels)

    assert int(token_count) == 4
    assert loss_sum.item() == pytest.approx(0.0, abs=1e-5)


def test_anchor_kl_positive_and_teacher_detached():
    policy = torch.randn(1, 4, 5, requires_grad=True)
    base = torch.randn(1, 4, 5, requires_grad=True)
    labels = _labels(1, 4, {2: 1})

    loss_sum, token_count = anchor_kl_loss(policy, base, labels)
    loss_sum.backward()

    assert int(token_count) == 1
    assert loss_sum.item() > 0
    assert policy.grad is not None
    assert policy.grad.abs().sum() > 0
    assert base.grad is None


def _one_token_kl_fixture(policy_row, base_row):
    """(1, 2, V) logits where exactly one prediction is supervised."""
    vocab = len(policy_row)
    policy = torch.zeros(1, 2, vocab)
    base = torch.zeros(1, 2, vocab)
    policy[0, 0] = torch.tensor(policy_row)
    base[0, 0] = torch.tensor(base_row)
    labels = torch.full((1, 2), -100)
    labels[0, 1] = 0
    return policy, base, labels


def _reference_kl(from_row, to_row, temperature=1.0):
    """KL(softmax(from/T) || softmax(to/T)) via explicit exp/sum arithmetic."""
    p_logits = torch.tensor(from_row, dtype=torch.float64) / temperature
    q_logits = torch.tensor(to_row, dtype=torch.float64) / temperature
    p = p_logits.exp() / p_logits.exp().sum()
    q = q_logits.exp() / q_logits.exp().sum()
    return float((p * (p / q).log()).sum())


def test_anchor_kl_is_forward_kl_from_the_frozen_base():
    policy_row = [0.0, 0.0, 0.0]  # uniform policy
    base_row = [2.0, 0.0, -2.0]  # peaked teacher
    policy, base, labels = _one_token_kl_fixture(policy_row, base_row)

    loss_sum, token_count = anchor_kl_loss(policy, base, labels)

    forward = _reference_kl(base_row, policy_row)
    reverse = _reference_kl(policy_row, base_row)
    assert int(token_count) == 1
    assert abs(forward - reverse) > 0.3
    assert loss_sum.item() == pytest.approx(forward, abs=1e-5)


def test_anchor_kl_applies_the_temperature_to_both_distributions():
    policy_row = [1.0, -1.0, 0.5, -0.5]
    base_row = [3.0, 0.0, -1.0, 1.0]
    policy, base, labels = _one_token_kl_fixture(policy_row, base_row)

    expected = {
        temperature: _reference_kl(base_row, policy_row, temperature)
        for temperature in (1.0, 4.0)
    }
    assert abs(expected[1.0] - expected[4.0]) > 0.05

    for temperature, want in expected.items():
        loss_sum, _ = anchor_kl_loss(policy, base, labels, temperature=temperature)
        assert loss_sum.item() == pytest.approx(want, abs=1e-5), f"T={temperature}"


def test_anchor_kl_masks_unsupervised_and_handles_zero_tokens():
    policy = torch.randn(1, 4, 5, requires_grad=True)
    base = policy.detach().clone()
    base[0, 0, 0] += 20
    labels = torch.full((1, 4), -100)

    loss_sum, token_count = anchor_kl_loss(policy, base, labels)
    loss_sum.backward()

    assert int(token_count) == 0
    assert loss_sum.item() == 0
    assert policy.grad is not None


def test_anchor_kl_validates_temperature_and_base_shape():
    policy = torch.randn(1, 4, 5)
    labels = _labels(1, 4, {2: 1})
    with pytest.raises(ValueError, match="temperature"):
        anchor_kl_loss(policy, policy, labels, temperature=0)
    with pytest.raises(ValueError, match="policy/base"):
        anchor_kl_loss(policy, torch.randn(1, 4, 6), labels)


def test_anchor_ce_matches_causal_reference():
    torch.manual_seed(2)
    batch, length, vocab = 2, 6, 7
    logits = torch.randn(batch, length, vocab)
    labels = torch.randint(0, vocab, (batch, length))
    labels[:, :2] = -100

    loss_sum, token_count = anchor_ce_loss(logits, labels)
    expected = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab),
        labels[:, 1:].reshape(-1),
        reduction="sum",
        ignore_index=-100,
    )

    assert torch.allclose(loss_sum, expected)
    assert int(token_count) == int(labels[:, 1:].ne(-100).sum())


def test_anchor_aux_kl_weight_and_accumulation():
    policy = torch.randn(1, 4, 5)
    base = policy.clone()
    base[0, 1, 0] += 3
    labels = _labels(1, 4, {2: 1})
    config = AnchorAuxConfig(loss_weight=0.2)

    weighted, raw = anchor_aux_loss(
        policy_logits=policy,
        base_logits=base,
        labels=labels,
        config=config,
        grad_accum_steps=4,
    )

    assert torch.allclose(weighted, raw * 0.2 / 4)
    assert not raw.requires_grad


def test_anchor_aux_ce_and_zero_token_clamp():
    policy = torch.randn(1, 4, 5)
    labels = torch.full((1, 4), -100)
    config = AnchorAuxConfig(loss_weight=1.0, kind="ce")

    weighted, raw = anchor_aux_loss(
        policy_logits=policy,
        base_logits=None,
        labels=labels,
        config=config,
    )

    assert weighted.item() == 0
    assert raw.item() == 0


def test_anchor_aux_validates_accumulation_and_kl_teacher():
    policy = torch.randn(1, 4, 5)
    labels = _labels(1, 4, {2: 1})
    config = AnchorAuxConfig(loss_weight=1.0)
    with pytest.raises(ValueError, match="grad_accum"):
        anchor_aux_loss(
            policy_logits=policy,
            base_logits=policy,
            labels=labels,
            config=config,
            grad_accum_steps=0,
        )
    with pytest.raises(ValueError, match="requires base_logits"):
        anchor_aux_loss(
            policy_logits=policy,
            base_logits=None,
            labels=labels,
            config=config,
        )


def _reference_ce(logits, labels):
    """The shifted cross entropy, summed, and its token count, computed apart from `anchor_ce_loss`."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_sum = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum",
                               ignore_index=-100)
    return loss_sum, shift_labels.ne(-100).sum()


def test_the_ce_loss_agrees_with_an_independent_reference():
    torch.manual_seed(3)
    logits = torch.randn(3, 7, 11)
    labels = torch.randint(0, 11, (3, 7))
    labels[0, :3] = -100
    labels[1, 2:5] = -100
    labels[2, :] = -100

    loss_sum, token_count = anchor_ce_loss(logits, labels)
    reference_loss, reference_count = _reference_ce(logits, labels)

    assert torch.allclose(loss_sum, reference_loss)
    assert int(token_count.item()) == int(reference_count.item())
    assert int(token_count.item()) == int(labels[:, 1:].ne(-100).sum())


def test_the_ce_loss_over_fully_masked_labels_is_zero_over_a_float32_count():
    logits = torch.randn(2, 5, 7, dtype=torch.float64)
    labels = torch.full((2, 5), -100)

    loss_sum, token_count = anchor_ce_loss(logits, labels)

    assert loss_sum.item() == 0.0 and token_count.item() == 0.0
    assert token_count.dtype == torch.float32


def test_anchor_kl_upcasts_reduced_precision_logits():
    torch.manual_seed(4)
    policy = torch.randn(1, 5, 9).to(torch.bfloat16)
    base = torch.randn(1, 5, 9).to(torch.bfloat16)
    labels = _labels(1, 5, {2: 1, 4: 3})
    loss_sum, count = anchor_kl_loss(policy, base, labels)
    reference, _ = anchor_kl_loss(policy.float(), base.float(), labels)
    assert loss_sum.dtype == torch.float32 and int(count) == 2
    assert loss_sum.item() == pytest.approx(reference.item(), abs=1e-6)


THINK_END = 9


def _row(n: int = 6, *, prompt: int = 2, weight: float | None = None, **extra) -> dict:
    row = {"input_ids": list(range(1, n + 1)), "labels": [-100] * prompt + list(range(prompt + 1, n + 1)), **extra}
    if weight is not None:
        row["weight"] = weight
    return row


def test_replay_rows_collate_right_padded_with_weights():
    batch = collate_replay_rows(
        [{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7], "weight": 0.5},
         {"input_ids": [8], "attention_mask": [1], "labels": [-100], "weight": 2.0}],
        pad_token_id=0,
    )
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [-100, -100, -100]]
    assert batch["weight"].tolist() == [0.5, 2.0] and batch["weight"].dtype == torch.float32


def test_the_answer_segment_is_what_follows_the_last_think_end():
    input_ids = torch.tensor([[1, THINK_END, 2, THINK_END, 3, 4], [1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[-100, 9, 2, 9, 3, 4], [-100, 2, 3, 4, 5, 6]])
    assert answer_segment_labels(input_ids, labels, THINK_END).tolist() == [
        [-100, -100, -100, -100, 3, 4],
        [-100] * 6,  # no </think>: no answer segment
    ]


def test_the_weighted_loss_is_each_rows_anchor_loss_at_its_weight():
    torch.manual_seed(6)
    config = AnchorAuxConfig(loss_weight=2.0)
    policy, base = torch.randn(3, 5, 7), torch.randn(3, 5, 7)
    labels = _labels(3, 5, {2: 1, 3: 4})
    labels[1, 3] = -100
    weights = torch.tensor([1.0, 0.5, 1.5])
    weighted, raw = weighted_anchor_loss(
        policy_logits=policy, base_logits=base, labels=labels, weights=weights, config=config
    )
    rows = [anchor_aux_loss(policy_logits=policy[i : i + 1], base_logits=base[i : i + 1], labels=labels[i : i + 1],
                            config=config) for i in range(3)]
    assert weighted.item() == pytest.approx(sum(w * r[0].item() for w, r in zip(weights.tolist(), rows)) / 3)
    assert raw.item() == pytest.approx(sum(r[1].item() for r in rows) / 3)
    one, one_raw = weighted_anchor_loss(policy_logits=policy[:1], base_logits=base[:1], labels=labels[:1],
                                        weights=torch.ones(1), config=config)
    assert (one.item(), one_raw.item()) == pytest.approx((rows[0][0].item(), rows[0][1].item()))
    ce, _ = weighted_anchor_loss(policy_logits=policy, base_logits=None, labels=labels, weights=weights,
                                 config=AnchorAuxConfig(loss_weight=1.0, kind="ce"))
    assert ce.item() > 0
    with pytest.raises(ValueError, match="one per row"):
        weighted_anchor_loss(policy_logits=policy, base_logits=base, labels=labels, weights=torch.ones(2),
                             config=config)


@pytest.mark.parametrize(
    "row,match",
    [
        ({"input_ids": [1, 2]}, "lacks"),
        ({"input_ids": [], "labels": []}, "non-empty"),
        ({"input_ids": [1, 2], "labels": [1]}, "one length"),
        ({"input_ids": [1, 2], "labels": [1, 2], "attention_mask": [1]}, "one length"),
        ({"input_ids": [1, 2], "labels": [1, -100]}, "supervises no token"),
        (_row(weight=0.0), "positive and finite"),
        (_row(weight=float("nan")), "positive and finite"),
    ],
)
def test_a_replay_row_that_would_train_nothing_or_misweigh_is_refused(row, match):
    with pytest.raises(ValueError, match=match):
        AnchorReplayManager([row], config=AnchorAuxConfig(loss_weight=1.0), pad_token_id=0, seed=0)


def _manager(rows=None, **kwargs) -> AnchorReplayManager:
    rows = rows if rows is not None else [_row(4 + i % 3, weight=None) for i in range(10)]
    settings = {"config": AnchorAuxConfig(loss_weight=1.0), "pad_token_id": 0, "seed": 11, **kwargs}
    return AnchorReplayManager(rows, **settings)


def _order(manager: AnchorReplayManager, n: int) -> list[tuple[int, ...]]:
    return [tuple(manager.next_batch()["input_ids"][0].tolist()) for _ in range(n)]


def test_the_replay_sampler_is_a_plain_random_sampler_over_the_rows():
    from torch.utils.data import RandomSampler

    manager = _manager([dict(_row(), input_ids=[i + 1] * 6) for i in range(10)])
    assert type(manager.sampler) is RandomSampler
    manager.start(0)
    firsts = [manager.next_batch()["input_ids"][0, 0].item() for _ in range(10)]
    assert sorted(firsts) == list(range(1, 11))
    assert manager.summary()["sampler"] == "RandomSampler" and manager.rows_served == 10


def test_the_replay_order_is_a_function_of_seed_and_start_step():
    rows = [dict(_row(), input_ids=[i + 1] * 6) for i in range(12)]
    first, again, resumed = _manager(rows), _manager(rows), _manager(rows)
    first.start(0)
    again.start(0)
    resumed.start(50)
    assert _order(first, 12) == _order(again, 12)
    assert _order(resumed, 12) != _order(_started(_manager(rows), 0), 12)
    assert replay_seed(11, 50, 0) == replay_seed(11, 50, 0) != replay_seed(11, 50, 1)
    other_seed = _manager(rows, seed=12)
    other_seed.start(0)
    assert _order(other_seed, 12) != _order(_started(_manager(rows), 0), 12)


def _started(manager: AnchorReplayManager, step: int) -> AnchorReplayManager:
    manager.start(step)
    return manager


def test_the_stream_cycles_into_a_fresh_pass():
    manager = _started(_manager([dict(_row(), input_ids=[i + 1] * 6) for i in range(3)]), 0)
    batches = _order(manager, 7)
    assert manager.cycle == 2 and sorted(batches[:3]) == sorted(batches[3:6])
    assert manager.summary()["cycle"] == 2 and manager.summary()["start_step"] == 0


def test_ranks_replay_disjoint_shards_of_one_permutation():
    rows = [dict(_row(), input_ids=[i + 1] * 6) for i in range(8)]
    shards = []
    for rank in range(2):
        manager = _started(_manager(rows, num_processes=2, process_index=rank), 5)
        shards.append({batch[0] for batch in _order(manager, 4)})
    assert shards[0].isdisjoint(shards[1]) and shards[0] | shards[1] == set(range(1, 9))


def test_row_weights_are_normalized_to_mean_one():
    manager = _manager([_row(weight=0.5), _row(weight=0.25), _row(weight=0.25), _row(weight=1.0)])
    assert manager.weight_scale == pytest.approx(2.0)
    weights = sorted(manager.dataset[index]["weight"] for index in range(len(manager.dataset)))
    assert weights == pytest.approx([0.5, 0.5, 1.0, 2.0]) and sum(weights) / 4 == pytest.approx(1.0)


def test_the_stream_refuses_misuse():
    with pytest.raises(ValueError, match="empty"):
        _manager([])
    with pytest.raises(ValueError, match="not one of"):
        _manager(num_processes=2, process_index=2)
    manager = _manager()
    assert not manager.started
    with pytest.raises(RuntimeError, match="not started"):
        manager.next_batch()


def test_a_resumed_stream_continues_its_row_count():
    manager = _started(_manager(rows_served=40, num_processes=2, process_index=1), 5)
    manager.next_batch()
    assert manager.rows_served == 41 and manager.summary()["num_processes"] == 2


def _replay_file(path, rows, *, blank_lines: bool = False) -> str:
    import hashlib
    import json

    text = "".join(json.dumps(row) + "\n" + ("\n" if blank_lines else "") for row in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def test_a_replay_file_is_indexed_and_its_rows_read_back_as_they_are_served(tmp_path):
    rows = [dict(_row(4 + i % 3), weight=0.5 if i % 2 else 0.25, scenario_id=f"s{i}") for i in range(7)]
    path = tmp_path / "replay.jsonl"
    digest = _replay_file(path, rows, blank_lines=True)
    stored = ReplayRows.from_jsonl(path, expected_sha256=digest)
    in_memory = ReplayRows.from_records(rows)
    assert len(stored) == 7 and stored.sha256 == digest and stored.weight_scale == pytest.approx(in_memory.weight_scale)
    assert [stored[i] for i in (6, 0, 3)] == [in_memory[i] for i in (6, 0, 3)]
    assert stored.summary() == {"source": str(path), "sha256": digest, "rows": 7,
                                "weight_scale": pytest.approx(1 / (sum(row["weight"] for row in rows) / 7))}
    first, again = _started(_manager(stored), 3), _started(_manager(in_memory), 3)
    assert [first.next_batch()["input_ids"].tolist() for _ in range(9)] == [
        again.next_batch()["input_ids"].tolist() for _ in range(9)]
    # the handle opened for the index is the one read: a file replaced afterwards is not
    path.write_text("")
    assert stored[1] == in_memory[1]


def test_a_replay_file_that_changed_or_is_not_rows_is_refused(tmp_path):
    path = tmp_path / "replay.jsonl"
    _replay_file(path, [_row()])
    with pytest.raises(ValueError, match="the file changed in between"):
        ReplayRows.from_jsonl(path, expected_sha256="0" * 64)
    path.write_text('{"input_ids": [1, 2]\n')
    with pytest.raises(ValueError, match="replay row 0 is not JSON"):
        ReplayRows.from_jsonl(path)
    path.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="replay row 0 is not a JSON object"):
        ReplayRows.from_jsonl(path)
    path.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        ReplayRows.from_jsonl(path)
    _replay_file(path, [_row(), _row(weight=-1.0)])
    with pytest.raises(ValueError, match="replay row 1 has weight"):
        ReplayRows.from_jsonl(path)


def test_an_indexed_replay_file_holds_a_few_bytes_per_row_not_the_rows(tmp_path):
    import gc
    import tracemalloc

    tokens = 400

    def held(count):
        rows = ({"input_ids": list(range(1, tokens + 1)), "labels": [-100] * 100 + list(range(101, tokens + 1)),
                 "weight": 0.5} for _ in range(count))
        path = tmp_path / f"replay-{count}.jsonl"
        _replay_file(path, rows)
        gc.collect()
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            stored = ReplayRows.from_jsonl(path)
            gc.collect()
            return len(stored), tracemalloc.get_traced_memory()[0] - before
        finally:
            tracemalloc.stop()

    (small, held_small), (large, held_large) = held(1500), held(3000)
    assert (small, large) == (1500, 3000)
    # an 8-byte offset per row, plus the array's growth slack; a parsed row here is ~40 KB of Python ints
    assert (held_large - held_small) / (large - small) < 64, (held_small, held_large)
