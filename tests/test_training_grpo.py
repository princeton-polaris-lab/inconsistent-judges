from __future__ import annotations

import pytest

from ruleforge.grpo.primitives import completion_text, should_run_anchor_step


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("text", "text"),
        (123, ""),
        ([], ""),
        ([123], ""),
        ([{"content": 123}], ""),
        ([{"role": "assistant", "content": "answer"}], "answer"),
    ],
)
def test_completion_text_formats(completion, expected):
    assert completion_text(completion) == expected


@pytest.mark.parametrize(
    "global_step,mix,sync,expected",
    [
        (0, 1, True, True),
        (0, 2, True, False),
        (1, 2, True, True),
        (1, 2, False, False),
    ],
)
def test_should_run_anchor_step(global_step, mix, sync, expected):
    assert should_run_anchor_step(
        global_step=global_step,
        mix_every_n_steps=mix,
        sync_gradients=sync,
    ) is expected


def test_should_run_anchor_step_validates_period():
    with pytest.raises(ValueError, match="must be positive"):
        should_run_anchor_step(
            global_step=0,
            mix_every_n_steps=0,
            sync_gradients=True,
        )


def test_the_anchor_step_fires_once_per_true_optimizer_step():
    grad_accum, optimizer_steps = 8, 10
    fired = []
    global_step = 0
    for micro in range(grad_accum * optimizer_steps):
        sync = (micro + 1) % grad_accum == 0
        if should_run_anchor_step(
            global_step=global_step, mix_every_n_steps=1, sync_gradients=sync
        ):
            fired.append(global_step)
        if sync:
            global_step += 1  # HF increments AFTER the step

    assert len(fired) == optimizer_steps
    assert fired == list(range(optimizer_steps))


def test_a_period_of_n_fires_every_nth_optimizer_step():
    grad_accum, optimizer_steps, period = 4, 12, 3
    fired = []
    global_step = 0
    for micro in range(grad_accum * optimizer_steps):
        sync = (micro + 1) % grad_accum == 0
        if should_run_anchor_step(
            global_step=global_step, mix_every_n_steps=period, sync_gradients=sync
        ):
            fired.append(global_step)
        if sync:
            global_step += 1

    assert len(fired) == optimizer_steps // period
    # Steps 2, 5, 8, 11 -- i.e. the last of each period, counting from one.
    assert fired == [2, 5, 8, 11]
