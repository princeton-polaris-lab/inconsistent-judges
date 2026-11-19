from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="needs the grpo group (CPU torch is enough)")

from ruleforge.grpo import truncation as tr  # noqa: E402

pytestmark = pytest.mark.torch

TRUNCATION = tr.Truncation(temperature=1.0, top_k=20, top_p=0.95)


def _vllm_reference(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """vLLM 0.28 ``apply_top_k_top_p_pytorch`` (sort ascending; top-k; top-p on the renormalized set), -inf masked."""
    rows = logits.reshape(-1, logits.size(-1)).clone()
    logits_sort, logits_idx = rows.sort(dim=-1, descending=False)
    if top_k > 0:
        kth = logits_sort.gather(1, torch.full((rows.size(0), 1), rows.size(-1) - top_k))
        logits_sort.masked_fill_(logits_sort < kth, -float("inf"))
    if top_p < 1.0:
        probs_sum = logits_sort.softmax(dim=-1).cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - top_p
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
    return rows.scatter(-1, logits_idx, logits_sort).view(logits.shape)


def _masked_logps(logits: torch.Tensor, truncation: tr.Truncation) -> torch.Tensor:
    """What TRL computes from the masked logits: log_softmax(masked / T)."""
    return torch.log_softmax(tr.mask_logits(logits, truncation).float() / truncation.temperature, dim=-1)


@pytest.mark.parametrize("top_k,top_p", [(20, 0.95), (5, 0.5), (0, 0.9), (20, 1.0), (64, 0.95), (1, 0.95), (3, 1e-6)])
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_the_mask_is_the_renormalized_truncated_distribution_in_vllms_order(top_k, top_p, temperature):
    torch.manual_seed(7)
    logits = torch.randn(3, 11, 64) * 3.0  # [batch, positions, vocab], fp32: no ties
    truncation = tr.Truncation(temperature=temperature, top_k=top_k, top_p=top_p)
    ours = _masked_logps(logits, truncation)
    reference = torch.log_softmax(_vllm_reference(logits / temperature, top_k, top_p), dim=-1)
    kept = torch.isfinite(reference)
    assert torch.equal(tr.keep_mask(logits, truncation), kept)
    torch.testing.assert_close(ours[kept], reference[kept], rtol=0, atol=1e-5)
    # renormalized: the kept mass is 1; everything else is below the outside threshold, finitely
    torch.testing.assert_close(ours.exp().sum(-1), torch.ones(3, 11), rtol=0, atol=1e-5)
    assert torch.isfinite(ours).all()
    assert (ours[~kept] < truncation.outside_logp).all()


def test_vllms_own_sampler_masks_the_same_tokens():
    ops = pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler", reason="needs the vllm group")
    torch.manual_seed(3)
    logits = torch.randn(40, 151) * 4.0
    for top_k, top_p in ((20, 0.95), (7, 0.6), (0, 0.8), (20, 1.0)):
        truncation = tr.Truncation(temperature=1.0, top_k=top_k, top_p=top_p)
        vllm_masked = ops.apply_top_k_top_p_pytorch(
            logits.clone(),
            torch.full((40,), top_k) if top_k else None,
            torch.full((40,), top_p) if top_p < 1 else None,
            allow_cpu_sync=True,
        )
        kept = torch.isfinite(vllm_masked)
        assert torch.equal(tr.keep_mask(logits, truncation), kept), (top_k, top_p)
        torch.testing.assert_close(
            _masked_logps(logits, truncation)[kept], torch.log_softmax(vllm_masked, dim=-1)[kept], rtol=0, atol=1e-5
        )


def test_the_mask_never_yields_nan_in_logps_entropy_or_gradient():
    """A token outside the kept set (the HF/vLLM boundary lag) gets a finite log-prob and IS weight ~0."""
    torch.manual_seed(1)
    for dtype in (torch.float32, torch.bfloat16):
        logits = (torch.randn(2, 5, 50) * 5).to(dtype).requires_grad_(True)
        masked = tr.mask_logits(logits, TRUNCATION)
        logps = torch.log_softmax(masked.float(), dim=-1)
        outside_token = (~tr.keep_mask(logits, TRUNCATION)).float().argmax(dim=-1)  # a token vLLM could not have kept
        picked = logps.gather(-1, outside_token.unsqueeze(-1)).squeeze(-1)
        assert torch.isfinite(picked).all() and (picked < TRUNCATION.outside_logp).all()
        entropy = -(logps.exp() * logps).sum(-1)  # 0 * finite, never 0 * -inf
        assert torch.isfinite(entropy).all()
        importance = torch.exp(picked - torch.full_like(picked, -3.0)).clamp(max=2.0)
        assert torch.equal(importance, torch.zeros_like(importance))
        (picked.sum() + entropy.sum()).backward()
        assert torch.isfinite(logits.grad).all()


def test_the_mask_is_computed_on_logits_over_temperature():
    """Top-p depends on the temperature: T scales the logits before the mask, as vLLM does."""
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    sharp = tr.keep_mask(logits, tr.Truncation(temperature=0.5, top_k=0, top_p=0.9))
    flat = tr.keep_mask(logits, tr.Truncation(temperature=2.0, top_k=0, top_p=0.9))
    assert sharp.sum() < flat.sum()


def test_ties_at_the_boundaries_keep_vllms_count_lowest_index_first():
    """vLLM's Triton kernel trims a tie at the k-th value, or at the top-p boundary, to the count, by index."""
    logits = torch.tensor([[5.0, 3.0, 3.0, 3.0, 1.0, 0.0]])
    assert tr.keep_mask(logits, tr.Truncation(temperature=1.0, top_k=2, top_p=1.0)).tolist() == [
        [True, True, False, False, False, False]
    ]
    even = torch.tensor([[2.0, 2.0, 2.0, -9.0]])  # exclusive masses 0, 1/3, 2/3: two are below 0.4
    assert tr.keep_mask(even, tr.Truncation(temperature=1.0, top_k=0, top_p=0.4)).tolist() == [[True, True, False, False]]
    # top-p over an exactly-k set: the k-th tie's third member is not in the renormalization
    assert tr.keep_mask(logits, tr.Truncation(temperature=1.0, top_k=3, top_p=0.99)).tolist() == [
        [True, True, True, False, False, False]
    ]


def _triton_rule(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """vLLM 0.28's Triton kernel's rule."""
    rows = logits.reshape(-1, logits.size(-1))
    out = torch.zeros(rows.shape, dtype=torch.bool)
    for index, row in enumerate(rows.tolist()):
        order = sorted(range(len(row)), key=lambda token: (-row[token], token))
        kept = order[:top_k] if 0 < top_k < len(row) else order
        if top_p < 1.0:
            probs = torch.softmax(torch.tensor([row[token] for token in kept]), dim=0)
            kept = kept[: max(1, int(((probs.cumsum(0) - probs) < top_p).sum()))]
        out[index, kept] = True
    return out.view(logits.shape)


def _tied_bf16_logits(rows: int = 96, vocab: int = 200) -> torch.Tensor:
    torch.manual_seed(11)
    return (torch.randn(rows, vocab) * 2.0).to(torch.bfloat16).float()  # bf16 values: ties at the boundaries


def _trimmed_ties(logits: torch.Tensor, keep: torch.Tensor) -> int:
    """Rows where a token equal to the smallest kept logit is dropped: a whole-tie rule would keep it."""
    smallest = logits.masked_fill(~keep, math.inf).min(dim=-1, keepdim=True).values
    return int(((logits == smallest) & ~keep).any(dim=-1).sum())


@pytest.mark.parametrize("top_k,top_p", [(20, 0.95), (20, 1.0), (0, 0.9), (5, 0.6)])
def test_on_tied_bf16_logits_the_mask_is_vllms_triton_rule(top_k, top_p):
    logits = _tied_bf16_logits()
    truncation = tr.Truncation(temperature=1.0, top_k=top_k, top_p=top_p)
    keep = tr.keep_mask(logits, truncation)
    assert torch.equal(keep, _triton_rule(logits, top_k, top_p))
    assert _trimmed_ties(logits, keep) > 0
    expected = top_k if top_p == 1.0 else None
    assert expected is None or (keep.sum(-1) == expected).all()


def test_on_tied_bf16_logits_vllms_pytorch_top_p_keeps_the_same_count_so_the_same_log_probs():
    """vLLM's small-batch path fills a top-p tie by its sort, possibly with other members."""
    ops = pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler", reason="needs the vllm group")
    logits = _tied_bf16_logits()
    truncation = tr.Truncation(temperature=1.0, top_k=0, top_p=0.9)
    vllm_masked = ops.apply_top_k_top_p_pytorch(logits.clone(), None, torch.full((logits.size(0),), 0.9),
                                                allow_cpu_sync=True)
    vllm_kept, keep = torch.isfinite(vllm_masked), tr.keep_mask(logits, truncation)
    assert torch.equal(vllm_kept.sum(-1), keep.sum(-1)) and _trimmed_ties(logits, keep) > 0
    joint = vllm_kept & keep
    torch.testing.assert_close(_masked_logps(logits, truncation)[joint], torch.log_softmax(vllm_masked, dim=-1)[joint],
                               rtol=0, atol=1e-5)


def test_an_outside_old_log_prob_zeroed_keeps_a_flipped_tokens_ratio_finite():
    old = torch.tensor([[-0.7, -10000.3, -2.0]])
    new = torch.tensor([[-0.6, -1.2, -2.1]], requires_grad=True)
    is_ratio = torch.tensor([[1.0, 0.0, 1.0]])  # the vLLM weight of the outside token: exp(old - sampled) = 0
    for zeroed, finite in ((False, False), (True, True)):
        reference = tr.zero_outside_logps(old, TRUNCATION) if zeroed else old
        loss = -(torch.exp(new - reference) * 0.5 * is_ratio).sum()
        new.grad = None
        loss.backward()
        assert bool(torch.isfinite(new.grad).all()) is finite
    assert torch.equal(tr.zero_outside_logps(old, TRUNCATION), torch.tensor([[-0.7, 0.0, -2.0]]))


def test_the_top_token_is_always_kept_and_a_disabled_truncation_masks_nothing():
    logits = torch.tensor([[0.3, 9.0, 0.1]])
    assert tr.keep_mask(logits, tr.Truncation(temperature=1.0, top_k=0, top_p=1e-9)).tolist() == [[False, True, False]]
    untruncated = tr.Truncation(temperature=1.0, top_k=0, top_p=1.0)
    assert not untruncated.truncates
    assert tr.mask_logits(logits, untruncated) is logits
    assert tr.keep_mask(logits, tr.Truncation(temperature=1.0, top_k=3, top_p=1.0)).all()


def test_the_chunked_mask_does_not_depend_on_the_chunk_size():
    torch.manual_seed(5)
    logits = torch.randn(4, 9, 33)
    whole = tr.keep_mask(logits, TRUNCATION, chunk_rows=1000)
    for chunk in (1, 5, 36):
        assert torch.equal(tr.keep_mask(logits, TRUNCATION, chunk_rows=chunk), whole)
    with pytest.raises(tr.TruncationError, match="chunk_rows"):
        tr.keep_mask(logits, TRUNCATION, chunk_rows=0)


def test_outside_kept_share_counts_only_the_masked_positions():
    logps = torch.tensor([[-1.0, -9000.0, -2.0], [-8000.0, -0.5, -7000.0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    assert tr.outside_kept_share(logps, mask, TRUNCATION) == (2, 5)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": math.inf}, "temperature"),
        ({"top_k": 2.5}, "top_k must be an integer"),
        ({"top_k": True}, "top_k must be an integer"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.5}, "top_p"),
        ({"min_p": 0.05}, "min_p"),
        ({"repetition_penalty": 1.1}, "repetition_penalty"),
    ],
)
def test_a_truncation_the_mask_cannot_reproduce_is_refused(kwargs, match):
    settings = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, **kwargs}
    with pytest.raises(tr.TruncationError, match=match):
        tr.Truncation(**settings)


def test_the_truncation_reads_and_writes_the_campaign_record():
    recorded = {"top_k": 20, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0}
    truncation = tr.Truncation.from_sampling(recorded, temperature=1)
    assert truncation == TRUNCATION and truncation.as_sampling() == recorded
    assert isinstance(truncation.temperature, float) and tr.Truncation(temperature=1.0, top_k=-1, top_p=1.0).top_k == 0
    with pytest.raises(tr.TruncationError, match="lacks"):
        tr.Truncation.from_sampling({"top_k": 20}, temperature=1.0)


class _Head(torch.nn.Module):
    def __init__(self, returns_tuple: bool = False) -> None:
        super().__init__()
        self.returns_tuple = returns_tuple

    def forward(self, logits):
        return (logits,) if self.returns_tuple else logits * 1.0


def test_the_hook_masks_only_while_active():
    torch.manual_seed(2)
    logits = torch.randn(2, 3, 40)
    head = _Head()
    mask = tr.TruncationMask(TRUNCATION).attach(head)
    assert torch.equal(head(logits), logits)
    with mask.active():
        with mask.active():  # nested (a logps pass inside another context) stays active
            assert mask.is_active
        assert torch.equal(head(logits), tr.mask_logits(logits, TRUNCATION))
    assert not mask.is_active and torch.equal(head(logits), logits)
    with pytest.raises(tr.TruncationError, match="already attached"):
        mask.attach(head)
    mask.detach()
    mask.detach()
    with mask.active():
        assert torch.equal(head(logits), logits)


def test_the_hook_refuses_a_module_that_returns_no_tensor():
    head = _Head(returns_tuple=True)
    mask = tr.TruncationMask(TRUNCATION).attach(head)
    with mask.active(), pytest.raises(tr.TruncationError, match="not a logits tensor"):
        head(torch.randn(1, 2, 30))


def _tiny_gemma(cap: float):
    transformers = pytest.importorskip("transformers")
    from tests.tiny_gemma import gemma_config

    torch.manual_seed(5)
    model = transformers.Gemma4ForConditionalGeneration(gemma_config(final_logit_softcapping=cap))
    with torch.no_grad():  # logits large enough that the cap reshapes the distribution
        model.lm_head.weight.mul_(40.0)
    return model.eval()


def test_a_soft_capping_model_is_masked_after_its_cap():
    peft = pytest.importorskip("peft")
    model = _tiny_gemma(cap=2.0)
    ids = torch.tensor([[3, 17, 42, 99, 5]])
    capped = model(input_ids=ids).logits
    raw = model.lm_head(model.model(input_ids=ids).last_hidden_state)
    assert not torch.allclose(capped, raw) and capped.abs().max() <= 2.0
    truncation = tr.Truncation(temperature=1.0, top_k=5, top_p=0.9)
    # under the PEFT wrapper the mask sits on the wrapped model, whose output logits are the capped ones
    wrapped = peft.get_peft_model(model, peft.LoraConfig(r=2, target_modules=r".*language_model.*\.q_proj"))
    assert tr.logits_module(wrapped) is model and tr.logits_module(model) is model
    mask = tr.TruncationMask(truncation).attach(tr.logits_module(wrapped))
    with torch.no_grad(), mask.active():
        masked = wrapped(input_ids=ids).logits
    expected = tr.mask_logits(capped, truncation)
    assert torch.equal(masked, expected) and int((masked == tr.TRUNCATED_LOGIT).sum()) > 0
    # on the head's output the kept set would be the uncapped distribution's, and the cap would squash the mask
    assert not torch.equal(tr.keep_mask(raw, truncation), tr.keep_mask(capped, truncation))
    assert torch.tanh(torch.tensor(tr.TRUNCATED_LOGIT / 2.0)) * 2.0 == -2.0
    mask.detach()


def test_a_model_without_a_cap_is_masked_at_its_head():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                      num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    model = transformers.Qwen3ForCausalLM(config)
    assert tr.logits_module(model) is model.lm_head
    import types

    bare = types.SimpleNamespace(config=types.SimpleNamespace(final_logit_softcapping=None),
                                 get_output_embeddings=lambda: "the head")
    assert tr.logits_module(bare) == "the head"
