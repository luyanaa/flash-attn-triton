"""Compare the Triton implementation against PyTorch SDPA.

Run with::

    python -m pytest tests/test_compare_implementations.py
"""

import pytest
import torch
import torch.nn.functional as F

from flash_attn_triton import flash_attn_func, flash_attn_func_cross
from flash_attn_triton import flash_attn_qkvpacked_func, flash_attn_varlen_func
from flash_attn_triton.triton_kernel.attention_kernel_int8 import attention_int8


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Triton attention"
)


DEVICE = torch.device("cuda")
ATOL = 2e-2
RTOL = 2e-2
# QK^T runs in INT8, so the output and every gradient derived from it carry
# quantization error (measured up to ~5e-2 against fp16 SDPA at head dim 64).
INT8_ATOL = 5e-2
INT8_RTOL = 2e-2


def _sdpa(q, k, v, causal=False, softmax_scale=None, attn_mask=None):
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=causal,
        scale=softmax_scale,
    )
    return output.transpose(1, 2)


def _inputs(batch, q_len, k_len, heads=2, head_dim=16):
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    q = torch.randn(batch, q_len, heads, head_dim, device=DEVICE,
                    dtype=torch.float16, generator=generator)
    k = torch.randn(batch, k_len, heads, head_dim, device=DEVICE,
                    dtype=torch.float16, generator=generator)
    v = torch.randn(batch, k_len, heads, head_dim, device=DEVICE,
                    dtype=torch.float16, generator=generator)
    return q, k, v


def _assert_output_and_gradients_match(implementation, q, k, v, *, causal,
                                       softmax_scale=None, atol=ATOL, rtol=RTOL):
    q_impl = q.detach().clone().requires_grad_()
    k_impl = k.detach().clone().requires_grad_()
    v_impl = v.detach().clone().requires_grad_()
    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()

    output_impl = implementation(
        q_impl, k_impl, v_impl, causal=causal, softmax_scale=softmax_scale
    )
    output_ref = _sdpa(
        q_ref, k_ref, v_ref, causal=causal, softmax_scale=softmax_scale
    )
    torch.testing.assert_close(output_impl, output_ref, atol=atol, rtol=rtol)

    generator = torch.Generator(device=DEVICE).manual_seed(7)
    grad_output = torch.randn(output_impl.shape, device=DEVICE,
                              dtype=output_impl.dtype, generator=generator)
    gradients_impl = torch.autograd.grad(
        (output_impl * grad_output).sum(), (q_impl, k_impl, v_impl)
    )
    gradients_ref = torch.autograd.grad(
        (output_ref * grad_output).sum(), (q_ref, k_ref, v_ref)
    )
    for gradient_impl, gradient_ref in zip(gradients_impl, gradients_ref):
        torch.testing.assert_close(gradient_impl, gradient_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("causal", [False, True])
def test_equal_length_attention_matches_sdpa(causal):
    q, k, v = _inputs(batch=2, q_len=17, k_len=17)
    _assert_output_and_gradients_match(
        flash_attn_func, q, k, v, causal=causal, softmax_scale=0.125
    )


@pytest.mark.parametrize("causal", [False, True])
def test_window_attention_matches_sdpa_with_gradients(causal):
    q, k, v = _inputs(batch=1, q_len=17, k_len=17)
    window = (3, 2)
    positions = torch.arange(17, device=DEVICE)
    attn_mask = torch.zeros((17, 17), device=DEVICE, dtype=q.dtype)
    valid = (
        (positions[None, :] >= positions[:, None] - window[0])
        & (positions[None, :] <= positions[:, None] + window[1])
    )
    if causal:
        valid &= positions[None, :] <= positions[:, None]
    attn_mask.masked_fill_(~valid, float("-inf"))

    q_impl = q.detach().clone().requires_grad_()
    k_impl = k.detach().clone().requires_grad_()
    v_impl = v.detach().clone().requires_grad_()
    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()
    output_impl = flash_attn_func(
        q_impl, k_impl, v_impl, causal=causal, window_size=window
    )
    output_ref = _sdpa(q_ref, k_ref, v_ref, attn_mask=attn_mask)
    torch.testing.assert_close(output_impl, output_ref, atol=ATOL, rtol=RTOL)

    generator = torch.Generator(device=DEVICE).manual_seed(7)
    grad_output = torch.randn(
        output_impl.shape, device=DEVICE, dtype=output_impl.dtype, generator=generator
    )
    gradients_impl = torch.autograd.grad(
        (output_impl * grad_output).sum(), (q_impl, k_impl, v_impl)
    )
    gradients_ref = torch.autograd.grad(
        (output_ref * grad_output).sum(), (q_ref, k_ref, v_ref)
    )
    for gradient_impl, gradient_ref in zip(gradients_impl, gradients_ref):
        torch.testing.assert_close(gradient_impl, gradient_ref, atol=ATOL, rtol=RTOL)

@pytest.mark.parametrize("causal", [False, True])
def test_rectangular_attention_matches_sdpa(causal):
    q, k, v = _inputs(batch=2, q_len=7, k_len=13)
    _assert_output_and_gradients_match(
        flash_attn_func_cross, q, k, v, causal=causal
    )


def test_varlen_attention_matches_sdpa():
    q_parts = [_inputs(1, q_len, q_len + 1)[0].squeeze(0) for q_len in (3, 5)]
    kv_parts = [_inputs(1, q_len, k_len)[1:] for q_len, k_len in ((3, 4), (5, 6))]
    k_parts = [part[0].squeeze(0) for part in kv_parts]
    v_parts = [part[1].squeeze(0) for part in kv_parts]
    q = torch.cat(q_parts, dim=0)
    k = torch.cat(k_parts, dim=0)
    v = torch.cat(v_parts, dim=0)
    cu_q = torch.tensor([0, 3, 8], device=DEVICE, dtype=torch.int32)
    cu_k = torch.tensor([0, 4, 10], device=DEVICE, dtype=torch.int32)

    output = flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, max_seqlen_q=5, max_seqlen_k=6, causal=True
    )
    expected = torch.cat([
        _sdpa(q_part.unsqueeze(0), k_part.unsqueeze(0), v_part.unsqueeze(0), causal=True)
        .squeeze(0)
        for q_part, k_part, v_part in zip(q_parts, k_parts, v_parts)
    ])
    torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)

def test_dropout_keeps_softmax_normalization():
    q = torch.zeros(1, 1, 1, 16, device=DEVICE, dtype=torch.float16)
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    output = flash_attn_func(q, k, v, dropout_p=0.5)
    kept = torch.isclose(output, torch.full_like(output, 2.0), atol=1e-3, rtol=0)
    dropped = torch.isclose(output, torch.zeros_like(output), atol=1e-3, rtol=0)
    assert torch.all(kept | dropped)


def test_cross_attention_bias_matches_sdpa():
    q, k, v = _inputs(batch=2, q_len=7, k_len=13)
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    attn_bias = torch.randn(2, 2, 7, 13, device=DEVICE,
                            dtype=torch.float16, generator=generator) * 0.1
    output = flash_attn_func_cross(q, k, v, attn_bias=attn_bias)
    expected = _sdpa(q, k, v, attn_mask=attn_bias)
    torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)


def test_non_contiguous_inputs_match_sdpa():
    # Slices of a packed buffer, i.e. the geometry flash_attn_qkvpacked_func
    # hands over: the head-major token stride has a 3x gap. The wrapper used to
    # normalise contiguity only when gradients were required, so inference on
    # input like this silently returned the wrong output.
    q, k, v = _inputs(batch=2, q_len=128, k_len=128)
    qs, ks, vs = torch.stack((q, k, v), dim=2).unbind(dim=2)
    assert not qs.is_contiguous() and not ks.is_contiguous() and not vs.is_contiguous()
    output = flash_attn_func(qs, ks, vs)
    expected = _sdpa(q, k, v)
    torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)


def test_qkvpacked_matches_sdpa():
    # unpacking the packed layout yields strided slices, which is the shape of
    # input that exposed the missing contiguity normalisation.
    q, k, v = _inputs(batch=2, q_len=128, k_len=128)
    qkv = torch.stack((q, k, v), dim=2)
    output = flash_attn_qkvpacked_func(qkv, causal=True)
    expected = _sdpa(q, k, v, causal=True)
    torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)


def test_alibi_slopes_matches_sdpa_bias():
    q, k, v = _inputs(batch=2, q_len=17, k_len=17)
    slopes = torch.randn(2, 2, device=DEVICE, dtype=torch.float32).abs() * 0.1
    output = flash_attn_func(q, k, v, causal=False, alibi_slopes=slopes)
    q_pos = torch.arange(17, device=DEVICE)[:, None]
    k_pos = torch.arange(17, device=DEVICE)[None, :]
    bias = (-slopes[:, :, None, None] * (q_pos - k_pos).abs()).half()
    expected = _sdpa(q, k, v, attn_mask=bias)
    torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)


def _attention_int8_like_sdpa(q, k, v, *, causal, softmax_scale):
    """attention_int8 takes [B, H, S, D]; the shared helpers use [B, S, H, D]."""
    q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    output = attention_int8(q, k, v, causal=causal, sm_scale=softmax_scale)
    return output.transpose(1, 2)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 5)),
    reason="the INT8 attention kernel targets Turing (sm_75)",
)
@pytest.mark.parametrize("causal", [False, True])
def test_int8_attention_matches_sdpa_with_gradients(causal):
    # Two query blocks and two key blocks per head. The backward walks both axes
    # by block, so a dropped block offset leaves the gradients finite but wrong:
    # the later blocks come out short and the first block absorbs every step.
    q, k, v = _inputs(batch=2, q_len=128, k_len=128, heads=2, head_dim=64)
    _assert_output_and_gradients_match(
        _attention_int8_like_sdpa, q, k, v, causal=causal, softmax_scale=0.125,
        atol=INT8_ATOL, rtol=INT8_RTOL,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
