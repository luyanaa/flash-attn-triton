"""
Main interfaces for the Triton-based Flash Attention implementation.
"""

import torch
import math
from typing import Optional, Tuple, Union
from .triton_kernel.attention_kernel import attention, attention_varlen

BLOCK_SIZE = 64




def _attention_with_probs(q, k, v, dropout_p, softmax_scale, causal,
                          window_size, softcap, alibi_slopes, attn_bias=None,
                          causal_offset=0):
    """Materialize probabilities for the dense reference/testing API option."""
    batch, q_len, q_heads, head_dim = q.shape
    k_len, kv_heads = k.shape[1], k.shape[2]
    if q_heads != kv_heads:
        repeats = q_heads // kv_heads
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
    scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    q_pos = torch.arange(q_len, device=q.device) + causal_offset
    k_pos = torch.arange(k_len, device=q.device)
    keep = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device)
    if causal:
        keep &= q_pos[:, None] >= k_pos[None, :]
    if window_size[0] >= 0:
        keep &= k_pos[None, :] >= q_pos[:, None] - window_size[0]
    if window_size[1] >= 0:
        keep &= k_pos[None, :] <= q_pos[:, None] + window_size[1]
    if alibi_slopes is not None:
        slopes = alibi_slopes if alibi_slopes.ndim == 2 else alibi_slopes[None, :]
        scores -= slopes[:, :, None, None] * (q_pos[:, None] - k_pos[None, :]).abs()
    if attn_bias is not None:
        scores = scores + attn_bias
    if softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    scores = scores.float().masked_fill(~keep[None, None], float("-inf"))
    probs = torch.nan_to_num(torch.softmax(scores, dim=-1))
    lse = torch.logsumexp(scores, dim=-1)
    if dropout_p:
        probs = torch.nn.functional.dropout(probs, p=dropout_p, training=True)
    out = torch.einsum("bhqk,bkhd->bqhd", probs.to(q.dtype), v)
    return out, lse, probs


def _apply_rotary(x, cos, sin, positions, interleaved):
    """Apply rotary cos/sin tables to a [B, S, H, D] tensor."""
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError("rotary tables exceed the head dimension")
    c = cos[positions].to(dtype=x.dtype).unsqueeze(0).unsqueeze(2)
    s = sin[positions].to(dtype=x.dtype).unsqueeze(0).unsqueeze(2)
    part = x[..., :rotary_dim]
    if interleaved:
        even, odd = part[..., 0::2], part[..., 1::2]
        rotated = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)
    else:
        first, second = part[..., :rotary_dim // 2], part[..., rotary_dim // 2:]
        rotated = torch.cat((first * c - second * s, first * s + second * c), dim=-1)
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False, use_native_gqa=False,
                    attn_bias=None):
    """Flash Attention implementation using Triton kernel.
    
    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention
            with SDPA-compatible upper-left alignment for rectangular inputs.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. For ALiBi attention bias
            added to the score before softcap.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass
            using a slower non-atomic Triton path.
        return_attn_probs: bool. Whether to return testing-only attention probabilities.
        use_native_gqa: bool. Avoid materializing repeated K/V heads and use grouped-head backward.
            
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    # Check unsupported features
    if q.dtype not in (torch.float16, torch.float32):
        raise ValueError("This Triton implementation supports FP16 and FP32 tensors")
    if softcap < 0.0:
        raise ValueError("softcap must be non-negative")
    if dropout_p < 0.0 or dropout_p >= 1.0:
        raise ValueError("dropout_p must be in the range [0, 1)")

    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("All attention tensors must be CUDA tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (batch, seqlen, heads, headdim)")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")


    # Validate input shapes
    batch_size, seq_len_q, n_heads_q, head_dim = q.shape
    key_batch_size, seq_len_k, n_heads_k, key_head_dim = k.shape
    value_batch_size, value_seq_len, value_heads, value_head_dim = v.shape
    if key_batch_size != batch_size or value_batch_size != batch_size:
        raise ValueError("q, k, and v must have the same batch size")
    if value_seq_len != seq_len_k:
        raise ValueError("k and v must have the same sequence length")
    if key_head_dim != head_dim or value_head_dim != head_dim:
        raise ValueError("q, k, and v must have the same head dimension")
    if value_heads != n_heads_k:
        raise ValueError("k and v must have the same number of heads")
    if n_heads_k == 0 or n_heads_q % n_heads_k != 0:
        raise ValueError(
            f"Number of heads in Q ({n_heads_q}) must be divisible by number of heads in K/V ({n_heads_k})"
        )
    if alibi_slopes is not None:
        alibi_slopes = torch.as_tensor(alibi_slopes, device=q.device, dtype=torch.float32)
        if alibi_slopes.ndim == 1:
            if alibi_slopes.shape[0] != n_heads_q:
                raise ValueError("alibi_slopes must have one value per query head")
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(batch_size, -1)
        elif alibi_slopes.shape != (batch_size, n_heads_q):
            raise ValueError("alibi_slopes must have shape (nheads,) or (batch, nheads)")
        alibi_slopes = alibi_slopes.contiguous()
    use_alibi = alibi_slopes is not None and bool(torch.any(alibi_slopes != 0))

    if return_attn_probs:
        return _attention_with_probs(
            q, k, v, dropout_p, softmax_scale, causal, window_size,
            softcap, alibi_slopes, attn_bias,
        )
    if attn_bias is not None:
        return _attention_with_probs(
            q, k, v, dropout_p, softmax_scale, causal, window_size,
            softcap, alibi_slopes, attn_bias,
        )[0]

    # FlashAttention's public API also accepts cross-attention through
    # flash_attn_func.  Keep the equal-length Triton path unchanged, but use
    # the dedicated unequal-length path instead of rejecting valid inputs.
    if seq_len_q != seq_len_k:
        return flash_attn_func_cross(
            q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale,
            causal=causal, use_native_gqa=use_native_gqa, softcap=softcap,
            alibi_slopes=alibi_slopes, window_size=window_size,
            deterministic=deterministic,
        )

    native_gqa = use_native_gqa and not deterministic and not (q.requires_grad or k.requires_grad or v.requires_grad) and n_heads_q != n_heads_k

    # Save original sequence lengths for unpadding later
    original_seq_len_q = seq_len_q
    original_seq_len_k = seq_len_k
    
    # Check if sequence lengths are multiples of 128 and pad if necessary
    pad_q = (BLOCK_SIZE - seq_len_q % BLOCK_SIZE) % BLOCK_SIZE
    pad_k = (BLOCK_SIZE - seq_len_k % BLOCK_SIZE) % BLOCK_SIZE
    
    if pad_q > 0:
        q_padding = torch.zeros(batch_size, pad_q, n_heads_q, head_dim, 
                               dtype=q.dtype, device=q.device)
        q = torch.cat([q, q_padding], dim=1)
        seq_len_q += pad_q
    
    if pad_k > 0:
        k_padding = torch.zeros(batch_size, pad_k, n_heads_k, head_dim, 
                               dtype=k.dtype, device=k.device)
        v_padding = torch.zeros(batch_size, pad_k, n_heads_k, head_dim, 
                               dtype=v.dtype, device=v.device)
        k = torch.cat([k, k_padding], dim=1)
        v = torch.cat([v, v_padding], dim=1)
        seq_len_k += pad_k
    
    # Check if we're dealing with MQA/GQA (not fully supported)
    if n_heads_q != n_heads_k and not native_gqa:
        # Materialize the same grouped-head mapping used by upstream.
        k = k.repeat_interleave(n_heads_q // n_heads_k, dim=2)
        v = v.repeat_interleave(n_heads_q // n_heads_k, dim=2)
    
    needs_grad = torch.is_grad_enabled() and (
        q.requires_grad or k.requires_grad or v.requires_grad
    )
    q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if needs_grad:
        # Dense backward currently assumes contiguous head-major tensors.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
    
    # Compute softmax scale if not provided
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Call the Triton implementation
    out = attention(
        q,
        k,
        v,
        causal,
        softmax_scale,
        dropout_p,
        seq_len_k=original_seq_len_k,
        seq_len_q=original_seq_len_q,
        kv_heads=n_heads_k if native_gqa else n_heads_q,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        window_left=window_size[0],
        window_right=window_size[1],
        use_alibi=use_alibi,
    )
    
    # Transpose the output back to the expected shape [batch, seq_len, heads, head_dim]
    out = out.transpose(1, 2).contiguous()  # Ensure output is contiguous

    # Slice the output to remove padding
    if pad_q > 0:
        out = out[:, :original_seq_len_q, :, :]

    if return_attn_probs:
        return out, None, None
    else:
        return out


def flash_attn_func_cross(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                          use_native_gqa=False, softcap=0.0, alibi_slopes=None,
                          window_size=(-1, -1), deterministic=False,
                          return_attn_probs=False, attn_bias=None, _causal_offset=0):
    """Triton cross-attention for unequal query and KV sequence lengths."""
    if q.dtype not in (torch.float16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("flash_attn_func_cross supports matching FP16 or FP32 tensors")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("All attention tensors must be CUDA tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (batch, seqlen, heads, headdim)")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if k.shape[0] != q.shape[0] or v.shape[:2] != k.shape[:2]:
        raise ValueError("q, k, and v have incompatible batch or sequence dimensions")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same head dimension")
    if q.shape[-1] not in (16, 32, 64, 128, 256):
        raise ValueError("head dimension must be one of 16, 32, 64, 128, or 256")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError("The number of query heads must be divisible by KV heads")
    if dropout_p < 0.0 or dropout_p >= 1.0:
        raise ValueError("dropout_p must be in the range [0, 1)")
    if softcap < 0.0:
        raise ValueError("softcap must be non-negative")

    q_heads, kv_heads = q.shape[2], k.shape[2]
    if q_heads % kv_heads != 0:
        raise ValueError("The number of query heads must be divisible by KV heads")
    if alibi_slopes is not None:
        alibi_slopes = torch.as_tensor(alibi_slopes, device=q.device, dtype=torch.float32)
        if alibi_slopes.ndim == 1:
            if alibi_slopes.shape[0] != q_heads:
                raise ValueError("alibi_slopes must have one value per query head")
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(q.shape[0], -1)
        elif alibi_slopes.shape != (q.shape[0], q_heads):
            raise ValueError("alibi_slopes must have shape (nheads,) or (batch, nheads)")
        alibi_slopes = alibi_slopes.contiguous()
    use_alibi = alibi_slopes is not None and bool(torch.any(alibi_slopes != 0))

    if return_attn_probs:
        return _attention_with_probs(
            q, k, v, dropout_p, softmax_scale, causal, window_size,
            softcap, alibi_slopes, attn_bias, causal_offset=_causal_offset,
        )
    if attn_bias is not None:
        return _attention_with_probs(
            q, k, v, dropout_p, softmax_scale, causal, window_size,
            softcap, alibi_slopes, attn_bias, causal_offset=_causal_offset,
        )[0]

    original_q_len, original_k_len = q.shape[1], k.shape[1]
    target_len = max(original_q_len, original_k_len)
    padded_len = (target_len + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    if original_q_len != padded_len:
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, padded_len - original_q_len))
    if original_k_len != padded_len:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, padded_len - original_k_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, padded_len - original_k_len))

    native_gqa = use_native_gqa and not deterministic and not (q.requires_grad or k.requires_grad or v.requires_grad)
    if q.shape[2] != k.shape[2] and not native_gqa:
        repeats = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
    out = attention(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        causal,
        softmax_scale,
        dropout_p,
        seq_len_k=original_k_len,
        seq_len_q=original_k_len - _causal_offset,
        kv_heads=k.shape[2],
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        window_left=window_size[0],
        window_right=window_size[1],
        use_alibi=use_alibi,
    )
    return out.transpose(1, 2).contiguous()[:, :original_q_len]






def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                              window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False,
                              return_attn_probs=False, use_native_gqa=False, attn_bias=None):
    """Flash Attention for packed QKV using Triton kernel.
    
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention
            with SDPA-compatible upper-left alignment for rectangular inputs.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. For ALiBi attention bias
            added to the score before softcap.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass
            using a slower non-atomic Triton path.
        return_attn_probs: bool. Whether to return attention probabilities (not supported in this
            implementation).
            
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    # Ensure input is contiguous
    qkv = qkv.contiguous() if not qkv.is_contiguous() else qkv
    
    # Unpack qkv
    batch_size, seqlen, _, n_heads, head_dim = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    
    # Call the unpacked version
    return flash_attn_func(
        q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes,
        deterministic, return_attn_probs, use_native_gqa, attn_bias
    )


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False,
                             window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False,
                             return_attn_probs=False, use_native_gqa=False, attn_bias=None):
    """Flash Attention for packed KV using Triton kernel.
    
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        kv: (batch_size, seqlen_k, 2, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention
            materialized for testing.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. For ALiBi attention bias
            added to the score before softcap.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass
            using a slower non-atomic Triton path.
            
    Return:
        out: (batch_size, seqlen_q, nheads, headdim).
    """
    # Ensure KV is contiguous
    kv = kv.contiguous() if not kv.is_contiguous() else kv
    
    # Unpack kv
    k, v = kv.unbind(dim=2)
    
    # Call the unpacked version
    return flash_attn_func(
        q, k, v, dropout_p, softmax_scale, causal, window_size, softcap,
        alibi_slopes, deterministic, return_attn_probs, use_native_gqa, attn_bias
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    rotary_interleaved=True,
    alibi_slopes=None,
):
    """Run attention against, and optionally append to, a KV cache."""
    paged = block_table is not None
    if (rotary_cos is None) != (rotary_sin is None):
        raise ValueError("rotary_cos and rotary_sin must be supplied together")
    if q.ndim != 4 or k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("q and KV caches must be rank-4 tensors")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    if paged:
        if block_table.ndim != 2 or block_table.shape[0] != q.shape[0]:
            raise ValueError("block_table must have shape (batch, max_num_pages)")
        if block_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("block_table must contain integer page indices")
        if torch.any(block_table < 0) or torch.any(block_table >= k_cache.shape[0]):
            raise ValueError("block_table contains an invalid physical page index")
        if cache_leftpad is not None:
            raise ValueError("cache_leftpad cannot be combined with paged KV caches")
        page_size = k_cache.shape[1]
        cache_capacity = block_table.shape[1] * page_size
    else:
        cache_capacity = k_cache.shape[1]
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError("q and KV cache dimensions are incompatible")
    if q.device != k_cache.device or q.device != v_cache.device:
        raise ValueError("q and KV caches must be on the same device")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q and KV caches must have the same dtype")
    if paged and cache_batch_idx is not None:
        raise ValueError("cache_batch_idx cannot be combined with paged KV caches")
    if cache_batch_idx is None:
        cache_rows = torch.arange(q.shape[0], device=q.device, dtype=torch.long)
    else:
        cache_rows = torch.as_tensor(cache_batch_idx, device=q.device, dtype=torch.long).flatten()
        if cache_rows.numel() != q.shape[0] or torch.any(cache_rows < 0) or torch.any(cache_rows >= k_cache.shape[0]):
            raise ValueError("cache_batch_idx must contain valid cache-row indices")
    if cache_leftpad is None:
        leftpads = torch.zeros(q.shape[0], device=q.device, dtype=torch.long)
    else:
        leftpads = torch.as_tensor(cache_leftpad, device=q.device, dtype=torch.long).flatten()
        if leftpads.numel() != q.shape[0] or torch.any(leftpads < 0):
            raise ValueError("cache_leftpad must contain one non-negative value per query batch")
    if k is not None or v is not None:
        if k is None or v is None or k.shape != v.shape:
            raise ValueError("k and v must be supplied together with matching shapes")
        if k.shape[0] != q.shape[0] or k.shape[2:] != k_cache.shape[2:]:
            raise ValueError("new k/v tensors are incompatible with the cache")
        if cache_seqlens is None:
            cache_seqlens = torch.zeros(q.shape[0], dtype=torch.int32, device=q.device)
        lengths = torch.as_tensor(cache_seqlens, device=q.device, dtype=torch.int64).flatten()
        if lengths.numel() != q.shape[0] or torch.any(lengths < 0) or torch.any(lengths + leftpads + k.shape[1] > cache_capacity):
            raise ValueError("new KV tensors do not fit in the cache")
        if rotary_cos is not None:
            positions = torch.arange(k.shape[1], device=q.device)[None, :] + lengths[:, None] + leftpads[:, None]
            k = torch.stack([_apply_rotary(k[i:i + 1], rotary_cos, rotary_sin, positions[i], rotary_interleaved)[0]
                             for i in range(q.shape[0])])
        logical = lengths[:, None] + leftpads[:, None] + torch.arange(k.shape[1], device=q.device)[None, :]
        if paged:
            page = logical // page_size
            offset = logical % page_size
            physical = block_table.gather(1, page).to(torch.long)
            k_cache[physical.reshape(-1), offset.reshape(-1)].copy_(k.reshape(-1, *k.shape[2:]))
            v_cache[physical.reshape(-1), offset.reshape(-1)].copy_(v.reshape(-1, *v.shape[2:]))
        else:
            rows = cache_rows[:, None].expand_as(logical)
            k_cache[rows.reshape(-1), logical.reshape(-1)].copy_(k.reshape(-1, *k.shape[2:]))
            v_cache[rows.reshape(-1), logical.reshape(-1)].copy_(v.reshape(-1, *v.shape[2:]))
        lengths = lengths + k.shape[1]
    else:
        if cache_seqlens is None:
            lengths = torch.full((q.shape[0],), cache_capacity, dtype=torch.int64, device=q.device)
        else:
            lengths = torch.as_tensor(cache_seqlens, device=q.device, dtype=torch.int64).flatten()
        if lengths.numel() != q.shape[0] or torch.any(lengths < 0) or torch.any(lengths + leftpads > cache_capacity):
            raise ValueError("cache_seqlens must contain valid lengths for every batch item")
    outputs = []
    for batch_idx, length in enumerate(lengths.tolist()):
        batch_alibi = alibi_slopes
        if alibi_slopes is not None and alibi_slopes.ndim == 2:
            batch_alibi = alibi_slopes[batch_idx:batch_idx + 1]
        row = cache_rows[batch_idx].item()
        start = leftpads[batch_idx].item()
        q_batch = q[batch_idx:batch_idx + 1]
        if paged:
            page_ids = block_table[batch_idx, :((start + length + page_size - 1) // page_size)].to(torch.long)
            k_batch = k_cache[page_ids].reshape(-1, k_cache.shape[2], k_cache.shape[3])[start:start + length].unsqueeze(0)
            v_batch = v_cache[page_ids].reshape(-1, v_cache.shape[2], v_cache.shape[3])[start:start + length].unsqueeze(0)
        else:
            k_batch = k_cache[row:row + 1, start:start + length]
            v_batch = v_cache[row:row + 1, start:start + length]
        if rotary_cos is not None:
            positions = torch.arange(q.shape[1], device=q.device) + length - q.shape[1] + start
            q_batch = _apply_rotary(q_batch, rotary_cos, rotary_sin, positions, rotary_interleaved)
        outputs.append(flash_attn_func_cross(
            q_batch, k_batch, v_batch,
            softmax_scale=softmax_scale, causal=causal, use_native_gqa=True,
            window_size=window_size, alibi_slopes=batch_alibi,
            _causal_offset=(length - q.shape[1]) if causal else 0,
        ))
    return torch.cat(outputs, dim=0) if outputs else q.new_empty(q.shape)

def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                           max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False,
                           window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False,
                           return_attn_probs=False, block_table=None,
                           seqused_q=None, seqused_k=None, **kwargs):
    """Triton attention for packed variable-length sequences."""
    if block_table is not None:
        if k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
            raise ValueError("paged varlen k and v must have shape (pages, page_size, heads, dim)")
        if block_table.ndim != 2 or block_table.shape[0] != cu_seqlens_k.numel() - 1:
            raise ValueError("block_table must have shape (batch, max_num_pages)")
        page_size = k.shape[1]
        page_parts, value_parts = [], []
        for batch_idx in range(block_table.shape[0]):
            start, end = cu_seqlens_k[batch_idx:batch_idx + 2].tolist()
            length = end - start
            page_count = (length + page_size - 1) // page_size
            page_ids = block_table[batch_idx, :page_count].to(torch.long)
            if torch.any(page_ids < 0) or torch.any(page_ids >= k.shape[0]):
                raise ValueError("block_table contains an invalid physical page index")
            page_parts.append(k[page_ids].reshape(-1, k.shape[2], k.shape[3])[:length])
            value_parts.append(v[page_ids].reshape(-1, v.shape[2], v.shape[3])[:length])
        k = torch.cat(page_parts, dim=0) if page_parts else k.new_empty((0, k.shape[2], k.shape[3]))
        v = torch.cat(value_parts, dim=0) if value_parts else v.new_empty((0, v.shape[2], v.shape[3]))
    batch_size = cu_seqlens_q.numel() - 1
    if seqused_q is not None:
        seqused_q = torch.as_tensor(seqused_q, device=q.device, dtype=torch.int64).flatten()
        if seqused_q.numel() != batch_size:
            raise ValueError("seqused_q must contain one value per sequence")
        q_parts, q_lengths = [], []
        for batch_idx in range(batch_size):
            start, end = cu_seqlens_q[batch_idx:batch_idx + 2].tolist()
            used = int(seqused_q[batch_idx].item())
            if used < 0 or used > end - start:
                raise ValueError("seqused_q exceeds the packed query sequence")
            q_parts.append(q[start:start + used])
            q_lengths.append(used)
        q = torch.cat(q_parts, dim=0) if q_parts else q[:0]
        cu_seqlens_q = torch.tensor([0] + list(torch.tensor(q_lengths).cumsum(0).tolist()), device=q.device, dtype=torch.int32)
    if seqused_k is not None:
        seqused_k = torch.as_tensor(seqused_k, device=k.device, dtype=torch.int64).flatten()
        if seqused_k.numel() != batch_size:
            raise ValueError("seqused_k must contain one value per sequence")
        k_parts, v_parts, k_lengths = [], [], []
        for batch_idx in range(batch_size):
            start, end = cu_seqlens_k[batch_idx:batch_idx + 2].tolist()
            used = int(seqused_k[batch_idx].item())
            if used < 0 or used > end - start:
                raise ValueError("seqused_k exceeds the packed KV sequence")
            k_parts.append(k[start:start + used])
            v_parts.append(v[start:start + used])
            k_lengths.append(used)
        k = torch.cat(k_parts, dim=0) if k_parts else k[:0]
        v = torch.cat(v_parts, dim=0) if v_parts else v[:0]
        cu_seqlens_k = torch.tensor([0] + list(torch.tensor(k_lengths).cumsum(0).tolist()), device=k.device, dtype=torch.int32)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or k.shape != v.shape:
        raise ValueError("packed q, k, and v must be rank-3 tensors with matching KV shapes")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cu_seqlens arrays must be one-dimensional")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch size")
    if cu_seqlens_q.numel() == 0 or cu_seqlens_q[0].item() != 0 or cu_seqlens_k[0].item() != 0:
        raise ValueError("cu_seqlens arrays must start at zero and contain a batch boundary")
    if (torch.any(cu_seqlens_q[1:] < cu_seqlens_q[:-1]) or
            torch.any(cu_seqlens_k[1:] < cu_seqlens_k[:-1])):
        raise ValueError("cu_seqlens arrays must be monotonically non-decreasing")
    if cu_seqlens_q[-1].item() > q.shape[0] or cu_seqlens_k[-1].item() > k.shape[0]:
        raise ValueError("cu_seqlens values exceed the packed tensor sizes")
    if q.dtype not in (torch.float16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("flash_attn_varlen_func supports matching FP16 or FP32 tensors")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("All attention tensors must be CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same head dimension")
    if q.shape[-1] not in (16, 32, 64, 128, 256):
        raise ValueError("head dimension must be one of 16, 32, 64, 128, or 256")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("The number of query heads must be divisible by KV heads")
    if dropout_p < 0.0 or dropout_p >= 1.0:
        raise ValueError("dropout_p must be in the range [0, 1)")
    if softcap < 0.0:
        raise ValueError("softcap must be non-negative")
    if cu_seqlens_q.device != q.device or cu_seqlens_k.device != q.device:
        raise ValueError("cu_seqlens arrays must be on the same CUDA device as q, k, and v")
    cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32).contiguous()
    cu_seqlens_k = cu_seqlens_k.to(dtype=torch.int32).contiguous()
    if max_seqlen_q < 0 or max_seqlen_k < 0:
        raise ValueError("max sequence lengths must be non-negative")
    if cu_seqlens_q[-1].item() != q.shape[0] or cu_seqlens_k[-1].item() != k.shape[0]:
        raise ValueError("cu_seqlens arrays must cover the packed q and KV tensors")
    if cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens arrays must describe at least one sequence")
    if torch.any(cu_seqlens_q[1:] < cu_seqlens_q[:-1]) or torch.any(cu_seqlens_k[1:] < cu_seqlens_k[:-1]):
        raise ValueError("cu_seqlens arrays must be monotonically non-decreasing")
    actual_max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    actual_max_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())
    if max_seqlen_q < actual_max_q or max_seqlen_k < actual_max_k:
        raise ValueError("max sequence lengths must cover every packed sequence")
    if max_seqlen_q == 0 or max_seqlen_k == 0:
        return q.new_zeros((q.shape[0], q.shape[1], q.shape[2]))
    provided_alibi = alibi_slopes is not None
    if alibi_slopes is None:
        alibi_slopes = q.new_zeros((cu_seqlens_q.numel() - 1, q.shape[1]), dtype=torch.float32)
    else:
        alibi_slopes = torch.as_tensor(alibi_slopes, device=q.device, dtype=torch.float32)
        if alibi_slopes.ndim == 1:
            if alibi_slopes.shape[0] != q.shape[1]:
                raise ValueError("alibi_slopes must have one value per query head")
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(cu_seqlens_q.numel() - 1, -1)
        elif alibi_slopes.shape != (cu_seqlens_q.numel() - 1, q.shape[1]):
            raise ValueError("alibi_slopes must have shape (nheads,) or (batch, nheads)")
        alibi_slopes = alibi_slopes.contiguous()
    use_alibi = provided_alibi and bool(torch.any(alibi_slopes != 0))
    if return_attn_probs:
        outputs = []
        lses = q.new_full((batch_size, q.shape[1], max_seqlen_q), float("-inf"), dtype=torch.float32)
        probabilities = q.new_zeros((batch_size, q.shape[1], max_seqlen_q, max_seqlen_k), dtype=torch.float32)
        for batch_idx in range(batch_size):
            q_start, q_end = cu_seqlens_q[batch_idx:batch_idx + 2].tolist()
            k_start, k_end = cu_seqlens_k[batch_idx:batch_idx + 2].tolist()
            out, lse, probs = _attention_with_probs(
                q[q_start:q_end].unsqueeze(0), k[k_start:k_end].unsqueeze(0),
                v[k_start:k_end].unsqueeze(0), dropout_p, softmax_scale, causal,
                window_size, softcap, alibi_slopes[batch_idx:batch_idx + 1],
            )
            outputs.append(out.squeeze(0))
            q_len = q_end - q_start
            k_len = k_end - k_start
            lses[batch_idx, :, :q_len] = lse[0]
            probabilities[batch_idx, :, :q_len, :k_len] = probs[0]
        return torch.cat(outputs, dim=0), lses, probabilities
    if deterministic:
        outputs = []
        for batch_idx in range(batch_size):
            q_start, q_end = cu_seqlens_q[batch_idx:batch_idx + 2].tolist()
            k_start, k_end = cu_seqlens_k[batch_idx:batch_idx + 2].tolist()
            q_part = q[q_start:q_end]
            k_part = k[k_start:k_end]
            v_part = v[k_start:k_end]
            if q_part.shape[0] == 0:
                outputs.append(q_part.new_empty((0, q.shape[1], q.shape[2])))
                continue
            if k_part.shape[0] == 0:
                outputs.append(q_part.new_zeros((q_part.shape[0], q.shape[1], q.shape[2])))
                continue
            outputs.append(flash_attn_func_cross(
                q_part.unsqueeze(0), k_part.unsqueeze(0), v_part.unsqueeze(0),
                dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
                use_native_gqa=False, softcap=softcap,
                alibi_slopes=alibi_slopes[batch_idx:batch_idx + 1],
                window_size=window_size, deterministic=True,
            ).squeeze(0))
        return torch.cat(outputs, dim=0)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    return attention_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        causal=causal, sm_scale=softmax_scale, dropout_p=dropout_p,
        softcap=softcap, alibi_slopes=alibi_slopes,
        window_left=window_size[0], window_right=window_size[1], use_alibi=use_alibi,
    )

def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                                    max_seqlen_k, dropout_p=0.0, softmax_scale=None,
                                    causal=False, **kwargs):
    k, v = kv.unbind(dim=1)
    return flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, **kwargs
    )


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0,
                                     softmax_scale=None, causal=False, **kwargs):
    q, k, v = qkv.unbind(dim=1)
    return flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, **kwargs
    )


class FlashSelfAttention(torch.nn.Module):
    """FlashAttention-compatible self-attention module."""

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 window_size=(-1, -1), alibi_slopes=None, deterministic=False):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = torch.nn.Dropout(attention_dropout)
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.deterministic = deterministic

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        dropout_p = self.drop.p if self.training else 0.0
        causal = self.causal if causal is None else causal
        if cu_seqlens is None:
            return flash_attn_qkvpacked_func(
                qkv, dropout_p, self.softmax_scale, causal, self.window_size,
                0.0, self.alibi_slopes, self.deterministic
            )
        return flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, dropout_p, self.softmax_scale, causal,
            window_size=self.window_size, alibi_slopes=self.alibi_slopes,
            deterministic=self.deterministic
        )


class FlashCrossAttention(torch.nn.Module):
    """FlashAttention-compatible cross-attention module."""

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 window_size=(-1, -1), alibi_slopes=None, deterministic=False):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = torch.nn.Dropout(attention_dropout)
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.deterministic = deterministic

    def forward(self, q, kv, causal=None, cu_seqlens=None, max_seqlen=None,
                cu_seqlens_k=None, max_seqlen_k=None):
        dropout_p = self.drop.p if self.training else 0.0
        causal = self.causal if causal is None else causal
        k, v = kv.unbind(dim=2)
        if cu_seqlens is None:
            return flash_attn_func_cross(
                q, k, v, dropout_p, self.softmax_scale, causal,
                use_native_gqa=not self.training,
                softcap=0.0,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
            )
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens_k, max_seqlen, max_seqlen_k,
            dropout_p, self.softmax_scale, causal, window_size=self.window_size,
            alibi_slopes=self.alibi_slopes, deterministic=self.deterministic
        )


class FlashAttention(torch.nn.Module):
    """
    Module implementation of Flash Attention using Triton kernel.
    
    Note: This is a limited implementation that doesn't support all features of Flash Attention 2.
    """
    def __init__(self, attention_dropout=0.0, softmax_scale=None):
        super().__init__()
        self.dropout_p = attention_dropout
        self.softmax_scale = softmax_scale
    
    def forward(self, q, k, v, causal=False, attn_bias=None):
        """
        Forward pass.
        
        Args:
            q: Query tensor of shape (batch_size, seqlen_q, num_heads, head_dim)
            k: Key tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
            v: Value tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
            causal: If True, applies causal masking
            attn_bias: Optional dense bias broadcastable to (batch, heads, seqlen_q, seqlen_k)
            
        Returns:
            output: Attention output of shape (batch_size, seqlen_q, num_heads, head_dim)
        """
        return flash_attn_func(
            q, k, v, 
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            attn_bias=attn_bias,
        )

    
