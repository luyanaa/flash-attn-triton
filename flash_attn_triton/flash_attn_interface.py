"""
Main interfaces for the Triton-based Flash Attention implementation.
"""

import torch
import math
from typing import Optional, Tuple, Union

# Import the attention function from the Triton kernel implementation
from .triton_kernel.attention_kernel import attention

BLOCK_SIZE = 128

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                    softcap=0.0,
                    return_attn_probs=False):
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
            (not supported in this implementation).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. For ALiBi attention bias
            (not supported in this implementation).
        deterministic: bool. Whether to use the deterministic implementation of the backward pass
            (not supported in this implementation).
        return_attn_probs: bool. Whether to return attention probabilities (not supported in this
            implementation).
            
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    # Check unsupported features
    if q.dtype not in (torch.float16, torch.float32):
        raise ValueError("This Triton implementation only supports float16 and float32 dtypes")
    if window_size != (-1, -1):
        raise ValueError("Warning: sliding window attention is not supported in this Triton implementation")
    if alibi_slopes is not None:
        raise ValueError("Warning: ALiBi is not supported in this Triton implementation")
    if softcap != 0.0:
        raise ValueError("Warning: Softcap is not supported in this Triton implementation")
    if deterministic:
        print("Warning: deterministic backward pass is not built into this Triton implementation")
    if return_attn_probs:
        print("Warning: returning attention probabilities is not built into this Triton implementation")
        
    # Ensure tensors are contiguous
    q = q.contiguous() if not q.is_contiguous() else q
    k = k.contiguous() if not k.is_contiguous() else k
    v = v.contiguous() if not v.is_contiguous() else v
    
    # Validate input shapes
    batch_size, seq_len_q, n_heads_q, head_dim = q.shape
    _, seq_len_k, n_heads_k, _ = k.shape

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
    if n_heads_q != n_heads_k:
        if n_heads_q % n_heads_k != 0:
            raise ValueError(
                f"Number of heads in Q ({n_heads_q}) must be divisible by number of heads in K/V ({n_heads_k})"
            )
        print("Warning: MQA/GQA will use simple reshaping which may not match Flash Attention's implementation")
        # Simple implementation: repeat k and v
        k = k.repeat_interleave(n_heads_q // n_heads_k, dim=2)
        v = v.repeat_interleave(n_heads_q // n_heads_k, dim=2)
    
    # Transpose q, k, v from [batch, seq_len, heads, head_dim] to [batch, heads, seq_len, head_dim]
    # which is what the Triton implementation expects
    q = q.transpose(1, 2).contiguous()  # Ensure contiguity after transpose
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    # Compute softmax scale if not provided
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Call the Triton implementation
    out = attention(q, k, v, causal, softmax_scale, dropout_p, seq_len_k=original_seq_len_k)
    
    # Transpose the output back to the expected shape [batch, seq_len, heads, head_dim]
    out = out.transpose(1, 2).contiguous()  # Ensure output is contiguous

    # Slice the output to remove padding
    if pad_q > 0:
        out = out[:, :original_seq_len_q, :, :]

    if return_attn_probs:
        return out, None, None
    else:
        return out
    

def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                              window_size=(-1, -1), alibi_slopes=None, deterministic=False, softcap=0.0,
                              return_attn_probs=False):
    """Flash Attention for packed QKV using Triton kernel.
    
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention
            (not supported in this implementation).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. For ALiBi attention bias
            (not supported in this implementation).
        deterministic: bool. Whether to use the deterministic implementation of the backward pass
            (not supported in this implementation).
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
        q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic, softcap, return_attn_probs
    )


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False,
                             window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                             softcap=0.0, return_attn_probs=False):
    """Flash Attention for packed KV using Triton kernel.
    
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        kv: (batch_size, seqlen_k, 2, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention
            (not supported in this implementation).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. For ALiBi attention bias
            (not supported in this implementation).
        deterministic: bool. Whether to use the deterministic implementation of the backward pass
            (not supported in this implementation).
            
    Return:
        out: (batch_size, seqlen_q, nheads, headdim).
    """
    # Ensure KV is contiguous
    kv = kv.contiguous() if not kv.is_contiguous() else kv
    
    # Unpack kv
    k, v = kv.unbind(dim=2)
    
    # Call the unpacked version
    return flash_attn_func(
        q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic, softcap, return_attn_probs
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
    raise NotImplementedError

def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError


def flash_attn_varlen_kvpacked_func(*args, **kwargs):
    raise NotImplementedError


def flash_attn_varlen_qkvpacked_func(*args, **kwargs):
    raise NotImplementedError


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
            attn_bias: Optional attention bias (not supported in this implementation)
            
        Returns:
            output: Attention output of shape (batch_size, seqlen_q, num_heads, head_dim)
        """
        if attn_bias is not None:
            print("Warning: attention bias is not supported in this Triton implementation")
        
        return flash_attn_func(
            q, k, v, 
            dropout_p=self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal=causal
        )

    
