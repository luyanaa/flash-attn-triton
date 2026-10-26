"""
SageBwd-inspired INT8 QK Flash Attention for Turing (sm_75).

The forward path quantizes K per key row in a prepass and Q per query row in
the attention kernel. Only QK^T uses INT8; it accumulates to INT32 and is
rescaled before the FP32 online softmax. PV remains FP16 with FP32
accumulation. Backward reuses the dense path: dot accumulators are FP32, and
dS is cast to the input element type before dQ/dK accumulation.

This entry point is explicitly gated to compute capability (7, 5) and has not
been validated on other architectures. Volta (sm_70) lacks INT8 Tensor Cores
and must use the FP16 path.

Inputs require equal contiguous Q/K/V sequences, sequence lengths divisible
by 64, and a power-of-two head dim >= 32. Window and ALiBi variants are not
implemented in this entry point.
"""
import math

import torch
import triton
import triton.language as tl

from .attention_kernel import _dense_backward


@triton.jit
def _round_i8(x):
    """Round to nearest then cast to int8 (Triton 3.2 has no tl.math.round)."""
    return tl.where(x >= 0, tl.math.floor(x + 0.5), tl.math.ceil(x - 0.5)).to(tl.int8)


@triton.jit
def _quantize_rows(x):
    """Per-row INT8 quantization. x: [BLOCK_M, D] -> (x_hat int8, s [BLOCK_M])."""
    amax = tl.max(tl.abs(x), axis=1)
    s = tl.maximum(amax, 1.0) / 127.0
    x_hat = _round_i8(x / s[:, None])
    return x_hat, s


@triton.jit
def _quantize_cols(x):
    """Per-column INT8 quantization. x: [D, BLOCK_N] -> (x_hat int8, s [BLOCK_N])."""
    amax = tl.max(tl.abs(x), axis=0)
    s = tl.maximum(amax, 1.0) / 127.0
    x_hat = _round_i8(x / s[None, :])
    return x_hat, s


@triton.jit
def _quantize_block(x):
    """Single per-block INT8 scale (factors out of any dot reduction)."""
    amax = tl.max(tl.abs(x))
    s = tl.maximum(amax, 1.0) / 127.0
    x_hat = _round_i8(x / s)
    return x_hat, s
@triton.jit
def _quantize_k_rows(K, K8, SK, N_CTX, HEAD_DIM: tl.constexpr, BLOCK_R: tl.constexpr):
    """Per-key-token INT8 quantization of K, hoisted out of the attention loop.

    K is re-read by every query block, so quantizing it inside the inner loop
    repeats the same work S/BLOCK_M times: on Turing that costs ~4x more than the
    INT8 mma it feeds (measured), which is why the in-loop form is a net loss.
    """
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, HEAD_DIM)
    k_base = K + bh.to(tl.int64) * N_CTX * HEAD_DIM
    k8_base = K8 + bh.to(tl.int64) * N_CTX * HEAD_DIM
    k = tl.load(k_base + offs_r[:, None] * HEAD_DIM + offs_d[None, :],
                mask=offs_r[:, None] < N_CTX, other=0.0)
    amax = tl.max(tl.abs(k), axis=1)
    s = tl.maximum(amax, 1.0) / 127.0
    k_hat = _round_i8(k * (1.0 / s)[:, None])
    tl.store(k8_base + offs_r[:, None] * HEAD_DIM + offs_d[None, :], k_hat,
             mask=offs_r[:, None] < N_CTX)
    tl.store(SK + bh.to(tl.int64) * N_CTX + offs_r, s, mask=offs_r < N_CTX)


@triton.jit
def _attn_fwd_int8_inner(acc, l_i, m_i, q, s_q,  #
                         K8_block_ptr, s_k_ptrs, V_block_ptr,  #
                         start_m, qk_scale,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                         STAGE: tl.constexpr, offs_m, offs_n,  #
                         N_CTX: tl.constexpr, SEQ_LEN_K: tl.constexpr, SEQ_LEN_Q: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    K8_block_ptr = tl.advance(K8_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # Triton 3.2 cannot feed an int8 kernel parameter into tl.dot, so quantize q
    # here from its fp16 form (once per query block).
    q_hat = _round_i8(q / s_q[:, None])
    for start_n in range(lo, hi, BLOCK_N):
        # -- QK^T in INT8, K loaded already quantized (see _quantize_k_rows) --
        k_hat = tl.load(K8_block_ptr)                # [HEAD_DIM, BLOCK_N] int8
        s_k = tl.load(s_k_ptrs + start_n + offs_n,
                      mask=(start_n + offs_n) < N_CTX, other=1.0)  # [BLOCK_N]
        qk = tl.dot(q_hat, k_hat, out_dtype=tl.int32).to(tl.float32)
        qk = qk * (s_q[:, None] * s_k[None, :]) * qk_scale
        query_positions = offs_m + SEQ_LEN_K - SEQ_LEN_Q
        if STAGE == 2:
            mask = query_positions[:, None] >= (start_n + offs_n[None, :])
            if SEQ_LEN_K < N_CTX:
                mask = mask & ((start_n + offs_n[None, :]) < SEQ_LEN_K)
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            if SEQ_LEN_K < N_CTX:
                pad_mask = ((start_n + offs_n[None, :]) < SEQ_LEN_K)
                qk = qk + tl.where(pad_mask, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            else:
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        # Keep PV in FP16. The unpatched SM75 lowering mispacked an INT8 operand
        # generated by the MMA/softmax chain; independent loaded or generated
        # dot operands were exact. A local Triton patch routes this conversion
        # through shared memory. PV INT8 remains disabled pending scaled,
        # end-to-end accuracy and performance validation.
        p16 = p.to(tl.float16)
        v = tl.load(V_block_ptr)                     # [BLOCK_N, HEAD_DIM]
        acc += tl.dot(p16, v, out_dtype=tl.float32)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K8_block_ptr = tl.advance(K8_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


def _int8_fwd_launch_config(head_dim, seqlen):
    """Tile choice from interleaved per-config measurements on a 2080 Ti.

    Pinned rather than autotuned: the candidates span 1.3-2.4x, and Triton's
    autotuner times each one once, in sequence, so on a noisy device it picks a
    slow tile about as often as the right one -- measured end-to-end it landed
    1.2-1.6x off the best config in this table.

    A 128-row query tile is the fastest choice at head dim 64 (1.3-1.6x over 64
    rows) but it is only usable when the sequence divides evenly: with a partial
    row block the rows in that block's first half come out wrong (verified
    against an fp32 reference on 3.1.0 -- 64/192/320 all diverge, 128/256/512 do
    not). BLOCK_M=64 always divides, since the caller requires a multiple of 64.

    BLOCK_M must also stay >= BLOCK_N: the causal stages split the key range on
    BLOCK_M boundaries and only the diagonal stage masks, so a wider key tile
    would walk past the split.
    """
    if head_dim <= 64 and seqlen % 128 == 0:
        return {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}
    return {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}


@triton.jit
def _attn_fwd_int8(Q, K8, SK, V, sm_scale, M, Out,  #
                   stride_qz, stride_qh, stride_qm, stride_qk,  #
                   stride_kz, stride_kh, stride_kn, stride_kk,  #
                   stride_vz, stride_vh, stride_vk, stride_vn,  #
                   stride_oz, stride_oh, stride_om, stride_on,  #
                   Z, H, N_CTX, HEAD_DIM: tl.constexpr,  #
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  #
                   STAGE: tl.constexpr, SEQ_LEN_K: tl.constexpr, SEQ_LEN_Q: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    kv_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vk, stride_vn),
        offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    K8_block_ptr = tl.make_block_ptr(
        base=K8 + off_hz.to(tl.int64) * N_CTX * HEAD_DIM, shape=(HEAD_DIM, N_CTX),
        strides=(1, HEAD_DIM), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    s_k_ptrs = SK + off_hz.to(tl.int64) * N_CTX
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    _, s_q = _quantize_rows(q)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_int8_inner(acc, l_i, m_i, q, s_q, K8_block_ptr, s_k_ptrs,
                                             V_block_ptr, start_m, qk_scale, BLOCK_M, HEAD_DIM,
                                             BLOCK_N, 4 - STAGE, offs_m, offs_n, N_CTX,
                                             SEQ_LEN_K, SEQ_LEN_Q)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_int8_inner(acc, l_i, m_i, q, s_q, K8_block_ptr, s_k_ptrs,
                                             V_block_ptr, start_m, qk_scale, BLOCK_M, HEAD_DIM,
                                             BLOCK_N, 2, offs_m, offs_n, N_CTX,
                                             SEQ_LEN_K, SEQ_LEN_Q)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class _attention_int8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, seq_len_k, seq_len_q):
        o = torch.empty_like(q)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        stage = 3 if causal else 1
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        # Hoist K's quantization out of the query loop; the attention kernel then
        # loads INT8 K directly instead of re-quantizing every key tile per query
        # block (which measured ~4x more expensive than the mma it feeds).
        k8 = torch.empty((BATCH * N_HEAD, N_CTX, q.shape[3]), device=q.device, dtype=torch.int8)
        s_k = torch.empty((BATCH * N_HEAD, N_CTX), device=q.device, dtype=torch.float32)
        _quantize_k_rows[(triton.cdiv(N_CTX, 64), BATCH * N_HEAD)](
            k, k8, s_k, N_CTX, HEAD_DIM=q.shape[3], BLOCK_R=64, num_warps=4,
        )
        cfg = _int8_fwd_launch_config(q.shape[3], q.shape[2])
        grid = (triton.cdiv(N_CTX, cfg["BLOCK_M"]), BATCH * N_HEAD, 1)
        _attn_fwd_int8[grid](
            q, k8, s_k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            HEAD_DIM=q.shape[3],
            STAGE=stage, SEQ_LEN_K=seq_len_k, SEQ_LEN_Q=seq_len_q,
            **cfg,
        )
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.seq_len_k = seq_len_k
        ctx.seq_len_q = seq_len_q
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        # QK^T is the only INT8 matmul in forward. The dense backward already handles
        # the same gradient math: dot accumulators are FP32, and dS is cast to the
        # input element type before dQ/dK accumulation. Reuse it instead of duplicating
        # the backward path.
        dq, dk, dv = _dense_backward(
            q, k, v, o, M, do,
            sm_scale=ctx.sm_scale, head_dim=q.shape[-1], causal=ctx.causal,
            seq_len_k=ctx.seq_len_k, seq_len_q=ctx.seq_len_q,
            kv_heads=k.shape[1],
        )
        return dq, dk, dv, None, None, None, None


def attention_int8(q, k, v, causal=False, sm_scale=None, seq_len_k=None, seq_len_q=None):
    """INT8 flash attention (Turing sm_75 only). q,k,v: [B, H, S, D]."""
    head_dim = q.shape[-1]
    if not q.is_cuda:
        raise ValueError("The INT8 kernel requires CUDA tensors")
    if torch.cuda.get_device_capability(q.device) != (7, 5):
        raise ValueError(
            "The INT8 kernel requires Turing (compute capability 7.5); other GPUs "
            "have no matching INT8 tensor-core path here, use the fp16 entry point"
        )
    if head_dim < 32 or head_dim & (head_dim - 1):
        raise ValueError(
            "The INT8 kernel requires a power-of-two head dim of at least 32"
        )
    if q.shape[2] % 64:
        raise ValueError("The INT8 kernel requires sequence lengths divisible by 64")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError(
            "The INT8 kernel requires contiguous inputs; its backward addresses "
            "K/V with Q's strides and a strided input silently gives wrong results"
        )
    if q.shape[2] != k.shape[2] or k.shape[2] != v.shape[2]:
        raise ValueError("The INT8 kernel requires equal sequence lengths")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    if seq_len_k is None:
        seq_len_k = q.shape[2]
    if seq_len_q is None:
        seq_len_q = q.shape[2]
    return _attention_int8.apply(q, k, v, causal, sm_scale, seq_len_k, seq_len_q)
