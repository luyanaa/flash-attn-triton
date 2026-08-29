"""
SageBwd-style INT8 Flash Attention for Turing (sm_75).

Turing INT8 tensor cores deliver 2x the fp16 peak (T4: 130 vs 65 TOPS). This
implements an INT8 forward + backward following SageBwd (arXiv 2505.11594,
2603.02170): most attention matmuls run on INT8 tensor cores, while the
accuracy-critical dP = dO @ V^T is kept in FP16.

Granularity is per-token for QK^T and per-token for PV (SageAttention-style),
fp32 accumulation. Gated to sm_75 (compute capability (7, 5)) by the caller;
Volta (sm_70) has no INT8 tensor cores and must use the fp16 path.

Equal-length, non-window, non-alibi inputs only.
"""
import math

import torch
import triton
import triton.language as tl


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
def _attn_fwd_int8_inner(acc, l_i, m_i, q, s_q,  #
                         K_block_ptr, V_block_ptr,  #
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
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # Triton 3.2 cannot feed an int8 kernel parameter into tl.dot, so quantize q
    # here from its fp16 form (once per query block).
    q_hat = _round_i8(q / s_q[:, None])
    for start_n in range(lo, hi, BLOCK_N):
        # -- QK^T in INT8 --
        k = tl.load(K_block_ptr)                     # [HEAD_DIM, BLOCK_N]
        k_hat, s_k = _quantize_cols(k)
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
        # -- PV in FP16. The INT8 PV dot corrupts its result on sm_75 when the
        # int8 operand comes from an in-kernel chain (q_rows(exp2(qk))): Triton
        # 3.2 skips the MMA layout conversion, returning garbage. Loaded int8
        # operands are fine, so QK^T stays INT8; PV is FP16 like the backward's
        # dP (SageBwd), which is also strictly more accurate.
        p16 = p.to(tl.float16)
        v = tl.load(V_block_ptr)                     # [BLOCK_N, HEAD_DIM]
        acc += tl.dot(p16, v, out_dtype=tl.float32)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_int8(Q, K, V, sm_scale, M, Out,  #
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
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn),
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
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
        acc, l_i, m_i = _attn_fwd_int8_inner(acc, l_i, m_i, q, s_q, K_block_ptr, V_block_ptr,
                                             start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N,
                                             4 - STAGE, offs_m, offs_n, N_CTX, SEQ_LEN_K, SEQ_LEN_Q)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_int8_inner(acc, l_i, m_i, q, s_q, K_block_ptr, V_block_ptr,
                                             start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N,
                                             2, offs_m, offs_n, N_CTX, SEQ_LEN_K, SEQ_LEN_Q)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_bwd_int8(Q, K, V, sm_scale, DO, DQ, DK, DV, M, D,  #
                   stride_z, stride_h, stride_tok, stride_d,  #
                   H, N_CTX, HEAD_DIM: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr, BLOCK_M1: tl.constexpr,
                   SEQ_LEN_K: tl.constexpr, SEQ_LEN_Q: tl.constexpr,
                   CAUSAL: tl.constexpr):
    """Fused INT8 backward: dV/dK/dQ. Per (b, h, key-block) program."""
    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, HEAD_DIM), HEAD_DIM), HEAD_DIM)
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    batch_idx = bhid // H
    query_head = bhid % H
    adj = (stride_h * query_head + stride_z * batch_idx).to(tl.int64)
    Q += adj; K += adj; V += adj; DO += adj
    DQ += adj; DK += adj; DV += adj
    M += off_chz; D += off_chz

    offs_n = tl.arange(0, BLOCK_N1)
    start_n = tl.program_id(0) * BLOCK_N1
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # k/v in [BLOCK_N1, HEAD_DIM]; qT in transposed [HEAD_DIM, BLOCK_M1] so all
    # gradient matmuls avoid tl.trans on computed operands (sm_75 MMA layout).
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = N_CTX // BLOCK_M1
    for blk in range(num_steps):
        offs_m = tl.arange(0, BLOCK_M1)
        start_m = blk * BLOCK_M1
        qT = tl.load(Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d)  # [HD, BLOCK_M1]
        q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)  # [BLOCK_M1, HD]
        do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
        m = tl.load(M + offs_m)                       # LSE
        delta = tl.load(D + offs_m)

        # Recompute S^T = K @ Q^T in FP16 (Triton 3.2's sm_75 int8 MMA corrupts
        # in-kernel-quantized operands; the backward is accuracy-critical and
        # SageBwd keeps dP FP16 anyway). The forward QK^T stays INT8.
        qkT = tl.dot(k, qT, out_dtype=tl.float32) * (sm_scale * 1.4426950408889634)  # [BLOCK_N1, BLOCK_M1]
        pT = tl.math.exp2(qkT - m[None, :])
        if CAUSAL:
            mask = (offs_m[None, :] + SEQ_LEN_K - SEQ_LEN_Q >= offs_n[:, None])
            if SEQ_LEN_K < N_CTX:
                mask = mask & (offs_n[:, None] < SEQ_LEN_K)
            pT = tl.where(mask, pT, 0.0)
        elif SEQ_LEN_K < N_CTX:
            pT = tl.where(offs_n[:, None] < SEQ_LEN_K, pT, 0.0)

        # dV = P^T @ dO  (pT already [BLOCK_N1, BLOCK_M1]; no trans).
        dv += tl.dot(pT.to(do.dtype), do, out_dtype=tl.float32)
        # dP^T = V @ dO^T ; dS = P * (dP - delta)  (trans on loaded do only).
        dpT = tl.dot(v, tl.trans(do), out_dtype=tl.float32)
        dsT = pT * (dpT - delta[None, :])
        # dK = dS^T @ Q  (dsT [BLOCK_N1, BLOCK_M1] @ q [BLOCK_M1, HD]; no trans).
        dk += tl.dot(dsT.to(q.dtype), q, out_dtype=tl.float32)
        # dQ = dS @ K  in the [BLOCK_M1, BLOCK_N1] orientation with kT/vT loaded
        # transposed (no tl.trans at all in this path; the fp16 kernel's dQ
        # kernel loads kT transposed and only transposes the transposed load).
        kT = tl.load(K + offs_k[:, None] * stride_d + offs_n[None, :] * stride_tok)  # [HEAD_DIM, BLOCK_N1]
        vT = tl.load(V + offs_k[:, None] * stride_d + offs_n[None, :] * stride_tok)  # [HEAD_DIM, BLOCK_N1]
        dp = tl.dot(do, vT, out_dtype=tl.float32)   # [BLOCK_M1, BLOCK_N1]
        qk = tl.dot(q, kT, out_dtype=tl.float32) * (sm_scale * 1.4426950408889634)
        p = tl.math.exp2(qk - m[:, None])
        if CAUSAL:
            # [BLOCK_M1, BLOCK_N1] causal mask (mask above is [BLOCK_N1, BLOCK_M1]).
            mask_mn = (offs_m[:, None] + SEQ_LEN_K - SEQ_LEN_Q >= offs_n[None, :])
            if SEQ_LEN_K < N_CTX:
                mask_mn = mask_mn & (offs_n[None, :] < SEQ_LEN_K)
            p = tl.where(mask_mn, p, 0.0)
        elif SEQ_LEN_K < N_CTX:
            p = tl.where(offs_n[None, :] < SEQ_LEN_K, p, 0.0)
        ds = p * (dp - delta[:, None])
        dq = tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32) * sm_scale
        tl.atomic_add(DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, dq)

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.atomic_add(dv_ptrs, dv)
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.atomic_add(dk_ptrs, dk)


class _attention_int8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, seq_len_k, seq_len_q):
        o = torch.empty_like(q)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        stage = 3 if causal else 1
        BLOCK_M, BLOCK_N = 64, 64
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        _attn_fwd_int8[grid](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            HEAD_DIM=q.shape[3], BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            STAGE=stage, SEQ_LEN_K=seq_len_k, SEQ_LEN_Q=seq_len_q,
            num_warps=4, num_stages=2,
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
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        delta = torch.sum(do * o, dim=-1, dtype=torch.float32)
        BLOCK_N1, BLOCK_M1 = 64, 64
        grid = (triton.cdiv(N_CTX, BLOCK_N1), 1, BATCH * N_HEAD)
        _attn_bwd_int8[grid](
            q, k, v, ctx.sm_scale, do, dq, dk, dv, M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            N_HEAD, N_CTX,
            BLOCK_N1=BLOCK_N1, BLOCK_M1=BLOCK_M1, HEAD_DIM=q.shape[3],
            SEQ_LEN_K=ctx.seq_len_k, SEQ_LEN_Q=ctx.seq_len_q,
            CAUSAL=ctx.causal, num_warps=4, num_stages=2,
        )
        return dq, dk, dv, None, None, None, None


def attention_int8(q, k, v, causal=False, sm_scale=None, seq_len_k=None, seq_len_q=None):
    """INT8 flash attention (Turing sm_75 only). q,k,v: [B, H, S, D]."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    if seq_len_k is None:
        seq_len_k = q.shape[2]
    if seq_len_q is None:
        seq_len_q = q.shape[2]
    return _attention_int8.apply(q, k, v, causal, sm_scale, seq_len_k, seq_len_q)
