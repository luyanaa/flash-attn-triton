"""
Fused Attention with Dropout
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import random

import pytest
import torch

import triton
import triton.language as tl

DEVICE = 'cuda'


@triton.jit
def _varlen_fwd(Q, K, V, O, LSE, CU_Q, CU_K, ALIBI_SLOPES,
                sm_scale, dropout_p, dropout_seed,
                stride_qt, stride_kt, stride_vt, stride_ot,
                H, H_KV, MAX_SEQLEN_Q, MAX_SEQLEN_K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
                USE_DROPOUT: tl.constexpr, USE_ALIBI: tl.constexpr,
                WINDOW_LEFT: tl.constexpr, WINDOW_RIGHT: tl.constexpr,
                SOFTCAP):
    batch = tl.program_id(1)
    head = tl.program_id(2)
    q_start = tl.load(CU_Q + batch)
    q_end = tl.load(CU_Q + batch + 1)
    k_start = tl.load(CU_K + batch)
    k_end = tl.load(CU_K + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    q_valid = rows < q_len
    # SDPA uses upper-left causal alignment for rectangular attention.
    q_pos = rows
    kv_head = head // (H // H_KV)
    d = tl.arange(0, HEAD_DIM)
    q_ptr = Q + (q_start + rows[:, None]) * stride_qt + head * HEAD_DIM + d[None, :]
    q = tl.load(q_ptr, mask=q_valid[:, None], other=0.0)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    slope = tl.load(ALIBI_SLOPES + batch * H + head) if USE_ALIBI else 0.0

    max_n = k_len
    if CAUSAL:
        # Every key block at or above this bound is fully masked.
        max_n = tl.minimum(k_len, (tl.program_id(0) + 1) * BLOCK_M)
    for start_n in range(0, MAX_SEQLEN_K, BLOCK_N):
        if not CAUSAL or start_n < max_n:
            cols = start_n + tl.arange(0, BLOCK_N)
            k_valid = cols < k_len
            key_pos = cols[None, :]
            valid = k_valid[None, :]
            if CAUSAL:
                valid = valid & (q_pos[:, None] >= key_pos)
            if WINDOW_LEFT >= 0:
                valid = valid & (key_pos >= q_pos[:, None] - WINDOW_LEFT)
            if WINDOW_RIGHT >= 0:
                valid = valid & (key_pos <= q_pos[:, None] + WINDOW_RIGHT)
            # Legacy V100 workaround (commented out for comparison):
            # Avoid pointer-loaded FP16 transposes in tl.dot on V100.
            # Directly loading K as [head_dim, block_n] preserves correctness.
            # k_t_ptr = K + (k_start + cols[None, :]) * stride_kt + kv_head * HEAD_DIM + d[:, None]
            v_ptr = V + (k_start + cols[:, None]) * stride_vt + kv_head * HEAD_DIM + d[None, :]
            # k_t = tl.load(k_t_ptr, mask=k_valid[None, :], other=0.0)
            k_ptr = K + (k_start + cols[:, None]) * stride_kt + kv_head * HEAD_DIM + d[None, :]
            k = tl.load(k_ptr, mask=k_valid[:, None], other=0.0)
            k_t = tl.trans(k)
            v = tl.load(v_ptr, mask=k_valid[:, None], other=0.0)
            scores = tl.dot(q, k_t, out_dtype=tl.float32) * sm_scale
            if USE_ALIBI:
                scores -= slope * tl.abs(q_pos[:, None] - key_pos)
            if SOFTCAP > 0:
                scores = SOFTCAP * (2.0 / (1.0 + tl.math.exp(-2.0 * (scores / SOFTCAP))) - 1.0)
            scores = scores * 1.4426950408889634
            scores = tl.where(valid & q_valid[:, None], scores, -float('inf'))
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.math.exp2(scores - m_ij[:, None])
            p = tl.where(valid & q_valid[:, None], p, 0.0)
            l_ij = tl.sum(p, axis=1)
            if USE_DROPOUT:
                rng = tl.rand(dropout_seed, (q_start + rows[:, None]) * MAX_SEQLEN_K + cols[None, :])
                p = tl.where(rng > dropout_p, p / (1.0 - dropout_p), 0.0)
            alpha = tl.where(m_i == -float('inf'), 0.0, tl.math.exp2(m_i - m_ij))
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32)
            m_i = m_ij

    lse = tl.where(l_i > 0, m_i + tl.math.log2(l_i), -float('inf'))
    out = acc / tl.where(l_i[:, None] > 0, l_i[:, None], 1.0)
    out_ptr = O + (q_start + rows[:, None]) * stride_ot + head * HEAD_DIM + d[None, :]
    lse_ptr = LSE + (q_start + rows) * H + head
    tl.store(out_ptr, out.to(O.type.element_ty), mask=q_valid[:, None])
    tl.store(lse_ptr, lse, mask=q_valid)


@triton.jit
def _varlen_bwd(Q, K, V, O, DO, DQ, LSE, CU_Q, CU_K, ALIBI_SLOPES,
                sm_scale, dropout_p, dropout_seed,
                stride_qt, stride_kt, stride_vt, stride_ot,
                H, H_KV, MAX_SEQLEN_Q, MAX_SEQLEN_K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
                USE_DROPOUT: tl.constexpr, USE_ALIBI: tl.constexpr,
                WINDOW_LEFT: tl.constexpr, WINDOW_RIGHT: tl.constexpr,
                SOFTCAP):
    batch = tl.program_id(1)
    head = tl.program_id(2)
    q_start = tl.load(CU_Q + batch)
    q_end = tl.load(CU_Q + batch + 1)
    k_start = tl.load(CU_K + batch)
    k_end = tl.load(CU_K + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_valid = rows < q_len
    # Keep backward masking identical to the SDPA-aligned forward pass.
    q_pos = rows
    kv_head = head // (H // H_KV)
    d = tl.arange(0, HEAD_DIM)
    q_ptr = Q + (q_start + rows[:, None]) * stride_qt + head * HEAD_DIM + d[None, :]
    o_ptr = O + (q_start + rows[:, None]) * stride_ot + head * HEAD_DIM + d[None, :]
    do_ptr = DO + (q_start + rows[:, None]) * stride_ot + head * HEAD_DIM + d[None, :]
    q = tl.load(q_ptr, mask=q_valid[:, None], other=0.0)
    o = tl.load(o_ptr, mask=q_valid[:, None], other=0.0)
    do = tl.load(do_ptr, mask=q_valid[:, None], other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    lse = tl.load(LSE + (q_start + rows) * H + head, mask=q_valid, other=0.0)
    slope = tl.load(ALIBI_SLOPES + batch * H + head) if USE_ALIBI else 0.0
    dq = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    max_n = k_len
    if CAUSAL:
        # Every key block at or above this bound is fully masked.
        max_n = tl.minimum(k_len, (pid_m + 1) * BLOCK_M)
    for start_n in range(0, MAX_SEQLEN_K, BLOCK_N):
        if not CAUSAL or start_n < max_n:
            cols = start_n + tl.arange(0, BLOCK_N)
            k_valid = cols < k_len
            key_pos = cols[None, :]
            valid = k_valid[None, :]
            if CAUSAL:
                valid = valid & (q_pos[:, None] >= key_pos)
            if WINDOW_LEFT >= 0:
                valid = valid & (key_pos >= q_pos[:, None] - WINDOW_LEFT)
            if WINDOW_RIGHT >= 0:
                valid = valid & (key_pos <= q_pos[:, None] + WINDOW_RIGHT)
            k_ptr = K + (k_start + cols[:, None]) * stride_kt + kv_head * HEAD_DIM + d[None, :]
            # Legacy direct-transpose pointer loads retained for comparison:
            # k_t_ptr = K + (k_start + cols[None, :]) * stride_kt + kv_head * HEAD_DIM + d[:, None]
            # v_t_ptr = V + (k_start + cols[None, :]) * stride_vt + kv_head * HEAD_DIM + d[:, None]
            v_ptr = V + (k_start + cols[:, None]) * stride_vt + kv_head * HEAD_DIM + d[None, :]
            k = tl.load(k_ptr, mask=k_valid[:, None], other=0.0)
            # k_t = tl.load(k_t_ptr, mask=k_valid[None, :], other=0.0)
            # v_t = tl.load(v_t_ptr, mask=k_valid[None, :], other=0.0)
            k_t = tl.trans(k)
            v = tl.load(v_ptr, mask=k_valid[:, None], other=0.0)
            v_t = tl.trans(v)
            scores = tl.dot(q, k_t, out_dtype=tl.float32) * sm_scale
            if USE_ALIBI:
                scores -= slope * tl.abs(q_pos[:, None] - key_pos)
            if SOFTCAP > 0:
                cap_tanh = 2.0 / (1.0 + tl.math.exp(-2.0 * (scores / SOFTCAP))) - 1.0
                scores = SOFTCAP * cap_tanh
                cap_grad = 1.0 - cap_tanh * cap_tanh
            else:
                cap_grad = tl.zeros_like(scores) + 1.0
            p = tl.math.exp2(scores * 1.4426950408889634 - lse[:, None])
            p = tl.where(lse[:, None] == -float('inf'), 0.0, p)
            p = tl.where(valid & q_valid[:, None], p, 0.0)
            if USE_DROPOUT:
                rng = tl.rand(dropout_seed, (q_start + rows[:, None]) * MAX_SEQLEN_K + cols[None, :])
                p = tl.where(rng > dropout_p, p / (1.0 - dropout_p), 0.0)
            dp = tl.dot(do, v_t, out_dtype=tl.float32)
            ds = p * (dp - delta[:, None]) * cap_grad
            ds16 = ds.to(q.type.element_ty)
            dq += tl.dot(ds16, k, out_dtype=tl.float32)

    dq_ptr = DQ + (q_start + rows[:, None]) * stride_qt + head * HEAD_DIM + d[None, :]
    tl.store(dq_ptr, (dq * sm_scale).to(DQ.type.element_ty), mask=q_valid[:, None])

@triton.jit
def _varlen_bwd_dkdv(Q, K, V, O, DO, DK, DV, LSE, CU_Q, CU_K, ALIBI_SLOPES,
                     sm_scale, dropout_p, dropout_seed,
                     stride_qt, stride_kt, stride_vt, stride_ot,
                     H, H_KV, MAX_SEQLEN_Q, MAX_SEQLEN_K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
                     USE_DROPOUT: tl.constexpr, USE_ALIBI: tl.constexpr,
                     WINDOW_LEFT: tl.constexpr, WINDOW_RIGHT: tl.constexpr,
                     GROUP_SIZE: tl.constexpr, SOFTCAP):
    # One program owns one K/V block, batch, and KV head. Query-head groups
    # are reduced inside the program, so every dK/dV element has one writer.
    batch = tl.program_id(1)
    kv_head = tl.program_id(2)
    q_start = tl.load(CU_Q + batch)
    q_end = tl.load(CU_Q + batch + 1)
    k_start = tl.load(CU_K + batch)
    k_end = tl.load(CU_K + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    start_n = tl.program_id(0) * BLOCK_N
    cols = start_n + tl.arange(0, BLOCK_N)
    k_valid = cols < k_len
    key_pos = cols[None, :]
    d = tl.arange(0, HEAD_DIM)
    # Legacy direct-transpose K/V loads retained for comparison:
    # k_t_ptr = K + (k_start + cols[None, :]) * stride_kt + kv_head * HEAD_DIM + d[:, None]
    # v_t_ptr = V + (k_start + cols[None, :]) * stride_vt + kv_head * HEAD_DIM + d[:, None]
    # k_t = tl.load(k_t_ptr, mask=k_valid[None, :], other=0.0)
    # v_t = tl.load(v_t_ptr, mask=k_valid[None, :], other=0.0)
    k_ptr = K + (k_start + cols[:, None]) * stride_kt + kv_head * HEAD_DIM + d[None, :]
    v_ptr = V + (k_start + cols[:, None]) * stride_vt + kv_head * HEAD_DIM + d[None, :]
    k = tl.load(k_ptr, mask=k_valid[:, None], other=0.0)
    k_t = tl.trans(k)
    v = tl.load(v_ptr, mask=k_valid[:, None], other=0.0)
    v_t = tl.trans(v)
    dk = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)

    for head_group in range(GROUP_SIZE):
        head = kv_head * GROUP_SIZE + head_group
        slope = tl.load(ALIBI_SLOPES + batch * H + head) if USE_ALIBI else 0.0
        for start_m in range(0, MAX_SEQLEN_Q, BLOCK_M):
            rows = start_m + tl.arange(0, BLOCK_M)
            q_valid = rows < q_len
            q_pos = rows
            q_ptr = Q + (q_start + rows[:, None]) * stride_qt + head * HEAD_DIM + d[None, :]
            # q_t_ptr = Q + (q_start + rows[None, :]) * stride_qt + head * HEAD_DIM + d[:, None]
            o_ptr = O + (q_start + rows[:, None]) * stride_ot + head * HEAD_DIM + d[None, :]
            do_ptr = DO + (q_start + rows[:, None]) * stride_ot + head * HEAD_DIM + d[None, :]
            # do_t_ptr = DO + (q_start + rows[None, :]) * stride_ot + head * HEAD_DIM + d[:, None]
            q = tl.load(q_ptr, mask=q_valid[:, None], other=0.0)
            # q_t = tl.load(q_t_ptr, mask=q_valid[None, :], other=0.0)
            q_t = tl.trans(q)
            o = tl.load(o_ptr, mask=q_valid[:, None], other=0.0)
            do = tl.load(do_ptr, mask=q_valid[:, None], other=0.0)
            # do_t = tl.load(do_t_ptr, mask=q_valid[None, :], other=0.0)
            do_t = tl.trans(do)
            delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
            lse = tl.load(LSE + (q_start + rows) * H + head, mask=q_valid, other=0.0)
            valid = k_valid[None, :]
            if CAUSAL:
                valid = valid & (q_pos[:, None] >= key_pos)
            if WINDOW_LEFT >= 0:
                valid = valid & (key_pos >= q_pos[:, None] - WINDOW_LEFT)
            if WINDOW_RIGHT >= 0:
                valid = valid & (key_pos <= q_pos[:, None] + WINDOW_RIGHT)
            scores = tl.dot(q, k_t, out_dtype=tl.float32) * sm_scale
            if USE_ALIBI:
                scores -= slope * tl.abs(q_pos[:, None] - key_pos)
            if SOFTCAP > 0:
                cap_tanh = 2.0 / (1.0 + tl.math.exp(-2.0 * (scores / SOFTCAP))) - 1.0
                scores = SOFTCAP * cap_tanh
                cap_grad = 1.0 - cap_tanh * cap_tanh
            else:
                cap_grad = tl.zeros_like(scores) + 1.0
            p = tl.math.exp2(scores * 1.4426950408889634 - lse[:, None])
            p = tl.where(lse[:, None] == -float('inf'), 0.0, p)
            p = tl.where(valid & q_valid[:, None], p, 0.0)
            if USE_DROPOUT:
                rng = tl.rand(dropout_seed, (q_start + rows[:, None]) * MAX_SEQLEN_K + cols[None, :])
                p = tl.where(rng > dropout_p, p / (1.0 - dropout_p), 0.0)
            dp = tl.dot(do, v_t, out_dtype=tl.float32)
            ds = p * (dp - delta[:, None]) * cap_grad
            ds16 = ds.to(q.type.element_ty)
            dk_t = tl.dot(q_t, ds16, out_dtype=tl.float32) * sm_scale
            dv_t = tl.dot(do_t, p.to(do.type.element_ty), out_dtype=tl.float32)
            dk += tl.trans(dk_t)
            dv += tl.trans(dv_t)

    dk_ptr = DK + (k_start + cols[:, None]) * stride_kt + kv_head * HEAD_DIM + d[None, :]
    dv_ptr = DV + (k_start + cols[:, None]) * stride_vt + kv_head * HEAD_DIM + d[None, :]
    tl.store(dk_ptr, dk.to(DK.type.element_ty), mask=k_valid[:, None])
    tl.store(dv_ptr, dv.to(DV.type.element_ty), mask=k_valid[:, None])


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m,  #
                    dropout_p, dropout_seed,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, USE_DROPOUT: tl.constexpr,
                    SEQ_LEN_K: tl.constexpr, SEQ_LEN_Q: tl.constexpr,
                    SOFTCAP: tl.constexpr, alibi_slope,
                    WINDOW_LEFT: tl.constexpr, WINDOW_RIGHT: tl.constexpr,
                    HAS_SOFTCAP: tl.constexpr, HAS_ALIBI: tl.constexpr,
                    HAS_WINDOW: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    dropout_scale = 1.0 / (1.0 - dropout_p) if USE_DROPOUT else 1.0

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k, out_dtype=tl.float32)
        query_positions = offs_m + SEQ_LEN_K - SEQ_LEN_Q
        if HAS_ALIBI:
            key_positions = start_n + offs_n[None, :]
            qk -= alibi_slope * 1.4426950408889634 * tl.abs(
                query_positions[:, None] - key_positions
            )
        if HAS_SOFTCAP:
            qk = SOFTCAP * (2.0 / (1.0 + tl.math.exp(-2.0 * (qk / SOFTCAP))) - 1.0)
        if HAS_WINDOW:
            window_mask = tl.full([BLOCK_M, BLOCK_N], True, tl.int1)
            if WINDOW_LEFT >= 0:
                window_mask = window_mask & ((start_n + offs_n[None, :]) >= (query_positions[:, None] - WINDOW_LEFT))
            if WINDOW_RIGHT >= 0:
                window_mask = window_mask & ((start_n + offs_n[None, :]) <= (query_positions[:, None] + WINDOW_RIGHT))
        else:
            window_mask = tl.full([BLOCK_M, BLOCK_N], True, tl.int1)
        if STAGE == 2:
            mask = query_positions[:, None] >= (start_n + offs_n[None, :])
            if HAS_WINDOW:
                mask = mask & window_mask
            if SEQ_LEN_K < N_CTX:
                mask = mask & ((start_n + offs_n[None, :]) < SEQ_LEN_K)
            qk = tl.where(mask, qk, -float('inf'))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            mask = tl.full([BLOCK_M, BLOCK_N], True, tl.int1)
            if SEQ_LEN_K < N_CTX:
                mask = mask & ((start_n + offs_n[None, :]) < SEQ_LEN_K)
            if HAS_WINDOW:
                mask = mask & window_mask
            m_ij = tl.maximum(m_i, tl.max(tl.where(mask, qk, -float('inf')), 1))
            qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        p = tl.where(mask, p, 0.0)
        l_ij = tl.sum(p, 1)

        # -- apply dropout if specified at compile time --
        if USE_DROPOUT:
            # Create a unique offset for each element in the p matrix
            # The offset needs to be unique across both dimensions of p
            row_offsets = offs_m[:, None]
            col_offsets = (start_n + offs_n[None, :])
            # Combine row and column offsets into a unique offset
            # Multiply row offset by N_CTX to ensure uniqueness across the matrix
            combined_offsets = row_offsets * N_CTX + col_offsets

            # Generate the dropout mask using combined offsets
            rng = tl.rand(dropout_seed, combined_offsets)
            dropout_mask = rng > dropout_p
            # Apply the dropout and scale by 1/(1-p)
            p = tl.where(dropout_mask, p / (1.0 - dropout_p), 0.0)
        # -- update m_i and l_i
        alpha = tl.where(m_i == -float('inf'), 0.0, tl.math.exp2(m_i - m_ij))
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(v.type.element_ty)
        acc = tl.dot(p, v, acc, out_dtype=tl.float32)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return BLOCK_M >= BLOCK_N

def _device_capability():
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def _is_volta():
    return _device_capability() == (7, 0)


def _forward_configs_for_device():
    candidates = list(filter(keep, configs))
    if _is_volta():
        return [config for config in candidates if config.num_stages <= 2]
    return candidates



def prune_configs_by_head_dim(configs, named_args, **kwargs):
    head_dim = kwargs["HEAD_DIM"]
    return [config for config in configs if config.kwargs["BLOCK_N"] <= head_dim]


@triton.autotune(
    _forward_configs_for_device(),
    key=["N_CTX", "HEAD_DIM", "STAGE", "USE_DROPOUT"],
    prune_configs_by={"early_config_prune": prune_configs_by_head_dim},
)
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, dropout_p, dropout_seed, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX, H_KV,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              USE_DROPOUT: tl.constexpr,  #
              SEQ_LEN_K: tl.constexpr, SEQ_LEN_Q: tl.constexpr,
              WINDOW_LEFT: tl.constexpr, WINDOW_RIGHT: tl.constexpr,
              SOFTCAP: tl.constexpr, ALIBI_SLOPES, USE_ALIBI: tl.constexpr,
              STORE_LSE: tl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    kv_head = off_h // (H // H_KV)
    kv_offset = off_z.to(tl.int64) * stride_kz + kv_head.to(tl.int64) * stride_kh
    alibi_slope = tl.load(ALIBI_SLOPES + off_z * H + off_h) if USE_ALIBI else 0.0

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = tl.max_contiguous(tl.multiple_of(start_m * BLOCK_M + tl.arange(0, BLOCK_M), BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # Pre-scale Q so every inner-loop dot avoids a repeated multiply.
    q_scale = sm_scale * 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * q_scale).to(Q.type.element_ty)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, dropout_p, dropout_seed,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, USE_DROPOUT,
                                        SEQ_LEN_K, SEQ_LEN_Q, SOFTCAP, alibi_slope,
                                        WINDOW_LEFT, WINDOW_RIGHT,
                                        HAS_SOFTCAP=SOFTCAP > 0,
                                        HAS_ALIBI=USE_ALIBI,
                                        HAS_WINDOW=(WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0))
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, dropout_p, dropout_seed,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, USE_DROPOUT,
                                        SEQ_LEN_K, SEQ_LEN_Q, SOFTCAP, alibi_slope,
                                        WINDOW_LEFT, WINDOW_RIGHT,
                                        HAS_SOFTCAP=SOFTCAP > 0,
                                        HAS_ALIBI=USE_ALIBI,
                                        HAS_WINDOW=(WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0))
    safe_l_i = tl.where(l_i > 0, l_i, 1.0)
    if STORE_LSE:
        lse = tl.where(l_i > 0, m_i + tl.math.log2(l_i), -float('inf'))
        m_ptrs = M + off_hz * N_CTX + offs_m
        # The tail block is partial whenever BLOCK_M does not divide N_CTX (the
        # caller only requires a multiple of 64, and the configs include
        # BLOCK_M=128). Both stores must therefore be guarded: an unguarded
        # block-pointer store writes the out-of-range rows over the start of the
        # head's own output, and an unguarded LSE store walks past M entirely.
        tl.store(m_ptrs, lse, mask=offs_m < N_CTX)
    acc = acc / safe_l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,))


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    # Calculate starting position
    start_idx = tl.program_id(0) * BLOCK_M
    # Get offsets and create mask for valid positions
    off_m = start_idx + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    
    # Create mask for valid positions (where offset < N_CTX)
    mask = off_m < N_CTX
    
    # Clamp offsets to stay within bounds
    off_m_valid = tl.where(mask, off_m, N_CTX - 1)
    
    # Load with mask
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m_valid[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m_valid[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    
    # Compute delta
    delta = tl.sum(o * do, axis=1)
    
    # Store with mask
    tl.store(Delta + off_hz * N_CTX + off_m_valid, delta, mask=mask)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   dropout_p, dropout_seed,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                    MASK: tl.constexpr, USE_DROPOUT: tl.constexpr,
                    SEQ_LEN_K: tl.constexpr, SEQ_LEN_Q: tl.constexpr,
                    WINDOW_LEFT: tl.constexpr, WINDOW_RIGHT: tl.constexpr,
                    softcap, alibi_slope,
                    HAS_SOFTCAP: tl.constexpr, HAS_ALIBI: tl.constexpr,
                    HAS_WINDOW: tl.constexpr):
    offs_m = tl.max_contiguous(tl.multiple_of(start_m + tl.arange(0, BLOCK_M1), BLOCK_M1), BLOCK_M1)
    offs_n = tl.max_contiguous(tl.multiple_of(start_n + tl.arange(0, BLOCK_N1), BLOCK_N1), BLOCK_N1)
    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, HEAD_DIM), HEAD_DIM), HEAD_DIM)
    q_ptrs = Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # q_t_ptrs = Q + offs_k[:, None] * stride_d + offs_m[None, :] * stride_tok
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # do_t_ptrs = DO + offs_k[:, None] * stride_d + offs_m[None, :] * stride_tok
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    dropout_scale = 1.0 / (1.0 - dropout_p) if USE_DROPOUT else 1.0
    for blk_idx in range(num_steps):
        q = tl.load(q_ptrs)
        # Legacy V100 workaround (commented out for comparison):
        # Avoid pointer-loaded FP16 transposes in tl.dot on V100.
        # Load the transposed operand directly instead.
        # q_t = tl.load(q_t_ptrs)
        q_t = tl.trans(q)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = tl.max_contiguous(tl.multiple_of(curr_m + tl.arange(0, BLOCK_M1), BLOCK_M1), BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, q_t, out_dtype=tl.float32) * (sm_scale * 1.4426950408889634)
        query_positions = offs_m + SEQ_LEN_K - SEQ_LEN_Q
        if HAS_WINDOW:
            window_mask = tl.full([BLOCK_N1, BLOCK_M1], True, tl.int1)
            if WINDOW_LEFT >= 0:
                window_mask = window_mask & (offs_n[:, None] >= query_positions[None, :] - WINDOW_LEFT)
            if WINDOW_RIGHT >= 0:
                window_mask = window_mask & (offs_n[:, None] <= query_positions[None, :] + WINDOW_RIGHT)
        else:
            window_mask = tl.full([BLOCK_N1, BLOCK_M1], True, tl.int1)
        if HAS_ALIBI:
            qkT -= alibi_slope * 1.4426950408889634 * tl.abs(
                (offs_m + SEQ_LEN_K - SEQ_LEN_Q)[None, :] - offs_n[:, None]
            )
        if HAS_SOFTCAP:
            cap_tanh = 2.0 / (1.0 + tl.math.exp(-2.0 * (qkT / softcap))) - 1.0
            qkT = softcap * cap_tanh
            cap_grad = 1.0 - cap_tanh * cap_tanh
        else:
            cap_grad = tl.zeros_like(qkT) + 1.0
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] + SEQ_LEN_K - SEQ_LEN_Q >= offs_n[:, None])
            if SEQ_LEN_K < N_CTX:
                mask = mask & (offs_n[:, None] < SEQ_LEN_K)
            mask = mask & window_mask
            pT = tl.where(mask, pT, 0.0)
        # Pad masking for non-causal attention with seq_len not divisible by block size.
        if not MASK and SEQ_LEN_K < N_CTX:
            pad_mask = offs_n[:, None] < SEQ_LEN_K
            pT = tl.where(pad_mask & window_mask, pT, 0.0)
        if not MASK:
            pT = tl.where(window_mask, pT, 0.0)
            
        # Apply the same dropout mask for backward pass
        if USE_DROPOUT:
            # Create unique offsets for the dropout mask
            row_offsets = offs_m[None, :]
            col_offsets = offs_n[:, None]
            combined_offsets = row_offsets * N_CTX + col_offsets
            
            # Generate the dropout mask (same as in forward pass)
            rng = tl.rand(dropout_seed, combined_offsets)
            dropout_mask = rng > dropout_p
            # Apply the dropout and scale by 1/(1-p)
            pT = tl.where(dropout_mask, pT * dropout_scale, 0.0)
            
        do = tl.load(do_ptrs)
        # Legacy V100 workaround (commented out for comparison):
        # Avoid pointer-loaded FP16 transposes in tl.dot on V100.
        # do_t = tl.load(do_t_ptrs)
        do_t = tl.trans(do)
        # Compute dV.
        ppT = pT.to(v.type.element_ty)
        dv += tl.dot(ppT, do, out_dtype=tl.float32)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, do_t, out_dtype=tl.float32)
        dsT = pT * (dpT - Di[None, :]) * cap_grad
        dsT = dsT.to(Q.type.element_ty)
        dk += tl.dot(dsT, q, out_dtype=tl.float32)
        # Increment pointers.
        curr_m += step_m
        q_ptrs += step_m * stride_tok
        # q_t_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
        # do_t_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 sm_scale, dropout_p, dropout_seed,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr, USE_DROPOUT: tl.constexpr,
                 SEQ_LEN_K: tl.constexpr, SEQ_LEN_Q: tl.constexpr,
                 WINDOW_LEFT: tl.constexpr, WINDOW_RIGHT: tl.constexpr,
                 softcap, alibi_slope,
                 HAS_SOFTCAP: tl.constexpr, HAS_ALIBI: tl.constexpr,
                 HAS_WINDOW: tl.constexpr):
    offs_m = tl.max_contiguous(tl.multiple_of(start_m + tl.arange(0, BLOCK_M2), BLOCK_M2), BLOCK_M2)
    offs_n = tl.max_contiguous(tl.multiple_of(start_n + tl.arange(0, BLOCK_N2), BLOCK_N2), BLOCK_N2)
    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, HEAD_DIM), HEAD_DIM), HEAD_DIM)
    k_ptrs = K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    v_ptrs = V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    # Legacy V100 workaround pointers retained for comparison:
    # k_t_ptrs = K + offs_k[:, None] * stride_d + offs_n[None, :] * stride_tok
    # v_t_ptrs = V + offs_k[:, None] * stride_d + offs_n[None, :] * stride_tok
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    dropout_scale = 1.0 / (1.0 - dropout_p) if USE_DROPOUT else 1.0
    for blk_idx in range(num_steps):
        k = tl.load(k_ptrs)
        # Avoid pointer-loaded FP16 transposes in tl.dot on V100.
        # Load K and V transposed directly instead.
        # k_t = tl.load(k_t_ptrs)
        # v_t = tl.load(v_t_ptrs)
        k_t = tl.trans(k)
        v = tl.load(v_ptrs)
        v_t = tl.trans(v)
        qk = tl.dot(q, k_t, out_dtype=tl.float32) * (1.4426950408889634 * sm_scale)
        query_positions = offs_m + SEQ_LEN_K - SEQ_LEN_Q
        key_positions = curr_n + tl.arange(0, BLOCK_N2)
        if HAS_WINDOW:
            window_mask = tl.full([BLOCK_M2, BLOCK_N2], True, tl.int1)
            if WINDOW_LEFT >= 0:
                window_mask = window_mask & (key_positions[None, :] >= query_positions[:, None] - WINDOW_LEFT)
            if WINDOW_RIGHT >= 0:
                window_mask = window_mask & (key_positions[None, :] <= query_positions[:, None] + WINDOW_RIGHT)
        else:
            window_mask = tl.full([BLOCK_M2, BLOCK_N2], True, tl.int1)
        if HAS_ALIBI:
            qk -= alibi_slope * 1.4426950408889634 * tl.abs(
                (offs_m + SEQ_LEN_K - SEQ_LEN_Q)[:, None]
                - curr_n - tl.arange(0, BLOCK_N2)[None, :]
            )
        if HAS_SOFTCAP:
            cap_tanh = 2.0 / (1.0 + tl.math.exp(-2.0 * (qk / softcap))) - 1.0
            qk = softcap * cap_tanh
            cap_grad = 1.0 - cap_tanh * cap_tanh
        else:
            cap_grad = tl.zeros_like(qk) + 1.0
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] + SEQ_LEN_K - SEQ_LEN_Q >= offs_n[None, :])
            p = tl.where(mask & window_mask, p, 0.0)
        # Pad masking for non-causal attention with seq_len not divisible by block size.
        if not MASK and SEQ_LEN_K < N_CTX:
            offs_n_curr = curr_n + tl.arange(0, BLOCK_N2)
            pad_mask = offs_n_curr[None, :] < SEQ_LEN_K
            p = tl.where(pad_mask & window_mask, p, 0.0)
        if not MASK:
            p = tl.where(window_mask, p, 0.0)
        # Apply dropout mask for backward pass
        if USE_DROPOUT:
            # Create unique offsets for the dropout mask
            row_offsets = offs_m[:, None]
            col_offsets = curr_n + tl.arange(0, BLOCK_N2)[None, :]
            combined_offsets = row_offsets * N_CTX + col_offsets
            
            # Generate the same dropout mask as in forward pass
            rng = tl.rand(dropout_seed, combined_offsets)
            dropout_mask = rng > dropout_p
            # Apply the dropout and scale by 1/(1-p)
            p = tl.where(dropout_mask, p * dropout_scale, 0.0)
            
        # Compute dP and dS.
        dp = tl.dot(do, v_t, out_dtype=tl.float32)
        ds = p * (dp - Di[:, None]) * cap_grad
        ds = ds.to(K.type.element_ty)
        dq += tl.dot(ds, k, out_dtype=tl.float32)
        # Increment pointers.
        curr_n += step_n
        k_ptrs += step_n * stride_tok
        v_ptrs += step_n * stride_tok
        # k_t_ptrs += step_n * stride_tok
        # v_t_ptrs += step_n * stride_tok
    return dq





backward_configs = [
    triton.Config({}, num_stages=2, num_warps=4),
    triton.Config({}, num_stages=3, num_warps=4),
    triton.Config({}, num_stages=2, num_warps=8),
    triton.Config({}, num_stages=3, num_warps=8),
]


def _is_turing():
    return _device_capability() == (7, 5)


def _backward_configs_for_device():
    if _is_turing():
        return [triton.Config({}, num_stages=1, num_warps=4)]
    if _is_volta():
        return [config for config in backward_configs if config.num_stages <= 2]
    return backward_configs


def _backward_block_sizes():
    if _is_turing() or _is_volta():
        # BLOCK_N1 must be a multiple of BLOCK_M1 in _attn_bwd_dkdv. Measured on a
        # V100: 32-row tiles are 1.20-1.25x faster than 16-row ones end to end
        # (22.97 vs 18.40 TFLOP/s at S=1024, 22.62 vs 18.78 at S=2048 causal),
        # with gradients matching an fp32 reference to the same tolerance.
        return 32, 64, 64, 32
    return 16, 64, 64, 16



@triton.autotune(
    _backward_configs_for_device(),
    key=["N_CTX", "HEAD_DIM", "CAUSAL", "USE_DROPOUT"],
)
@triton.jit
def _attn_bwd(Q, K, V, sm_scale,
              DO,
              DQ, DK, DV,
              M, D,
              dropout_p, dropout_seed,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_z_kv, stride_tok, stride_d,
              H, N_CTX, H_KV,
              BLOCK_M1: tl.constexpr,
              BLOCK_N1: tl.constexpr,
              BLOCK_M2: tl.constexpr,
              BLOCK_N2: tl.constexpr,
              BLK_SLICE_FACTOR: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              USE_DROPOUT: tl.constexpr,
              CAUSAL: tl.constexpr,
              SEQ_LEN_K: tl.constexpr, SEQ_LEN_Q: tl.constexpr,
              WINDOW_LEFT: tl.constexpr, WINDOW_RIGHT: tl.constexpr,
              SOFTCAP, ALIBI_SLOPES, USE_ALIBI: tl.constexpr,
              NATIVE_GQA: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    alibi_slope = tl.load(ALIBI_SLOPES + bhid) if USE_ALIBI else 0.0
    off_chz = (bhid * N_CTX).to(tl.int64)
    batch_idx = bhid // H
    query_head = bhid % H
    kv_head = query_head // (H // H_KV)
    q_adj = (stride_h * query_head + stride_z * batch_idx).to(tl.int64)
    kv_adj = (stride_h * kv_head + stride_z_kv * batch_idx).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += q_adj
    K += kv_adj
    V += kv_adj
    DO += q_adj
    DQ += q_adj
    DK += kv_adj
    DV += kv_adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:
        # For causal attention, we need to handle the masked (diagonal) blocks first
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(dk, dv,
                                Q, k, v, sm_scale,
                                DO,
                                M, D,
                                dropout_p, dropout_seed,
                                stride_tok, stride_d,
                                H, N_CTX,
                                MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
                                start_n, start_m, num_steps,
                                MASK=True,  # Use masking for causal section
                                USE_DROPOUT=USE_DROPOUT,
                                SEQ_LEN_K=SEQ_LEN_K,
                                SEQ_LEN_Q=SEQ_LEN_Q,
                                WINDOW_LEFT=WINDOW_LEFT,
                                WINDOW_RIGHT=WINDOW_RIGHT,
                                softcap=SOFTCAP,
                                alibi_slope=alibi_slope,
                                HAS_SOFTCAP=SOFTCAP > 0,
                                HAS_ALIBI=USE_ALIBI,
                                HAS_WINDOW=(WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0),
                                )
        start_m += num_steps * MASK_BLOCK_M1
        num_steps = (N_CTX - start_m) // BLOCK_M1
        
        # Then handle the non-masked blocks
        if num_steps > 0:
            dk, dv = _attn_bwd_dkdv(
                dk, dv,
                Q, k, v, sm_scale,
                DO,
                M, D,
                dropout_p, dropout_seed,
                stride_tok, stride_d,
                H, N_CTX,
                BLOCK_M1, BLOCK_N1, HEAD_DIM,
                start_n, start_m, num_steps,
                MASK=False,
                USE_DROPOUT=USE_DROPOUT,
                SEQ_LEN_K=SEQ_LEN_K,
                SEQ_LEN_Q=SEQ_LEN_Q,
                WINDOW_LEFT=WINDOW_LEFT,
                WINDOW_RIGHT=WINDOW_RIGHT,
                softcap=SOFTCAP,
                alibi_slope=alibi_slope,
                HAS_SOFTCAP=SOFTCAP > 0,
                HAS_ALIBI=USE_ALIBI,
                HAS_WINDOW=(WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0),
            )
    else:
        # For non-causal attention, process all blocks without masking
        # This is the key difference for supporting non-causal backward pass
        num_steps = N_CTX // BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk, dv,
            Q, k, v, sm_scale,
            DO,
            M, D,
            dropout_p, dropout_seed,
            stride_tok, stride_d,
            H, N_CTX,
            BLOCK_M1, BLOCK_N1, HEAD_DIM,
            start_n, 0, num_steps,
            MASK=False,  # No masking for non-causal attention
            USE_DROPOUT=USE_DROPOUT,
            SEQ_LEN_K=SEQ_LEN_K,
            SEQ_LEN_Q=SEQ_LEN_Q,
            WINDOW_LEFT=WINDOW_LEFT,
            WINDOW_RIGHT=WINDOW_RIGHT,
            softcap=SOFTCAP,
            alibi_slope=alibi_slope,
            HAS_SOFTCAP=SOFTCAP > 0,
            HAS_ALIBI=USE_ALIBI,
            HAS_WINDOW=(WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0),
        )

    # Store dV and dK
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    if NATIVE_GQA:
        tl.atomic_add(dv_ptrs, dv.to(DV.type.element_ty))
    else:
        tl.store(dv_ptrs, dv.to(DV.type.element_ty))
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    if NATIVE_GQA:
        tl.atomic_add(dk_ptrs, dk.to(DK.type.element_ty))
    else:
        tl.store(dk_ptrs, dk.to(DK.type.element_ty))

    # Calculate dQ
    start_m = pid * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    if CAUSAL:
        # For causal attention, handle both masked and non-masked parts
        end_n = start_m + BLOCK_M2
        MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
        
        # Compute dQ for masked (diagonal) blocks
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(dq, q, K, V,
                          do, m, D,
                          sm_scale,
                          dropout_p, dropout_seed,
                          stride_tok, stride_d,
                          H, N_CTX,
                          BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
                          start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,
                          MASK=True,
                          USE_DROPOUT=USE_DROPOUT,
                          SEQ_LEN_K=SEQ_LEN_K,
                          SEQ_LEN_Q=SEQ_LEN_Q,
                          WINDOW_LEFT=WINDOW_LEFT,
                          WINDOW_RIGHT=WINDOW_RIGHT,
                          softcap=SOFTCAP,
                          alibi_slope=alibi_slope,
                          HAS_SOFTCAP=SOFTCAP > 0,
                          HAS_ALIBI=USE_ALIBI,
                          HAS_WINDOW=(WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0),
                          )
        end_n -= num_steps * MASK_BLOCK_N2
        
        # Compute dQ for non-masked blocks (if any)
        if end_n > 0:
            num_steps = end_n // BLOCK_N2
            dq = _attn_bwd_dq(dq, q, K, V,
                              do, m, D,
                              sm_scale,
                              dropout_p, dropout_seed,
                              stride_tok, stride_d,
                              H, N_CTX,
                              BLOCK_M2, BLOCK_N2, HEAD_DIM,
                              start_m, 0, num_steps,
                              MASK=False,
                              USE_DROPOUT=USE_DROPOUT,
                              SEQ_LEN_K=SEQ_LEN_K,
                              SEQ_LEN_Q=SEQ_LEN_Q,
                              WINDOW_LEFT=WINDOW_LEFT,
                              WINDOW_RIGHT=WINDOW_RIGHT,
                              softcap=SOFTCAP,
                              alibi_slope=alibi_slope,
                              HAS_SOFTCAP=SOFTCAP > 0,
                              HAS_ALIBI=USE_ALIBI,
                              HAS_WINDOW=(WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0),
                              )
    else:
        # For non-causal attention, process all tokens without masking
        num_steps = N_CTX // BLOCK_N2
        dq = _attn_bwd_dq(dq, q, K, V,
                          do, m, D,
                          sm_scale,
                          dropout_p, dropout_seed,
                          stride_tok, stride_d,
                          H, N_CTX,
                          BLOCK_M2, BLOCK_N2, HEAD_DIM,
                          start_m, 0, num_steps,
                          MASK=False,  # No masking for non-causal
                          USE_DROPOUT=USE_DROPOUT,
                          SEQ_LEN_K=SEQ_LEN_K,
                          SEQ_LEN_Q=SEQ_LEN_Q,
                          WINDOW_LEFT=WINDOW_LEFT,
                          WINDOW_RIGHT=WINDOW_RIGHT,
                          softcap=SOFTCAP,
                          alibi_slope=alibi_slope,
                          HAS_SOFTCAP=SOFTCAP > 0,
                          HAS_ALIBI=USE_ALIBI,
                          HAS_WINDOW=(WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0),
                          )

    # Write back dQ
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= sm_scale
    tl.store(dq_ptrs, dq.to(DQ.type.element_ty))


def _launch_dense_forward(
    q, k, v, causal, sm_scale, dropout_p, dropout_seed, seq_len_k,
    seq_len_q, kv_heads, softcap, alibi_slopes, window_left,
    window_right, use_alibi, *, store_lse
):
    head_dim_q, head_dim_k, head_dim_v = q.shape[-1], k.shape[-1], v.shape[-1]
    assert head_dim_q == head_dim_k == head_dim_v
    assert head_dim_k in {16, 32, 64, 128, 256}
    out = torch.empty_like(q)
    metadata = (
        torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]),
            device=q.device,
            dtype=torch.float32,
        )
        if store_lse
        else q.new_empty((1,), dtype=torch.float32)
    )
    stage = 3 if causal else 1
    use_dropout = dropout_p > 0.0
    grid = lambda args: (
        triton.cdiv(q.shape[2], args["BLOCK_M"]),
        q.shape[0] * q.shape[1],
        1,
    )
    _attn_fwd[grid](
        q, k, v, sm_scale, dropout_p, dropout_seed, metadata, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        q.shape[0], q.shape[1], q.shape[2], kv_heads,
        HEAD_DIM=head_dim_k,
        STAGE=stage,
        USE_DROPOUT=use_dropout,
        SEQ_LEN_K=seq_len_k,
        SEQ_LEN_Q=seq_len_q,
        SOFTCAP=softcap * 1.4426950408889634,
        ALIBI_SLOPES=alibi_slopes,
        USE_ALIBI=use_alibi,
        STORE_LSE=store_lse,
        WINDOW_LEFT=window_left,
        WINDOW_RIGHT=window_right,
    )
    return out, metadata


def _dense_backward(q, k, v, o, M, do, *, sm_scale, head_dim, causal, seq_len_k,
                    seq_len_q, kv_heads, dropout_p=0.0, dropout_seed=0,
                    softcap=0.0, alibi_slopes=None, use_alibi=False,
                    window_left=-1, window_right=-1):
    """Shared dense backward: direct-store dQ/dK/dV, no atomics.

    Both FP16 and INT8 forward paths reuse this backward. The INT8 path quantizes
    only QK^T in forward; backward keeps input-typed operands, accumulates dots
    in FP32, and casts dS to the input element type before dQ/dK. It therefore
    adds no INT8 gradient acceleration; the previous fused atomic variant was
    measured 2.4x slower than this path.
    """
    if not do.is_contiguous():
        do = do.contiguous()
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    BATCH, N_HEAD, N_CTX = q.shape[:3]
    PRE_BLOCK = 128
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = _backward_block_sizes()
    BLK_SLICE_FACTOR = 1
    pre_grid = (triton.cdiv(N_CTX, PRE_BLOCK), BATCH * N_HEAD)
    delta = torch.empty_like(M)
    alibi = alibi_slopes.reshape(-1) if use_alibi else q.new_zeros((BATCH * N_HEAD,), dtype=torch.float32)
    _attn_bwd_preprocess[pre_grid](
        o, do,  #
        delta,  #
        BATCH, N_HEAD, N_CTX,  #
        BLOCK_M=PRE_BLOCK, HEAD_DIM=head_dim  #
    )
    grid = (triton.cdiv(N_CTX, BLOCK_N1), 1, BATCH * N_HEAD)
    _attn_bwd[grid](
        q, k, v, sm_scale, do, dq, dk, dv,  #
        M, delta,  #
        dropout_p, dropout_seed,  #
        q.stride(0), q.stride(1), k.stride(0), q.stride(2), q.stride(3),  #
        N_HEAD, N_CTX, kv_heads,  #
        BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
        BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
        HEAD_DIM=head_dim,  #
        USE_DROPOUT=dropout_p > 0.0,  #
        CAUSAL=causal,      #
        SEQ_LEN_K=seq_len_k,
        SEQ_LEN_Q=seq_len_q,
        WINDOW_LEFT=window_left,
        WINDOW_RIGHT=window_right,
        SOFTCAP=softcap * 1.4426950408889634,
        ALIBI_SLOPES=alibi,
        USE_ALIBI=use_alibi,
        NATIVE_GQA=kv_heads != N_HEAD
    )
    return dq, dk, dv


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, dropout_p, dropout_seed,
                seq_len_k, seq_len_q, kv_heads, softcap, alibi_slopes,
                window_left, window_right, use_alibi):
        o, metadata = _launch_dense_forward(
            q, k, v, causal, sm_scale, dropout_p, dropout_seed,
            seq_len_k, seq_len_q, kv_heads, softcap, alibi_slopes,
            window_left, window_right, use_alibi, store_lse=True,
        )
        ctx.save_for_backward(q, k, v, o, metadata)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = q.shape[-1]
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.dropout_seed = dropout_seed
        ctx.use_dropout = dropout_p > 0.0
        ctx.seq_len_k = seq_len_k
        ctx.seq_len_q = seq_len_q
        ctx.kv_heads = kv_heads
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.use_alibi = use_alibi
        ctx.window_left = window_left
        ctx.window_right = window_right
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        dq, dk, dv = _dense_backward(
            q, k, v, o, M, do,
            sm_scale=ctx.sm_scale, head_dim=ctx.HEAD_DIM, causal=ctx.causal,
            seq_len_k=ctx.seq_len_k, seq_len_q=ctx.seq_len_q,
            kv_heads=ctx.kv_heads, dropout_p=ctx.dropout_p,
            dropout_seed=ctx.dropout_seed, softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes, use_alibi=ctx.use_alibi,
            window_left=ctx.window_left, window_right=ctx.window_right,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


def attention(q, k, v, causal=False, sm_scale=None, dropout_p=0.0, dropout_seed=None,
              seq_len_k=None, seq_len_q=None, kv_heads=None, softcap=0.0, alibi_slopes=None,
              window_left=-1, window_right=-1, use_alibi=None):
    """
    Flash attention implementation with dropout using Triton.
    
    Args:
        q: Query tensor of shape (batch_size, num_heads, seqlen_q, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seqlen_k, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seqlen_v, head_dim)
        causal: If True, applies causal masking
        sm_scale: Softmax scaling factor (if None, defaults to 1/sqrt(head_dim))
        dropout_p: Dropout probability (default: 0.0)
        dropout_seed: Random seed for dropout (if None, a random seed will be generated)
        seq_len_k: Original (unpadded) key sequence length. Used for masking padded
            positions in non-causal attention. If None, defaults to seqlen_q (no padding).
        
    Returns:
        output: Attention output of shape (batch_size, num_heads, seqlen_q, head_dim)
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    if q.shape[2] != k.shape[2] or k.shape[2] != v.shape[2]:
        raise ValueError("The low-level attention kernel requires equal sequence lengths")
    if q.shape[2] % 64 != 0:
        raise ValueError("The low-level attention kernel requires sequence lengths divisible by 64")
    if kv_heads is None:
        kv_heads = k.shape[1]
    if q.shape[1] % kv_heads != 0:
        raise ValueError("The number of query heads must be divisible by KV heads")
    if alibi_slopes is None:
        alibi_slopes = q.new_empty((1,), dtype=torch.float32)
        use_alibi = False
    elif use_alibi is None:
        use_alibi = bool(torch.any(alibi_slopes != 0))

    if dropout_seed is None:
        dropout_seed = random.randrange(2**31 - 1)

    if seq_len_k is None:
        seq_len_k = q.shape[2]
    if seq_len_q is None:
        seq_len_q = q.shape[2]

    needs_grad = torch.is_grad_enabled() and (
        q.requires_grad or k.requires_grad or v.requires_grad
    )
    if not needs_grad:
        return _launch_dense_forward(
            q, k, v, causal, sm_scale, dropout_p, dropout_seed,
            seq_len_k, seq_len_q, kv_heads, softcap, alibi_slopes,
            window_left, window_right, use_alibi, store_lse=False,
        )[0]
    return _attention.apply(
        q, k, v, causal, sm_scale, dropout_p, dropout_seed,
        seq_len_k, seq_len_q, kv_heads, softcap, alibi_slopes,
        window_left, window_right, use_alibi,
    )


def _varlen_launch_options(head_dim, device):
    """Choose conservative launch settings for pre-Ampere GPUs."""
    major, _ = torch.cuda.get_device_capability(device)
    num_warps = 4 if head_dim <= 64 else 8
    num_stages = 2 if major == 7 else 3
    return num_warps, num_stages


def _varlen_backward_block_size(head_dim, device):
    """Keep V100 variable-length backward within shared-memory limits."""
    major, _ = torch.cuda.get_device_capability(device)
    if major == 7:
        if head_dim >= 256:
            return 16
        if head_dim >= 128:
            return 32
    return 64


class _attention_varlen(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                max_seqlen_k, causal, sm_scale, dropout_p, dropout_seed,
                softcap, alibi_slopes, window_left, window_right, use_alibi):
        head_dim = q.shape[-1]
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        block_m = 64
        block_n = 64
        out = torch.empty_like(q)
        lse = torch.empty((q.shape[0], q_heads), device=q.device, dtype=torch.float32)
        use_dropout = dropout_p > 0.0
        batch = cu_seqlens_q.numel() - 1
        grid = (triton.cdiv(max_seqlen_q, block_m), batch, q_heads)
        num_warps, num_stages = _varlen_launch_options(head_dim, q.device)
        _varlen_fwd[grid](
            q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, alibi_slopes,
            sm_scale, dropout_p, dropout_seed,
            q.stride(0), k.stride(0), v.stride(0), out.stride(0),
            q_heads, kv_heads, max_seqlen_q, max_seqlen_k,
            BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=head_dim,
            CAUSAL=causal, USE_DROPOUT=use_dropout, USE_ALIBI=use_alibi,
            WINDOW_LEFT=window_left, WINDOW_RIGHT=window_right, SOFTCAP=softcap,
            num_warps=num_warps, num_stages=num_stages,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, alibi_slopes)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        ctx.dropout_p = dropout_p
        ctx.dropout_seed = dropout_seed
        ctx.softcap = softcap
        ctx.window_left = window_left
        ctx.window_right = window_right
        ctx.use_dropout = use_dropout
        ctx.use_alibi = use_alibi
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, alibi_slopes = ctx.saved_tensors
        if not do.is_contiguous():
            do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        block_m = _varlen_backward_block_size(q.shape[-1], q.device)
        block_n = block_m
        batch = cu_seqlens_q.numel() - 1
        grid_dq = (triton.cdiv(ctx.max_seqlen_q, block_m), batch, q.shape[1])
        grid_dkdv = (triton.cdiv(ctx.max_seqlen_k, block_n), batch, k.shape[1])
        num_warps, num_stages = _varlen_launch_options(q.shape[-1], q.device)
        _varlen_bwd[grid_dq](
            q, k, v, out, do, dq, lse,
            cu_seqlens_q, cu_seqlens_k, alibi_slopes,
            ctx.sm_scale, ctx.dropout_p, ctx.dropout_seed,
            q.stride(0), k.stride(0), v.stride(0), out.stride(0),
            q.shape[1], k.shape[1], ctx.max_seqlen_q, ctx.max_seqlen_k,
            BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=q.shape[-1],
            CAUSAL=ctx.causal, USE_DROPOUT=ctx.use_dropout,
            USE_ALIBI=ctx.use_alibi, WINDOW_LEFT=ctx.window_left,
            WINDOW_RIGHT=ctx.window_right, SOFTCAP=ctx.softcap,
            num_warps=num_warps, num_stages=num_stages,
        )
        _varlen_bwd_dkdv[grid_dkdv](
            q, k, v, out, do, dk, dv, lse,
            cu_seqlens_q, cu_seqlens_k, alibi_slopes,
            ctx.sm_scale, ctx.dropout_p, ctx.dropout_seed,
            q.stride(0), k.stride(0), v.stride(0), out.stride(0),
            q.shape[1], k.shape[1], ctx.max_seqlen_q, ctx.max_seqlen_k,
            BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=q.shape[-1],
            CAUSAL=ctx.causal, USE_DROPOUT=ctx.use_dropout,
            USE_ALIBI=ctx.use_alibi, WINDOW_LEFT=ctx.window_left,
            WINDOW_RIGHT=ctx.window_right,
            GROUP_SIZE=q.shape[1] // k.shape[1], SOFTCAP=ctx.softcap,
            num_warps=num_warps, num_stages=num_stages,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None


def attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                     max_seqlen_k, causal=False, sm_scale=None, dropout_p=0.0,
                     dropout_seed=None, softcap=0.0, alibi_slopes=None,
                     window_left=-1, window_right=-1, use_alibi=None):
    """Fused packed variable-length attention using one grid for all sequences."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    if dropout_seed is None:
        dropout_seed = random.randrange(2**31 - 1)
    if alibi_slopes is None:
        alibi_slopes = q.new_zeros((q.shape[0], q.shape[1]), dtype=torch.float32)
    if use_alibi is None:
        use_alibi = bool(torch.any(alibi_slopes != 0))
    return _attention_varlen.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        causal, sm_scale, dropout_p, dropout_seed, softcap, alibi_slopes,
        window_left, window_right, use_alibi)


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dropout_p", [0.0])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dropout_p, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    dropout_seed = 1337
    dout = torch.randn_like(q)
    
    # Set a fixed dropout mask for reference implementation
    torch.manual_seed(dropout_seed)
    dropout = torch.nn.Dropout(dropout_p)
    
    # Reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    
    # Apply dropout to attention weights
    p_dropped = dropout(p)
    
    ref_out = torch.matmul(p_dropped, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    
    # Reset seed for triton implementation
    torch.manual_seed(20)
    
    # Triton implementation
    tri_out = attention(q, k, v, causal, sm_scale, dropout_p, dropout_seed).half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    
    # Compare - note that with dropout, we expect some differences due to implementation details
    # but the test should still pass with reasonable tolerances
    atol = 1e-1 if dropout_p > 0 else 1e-2
    rtol = 1e-1 if dropout_p > 0 else 0.0
    
    assert torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dv, tri_dv, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dk, tri_dk, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dq, tri_dq, atol=atol, rtol=rtol)

HAS_FLASH = False
import math

BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True, False]:
        for dropout_p in [0.0]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(10, 15)],
                    line_arg="provider",
                    line_vals=["triton-fp16-dropout" if dropout_p > 0 else "triton-fp16"],
                    line_names=["Triton [FP16 + Dropout]" if dropout_p > 0 else "Triton [FP16]"],
                    styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                    ylabel="TFLOPS",
                    plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-dropout={dropout_p}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "HEAD_DIM": HEAD_DIM,
                        "mode": mode,
                        "causal": causal,
                        "dropout_p": dropout_p,
                    },
                ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, dropout_p=0.0, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        dropout_seed = 1337 if dropout_p > 0 else None
        
        fn = lambda: attention(q, k, v, causal, sm_scale, dropout_p, dropout_seed)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == "flash":
        from flash_attn_triton import flash_attn_func
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    # Test basic functionality
    test_op(1, 2, 1024, 64, True, 0.0)
    print("All tests passed!")
    
    # Run benchmarks
    bench_flash_attention.run(save_path=".", print_data=True)
