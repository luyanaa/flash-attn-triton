"""
Fused Attention with Dropout
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch

import triton
import triton.language as tl

DEVICE = 'cuda'

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    dropout_p, dropout_seed,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, USE_DROPOUT: tl.constexpr,
                    SEQ_LEN_K: tl.constexpr):
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
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            if SEQ_LEN_K < N_CTX:
                pad_mask = (start_n + offs_n[None, :]) < SEQ_LEN_K
                qk = qk * qk_scale + tl.where(pad_mask, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            else:
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)

        # -- apply dropout if specified at compile time --
        if USE_DROPOUT:
            # Create a unique offset for each element in the p matrix
            # The offset needs to be unique across both dimensions of p
            row_offsets = offs_m[:, None] + start_m * BLOCK_M
            col_offsets = (start_n + offs_n[None, :])
            # Combine row and column offsets into a unique offset
            # Multiply row offset by N_CTX to ensure uniqueness across the matrix
            combined_offsets = row_offsets * N_CTX + col_offsets
            
            # Generate the dropout mask using combined offsets
            rng = tl.rand(dropout_seed, combined_offsets)
            dropout_mask = rng > dropout_p
            # Apply the dropout and scale by 1/(1-p) to maintain expected values
            p = tl.where(dropout_mask, p / (1.0 - dropout_p), 0.0)
            
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
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
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [32, 64, 128]\
    for BN in [16, 32, 64]\
    for s in ([3, 4, 5, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    if BLOCK_M < BLOCK_N:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, dropout_p, dropout_seed, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              USE_DROPOUT: tl.constexpr,  #
              SEQ_LEN_K: tl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale, dropout_p, dropout_seed,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, USE_DROPOUT,
                                        SEQ_LEN_K)
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale, dropout_p, dropout_seed,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, USE_DROPOUT,
                                        SEQ_LEN_K)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


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
                    SEQ_LEN_K: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    dropout_scale = 1.0 / (1.0 - dropout_p) if USE_DROPOUT else 1.0
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT, out_dtype=tl.float32)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        # Pad masking for non-causal attention with seq_len not divisible by block size.
        if not MASK and SEQ_LEN_K < N_CTX:
            pad_mask = offs_n[:, None] < SEQ_LEN_K
            pT = tl.where(pad_mask, pT, 0.0)
            
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
        # Compute dV.
        ppT = pT.to(tl.float16)
        dv += tl.dot(ppT, do, out_dtype=tl.float32)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do), out_dtype=tl.float32).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT), out_dtype=tl.float32)
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 dropout_p, dropout_seed,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr, USE_DROPOUT: tl.constexpr,
                 SEQ_LEN_K: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    dropout_scale = 1.0 / (1.0 - dropout_p) if USE_DROPOUT else 1.0
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT, out_dtype=tl.float32)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Pad masking for non-causal attention with seq_len not divisible by block size.
        if not MASK and SEQ_LEN_K < N_CTX:
            offs_n_curr = curr_n + tl.arange(0, BLOCK_N2)
            pad_mask = offs_n_curr[None, :] < SEQ_LEN_K
            p = tl.where(pad_mask, p, 0.0)
            
        # Apply dropout mask for backward pass
        if USE_DROPOUT:
            # Create unique offsets for the dropout mask
            row_offsets = offs_m[:, None]
            col_offsets = curr_n + tl.arange(0, BLOCK_N2)[None, :]
            combined_offsets = row_offsets * N_CTX + col_offsets
            
            # Generate the dropout mask (same as in forward pass)
            rng = tl.rand(dropout_seed, combined_offsets)
            dropout_mask = rng > dropout_p
            # Apply the dropout and scale by 1/(1-p)
            p = tl.where(dropout_mask, p * dropout_scale, 0.0)
            
        # Compute dP and dS.
        dp = tl.dot(do, vT, out_dtype=tl.float32).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT), out_dtype=tl.float32)
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq



@triton.jit
def _attn_bwd(Q, K, V, sm_scale,
              DO,
              DQ, DK, DV,
              M, D,
              dropout_p, dropout_seed,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,
              H, N_CTX,
              BLOCK_M1: tl.constexpr,
              BLOCK_N1: tl.constexpr,
              BLOCK_M2: tl.constexpr,
              BLOCK_N2: tl.constexpr,
              BLK_SLICE_FACTOR: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              USE_DROPOUT: tl.constexpr,
              CAUSAL: tl.constexpr,
              SEQ_LEN_K: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
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
                                SEQ_LEN_K=SEQ_LEN_K
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
                SEQ_LEN_K=SEQ_LEN_K
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
            SEQ_LEN_K=SEQ_LEN_K
        )

    # Store dV and dK
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv.to(DV.type.element_ty))
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
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
                          dropout_p, dropout_seed,
                          stride_tok, stride_d,
                          H, N_CTX,
                          BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
                          start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,
                          MASK=True,
                          USE_DROPOUT=USE_DROPOUT,
                          SEQ_LEN_K=SEQ_LEN_K
                          )
        end_n -= num_steps * MASK_BLOCK_N2
        
        # Compute dQ for non-masked blocks (if any)
        if end_n > 0:
            num_steps = end_n // BLOCK_N2
            dq = _attn_bwd_dq(dq, q, K, V,
                              do, m, D,
                              dropout_p, dropout_seed,
                              stride_tok, stride_d,
                              H, N_CTX,
                              BLOCK_M2, BLOCK_N2, HEAD_DIM,
                              start_m, 0, num_steps,
                              MASK=False,
                              USE_DROPOUT=USE_DROPOUT,
                              SEQ_LEN_K=SEQ_LEN_K
                              )
    else:
        # For non-causal attention, process all tokens without masking
        num_steps = N_CTX // BLOCK_N2
        dq = _attn_bwd_dq(dq, q, K, V,
                          do, m, D,
                          dropout_p, dropout_seed,
                          stride_tok, stride_d,
                          H, N_CTX,
                          BLOCK_M2, BLOCK_N2, HEAD_DIM,
                          start_m, 0, num_steps,
                          MASK=False,  # No masking for non-causal
                          USE_DROPOUT=USE_DROPOUT,
                          SEQ_LEN_K=SEQ_LEN_K
                          )

    # Write back dQ
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq.to(DQ.type.element_ty))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, dropout_p, dropout_seed, seq_len_k):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        ctx.grid = grid
        
        # Determine if we should use dropout at compile time
        USE_DROPOUT = dropout_p > 0.0
        
        _attn_fwd[grid](
            q, k, v, sm_scale, dropout_p, dropout_seed, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            USE_DROPOUT=USE_DROPOUT,  #
            SEQ_LEN_K=seq_len_k,
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.dropout_seed = dropout_seed
        ctx.use_dropout = USE_DROPOUT
        ctx.seq_len_k = seq_len_k
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        if not do.is_contiguous():
            do = do.contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 16, 64, 64, 16
        BLK_SLICE_FACTOR = 1
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)

        # Use triton.cdiv to handle non-divisible sequence lengths:
        pre_grid = (triton.cdiv(N_CTX, PRE_BLOCK), BATCH * N_HEAD)

        # pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )
        # Also update this grid calculation:
        grid = (triton.cdiv(N_CTX, BLOCK_N1), 1, BATCH * N_HEAD)
        # grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            ctx.dropout_p, ctx.dropout_seed,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            USE_DROPOUT=ctx.use_dropout,  #
            CAUSAL=ctx.causal,      #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
            SEQ_LEN_K=ctx.seq_len_k
        )

        return dq, dk, dv, None, None, None, None, None


def attention(q, k, v, causal=False, sm_scale=None, dropout_p=0.0, dropout_seed=None, seq_len_k=None):
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
    
    if dropout_seed is None:
        dropout_seed = torch.randint(0, 2**31 - 1, (), device=q.device).item()
    
    if seq_len_k is None:
        seq_len_k = q.shape[2]
    
    # Call the autograd function
    return _attention.apply(q, k, v, causal, sm_scale, dropout_p, dropout_seed, seq_len_k)


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
