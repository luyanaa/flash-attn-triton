// Pure-CUDA Flash Attention (FP16) for Turing (sm_75) and Volta (sm_70).
// Uses nvcuda::wmma m16n16k16 fragments, which compile for both architectures.
// Structure follows FA2 (Tri Dao): forward = online softmax + causal mask;
// backward = split dQ kernel + dKV kernel.
//
// Buffers are non-overlapping (P is never written over S's fp32 storage) to
// avoid cross-thread shared-memory races in the per-row softmax passes.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <vector>

namespace fmha_cuda {

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ---------------------------------------------------------------------------
// FORWARD  (BLOCK_M=64, BLOCK_N=32)
// Each block: one (batch, head, m-block). Grid = (M/BM, B*H).
// smem: sQ[BM,D] fp16, sK[BN,D] fp16, sV[BN,D] fp16, sS[BM,BN] fp32,
//       sP[BM,BN] fp16, sPV[BM,D] fp32, sO[BM,D] fp32 (running output).
// ---------------------------------------------------------------------------
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128) fmha_fwd_kernel(
    const __half *__restrict__ Q, const __half *__restrict__ K,
    const __half *__restrict__ V, __half *__restrict__ O,
    float *__restrict__ LSE, const int M, const int N, const int seqlen_k,
    const float sm_scale, const bool causal) {
  const int tid = threadIdx.x;
  const int off_m = blockIdx.x * BLOCK_M;
  const long bh = blockIdx.y;
  const long qkv_off = bh * (long)M * HEAD_DIM;
  const __half *q_ptr = Q + qkv_off + (long)off_m * HEAD_DIM;
  const __half *k_base = K + qkv_off;
  const __half *v_base = V + qkv_off;

  extern __shared__ __half sdata[];
  __half *sQ = sdata;
  __half *sK = sQ + BLOCK_M * HEAD_DIM;
  __half *sV = sK + BLOCK_N * HEAD_DIM;
  float *sS = (float *)(sV + BLOCK_N * HEAD_DIM);          // [BM, BN] fp32
  __half *sP = (__half *)(sS + BLOCK_M * BLOCK_N);         // [BM, BN] fp16
  float *sPV = (float *)(sP + BLOCK_M * BLOCK_N);          // [BM, HEAD_DIM] fp32
  float *sO = sPV + BLOCK_M * HEAD_DIM;                    // [BM, HEAD_DIM] fp32
  __shared__ float sMax[BLOCK_M];
  __shared__ float sSum[BLOCK_M];
  __shared__ float sAlpha[BLOCK_M];

  {
    const int total = BLOCK_M * HEAD_DIM / 2;
    for (int i = tid; i < total; i += blockDim.x) {
      const int r = (i * 2) / HEAD_DIM;
      const int c = (i * 2) % HEAD_DIM;
      *reinterpret_cast<__half2 *>(&sQ[r * HEAD_DIM + c]) =
          *reinterpret_cast<const __half2 *>(q_ptr + r * HEAD_DIM + c);
    }
  }
  if (tid < BLOCK_M) {
    sMax[tid] = -CUDART_INF_F;
    sSum[tid] = 0.f;
  }
  for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) sO[i] = 0.f;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> bT_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> bR_frag;
  // acc is indexed [BLOCK_M/WMMA_M][HEAD_DIM/WMMA_N]; the QK loop only uses
  // the first BLOCK_N/WMMA_N columns (they are dead once S is stored to smem).
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      acc[BLOCK_M / WMMA_M][HEAD_DIM / WMMA_N];

  for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
    const __half *k_ptr = k_base + (long)n0 * HEAD_DIM;
    const __half *v_ptr = v_base + (long)n0 * HEAD_DIM;
    for (int i = tid; i < BLOCK_N * HEAD_DIM / 2; i += blockDim.x) {
      const int r = (i * 2) / HEAD_DIM;
      const int c = (i * 2) % HEAD_DIM;
      *reinterpret_cast<__half2 *>(&sK[r * HEAD_DIM + c]) =
          *reinterpret_cast<const __half2 *>(k_ptr + r * HEAD_DIM + c);
      *reinterpret_cast<__half2 *>(&sV[r * HEAD_DIM + c]) =
          *reinterpret_cast<const __half2 *>(v_ptr + r * HEAD_DIM + c);
    }
    __syncthreads();

    // S = Q @ K^T
    #pragma unroll
    for (int mt = 0; mt < BLOCK_M / WMMA_M; ++mt) {
      #pragma unroll
      for (int nt = 0; nt < BLOCK_N / WMMA_N; ++nt) {
        wmma::fill_fragment(acc[mt][nt], 0.0f);
        #pragma unroll
        for (int kt = 0; kt < HEAD_DIM / WMMA_K; ++kt) {
          wmma::load_matrix_sync(a_frag, &sQ[mt * WMMA_M * HEAD_DIM + kt * WMMA_K], HEAD_DIM);
          wmma::load_matrix_sync(bT_frag, &sK[nt * WMMA_N * HEAD_DIM + kt * WMMA_K], HEAD_DIM);
          wmma::mma_sync(acc[mt][nt], a_frag, bT_frag, acc[mt][nt]);
        }
        wmma::store_matrix_sync(&sS[mt * WMMA_M * BLOCK_N + nt * WMMA_N], acc[mt][nt],
                                BLOCK_N, wmma::mem_row_major);
      }
    }
    __syncthreads();

    // Online softmax. Two threads per row (BLOCK_M * 2 <= blockDim.x).
    // Each computes a partial max over half the columns, then a partial sum
    // over the same half; the row's two partials are reduced via a shared
    // scratch slot indexed by (row, half).
    {
      const int rows_per_half = BLOCK_N / 2;
      const int tid_local = tid;
      if (tid_local < 2 * BLOCK_M) {
        const int row = tid_local / 2;
        const int half = tid_local % 2;
        const int qpos = off_m + row;
        float rowmax = -CUDART_INF_F;
        for (int j = half * rows_per_half; j < (half + 1) * rows_per_half; ++j) {
          float v = sS[row * BLOCK_N + j] * sm_scale;
          const int kpos = n0 + j;
          const bool valid = (qpos < M) && (kpos < seqlen_k) && (!causal || qpos >= kpos);
          if (!valid) v = -CUDART_INF_F;
          rowmax = fmaxf(rowmax, v);
        }
        // combine the two halves' maxima
        const float rowmax_full = fmaxf(rowmax, __shfl_xor_sync(0xffffffff, rowmax, 1));
        __syncthreads();
        const float old_max = sMax[row];
        const float new_max = fmaxf(old_max, rowmax_full);
        const float alpha = __expf(old_max - new_max);
        float sum = 0.f;
        __half *sProw = &sP[row * BLOCK_N];
        for (int j = half * rows_per_half; j < (half + 1) * rows_per_half; ++j) {
          float v = sS[row * BLOCK_N + j] * sm_scale;
          const int kpos = n0 + j;
          const bool valid = (qpos < M) && (kpos < seqlen_k) && (!causal || qpos >= kpos);
          if (!valid) v = -CUDART_INF_F;
          const float p = __expf(v - new_max);
          sum += p;
          sProw[j] = __float2half(p);
        }
        // combine the two halves' sums; only half 0 commits running stats
        const float sum_full = sum + __shfl_xor_sync(0xffffffff, sum, 1);
        if (half == 0) {
          sSum[row] = sSum[row] * alpha + sum_full;
          sMax[row] = new_max;
          sAlpha[row] = alpha;
        }
      }
      __syncthreads();
    }

    // PV = P @ V into register acc (A=P fp16 [BM,BN], B=V fp16 [BN,D]).
    #pragma unroll
    for (int mt = 0; mt < BLOCK_M / WMMA_M; ++mt) {
      #pragma unroll
      for (int dt = 0; dt < HEAD_DIM / WMMA_N; ++dt) {
        wmma::fill_fragment(acc[mt][dt], 0.0f);
        #pragma unroll
        for (int nt = 0; nt < BLOCK_N / WMMA_M; ++nt) {
          wmma::load_matrix_sync(a_frag, &sP[mt * WMMA_M * BLOCK_N + nt * WMMA_M], BLOCK_N);
          wmma::load_matrix_sync(bR_frag, &sV[nt * WMMA_N * HEAD_DIM + dt * WMMA_K], HEAD_DIM);
          wmma::mma_sync(acc[mt][dt], a_frag, bR_frag, acc[mt][dt]);
        }
        wmma::store_matrix_sync(&sPV[mt * WMMA_M * HEAD_DIM + dt * WMMA_N], acc[mt][dt],
                                HEAD_DIM, wmma::mem_row_major);
      }
    }
    __syncthreads();

    // sO = sO * alpha + PV  (parallel across all threads)
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
      const int r = i / HEAD_DIM;
      sO[i] = sO[i] * sAlpha[r] + sPV[i];
    }
    __syncthreads();
  }

  // Store O / sum and LSE = max + log(sum).
  if (tid < BLOCK_M) {
    const int row = tid;
    const int qpos = off_m + row;
    const float l = sSum[row];
    const float m = sMax[row];
    if (qpos < M) {
      LSE[bh * (long)M + qpos] = m + logf(l);
      const float inv = 1.0f / l;
      __half *o_ptr = O + bh * (long)M * HEAD_DIM + (long)qpos * HEAD_DIM;
      const float *orow = &sO[row * HEAD_DIM];
      for (int d = 0; d < HEAD_DIM; d += 2)
        *reinterpret_cast<__half2 *>(o_ptr + d) =
            __floats2half2_rn(orow[d] * inv, orow[d + 1] * inv);
    }
  }
}

// ---------------------------------------------------------------------------
// BACKWARD dKV: per (b, h, key-block). dK and dV accumulate in wmma fragments.
// smem: sK,sV [BN,D], sQ,sDO [BM,D] (fp16), sST [BN,BM] fp32 (S^T),
//       sPT [BN,BM] fp16 (P^T then dS^T), sDPT [BN,BM] fp32 (dP^T).
// ---------------------------------------------------------------------------
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128) fmha_bwd_dkdv_kernel(
    const __half *__restrict__ Q, const __half *__restrict__ K,
    const __half *__restrict__ V, const __half *__restrict__ DO,
    const float *__restrict__ LSE, const float *__restrict__ Delta,
    __half *__restrict__ DK, __half *__restrict__ DV,
    const int M, const int N, const int seqlen_k, const float sm_scale, const bool causal) {
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * BLOCK_N;
  const long bh = blockIdx.y;
  const long qkv_off = bh * (long)M * HEAD_DIM;
  if (n0 >= seqlen_k) return;

  extern __shared__ __half sdata[];
  __half *sK = sdata;                                  // [BLOCK_N, HEAD_DIM]
  __half *sV = sK + BLOCK_N * HEAD_DIM;                // [BLOCK_N, HEAD_DIM]
  __half *sQ = sV + BLOCK_N * HEAD_DIM;                // [BLOCK_M, HEAD_DIM]
  __half *sDO = sQ + BLOCK_M * HEAD_DIM;               // [BLOCK_M, HEAD_DIM]
  float *sST = (float *)(sDO + BLOCK_M * HEAD_DIM);    // [BLOCK_N, BLOCK_M] fp32
  __half *sPT = (__half *)(sST + BLOCK_N * BLOCK_M);   // [BLOCK_N, BLOCK_M] fp16
  float *sDPT = (float *)(sPT + BLOCK_N * BLOCK_M);    // [BLOCK_N, BLOCK_M] fp32
  __shared__ float sLse[BLOCK_M];
  __shared__ float sDelta[BLOCK_M];

  const __half *k_ptr = K + qkv_off + (long)n0 * HEAD_DIM;
  const __half *v_ptr = V + qkv_off + (long)n0 * HEAD_DIM;
  for (int i = tid; i < BLOCK_N * HEAD_DIM / 2; i += blockDim.x) {
    const int r = (i * 2) / HEAD_DIM;
    const int c = (i * 2) % HEAD_DIM;
    *reinterpret_cast<__half2 *>(&sK[r * HEAD_DIM + c]) =
        *reinterpret_cast<const __half2 *>(k_ptr + r * HEAD_DIM + c);
    *reinterpret_cast<__half2 *>(&sV[r * HEAD_DIM + c]) =
        *reinterpret_cast<const __half2 *>(v_ptr + r * HEAD_DIM + c);
  }

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      dv_acc[BLOCK_N / WMMA_M][HEAD_DIM / WMMA_N];
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      dk_acc[BLOCK_N / WMMA_M][HEAD_DIM / WMMA_N];
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> bR_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> bT_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      sT_acc[BLOCK_N / WMMA_M][BLOCK_M / WMMA_N];

  #pragma unroll
  for (int i = 0; i < BLOCK_N / WMMA_M; ++i)
    #pragma unroll
    for (int j = 0; j < HEAD_DIM / WMMA_N; ++j) {
      wmma::fill_fragment(dv_acc[i][j], 0.0f);
      wmma::fill_fragment(dk_acc[i][j], 0.0f);
    }

  for (int m0 = 0; m0 < M; m0 += BLOCK_M) {
    const __half *q_ptr = Q + qkv_off + (long)m0 * HEAD_DIM;
    const __half *do_ptr = DO + qkv_off + (long)m0 * HEAD_DIM;
    for (int i = tid; i < BLOCK_M * HEAD_DIM / 2; i += blockDim.x) {
      const int r = (i * 2) / HEAD_DIM;
      const int c = (i * 2) % HEAD_DIM;
      *reinterpret_cast<__half2 *>(&sQ[r * HEAD_DIM + c]) =
          *reinterpret_cast<const __half2 *>(q_ptr + r * HEAD_DIM + c);
      *reinterpret_cast<__half2 *>(&sDO[r * HEAD_DIM + c]) =
          *reinterpret_cast<const __half2 *>(do_ptr + r * HEAD_DIM + c);
    }
    if (tid < BLOCK_M) {
      sLse[tid] = LSE[bh * (long)M + m0 + tid];
      sDelta[tid] = Delta[bh * (long)M + m0 + tid];
    }
    __syncthreads();

    // S^T = K @ Q^T -> [BLOCK_N, BLOCK_M]
    #pragma unroll
    for (int nt = 0; nt < BLOCK_N / WMMA_M; ++nt) {
      #pragma unroll
      for (int mt = 0; mt < BLOCK_M / WMMA_N; ++mt) {
        wmma::fill_fragment(sT_acc[nt][mt], 0.0f);
        #pragma unroll
        for (int kt = 0; kt < HEAD_DIM / WMMA_K; ++kt) {
          wmma::load_matrix_sync(a_frag, &sK[nt * WMMA_M * HEAD_DIM + kt * WMMA_K], HEAD_DIM);
          wmma::load_matrix_sync(bT_frag, &sQ[mt * WMMA_N * HEAD_DIM + kt * WMMA_K], HEAD_DIM);
          wmma::mma_sync(sT_acc[nt][mt], a_frag, bT_frag, sT_acc[nt][mt]);
        }
        wmma::store_matrix_sync(&sST[nt * WMMA_M * BLOCK_M + mt * WMMA_N], sT_acc[nt][mt],
                                BLOCK_M, wmma::mem_row_major);
      }
    }
    __syncthreads();

    // P^T = exp(S^T*sm_scale - LSE[m]) masked; fp16 into sPT.
    if (tid < BLOCK_N) {
      __half *sPTrow = &sPT[tid * BLOCK_M];
      const int kpos = n0 + tid;
      for (int j = 0; j < BLOCK_M; ++j) {
        const int qpos = m0 + j;
        const bool valid = (qpos < M) && (kpos < seqlen_k) && (!causal || qpos >= kpos);
        const float p = valid ? __expf(sST[tid * BLOCK_M + j] * sm_scale - sLse[j]) : 0.f;
        sPTrow[j] = __float2half(p);
      }
    }
    __syncthreads();

    // dV += P^T @ dO  (P^T fp16, dO fp16)
    #pragma unroll
    for (int nt = 0; nt < BLOCK_N / WMMA_M; ++nt) {
      #pragma unroll
      for (int dt = 0; dt < HEAD_DIM / WMMA_N; ++dt) {
        #pragma unroll
        for (int mt = 0; mt < BLOCK_M / WMMA_M; ++mt) {
          wmma::load_matrix_sync(a_frag, &sPT[nt * WMMA_M * BLOCK_M + mt * WMMA_M], BLOCK_M);
          wmma::load_matrix_sync(bR_frag, &sDO[mt * WMMA_M * HEAD_DIM + dt * WMMA_N], HEAD_DIM);
          wmma::mma_sync(dv_acc[nt][dt], a_frag, bR_frag, dv_acc[nt][dt]);
        }
      }
    }
    __syncthreads();

    // dP^T = V @ dO^T -> sDPT (fp32). P^T still in sPT.
    #pragma unroll
    for (int nt = 0; nt < BLOCK_N / WMMA_M; ++nt) {
      #pragma unroll
      for (int mt = 0; mt < BLOCK_M / WMMA_N; ++mt) {
        wmma::fill_fragment(sT_acc[nt][mt], 0.0f);
        #pragma unroll
        for (int kt = 0; kt < HEAD_DIM / WMMA_K; ++kt) {
          wmma::load_matrix_sync(a_frag, &sV[nt * WMMA_M * HEAD_DIM + kt * WMMA_K], HEAD_DIM);
          wmma::load_matrix_sync(bT_frag, &sDO[mt * WMMA_N * HEAD_DIM + kt * WMMA_K], HEAD_DIM);
          wmma::mma_sync(sT_acc[nt][mt], a_frag, bT_frag, sT_acc[nt][mt]);
        }
        wmma::store_matrix_sync(&sDPT[nt * WMMA_M * BLOCK_M + mt * WMMA_N], sT_acc[nt][mt],
                                BLOCK_M, wmma::mem_row_major);
      }
    }
    __syncthreads();

    // dS^T = P^T * (dP^T - delta[m]) in fp16 into sPT.
    if (tid < BLOCK_N) {
      __half *sPTrow = &sPT[tid * BLOCK_M];
      for (int j = 0; j < BLOCK_M; ++j) {
        const float p = __half2float(sPTrow[j]);
        const float dst = p * (sDPT[tid * BLOCK_M + j] - sDelta[j]);
        sPTrow[j] = __float2half(dst);
      }
    }
    __syncthreads();

    // dK += dS^T @ Q  (dS^T fp16 in sPT; Q fp16)
    #pragma unroll
    for (int nt = 0; nt < BLOCK_N / WMMA_M; ++nt) {
      #pragma unroll
      for (int dt = 0; dt < HEAD_DIM / WMMA_N; ++dt) {
        #pragma unroll
        for (int mt = 0; mt < BLOCK_M / WMMA_M; ++mt) {
          wmma::load_matrix_sync(a_frag, &sPT[nt * WMMA_M * BLOCK_M + mt * WMMA_M], BLOCK_M);
          wmma::load_matrix_sync(bR_frag, &sQ[mt * WMMA_M * HEAD_DIM + dt * WMMA_N], HEAD_DIM);
          wmma::mma_sync(dk_acc[nt][dt], a_frag, bR_frag, dk_acc[nt][dt]);
        }
      }
    }
    __syncthreads();
  }

  // Store dK (x sm_scale) and dV via sST staging.
  __half *dk_ptr = DK + qkv_off + (long)n0 * HEAD_DIM;
  __half *dv_ptr = DV + qkv_off + (long)n0 * HEAD_DIM;
  const int valid_rows = min(BLOCK_N, seqlen_k - n0);
  {
    #pragma unroll
    for (int nt = 0; nt < BLOCK_N / WMMA_M; ++nt)
      #pragma unroll
      for (int dt = 0; dt < HEAD_DIM / WMMA_N; ++dt)
        wmma::store_matrix_sync(&sST[nt * WMMA_M * HEAD_DIM + dt * WMMA_N], dk_acc[nt][dt],
                                HEAD_DIM, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < valid_rows * HEAD_DIM; i += blockDim.x)
      dk_ptr[i] = __float2half(sST[i] * sm_scale);
  }
  __syncthreads();
  {
    #pragma unroll
    for (int nt = 0; nt < BLOCK_N / WMMA_M; ++nt)
      #pragma unroll
      for (int dt = 0; dt < HEAD_DIM / WMMA_N; ++dt)
        wmma::store_matrix_sync(&sST[nt * WMMA_M * HEAD_DIM + dt * WMMA_N], dv_acc[nt][dt],
                                HEAD_DIM, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < valid_rows * HEAD_DIM; i += blockDim.x)
      dv_ptr[i] = __float2half(sST[i]);
  }
}

// ---------------------------------------------------------------------------
// BACKWARD dQ: per (b, h, q-block).
// smem: sQ,sDO [BM,D], sK,sV [BN,D] (fp16), sS [BM,BN] fp32 (S),
//       sP [BM,BN] fp16 (P then dS), sDP [BM,BN] fp32 (dP).
// ---------------------------------------------------------------------------
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128) fmha_bwd_dq_kernel(
    const __half *__restrict__ Q, const __half *__restrict__ K,
    const __half *__restrict__ V, const __half *__restrict__ DO,
    const float *__restrict__ LSE, const float *__restrict__ Delta,
    __half *__restrict__ DQ, const int M, const int N, const int seqlen_k,
    const float sm_scale, const bool causal) {
  const int tid = threadIdx.x;
  const int m0 = blockIdx.x * BLOCK_M;
  const long bh = blockIdx.y;
  const long qkv_off = bh * (long)M * HEAD_DIM;

  extern __shared__ __half sdata[];
  __half *sQ = sdata;                                 // [BLOCK_M, HEAD_DIM]
  __half *sDO = sQ + BLOCK_M * HEAD_DIM;              // [BLOCK_M, HEAD_DIM]
  __half *sK = sDO + BLOCK_M * HEAD_DIM;              // [BLOCK_N, HEAD_DIM]
  __half *sV = sK + BLOCK_N * HEAD_DIM;               // [BLOCK_N, HEAD_DIM]
  float *sS = (float *)(sV + BLOCK_N * HEAD_DIM);     // [BLOCK_M, BLOCK_N] fp32
  __half *sP = (__half *)(sS + BLOCK_M * BLOCK_N);    // [BLOCK_M, BLOCK_N] fp16
  float *sDP = (float *)(sP + BLOCK_M * BLOCK_N);     // [BLOCK_M, BLOCK_N] fp32
  __shared__ float sLse[BLOCK_M];
  __shared__ float sDelta[BLOCK_M];

  const __half *q_ptr = Q + qkv_off + (long)m0 * HEAD_DIM;
  const __half *do_ptr = DO + qkv_off + (long)m0 * HEAD_DIM;
  for (int i = tid; i < BLOCK_M * HEAD_DIM / 2; i += blockDim.x) {
    const int r = (i * 2) / HEAD_DIM;
    const int c = (i * 2) % HEAD_DIM;
    *reinterpret_cast<__half2 *>(&sQ[r * HEAD_DIM + c]) =
        *reinterpret_cast<const __half2 *>(q_ptr + r * HEAD_DIM + c);
    *reinterpret_cast<__half2 *>(&sDO[r * HEAD_DIM + c]) =
        *reinterpret_cast<const __half2 *>(do_ptr + r * HEAD_DIM + c);
  }
  if (tid < BLOCK_M) {
    sLse[tid] = LSE[bh * (long)M + m0 + tid];
    sDelta[tid] = Delta[bh * (long)M + m0 + tid];
  }

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      dq_acc[BLOCK_M / WMMA_M][HEAD_DIM / WMMA_N];
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> bR_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> bT_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      s_acc[BLOCK_M / WMMA_M][BLOCK_N / WMMA_N];
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      dp_acc[BLOCK_M / WMMA_M][BLOCK_N / WMMA_N];

  #pragma unroll
  for (int i = 0; i < BLOCK_M / WMMA_M; ++i)
    #pragma unroll
    for (int j = 0; j < HEAD_DIM / WMMA_N; ++j)
      wmma::fill_fragment(dq_acc[i][j], 0.0f);

  for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
    const __half *k_ptr = K + qkv_off + (long)n0 * HEAD_DIM;
    const __half *v_ptr = V + qkv_off + (long)n0 * HEAD_DIM;
    for (int i = tid; i < BLOCK_N * HEAD_DIM / 2; i += blockDim.x) {
      const int r = (i * 2) / HEAD_DIM;
      const int c = (i * 2) % HEAD_DIM;
      *reinterpret_cast<__half2 *>(&sK[r * HEAD_DIM + c]) =
          *reinterpret_cast<const __half2 *>(k_ptr + r * HEAD_DIM + c);
      *reinterpret_cast<__half2 *>(&sV[r * HEAD_DIM + c]) =
          *reinterpret_cast<const __half2 *>(v_ptr + r * HEAD_DIM + c);
    }
    __syncthreads();

    // S = Q @ K^T -> sS
    #pragma unroll
    for (int mt = 0; mt < BLOCK_M / WMMA_M; ++mt) {
      #pragma unroll
      for (int nt = 0; nt < BLOCK_N / WMMA_N; ++nt) {
        wmma::fill_fragment(s_acc[mt][nt], 0.0f);
        #pragma unroll
        for (int kt = 0; kt < HEAD_DIM / WMMA_K; ++kt) {
          wmma::load_matrix_sync(a_frag, &sQ[mt * WMMA_M * HEAD_DIM + kt * WMMA_K], HEAD_DIM);
          wmma::load_matrix_sync(bT_frag, &sK[nt * WMMA_N * HEAD_DIM + kt * WMMA_K], HEAD_DIM);
          wmma::mma_sync(s_acc[mt][nt], a_frag, bT_frag, s_acc[mt][nt]);
        }
        wmma::store_matrix_sync(&sS[mt * WMMA_M * BLOCK_N + nt * WMMA_N], s_acc[mt][nt],
                                BLOCK_N, wmma::mem_row_major);
      }
    }
    __syncthreads();

    // P = exp(S*sm_scale - LSE) masked; fp16 into sP.
    if (tid < BLOCK_M) {
      __half *sProw = &sP[tid * BLOCK_N];
      const int qpos = m0 + tid;
      for (int j = 0; j < BLOCK_N; ++j) {
        const int kpos = n0 + j;
        const bool valid = (qpos < M) && (kpos < seqlen_k) && (!causal || qpos >= kpos);
        const float p = valid ? __expf(sS[tid * BLOCK_N + j] * sm_scale - sLse[tid]) : 0.f;
        sProw[j] = __float2half(p);
      }
    }
    __syncthreads();

    // dP = dO @ V^T -> sDP (fp32)
    #pragma unroll
    for (int mt = 0; mt < BLOCK_M / WMMA_M; ++mt) {
      #pragma unroll
      for (int nt = 0; nt < BLOCK_N / WMMA_N; ++nt) {
        wmma::fill_fragment(dp_acc[mt][nt], 0.0f);
        #pragma unroll
        for (int kt = 0; kt < HEAD_DIM / WMMA_K; ++kt) {
          wmma::load_matrix_sync(a_frag, &sDO[mt * WMMA_M * HEAD_DIM + kt * WMMA_K], HEAD_DIM);
          wmma::load_matrix_sync(bT_frag, &sV[nt * WMMA_N * HEAD_DIM + kt * WMMA_K], HEAD_DIM);
          wmma::mma_sync(dp_acc[mt][nt], a_frag, bT_frag, dp_acc[mt][nt]);
        }
        wmma::store_matrix_sync(&sDP[mt * WMMA_M * BLOCK_N + nt * WMMA_N], dp_acc[mt][nt],
                                BLOCK_N, wmma::mem_row_major);
      }
    }
    __syncthreads();

    // dS = P * (dP - delta) in fp16 into sP.
    if (tid < BLOCK_M) {
      __half *sProw = &sP[tid * BLOCK_N];
      for (int j = 0; j < BLOCK_N; ++j) {
        const float p = __half2float(sProw[j]);
        const float ds = p * (sDP[tid * BLOCK_N + j] - sDelta[tid]);
        sProw[j] = __float2half(ds);
      }
    }
    __syncthreads();

    // dQ += dS @ K  (dS fp16 in sP; K fp16)
    #pragma unroll
    for (int mt = 0; mt < BLOCK_M / WMMA_M; ++mt) {
      #pragma unroll
      for (int dt = 0; dt < HEAD_DIM / WMMA_N; ++dt) {
        #pragma unroll
        for (int nt = 0; nt < BLOCK_N / WMMA_M; ++nt) {
          wmma::load_matrix_sync(a_frag, &sP[mt * WMMA_M * BLOCK_N + nt * WMMA_M], BLOCK_N);
          wmma::load_matrix_sync(bR_frag, &sK[nt * WMMA_N * HEAD_DIM + dt * WMMA_K], HEAD_DIM);
          wmma::mma_sync(dq_acc[mt][dt], a_frag, bR_frag, dq_acc[mt][dt]);
        }
      }
    }
    __syncthreads();
  }

  // Store dQ * sm_scale via sS staging ([BM, HEAD_DIM] fp32; region overlaps
  // sS+sP+sDP which are dead after the loop).
  __half *dq_ptr = DQ + qkv_off + (long)m0 * HEAD_DIM;
  const int valid_rows = min(BLOCK_M, M - m0);
  {
    float *st = sS;
    #pragma unroll
    for (int mt = 0; mt < BLOCK_M / WMMA_M; ++mt)
      #pragma unroll
      for (int dt = 0; dt < HEAD_DIM / WMMA_N; ++dt)
        wmma::store_matrix_sync(&st[mt * WMMA_M * HEAD_DIM + dt * WMMA_N], dq_acc[mt][dt],
                                HEAD_DIM, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < valid_rows * HEAD_DIM; i += blockDim.x)
      dq_ptr[i] = __float2half(st[i] * sm_scale);
  }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
constexpr int kBlockM = 64;
constexpr int kBlockN = 32;  // forward
constexpr int kBlockNBwd = 32;  // backward dKV / dQ

std::vector<torch::Tensor> fmha_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                        double sm_scale, bool causal) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf, "fmha_cuda forward requires fp16 CUDA");
  TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, M, D]");
  const int B = q.size(0), H = q.size(1), M = q.size(2), D = q.size(3);
  TORCH_CHECK(D == 64, "fmha_cuda supports head_dim 64");
  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
  const int pad = (kBlockM - M % kBlockM) % kBlockM;
  const int Mp = M + pad;
  auto qp = torch::zeros({B, H, Mp, D}, q.options());
  auto kp = torch::zeros({B, H, Mp, D}, q.options());
  auto vp = torch::zeros({B, H, Mp, D}, q.options());
  qp.narrow(2, 0, M).copy_(qc);
  kp.narrow(2, 0, M).copy_(kc);
  vp.narrow(2, 0, M).copy_(vc);
  auto out = torch::empty({B, H, Mp, D}, q.options());
  auto lse = torch::empty({B, H, Mp}, q.options().dtype(at::kFloat));

  // smem: sQ + sK + sV + sP (fp16), sS + sPV + sO (fp32)
  size_t shmem = (kBlockM * D + 2 * kBlockN * D + kBlockM * kBlockN) * sizeof(__half) +
                 (kBlockM * kBlockN + kBlockM * D + kBlockM * D) * sizeof(float);
  // Dynamic + static (sMax/sSum/sAlpha) must fit 64KB; use a safe budget.
  const int static_bytes = 3 * kBlockM * sizeof(float);
  cudaError_t attr_err = cudaFuncSetAttribute(
      fmha_fwd_kernel<kBlockM, kBlockN, 64>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024 - static_bytes);
  TORCH_CHECK(attr_err == cudaSuccess,
              "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));
  dim3 grid((Mp + kBlockM - 1) / kBlockM, B * H);
  auto stream = at::cuda::getCurrentCUDAStream();
  fmha_fwd_kernel<kBlockM, kBlockN, 64><<<grid, 128, shmem, stream>>>(
      reinterpret_cast<const __half *>(qp.data_ptr()),
      reinterpret_cast<const __half *>(kp.data_ptr()),
      reinterpret_cast<const __half *>(vp.data_ptr()),
      reinterpret_cast<__half *>(out.data_ptr()),
      reinterpret_cast<float *>(lse.data_ptr()),
      Mp, Mp, M, (float)sm_scale, causal);
  C10_CUDA_CHECK(cudaGetLastError());
  return {out.narrow(2, 0, M).contiguous(), lse.narrow(2, 0, M)};
}

std::vector<torch::Tensor> fmha_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                         torch::Tensor o, torch::Tensor do_,
                                         torch::Tensor lse, double sm_scale, bool causal) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf, "fmha_cuda backward requires fp16 CUDA");
  const int B = q.size(0), H = q.size(1), M = q.size(2), D = q.size(3);
  TORCH_CHECK(D == 64, "fmha_cuda supports head_dim 64");
  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
  auto oc = o.contiguous(), doc = do_.contiguous();
  auto lsec = lse.contiguous();
  const int pad = (kBlockM - M % kBlockM) % kBlockM;
  const int Mp = M + pad;
  auto qp = torch::zeros({B, H, Mp, D}, q.options());
  auto kp = torch::zeros({B, H, Mp, D}, q.options());
  auto vp = torch::zeros({B, H, Mp, D}, q.options());
  auto op = torch::zeros({B, H, Mp, D}, q.options());
  auto dop = torch::zeros({B, H, Mp, D}, q.options());
  qp.narrow(2, 0, M).copy_(qc);
  kp.narrow(2, 0, M).copy_(kc);
  vp.narrow(2, 0, M).copy_(vc);
  op.narrow(2, 0, M).copy_(oc);
  dop.narrow(2, 0, M).copy_(doc);
  auto lsep = torch::full({B, H, Mp}, -INFINITY, q.options().dtype(at::kFloat));
  lsep.narrow(2, 0, M).copy_(lsec);

  auto dq = torch::zeros({B, H, Mp, D}, q.options());
  auto dk = torch::zeros({B, H, Mp, D}, q.options());
  auto dv = torch::zeros({B, H, Mp, D}, q.options());
  auto delta = (doc * oc).sum(-1).to(at::kFloat);
  auto deltap = torch::zeros({B, H, Mp}, delta.options());
  deltap.narrow(2, 0, M).copy_(delta);

  auto stream = at::cuda::getCurrentCUDAStream();
  {
    // dKV smem: sK + sV + sQ + sDO (fp16) + sST + sDPT (fp32) + sPT (fp16)
    size_t shmem = (2 * kBlockNBwd * D + 2 * kBlockM * D + kBlockNBwd * kBlockM) * sizeof(__half) +
                   (kBlockNBwd * kBlockM + kBlockNBwd * kBlockM) * sizeof(float);
    cudaFuncSetAttribute(fmha_bwd_dkdv_kernel<kBlockM, kBlockNBwd, 64>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         64 * 1024 - 2 * kBlockM * sizeof(float));
    dim3 grid((Mp + kBlockNBwd - 1) / kBlockNBwd, B * H);
    fmha_bwd_dkdv_kernel<kBlockM, kBlockNBwd, 64><<<grid, 128, shmem, stream>>>(
        reinterpret_cast<const __half *>(qp.data_ptr()),
        reinterpret_cast<const __half *>(kp.data_ptr()),
        reinterpret_cast<const __half *>(vp.data_ptr()),
        reinterpret_cast<const __half *>(dop.data_ptr()),
        reinterpret_cast<const float *>(lsep.data_ptr()),
        reinterpret_cast<const float *>(deltap.data_ptr()),
        reinterpret_cast<__half *>(dk.data_ptr()),
        reinterpret_cast<__half *>(dv.data_ptr()),
        Mp, Mp, M, (float)sm_scale, causal);
    C10_CUDA_CHECK(cudaGetLastError());
  }
  {
    // dQ smem: sQ + sDO + sK + sV (fp16) + sS + sDP (fp32) + sP (fp16)
    size_t shmem = (2 * kBlockM * D + 2 * kBlockNBwd * D + kBlockM * kBlockNBwd) * sizeof(__half) +
                   (kBlockM * kBlockNBwd + kBlockM * kBlockNBwd) * sizeof(float);
    cudaFuncSetAttribute(fmha_bwd_dq_kernel<kBlockM, kBlockNBwd, 64>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         64 * 1024 - 2 * kBlockM * sizeof(float));
    dim3 grid((Mp + kBlockM - 1) / kBlockM, B * H);
    fmha_bwd_dq_kernel<kBlockM, kBlockNBwd, 64><<<grid, 128, shmem, stream>>>(
        reinterpret_cast<const __half *>(qp.data_ptr()),
        reinterpret_cast<const __half *>(kp.data_ptr()),
        reinterpret_cast<const __half *>(vp.data_ptr()),
        reinterpret_cast<const __half *>(dop.data_ptr()),
        reinterpret_cast<const float *>(lsep.data_ptr()),
        reinterpret_cast<const float *>(deltap.data_ptr()),
        reinterpret_cast<__half *>(dq.data_ptr()),
        Mp, Mp, M, (float)sm_scale, causal);
    C10_CUDA_CHECK(cudaGetLastError());
  }
  return {dq.narrow(2, 0, M).contiguous(), dk.narrow(2, 0, M).contiguous(),
          dv.narrow(2, 0, M).contiguous()};
}

}  // namespace fmha_cuda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fmha_forward", &fmha_cuda::fmha_forward, "FMHA CUDA forward (fp16)");
  m.def("fmha_backward", &fmha_cuda::fmha_backward, "FMHA CUDA backward (fp16)");
}
