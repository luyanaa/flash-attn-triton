# Flash Attention Triton

This repository provides a Triton implementation of Flash Attention with a Flash Attention 2-compatible API. Supported attention paths follow PyTorch `scaled_dot_product_attention` semantics, including upper-left causal alignment for rectangular inputs. It targets Volta (eg. V100, Titan V) and Turing (eg. 2080 Ti, T4) GPUs not supported by the original FA2 CUDA package.

## Requirements
- PyTorch 2.6, Triton 3.2 (last Turing-compatible Triton.)
- CUDA Toolkit 12.8 system-wide (https://github.com/googlecolab/backend-info#202507) 
- CUDA-compatible GPU (compute capability 7.0+)

## Usage

The API is designed to be a drop-in replacement for Flash Attention 2. You can use it in the same way:

```python
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, FlashAttention

# Basic usage
out = flash_attn_func(q, k, v, causal=True)

# Packed QKV
out = flash_attn_qkvpacked_func(qkv, causal=True)

# Module interface
flash_attn = FlashAttention()
out = flash_attn(q, k, v, causal=True)
```

## Currently Supported Features

- Basic equal-length self-attention (forward and backward)
- Dedicated unequal-length cross-attention via `flash_attn_func_cross`
- Packed variable-length attention via `flash_attn_varlen_func` and packed variants
- FP16 and FP32 inputs with FP32 accumulation on Turing and Volta GPUs.
- Softcap score transformation in forward and backward.
- ALiBi slopes with `(nheads,)` or `(batch, nheads)` inputs.
- Optional native GQA/MQA addressing for inference and training (`use_native_gqa=True`)
- KV-cache attention and cache appends via `flash_attn_with_kvcache`, including rotary, remapped, and paged caches
- Sliding-window attention through the Triton forward and backward kernels
- Causal and non-causal masking
- Softmax scaling
- MQA/GQA support via native grouped-head addressing or K/V repetition
- Head dims 16, 32, 64, 128, 256
- Experimental pure-CUDA FP16 backend for Turing (sm_75) and Volta (sm_70) via
  `flash_attn_cuda_func` (see below)

## Pure-CUDA FP16 backend (`flash_attn_cuda_func`)

`csrc/fmha_cuda.cu` is a hand-written CUDA implementation of FlashAttention-2
for the pre-Ampere tensor-core generations, mirroring the approach of the
community V100 implementations (ai-bond, ZRayZzz, zhinianqin
`flash-attention-v100` repos). It uses `nvcuda::wmma` m16n16k16 fragments,
which compile for both Volta and Turing, with the dQ/dKV backward split and
the causal mask applied in the per-row softmax. Volta has no INT8 tensor
cores, so this backend is FP16-only; the INT8 path remains the Triton
experimental path for Turing (sm_75) in `flash_attn_func_int8`.

Build (requires nvcc + CUDA Toolkit 12.8, torch 2.6):

```bash
python setup_cuda.py build_ext --inplace   # produces flash_attn_triton/fmha_cuda.*.so
```

Use:

```python
from flash_attn_triton import flash_attn_cuda_func
out = flash_attn_cuda_func(q, k, v, causal=True)   # fp16, head_dim 64
```

`flash_attn_cuda_func` validates inputs (fp16, head_dim 64, equal lengths),
dispatches to the CUDA kernel on sm_70/sm_75 when the extension is built, and
falls back to the Triton `flash_attn_func` otherwise. Verified on T4: forward
rel err 1e-4 and backward dQ/dK/dV rel err ~5e-4 vs PyTorch SDPA (fp16
precision). Performance work is ongoing — see the `build_cuda_colab.sh`
script for the one-shot build-and-verify flow on a Colab T4.

## Benchmarks
### RTX 2080 Ti (Turing)

```
fused-attention-batch4-head32-d64-fwd-causal=True-dropout=0.0:
     N_CTX  Triton [FP16]
0   1024.0      29.258471
1   2048.0      41.382117
2   4096.0      46.972266
3   8192.0      49.315714
4  16384.0      50.443531
fused-attention-batch4-head32-d64-fwd-causal=False-dropout=0.0:
     N_CTX  Triton [FP16]
0   1024.0      38.110175
1   2048.0      47.640577
2   4096.0      50.301599
3   8192.0      51.136501
4  16384.0      51.826783
fused-attention-batch4-head32-d64-bwd-causal=True-dropout=0.0:
     N_CTX  Triton [FP16]
0   1024.0      22.085938
1   2048.0      26.173398
2   4096.0      28.565586
3   8192.0      30.030201
4  16384.0      31.082861
fused-attention-batch4-head32-d64-bwd-causal=False-dropout=0.0:
     N_CTX  Triton [FP16]
0   1024.0      27.756566
1   2048.0      30.274265
2   4096.0      31.471025
3   8192.0      32.253811
4  16384.0      32.614130
```

### NVIDIA T4 (Turing) — vs PyTorch SDPA

Measured on a Colab T4 with the pinned torch 2.6.0+cu124 / Triton 3.2.0, FP16, batch 4, 32 heads, head dim 64. Latency is the mean over 30 iterations after warmup; TFLOPS uses the standard fused-attention accounting (4·B·H·S²·D, causal ×0.5, backward ×2.5).

**Mid/large sequence lengths** (typical LLM inference/training range) are faster than PyTorch `scaled_dot_product_attention`:

| N_CTX | mode | ours (ms) | SDPA (ms) | ours (TFLOP/s) | SDPA (TFLOP/s) | speedup |
|------:|:----:|----------:|----------:|---------------:|---------------:|--------:|
|  2048 | cfwd |     5.229 |     5.710 |          13.14 |          12.03 |    1.09 |
|  2048 | fwd  |     7.773 |    10.543 |          17.68 |          13.04 |    1.36 |
|  4096 | cfwd |    15.756 |    21.342 |          17.45 |          12.88 |    1.35 |
|  4096 | fwd  |    26.243 |    40.620 |          20.95 |          13.53 |    1.55 |
|  2048 | cbwd |    19.441 |    25.973 |           8.84 |           6.61 |    1.34 |
|  4096 | cbwd |    67.089 |    93.872 |          10.24 |           7.32 |    1.40 |

**Short sequence lengths** — per-call GPU→CPU synchronization (ALiBi detection and dropout-seed generation) was removed from the hot path, giving a 1.2–2.5× forward improvement:

| N_CTX | before (ms) | after (ms) | speedup |
|------:|------------:|-----------:|--------:|
|   128 |       0.722 |      0.378 |    1.9× |
|   256 |       0.923 |      0.371 |    2.5× |
|   512 |       1.029 |      0.597 |    1.7× |
|  1024 |       2.103 |      1.749 |    1.2× |

**Flash Attention v2 tuning (T4, repeated-median over 5 runs)** — memory-coalescing hints (`tl.max_contiguous`/`tl.multiple_of`) in forward and backward, plus Turing/V100-safe autotune configs (`num_stages ≤ 3`; `num_stages=4` risks shared-memory exhaustion on sm_7x):

| N_CTX | mode | before (ms) | after (ms) | speedup |
|------:|:----:|------------:|-----------:|--------:|
|  1024 | cfwd |       2.197 |      2.095 |    1.05 |
|  2048 | cfwd |       6.036 |      5.949 |    1.02 |
|  2048 | fwd  |       8.939 |      8.507 |    1.05 |
|  4096 | cfwd |      18.984 |     17.965 |    1.06 |
|  4096 | fwd  |      31.741 |     30.597 |    1.04 |
|  2048 | cbwd |      22.936 |     24.020 |    0.96 |
|  4096 | cbwd |      77.760 |     79.848 |    0.97 |

Forward gains 1–6%; the backward config widening was reverted (it caused a 2–4% regression). All numbers are contention-inflated (~1.2× the clean SDPA-comparison baselines above), so relative gains are real but modest.

**SageBwd-style INT8 attention (Turing sm_75 only, experimental)** — `flash_attn_func_int8` runs the forward QK^T on the INT8 tensor cores, keeping PV and the backward dP in FP16 per SageBwd (arXiv 2505.11594, 2603.02170). Forward accuracy is now solid (0.5–2% rel err vs the fp16 path, causal and noncausal; the earlier scale-folding bug that caused 25–63% error is fixed). Speed remains the blocker: measured ~0.66× fp16 end-to-end — Triton 3.2's INT8 MMA gives only ~6% raw dot speedup at the kernel's 64×64×64 blocks on T4, and in-kernel quantize/dequantize overhead cancels it. The backward is functional but its FP16 S-recompute, normalized by the INT8 forward's LSE, degrades with sequence length (~2× gradient error at S=64, worse at S=256). Matching SageBwd's full accuracy/speed requires two-level P scaling, K-smoothing, and fused quantization; `flash_attn_func_int8` is an experimental building block, not a performance path. Volta (sm_70) has no INT8 tensor cores and falls back to fp16.

## Acknowledgements

This implementation is based on the Triton attention implementation from the original Flash Attention 2 repository by TriDao and the Triton tutorial on fused attention.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
