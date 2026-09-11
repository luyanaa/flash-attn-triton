# Flash Attention Triton

This repository provides a Triton implementation of Flash Attention with a
Flash Attention 2-compatible API. Supported attention paths follow PyTorch
`scaled_dot_product_attention` semantics, including upper-left causal
alignment for rectangular inputs. All attention kernels are implemented in
Triton; no handwritten CUDA extension is required.

## Requirements

- PyTorch <=2.6
- Triton <=3.2 (last Turing-compatible Triton release)
- CUDA-compatible GPU (compute capability 7.0+)

## Usage

The API is designed to be a drop-in replacement for Flash Attention 2:

```python
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, FlashAttention

out = flash_attn_func(q, k, v, causal=True)
out = flash_attn_qkvpacked_func(qkv, causal=True)
out = FlashAttention()(q, k, v, causal=True)
```

## Supported features

- Equal-length self-attention with forward and backward
- Unequal-length cross-attention via `flash_attn_func_cross`
- Packed variable-length attention and packed variants
- FP16 and FP32 inputs with FP32 accumulation
- Softcap score transformation
- ALiBi slopes with `(nheads,)` or `(batch, nheads)` inputs
- Native inference-only GQA/MQA addressing through `use_native_gqa=True`
- KV-cache attention and cache appends, including rotary, remapped, and paged caches
- Sliding-window attention
- Causal and non-causal masking
- Configurable softmax scaling
- Head dimensions 16, 32, 64, 128, and 256

## Triton backend notes

The dense equal-length FP16 path is the primary execution path. The code
validates dtype, device, shape, head layout, and attention options before
launching Triton kernels. Unequal query and KV lengths use the dedicated
cross-attention path with SDPA-compatible upper-left causal alignment.

Dense forward masking applies `-inf` before the row maximum, preserving valid
rows whose scores are extremely negative. Dense and variable-length epilogues
use safe normalization and write `-inf` LSE for empty rows. Variable-length
backward uses matching operand dtypes for Triton matrix products on pre-Ampere
targets.

The public dense gradient path uses direct dK/dV stores. Packed variable-length
backward now uses separate dQ and dK/dV grids, so `tl.atomic_add` remains only
in the low-level native-GQA path, where it preserves grouped-head accumulation
for callers that bypass the public inference-only restriction.

Native GQA remains inference-only when gradients are required. Deterministic
variable-length fallback, paged-cache reconstruction, materialized attention
probabilities, and materialized attention bias are correctness/API paths rather
than throughput paths.

Triton is the sole project-owned attention-kernel backend.

## Benchmark: Tesla V100

Measured on a Tesla V100-PCIE-32GB with PyTorch `2.5.1+cu124`,
CUDA `12.4`, Triton `3.1.0`, and xFormers `0.0.29`. Values use the same
effective-FLOP convention as the T4 benchmark. The xFormers column uses
the CUTLASS forward/backward operators.

The Volta path now prunes fully masked causal key blocks in packed
variable-length attention, specializes optional score features at compile
time, pre-scales Q for dense forward, and excludes 3-stage dense configs.

### Forward effective throughput (TFLOP/s)

| B | S | H | D | Causal | Project | xFormers CUTLASS |
|---:|---:|---:|---:|:---:|---:|---:|
| 4 | 512 | 32 | 64 | no | 20.00 | 16.71 |
| 4 | 1024 | 32 | 64 | no | 32.06 | 25.08 |
| 4 | 2048 | 32 | 64 | yes | 33.55 | 24.63 |
| 8 | 512 | 32 | 64 | no | 27.15 | 21.47 |
| 2 | 1024 | 64 | 64 | yes | 22.74 | 17.14 |
| 8 | 512 | 16 | 128 | yes | 16.61 | 16.15 |
| 1 | 4096 | 16 | 64 | no | 35.45 | 25.32 |
| 1 | 4096 | 16 | 64 | yes | 29.12 | 21.40 |

### Backward effective throughput (TFLOP/s)

| B | S | H | D | Causal | Project | xFormers CUTLASS |
|---:|---:|---:|---:|:---:|---:|---:|
| 4 | 512 | 32 | 64 | no | 11.77 | 11.13 |
| 4 | 1024 | 32 | 64 | no | 17.08 | 14.23 |
| 4 | 2048 | 32 | 64 | yes | 17.76 | 14.09 |
| 8 | 512 | 32 | 64 | no | 15.08 | 12.67 |
| 2 | 1024 | 64 | 64 | yes | 14.69 | 11.34 |
| 8 | 512 | 16 | 128 | yes | 10.99 | 9.49 |
| 1 | 4096 | 16 | 64 | no | 16.94 | 15.58 |
| 1 | 4096 | 16 | 64 | yes | 16.77 | 13.47 |

## Benchmark: Tesla T4

Measured on a Kaggle Tesla T4 (14.56 GiB) with PyTorch `2.6.0+cu124`,
CUDA `12.4`, and Triton `3.2.0`. Inputs are FP16. Each timing is the
median of CUDA-event measurements after warmup (20 forward iterations or
12 backward iterations). The effective FLOP model counts the two attention
matrix multiplications, uses the logical sequence length, and treats
backward as 2x forward FLOPs. Softmax, masking, LSE, and other elementwise
work are excluded.

The Turing backward launch uses tested `(32, 64, 64, 32)` blocks and one
pipeline stage. The larger valid tile improves backward throughput on the
tested FP16 workloads while staying within the T4's 64 KiB shared-memory limit.
The `(64, 32)` M1/N1 ordering is not valid for this kernel's dK/dV
static divisibility constraint; `(32, 64, 64, 32)` was the best-performing
valid tested alternative. Dense backward dQ and dK/dV now compile-time
specialize the optional softcap, ALiBi, and window branches.

### Forward effective throughput (TFLOP/s)

| B | S | H | D | Causal | Project | PyTorch SDPA |
|---:|---:|---:|---:|:---:|---:|---:|
| 4 | 512 | 32 | 64 | no | 18.28 | 16.36 |
| 4 | 1024 | 32 | 64 | no | 29.11 | 10.65 |
| 4 | 2048 | 32 | 64 | yes | 18.50 | 11.67 |
| 8 | 512 | 32 | 64 | no | 18.74 | 10.83 |
| 2 | 1024 | 64 | 64 | yes | 14.82 | 9.45 |
| 8 | 512 | 16 | 128 | yes | 12.45 | 11.27 |
| 1 | 4096 | 16 | 64 | yes | 24.45 | 11.18 |

### Backward effective throughput (TFLOP/s)

| B | S | H | D | Causal | Project | PyTorch SDPA |
|---:|---:|---:|---:|:---:|---:|---:|
| 4 | 512 | 32 | 64 | no | 5.02 | 3.88 |
| 4 | 1024 | 32 | 64 | no | 10.42 | 7.21 |
| 4 | 2048 | 32 | 64 | yes | 6.01 | 6.71 |
| 8 | 512 | 32 | 64 | no | 5.27 | 4.03 |
| 2 | 1024 | 64 | 64 | yes | 5.21 | 3.62 |
| 8 | 512 | 16 | 128 | yes | 2.33 | 4.88 |
| 1 | 4096 | 16 | 64 | yes | 6.22 | 6.20 |

For causal attention, forward FLOPs are
`2 * B * H * D * S * (S + 1)`; for non-causal attention they are
`4 * B * H * D * S * S`. Effective TFLOP/s is FLOPs divided by median
milliseconds and `1e9`.

The T4 measurements use the pinned PyTorch/Triton runtime listed above.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
