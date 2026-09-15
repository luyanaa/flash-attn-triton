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
- Turing INT8 forward (`attention_int8`) with the dense FP16 backward

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

The Turing INT8 path (`attention_int8`, compute capability 7.5 only) quantizes K
per key token in a prepass kernel and loads the INT8 keys directly, so `QK^T`
runs on the INT8 tensor cores with FP32 accumulation. Quantizing K inside the
query loop instead measures roughly four times more expensive than the INT8 MMA
it feeds on this generation, because every query block redoes every key tile.
The per-token scales are then folded into the score epilogue, so no per-tile
quantization survives in the inner loop.

PV and the entire backward stay in FP16. Triton mis-lowers an INT8 operand
produced in-kernel on sm_75 -- `P = exp2(S - m)` reaches `tl.dot` as garbage, not
merely slowly, on both 3.1.0 and 3.2.0 -- so neither INT8 PV nor an INT8 backward
is reachable from here, and the gradient path is the same dense FP16 one
described above. The head dim must be a power of two of at least 32, and
sequence lengths must be multiples of 64, the same requirement the dense FP16 entry point imposes.

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
pipeline stage. Backward tile dimensions are currently fixed rather than
included in the Triton autotune configs; the single fused grid requires
`BLOCK_M2 == BLOCK_N1` for its unmasked dQ loads. The `tl.zeros` dK/dV
accumulators are FP32 local fragments; at D=128 they represent 64 KiB of
logical storage, so the relevant risk is register pressure or spilling, not
a 64 KiB shared-memory allocation. Dense backward dQ and dK/dV now
compile-time specialize the optional softcap, ALiBi, and window branches.

The forward kernel already autotunes `BLOCK_M`/`BLOCK_N`; `BLOCK_M=128` is a
valid forward candidate. T4 D=128 measurements did not justify removing it
globally or pinning a new stage/tile choice.

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

## Benchmark: RTX 2080 Ti (INT8 vs FP16)

Measured on an RTX 2080 Ti (compute capability 7.5) with PyTorch `2.5.1+cu124`
and Triton `3.1.0`; the same code on PyTorch `2.6.0+cu126` with Triton `3.2.0`
measured within about 10% of these numbers. Inputs are FP16. The INT8 column is
`attention_int8`: K is quantized per key token in the prepass and `PV` stays in
FP16. Per row, both columns come from the same run, and the FLOP model is the one
used for the T4 tables above; forward and backward are timed separately, the
backward on a reused graph so it measures gradient work alone.

### Forward effective throughput (TFLOP/s)

| B | S | H | D | Causal | FP16 | INT8 | INT8 / FP16 |
|---:|---:|---:|---:|:---:|---:|---:|---:|
| 4 | 512 | 32 | 64 | no | 45.66 | 44.31 | 0.97 |
| 4 | 1024 | 32 | 64 | no | 48.84 | 53.92 | 1.10 |
| 4 | 2048 | 32 | 64 | yes | 47.61 | 46.44 | 0.98 |
| 8 | 512 | 32 | 64 | no | 44.86 | 43.57 | 0.97 |
| 2 | 1024 | 64 | 64 | yes | 36.64 | 34.24 | 0.93 |
| 8 | 512 | 16 | 128 | yes | 29.42 | 25.91 | 0.88 |
| 1 | 4096 | 16 | 64 | yes | 45.63 | 45.14 | 0.99 |
| 1 | 4096 | 16 | 64 | no | 45.96 | 61.16 | 1.33 |

INT8 pays off once the two attention matrix multiplications dominate the per-tile
work: 1.10x at `S=1024` and 1.33x at `S=4096` for non-causal attention. It gives
up ground where a causal diagonal shrinks the attention work while the per-call K
quantization stays proportional to `S`, so the prepass is not amortized, and at
head dim 128, where the quantization and scale traffic double relative to the
matmuls.

The INT8 launch tiles are pinned by head dim and sequence length rather than
autotuned. The candidates span 1.3-2.4x, and Triton's autotuner times each config
once in sequence, so on a noisy device it lands on a slow tile often enough that
results varied up to 40% run to run. `_int8_fwd_launch_config` carries the
measured choice: a 128-row query tile at head dim 64, 64 rows otherwise. The
128-row tile is also used only when the sequence divides by 128 -- with a partial
row block its first half computes the wrong softmax denominator, which a masked
tail does not repair (64, 192 and 320 diverge against an fp32 reference while
128, 256 and 512 do not).

Causal rows move more between runs than non-causal ones (down to 0.8x) because
the causal stages leave the later query blocks with more work than the early
ones, so the result depends on how the blocks land across SMs.

### Backward effective throughput (TFLOP/s)

Both paths run the same dense FP16 gradient kernels, so there is one column:

| B | S | H | D | Causal | Backward |
|---:|---:|---:|---:|:---:|---:|
| 4 | 512 | 32 | 64 | no | 22.4 |
| 4 | 1024 | 32 | 64 | no | 23.9 |
| 4 | 2048 | 32 | 64 | yes | 23.9 |
| 8 | 512 | 32 | 64 | no | 21.5 |
| 2 | 1024 | 64 | 64 | yes | 21.5 |
| 8 | 512 | 16 | 128 | yes | 10.3 |
| 1 | 4096 | 16 | 64 | yes | 24.7 |
| 1 | 4096 | 16 | 64 | no | 24.8 |

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
