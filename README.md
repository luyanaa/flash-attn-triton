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

Dense forward guards both of its dense writes against a partial tail block. The
tile width need not divide the sequence -- the entry point requires only a
multiple of 64, while the forward autotune set contains 128-row tiles -- and an
unguarded block-pointer store writes those out-of-range rows over the start of
the same head's output, which silently corrupts the first rows of every head.

Inputs are made contiguous before the head-major transpose, on every call rather
than only when gradients are required. The dense kernel is correct only for the
two head-major layouts this wrapper produces -- a transposed view of a contiguous
tensor, or a contiguous head-major one -- and any other stride pattern returns
silently wrong output: the slices of a packed buffer that
`flash_attn_qkvpacked_func` unpacks measured 0.6 to 7.1 absolute error against
SDPA, and a non-unit head-dim stride produced `inf`. Normalizing is free for
contiguous input, so the dense path is unaffected; unpacking a packed buffer
costs one copy per tensor (packed forward on a V100: 21.7 TFLOP/s against 28.2
before, for a correct result instead of a wrong one).

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

PV and backward use FP16 operands with FP32 dot accumulation; backward casts dS
to the input element type before dQ/dK. A local repro on stock Triton 3.1.0
showed standalone loaded/generated INT8 dots are exact, while an INT8 operand
generated from an MMA/softmax chain is mispacked by the SM75 register-to-dot
shortcut. TTGIR contains the conversion; the failure is in lowering. The local
Triton source patch materializes that conversion through shared memory and adds
a regression test. PV stays FP16 until the patch and a scaled INT8 PV path pass
end-to-end validation. The entry point validates equal contiguous inputs,
power-of-two D >= 32, and sequence lengths divisible by 64; its backward
addresses K/V using Q's strides.

Triton is the sole project-owned attention-kernel backend.

## Benchmark: Tesla V100

Measured on a Tesla V100-PCIE-32GB with PyTorch `2.5.1+cu124`,
CUDA `12.4`, Triton `3.1.0`, and xFormers `0.0.29`. Values use the same
effective-FLOP convention as the T4 benchmark. The xFormers column uses
the CUTLASS forward/backward operators.

The Volta path now prunes fully masked causal key blocks in packed
variable-length attention, specializes optional score features at compile
time, pre-scales Q for dense forward, and excludes 3-stage dense configs.
The Project columns were re-measured on the current revision; the xFormers
column dates from the earlier revision, so it understates the current margin.
The backward block size for Volta is 32 rows, which measured 1.20x (S=1024,
non-causal) to 1.25x (S=2048, causal) over the previous 16-row tiles.

Volta reaches `mma.sync.aligned.m8n8k4` through Triton (MMAv1 -- an fp16
accumulator does not even compile, `f32.f16.f16.f32` is hardcoded), so a V100 SM
must issue 2 MMA per clock to reach its 113 TFLOP/s peak, half of its four
schedulers' issue slots, against 0.5 MMA per clock (0.125 slots) on Turing. A
pure dot loop does reach 97.7 TFLOP/s (86% of peak) with register-resident
operands, but the forward kernel in these tables runs at 31-39 TFLOP/s, about
56% of the best tile it can use (`BLOCK_N` is capped at `HEAD_DIM`). The gap is
that issue bandwidth: the same softmax, mask and fragment-load work shares a
stream where the MMA alone is four times as dense.

### Forward effective throughput (TFLOP/s)

| B | S | H | D | Causal | Project | xFormers CUTLASS |
|---:|---:|---:|---:|:---:|---:|---:|
| 4 | 512 | 32 | 64 | no | 34.20 | 16.71 |
| 4 | 1024 | 32 | 64 | no | 37.56 | 25.08 |
| 4 | 2048 | 32 | 64 | yes | 36.56 | 24.63 |
| 8 | 512 | 32 | 64 | no | 35.59 | 21.47 |
| 2 | 1024 | 64 | 64 | yes | 31.56 | 17.14 |
| 8 | 512 | 16 | 128 | yes | 23.00 | 16.15 |
| 1 | 4096 | 16 | 64 | no | 39.23 | 25.32 |
| 1 | 4096 | 16 | 64 | yes | 30.13 | 21.40 |

### Backward effective throughput (TFLOP/s)

| B | S | H | D | Causal | Project | xFormers CUTLASS |
|---:|---:|---:|---:|:---:|---:|---:|
| 4 | 512 | 32 | 64 | no | 19.70 | 11.13 |
| 4 | 1024 | 32 | 64 | no | 23.12 | 14.23 |
| 4 | 2048 | 32 | 64 | yes | 22.65 | 14.09 |
| 8 | 512 | 32 | 64 | no | 21.29 | 12.67 |
| 2 | 1024 | 64 | 64 | yes | 20.10 | 11.34 |
| 8 | 512 | 16 | 128 | yes | 14.50 | 9.49 |
| 1 | 4096 | 16 | 64 | no | 22.78 | 15.58 |
| 1 | 4096 | 16 | 64 | yes | 22.20 | 13.47 |

## Benchmark: Tesla T4

Measured on a Kaggle Tesla T4 (14.56 GiB) with PyTorch `2.5.1+cu124`,
CUDA `12.4`, and Triton `3.1.0`, the same runtime stack as the RTX 2080 Ti
table below. Inputs are FP16, the INT8 column runs the Turing
`mma.m16n8k32` path, and the PyTorch column is
`torch.nn.functional.scaled_dot_product_attention` on the same tensors.
Each timing is the best of five CUDA-event batches (15 forward iterations
or 2 backward iterations per batch) after warmup. The effective FLOP model
counts the two attention matrix multiplications, uses the logical sequence
length, and treats backward as 2x forward FLOPs. Softmax, masking, LSE, and
other elementwise work are excluded.

All three columns are measured in one process, because Kaggle T4 instances
are shared and throttle: two runs of this same revision spread 1.0-1.5x on
most rows and up to 2.6x on the small-tile rows (`2/1024/64/64` and
`8/512/16/128`, whose programs finish in tens of microseconds). Row-to-row
forward differences inside that band are not meaningful, which is why the
numbers here are not compared row by row against earlier revisions. The two
comparisons that survive the spread are Project-vs-SDPA within a row, and
the backward: its rows sit 1.4-2.1x above the previous table (six of seven
rows; `4/1024/32/64` non-causal is unchanged), a gap that holds across three
independent runs whose own backward spread is 1.0-1.3x.

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

| B | S | H | D | Causal | Project | INT8 | PyTorch SDPA |
|---:|---:|---:|---:|:---:|---:|---:|---:|
| 4 | 512 | 32 | 64 | no | 24.15 | 18.02 | 12.14 |
| 4 | 1024 | 32 | 64 | no | 22.43 | 21.87 | 11.79 |
| 4 | 2048 | 32 | 64 | yes | 21.25 | 18.61 | 11.47 |
| 8 | 512 | 32 | 64 | no | 20.56 | 17.37 | 10.80 |
| 2 | 1024 | 64 | 64 | yes | 14.95 | 14.74 | 10.32 |
| 8 | 512 | 16 | 128 | yes | 19.24 | 10.56 | 9.02 |
| 1 | 4096 | 16 | 64 | yes | 22.95 | 20.61 | 11.75 |
| 1 | 4096 | 16 | 64 | no | 22.24 | 27.15 | 12.21 |

### Backward effective throughput (TFLOP/s)

The INT8 path runs the same dense FP16 gradient kernels as the FP16 path
(see the 2080 Ti section), so its column tracks FP16 and differs only by
measurement spread.

| B | S | H | D | Causal | Project | INT8 | PyTorch SDPA |
|---:|---:|---:|---:|:---:|---:|---:|---:|
| 4 | 512 | 32 | 64 | no | 10.69 | 10.43 | 7.40 |
| 4 | 1024 | 32 | 64 | no | 10.59 | 10.35 | 6.96 |
| 4 | 2048 | 32 | 64 | yes | 8.12 | 10.06 | 6.34 |
| 8 | 512 | 32 | 64 | no | 9.26 | 9.13 | 6.40 |
| 2 | 1024 | 64 | 64 | yes | 9.07 | 9.26 | 5.56 |
| 8 | 512 | 16 | 128 | yes | 5.56 | 4.64 | 5.38 |
| 1 | 4096 | 16 | 64 | yes | 11.64 | 10.68 | 6.87 |
| 1 | 4096 | 16 | 64 | no | 11.13 | 11.08 | 7.71 |

INT8 is at or below FP16 on every forward row except the longest non-causal
shape (`1/4096/16/64`, 1.22x), which matches the 2080 Ti table below: K's
quantization is hoisted out of the query loop, so the prepass is amortised
only by long non-causal work. On a 40-SM T4 the small shapes pay
proportionally more for that prepass than they do on a 68-SM 2080 Ti, which
is why the INT8/FP16 ratios here are consistently lower.

For causal attention, forward FLOPs are
`2 * B * H * D * S * (S + 1)`; for non-causal attention they are
`4 * B * H * D * S * S`. Effective TFLOP/s is FLOPs divided by the measured
batch time and `1e9`.

The T4 measurements use the pinned PyTorch/Triton runtime listed above. On
that stack the repository's test file (`14 passed`, including the INT8 path
on a Turing part) and the 19-entry-point public API smoke suite both pass on
the same T4 session that produced these tables.

## Benchmark: RTX 2080 Ti (INT8 vs FP16)

Measured on an RTX 2080 Ti (compute capability 7.5) with PyTorch `2.5.1+cu124`
and Triton `3.1.0`; the same code on PyTorch `2.6.0+cu126` with Triton `3.2.0`
measured within about 10% of these numbers. Inputs are FP16. The INT8 column is
`attention_int8`: K is quantized per key token in the prepass and `PV` stays in
FP16. Per row, both columns come from the same run, and the FLOP model is the one
used for the T4 tables above; forward and backward are timed separately, the
backward on a reused graph so it measures gradient work alone.

The simplified FLOP model counts six canonical matrix-multiplication roles:
forward QK^T/PV and backward dP/dV/dQ/dK. Only forward QK^T uses INT8; PV and
backward use FP16 operands with FP32 dot accumulators, and backward casts dS to
the input element type before dQ/dK. `_attention_int8.backward` reuses the dense
backward kernel.

This six-role model omits recomputation in this implementation: both
`_attn_bwd_dkdv` and `_attn_bwd_dq` recompute QK^T and dP in their separate
backward passes. For full non-causal attention that gives nine source-level
attention dot products (two forward, four dK/dV, three dQ). If one equally
costly dot were 2x faster, the idealized dot-only ceiling for one forward INT8
dot is 9/8.5, or about 1.06x; K quantization, softmax, and launch costs lower
it. The measured per-phase numbers give 0.98-1.06x end-to-end on the 2080 Ti
and 0.92-1.03x on the T4, with the best case `1/4096/16/64` non-causal
(forward 1.33x on the 2080 Ti, 1.22x on the T4). The next performance lever is
the repeated backward QK^T work, which remains FP16 and needs its own
quantization and precision validation.

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
