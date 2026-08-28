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

## Acknowledgements

This implementation is based on the Triton attention implementation from the original Flash Attention 2 repository by TriDao and the Triton tutorial on fused attention.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
