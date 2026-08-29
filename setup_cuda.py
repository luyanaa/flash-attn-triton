import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Pure-CUDA FP16 Flash Attention for Turing (sm_75) and Volta (sm_70).
# Volta has no INT8 tensor cores, so this backend is fp16-only; wmma m16n16k16
# fragments are the portable unit across both architectures.
sources = ["csrc/fmha_cuda.cu"]

extra_cuda_cflags = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-gencode=arch=compute_70,code=sm_70",   # Volta V100
    "-gencode=arch=compute_75,code=sm_75",   # Turing T4 / RTX 20xx
]

ext_modules = [
    CUDAExtension(
        "flash_attn_triton.fmha_cuda",
        sources,
        extra_cuda_cflags=extra_cuda_cflags,
    )
]

setup(
    name="fmha_cuda",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
