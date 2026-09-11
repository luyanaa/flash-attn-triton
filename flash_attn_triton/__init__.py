"""Triton-based implementation of Flash Attention with Flash Attention 2 compatible API."""

__version__ = "0.1.1"

# Import the main interfaces to expose at the package level
from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_func_cross,
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_with_kvcache,
    FlashAttention,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    FlashSelfAttention,
    FlashCrossAttention,
)
