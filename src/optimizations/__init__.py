"""Optimizations module."""
from .flash_attention import FlashAttention, flash_attention_available
from .quantization import quantize_model, dequantize_model
from .fused_kernels import fused_attention
