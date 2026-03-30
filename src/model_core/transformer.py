"""Transformer block."""
import torch
from torch import nn
from .attention import SelfAttention
from .mlp import MLP


class TransformerBlock(nn.Module):
    """Single transformer block with attention + MLP."""
    
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads)
        self.mlp = MLP(dim)
    
    def forward(self, x, kv_cache=None, layer_id=None):
        attn_out, K_new, V_new = self.attn(self.ln1(x), kv_cache, layer_id)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, K_new, V_new
