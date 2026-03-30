"""Self-attention module."""
import torch
from torch import nn


class SelfAttention(nn.Module):
    """Multi-head self-attention with KV cache support."""
    
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x, kv_cache=None, layer_id=None):
        B, T, D = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # use KV cache if available
        if kv_cache is not None and kv_cache.curr_len > 0:
            K_cached, V_cached = kv_cache.get(layer_id)
            K_all = torch.cat([K_cached, k], dim=2)
            V_all = torch.cat([V_cached, v], dim=2)
        else:
            K_all, V_all = k, v
        
        # attention computation
        attn = (q @ K_all.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ V_all
        
        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out), k, v
