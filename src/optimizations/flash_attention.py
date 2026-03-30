"""Flash Attention implementation."""
import torch
from torch import nn
import math

# Check if flash_attn is available
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


def flash_attention_available():
    """Check if flash attention is available."""
    return FLASH_ATTN_AVAILABLE


class FlashAttention(nn.Module):
    """
    Flash Attention wrapper.
    Falls back to standard attention if flash_attn is not available.
    """
    
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout
        
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
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        
        # use KV cache if available
        if kv_cache is not None and kv_cache.curr_len > 0:
            K_cached, V_cached = kv_cache.get(layer_id)
            # transpose for concat: (B, n_heads, seq, head_dim) -> (B, seq, n_heads, head_dim)
            K_cached = K_cached.transpose(1, 2)
            V_cached = V_cached.transpose(1, 2)
            k = torch.cat([K_cached, k], dim=1)
            v = torch.cat([V_cached, v], dim=1)
        
        if FLASH_ATTN_AVAILABLE and x.is_cuda:
            # Use flash attention (expects B, T, n_heads, head_dim)
            out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        else:
            # Fallback to standard attention
            out = self._standard_attention(q, k, v)
        
        # reshape output
        out = out.reshape(B, T, D)
        
        # for KV cache, return k, v in (B, n_heads, T, head_dim) format
        k_out = k[:, -T:, :, :].transpose(1, 2) if kv_cache else k.transpose(1, 2)
        v_out = v[:, -T:, :, :].transpose(1, 2) if kv_cache else v.transpose(1, 2)
        
        return self.out_proj(out), k_out, v_out
    
    def _standard_attention(self, q, k, v):
        """Standard scaled dot-product attention."""
        # transpose to (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        
        # transpose back to (B, T, n_heads, head_dim)
        return out.transpose(1, 2)
