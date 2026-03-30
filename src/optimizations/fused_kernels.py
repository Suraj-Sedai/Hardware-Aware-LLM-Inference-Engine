"""Fused kernel operations."""
import torch
import math


def fused_attention(q, k, v, mask=None):
    """
    Fused attention operation.
    Combines QK^T, scaling, masking, softmax, and V multiplication.
    
    Args:
        q: Query tensor (B, n_heads, T_q, head_dim)
        k: Key tensor (B, n_heads, T_k, head_dim)
        v: Value tensor (B, n_heads, T_k, head_dim)
        mask: Optional attention mask
    
    Returns:
        Attention output (B, n_heads, T_q, head_dim)
    """
    head_dim = q.size(-1)
    scale = 1.0 / math.sqrt(head_dim)
    
    # fused QK^T with scaling
    attn_weights = torch.baddbmm(
        torch.zeros(q.size(0) * q.size(1), q.size(2), k.size(2), device=q.device, dtype=q.dtype),
        q.reshape(-1, q.size(2), q.size(3)),
        k.reshape(-1, k.size(2), k.size(3)).transpose(-2, -1),
        beta=0.0,
        alpha=scale
    )
    
    # reshape back
    attn_weights = attn_weights.view(q.size(0), q.size(1), q.size(2), k.size(2))
    
    # apply mask if provided
    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
    
    # softmax
    attn_weights = torch.softmax(attn_weights, dim=-1)
    
    # attention @ values
    out = torch.matmul(attn_weights, v)
    
    return out


def fused_gelu(x):
    """
    Fused GELU activation.
    Uses the approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return x * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def fused_layer_norm(x, weight, bias, eps=1e-5):
    """
    Fused layer normalization.
    Combines mean, variance, normalization, and affine transform.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * weight + bias


class FusedMLP(torch.nn.Module):
    """MLP with fused GELU activation."""
    
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = torch.nn.Linear(dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = fused_gelu(x)
        x = self.fc2(x)
        return x
