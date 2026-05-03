import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        """
        Args:
            hidden_size: Model dimension (D).
            num_heads: Number of attention heads (H).
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # Hd = D / H

        # Projection layers for q, k, v
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None, cache=None):
        """
        Args:
            x: Input tensor of shape [B, T, D].
            mask: Optional causal mask of shape [B, 1, T, T].
            cache: Optional dictionary for cached attention.
        Returns:
            Tensor of shape [B, T, D].
        """
        B, T, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        # Project input to q, k, v
        q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)  # [B, H, T, Hd]
        k = self.k_proj(x).view(B, T, H, Hd).transpose(1, 2)  # [B, H, T, Hd]
        v = self.v_proj(x).view(B, T, H, Hd).transpose(1, 2)  # [B, H, T, Hd]

        # Handle cache for decoding
        if cache is not None:
            if cache["k"] is None or cache["v"] is None:
                # Initialize cache with current k and v
                cache["k"], cache["v"] = k, v
            else:
                # Append new keys and values to the cache
                cache["k"] = torch.cat([cache["k"], k], dim=2)  # [B, H, T_cache + T, Hd]
                cache["v"] = torch.cat([cache["v"], v], dim=2)  # [B, H, T_cache + T, Hd]
            k, v = cache["k"], cache["v"]

        # Compute scaled dot-product attention
        q = q / (Hd ** 0.5)  # Scale query by sqrt(Hd)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [B, H, T, T]

        # Apply causal mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)  # [B, H, T, T]
        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, Hd]

        # Combine heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        output = self.out_proj(attn_output)  # [B, T, D]

        return output