import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention
from .mlp import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_hidden_size, dropout_prob=0.1):
        """
        Args:
            hidden_size: Model dimension (D).
            num_heads: Number of attention heads (H).
            ff_hidden_size: Feed-forward hidden dimension (D_ff).
            dropout_prob: Dropout probability.
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Pre-LN components
        self.ln1 = nn.LayerNorm(hidden_size)  # LayerNorm before attention
        self.ln2 = nn.LayerNorm(hidden_size)  # LayerNorm before MLP

        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads)

        # Feed-forward block
        self.mlp = FeedForward(hidden_size, ff_hidden_size)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer

    def forward(self, x, mask=None, cache=None):
        """
        Args:
            x: Input tensor of shape [B, T, D].
            mask: Optional causal mask of shape [B, 1, T, T].
            cache: Optional dictionary for cached attention.
        Returns:
            Tensor of shape [B, T, D].
        """
        # Pre-LN + Attention
        residual = x
        x = self.ln1(x)  # Apply LayerNorm
        x = self.attention(x, mask=mask, cache=cache)  # Self-attention
        x = self.dropout(x)  # Apply dropout
        x = x + residual  # Add residual connection

        # Pre-LN + MLP
        residual = x
        x = self.ln2(x)  # Apply LayerNorm
        x = self.mlp(x)  # Feed-forward block
        x = self.dropout(x)  # Apply dropout
        x = x + residual  # Add residual connection

        return x