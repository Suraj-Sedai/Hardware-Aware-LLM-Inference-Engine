import torch
import torch.nn as nn
from attention import MultiHeadSelfAttention
from mlp import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_hidden_size, dropout_prob=0.1):
        super().__init__(self)
        self.hidden_size = hidden_size

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        #multi-head self attention
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads)

        #Feed-forwaed block
        self.mlp = FeedForward(hidden_size,ff_hidden_size)

        #Dropout

    def forward(self, x, mask= None, cache=None):
        #pre LN + attention
        residual = x
        x= self.ln1(x)
        x= self.attention(x, mask=mask, cache=cache)
        x= self.dropout(x)#apply dropout after attention
        x=x+residual#residual connection

        #pre LN + MLP
        residual = x
        x= self.ln2(x)
        x= self.mlp(x)
        x= self.dropout(x)
        x=x+residual

        return 
    
    # Test the TransformerBlock
B, T, D, H, D_ff = 8, 16, 64, 8, 256  # Batch size, sequence length, model dim, num heads, feed-forward dim
x = torch.randn(B, T, D)
mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(1)  # Causal mask

block = TransformerBlock(hidden_size=D, num_heads=H, ff_hidden_size=D_ff)
output = block(x, mask=mask)
print(output.shape)  # Should print: torch.Size([8, 16, 64]) 