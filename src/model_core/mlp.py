"""MLP (feed-forward) module."""
import torch
from torch import nn


class MLP(nn.Module):
    """Simple MLP with GELU activation."""
    
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
