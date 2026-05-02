import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, hidden_size, ff_hidden_size, activation="gelu"):
        super().__init__(self)
        self.fc1 = nn.Linear(hidden_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, hidden_size)

        if activation=="gelu":
            self.activation = F.gelu
        elif activation=="silu":
            self.activation = F.silu
        else:
            raise ValueError("Unsupported activation. Use 'gelu' or 'silu'.")
        
    def forward(self,x):
        x = self.fc1(x)  # [B, T, D_ff]
        x = self.activation(x)  # Apply activation
        x = self.fc2(x)  # [B, T, D]
        return x