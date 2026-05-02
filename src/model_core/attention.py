import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init_(self, hidden_size, num_heads):
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

    def forward(self, x, mask = None, cache = None):
        B,T,D = x.shape  # Batch size, sequence length, hidden size
        H, HD = self.num_heads, self.head_dim

        #project input to q,k,v
        q = self.q_proj(x).view(B,T,H,HD).transpose(1,2)
        k = self.k_proj(x).view(B,T,H,HD).transpose(1,2)
        v = self.v_proj(x).view(B,T,H,HD).transpose(1,2)

        #chache for decoding
        if cache is not None:
            if "k" in cache and "v" in cache:
                k = torch.cat([cache['k'], k], dim=2)
                v = torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k,v

        #compute scaler dot product attention
        q = q / (HD ** 0.5)  #scale query
        attn_weights = torch.matmul(q, k.transpose(-2,-1))

        #apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_output = torch.matmul(attn_weights, v)

        #combine head and project output
        attn_output = attn_output.transpose(1,2).contiguous().view(B,T,D)
        output = self.out_proj(attn_output)

        return output

def causal_mask(size, device):
    return torch.tril(torch.ones(size, size, device=device)).unsqueeze(0).unsqueeze(1)

def causal_mask_for_decoding(current_len, cache_len, device):
    total_len = current_len + cache_len
    mask = torch.tril(torch.ones(total_len, total_len, device=device))
    return mask[-current_len:, :].unsqueeze(0).unsqueeze(1)  # [1, 1, current_len, total_len]