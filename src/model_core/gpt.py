"""GPT model."""
import torch
from torch import nn
from .transformer import TransformerBlock


class GPT(nn.Module):
    """Simple GPT model for inference experiments."""
    
    def __init__(self, vocab_size, dim, n_heads, n_layers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.max_seq_len = max_seq_len
    
    def forward(self, input_ids, kv_cache=None):
        B, T = input_ids.shape
        
        # get position offset from cache
        pos_start = kv_cache.curr_len if kv_cache and kv_cache.curr_len > 0 else 0
        positions = torch.arange(pos_start, pos_start + T, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(B, -1)
        
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        new_kvs = []
        for layer_id, block in enumerate(self.blocks):
            x, K_new, V_new = block(x, kv_cache, layer_id)
            new_kvs.append((K_new, V_new))
        
        # update cache
        if kv_cache is not None:
            for layer_id, (K_new, V_new) in enumerate(new_kvs):
                kv_cache.update(layer_id, K_new, V_new)
            kv_cache.curr_len += T
        
        x = self.ln_f(x)
        return self.head(x)
