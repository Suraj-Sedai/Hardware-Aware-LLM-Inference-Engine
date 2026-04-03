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
    
    def forward(self, input_ids, kv_cache=None, latency_tracker=None):
        B, T = input_ids.shape
        
        # get position offset from cache
        pos_start = kv_cache.curr_len if kv_cache and kv_cache.curr_len > 0 else 0
        positions = torch.arange(pos_start, pos_start + T, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(B, -1)
        
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        for layer_id, block in enumerate(self.blocks):
            if latency_tracker:
                latency_tracker.start_layer(layer_id)
            
            # Cache is updated inside attention via get_for_attention()
            x, _, _ = block(x, kv_cache, layer_id)
            
            if latency_tracker:
                latency_tracker.end_layer(layer_id)
        
        # Update cache length after all layers processed
        if kv_cache is not None:
            kv_cache.curr_len += T
        
        x = self.ln_f(x)
        return self.head(x)
