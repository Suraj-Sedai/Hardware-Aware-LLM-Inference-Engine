"""KV Cache for efficient autoregressive generation."""
import torch


class KVCacheManager:
    """Manages key-value cache for transformer layers."""
    
    def __init__(self, n_layers, n_heads, max_seq_len, dim_head, device, batch_size):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.dim_head = dim_head
        self.device = device
        self.batch_size = batch_size
        
        # preallocate KV cache for all layers
        self.K = [torch.zeros(batch_size, n_heads, max_seq_len, dim_head, device=device) 
                  for _ in range(n_layers)]
        self.V = [torch.zeros(batch_size, n_heads, max_seq_len, dim_head, device=device) 
                  for _ in range(n_layers)]
        self.curr_len = 0
    
    def update(self, layer_id, K_new, V_new):
        """Update cache with new keys and values."""
        B, n_heads, T_new, dim_head = K_new.shape
        
        if self.curr_len + T_new > self.max_seq_len:
            raise ValueError("KV cache out of capacity!")
        
        self.K[layer_id][:, :, self.curr_len:self.curr_len + T_new, :] = K_new
        self.V[layer_id][:, :, self.curr_len:self.curr_len + T_new, :] = V_new
    
    def get(self, layer_id):
        """Get cached keys and values for a layer."""
        return (self.K[layer_id][:, :, :self.curr_len, :], 
                self.V[layer_id][:, :, :self.curr_len, :])
    
    def reset(self):
        """Reset cache to empty state."""
        self.curr_len = 0
