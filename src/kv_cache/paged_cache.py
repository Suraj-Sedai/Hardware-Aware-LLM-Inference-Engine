"""Paged KV Cache implementation."""
import torch


class PagedKVCache:
    """Manages KV cache using paging to reduce fragmentation."""
    
    def __init__(self, n_layers, n_heads, dim_head, block_size, num_blocks, device, batch_size):
        # Implementation will go here
        self.num_layers = n_layers
        self.num_heads = n_heads
        self.head_dim = dim_head
        self.page_size = block_size
        self.max_pages = num_blocks
        self.batch_size = batch_size
        self.device = device
        self.dtype = torch.dtype
        self.device = device
        self.curr_len = 0
        self.k_pages = torch.zeros(
            num_blocks, batch_size, n_heads, block_size, dim_head, device=device, dtype = self.dtype)
        self.v_pages = torch.zeros(
            num_blocks, batch_size, n_heads, block_size, dim_head, device=device, dtype = self.dtype)
        self.page_table = [[None] * self.max_pages for _ in range(self.num_layers)]                                                                                                
        self.free_pages = [list(range(self.max_pages)) for _ in range(self.num_layers)]

    def _num_used_pages(self):
        if self.curr_len == 0:
            return 0
        return (self.curr_len+ self.page_size - 1) // self.page_size

    def _get_or_allocate_page(self, layer_id, logical_page_idx):
        physical_page = self.page_table[layer_id][logical_page_idx]
        if physical_page is None:
            if not self.free_pages[layer_id]:
                raise ValueError("KV cache out of pages!")
            physical_page = self.free_pages[layer_id].pop(0)
            self.page_table[layer_id][logical_page_idx] = physical_page
        return physical_page

    def update(self, layer_id, K_new, V_new):
        pass
    def get_for_attention(self, layer_id, K_new, V_new):
        pass
    
    def reset(self):
        pass
