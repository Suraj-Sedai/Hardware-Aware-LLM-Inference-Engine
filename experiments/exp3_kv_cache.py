"""
Experiment 3: KV Cache Efficiency
Tests KV cache memory usage and performance impact.

The KV cache speeds up autoregressive generation by storing computed
key/value pairs, avoiding redundant computation on previous tokens.
"""
import time
import torch

from src.model_core import GPT
from src.kv_cache import KVCacheManager
from src.sampling import sample_top_k
from src.profiling import get_device


def get_cache_memory_mb(kv_cache):
    """Calculate KV cache memory in MB."""
    total_bytes = 0
    for k_tensor in kv_cache.K:
        total_bytes += k_tensor.nelement() * k_tensor.element_size()
    for v_tensor in kv_cache.V:
        total_bytes += v_tensor.nelement() * v_tensor.element_size()
    return total_bytes / (1024 * 1024)


def run_kv_cache_experiment(
    seq_lengths=None,
    batch_size=1,
    max_new_tokens=100,
    vocab_size=1000,
    dim=512,
    n_heads=8,
    n_layers=8
):
    """
    Benchmark KV cache efficiency across different configurations.
    
    Uses larger model (dim=512, 8 layers) to show cache benefits clearly.
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024]
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: batch_size={batch_size}, max_new_tokens={max_new_tokens}")
    print(f"Model: vocab={vocab_size}, dim={dim}, heads={n_heads}, layers={n_layers}")
    print("-" * 60)
    
    results = []
    
    for seq_len in seq_lengths:
        # Create model (larger for meaningful cache benefits)
        model = GPT(vocab_size, dim, n_heads, n_layers, seq_len).to(device)
        model.eval()
        
        prompt_len = 16
        effective_new_tokens = min(max_new_tokens, seq_len - prompt_len - 10)
        if effective_new_tokens < 10:
            continue
            
        input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)
        
        # ===== WARMUP =====
        warmup_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids, warmup_cache)
                warmup_cache.curr_len = 0  # Reset for next warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # ===== Test WITH KV cache =====
        kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        cache_memory = get_cache_memory_mb(kv_cache)
        
        with torch.no_grad():
            # Prefill (not timed)
            logits = model(input_ids, kv_cache)
            next_token = sample_top_k(logits[:, -1, :])
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Decode: generate one token at a time using cache
            start_cached = time.perf_counter()
            for _ in range(effective_new_tokens):
                logits = model(next_token, kv_cache)
                next_token = sample_top_k(logits[:, -1, :])
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time_cached = time.perf_counter() - start_cached
        
        # ===== Test WITHOUT KV cache =====
        with torch.no_grad():
            generated = input_ids.clone()
            
            # Warmup without cache
            for _ in range(2):
                _ = model(generated, kv_cache=None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            generated = input_ids.clone()
            start_no_cache = time.perf_counter()
            for _ in range(effective_new_tokens):
                logits = model(generated, kv_cache=None)
                next_token = sample_top_k(logits[:, -1, :])
                generated = torch.cat([generated, next_token], dim=1)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time_no_cache = time.perf_counter() - start_no_cache
        
        speedup = time_no_cache / time_cached if time_cached > 0 else 0
        
        result = {
            "seq_len": seq_len,
            "cache_memory_mb": cache_memory,
            "time_with_cache": time_cached,
            "time_without_cache": time_no_cache,
            "speedup": speedup,
            "throughput_cached": effective_new_tokens / time_cached,
            "throughput_no_cache": effective_new_tokens / time_no_cache
        }
        results.append(result)
        
        print(f"seq_len={seq_len:4d} | cache={cache_memory:.2f}MB | "
              f"cached={time_cached:.4f}s | no_cache={time_no_cache:.4f}s | "
              f"speedup={speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    run_kv_cache_experiment()
