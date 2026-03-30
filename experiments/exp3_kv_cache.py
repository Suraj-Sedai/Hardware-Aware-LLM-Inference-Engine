"""
Experiment 3: KV Cache Efficiency
Tests KV cache memory usage and performance impact.
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
    max_new_tokens=10,
    vocab_size=100,
    dim=64,
    n_heads=4,
    n_layers=4
):
    """
    Benchmark KV cache efficiency across different configurations.
    """
    if seq_lengths is None:
        seq_lengths = [32, 64, 128, 256]
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: batch_size={batch_size}, max_new_tokens={max_new_tokens}")
    print(f"Model: vocab={vocab_size}, dim={dim}, heads={n_heads}, layers={n_layers}")
    print("-" * 60)
    
    results = []
    
    for seq_len in seq_lengths:
        # Create model
        model = GPT(vocab_size, dim, n_heads, n_layers, seq_len).to(device)
        model.eval()
        
        # Test WITH KV cache
        kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        cache_memory = get_cache_memory_mb(kv_cache)
        
        prompt_len = min(4, seq_len - max_new_tokens)
        input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)
        
        with torch.no_grad():
            # With cache
            _ = model(input_ids, kv_cache)
            generated = input_ids.clone()
            
            start_cached = time.time()
            for _ in range(max_new_tokens):
                x = generated[:, -1:]
                logits = model(x, kv_cache)
                next_token = sample_top_k(logits[:, -1, :])
                generated = torch.cat([generated, next_token], dim=1)
            time_cached = time.time() - start_cached
        
        # Test WITHOUT KV cache (recompute all)
        with torch.no_grad():
            generated = input_ids.clone()
            
            start_no_cache = time.time()
            for _ in range(max_new_tokens):
                # Recompute full sequence each time (no cache)
                logits = model(generated, kv_cache=None)
                next_token = sample_top_k(logits[:, -1, :])
                generated = torch.cat([generated, next_token], dim=1)
            time_no_cache = time.time() - start_no_cache
        
        speedup = time_no_cache / time_cached if time_cached > 0 else 0
        
        result = {
            "seq_len": seq_len,
            "cache_memory_mb": cache_memory,
            "time_with_cache": time_cached,
            "time_without_cache": time_no_cache,
            "speedup": speedup,
            "throughput_cached": max_new_tokens / time_cached,
            "throughput_no_cache": max_new_tokens / time_no_cache
        }
        results.append(result)
        
        print(f"seq_len={seq_len:4d} | cache={cache_memory:.2f}MB | "
              f"cached={time_cached:.4f}s | no_cache={time_no_cache:.4f}s | "
              f"speedup={speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    run_kv_cache_experiment()
