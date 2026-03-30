"""
Experiment 1: Sequence Length Impact
Tests how different sequence lengths affect inference performance.
"""
import time
import torch

from src.model_core import GPT
from src.kv_cache import KVCacheManager
from src.sampling import sample_top_k
from src.profiling import get_device, get_gpu_utilization, get_cpu_utilization


def run_sequence_length_experiment(
    seq_lengths=None,
    batch_size=1,
    max_new_tokens=10,
    vocab_size=100,
    dim=32,
    n_heads=4,
    n_layers=2
):
    """
    Benchmark different sequence lengths with fixed batch size.
    """
    if seq_lengths is None:
        seq_lengths = [16, 32, 64, 128]
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: batch_size={batch_size}, max_new_tokens={max_new_tokens}")
    print(f"Model: vocab={vocab_size}, dim={dim}, heads={n_heads}, layers={n_layers}")
    print("-" * 60)
    
    results = []
    
    for seq_len in seq_lengths:
        # Create model and cache
        model = GPT(vocab_size, dim, n_heads, n_layers, seq_len).to(device)
        model.eval()
        
        # Create input
        prompt_len = min(4, seq_len - max_new_tokens)
        input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)
        
        # Warm-up run
        kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        with torch.no_grad():
            _ = model(input_ids, kv_cache)
        
        # Benchmark run
        kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        
        with torch.no_grad():
            # Prefill
            _ = model(input_ids, kv_cache)
            
            # Decode
            generated = input_ids.clone()
            start = time.time()
            
            for _ in range(max_new_tokens):
                x = generated[:, -1:]
                logits = model(x, kv_cache)
                next_token = sample_top_k(logits[:, -1, :])
                generated = torch.cat([generated, next_token], dim=1)
            
            elapsed = time.time() - start
        
        throughput = (batch_size * max_new_tokens) / elapsed
        latency = elapsed / max_new_tokens
        
        result = {
            "seq_len": seq_len,
            "throughput": throughput,
            "latency": latency,
            "total_time": elapsed,
            "gpu_util": get_gpu_utilization(),
            "cpu_util": get_cpu_utilization()
        }
        results.append(result)
        
        print(f"seq_len={seq_len:4d} | {throughput:7.2f} tok/s | latency={latency:.4f}s | "
              f"GPU={result['gpu_util']:.1f}% | CPU={result['cpu_util']:.1f}%")
    
    return results


if __name__ == "__main__":
    run_sequence_length_experiment()
