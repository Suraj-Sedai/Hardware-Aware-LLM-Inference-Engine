"""
Experiment 2: Batch Size Impact
Tests how different batch sizes affect inference performance.
"""
import time
import torch

from src.model_core import GPT
from src.kv_cache import KVCacheManager
from src.sampling import sample_top_k
from src.profiling import get_device, get_gpu_utilization, get_cpu_utilization


def run_batch_size_experiment(
    batch_sizes=None,
    seq_len=64,
    max_new_tokens=10,
    vocab_size=100,
    dim=32,
    n_heads=4,
    n_layers=2
):
    """
    Benchmark different batch sizes with fixed sequence length.
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: seq_len={seq_len}, max_new_tokens={max_new_tokens}")
    print(f"Model: vocab={vocab_size}, dim={dim}, heads={n_heads}, layers={n_layers}")
    print("-" * 60)
    
    results = []
    
    for batch_size in batch_sizes:
        # Create model
        model = GPT(vocab_size, dim, n_heads, n_layers, seq_len).to(device)
        model.eval()
        
        # Create input
        prompt_len = 4
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
        
        total_tokens = batch_size * max_new_tokens
        throughput = total_tokens / elapsed
        latency = elapsed / max_new_tokens
        
        result = {
            "batch_size": batch_size,
            "throughput": throughput,
            "latency": latency,
            "total_time": elapsed,
            "tokens_generated": total_tokens,
            "gpu_util": get_gpu_utilization(),
            "cpu_util": get_cpu_utilization()
        }
        results.append(result)
        
        print(f"batch={batch_size:3d} | {throughput:8.2f} tok/s | latency={latency:.4f}s | "
              f"GPU={result['gpu_util']:.1f}% | CPU={result['cpu_util']:.1f}%")
    
    return results


if __name__ == "__main__":
    run_batch_size_experiment()
