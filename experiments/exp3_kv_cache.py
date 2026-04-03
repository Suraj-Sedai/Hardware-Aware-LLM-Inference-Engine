"""
Experiment 3: KV Cache Efficiency
Tests KV cache memory usage and performance impact.

The KV cache speeds up autoregressive generation by storing computed
key/value pairs, avoiding redundant computation on previous tokens.
"""
import json
import time
import torch
from pathlib import Path

from src.model_core import GPT
from src.kv_cache import KVCacheManager
from src.runtime.controller import InferenceController
from src.profiling import get_device, calculate_metrics
from src.sampling import sample_top_k


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
    n_layers=8,
    save_results=True
):
    """
    Benchmark KV cache efficiency across different configurations.
    Uses InferenceController for cached version and manual generation for comparison.
    
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
        
        # ===== Test WITH KV cache using InferenceController =====
        kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        controller = InferenceController(model, kv_cache, device)
        
        # Warmup
        controller.warmup(input_ids, effective_new_tokens, trials=2)
        
        # Benchmark run with cache
        kv_cache.reset()
        cache_memory = get_cache_memory_mb(kv_cache)
        
        with torch.no_grad():
            gen_result = controller.generate(input_ids, effective_new_tokens)
        
        time_cached = sum(gen_result["latencies"])
        latencies_cached = gen_result["latencies"]
        metrics_cached = calculate_metrics(latencies_cached, effective_new_tokens, time_cached)
        
        # ===== Test WITHOUT KV cache (manual generation) =====
        # Warmup without cache
        with torch.no_grad():
            for _ in range(2):
                _ = model(input_ids, kv_cache=None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Benchmark without cache
        with torch.no_grad():
            generated = input_ids.clone()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_no_cache = time.perf_counter()
            for _ in range(effective_new_tokens):
                logits = model(generated, kv_cache=None)
                next_token = sample_top_k(logits[:, -1, :])
                generated = torch.cat([generated, next_token], dim=1)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time_no_cache = time.perf_counter() - start_no_cache
        
        speedup = time_no_cache / time_cached if time_cached > 0 else 0
        
        # Construct structured result
        result = {
            "experiment": "kv_cache",
            "seq_len": seq_len,
            "prompt_len": prompt_len,
            "decode_len": effective_new_tokens,
            "batch_size": batch_size,
            "cache_memory_mb": cache_memory,
            "time_with_cache_s": time_cached,
            "time_without_cache_s": time_no_cache,
            "speedup": speedup,
            "ttft_ms": metrics_cached["ttft_ms"],
            "tpot_avg_ms": metrics_cached["tpot_avg_ms"],
            "tpot_p95_ms": metrics_cached["tpot_p95_ms"],
            "throughput_tokens_per_sec": metrics_cached["throughput_tokens_per_sec"],
            "throughput_no_cache": effective_new_tokens / time_no_cache,
            "peak_memory_mb": gen_result["peak_memory_mb"],
            "phase_times": gen_result["phase_times"],
        }
        results.append(result)
        
        print(f"seq_len={seq_len:4d} | cache={cache_memory:.2f}MB | "
              f"cached={time_cached:.4f}s | no_cache={time_no_cache:.4f}s | "
              f"speedup={speedup:.2f}x | peak_mem={result['peak_memory_mb']:.2f}MB")
    
    # Save results
    if save_results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / "exp3_kv_cache.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    run_kv_cache_experiment()
