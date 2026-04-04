"""
Experiment 3: KV Cache Efficiency
Tests KV cache memory usage and performance impact.

The KV cache speeds up autoregressive generation by storing computed
key/value pairs, avoiding redundant computation on previous tokens.
"""
import json
import torch
from pathlib import Path

from src.model_core import GPT
from src.kv_cache import KVCacheManager
from src.inference.controller import InferenceController
from src.profiling import get_device, build_benchmark_result


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
    Uses the same controller-driven measurement path for cached and no-cache runs.
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
        
        res_cached = build_benchmark_result(
            experiment_name="kv_cache",
            model_name=f"gpt_{dim}d_{n_layers}l",
            device=device,
            gen_result=gen_result,
            total_tokens=batch_size * effective_new_tokens,
            config={
                "seq_len": seq_len,
                "prompt_len": prompt_len,
                "decode_len": effective_new_tokens,
                "batch_size": batch_size,
            },
            variant_name="kv_cache_on",
            extras={
                "kv_cache_enabled": True,
                "cache_memory_mb": cache_memory,
            }
        )
        results.append(res_cached)
        
        # ===== Test WITHOUT KV cache via the same controller =====
        no_cache_controller = InferenceController(model, None, device)
        no_cache_controller.warmup(input_ids, effective_new_tokens, trials=2)

        with torch.no_grad():
            gen_result_no_cache = no_cache_controller.generate(
                input_ids,
                effective_new_tokens,
                use_kv_cache=False,
            )

        res_no_cache = build_benchmark_result(
            experiment_name="kv_cache",
            model_name=f"gpt_{dim}d_{n_layers}l",
            device=device,
            gen_result=gen_result_no_cache,
            total_tokens=batch_size * effective_new_tokens,
            config={
                "seq_len": seq_len,
                "prompt_len": prompt_len,
                "decode_len": effective_new_tokens,
                "batch_size": batch_size,
            },
            variant_name="kv_cache_off",
            extras={
                "kv_cache_enabled": False,
                "cache_memory_mb": 0.0,
            }
        )
        results.append(res_no_cache)
        
        cached_total_ms = res_cached["metrics"]["total_latency_ms"]
        no_cache_total_ms = res_no_cache["metrics"]["total_latency_ms"]
        speedup = no_cache_total_ms / cached_total_ms if cached_total_ms > 0 else 0
        
        print(f"seq_len={seq_len:4d} | cache={cache_memory:.2f}MB | "
              f"cached={cached_total_ms/1000:.4f}s | no_cache={no_cache_total_ms/1000:.4f}s | "
              f"speedup={speedup:.2f}x | peak_mem={res_cached['metrics']['peak_memory_mb']:.2f}MB")
    
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
