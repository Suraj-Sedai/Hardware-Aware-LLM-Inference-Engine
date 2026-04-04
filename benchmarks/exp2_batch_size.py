"""
Experiment 2: Batch Size Impact
Tests how different batch sizes affect inference performance.
"""
import json
import torch
from pathlib import Path

from src.model_core import GPT
from src.kv_cache import KVCacheManager
from src.inference.controller import InferenceController
from src.profiling import get_device, build_benchmark_result


def run_batch_size_experiment(
    batch_sizes=None,
    seq_len=64,
    max_new_tokens=10,
    vocab_size=100,
    dim=32,
    n_heads=4,
    n_layers=2,
    save_results=True
):
    """
    Benchmark different batch sizes with fixed sequence length.
    Uses InferenceController and structured metrics.
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
        
        # Create controller
        kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        controller = InferenceController(model, kv_cache, device)
        
        # Warmup
        controller.warmup(input_ids, max_new_tokens, trials=2)
        
        # Benchmark run
        kv_cache.reset()
        with torch.no_grad():
            gen_result = controller.generate(input_ids, max_new_tokens)
        
        result = build_benchmark_result(
            experiment_name="batch_size",
            model_name=f"gpt_{dim}d_{n_layers}l",
            device=device,
            gen_result=gen_result,
            total_tokens=batch_size * max_new_tokens,
            config={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "prompt_len": prompt_len,
                "decode_len": max_new_tokens,
            }
        )
        results.append(result)
        
        print(f"batch={batch_size:3d} | TTFT={result['metrics']['ttft_ms']:7.2f}ms | "
              f"TPOT={result['metrics']['tpot_avg_ms']:6.2f}ms | "
              f"throughput={result['metrics']['throughput_tokens_per_sec']:8.2f} tok/s | "
              f"peak_mem={result['metrics']['peak_memory_mb']:.2f}MB")
    
    # Save results
    if save_results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / "exp2_batch_size.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    run_batch_size_experiment()
