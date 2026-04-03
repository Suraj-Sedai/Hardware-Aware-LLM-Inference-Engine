"""
Experiment 1: Sequence Length Impact
Tests how different sequence lengths affect inference performance.
"""
import json
import torch
from pathlib import Path

from src.model_core import GPT
from src.kv_cache import KVCacheManager
from src.inference.controller import InferenceController
from src.profiling import get_device, calculate_metrics, format_benchmark_result


def run_sequence_length_experiment(
    seq_lengths=None,
    batch_size=1,
    max_new_tokens=10,
    vocab_size=100,
    dim=32,
    n_heads=4,
    n_layers=2,
    save_results=True
):
    """
    Benchmark different sequence lengths with fixed batch size.
    Uses InferenceController and structured metrics.
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
        
        # Create controller
        kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        controller = InferenceController(model, kv_cache, device)
        
        # Warmup
        controller.warmup(input_ids, max_new_tokens, trials=2)
        
        # Benchmark run
        kv_cache.reset()
        with torch.no_grad():
            gen_result = controller.generate(input_ids, max_new_tokens)
        
        # Calculate metrics
        metrics = calculate_metrics(
            gen_result["latencies"], 
            batch_size * max_new_tokens, 
            sum(gen_result["latencies"])
        )
        
        # Construct standardized result
        result = format_benchmark_result(
            experiment_name="sequence_length",
            model_name=f"gpt_{dim}d_{n_layers}l",
            gen_result=gen_result,
            metrics=metrics,
            config_overrides={
                "seq_len": seq_len,
                "prompt_len": prompt_len,
                "decode_len": max_new_tokens,
                "batch_size": batch_size,
            }
        )
        results.append(result)
        
        print(f"seq_len={seq_len:4d} | TTFT={result['ttft_ms']:7.2f}ms | "
              f"TPOT={result['tpot_avg_ms']:6.2f}ms | "
              f"throughput={result['throughput_tokens_per_sec']:7.2f} tok/s | "
              f"peak_mem={result['peak_memory_mb']:.2f}MB")
    
    # Save results
    if save_results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / "exp1_sequence_length.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    run_sequence_length_experiment()
