"""
Experiment 4: Quantization Impact
Tests the impact of INT8 quantization on model size and performance.

Compares:
1. FP32 baseline on GPU
2. Manual INT8 with cached dequantization on GPU  
3. PyTorch dynamic quantization on CPU (optimized INT8 kernels)
"""
import json
import torch
from pathlib import Path

from src.model_core import GPT
from src.kv_cache import KVCacheManager
from src.inference.controller import InferenceController
from src.profiling import get_device, calculate_metrics, format_benchmark_result
from src.optimizations.quantization import (
    quantize_model, 
    quantize_model_dynamic,
    get_model_size_mb
)


def run_quantization_experiment(
    model_configs=None,
    batch_size=1,
    seq_len=128,
    max_new_tokens=50,
    vocab_size=1000,
    save_results=True
):
    """
    Benchmark quantization impact on model size and inference speed.
    Uses InferenceController and structured metrics.
    """
    if model_configs is None:
        model_configs = [
            {"dim": 256, "n_heads": 4, "n_layers": 4, "name": "small"},
            {"dim": 512, "n_heads": 8, "n_layers": 6, "name": "medium"},
            {"dim": 768, "n_heads": 12, "n_layers": 8, "name": "large"},
        ]
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, max_new_tokens={max_new_tokens}")
    print("-" * 70)
    
    results = []
    
    for config in model_configs:
        dim = config["dim"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        name = config["name"]
        
        # ===== FP32 Baseline =====
        model_fp32 = GPT(vocab_size, dim, n_heads, n_layers, seq_len).to(device)
        model_fp32.eval()
        size_fp32 = get_model_size_mb(model_fp32)
        
        prompt_len = 8
        input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)
        
        kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        controller = InferenceController(model_fp32, kv_cache, device)
        
        # Warmup
        controller.warmup(input_ids, max_new_tokens, trials=2)
        
        # Benchmark FP32
        kv_cache.reset()
        with torch.no_grad():
            res_fp32 = controller.generate(input_ids, max_new_tokens)
        
        metrics_fp32 = calculate_metrics(res_fp32["latencies"], batch_size * max_new_tokens, sum(res_fp32["latencies"]))
        
        formatted_fp32 = format_benchmark_result(
            experiment_name="quantization",
            model_name=f"{name}_fp32",
            gen_result=res_fp32,
            metrics=metrics_fp32,
            config_overrides={
                "precision": "fp32",
                "size_mb": size_fp32,
                "prompt_len": prompt_len,
                "decode_len": max_new_tokens,
                "batch_size": batch_size,
            }
        )
        results.append(formatted_fp32)
        
        # ===== INT8 Manual (GPU if available) =====
        if device == "cuda":
            model_int8 = GPT(vocab_size, dim, n_heads, n_layers, seq_len).to(device)
            model_int8.load_state_dict(model_fp32.state_dict())
            model_int8 = quantize_model(model_int8, use_pytorch_quantization=False)
            model_int8.eval()
            size_int8 = get_model_size_mb(model_int8)
            
            kv_cache_int8 = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
            controller_int8 = InferenceController(model_int8, kv_cache_int8, device)
            
            # Warmup
            controller_int8.warmup(input_ids, max_new_tokens, trials=2)
            
            # Benchmark INT8
            kv_cache_int8.reset()
            with torch.no_grad():
                res_int8 = controller_int8.generate(input_ids, max_new_tokens)
            
            metrics_int8 = calculate_metrics(res_int8["latencies"], batch_size * max_new_tokens, sum(res_int8["latencies"]))
            
            formatted_int8 = format_benchmark_result(
                experiment_name="quantization",
                model_name=f"{name}_int8_manual",
                gen_result=res_int8,
                metrics=metrics_int8,
                config_overrides={
                    "precision": "int8_manual",
                    "size_mb": size_int8,
                    "prompt_len": prompt_len,
                    "decode_len": max_new_tokens,
                    "batch_size": batch_size,
                }
            )
            results.append(formatted_int8)
            
            print(f"{name:8s} | FP32: size={size_fp32:6.1f}MB, throughput={metrics_fp32['throughput_tokens_per_sec']:8.2f} tok/s")
            print(f"         | INT8: size={size_int8:6.1f}MB, throughput={metrics_int8['throughput_tokens_per_sec']:8.2f} tok/s")
        else:
            print(f"{name:8s} | FP32: size={size_fp32:6.1f}MB, throughput={metrics_fp32['throughput_tokens_per_sec']:8.2f} tok/s")
            print(f"         | INT8 manual quantization skipped (no CUDA)")

    # Save results
    if save_results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / "exp4_quantization.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    run_quantization_experiment()
