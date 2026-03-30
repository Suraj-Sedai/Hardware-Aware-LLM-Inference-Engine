"""
Experiment 4: Quantization Impact
Tests the impact of INT8 quantization on model size and performance.

Compares:
1. FP32 baseline on GPU
2. Manual INT8 with cached dequantization on GPU  
3. PyTorch dynamic quantization on CPU (optimized INT8 kernels)
"""
import time
import copy
import torch

from src.model_core import GPT
from src.kv_cache import KVCacheManager
from src.sampling import sample_top_k
from src.profiling import get_device
from src.optimizations.quantization import (
    quantize_model, 
    quantize_model_dynamic,
    get_model_size_mb
)


def benchmark_model(model, input_ids, n_layers, n_heads, seq_len, dim, 
                   device, batch_size, max_new_tokens, num_runs=3):
    """Benchmark a model with warmup and multiple runs."""
    # Warmup
    kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
    with torch.no_grad():
        _ = model(input_ids, kv_cache)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark runs
    times = []
    for _ in range(num_runs):
        kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        with torch.no_grad():
            _ = model(input_ids, kv_cache)
            next_tok = input_ids[:, -1:]
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(max_new_tokens):
                logits = model(next_tok, kv_cache)
                next_tok = torch.argmax(logits[:, -1:], dim=-1)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    return min(times)  # Return best time


def run_quantization_experiment(
    model_configs=None,
    batch_size=1,
    seq_len=128,
    max_new_tokens=50,
    vocab_size=1000
):
    """
    Benchmark quantization impact on model size and inference speed.
    
    Tests larger models to show meaningful differences.
    """
    if model_configs is None:
        model_configs = [
            {"dim": 256, "n_heads": 4, "n_layers": 4, "name": "small"},
            {"dim": 512, "n_heads": 8, "n_layers": 6, "name": "medium"},
            {"dim": 768, "n_heads": 12, "n_layers": 8, "name": "large"},
        ]
    
    gpu_device = get_device()
    print(f"GPU Device: {gpu_device}")
    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, max_new_tokens={max_new_tokens}")
    print("-" * 70)
    
    results = []
    
    for config in model_configs:
        dim = config["dim"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        name = config["name"]
        
        # ===== FP32 on GPU =====
        model_fp32_gpu = GPT(vocab_size, dim, n_heads, n_layers, seq_len).to(gpu_device)
        model_fp32_gpu.eval()
        size_fp32 = get_model_size_mb(model_fp32_gpu)
        
        input_ids_gpu = torch.randint(0, vocab_size, (batch_size, 8), device=gpu_device)
        time_fp32_gpu = benchmark_model(
            model_fp32_gpu, input_ids_gpu, n_layers, n_heads, seq_len, dim,
            gpu_device, batch_size, max_new_tokens
        )
        
        # ===== INT8 (manual, cached) on GPU =====
        model_int8_gpu = GPT(vocab_size, dim, n_heads, n_layers, seq_len).to(gpu_device)
        model_int8_gpu.load_state_dict(model_fp32_gpu.state_dict())
        model_int8_gpu = quantize_model(model_int8_gpu, use_pytorch_quantization=False)
        model_int8_gpu.eval()
        size_int8_manual = get_model_size_mb(model_int8_gpu)
        
        time_int8_gpu = benchmark_model(
            model_int8_gpu, input_ids_gpu, n_layers, n_heads, seq_len, dim,
            gpu_device, batch_size, max_new_tokens
        )
        
        # ===== PyTorch Dynamic Quantization on CPU =====
        model_fp32_cpu = GPT(vocab_size, dim, n_heads, n_layers, seq_len).cpu()
        model_fp32_cpu.load_state_dict(model_fp32_gpu.cpu().state_dict())
        model_fp32_cpu.eval()
        
        input_ids_cpu = torch.randint(0, vocab_size, (batch_size, 8))
        
        # FP32 CPU baseline
        time_fp32_cpu = benchmark_model(
            model_fp32_cpu, input_ids_cpu, n_layers, n_heads, seq_len, dim,
            'cpu', batch_size, max_new_tokens, num_runs=2
        )
        
        # PyTorch optimized INT8 on CPU
        try:
            model_int8_cpu = quantize_model_dynamic(model_fp32_cpu)
            model_int8_cpu.eval()
            size_int8_pytorch = get_model_size_mb(model_int8_cpu)
            
            time_int8_cpu = benchmark_model(
                model_int8_cpu, input_ids_cpu, n_layers, n_heads, seq_len, dim,
                'cpu', batch_size, max_new_tokens, num_runs=2
            )
            cpu_speedup = time_fp32_cpu / time_int8_cpu
        except Exception as e:
            print(f"  PyTorch quantization failed: {e}")
            time_int8_cpu = time_fp32_cpu
            size_int8_pytorch = size_fp32
            cpu_speedup = 1.0
        
        # Calculate metrics
        compression_manual = size_fp32 / size_int8_manual if size_int8_manual > 0 else 0
        gpu_speedup = time_fp32_gpu / time_int8_gpu if time_int8_gpu > 0 else 0
        
        result = {
            "model": name,
            "dim": dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "size_fp32_mb": size_fp32,
            "size_int8_mb": size_int8_manual,
            "compression_ratio": compression_manual,
            "time_fp32_gpu": time_fp32_gpu,
            "time_int8_gpu": time_int8_gpu,
            "gpu_speedup": gpu_speedup,
            "time_fp32_cpu": time_fp32_cpu,
            "time_int8_cpu": time_int8_cpu,
            "cpu_speedup": cpu_speedup,
            "throughput_fp32": max_new_tokens / time_fp32_gpu,
            "throughput_int8": max_new_tokens / time_int8_gpu
        }
        results.append(result)
        
        print(f"{name:8s} | FP32={size_fp32:.1f}MB INT8={size_int8_manual:.1f}MB ({compression_manual:.1f}x compression)")
        print(f"         | GPU: FP32={time_fp32_gpu:.3f}s INT8={time_int8_gpu:.3f}s ({gpu_speedup:.2f}x)")
        print(f"         | CPU: FP32={time_fp32_cpu:.3f}s INT8={time_int8_cpu:.3f}s ({cpu_speedup:.2f}x) [PyTorch optimized]")
    
    return results


if __name__ == "__main__":
    run_quantization_experiment()
