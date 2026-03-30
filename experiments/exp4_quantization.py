"""
Experiment 4: Quantization Impact
Tests the impact of INT8 quantization on model size and performance.
"""
import time
import torch

from src.model_core import GPT
from src.kv_cache import KVCacheManager
from src.sampling import sample_top_k
from src.profiling import get_device
from src.optimizations.quantization import quantize_model, get_model_size_mb


def run_quantization_experiment(
    model_configs=None,
    batch_size=1,
    seq_len=64,
    max_new_tokens=10,
    vocab_size=100
):
    """
    Benchmark quantization impact on model size and inference speed.
    """
    if model_configs is None:
        model_configs = [
            {"dim": 32, "n_heads": 4, "n_layers": 2, "name": "small"},
            {"dim": 64, "n_heads": 4, "n_layers": 4, "name": "medium"},
            {"dim": 128, "n_heads": 8, "n_layers": 4, "name": "large"},
        ]
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, max_new_tokens={max_new_tokens}")
    print("-" * 60)
    
    results = []
    
    for config in model_configs:
        dim = config["dim"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        name = config["name"]
        
        # Create FP32 model
        model_fp32 = GPT(vocab_size, dim, n_heads, n_layers, seq_len).to(device)
        model_fp32.eval()
        size_fp32 = get_model_size_mb(model_fp32)
        
        # Create quantized model (copy first)
        model_int8 = GPT(vocab_size, dim, n_heads, n_layers, seq_len).to(device)
        model_int8.load_state_dict(model_fp32.state_dict())
        model_int8 = quantize_model(model_int8)
        model_int8.eval()
        size_int8 = get_model_size_mb(model_int8)
        
        # Test input
        input_ids = torch.randint(0, vocab_size, (batch_size, 4), device=device)
        
        # Benchmark FP32
        kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        with torch.no_grad():
            _ = model_fp32(input_ids, kv_cache)
            generated = input_ids.clone()
            
            start = time.time()
            for _ in range(max_new_tokens):
                x = generated[:, -1:]
                logits = model_fp32(x, kv_cache)
                next_token = sample_top_k(logits[:, -1, :])
                generated = torch.cat([generated, next_token], dim=1)
            time_fp32 = time.time() - start
        
        # Benchmark INT8
        kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        with torch.no_grad():
            _ = model_int8(input_ids, kv_cache)
            generated = input_ids.clone()
            
            start = time.time()
            for _ in range(max_new_tokens):
                x = generated[:, -1:]
                logits = model_int8(x, kv_cache)
                next_token = sample_top_k(logits[:, -1, :])
                generated = torch.cat([generated, next_token], dim=1)
            time_int8 = time.time() - start
        
        compression = size_fp32 / size_int8 if size_int8 > 0 else 0
        speedup = time_fp32 / time_int8 if time_int8 > 0 else 0
        
        result = {
            "model": name,
            "dim": dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "size_fp32_mb": size_fp32,
            "size_int8_mb": size_int8,
            "compression_ratio": compression,
            "time_fp32": time_fp32,
            "time_int8": time_int8,
            "speedup": speedup,
            "throughput_fp32": max_new_tokens / time_fp32,
            "throughput_int8": max_new_tokens / time_int8
        }
        results.append(result)
        
        print(f"{name:8s} | FP32={size_fp32:.2f}MB INT8={size_int8:.2f}MB ({compression:.2f}x) | "
              f"FP32={time_fp32:.4f}s INT8={time_int8:.4f}s ({speedup:.2f}x)")
    
    return results


if __name__ == "__main__":
    run_quantization_experiment()
