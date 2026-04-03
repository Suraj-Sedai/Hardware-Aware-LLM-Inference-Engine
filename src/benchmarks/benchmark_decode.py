"""Benchmark the decoding phase of LLM inference."""
import torch
import time
from ..model_core.gpt import GPT
from ..kv_cache.contiguous_cache import KVCacheManager
from ..runtime.controller import InferenceController
from ..profiling.metrics import calculate_metrics


def run_decode_benchmark(model_config, batch_size, decode_len, device="cuda"):
    """Run a single decode benchmark."""
    model = GPT(**model_config).to(device)
    kv_cache = KVCacheManager(
        n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"],
        max_seq_len=model_config["max_seq_len"],
        dim_head=model_config["dim"] // model_config["n_heads"],
        device=device,
        batch_size=batch_size
    )
    
    controller = InferenceController(model, kv_cache, device)
    
    # Warmup
    prompt_ids = torch.randint(1, model_config["vocab_size"], (batch_size, 1), device=device)
    controller.warmup(prompt_ids, decode_len, trials=2)
    
    # Benchmark
    kv_cache.reset()
    start_time = time.perf_counter()
    res = controller.generate(prompt_ids, decode_len)
    end_time = time.perf_counter()
    
    metrics = calculate_metrics(res["latencies"], batch_size * decode_len, end_time - start_time)
    return metrics


if __name__ == "__main__":
    config = {
        "vocab_size": 1000,
        "dim": 512,
        "n_heads": 8,
        "n_layers": 12,
        "max_seq_len": 1024,
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running decode benchmark on {device}...")
    metrics = run_decode_benchmark(config, batch_size=1, decode_len=100, device=device)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
