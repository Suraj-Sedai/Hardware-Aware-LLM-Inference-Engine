"""Benchmark the prefill phase of LLM inference."""
import torch
import time
from ..model_core.gpt import GPT
from ..kv_cache.contiguous_cache import KVCacheManager
from ..runtime.controller import InferenceController


def run_prefill_benchmark(model_config, batch_size, prompt_len, device="cuda"):
    """Run a single prefill benchmark."""
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
    prompt_ids = torch.randint(1, model_config["vocab_size"], (batch_size, prompt_len), device=device)
    controller.warmup(prompt_ids, 1, trials=2)
    
    # Benchmark
    kv_cache.reset()
    res = controller.generate(prompt_ids, 1) # 1 token to trigger prefill
    
    return {
        "prefill_time_ms": res["phase_times"]["prefill"],
        "prefill_throughput_tokens_per_sec": (batch_size * prompt_len) / (res["phase_times"]["prefill"] / 1000)
    }


if __name__ == "__main__":
    config = {
        "vocab_size": 1000,
        "dim": 512,
        "n_heads": 8,
        "n_layers": 12,
        "max_seq_len": 1024,
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running prefill benchmark on {device}...")
    metrics = run_prefill_benchmark(config, batch_size=1, prompt_len=512, device=device)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
