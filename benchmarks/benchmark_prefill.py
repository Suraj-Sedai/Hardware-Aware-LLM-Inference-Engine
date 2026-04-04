"""Benchmark the prefill phase of LLM inference."""
import torch
from ..model_core.gpt import GPT
from ..kv_cache.contiguous_cache import KVCacheManager
from ..profiling import benchmark_generation


def run_prefill_benchmark(model_config, batch_size, prompt_len, device="cuda"):
    """Run a single prefill benchmark using the shared schema."""
    model = GPT(**model_config).to(device)
    kv_cache = KVCacheManager(
        n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"],
        max_seq_len=model_config["max_seq_len"],
        dim_head=model_config["dim"] // model_config["n_heads"],
        device=device,
        batch_size=batch_size
    )
    
    prompt_ids = torch.randint(1, model_config["vocab_size"], (batch_size, prompt_len), device=device)
    return benchmark_generation(
        model,
        prompt_ids,
        kv_cache,
        1,
        experiment_name="prefill",
        model_name=f"gpt_{model_config['dim']}d_{model_config['n_layers']}l",
        variant_name="kv_cache_on",
        config={"seq_len": model_config["max_seq_len"]},
    )


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
    result = run_prefill_benchmark(config, batch_size=1, prompt_len=512, device=device)
    for key, value in result["metrics"].items():
        print(f"{key}: {value:.4f}")
