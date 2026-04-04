"""Run inference benchmarks."""
import sys
sys.path.insert(0, "..")

import json
import torch
from src.model_core import GPT
from src.kv_cache import KVCacheManager
from src.sampling import sample_top_k
from src.profiling import get_device, benchmark_generation


def main():
    device = get_device()
    print(f"Device: {device}")
    
    # model config
    vocab_size, dim, n_heads, n_layers = 100, 32, 4, 2
    max_seq_len = 128
    batch_size = 1
    
    model = GPT(vocab_size, dim, n_heads, n_layers, max_seq_len).to(device)
    
    # test different sequence lengths
    for seq_len in [16, 32, 64, 128]:
        kv_cache = KVCacheManager(n_layers, n_heads, seq_len, dim // n_heads, device, batch_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, 4), device=device)
        
        result = benchmark_generation(
            model,
            input_ids,
            kv_cache,
            10,
            sample_top_k,
            experiment_name="generation_smoke",
            model_name=f"gpt_{dim}d_{n_layers}l",
            variant_name="kv_cache_on",
            config={"seq_len": seq_len},
        )
        print(
            f"seq_len={seq_len:3d} | "
            f"{result['metrics']['throughput_tokens_per_sec']:.2f} tok/s | "
            f"latency={result['metrics']['tpot_avg_ms']:.2f}ms"
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
