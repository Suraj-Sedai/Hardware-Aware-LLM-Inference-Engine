"""Benchmark the system under realistic workloads."""
import torch
import numpy as np
from ..model_core.gpt import GPT
from ..kv_cache.contiguous_cache import KVCacheManager
from ..inference.controller import InferenceController
from ..inference.workload_simulator import WorkloadSimulator, run_workload
from ..profiling.metrics import calculate_metrics


def run_workload_benchmark(model_config, num_requests, device="cuda"):
    """Run a workload benchmark."""
    model = GPT(**model_config).to(device)
    # We use a large enough batch size for sequential simulation
    kv_cache = KVCacheManager(
        n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"],
        max_seq_len=model_config["max_seq_len"],
        dim_head=model_config["dim"] // model_config["n_heads"],
        device=device,
        batch_size=1 
    )
    
    controller = InferenceController(model, kv_cache, device)
    
    simulator = WorkloadSimulator(avg_arrival_rate=2.0, prompt_range=(16, 128), decode_range=(32, 256))
    requests = simulator.generate_requests(num_requests)
    
    results = run_workload(controller, requests)
    
    # Aggregate metrics
    all_ttfts = [res["phase_times"]["prefill"] for res in results]
    all_tpots = []
    for res in results:
        # res["latencies"] includes prefill and all decodes
        # TPOT is from latencies[1:]
        if len(res["latencies"]) > 1:
            all_tpots.extend(res["latencies"][1:])
    
    total_tokens = sum([res["tokens"].shape[1] - (requests[i]["prompt_len"]) for i, res in enumerate(results)])
    total_time = sum([sum(res["latencies"]) for res in results])
    
    return {
        "avg_ttft_ms": np.mean(all_ttfts),
        "p95_ttft_ms": np.percentile(all_ttfts, 95),
        "avg_tpot_ms": np.mean(all_tpots) * 1000,
        "p95_tpot_ms": np.percentile(all_tpots, 95) * 1000,
        "throughput_tokens_per_sec": total_tokens / total_time
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
    print(f"Running workload benchmark on {device}...")
    metrics = run_workload_benchmark(config, num_requests=10, device=device)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
