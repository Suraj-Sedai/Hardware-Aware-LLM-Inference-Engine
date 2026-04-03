"""Metrics calculation for inference performance."""
import numpy as np


def calculate_metrics(latencies, total_tokens, throughput_time):
    """Calculate TTFT, TPOT, throughput, and latency percentiles.
    
    latencies: List of token-to-token latencies in seconds.
    total_tokens: Total number of tokens generated.
    throughput_time: Total time for generation in seconds.
    """
    if not latencies:
        return {}
    
    ttft = latencies[0] * 1000  # ms
    tpot_list = latencies[1:] if len(latencies) > 1 else latencies
    tpot_ms = [t * 1000 for t in tpot_list]
    
    metrics = {
        "ttft_ms": ttft,
        "tpot_avg_ms": np.mean(tpot_ms),
        "tpot_p50_ms": np.percentile(tpot_ms, 50),
        "tpot_p95_ms": np.percentile(tpot_ms, 95),
        "tpot_p99_ms": np.percentile(tpot_ms, 99),
        "throughput_tokens_per_sec": total_tokens / throughput_time if throughput_time > 0 else 0,
    }
    
    return metrics
