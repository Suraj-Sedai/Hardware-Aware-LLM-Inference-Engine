"""Metrics calculation for inference performance."""
import numpy as np
from datetime import datetime


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


def format_benchmark_result(
    experiment_name,
    model_name,
    gen_result,
    metrics,
    config_overrides=None
):
    """Format benchmark result into a standardized schema as required by prompt.md."""
    phase_times = gen_result.get("phase_times", {})
    prefill_ms = phase_times.get("prefill", 0) * 1000
    decode_ms = phase_times.get("decode", 0) * 1000
    total_ms = (sum(gen_result.get("latencies", []))) * 1000
    
    result = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        
        # Performance metrics
        "ttft_ms": metrics.get("ttft_ms", 0),
        "tpot_avg_ms": metrics.get("tpot_avg_ms", 0),
        "tpot_p50_ms": metrics.get("tpot_p50_ms", 0),
        "tpot_p95_ms": metrics.get("tpot_p95_ms", 0),
        "tpot_p99_ms": metrics.get("tpot_p99_ms", 0),
        "throughput_tokens_per_sec": metrics.get("throughput_tokens_per_sec", 0),
        
        # Latency breakdown
        "prefill_latency_ms": prefill_ms,
        "decode_latency_ms": decode_ms,
        "total_latency_ms": total_ms,
        
        # Memory metrics
        "peak_memory_mb": gen_result.get("peak_memory_mb", 0),
        "current_memory_mb": gen_result.get("current_memory_mb", 0),
    }
    
    # Add optional config fields
    if config_overrides:
        result.update(config_overrides)
        
    return result
