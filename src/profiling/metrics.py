"""Metrics calculation and result schema helpers for inference benchmarks."""
from datetime import datetime

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


def normalize_phase_times_ms(phase_times):
    """Return phase timings in milliseconds with consistent numeric values."""
    normalized = {}
    for phase_name, phase_time in (phase_times or {}).items():
        normalized[phase_name] = float(phase_time)
    return normalized


def build_benchmark_result(
    experiment_name,
    model_name,
    device,
    gen_result,
    total_tokens,
    config=None,
    variant_name=None,
    extras=None,
):
    """Build a standardized benchmark result shared across experiments."""
    latencies = gen_result.get("latencies", [])
    throughput_time = sum(latencies)
    metrics = calculate_metrics(latencies, total_tokens, throughput_time) if latencies else {}
    phase_times_ms = normalize_phase_times_ms(gen_result.get("phase_times", {}))
    tokens = gen_result.get("tokens")

    prompt_tokens = int(config.get("prompt_len", 0)) if config else 0
    output_tokens = int(tokens.shape[1]) if tokens is not None else prompt_tokens + total_tokens
    generated_tokens = max(output_tokens - prompt_tokens, 0)

    return {
        "experiment_name": experiment_name,
        "variant_name": variant_name or "default",
        "model_name": model_name,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "config": config or {},
        "metrics": {
            "ttft_ms": float(metrics.get("ttft_ms", 0.0)),
            "tpot_avg_ms": float(metrics.get("tpot_avg_ms", 0.0)),
            "tpot_p50_ms": float(metrics.get("tpot_p50_ms", 0.0)),
            "tpot_p95_ms": float(metrics.get("tpot_p95_ms", 0.0)),
            "tpot_p99_ms": float(metrics.get("tpot_p99_ms", 0.0)),
            "throughput_tokens_per_sec": float(metrics.get("throughput_tokens_per_sec", 0.0)),
            "prefill_latency_ms": float(phase_times_ms.get("prefill", 0.0)),
            "decode_latency_ms": float(phase_times_ms.get("decode", 0.0)),
            "total_latency_ms": float(throughput_time * 1000),
            "peak_memory_mb": float(gen_result.get("peak_memory_mb", 0.0)),
            "current_memory_mb": float(gen_result.get("current_memory_mb", 0.0)),
        },
        "phases_ms": phase_times_ms,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "output_tokens": output_tokens,
        "extras": extras or {},
    }


def format_benchmark_result(
    experiment_name,
    model_name,
    gen_result,
    metrics=None,
    config_overrides=None,
    device="unknown",
    variant_name=None,
    extras=None,
):
    """Backward-compatible wrapper around the standardized result schema."""
    total_tokens = int((config_overrides or {}).get("batch_size", 1) * (config_overrides or {}).get("decode_len", 0))
    result = build_benchmark_result(
        experiment_name=experiment_name,
        model_name=model_name,
        device=device,
        gen_result=gen_result,
        total_tokens=total_tokens,
        config=config_overrides,
        variant_name=variant_name,
        extras=extras,
    )
    if metrics:
        result["metrics"].update({
            "ttft_ms": float(metrics.get("ttft_ms", result["metrics"]["ttft_ms"])),
            "tpot_avg_ms": float(metrics.get("tpot_avg_ms", result["metrics"]["tpot_avg_ms"])),
            "tpot_p50_ms": float(metrics.get("tpot_p50_ms", result["metrics"]["tpot_p50_ms"])),
            "tpot_p95_ms": float(metrics.get("tpot_p95_ms", result["metrics"]["tpot_p95_ms"])),
            "tpot_p99_ms": float(metrics.get("tpot_p99_ms", result["metrics"]["tpot_p99_ms"])),
            "throughput_tokens_per_sec": float(metrics.get("throughput_tokens_per_sec", result["metrics"]["throughput_tokens_per_sec"])),
        })
    return result
