"""Profiling and benchmarking utilities."""
import time
import torch

try:
    import psutil
except ImportError:
    psutil = None

try:
    import GPUtil
except ImportError:
    GPUtil = None


def get_device():
    """Get best available device with fallback to CPU."""
    if torch.cuda.is_available():
        try:
            test = torch.zeros(1, device="cuda")
            del test
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


def get_gpu_utilization():
    """Get GPU utilization percentage."""
    if GPUtil is None:
        return 0.0
    try:
        gpus = GPUtil.getGPUs()
        return gpus[0].load * 100 if gpus else 0.0
    except Exception:
        return 0.0


def get_cpu_utilization():
    """Get CPU utilization percentage."""
    if psutil is None:
        return 0.0
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception:
        return 0.0


def benchmark_generation(model, input_ids, kv_cache, max_new_tokens, sample_fn):
    """Benchmark token generation speed."""
    generated = input_ids.clone()
    
    start = time.time()
    for _ in range(max_new_tokens):
        x = generated[:, -1:]
        logits = model(x, kv_cache)
        next_token = sample_fn(logits[:, -1, :])
        generated = torch.cat([generated, next_token], dim=1)
    elapsed = time.time() - start
    
    return {
        "total_time": elapsed,
        "tokens_per_sec": max_new_tokens / elapsed,
        "latency_per_token": elapsed / max_new_tokens,
        "gpu_util": get_gpu_utilization(),
        "cpu_util": get_cpu_utilization(),
    }
