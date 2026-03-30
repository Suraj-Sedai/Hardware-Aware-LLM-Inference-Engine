"""Profiling and benchmarking utilities."""
import time
import torch

try:
    import psutil
except ImportError:
    psutil = None

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
    nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False


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
    """Get GPU utilization percentage using pynvml."""
    if HAS_NVML:
        try:
            handle = nvmlDeviceGetHandleByIndex(0)
            util = nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except Exception:
            pass
    # Fallback: return memory utilization as proxy
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            return (allocated / total) * 100
        except Exception:
            pass
    return 0.0


def get_gpu_memory_mb():
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        try:
            return torch.cuda.memory_allocated(0) / (1024 * 1024)
        except Exception:
            pass
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
