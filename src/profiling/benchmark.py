"""Profiling and benchmarking utilities."""
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


def benchmark_generation(
    model,
    input_ids,
    kv_cache,
    max_new_tokens,
    sample_fn=None,
    *,
    experiment_name="generation",
    model_name="model",
    variant_name="default",
    config=None,
    use_kv_cache=True,
):
    """Benchmark generation and return the shared result schema."""
    from ..inference.controller import InferenceController
    from .metrics import build_benchmark_result

    device = input_ids.device
    controller = InferenceController(model, kv_cache, device)
    
    # Warmup
    controller.warmup(input_ids, max_new_tokens, trials=1)
    
    # Benchmark
    if kv_cache is not None:
        kv_cache.reset()
    result = controller.generate(input_ids, max_new_tokens, sample_fn=sample_fn, use_kv_cache=use_kv_cache)

    benchmark_config = {
        "batch_size": int(input_ids.shape[0]),
        "prompt_len": int(input_ids.shape[1]),
        "decode_len": int(max_new_tokens),
    }
    if config:
        benchmark_config.update(config)

    return build_benchmark_result(
        experiment_name=experiment_name,
        model_name=model_name,
        device=device,
        gen_result=result,
        total_tokens=input_ids.shape[0] * max_new_tokens,
        config=benchmark_config,
        variant_name=variant_name,
        extras={
            "gpu_util": float(get_gpu_utilization()),
            "cpu_util": float(get_cpu_utilization()),
            "kv_cache_enabled": bool(use_kv_cache and kv_cache is not None),
        },
    )
