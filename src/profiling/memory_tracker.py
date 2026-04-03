"""Memory tracking for exact GPU accounting."""
import torch


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / (1024 * 1024)
    return 0.0


def get_gpu_max_memory_mb():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    return 0.0


def reset_gpu_memory_stats():
    """Reset peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)


def get_memory_stats():
    """Get detailed GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}
    
    stats = torch.cuda.memory_stats(0)
    return {
        "current_allocated_mb": stats.get("allocated_bytes.all.current", 0) / (1024 * 1024),
        "peak_allocated_mb": stats.get("allocated_bytes.all.peak", 0) / (1024 * 1024),
        "reserved_mb": stats.get("reserved_bytes.all.current", 0) / (1024 * 1024),
        "active_mb": stats.get("active_bytes.all.current", 0) / (1024 * 1024),
    }
