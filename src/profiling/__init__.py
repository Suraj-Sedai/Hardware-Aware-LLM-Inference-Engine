"""Profiling module."""
from .memory_tracker import get_gpu_memory_mb, get_gpu_max_memory_mb, get_memory_stats
from .latency_tracker import LatencyTracker
from .metrics import calculate_metrics
