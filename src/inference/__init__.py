"""Inference module."""
from .generate import generate, generate_greedy
from .controller import InferenceController
from .workload_simulator import WorkloadSimulator, run_workload
