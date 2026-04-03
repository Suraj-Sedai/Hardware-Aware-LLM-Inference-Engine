"""
Run all benchmarks and generate results.
Usage: python -m benchmarks.run_all
"""
import os
import sys
import json
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.exp1_sequence_length import run_sequence_length_experiment
from benchmarks.exp2_batch_size import run_batch_size_experiment
from benchmarks.exp3_kv_cache import run_kv_cache_experiment
from benchmarks.exp4_quantization import run_quantization_experiment
from benchmarks.visualize import plot_all_results


def run_all_experiments(output_dir="results"):
    """Run all experiments and save results."""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 60)
    print("Hardware-Aware LLM Inference Engine - Experiments")
    print("=" * 60)
    print(f"Results will be saved to: {results_dir}\n")
    
    all_results = {}
    
    # Experiment 1: Sequence Length Impact
    print("\n" + "=" * 60)
    print("Experiment 1: Sequence Length Impact")
    print("=" * 60)
    exp1_results = run_sequence_length_experiment()
    all_results["sequence_length"] = exp1_results
    
    # Experiment 2: Batch Size Impact
    print("\n" + "=" * 60)
    print("Experiment 2: Batch Size Impact")
    print("=" * 60)
    exp2_results = run_batch_size_experiment()
    all_results["batch_size"] = exp2_results
    
    # Experiment 3: KV Cache Efficiency
    print("\n" + "=" * 60)
    print("Experiment 3: KV Cache Efficiency")
    print("=" * 60)
    exp3_results = run_kv_cache_experiment()
    all_results["kv_cache"] = exp3_results
    
    # Experiment 4: Quantization Impact
    print("\n" + "=" * 60)
    print("Experiment 4: Quantization Impact")
    print("=" * 60)
    exp4_results = run_quantization_experiment()
    all_results["quantization"] = exp4_results
    
    # Save results as JSON
    results_file = os.path.join(results_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_all_results(all_results, results_dir)
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    run_all_experiments()
