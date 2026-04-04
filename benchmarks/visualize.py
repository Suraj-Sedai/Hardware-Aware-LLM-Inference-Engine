"""
Visualization utilities for experiment results.
Generates plots and saves them as images.
"""
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will not be generated.")


def _config(result, key, default=0):
    return result.get("config", {}).get(key, default)


def _metric(result, key, default=0.0):
    return result.get("metrics", {}).get(key, default)


def _extra(result, key, default=0):
    return result.get("extras", {}).get(key, default)


def plot_sequence_length_results(results, save_path=None):
    """Plot sequence length experiment results."""
    if not HAS_MATPLOTLIB or not results:
        return
    
    seq_lens = [_config(r, "seq_len") for r in results]
    throughputs = [_metric(r, "throughput_tokens_per_sec") for r in results]
    latencies = [_metric(r, "total_latency_ms") / 1000 for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Throughput plot
    ax1.plot(seq_lens, throughputs, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Throughput (tokens/sec)')
    ax1.set_title('Throughput vs Sequence Length')
    ax1.grid(True, alpha=0.3)
    
    # Latency plot
    ax2.plot(seq_lens, latencies, 'r-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Latency (s)')
    ax2.set_title('Latency vs Sequence Length')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_batch_size_results(results, save_path=None):
    """Plot batch size experiment results."""
    if not HAS_MATPLOTLIB or not results:
        return
    
    batch_sizes = [_config(r, "batch_size") for r in results]
    throughputs = [_metric(r, "throughput_tokens_per_sec") for r in results]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.bar(range(len(batch_sizes)), throughputs, color='steelblue')
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (tokens/sec)')
    ax.set_title('Throughput vs Batch Size')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(throughputs):
        ax.text(i, v + max(throughputs) * 0.02, f'{v:.0f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_kv_cache_results(results, save_path=None):
    """Plot KV cache experiment results."""
    if not HAS_MATPLOTLIB or not results:
        return
    
    # Extract only cached results for sequence length and memory
    cached_results = [r for r in results if _extra(r, "kv_cache_enabled", True)]
    
    # Extract speedup (needs both cached and no-cache for same seq_len)
    seq_lens = []
    speedups = []
    
    seq_to_cached = {_config(r, "seq_len"): r for r in results if _extra(r, "kv_cache_enabled", True)}
    seq_to_nocache = {_config(r, "seq_len"): r for r in results if not _extra(r, "kv_cache_enabled", True)}
    
    for seq in sorted(seq_to_cached.keys()):
        if seq in seq_to_nocache:
            seq_lens.append(seq)
            cached_lat = _metric(seq_to_cached[seq], "total_latency_ms")
            nocache_lat = _metric(seq_to_nocache[seq], "total_latency_ms")
            speedups.append(nocache_lat / cached_lat if cached_lat > 0 else 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Speedup plot
    ax1.bar(range(len(seq_lens)), speedups, color='green', alpha=0.7)
    ax1.set_xticks(range(len(seq_lens)))
    ax1.set_xticklabels(seq_lens)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Speedup (x)')
    ax1.set_title('KV Cache Speedup')
    ax1.axhline(y=1, color='r', linestyle='--', label='No speedup')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # Memory plot
    cached_seqs = [_config(r, "seq_len") for r in cached_results]
    memory = [_extra(r, "cache_memory_mb") for r in cached_results]
    ax2.plot(cached_seqs, memory, 'purple', marker='o', linewidth=2, markersize=8)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Cache Memory (MB)')
    ax2.set_title('KV Cache Memory Usage')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_quantization_results(results, save_path=None):
    """Plot quantization experiment results."""
    if not HAS_MATPLOTLIB or not results:
        return
    
    # Group results by model base name (e.g., small, medium, large)
    model_groups = {}
    for r in results:
        base_name = r["model_name"].split('_')[0]
        if base_name not in model_groups:
            model_groups[base_name] = {}
        model_groups[base_name][_extra(r, "precision")] = r
    
    bases = sorted(model_groups.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = range(len(bases))
    width = 0.35
    
    sizes_fp32 = [_extra(model_groups[b]["fp32"], "size_mb") for b in bases]
    # Handle optional int8
    sizes_int8 = [_extra(model_groups[b].get("int8_manual", {}), "size_mb") for b in bases]
    
    tp_fp32 = [_metric(model_groups[b]["fp32"], "throughput_tokens_per_sec") for b in bases]
    tp_int8 = [_metric(model_groups[b].get("int8_manual", {}), "throughput_tokens_per_sec") for b in bases]
    
    # Size comparison
    ax1.bar([i - width/2 for i in x], sizes_fp32, width, label='FP32', color='steelblue')
    ax1.bar([i + width/2 for i in x], sizes_int8, width, label='INT8', color='coral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bases)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Size (MB)')
    ax1.set_title('Model Size: FP32 vs INT8')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Throughput comparison
    ax2.bar([i - width/2 for i in x], tp_fp32, width, label='FP32', color='steelblue')
    ax2.bar([i + width/2 for i in x], tp_int8, width, label='INT8', color='coral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bases)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Throughput (tokens/sec)')
    ax2.set_title('Throughput: FP32 vs INT8')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_all_results(all_results, output_dir):
    """Generate all plots from experiment results."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots - matplotlib not available")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    if "sequence_length" in all_results:
        plot_sequence_length_results(
            all_results["sequence_length"],
            os.path.join(output_dir, "exp1_sequence_length.png")
        )
    
    if "batch_size" in all_results:
        plot_batch_size_results(
            all_results["batch_size"],
            os.path.join(output_dir, "exp2_batch_size.png")
        )
    
    if "kv_cache" in all_results:
        plot_kv_cache_results(
            all_results["kv_cache"],
            os.path.join(output_dir, "exp3_kv_cache.png")
        )
    
    if "quantization" in all_results:
        plot_quantization_results(
            all_results["quantization"],
            os.path.join(output_dir, "exp4_quantization.png")
        )
    
    # Summary plot
    plot_summary(all_results, os.path.join(output_dir, "summary.png"))


def plot_summary(all_results, save_path=None):
    """Create a summary plot of all experiments."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hardware-Aware LLM Inference Engine - Experiment Summary', fontsize=14)
    
    # Exp 1: Sequence Length
    if "sequence_length" in all_results:
        results = all_results["sequence_length"]
        seq_lens = [_config(r, "seq_len") for r in results]
        throughputs = [_metric(r, "throughput_tokens_per_sec") for r in results]
        axes[0, 0].plot(seq_lens, throughputs, 'b-o', linewidth=2)
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('Throughput (tok/s)')
        axes[0, 0].set_title('1. Sequence Length Impact')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Exp 2: Batch Size
    if "batch_size" in all_results:
        results = all_results["batch_size"]
        batch_sizes = [_config(r, "batch_size") for r in results]
        throughputs = [_metric(r, "throughput_tokens_per_sec") for r in results]
        axes[0, 1].bar(range(len(batch_sizes)), throughputs, color='steelblue')
        axes[0, 1].set_xticks(range(len(batch_sizes)))
        axes[0, 1].set_xticklabels(batch_sizes)
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Throughput (tok/s)')
        axes[0, 1].set_title('2. Batch Size Impact')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Exp 3: KV Cache
    if "kv_cache" in all_results:
        # Extract only cached results for sequence length and speedup
        seq_to_cached = {_config(r, "seq_len"): r for r in all_results["kv_cache"] if _extra(r, "kv_cache_enabled", True)}
        seq_to_nocache = {_config(r, "seq_len"): r for r in all_results["kv_cache"] if not _extra(r, "kv_cache_enabled", True)}
        
        plot_seqs = sorted(seq_to_cached.keys())
        plot_speedups = []
        for s in plot_seqs:
            if s in seq_to_nocache:
                plot_speedups.append(_metric(seq_to_nocache[s], "total_latency_ms") / _metric(seq_to_cached[s], "total_latency_ms"))
            else:
                plot_speedups.append(0)
                
        axes[1, 0].bar(range(len(plot_seqs)), plot_speedups, color='green', alpha=0.7)
        axes[1, 0].set_xticks(range(len(plot_seqs)))
        axes[1, 0].set_xticklabels(plot_seqs)
        axes[1, 0].set_xlabel('Sequence Length')
        axes[1, 0].set_ylabel('Speedup (x)')
        axes[1, 0].set_title('3. KV Cache Speedup')
        axes[1, 0].axhline(y=1, color='r', linestyle='--')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Exp 4: Quantization
    if "quantization" in all_results:
        # Group by precision
        res = all_results["quantization"]
        precisions = ["fp32", "int8_manual"]
        # Find speedups if both exist for same model
        model_to_prec = {}
        for r in res:
            m = r["model_name"].split('_')[0]
            if m not in model_to_prec: model_to_prec[m] = {}
            model_to_prec[m][_extra(r, "precision")] = r
            
        bases = sorted(model_to_prec.keys())
        speedups = []
        for b in bases:
            if "int8_manual" in model_to_prec[b]:
                speedups.append(_metric(model_to_prec[b]["int8_manual"], "throughput_tokens_per_sec") / _metric(model_to_prec[b]["fp32"], "throughput_tokens_per_sec"))
            else:
                speedups.append(1.0)
                
        axes[1, 1].bar(range(len(bases)), speedups, color='coral')
        axes[1, 1].set_xticks(range(len(bases)))
        axes[1, 1].set_xticklabels(bases)
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Speedup (x)')
        axes[1, 1].set_title('4. INT8 vs FP32 Speedup')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def save_plots(results, output_dir):
    """Alias for plot_all_results."""
    plot_all_results(results, output_dir)


def plot_results(results, output_dir):
    """Alias for plot_all_results."""
    plot_all_results(results, output_dir)
