"""Inference Controller for managing LLM generation."""
import torch
import time
from ..profiling.latency_tracker import LatencyTracker
from ..profiling.memory_tracker import get_gpu_memory_mb, reset_gpu_memory_stats


class InferenceController:
    """Manages the generation process with optimizations and profiling."""
    
    def __init__(self, model, kv_cache, device, ablation_flags=None):
        self.model = model
        self.kv_cache = kv_cache
        self.device = device
        self.ablation_flags = ablation_flags or {}
        self.latency_tracker = LatencyTracker(device)
    
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=50):
        """Generate tokens with preallocated buffer and profiling."""
        B, T_prompt = input_ids.shape
        max_total_len = T_prompt + max_new_tokens
        
        # Preallocate output token buffer
        # This avoids torch.cat at every step
        output_tokens = torch.zeros((B, max_total_len), dtype=torch.long, device=self.device)
        output_tokens[:, :T_prompt] = input_ids
        
        token_latencies = []
        
        # Reset memory stats
        reset_gpu_memory_stats()
        start_mem = get_gpu_memory_mb()
        
        # 1. Prefill Phase
        self.latency_tracker.start_phase("prefill")
        step_start = time.perf_counter()
        
        # Prefill forward pass
        # We wrap the model forward to allow per-layer profiling if needed
        logits = self.model(input_ids, self.kv_cache, latency_tracker=self.latency_tracker)
        
        self.latency_tracker.end_phase("prefill")
        token_latencies.append(time.perf_counter() - step_start)
        
        # 2. Decode Phase
        self.latency_tracker.start_phase("decode")
        
        curr_len = T_prompt
        for i in range(max_new_tokens):
            step_start = time.perf_counter()
            
            # Use only the last token for decode step
            x = output_tokens[:, curr_len-1:curr_len]
            
            # Forward pass
            logits = self.model(x, self.kv_cache, latency_tracker=self.latency_tracker)
            
            # Sample next token (greedy for simplicity here, can be extended)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            # Store in preallocated buffer
            output_tokens[:, curr_len:curr_len+1] = next_token
            
            token_latencies.append(time.perf_counter() - step_start)
            curr_len += 1
            
            if (next_token == 0).all(): # Assume 0 is EOS for this simple example
                break
        
        self.latency_tracker.end_phase("decode")
        self.latency_tracker.synchronize()
        
        end_mem = get_gpu_memory_mb()
        
        return {
            "tokens": output_tokens[:, :curr_len],
            "latencies": token_latencies,
            "phase_times": self.latency_tracker.get_times(),
            "memory_mb": end_mem - start_mem,
            "peak_memory_mb": end_mem, # Simplified
        }

    def warmup(self, input_ids, max_new_tokens, trials=3):
        """Run warmup trials."""
        for _ in range(trials):
            self.kv_cache.reset()
            _ = self.generate(input_ids, max_new_tokens)
        self.kv_cache.reset()
