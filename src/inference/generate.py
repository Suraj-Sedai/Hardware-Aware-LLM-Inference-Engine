"""Text generation with KV cache using InferenceController.

This module now delegates to InferenceController which uses preallocated
buffer logic for efficient generation. The old torch.cat approach has been
replaced with the controller's optimized implementation.
"""
from ..runtime.controller import InferenceController


def generate(model, input_ids, kv_cache, max_new_tokens, temperature=1.0, top_k=50, device="cuda"):
    """Generate tokens autoregressively using InferenceController.
    
    Args:
        model: The language model
        input_ids: Input token IDs (B, T)
        kv_cache: KV cache manager
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (currently unused, greedy only)
        top_k: Top-k sampling parameter (currently unused, greedy only)
        device: Device to run on
        
    Returns:
        Generated tokens (B, T + max_new_tokens)
    """
    controller = InferenceController(model, kv_cache, device)
    result = controller.generate(input_ids, max_new_tokens, temperature=temperature, top_k=top_k)
    return result["tokens"]


def generate_greedy(model, input_ids, kv_cache, max_new_tokens, device="cuda"):
    """Simple greedy generation using InferenceController.
    
    Args:
        model: The language model
        input_ids: Input token IDs (B, T)
        kv_cache: KV cache manager
        max_new_tokens: Number of tokens to generate
        device: Device to run on
        
    Returns:
        Generated tokens (B, T + max_new_tokens)
    """
    controller = InferenceController(model, kv_cache, device)
    result = controller.generate(input_ids, max_new_tokens, temperature=1.0, top_k=1)
    return result["tokens"]
