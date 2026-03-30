"""Text generation with KV cache."""
import torch
from ..sampling.strategies import sample_top_k


def generate(model, input_ids, kv_cache, max_new_tokens, temperature=1.0, top_k=50):
    """Generate tokens autoregressively."""
    generated = input_ids.clone()
    
    # prefill: process the prompt
    _ = model(input_ids, kv_cache)
    
    # decode: generate new tokens one by one
    for _ in range(max_new_tokens):
        x = generated[:, -1:]
        logits = model(x, kv_cache)
        next_token = sample_top_k(logits[:, -1, :], k=top_k, temperature=temperature)
        generated = torch.cat([generated, next_token], dim=1)
    
    return generated


def generate_greedy(model, input_ids, kv_cache, max_new_tokens):
    """Simple greedy generation (argmax)."""
    generated = input_ids.clone()
    
    _ = model(input_ids, kv_cache)
    
    for _ in range(max_new_tokens):
        x = generated[:, -1:]
        logits = model(x, kv_cache)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    
    return generated
