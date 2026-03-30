"""Sampling strategies for text generation."""
import torch


def sample_top_k(logits, k=50, temperature=1.0):
    """Sample from top-k logits with temperature."""
    logits = logits / temperature
    k = min(k, logits.size(-1))
    top_k_vals, top_k_idx = torch.topk(logits, k)
    probs = torch.softmax(top_k_vals, dim=-1)
    idx = torch.multinomial(probs, 1)
    return top_k_idx.gather(-1, idx)


def sample_greedy(logits):
    """Greedy sampling (argmax)."""
    return torch.argmax(logits, dim=-1, keepdim=True)


def sample_top_p(logits, p=0.9, temperature=1.0):
    """Nucleus (top-p) sampling."""
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # remove tokens with cumulative prob above threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float("-inf")
    
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)
