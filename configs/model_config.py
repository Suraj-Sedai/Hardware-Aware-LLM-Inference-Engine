"""Model configuration."""

# Default GPT config (small for testing)
DEFAULT_CONFIG = {
    "vocab_size": 100,
    "dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "max_seq_len": 128,
}

def get_config(size="small"):
    """Get model config by size."""
    configs = {
        "small": DEFAULT_CONFIG,
        "medium": {**DEFAULT_CONFIG, "dim": 64, "n_heads": 8, "n_layers": 4},
        "large": {**DEFAULT_CONFIG, "dim": 128, "n_heads": 8, "n_layers": 6},
    }
    return configs.get(size, DEFAULT_CONFIG)
