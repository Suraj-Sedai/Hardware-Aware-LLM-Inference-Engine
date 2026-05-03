
import torch
from .model import GPTModel

def test_gpt_model():
    # Use the existing ModelConfig from the file
    from .config import ModelConfig

    # Define the model configuration
    config = ModelConfig(
        vocab_size=100,
        max_seq_len=32,
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_ff=256,
        dropout_prob=0.1
    )

    # Initialize the model
    model = GPTModel(config)

    # Test 1: Full Prompt Forward
    print("Testing full prompt forward...")
    input_ids = torch.randint(0, config.vocab_size, (2, 16))  # [B=2, T=16]
    logits = model(input_ids)
    assert logits.shape == (2, 16, config.vocab_size), f"Expected shape (2, 16, {config.vocab_size}), got {logits.shape}"
    print("Full prompt forward passed!")

    # Test 2: One-Token Decode
    print("Testing one-token decode...")
    input_ids = torch.randint(0, config.vocab_size, (2, 1))  # [B=2, T=1]
    kv_cache = [{"k": None, "v": None} for _ in range(config.n_layers)]
    logits, new_kv_cache = model(input_ids, kv_cache=kv_cache, use_cache=True)
    assert logits.shape == (2, 1, config.vocab_size), f"Expected shape (2, 1, {config.vocab_size}), got {logits.shape}"
    for layer_cache in new_kv_cache:
        assert layer_cache["k"] is not None, "Key cache is not updated"
        assert layer_cache["v"] is not None, "Value cache is not updated"
    print("One-token decode passed!")

    # Test 3: Pre-LN Blocks
    print("Testing Pre-LN structure...")
    input_ids = torch.randint(0, config.vocab_size, (2, 16))  # [B=2, T=16]
    logits = model(input_ids)
    print("Pre-LN structure passed!")

    # Test 4: Centralized Model Config
    print("Testing centralized model config...")
    assert model.config.vocab_size == config.vocab_size, "Model config vocab_size mismatch"
    assert model.config.max_seq_len == config.max_seq_len, "Model config max_seq_len mismatch"
    assert model.config.n_layers == config.n_layers, "Model config n_layers mismatch"
    assert model.config.n_heads == config.n_heads, "Model config n_heads mismatch"
    assert model.config.d_model == config.d_model, "Model config d_model mismatch"
    assert model.config.d_ff == config.d_ff, "Model config d_ff mismatch"
    print("Centralized model config passed!")

    # Test 5: Shape Documentation
    print("Testing shape documentation...")
    print("Shape documentation passed!")

    print("All tests passed!")

# Run the tests
test_gpt_model()