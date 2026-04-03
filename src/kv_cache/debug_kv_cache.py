"""Debug script to isolate KV cache performance issue."""
import torch
import time

from src.model_core import GPT
from src.kv_cache import KVCacheManager

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Model config - LARGER model to actually stress the GPU
vocab, dim, heads, layers, max_seq = 50257, 1024, 16, 24, 1024
model = GPT(vocab, dim, heads, layers, max_seq).to(device).eval()

# Calculate model size
param_count = sum(p.numel() for p in model.parameters())
print(f"Model: dim={dim}, heads={heads}, layers={layers}")
print(f"Parameters: {param_count/1e6:.1f}M ({param_count*4/1e9:.2f} GB)")
print("-" * 50)

# Warmup
with torch.no_grad():
    _ = model(torch.randint(0, vocab, (1, 10), device=device), None)
torch.cuda.synchronize() if device == 'cuda' else None

# =============================================================
# TEST 1: Batch forward (what no-cache does per step)
# =============================================================
seq_lens = [32, 64, 128, 256]
print("\n[TEST 1] Batch forward pass timing (no cache simulation):")
for seq_len in seq_lens:
    x = torch.randint(0, vocab, (1, seq_len), device=device)
    torch.cuda.synchronize() if device == 'cuda' else None
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x, None)
    torch.cuda.synchronize() if device == 'cuda' else None
    
    elapsed = time.perf_counter() - start
    print(f"  seq_len={seq_len:3d}: {elapsed:.3f}s for 100 iters ({elapsed/100*1000:.2f}ms/iter)")

# =============================================================
# TEST 2: Single token with cache (what cached decode does)
# =============================================================
print("\n[TEST 2] Single token + cache timing (cached decode simulation):")
for cache_len in [31, 63, 127, 255]:
    # Setup cache with prefilled tokens
    cache = KVCacheManager(layers, heads, max_seq, dim//heads, device, 1)
    prefill = torch.randint(0, vocab, (1, cache_len), device=device)
    with torch.no_grad():
        _ = model(prefill, cache)
    
    x_one = torch.randint(0, vocab, (1, 1), device=device)
    torch.cuda.synchronize() if device == 'cuda' else None
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x_one, cache)
            cache.curr_len -= 1  # Reset to keep cache size constant
    torch.cuda.synchronize() if device == 'cuda' else None
    
    elapsed = time.perf_counter() - start
    print(f"  cache_len={cache_len:3d}: {elapsed:.3f}s for 100 iters ({elapsed/100*1000:.2f}ms/iter)")

# =============================================================
# TEST 3: Realistic generation comparison
# =============================================================
print("\n[TEST 3] Realistic generation (50 tokens):")

# With cache
cache = KVCacheManager(layers, heads, max_seq, dim//heads, device, 1)
prompt = torch.randint(0, vocab, (1, 16), device=device)

with torch.no_grad():
    logits = model(prompt, cache)  # Prefill
torch.cuda.synchronize() if device == 'cuda' else None

start = time.perf_counter()
with torch.no_grad():
    next_tok = torch.argmax(logits[:, -1:], dim=-1)
    for _ in range(50):
        logits = model(next_tok, cache)
        next_tok = torch.argmax(logits[:, -1:], dim=-1)
torch.cuda.synchronize() if device == 'cuda' else None
cached_time = time.perf_counter() - start
print(f"  WITH cache:    {cached_time:.4f}s")

# Without cache
generated = prompt.clone()
torch.cuda.synchronize() if device == 'cuda' else None

start = time.perf_counter()
with torch.no_grad():
    for _ in range(50):
        logits = model(generated, None)
        next_tok = torch.argmax(logits[:, -1:], dim=-1)
        generated = torch.cat([generated, next_tok], dim=1)
torch.cuda.synchronize() if device == 'cuda' else None
no_cache_time = time.perf_counter() - start
print(f"  WITHOUT cache: {no_cache_time:.4f}s")
print(f"  Speedup:       {no_cache_time/cached_time:.2f}x")

# =============================================================
# TEST 4: Check if cache is actually being used
# =============================================================
print("\n[TEST 4] Cache state verification:")
cache = KVCacheManager(layers, heads, max_seq, dim//heads, device, 1)
x = torch.randint(0, vocab, (1, 5), device=device)

with torch.no_grad():
    _ = model(x, cache)
print(f"  After 5-token prefill: cache.curr_len = {cache.curr_len}")

with torch.no_grad():
    _ = model(x[:, :1], cache)
print(f"  After 1-token decode:  cache.curr_len = {cache.curr_len}")

# Check K values are actually stored
k_layer0 = cache.K[0][:, :, :cache.curr_len, :]
print(f"  K[0] shape in cache: {k_layer0.shape}")
print(f"  K[0] non-zero: {(k_layer0.abs() > 1e-6).sum().item()} / {k_layer0.numel()} elements")
