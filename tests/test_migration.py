import torch
import unittest
from src.model_core import GPT
from src.kv_cache import KVCacheManager
from src.inference.controller import InferenceController
from src.profiling.metrics import calculate_metrics

class TestInferenceMigration(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.vocab_size = 100
        self.dim = 32
        self.n_heads = 4
        self.n_layers = 2
        self.max_seq_len = 128
        self.batch_size = 1
        
        self.model = GPT(self.vocab_size, self.dim, self.n_heads, self.n_layers, self.max_seq_len).to(self.device)
        self.model.eval()
        
    def test_controller_generation_shape(self):
        """Verify output tokens shape and preallocated buffer logic."""
        prompt_len = 8
        max_new_tokens = 10
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, prompt_len), device=self.device)
        
        kv_cache = KVCacheManager(self.n_layers, self.n_heads, self.max_seq_len, self.dim // self.n_heads, self.device, self.batch_size)
        controller = InferenceController(self.model, kv_cache, self.device)
        
        result = controller.generate(input_ids, max_new_tokens)
        tokens = result["tokens"]
        
        # Expected shape: (batch_size, prompt_len + actual_new_tokens)
        # It might be less than max_new_tokens if EOS is hit, but we don't have EOS logic here really
        self.assertEqual(tokens.shape, (self.batch_size, prompt_len + max_new_tokens))
        
    def test_metrics_consistency(self):
        """Verify TTFT, TPOT and throughput calculations."""
        latencies = [0.1, 0.05, 0.05, 0.05] # 4 tokens
        total_tokens = 4
        total_time = sum(latencies)
        
        metrics = calculate_metrics(latencies, total_tokens, total_time)
        
        self.assertEqual(metrics["ttft_ms"], 100.0)
        self.assertAlmostEqual(metrics["tpot_avg_ms"], 50.0)
        self.assertAlmostEqual(metrics["throughput_tokens_per_sec"], 4 / 0.25)
        
    def test_cache_reset(self):
        """Verify KV cache reset behavior."""
        kv_cache = KVCacheManager(self.n_layers, self.n_heads, self.max_seq_len, self.dim // self.n_heads, self.device, self.batch_size)
        
        # Fill cache a bit
        kv_cache.K[0][:, :, :5, :] = 1.0
        kv_cache.curr_len = 5
        
        kv_cache.reset()
        self.assertEqual(kv_cache.curr_len, 0)
        
    def test_peak_memory_reporting(self):
        """Verify peak memory is reported and >= 0."""
        prompt_len = 4
        max_new_tokens = 2
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, prompt_len), device=self.device)
        
        kv_cache = KVCacheManager(self.n_layers, self.n_heads, self.max_seq_len, self.dim // self.n_heads, self.device, self.batch_size)
        controller = InferenceController(self.model, kv_cache, self.device)
        
        result = controller.generate(input_ids, max_new_tokens)
        self.assertIn("peak_memory_mb", result)
        self.assertGreaterEqual(result["peak_memory_mb"], 0)

if __name__ == "__main__":
    unittest.main()
