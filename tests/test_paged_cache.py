import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kv_cache import KVCacheManager, PagedKVCache


class TestPagedKVCache(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.batch_size = 2
        self.n_layers = 3
        self.n_heads = 2
        self.head_dim = 4
        self.max_seq_len = 12
        self.page_size = 4
        self.num_blocks = self.max_seq_len // self.page_size

    def _make_caches(self):
        contiguous = KVCacheManager(
            self.n_layers,
            self.n_heads,
            self.max_seq_len,
            self.head_dim,
            self.device,
            self.batch_size,
        )
        paged = PagedKVCache(
            self.n_layers,
            self.n_heads,
            self.head_dim,
            self.page_size,
            self.num_blocks,
            self.device,
            self.batch_size,
        )
        return contiguous, paged

    def test_prefill_matches_contiguous_cache(self):
        contiguous, paged = self._make_caches()
        t_new = 6

        for layer_id in range(self.n_layers):
            k_new = torch.randn(self.batch_size, self.n_heads, t_new, self.head_dim)
            v_new = torch.randn(self.batch_size, self.n_heads, t_new, self.head_dim)

            k_contig, v_contig = contiguous.get_for_attention(layer_id, k_new, v_new)
            k_paged, v_paged = paged.get_for_attention(layer_id, k_new, v_new)

            self.assertTrue(torch.allclose(k_contig, k_paged))
            self.assertTrue(torch.allclose(v_contig, v_paged))

        contiguous.curr_len += t_new
        paged.curr_len += t_new
        self.assertEqual(contiguous.curr_len, paged.curr_len)

    def test_decode_step_matches_contiguous_cache(self):
        contiguous, paged = self._make_caches()
        prefill_len = 5
        decode_len = 1

        for layer_id in range(self.n_layers):
            k_prefill = torch.randn(self.batch_size, self.n_heads, prefill_len, self.head_dim)
            v_prefill = torch.randn(self.batch_size, self.n_heads, prefill_len, self.head_dim)
            contiguous.get_for_attention(layer_id, k_prefill, v_prefill)
            paged.get_for_attention(layer_id, k_prefill, v_prefill)

        contiguous.curr_len += prefill_len
        paged.curr_len += prefill_len

        for layer_id in range(self.n_layers):
            k_decode = torch.randn(self.batch_size, self.n_heads, decode_len, self.head_dim)
            v_decode = torch.randn(self.batch_size, self.n_heads, decode_len, self.head_dim)

            k_contig, v_contig = contiguous.get_for_attention(layer_id, k_decode, v_decode)
            k_paged, v_paged = paged.get_for_attention(layer_id, k_decode, v_decode)

            self.assertTrue(torch.allclose(k_contig, k_paged))
            self.assertTrue(torch.allclose(v_contig, v_paged))
            self.assertEqual(k_paged.shape[2], prefill_len + decode_len)

        contiguous.curr_len += decode_len
        paged.curr_len += decode_len
        self.assertEqual(contiguous.curr_len, paged.curr_len)

    def test_reset_releases_pages(self):
        _, paged = self._make_caches()
        k_new = torch.randn(self.batch_size, self.n_heads, 3, self.head_dim)
        v_new = torch.randn(self.batch_size, self.n_heads, 3, self.head_dim)

        paged.get_for_attention(0, k_new, v_new)
        paged.curr_len += 3
        self.assertIsNotNone(paged.page_table[0][0])

        paged.reset()

        self.assertEqual(paged.curr_len, 0)
        self.assertTrue(all(entry is None for entry in paged.page_table[0]))
        self.assertEqual(paged.free_pages[0], list(range(self.num_blocks)))


if __name__ == "__main__":
    unittest.main()
