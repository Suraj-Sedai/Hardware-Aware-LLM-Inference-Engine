"""Chunked prefill implementation."""


class ChunkedPrefill:
    """Splits large prompts into smaller chunks for prefill."""
    
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size
    
    def get_chunks(self, input_ids):
        T = input_ids.shape[1]
        chunks = []
        for i in range(0, T, self.chunk_size):
            chunks.append(input_ids[:, i:i+self.chunk_size])
        return chunks
