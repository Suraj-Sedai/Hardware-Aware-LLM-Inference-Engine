import torch
import torch.nn as nn
from .embeddings import TokenEmbedding
from .transformer_block import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        #embedding layer
        self.embedding = TokenEmbedding(config)

        #transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.d_model,
                num_heads=config.n_heads,
                ff_hidden_size=config.d_ff,
                dropout_prob=config.dropout_prob
            )
            for _ in range(config.n_layers)
        ])

        #final linear layer to project hidden states to logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, kv_cache= None, use_cache=False):
        B,T = input_ids.shape

        #embbeding layer
        x = self.embedding(input_ids)

        #pass though transformer blocks
        new_kv_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            cache = kv_cache[i] if (kv_cache is not None and use_cache) else None
            x = layer(x, cache=cache)

            if use_cache:
                new_kv_cache.append(cache)

        #project to logits
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_kv_cache
        return logits