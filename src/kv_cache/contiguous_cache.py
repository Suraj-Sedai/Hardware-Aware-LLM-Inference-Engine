import torch

class ContiguousKVCache:
    def __init__(self, config, batch_size, max_seq_len, device, dtype):
        """
        Initialize the KV cache with preallocated memory.
        """
        self.num_layers = config.num_layers  # Number of layers (L)
        self.batch_size = batch_size  # Batch size (B)
        self.num_heads = config.num_heads  # Number of heads (H)
        self.max_seq_len = max_seq_len  # Maximum sequence length (MaxSeq)
        self.head_dim = config.head_dim  # Head dimension (Hd)
        self.device = device  # Device (e.g., 'cuda' or 'cpu')
        self.dtype = dtype  # Data type (e.g., torch.float32)

        # Preallocate memory for keys and values
        self.keys = torch.zeros(
            (self.num_layers, self.batch_size, self.num_heads, self.max_seq_len, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        self.values = torch.zeros(
            (self.num_layers, self.batch_size, self.num_heads, self.max_seq_len, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )

        # Initialize sequence length
        self.seq_len = 0

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> None:
        """
        Write new keys and values to the cache at the specified layer and position.
        Args:
            layer_idx (int): Layer index (0 <= layer_idx < L).
            k (torch.Tensor): Keys tensor of shape [B, H, T, Hd].
            v (torch.Tensor): Values tensor of shape [B, H, T, Hd].
            start_pos (int): Starting position in the sequence.
        """
        T = k.size(2)  # Sequence length of the new data (T)
        self.keys[layer_idx, :, :, start_pos:start_pos + T, :] = k
        self.values[layer_idx, :, :, start_pos:start_pos + T, :] = v

    def read(self, layer_idx: int, upto_pos: int):
        """
        Read the active slice of keys and values for the specified layer.
        Args:
            layer_idx (int): Layer index (0 <= layer_idx < L).
            upto_pos (int): Sequence length to read up to.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Slices of keys and values.
        """
        return (
            self.keys[layer_idx, :, :, :upto_pos, :],
            self.values[layer_idx, :, :, :upto_pos, :]
        )

    def advance(self, amount: int) -> None:
        """
        Advance the global sequence length by the specified amount.
        Args:
            amount (int): Number of tokens to advance.
        """
        self.seq_len += amount

    def reset(self) -> None:
        """
        Reset the cache by clearing all data.
        """
        self.seq_len = 0
        self.keys.zero_()
        self.values.zero_()

    @property
    def current_seq_len(self) -> int:
        """
        Get the current sequence length.
        Returns:
            int: Current sequence length.
        """
        return self.seq_len