import glob

import numpy as np
import torch

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return int(ntok) # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

def create_random_chunk_mask(batch_size: int, n_chunks: int, mask_ratio: float = 0.15,
                             device: torch.device = None) -> torch.BoolTensor:
    """
    Create random chunk masks for JEPA training with optimized vectorized operations.

    Args:
        batch_size: Number of samples in batch
        n_chunks: Number of chunks per sample
        mask_ratio: Fraction of chunks to mask (default 0.15)
        device: Device to create tensor on

    Returns:
        Boolean mask of shape (batch_size, n_chunks) where True = masked
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create base mask
    mask = torch.rand(batch_size, n_chunks, device=device) < mask_ratio
    
    # Vectorized correction to ensure at least one visible and one masked
    if n_chunks > 1:
        # Find samples that need correction
        all_masked = mask.all(dim=1)
        none_masked = ~mask.any(dim=1)
        
        # Fix all-masked samples (unmask one random position)
        if all_masked.any():
            unmask_indices = torch.randint(0, n_chunks, (all_masked.sum(),), device=device)
            mask[all_masked, unmask_indices] = False
        
        # Fix none-masked samples (mask one random position)
        if none_masked.any():
            mask_indices = torch.randint(0, n_chunks, (none_masked.sum(),), device=device)
            mask[none_masked, mask_indices] = True

    return mask


class DataLoader:
    def __init__(self, filename_pattern, B, chunk_size, n_chunks, mask_ratio=0.15, device='cuda'):
        """
        Args:
            filename_pattern: Pattern to match data files
            B: Batch size
            chunk_size: Size of each chunk (including CLS token)
            n_chunks: Number of chunks per sample
            mask_ratio: Fraction of chunks to mask for JEPA training
            device: Device to load data to ('cuda' or 'cpu')
        """
        self.B = B
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.mask_ratio = mask_ratio
        self.device = torch.device(device) if isinstance(device, str) else device

        # Calculate total tokens needed per sample
        # Each chunk needs (chunk_size - 1) tokens since CLS is added
        self.tokens_per_chunk = chunk_size - 1
        self.T = n_chunks * self.tokens_per_chunk  # Total tokens per sample

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")
        print(f"DataLoader: chunk_size={chunk_size}, n_chunks={n_chunks}, tokens_per_sample={self.T}")
        
        # Pre-allocate CPU staging buffer for efficient GPU transfer
        self.cpu_staging_buffer = torch.empty(B * self.T, dtype=torch.long, pin_memory=True)
        
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        """
        Returns:
            dict with:
                - tokens: (B, T) token indices for ChunkEncoder
                - chunk_mask: (B, n_chunks) boolean mask for Encoder
                - target_positions: (B, n_target) positions to predict
                - n_chunks: number of chunks
        """
        B = self.B
        T = self.T

        # Get tokens from current position
        buf = self.tokens[self.current_position : self.current_position+B*T]
        
        # Use pre-allocated CPU buffer with pinned memory for faster GPU transfer
        self.cpu_staging_buffer[:len(buf)] = torch.from_numpy(buf.astype(np.int32))
        
        # Efficient GPU transfer with non-blocking
        tokens = self.cpu_staging_buffer[:len(buf)].to(self.device, non_blocking=True)
        tokens = tokens.view(B, T)

        # Create chunk mask using optimized function
        chunk_mask = create_random_chunk_mask(B, self.n_chunks, self.mask_ratio, self.device)

        # Optimized target position extraction
        n_masked_per_batch = chunk_mask.sum(dim=1)
        max_targets = n_masked_per_batch.max().item()
        
        # Pre-allocate and fill target positions efficiently
        target_positions = torch.zeros(B, max_targets, dtype=torch.long, device=self.device)
        for b in range(B):
            n_masked = n_masked_per_batch[b].item()
            if n_masked > 0:
                # Get masked positions efficiently using nonzero
                masked_indices = chunk_mask[b].nonzero(as_tuple=False).squeeze(-1)
                target_positions[b, :n_masked] = masked_indices[:n_masked]

        # Advance position
        self.current_position += B * T
        if self.current_position + (B * T) > len(self.tokens):
            self.advance()

        return {
            'tokens': tokens,
            'chunk_mask': chunk_mask,
            'target_positions': target_positions,
            'n_chunks': self.n_chunks
        }