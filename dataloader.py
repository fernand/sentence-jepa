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
    Create random chunk masks for JEPA training.

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
    mask = torch.rand(batch_size, n_chunks, device=device) < mask_ratio
    # Ensure at least one chunk is visible and one is masked per sample (if possible)
    for i in range(batch_size):
        if mask[i].all() and n_chunks > 1:  # All masked, unmask one
            unmask_idx = torch.randint(0, n_chunks, (1,), device=device)
            mask[i, unmask_idx] = False
        elif not mask[i].any() and n_chunks > 1:  # None masked, mask one
            mask_idx = torch.randint(0, n_chunks, (1,), device=device)
            mask[i, mask_idx] = True

    return mask


class DataLoader:
    def __init__(self, filename_pattern, B, T):
        self.B = B
        self.T = T

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")
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
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf.view(B, T) # inputs
        self.current_position += B * T
        if self.current_position + (B * T) > len(self.tokens):
            self.advance()
        return x.cuda()
