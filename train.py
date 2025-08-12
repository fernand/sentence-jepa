import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch.nn.functional as F

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                momentum_buffer = state['momentum_buffer']
                p.mul_(1 - group['weight_decay'])
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)
                v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                p.add_(other=v, alpha=-group['lr'])

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, chunk_size=None):
        seq_len = x.shape[1]

        # If chunk_size is provided, create position indices that reset per chunk
        if chunk_size is not None:
            device = x.device
            n_chunks = seq_len // chunk_size
            # Create position indices that reset for each chunk (0, 1, ..., chunk_size-1, 0, 1, ...)
            t = torch.arange(chunk_size, device=device).repeat(n_chunks).type_as(self.inv_freq)
        else:
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq).to(x.device)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps:float=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class ChunkedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.chunk_size = config.chunk_size
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        chunk_size = self.chunk_size

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # Apply rotary embeddings (with positions resetting per chunk)
        cos, sin = self.rotary(q, chunk_size=chunk_size)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Transpose for attention: (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        def make_chunk_mask(b, h, q_idx, kv_idx):
            # Each position can only attend within its chunk
            q_chunk = q_idx // chunk_size
            kv_chunk = kv_idx // chunk_size
            return q_chunk == kv_chunk

        block_mask = create_block_mask(
            make_chunk_mask,
            B=B,
            H=self.n_head,
            Q_LEN=T,
            KV_LEN=T,
            device=x.device,
            _compile=True
        )

        y = flex_attention(q, k, v, block_mask=block_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wup = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.wdown = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.wup(x)
        x = F.relu(x).square()
        x = self.wdown(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = ChunkedSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / math.sqrt(2 * config.n_layer))

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

@dataclass
class EncoderConfig:
    vocab_size: int
    chunk_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.n_embd))
        self.cls_pos_embedding = nn.Parameter(torch.zeros(1, 1, config.n_embd))

    def forward(self, idx: torch.LongTensor):
        """
        Args:
            idx: Token indices of shape (B, k * (chunk_size - 1))
                 where k is the number of chunks
        Returns:
            chunk_embeddings: Tensor of shape (B, k, n_embd)
                             containing the CLS token embedding for each chunk
        """
        B = idx.shape[0]
        chunk_size = self.config.chunk_size

        tokens_per_chunk = chunk_size - 1
        total_tokens = idx.shape[1]
        n_chunks = total_tokens // tokens_per_chunk
        assert total_tokens % tokens_per_chunk == 0, \
            f"Input length {total_tokens} must be divisible by {tokens_per_chunk}"

        # Embed input tokens
        x = self.wte(idx)  # (B, k*(chunk_size-1), n_embd)
        # Reshape to separate chunks
        x = x.view(B, n_chunks, tokens_per_chunk, self.config.n_embd)
        # Prepare CLS tokens for each chunk
        cls_tokens = self.cls_token.expand(B, n_chunks, 1, self.config.n_embd)
        cls_tokens = cls_tokens + self.cls_pos_embedding
        x = torch.cat([cls_tokens, x], dim=2)  # (B, n_chunks, chunk_size, n_embd)
        x = x.view(B, n_chunks * chunk_size, self.config.n_embd)
        for block in self.transformer:
            x = block(x)
        x = rmsnorm(x)

        # Reshape to chunks and extract CLS tokens
        x = x.view(B, n_chunks, chunk_size, self.config.n_embd)
        chunk_embeddings = x[:, :, 0, :]  # Extract first token (CLS) of each chunk

        return chunk_embeddings  # (B, k, n_embd)

    def configure_optimizers(self, wd, adam_lr, adam_betas):
        return [
            Muon(self.transformer.parameters(), lr=10*adam_lr, weight_decay=0, momentum=0.95),
            torch.optim.AdamW(self.wte.parameters(), lr=adam_lr, weight_decay=wd, betas=adam_betas)
        ]

@dataclass
class PredictorConfig:
    pass
