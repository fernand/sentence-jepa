import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch.nn.functional as F

class ChunkedRotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.chunk_size = max_seq_len
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        B, T, H, D = x_BTHD.shape
        # For chunked mode, repeat the same positional encoding for each chunk
        assert T % self.chunk_size == 0, f"Sequence length {T} must be divisible by chunk_size {self.chunk_size}"
        n_chunks = T // self.chunk_size
        # Get positional encodings for one chunk
        cos_chunk = self.cos[None, :self.chunk_size, None, :]  # (1, chunk_size, 1, D)
        sin_chunk = self.sin[None, :self.chunk_size, None, :]  # (1, chunk_size, 1, D)
        # Repeat for all chunks
        cos = cos_chunk.repeat(1, n_chunks, 1, 1)  # (1, T, 1, D)
        sin = sin_chunk.repeat(1, n_chunks, 1, 1)  # (1, T, 1, D)
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

def norm(x: torch.Tensor):
    return F.rms_norm(x, (x.size(-1),))

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
        self.rotary = ChunkedRotary(self.head_dim, self.chunk_size, chunked=True)

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
        # QK norm
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
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

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        # QK norm
        q, k = norm(q), norm(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_kv = nn.Linear(self.n_embd, 2 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, queries, context):
        """
        Args:
            queries: Tensor of shape (B, T_q, C) - positions to predict
            context: Tensor of shape (B, T_c, C) - context chunks
        Returns:
            Tensor of shape (B, T_q, C)
        """
        B, T_q, C = queries.size()
        _, T_c, _ = context.size()
        # Compute queries from prediction positions
        q = self.c_q(queries).view(B, T_q, self.n_head, self.head_dim)
        # Compute keys and values from context
        kv = self.c_kv(context)
        k, v = kv.split(self.n_embd, dim=2)
        k = k.view(B, T_c, self.n_head, self.head_dim)
        v = v.view(B, T_c, self.n_head, self.head_dim)
        # QK norm
        q, k = norm(q), norm(k)
        # Compute attention
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
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
    def __init__(self, config, chunked: bool = False):
        super().__init__()
        self.attn = ChunkedSelfAttention(config) if chunked else SelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / math.sqrt(2 * config.n_layer))

    def forward(self, x):
        x = x + self.attn_scale * self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x

class PredictorBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = SelfAttention(config)
        self.cross_attn = CrossAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / math.sqrt(2 * config.n_layer))

    def forward(self, x, context):
        # Self-attention among predictions
        x = x + self.attn_scale * self.self_attn(norm(x))
        # Cross-attention to context
        x = x + self.attn_scale * self.cross_attn(norm(x), context)
        # MLP
        x = x + self.mlp(norm(x))
        return x

@dataclass
class ChunkEncoderConfig:
    vocab_size: int
    chunk_size: int
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768

class ChunkEncoder(nn.Module):
    """Encodes token sequences into chunk-level embeddings."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer = nn.ModuleList([Block(config, chunked=True) for _ in range(config.n_layer)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.n_embd))

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
        x = torch.cat([cls_tokens, x], dim=2)  # (B, n_chunks, chunk_size, n_embd)
        x = x.view(B, n_chunks * chunk_size, self.config.n_embd)
        for block in self.transformer:
            x = block(x)
        x = norm(x)

        # Reshape to chunks and extract CLS tokens
        x = x.view(B, n_chunks, chunk_size, self.config.n_embd)
        chunk_embeddings = x[:, :, 0, :]  # Extract first token (CLS) of each chunk

        return chunk_embeddings  # (B, k, n_embd)

    def configure_optimizers(self, wd, adam_lr, adam_betas):
        return torch.optim.AdamW(self.parameters(), lr=adam_lr, weight_decay=wd, betas=adam_betas)

@dataclass
class EncoderConfig:
    max_chunks: int = 32
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class Encoder(nn.Module):
    """Processes chunk embeddings (with optional masking) to produce contextual representations."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.chunk_pos_embedding = nn.Parameter(torch.zeros(1, config.max_chunks, config.n_embd))
        self.blocks = nn.ModuleList([Block(config, chunked=False) for _ in range(config.n_layer)])

    def forward(self, chunk_embeddings: torch.Tensor, chunk_positions: torch.LongTensor = None):
        """
        Args:
            chunk_embeddings: Tensor of shape (B, k, n_embd) containing chunk embeddings
            chunk_positions: Optional tensor of shape (B, k) containing original chunk positions
        Returns:
            Tensor of shape (B, k, n_embd) containing contextualized chunk representations
        """
        B, k, D = chunk_embeddings.shape
        
        if chunk_positions is not None:
            # Use provided positions to gather positional embeddings
            pos_embeddings = self.chunk_pos_embedding.expand(B, -1, -1)  # (B, max_chunks, n_embd)
            # Gather positional embeddings for the specified positions
            x = chunk_embeddings + torch.gather(pos_embeddings, 1, 
                                               chunk_positions.unsqueeze(-1).expand(-1, -1, D))
        else:
            # Fall back to sequential positions (backward compatibility)
            x = chunk_embeddings + self.chunk_pos_embedding[:, :k, :]
        
        for block in self.blocks:
            x = block(x)
        x = norm(x)
        return x

    def configure_optimizers(self, wd, adam_lr, adam_betas):
        return torch.optim.AdamW(self.parameters(), lr=adam_lr, weight_decay=wd, betas=adam_betas)


@dataclass
class PredictorConfig:
    max_chunks: int = 32
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    n_encoder_embd: int = 768

class Predictor(nn.Module):
    """Predicts masked chunk representations from context chunks."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_encoder_embd = config.n_encoder_embd
        self.context_proj = nn.Linear(config.n_encoder_embd, config.n_embd, bias=False)
        self.position_queries = nn.Parameter(torch.randn(1, config.max_chunks, config.n_embd))
        self.target_pos_embedding = nn.Parameter(torch.zeros(1, config.max_chunks, config.n_embd))
        self.context_pos_embedding = nn.Parameter(torch.zeros(1, config.max_chunks, config.n_embd))
        self.blocks = nn.ModuleList([PredictorBlock(config) for _ in range(config.n_layer)])
        self.output_proj = nn.Linear(config.n_embd, config.n_encoder_embd, bias=False)

    def forward(self, context_embeddings: torch.Tensor, target_positions: torch.LongTensor, 
                context_positions: torch.LongTensor = None):
        """
        Args:
            context_embeddings: Tensor of shape (B, n_context, n_encoder_embd) - visible chunk embeddings
            target_positions: Tensor of shape (B, n_target) - positions of chunks to predict
            context_positions: Optional tensor of shape (B, n_context) - original positions of context chunks

        Returns:
            Tensor of shape (B, n_target, n_encoder_embd) - predicted embeddings for masked positions
        """
        B, n_context, D_enc = context_embeddings.shape
        n_target = target_positions.shape[1]
        # Project context embeddings to predictor dimension
        context_embeddings = self.context_proj(context_embeddings)  # (B, n_context, n_embd)
        
        # Add positional embeddings to context if positions are provided
        if context_positions is not None:
            context_pos_emb = self.context_pos_embedding.expand(B, -1, -1)  # (B, max_chunks, n_embd)
            # Gather positional embeddings for context positions
            context_embeddings = context_embeddings + torch.gather(
                context_pos_emb, 1, 
                context_positions.unsqueeze(-1).expand(-1, -1, self.n_embd)
            )
        
        # Get position queries for target positions with positional embeddings
        queries = self.position_queries.expand(B, -1, -1)  # (B, max_chunks, n_embd)
        target_pos_emb = self.target_pos_embedding.expand(B, -1, -1)  # (B, max_chunks, n_embd)
        
        # Gather both position queries and positional embeddings for target positions
        target_queries = torch.gather(queries, 1,
                                     target_positions.unsqueeze(-1).expand(-1, -1, self.n_embd))  # (B, n_target, n_embd)
        target_pos = torch.gather(target_pos_emb, 1,
                                 target_positions.unsqueeze(-1).expand(-1, -1, self.n_embd))  # (B, n_target, n_embd)
        
        # Combine position queries with positional embeddings
        x = target_queries + target_pos
        
        for block in self.blocks:
            x = block(x, context_embeddings)
        x = self.output_proj(norm(x))
        return x

    def configure_optimizers(self, wd, adam_lr, adam_betas):
        return torch.optim.AdamW(self.parameters(), lr=adam_lr, weight_decay=wd, betas=adam_betas)
