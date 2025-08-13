"""STS-B validation utilities for training."""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset


def encode_sentence_for_sts(text, tokenizer, chunk_encoder, encoder, chunk_size, device, max_chunks=64, eos_token_id=None):
    """
    Encode a single sentence into a fixed-size representation for STS evaluation.
    Uses attention masking to handle padding properly.
    """
    # Tokenize the text using tokenizers library
    encoded = tokenizer.encode(text)
    tokens = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(device)

    # Calculate how many chunks we need
    tokens_per_chunk = chunk_size - 1  # Reserve 1 for CLS token
    original_token_count = tokens.shape[1]

    # Calculate the actual number of chunks needed for this text
    n_chunks_needed = (original_token_count + tokens_per_chunk - 1) // tokens_per_chunk
    n_chunks_to_process = min(n_chunks_needed, max_chunks)

    # Truncate if needed
    if n_chunks_needed > max_chunks:
        tokens = tokens[:, :max_chunks * tokens_per_chunk]
        original_token_count = tokens.shape[1]

    # Calculate total tokens needed (must be multiple of tokens_per_chunk)
    total_tokens_needed = n_chunks_to_process * tokens_per_chunk

    # Create attention mask for original tokens
    attention_mask = torch.ones(1, original_token_count, dtype=torch.bool, device=device)

    # Pad tokens and mask if needed
    if original_token_count < total_tokens_needed:
        padding_needed = total_tokens_needed - original_token_count
        # Use EOS token for padding (provided as parameter)
        pad_token_id = eos_token_id if eos_token_id is not None else 0
        tokens = F.pad(tokens, (0, padding_needed), value=pad_token_id)
        # Extend attention mask with False for padded positions
        attention_mask = F.pad(attention_mask, (0, padding_needed), value=False)

    with torch.no_grad():
        # Process through chunk encoder WITH attention mask
        chunk_embeddings = chunk_encoder(tokens, attention_mask)  # (1, n_chunks, n_embd)

        # Create positions for all chunks
        all_positions = torch.arange(n_chunks_to_process, device=device).unsqueeze(0)

        # Encode chunks with positional information
        context_embeddings = encoder(chunk_embeddings, all_positions)  # (1, n_chunks, n_embd)

        # Mean pooling across chunks
        sentence_embedding = context_embeddings.mean(dim=1)  # (1, n_embd)

    return sentence_embedding


def compute_stsb_spearman(chunk_encoder, encoder, target_chunk_encoder, target_encoder, tokenizer, chunk_size, device, num_samples=None):
    """
    Compute Spearman correlation on STS-B validation set.
    Uses the EMA (target) encoders for more stable evaluation.
    Expects a tokenizers.Tokenizer object.
    
    Args:
        num_samples: If provided and less than dataset size, will sample a subset.
                    If None, uses the entire dataset.
    """
    chunk_encoder.eval()
    encoder.eval()
    target_chunk_encoder.eval()
    target_encoder.eval()

    # Use EMA encoders for evaluation
    chunk_enc = target_chunk_encoder
    enc = target_encoder

    # Get EOS token ID for padding
    eos_token_id = tokenizer.token_to_id('<|endoftext|>')

    # Load STS-B validation data
    dataset = load_dataset('glue', 'stsb', split='validation')

    # Only sample if num_samples is provided and less than dataset size
    if num_samples is not None and num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = dataset.select(indices)

    sentences1 = dataset['sentence1']
    sentences2 = dataset['sentence2']
    scores = np.array(dataset['label'])

    # Encode all sentence pairs
    embeddings1 = []
    embeddings2 = []

    for sent1, sent2 in zip(sentences1, sentences2):
        emb1 = encode_sentence_for_sts(sent1, tokenizer, chunk_enc, enc, chunk_size, device, eos_token_id=eos_token_id)
        emb2 = encode_sentence_for_sts(sent2, tokenizer, chunk_enc, enc, chunk_size, device, eos_token_id=eos_token_id)
        embeddings1.append(emb1.cpu())
        embeddings2.append(emb2.cpu())

    # Stack embeddings
    embeddings1 = torch.cat(embeddings1, dim=0).numpy()
    embeddings2 = torch.cat(embeddings2, dim=0).numpy()

    # Compute cosine similarities
    embeddings1_norm = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
    embeddings2_norm = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
    cosine_sims = np.sum(embeddings1_norm * embeddings2_norm, axis=1)

    # Scale to [0, 5] range
    predicted_scores = (cosine_sims + 1) * 2.5

    # Compute Spearman correlation
    spearman_corr = spearmanr(scores, predicted_scores)[0]

    chunk_encoder.train()
    encoder.train()
    target_chunk_encoder.train()
    target_encoder.train()

    return spearman_corr