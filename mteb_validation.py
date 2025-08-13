"""MTEB validation utilities with caching disabled for training."""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
import mteb
from typing import Optional
from mteb.encoder_interface import PromptType
import os
import shutil


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


def encode_batch_for_sts(texts, tokenizer, chunk_encoder, encoder, chunk_size, device, max_chunks=64, eos_token_id=None):
    """
    Encode a batch of sentences into fixed-size representations.
    Process each text individually to handle variable lengths properly.
    """
    if not texts:
        return torch.empty(0, encoder.config.n_embd)

    all_embeddings = []

    # Process each text individually to handle variable lengths
    for text in texts:
        embedding = encode_sentence_for_sts(text, tokenizer, chunk_encoder, encoder, chunk_size, device, max_chunks, eos_token_id)
        all_embeddings.append(embedding)

    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, encoder.config.n_embd)


def compute_stsb_spearman(chunk_encoder, encoder, target_chunk_encoder, target_encoder, tokenizer, chunk_size, device, num_samples=None, batch_size=64):
    """
    Compute Spearman correlation on STS-B validation set.
    Uses the EMA (target) encoders for more stable evaluation.
    Expects a tokenizers.Tokenizer object.

    Args:
        num_samples: If provided and less than dataset size, will sample a subset.
                    If None, uses the entire dataset.
        batch_size: Number of sentences to process at once (default: 64)
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

    # Encode all sentence pairs in batches
    embeddings1 = []
    embeddings2 = []

    # Process in batches
    for i in range(0, len(sentences1), batch_size):
        batch_sent1 = sentences1[i:i+batch_size]
        batch_sent2 = sentences2[i:i+batch_size]

        batch_emb1 = encode_batch_for_sts(batch_sent1, tokenizer, chunk_enc, enc, chunk_size, device, eos_token_id=eos_token_id)
        batch_emb2 = encode_batch_for_sts(batch_sent2, tokenizer, chunk_enc, enc, chunk_size, device, eos_token_id=eos_token_id)

        embeddings1.append(batch_emb1.cpu())
        embeddings2.append(batch_emb2.cpu())

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


class JEPAModelForMTEB:
    """
    Wrapper class for JEPA model to work with MTEB evaluation framework.
    Uses the target (EMA) encoders for evaluation.
    """

    def __init__(
        self,
        target_chunk_encoder,
        target_encoder,
        tokenizer,
        chunk_size,
        device='cuda',
        batch_size=32,
        max_chunks=64,
    ):
        self.chunk_encoder = target_chunk_encoder.to(device).eval()
        self.encoder = target_encoder.to(device).eval()
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.device = device
        self.batch_size = batch_size
        self.max_chunks = max_chunks
        self.eos_token_id = tokenizer.token_to_id('<|endoftext|>')

    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: Optional[PromptType] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Encodes the given sentences using the JEPA encoder.
        """
        if not sentences:
            return np.empty((0, self.encoder.config.n_embd))

        all_embeddings = []

        # Process in batches for efficiency
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i:i + self.batch_size]
            batch_embeddings = encode_batch_for_sts(
                batch_sentences,
                self.tokenizer,
                self.chunk_encoder,
                self.encoder,
                self.chunk_size,
                self.device,
                max_chunks=self.max_chunks,
                eos_token_id=self.eos_token_id
            )
            all_embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


def compute_arxiv_hcp2p_score(chunk_encoder, encoder, target_chunk_encoder, target_encoder,
                              tokenizer, chunk_size, device, batch_size=32):
    """
    Compute ArXivHierarchicalClusteringP2P V-measure using MTEB.
    Uses the EMA (target) encoders for more stable evaluation.
    Returns:
        v_measure: The V-measure score from the clustering task
    """
    chunk_encoder.eval()
    encoder.eval()
    target_chunk_encoder.eval()
    target_encoder.eval()

    # Remove cache before evaluation.
    shutil.rmtree('results')

    # Create MTEB wrapper model with unique name
    model = JEPAModelForMTEB(
        target_chunk_encoder=target_chunk_encoder,
        target_encoder=target_encoder,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        device=device,
        batch_size=batch_size,
    )

    # Get the ArXivHierarchicalClusteringP2P task
    tasks = mteb.get_tasks(tasks=['ArXivHierarchicalClusteringP2P'])

    if not tasks:
        print("Warning: ArXivHierarchicalClusteringP2P task not found")
        return 0.0

    # Create MTEB evaluation object with minimal verbosity
    evaluation = mteb.MTEB(tasks=tasks, verbosity=0)

    # Run evaluation
    results = evaluation.run(model, verbosity=0)

    # Remove cache after evaluation
    shutil.rmtree('results')

    chunk_encoder.train()
    encoder.train()
    target_chunk_encoder.train()
    target_encoder.train()

    # Extract V-measure score
    if results and len(results) > 0:
        # MTEB returns a list of task results
        task_result = results[0]
        # Access the test scores correctly
        if hasattr(task_result, 'scores') and 'test' in task_result.scores:
            test_scores = task_result.scores['test']
            if isinstance(test_scores, list) and len(test_scores) > 0:
                # For clustering tasks, scores is a list with one element
                v_measure = test_scores[0]['main_score']
            else:
                v_measure = test_scores.get('main_score', 0.0)
            return v_measure
    return 0.0