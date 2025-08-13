import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F
import ir_datasets
import os
import pickle

from model import (
    ChunkEncoder, ChunkEncoderConfig,
    Encoder, EncoderConfig,
)

def load_model(checkpoint_path):
    """Load the trained JEPA model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Model configuration (matching train.py)
    vocab_size = 65024  # Falcon tokenizer
    chunk_size = 32
    n_embd = 768

    chunk_enc_config = ChunkEncoderConfig(
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        n_layer=6,
        n_head=12,
        n_embd=n_embd
    )
    enc_config = EncoderConfig(
        max_chunks=64,
        n_layer=12,
        n_head=12,
        n_embd=n_embd
    )

    # Initialize models
    chunk_encoder = ChunkEncoder(chunk_enc_config)
    encoder = Encoder(enc_config)
    target_chunk_encoder = ChunkEncoder(chunk_enc_config)
    target_encoder = Encoder(enc_config)

    # Load state dicts - handle compiled model prefixes
    def fix_state_dict(state_dict):
        """Remove '_orig_mod.' prefix from compiled model state dict."""
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    chunk_encoder.load_state_dict(fix_state_dict(checkpoint['chunk_encoder']))
    encoder.load_state_dict(fix_state_dict(checkpoint['context_encoder']))
    target_chunk_encoder.load_state_dict(fix_state_dict(checkpoint['target_chunk_encoder']))
    target_encoder.load_state_dict(fix_state_dict(checkpoint['target_encoder']))

    return chunk_encoder, encoder, target_chunk_encoder, target_encoder, chunk_size

def encode_text_to_chunks(text, tokenizer, chunk_encoder, encoder, chunk_size, device, max_tokens=1024):
    """
    Encode text into chunk embeddings (not pooled into single vector).
    Returns all chunk CLS embeddings for late interaction.
    """
    # Tokenize the text
    tokens = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors='pt')['input_ids'].to(device)

    # Calculate how many chunks we need
    tokens_per_chunk = chunk_size - 1  # Reserve 1 for CLS token
    original_token_count = tokens.shape[1]

    # Calculate the actual number of chunks needed
    n_chunks_needed = (original_token_count + tokens_per_chunk - 1) // tokens_per_chunk

    # Calculate total tokens needed (must be multiple of tokens_per_chunk)
    total_tokens_needed = n_chunks_needed * tokens_per_chunk

    # Create attention mask for original tokens
    attention_mask = torch.ones(1, original_token_count, dtype=torch.bool, device=device)

    # Pad tokens and mask if needed
    if original_token_count < total_tokens_needed:
        padding_needed = total_tokens_needed - original_token_count
        # Use EOS token for padding
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        tokens = F.pad(tokens, (0, padding_needed), value=pad_token_id)
        # Extend attention mask with False for padded positions
        attention_mask = F.pad(attention_mask, (0, padding_needed), value=False)

    with torch.no_grad():
        # Process through chunk encoder WITH attention mask
        chunk_embeddings = chunk_encoder(tokens, attention_mask)  # (1, n_chunks, n_embd)

        # Create positions for all chunks
        all_positions = torch.arange(n_chunks_needed, device=device).unsqueeze(0)

        # Encode chunks with positional information
        context_embeddings = encoder(chunk_embeddings, all_positions)  # (1, n_chunks, n_embd)

        # Return all chunk embeddings (not pooled)
        return context_embeddings.squeeze(0)  # (n_chunks, n_embd)

def late_interaction_score(query_chunks, doc_chunks):
    """
    Compute late interaction score (MaxSim) between query and document chunks.
    """
    # Normalize chunks for cosine similarity
    query_chunks = F.normalize(query_chunks, p=2, dim=-1)
    doc_chunks = F.normalize(doc_chunks, p=2, dim=-1)

    # Compute similarity matrix: (n_query_chunks, n_doc_chunks)
    similarity_matrix = torch.matmul(query_chunks, doc_chunks.T)

    # MaxSim: for each query chunk, take max similarity to any doc chunk
    max_sims = similarity_matrix.max(dim=1)[0]  # (n_query_chunks,)

    # Average across query chunks
    score = max_sims.mean().item()

    return score

def evaluate_msmarco_simple(model_components, tokenizer, device, num_queries=1000):
    """
    Evaluate on MS MARCO dev small dataset.
    """
    chunk_encoder, encoder, target_chunk_encoder, target_encoder, chunk_size = model_components

    # Move models to device and set to eval mode
    chunk_encoder = chunk_encoder.to(device).eval()
    encoder = encoder.to(device).eval()
    target_chunk_encoder = target_chunk_encoder.to(device).eval()
    target_encoder = target_encoder.to(device).eval()

    # Use EMA encoders
    chunk_enc = target_chunk_encoder
    enc = target_encoder

    print("Loading MS MARCO dev small dataset...")
    dataset = ir_datasets.load("msmarco-passage/dev/small")

    # Load queries
    print(f"Loading queries (limit: {num_queries})...")
    queries = {}
    for i, query in enumerate(dataset.queries_iter()):
        if i >= num_queries:
            break
        queries[query.query_id] = query.text

    print(f"Loaded {len(queries)} queries")

    # Load qrels (relevance judgments)
    print("Loading relevance judgments...")
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id in queries:
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    # Load documents (passages)
    print("Loading passages...")
    all_doc_ids = set()
    for query_id in qrels:
        all_doc_ids.update(qrels[query_id].keys())

    docs = {}
    for doc in tqdm(dataset.docs_iter(), desc="Loading docs"):
        if doc.doc_id in all_doc_ids:
            docs[doc.doc_id] = doc.text
        if len(docs) >= len(all_doc_ids):
            break

    print(f"Loaded {len(docs)} relevant passages")

    # For each query, we'll also sample some negative passages
    print("Sampling negative passages...")
    all_docs_list = []
    doc_count = 0
    for doc in dataset.docs_iter():
        all_docs_list.append((doc.doc_id, doc.text))
        doc_count += 1
        if doc_count >= 10000:  # Sample from first 10k docs
            break

    # Cache directory
    cache_dir = 'msmarco_cache_simple'
    os.makedirs(cache_dir, exist_ok=True)

    # Encode queries
    print("Encoding queries...")
    query_embeddings = {}
    for query_id, query_text in tqdm(queries.items(), desc="Encoding queries"):
        query_chunks = encode_text_to_chunks(
            query_text, tokenizer, chunk_enc, enc, chunk_size, device
        )
        query_embeddings[query_id] = query_chunks.cpu()

    # Encode documents
    print("Encoding passages...")
    doc_embeddings = {}

    # First encode relevant docs
    for doc_id, doc_text in tqdm(docs.items(), desc="Encoding relevant docs"):
        doc_chunks = encode_text_to_chunks(
            doc_text, tokenizer, chunk_enc, enc, chunk_size, device
        )
        doc_embeddings[doc_id] = doc_chunks.cpu()

    # Add some negative samples for better evaluation
    negative_sample_size = min(1000, len(all_docs_list))
    sampled_indices = np.random.choice(len(all_docs_list), negative_sample_size, replace=False)

    for idx in tqdm(sampled_indices, desc="Encoding negative samples"):
        doc_id, doc_text = all_docs_list[idx]
        if doc_id not in doc_embeddings:
            doc_chunks = encode_text_to_chunks(
                doc_text, tokenizer, chunk_enc, enc, chunk_size, device
            )
            doc_embeddings[doc_id] = doc_chunks.cpu()

    print(f"Total passages encoded: {len(doc_embeddings)}")

    # Compute rankings
    print("Computing rankings...")
    reciprocal_ranks = []
    recall_at = {1: [], 10: [], 100: []}

    for query_id in tqdm(queries.keys(), desc="Ranking"):
        if query_id not in qrels:
            continue

        query_chunks = query_embeddings[query_id].to(device)

        # Score all documents
        scores = []
        for doc_id, doc_chunks_cpu in doc_embeddings.items():
            doc_chunks = doc_chunks_cpu.to(device)
            score = late_interaction_score(query_chunks, doc_chunks)
            scores.append((doc_id, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Calculate metrics
        relevant_docs = qrels[query_id]
        found = False
        for rank, (doc_id, _) in enumerate(scores, 1):
            if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
                if not found:
                    reciprocal_ranks.append(1.0 / rank)
                    found = True

                # Update recall
                for k in recall_at.keys():
                    if rank <= k:
                        recall_at[k].append(1.0)
                        break

        if not found:
            reciprocal_ranks.append(0.0)
            for k in recall_at.keys():
                recall_at[k].append(0.0)

    # Calculate final metrics
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    print(f"\nResults on MS MARCO dev small:")
    print(f"MRR@10: {mrr:.4f}")
    for k, recalls in recall_at.items():
        if recalls:
            print(f"Recall@{k}: {np.mean(recalls):.4f}")

    print(f"\nDataset statistics:")
    print(f"Queries evaluated: {len(reciprocal_ranks)}")
    print(f"Passages ranked per query: {len(doc_embeddings)}")

    return {
        'mrr': mrr,
        'recall_at_1': np.mean(recall_at[1]) if recall_at[1] else 0,
        'recall_at_10': np.mean(recall_at[10]) if recall_at[10] else 0,
        'recall_at_100': np.mean(recall_at[100]) if recall_at[100] else 0,
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate JEPA model on MS MARCO')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tokenizer', type=str, default='tiiuae/falcon-7b')
    parser.add_argument('--num_queries', type=int, default=100,
                        help='Number of queries to evaluate (default: 100)')
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model_components = load_model(args.checkpoint)

    print(f"Loading tokenizer {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = evaluate_msmarco_simple(
        model_components,
        tokenizer,
        torch.device(args.device),
        args.num_queries
    )

    print("\nEvaluation complete!")

    # Save results
    results_file = args.checkpoint.replace('.pt', f'_msmarco_simple_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'num_queries': args.num_queries,
            'mrr': float(results['mrr']),
            'recall_at_1': float(results['recall_at_1']),
            'recall_at_10': float(results['recall_at_10']),
            'recall_at_100': float(results['recall_at_100']),
        }, f, indent=2)
    print(f"Results saved to {results_file}")

if __name__ == '__main__':
    main()