import argparse

import torch
import numpy as np
import tokenizers
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from model import (
    ChunkEncoder, ChunkEncoderConfig,
    Encoder, EncoderConfig,
)
from sts_validation import (
    encode_sentence_for_sts,
    encode_batch_for_sts,
    compute_stsb_spearman
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
                new_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix
            else:
                new_state_dict[k] = v
        return new_state_dict

    chunk_encoder.load_state_dict(fix_state_dict(checkpoint['chunk_encoder']))
    encoder.load_state_dict(fix_state_dict(checkpoint['context_encoder']))
    target_chunk_encoder.load_state_dict(fix_state_dict(checkpoint['target_chunk_encoder']))
    target_encoder.load_state_dict(fix_state_dict(checkpoint['target_encoder']))

    return chunk_encoder, encoder, target_chunk_encoder, target_encoder, chunk_size



def evaluate_sts_benchmark_simple(model_components, tokenizer, device, num_samples=None):
    """
    Simple evaluation using compute_stsb_spearman from sts_validation.py.
    
    Args:
        model_components: Tuple of (chunk_encoder, encoder, target_chunk_encoder, target_encoder, chunk_size)
        tokenizer: The tokenizer to use
        device: The device to run on
        num_samples: Number of samples to evaluate (None for full dataset)
    """
    chunk_encoder, encoder, target_chunk_encoder, target_encoder, chunk_size = model_components
    
    # Move models to device
    chunk_encoder = chunk_encoder.to(device)
    encoder = encoder.to(device)
    target_chunk_encoder = target_chunk_encoder.to(device)
    target_encoder = target_encoder.to(device)
    
    # Use the compute_stsb_spearman function directly
    spearman_corr = compute_stsb_spearman(
        chunk_encoder, encoder, target_chunk_encoder, target_encoder,
        tokenizer, chunk_size, device, num_samples=num_samples
    )
    
    return spearman_corr

def evaluate_sts_benchmark(model_components, tokenizer, device, dataset_name='stsb', batch_size=32):
    """
    Evaluate the model on STS benchmark tasks with detailed analysis.

    Args:
        model_components: Tuple of (chunk_encoder, encoder, target_chunk_encoder, target_encoder, chunk_size)
        tokenizer: The tokenizer to use
        device: The device to run on
        dataset_name: Which STS dataset to evaluate on ('stsb' or specific year like 'sts12')
        batch_size: Batch size for encoding
    """
    chunk_encoder, encoder, target_chunk_encoder, target_encoder, chunk_size = model_components

    # Move models to device and set to eval mode
    chunk_encoder = chunk_encoder.to(device).eval()
    encoder = encoder.to(device).eval()
    target_chunk_encoder = target_chunk_encoder.to(device).eval()
    target_encoder = target_encoder.to(device).eval()

    # Load the dataset
    if dataset_name == 'stsb':
        # Load STS-B from GLUE
        dataset = load_dataset('glue', 'stsb')
        test_data = dataset['validation']
        sentences1 = test_data['sentence1']
        sentences2 = test_data['sentence2']
        scores = np.array(test_data['label'])
    elif dataset_name.startswith('mteb/sts'):
        dataset = load_dataset('mteb/sts16-sts')
        test_data = dataset['test']
        sentences1 = test_data['sentence1']
        sentences2 = test_data['sentence2']
        scores = np.array(test_data['score'])
    else:
        # For other STS datasets, you might need different loading logic
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")

    # First, analyze token lengths
    print(f"\nAnalyzing token lengths for {len(sentences1)} sentence pairs...")
    token_lengths = []
    for sent1, sent2 in zip(sentences1, sentences2):
        tokens1 = tokenizer.encode(sent1).ids
        tokens2 = tokenizer.encode(sent2).ids
        token_lengths.append(len(tokens1))
        token_lengths.append(len(tokens2))

    # Calculate percentiles
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    token_percentiles = np.percentile(token_lengths, percentiles)

    print("\nToken length distribution (all sentences):")
    print(f"  Min tokens: {np.min(token_lengths)}")
    print(f"  Max tokens: {np.max(token_lengths)}")
    print(f"  Mean tokens: {np.mean(token_lengths):.1f}")
    print(f"  Std tokens: {np.std(token_lengths):.1f}")
    print("\nPercentiles:")
    for p, val in zip(percentiles, token_percentiles):
        print(f"  {p:3d}%: {val:6.1f} tokens")

    # Calculate how many chunks these translate to
    tokens_per_chunk = chunk_size - 1  # Reserve 1 for CLS token
    print(f"\nChunk analysis (chunk_size={chunk_size}, tokens_per_chunk={tokens_per_chunk}):")
    for p, val in zip(percentiles, token_percentiles):
        n_chunks = int(np.ceil(val / tokens_per_chunk))
        print(f"  {p:3d}%: {val:6.1f} tokens → {n_chunks} chunk(s)")

    # Encode all sentences
    embeddings1 = []
    embeddings2 = []

    print(f"\nEncoding {len(sentences1)} sentence pairs...")

    # Use the target (EMA) encoders for evaluation as they are typically more stable
    use_ema = True
    if use_ema:
        print("Using EMA (target) encoders for evaluation")
        chunk_enc = target_chunk_encoder
        enc = target_encoder
    else:
        print("Using regular encoders for evaluation")
        chunk_enc = chunk_encoder
        enc = encoder

    # Get EOS token ID for padding
    eos_token_id = tokenizer.token_to_id('<|endoftext|>')

    # Process in batches
    for i in tqdm(range(0, len(sentences1), batch_size)):
        batch_sent1 = sentences1[i:i+batch_size]
        batch_sent2 = sentences2[i:i+batch_size]

        batch_emb1 = encode_batch_for_sts(batch_sent1, tokenizer, chunk_enc, enc, chunk_size, device, eos_token_id=eos_token_id)
        batch_emb2 = encode_batch_for_sts(batch_sent2, tokenizer, chunk_enc, enc, chunk_size, device, eos_token_id=eos_token_id)

        embeddings1.append(batch_emb1.cpu())
        embeddings2.append(batch_emb2.cpu())

    # Stack embeddings
    embeddings1 = torch.cat(embeddings1, dim=0).numpy()  # (N, n_embd)
    embeddings2 = torch.cat(embeddings2, dim=0).numpy()  # (N, n_embd)

    # Compute cosine similarities
    # Normalize embeddings
    embeddings1_norm = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
    embeddings2_norm = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarities
    cosine_sims = np.sum(embeddings1_norm * embeddings2_norm, axis=1)

    # Scale scores to [0, 5] range to match STS-B
    # Cosine similarity is in [-1, 1], map to [0, 5]
    predicted_scores = (cosine_sims + 1) * 2.5

    # Compute correlations
    spearman_corr = spearmanr(scores, predicted_scores)[0]
    pearson_corr = pearsonr(scores, predicted_scores)[0]

    print(f"\nResults on {dataset_name}:")
    print(f"Spearman correlation: {spearman_corr:.4f}")
    print(f"Pearson correlation: {pearson_corr:.4f}")

    # Additional analysis
    print(f"\nScore statistics:")
    print(f"Ground truth - Mean: {scores.mean():.2f}, Std: {scores.std():.2f}, Min: {scores.min():.2f}, Max: {scores.max():.2f}")
    print(f"Predictions - Mean: {predicted_scores.mean():.2f}, Std: {predicted_scores.std():.2f}, Min: {predicted_scores.min():.2f}, Max: {predicted_scores.max():.2f}")

    return {
        'spearman': spearman_corr,
        'pearson': pearson_corr,
        'predicted_scores': predicted_scores,
        'ground_truth_scores': scores
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate JEPA model on STS benchmarks')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--dataset', type=str, default='stsb', help='Which STS dataset to evaluate on')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation')
    parser.add_argument('--tokenizer', type=str, default='data/falcon-7b-instruct_tokenizer.json',
                        help='Path to tokenizer file')
    parser.add_argument('--simple', action='store_true', help='Use simple evaluation (compute_stsb_spearman only)')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate (for simple mode)')
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model_components = load_model(args.checkpoint)

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = tokenizers.Tokenizer.from_file(args.tokenizer)

    if args.simple:
        print(f"Using simple evaluation on STS-B...")
        if args.num_samples:
            print(f"Evaluating on {args.num_samples} samples")
        spearman_corr = evaluate_sts_benchmark_simple(
            model_components,
            tokenizer,
            torch.device(args.device),
            num_samples=args.num_samples
        )
        print(f"\nSTS-B Spearman correlation: {spearman_corr:.4f}")
        
        # Save simple results
        import json
        from pathlib import Path
        checkpoint_dir = Path(args.checkpoint).parent
        checkpoint_name = Path(args.checkpoint).name
        results_file = checkpoint_dir / checkpoint_name.replace('.pt', '_stsb_simple_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'checkpoint': args.checkpoint,
                'dataset': 'stsb',
                'spearman': float(spearman_corr),
                'num_samples': args.num_samples,
            }, f, indent=2)
        print(f"Results saved to {results_file}")
    else:
        print(f"Evaluating on {args.dataset}...")
        results = evaluate_sts_benchmark(
            model_components,
            tokenizer,
            torch.device(args.device),
            args.dataset
        )

        print("\nEvaluation complete!")

        # Optionally save results
        import json
        from pathlib import Path
        checkpoint_dir = Path(args.checkpoint).parent
        checkpoint_name = Path(args.checkpoint).name
        results_file = checkpoint_dir / checkpoint_name.replace('.pt', f'_{args.dataset.replace('/', '_')}_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'checkpoint': args.checkpoint,
                'dataset': args.dataset,
                'spearman': float(results['spearman']),
                'pearson': float(results['pearson']),
            }, f, indent=2)
        print(f"Results saved to {results_file}")

if __name__ == '__main__':
    main()