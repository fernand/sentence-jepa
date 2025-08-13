#!/usr/bin/env python3
"""
Evaluate JEPA model on MTEB P2P (Pairwise) tasks.
Uses the target (EMA) encoders for more stable evaluation.
"""

import argparse
import torch
import tokenizers

import mteb

from model import (
    ChunkEncoder, ChunkEncoderConfig,
    Encoder, EncoderConfig,
)
from mteb_validation import JEPAModelForMTEB


# JEPAModelForMTEB class is now imported from mteb_validation.py


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


def main():
    parser = argparse.ArgumentParser(description='Evaluate JEPA model on MTEB P2P tasks')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['ArXivHierarchicalClusteringP2P'],
                        help='MTEB P2P tasks to evaluate on')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation')
    parser.add_argument('--tokenizer', type=str, default='data/falcon-7b-instruct_tokenizer.json',
                        help='Path to tokenizer file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    parser.add_argument('--verbosity', type=int, default=2,
                        help='Verbosity level for MTEB (0-3)')
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    chunk_encoder, encoder, target_chunk_encoder, target_encoder, chunk_size = load_model(args.checkpoint)

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = tokenizers.Tokenizer.from_file(args.tokenizer)

    model = JEPAModelForMTEB(
        target_chunk_encoder=target_chunk_encoder,
        target_encoder=target_encoder,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        device=torch.device(args.device),
        batch_size=args.batch_size,
    )

    print(f"\nEvaluating on tasks: {args.tasks}")

    # Get MTEB tasks
    tasks = mteb.get_tasks(tasks=args.tasks)

    if not tasks:
        print(f"Warning: No valid tasks found for {args.tasks}")
        print("\nAvailable P2P tasks:")
        all_tasks = mteb.get_tasks()
        p2p_tasks = [t.metadata.name for t in all_tasks if 'P2P' in t.metadata.name or 'Pair' in str(t.metadata.type)]
        for task in p2p_tasks:
            print(f"  - {task}")
        return

    # Create MTEB evaluation object
    evaluation = mteb.MTEB(tasks=tasks, verbosity=args.verbosity)

    # Run evaluation
    print("\nStarting MTEB evaluation...")
    results = evaluation.run(model)
    print("\nEvaluation complete!")

    if results:
        print("\nResults summary:")
        for task_result in results:
            print(f"\n{task_result.task_name}:")
            print(f"\n{task_result.scores['test']}")


if __name__ == '__main__':
    main()