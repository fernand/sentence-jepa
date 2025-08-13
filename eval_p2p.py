#!/usr/bin/env python3
"""
Evaluate JEPA model on MTEB P2P (Pairwise) tasks.
Uses the target (EMA) encoders for more stable evaluation.
"""

import argparse
from typing import Optional
import numpy as np
import torch
import tokenizers
from tqdm import tqdm

import mteb
from mteb.encoder_interface import PromptType

from model import (
    ChunkEncoder, ChunkEncoderConfig,
    Encoder, EncoderConfig,
)
from sts_validation import encode_batch_for_sts


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
        max_chunks=64
    ):
        """
        Initialize the MTEB wrapper.

        Args:
            target_chunk_encoder: The EMA chunk encoder model
            target_encoder: The EMA context encoder model
            tokenizer: Tokenizer object
            chunk_size: Size of each chunk
            device: Device to run on
            batch_size: Batch size for encoding
            max_chunks: Maximum number of chunks per text
        """
        self.chunk_encoder = target_chunk_encoder.to(device).eval()
        self.encoder = target_encoder.to(device).eval()
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.device = device
        self.batch_size = batch_size
        self.max_chunks = max_chunks

        # Get EOS token ID for padding
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

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task (for logging/debugging).
            prompt_type: The prompt type to use (not used in JEPA).
            **kwargs: Additional arguments (not used).

        Returns:
            The encoded sentences as a numpy array of shape (n_sentences, n_embd).
        """
        if not sentences:
            return np.empty((0, self.encoder.config.n_embd))

        all_embeddings = []

        # Process in batches for efficiency
        for i in tqdm(range(0, len(sentences), self.batch_size),
                     desc=f"Encoding for {task_name}", leave=False):
            batch_sentences = sentences[i:i + self.batch_size]

            # Use the same encoding function as in STS evaluation
            # This ensures proper attention masking for padding
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

            # Move to CPU and convert to numpy
            all_embeddings.append(batch_embeddings.cpu().numpy())

        # Concatenate all batches
        return np.vstack(all_embeddings)


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
    parser.add_argument('--output_folder', type=str, default='mteb_results',
                        help='Folder to save MTEB results')
    parser.add_argument('--verbosity', type=int, default=2,
                        help='Verbosity level for MTEB (0-3)')
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    chunk_encoder, encoder, target_chunk_encoder, target_encoder, chunk_size = load_model(args.checkpoint)

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = tokenizers.Tokenizer.from_file(args.tokenizer)

    print(f"Initializing JEPA model wrapper for MTEB...")
    model = JEPAModelForMTEB(
        target_chunk_encoder=target_chunk_encoder,
        target_encoder=target_encoder,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        device=torch.device(args.device),
        batch_size=args.batch_size
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
    results = evaluation.run(
        model,
        output_folder=args.output_folder,
        overwrite_results=False
    )

    print("\nEvaluation complete!")
    print(f"Results saved to {args.output_folder}")

    # Print summary of results
    if results:
        print("\nResults summary:")
        for task_name, task_results in results.items():
            print(f"\n{task_name}:")
            if isinstance(task_results, dict):
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
                    elif isinstance(value, dict) and 'main_score' in value:
                        print(f"  {metric}: {value['main_score']:.4f}")


if __name__ == '__main__':
    main()