import argparse
import copy
import math
import os
import subprocess
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
import tokenizers

from dataloader import DataLoader
from model import (
    ChunkEncoder, ChunkEncoderConfig,
    Encoder, EncoderConfig,
    Predictor, PredictorConfig
)
from mteb_validation import compute_stsb_spearman, compute_arxiv_hcp2p_score

def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_module(model):
    """Get the underlying module from DDP wrapper if needed."""
    return model.module if hasattr(model, 'module') else model

def extract_visible_chunks(chunk_embeddings, chunk_mask):
    """
    Extract only visible (non-masked) chunks from chunk embeddings and their positions.
    Args:
        chunk_embeddings: (B, n_chunks, D) all chunk embeddings
        chunk_mask: (B, n_chunks) boolean mask where True = masked
    Returns:
        visible_chunks: (B, max_visible, D) tensor containing only visible chunks
        visible_positions: (B, max_visible) tensor containing original positions of visible chunks
    """
    B, n_chunks, D = chunk_embeddings.shape
    visible_mask = ~chunk_mask  # True for visible positions
    # Gather only visible chunks for context encoder
    n_visible_per_batch = visible_mask.sum(dim=1)  # (B,)
    max_visible = n_visible_per_batch.max().item()
    # Create tensor with only visible chunks (pre-allocate for efficiency)
    visible_chunks = torch.zeros(B, max_visible, D, device=chunk_embeddings.device, dtype=chunk_embeddings.dtype)
    visible_positions = torch.zeros(B, max_visible, device=chunk_embeddings.device, dtype=torch.long)
    # Create position indices
    positions = torch.arange(n_chunks, device=chunk_embeddings.device)
    # Vectorized gathering with better memory patterns
    for b in range(B):
        n_visible = n_visible_per_batch[b].item()
        if n_visible > 0:
            # Use contiguous slicing for better memory access
            visible_chunks[b, :n_visible] = chunk_embeddings[b, visible_mask[b]]
            visible_positions[b, :n_visible] = positions[visible_mask[b]]
    return visible_chunks, visible_positions

def compute_jepa_loss(predicted_embeddings, target_embeddings, target_positions, chunk_mask, cosine_weight):
    """
    Compute JEPA loss between predicted and target embeddings.
    Args:
        predicted_embeddings: (B, n_target, D) predicted embeddings
        target_embeddings: (B, n_chunks, D) all target embeddings
        target_positions: (B, n_target) positions of masked chunks
        chunk_mask: (B, n_chunks) boolean mask
    Returns:
        loss: scalar loss value
    """
    B, max_targets, D = predicted_embeddings.shape
    device = predicted_embeddings.device
    # Gather target embeddings for masked positions using advanced indexing
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, max_targets)
    gathered_targets = target_embeddings[batch_indices, target_positions]
    # Create valid mask for non-padding positions
    n_masked_per_batch = chunk_mask.sum(dim=1)  # (B,)
    valid_mask = torch.arange(max_targets, device=device).unsqueeze(0) < n_masked_per_batch.unsqueeze(1)
    if valid_mask.any():
        loss = cosine_weight * (
            1 -  F.cosine_similarity(predicted_embeddings[valid_mask], gathered_targets[valid_mask], 1, 1e-8).mean()
            ) + (1 - cosine_weight) * F.l1_loss(
                predicted_embeddings[valid_mask],
                gathered_targets[valid_mask],
                reduction='mean'
            )
        return loss
    else:
        return torch.tensor(0.0, device=device, dtype=predicted_embeddings.dtype)

def train_step(
    chunk_encoder, encoder, target_chunk_encoder, target_encoder, predictor, batch, optimizers, cosine_weight):
    tokens = batch['tokens']
    chunk_mask = batch['chunk_mask']
    target_positions = batch['target_positions']
    amp_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    with amp_context:
        # Encode tokens to chunks
        chunk_embeddings = chunk_encoder(tokens)
        # I-JEPA style: Extract ONLY visible chunks for context encoder
        visible_chunks, visible_positions = extract_visible_chunks(chunk_embeddings, chunk_mask)
        # Context encoder processes ONLY visible chunks with their positions
        context_embeddings = encoder(visible_chunks, visible_positions)
        # Target path uses EMA versions (no gradients)
        with torch.no_grad():
            # Use target (EMA) chunk encoder for target path
            target_chunk_embeddings = target_chunk_encoder(tokens)
            # Create positions for all chunks (0, 1, 2, ..., n_chunks-1)
            all_positions = torch.arange(target_chunk_embeddings.shape[1], device=target_chunk_embeddings.device)
            all_positions = all_positions.unsqueeze(0).expand(target_chunk_embeddings.shape[0], -1)
            target_embeddings = target_encoder(target_chunk_embeddings, all_positions)
        # Predict masked chunks using context from visible chunks only
        predicted_embeddings = predictor(context_embeddings, target_positions, visible_positions)
        loss = compute_jepa_loss(
            predicted_embeddings, target_embeddings, target_positions, chunk_mask, cosine_weight)
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description='Train JEPA model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Max learning rate')
    parser.add_argument('--cosine_weight', type=float, default=0.3, help='How much to weight the cosine similarity loss.')
    parser.add_argument('--weight_decay', type=float, default=0.04, help='Starting weight decay for AdamW optimizer')
    parser.add_argument('--final_weight_decay', type=float, default=0.4, help='Final weight decay for AdamW optimizer')
    parser.add_argument('--ema_start', type=float, default=0.996, help='Starting EMA decay rate')
    parser.add_argument('--ema_end', type=float, default=1.0, help='Final EMA decay rate')
    parser.add_argument('--mask_ratio', type=float, default=0.50, help='Chunk masking ratio')
    parser.add_argument('--val_loss_every', type=int, default=250, help='Validation frequency')
    parser.add_argument('--project_name', type=str, default='sentence-jepa', help='Comet ML project name')
    parser.add_argument('--num_steps', type=int, default=None, help='Number of training steps')
    parser.add_argument('--dataset_path', type=str, default='data/fineweb-edu_10B', help='Path to dataset')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Number of warmup steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--use_comet', action='store_true', help='Use Comet ML for logging')
    parser.add_argument('--eval_arxiv_p2p', action='store_true', help='Evaluate ArXivHierarchicalClusteringP2P during validation')
    parser.add_argument('--eval_stsb', action='store_true', help='Evaluate STS-B Spearman correlation during validation')
    args = parser.parse_args()

    rank, world_size, local_rank, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    torch.manual_seed(args.seed + rank)

    # Initialize tokenizer for STS-B evaluation if needed
    tokenizer = None
    if args.eval_stsb:
        tokenizer = tokenizers.Tokenizer.from_file('data/falcon-7b-instruct_tokenizer.json')

    seq_len = 1024
    if args.num_steps is None:
        tokens_per_step = seq_len * args.batch_size * world_size
        args.num_steps = 10_000_000_000 // tokens_per_step
    if args.warmup_steps is None:
        args.warmup_steps = int(0.03 * args.num_steps)

    # Model configuration
    vocab_size = 65024  # Falcon tokenizer
    chunk_size = 32
    n_chunks = seq_len // chunk_size
    n_embd = 768

    # Beta2 scaling for optimizer
    beta2 = (0.95)**(1.0/(512/args.batch_size))

    if rank == 0:
        print(f'Training configuration:')
        print(f'  World size: {world_size}')
        print(f'  Batch size per GPU: {args.batch_size}')
        print(f'  Total batch size: {args.batch_size * world_size}')
        print(f'  Sequence length: {seq_len}')
        print(f'  Number of steps: {args.num_steps}')
        print(f'  Learning rate: {args.learning_rate}')
        print(f'  Weight decay: {args.weight_decay:.4f} -> {args.final_weight_decay:.4f}')
        print(f'  Warmup steps: {args.warmup_steps}')
        print(f'  Mask ratio: {args.mask_ratio}')
        print(f'  EMA decay: {args.ema_start:.4f} -> {args.ema_end:.4f}')
        print(f'  Beta2: {beta2:.6f}')

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
    pred_config = PredictorConfig(
        max_chunks=64,
        n_layer=6,
        n_head=6,
        n_embd=384,
        n_encoder_embd=n_embd
    )

    experiment = None
    if args.use_comet and rank == 0:
        import comet_ml
        experiment = comet_ml.Experiment(
            project_name=args.project_name,
            auto_metric_logging=True,
            auto_param_logging=True,
        )
        experiment.log_parameters(vars(args))
        experiment.log_parameter('total_batch_size', args.batch_size * world_size)
        experiment.log_parameter('world_size', world_size)
        experiment.log_parameter('tokens_per_step', seq_len * args.batch_size * world_size)
        experiment.log_parameter('seq_len', seq_len)
        experiment.log_parameter('beta2', beta2)
        experiment.log_parameter('ema_start', args.ema_start)
        experiment.log_parameter('ema_end', args.ema_end)
        experiment.log_parameter('weight_decay_start', args.weight_decay)
        experiment.log_parameter('weight_decay_final', args.final_weight_decay)
        model_dir = f'models/{experiment.id}'
    elif not args.use_comet:
        import uuid
        model_dir = f'models/{uuid.uuid4()}'
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)

    torch.set_float32_matmul_precision('high')
    chunk_encoder = ChunkEncoder(chunk_enc_config).to(device)
    encoder = Encoder(enc_config).to(device)
    target_chunk_encoder = copy.deepcopy(chunk_encoder)  # EMA version of chunk encoder
    target_encoder = copy.deepcopy(encoder)  # EMA version of context encoder
    predictor = Predictor(pred_config).to(device)

    if args.compile:
        chunk_encoder = torch.compile(chunk_encoder)
        encoder = torch.compile(encoder)
        target_chunk_encoder = torch.compile(target_chunk_encoder)
        target_encoder = torch.compile(target_encoder)
        predictor = torch.compile(predictor)

    if is_distributed:
        chunk_encoder = DDP(chunk_encoder, device_ids=[local_rank])
        encoder = DDP(encoder, device_ids=[local_rank])
        predictor = DDP(predictor, device_ids=[local_rank])
        # Note: target_encoder is not wrapped in DDP as it doesn't need gradients

    # Optimizers with beta2 scaling from https://arxiv.org/abs/2507.07101
    chunk_enc_optimizers = get_module(chunk_encoder).configure_optimizers(
        wd=args.weight_decay, adam_lr=args.learning_rate, adam_betas=(0.9, beta2)
    )
    context_enc_optimizer = get_module(encoder).configure_optimizers(
        wd=args.weight_decay, adam_lr=args.learning_rate, adam_betas=(0.9, beta2)
    )
    predictor_optimizer = get_module(predictor).configure_optimizers(
        wd=args.weight_decay, adam_lr=args.learning_rate, adam_betas=(0.9, beta2)
    )

    optimizers = [chunk_enc_optimizers, context_enc_optimizer, predictor_optimizer]

    schedulers = []
    for optimizer in optimizers:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.num_steps - args.warmup_steps,
        )
        schedulers.append(scheduler)

    train_pattern = os.path.join(args.dataset_path, 'fineweb-edu_train_*.bin')
    val_pattern = os.path.join(args.dataset_path, 'fineweb-edu_val_*.bin')
    train_loader = DataLoader(
        filename_pattern=train_pattern,
        B=args.batch_size,
        chunk_size=chunk_size,
        n_chunks=n_chunks,
        mask_ratio=args.mask_ratio,
        device=device
    )
    val_loader = DataLoader(
        filename_pattern=val_pattern,
        B=args.batch_size,
        chunk_size=chunk_size,
        n_chunks=n_chunks,
        mask_ratio=args.mask_ratio,
        device=device
    )

    if rank == 0:
        print('\nStarting training...')

    # Store initial learning rates
    initial_lrs = []
    for optimizer in optimizers:
        initial_lrs.append([group['lr'] for group in optimizer.param_groups])

    # Create EMA decay schedule (linear from ema_start to ema_end)
    def get_ema_decay(step):
        progress = min(step / args.num_steps, 1.0)
        return args.ema_start + progress * (args.ema_end - args.ema_start)

    # Create weight decay schedule (cosine from weight_decay to final_weight_decay)
    def get_weight_decay(step):
        if step < args.warmup_steps:
            # Use initial weight decay during warmup
            return args.weight_decay
        else:
            # Cosine schedule after warmup
            progress = (step - args.warmup_steps) / max(1, args.num_steps - args.warmup_steps)
            progress = min(progress, 1.0)
            # Cosine annealing from weight_decay to final_weight_decay
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return args.final_weight_decay + (args.weight_decay - args.final_weight_decay) * cosine_factor

    for step in range(args.num_steps):
        batch_start_time = time.perf_counter()
        batch = train_loader.next_batch()
        # Apply warmup or cosine schedule for learning rate
        if step < args.warmup_steps:
            # Linear warmup
            lr_scale = (step + 1) / args.warmup_steps
            for opt_idx, optimizer in enumerate(optimizers):
                for group_idx, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = initial_lrs[opt_idx][group_idx] * lr_scale

        # Update weight decay for all optimizers
        current_wd = get_weight_decay(step)
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = current_wd

        loss = train_step(
            chunk_encoder, encoder, target_chunk_encoder, target_encoder, predictor, batch,
            optimizers, args.cosine_weight
        )

        # Update target encoders with EMA using scheduled decay
        current_ema = get_ema_decay(step)
        with torch.no_grad():
            # Update target chunk encoder
            chunk_model = get_module(chunk_encoder)
            for param_q, param_k in zip(chunk_model.parameters(), target_chunk_encoder.parameters()):
                param_k.data.mul_(current_ema).add_(param_q.data, alpha=1 - current_ema)
            # Update target context encoder
            context_model = get_module(encoder)
            for param_q, param_k in zip(context_model.parameters(), target_encoder.parameters()):
                param_k.data.mul_(current_ema).add_(param_q.data, alpha=1 - current_ema)

        batch_time = time.perf_counter() - batch_start_time

        # Update learning rate schedulers (after warmup)
        if step >= args.warmup_steps:
            for scheduler in schedulers:
                scheduler.step()

        if rank == 0 and step % 10 == 0:
            current_lr = optimizers[0].param_groups[0]['lr']
            print(f'Step {step}/{args.num_steps} | Loss: {loss:.4f} | LR: {current_lr:.6f} | Time: {batch_time*1e3:.0f}ms')
            if experiment:
                experiment.log_metric('train_loss', loss, step=step)
                experiment.log_metric('lr', current_lr, step=step)
                experiment.log_metric('weight_decay', current_wd, step=step)
                experiment.log_metric('ema_decay', current_ema, step=step)

        if step % args.val_loss_every == 0 and step > 0:
            # Compute ArXivHierarchicalClusteringP2P V-measure if requested
            arxiv_score = None
            if args.eval_arxiv_p2p and rank == 0:
                arxiv_score = compute_arxiv_hcp2p_score(
                    chunk_encoder, encoder, target_chunk_encoder, target_encoder,
                    tokenizer, chunk_size, device, model_suffix=f"step_{step}"
                )

            # Compute STS-B Spearman correlation if requested
            stsb_spearman = None
            if args.eval_stsb and rank == 0:
                stsb_spearman = compute_stsb_spearman(
                    chunk_encoder, encoder, target_chunk_encoder, target_encoder, tokenizer, chunk_size, device)

            if rank == 0:
                metrics_str = f'Step {step}'
                if arxiv_score is not None:
                    metrics_str += f' | ArXiv P2P: {arxiv_score:.4f}'
                    if experiment:
                        experiment.log_metric('arxiv_hcp2p', arxiv_score, step=step)
                if stsb_spearman is not None:
                    metrics_str += f' | STS-B Spearman: {stsb_spearman:.4f}'
                    if experiment:
                        experiment.log_metric('stsb_spearman', stsb_spearman, step=step)
                print(metrics_str)

        if rank == 0 and step % 5000 == 0 and step > 0:
            checkpoint = {
                'step': step,
                'chunk_encoder': get_module(chunk_encoder).state_dict(),
                'context_encoder': get_module(encoder).state_dict(),
                'target_chunk_encoder': target_chunk_encoder.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'predictor': get_module(predictor).state_dict(),
                'optimizers': [opt.state_dict() for opt in optimizers],
                'schedulers': [sched.state_dict() for sched in schedulers],
                'current_ema': current_ema,
                'args': args,
            }
            torch.save(checkpoint, f'{model_dir}/checkpoint_step_{step}.pt')
            print(f'Saved checkpoint at step {step}')

    if rank == 0:
        checkpoint = {
            'step': args.num_steps,
            'chunk_encoder': get_module(chunk_encoder).state_dict(),
            'context_encoder': get_module(encoder).state_dict(),
            'target_chunk_encoder': target_chunk_encoder.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'predictor': get_module(predictor).state_dict(),
            'optimizers': [opt.state_dict() for opt in optimizers],
            'schedulers': [sched.state_dict() for sched in schedulers],
            'args': args,
        }
        torch.save(checkpoint, f'{model_dir}/checkpoint_final.pt')
        print('Training complete! Saved final checkpoint.')

    cleanup_distributed()
    if experiment:
        experiment.end()

    pod_id = os.environ.get('RUNPOD_POD_ID')
    if pod_id:
        subprocess.run(['runpodctl', 'remove', 'pod', pod_id])

if __name__ == '__main__':
    main()
