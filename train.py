import argparse
import copy
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader import DataLoader
from model import (
    ChunkEncoder, ChunkEncoderConfig,
    Encoder, EncoderConfig,
    Predictor, PredictorConfig
)

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

def compute_jepa_loss(predicted_embeddings, target_embeddings, target_positions, chunk_mask):
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
    # Positions are padded with 0, but 0 could be a valid position
    # So we check against the actual mask
    n_masked_per_batch = chunk_mask.sum(dim=1)  # (B,)
    position_indices = torch.arange(max_targets, device=device).unsqueeze(0).expand(B, max_targets)
    valid_mask = position_indices < n_masked_per_batch.unsqueeze(1)

    if valid_mask.any():
        # Compute L1 loss only on valid positions
        loss = F.l1_loss(
            predicted_embeddings[valid_mask],
            gathered_targets[valid_mask],
            reduction='mean'
        )
        return loss
    else:
        return torch.tensor(0.0, device=device)

def train_step(
    chunk_encoder, context_encoder, target_encoder, predictor, batch, optimizers):
    tokens = batch['tokens']
    chunk_mask = batch['chunk_mask']
    target_positions = batch['target_positions']
    amp_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    with amp_context:
        # 1. Encode tokens to chunks
        chunk_embeddings = chunk_encoder(tokens)
        # 2. Context path - MUST use mask to hide target chunks
        # The mask prevents information leakage from target chunks to context
        context_embeddings = context_encoder(chunk_embeddings, chunk_mask=chunk_mask)
        # 3. Target path (no masking, no gradients)
        with torch.no_grad():
            target_embeddings = target_encoder(chunk_embeddings, chunk_mask=None)
        # 4. Extract only visible (non-masked) context chunks for predictor
        # The predictor should only attend to visible chunks, not masked ones
        B, n_chunks, D = context_embeddings.shape
        visible_mask = ~chunk_mask  # True for visible positions

        # Efficient vectorized gathering of visible chunks
        # Count visible chunks per batch
        n_visible_per_batch = visible_mask.sum(dim=1)  # (B,)
        max_visible = n_visible_per_batch.max().item()

        # Create padded context tensor
        padded_context = torch.zeros(B, max_visible, D, device=context_embeddings.device, dtype=context_embeddings.dtype)

        # Use advanced indexing to gather visible chunks efficiently
        for b in range(B):
            n_visible = n_visible_per_batch[b].item()
            if n_visible > 0:
                # Get visible chunks for this batch element
                visible_chunks = context_embeddings[b][visible_mask[b]]
                padded_context[b, :n_visible] = visible_chunks
            else:
                # Edge case: all chunks masked (shouldn't happen with proper masking)
                padded_context[b, 0] = context_embeddings[b, 0]

        # 5. Predict masked chunks
        predicted_embeddings = predictor(padded_context, target_positions)
        # 6. Compute loss
        loss = compute_jepa_loss(predicted_embeddings, target_embeddings, target_positions, chunk_mask)
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    return loss.item()

def validate(chunk_encoder, context_encoder, target_encoder, predictor, val_loader):
    chunk_encoder.eval()
    context_encoder.eval()
    target_encoder.eval()
    predictor.eval()
    val_losses = []
    amp_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        for _ in range(50):  # Validate on 50 batches
            batch = val_loader.next_batch()
            with amp_context:
                chunk_embeddings = chunk_encoder(batch['tokens'])
                chunk_mask = batch['chunk_mask']
                context_embeddings = context_encoder(chunk_embeddings, chunk_mask=chunk_mask)
                target_embeddings = target_encoder(chunk_embeddings, chunk_mask=None)

                # Extract only visible context chunks for predictor
                B, n_chunks, D = context_embeddings.shape
                visible_mask = ~chunk_mask

                # Efficient vectorized gathering of visible chunks
                n_visible_per_batch = visible_mask.sum(dim=1)
                max_visible = n_visible_per_batch.max().item()

                # Create padded context tensor
                padded_context = torch.zeros(B, max_visible, D, device=context_embeddings.device, dtype=context_embeddings.dtype)

                # Use advanced indexing to gather visible chunks
                for b in range(B):
                    n_visible = n_visible_per_batch[b].item()
                    if n_visible > 0:
                        visible_chunks = context_embeddings[b][visible_mask[b]]
                        padded_context[b, :n_visible] = visible_chunks
                    else:
                        padded_context[b, 0] = context_embeddings[b, 0]

                predicted_embeddings = predictor(padded_context, batch['target_positions'])
                loss = compute_jepa_loss(
                    predicted_embeddings, target_embeddings,
                    batch['target_positions'], batch['chunk_mask']
                )
                val_losses.append(loss.item())
    chunk_encoder.train()
    context_encoder.train()
    target_encoder.train()
    predictor.train()
    return sum(val_losses) / len(val_losses) if val_losses else 0.0

def main():
    parser = argparse.ArgumentParser(description='Train JEPA model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--val_loss_every', type=int, default=250, help='Validation frequency')
    parser.add_argument('--project_name', type=str, default='sentence-jepa', help='Comet ML project name')
    parser.add_argument('--num_steps', type=int, default=None, help='Number of training steps')
    parser.add_argument('--dataset_path', type=str, default='data/fineweb-edu_10B', help='Path to dataset')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Number of warmup steps')
    parser.add_argument('--ema_decay', type=float, default=0.996, help='EMA decay rate')
    parser.add_argument('--mask_ratio', type=float, default=0.25, help='Chunk masking ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--use_comet', action='store_true', help='Use Comet ML for logging')
    args = parser.parse_args()

    rank, world_size, local_rank, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    torch.manual_seed(args.seed + rank)

    seq_len = 1024
    if args.num_steps is None:
        tokens_per_step = seq_len * args.batch_size * world_size
        args.num_steps = 10_000_000_000 // tokens_per_step
    if args.warmup_steps is None:
        args.warmup_steps = int(0.03 * args.num_steps)

    if rank == 0:
        print(f'Training configuration:')
        print(f'  World size: {world_size}')
        print(f'  Batch size per GPU: {args.batch_size}')
        print(f'  Total batch size: {args.batch_size * world_size}')
        print(f'  Sequence length: {seq_len}')
        print(f'  Number of steps: {args.num_steps}')
        print(f'  Learning rate: {args.learning_rate}')
        print(f'  Warmup steps: {args.warmup_steps}')
        print(f'  Mask ratio: {args.mask_ratio}')

    experiment = None
    if args.use_comet and rank == 0:
        try:
            import comet_ml
            experiment = comet_ml.Experiment(
                project_name=args.project_name,
                auto_metric_logging=True,
                auto_param_logging=True,
            )
            experiment.log_parameters(vars(args))
        except ImportError:
            print('Comet ML not installed, continuing without logging')

    vocab_size = 65024  # Falcon tokenizer
    chunk_size = 32
    n_chunks = seq_len // chunk_size
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
    pred_config = PredictorConfig(
        max_chunks=64,
        n_layer=6,
        n_head=6,
        n_embd=384,
        n_encoder_embd=n_embd
    )

    chunk_encoder = ChunkEncoder(chunk_enc_config).to(device)
    context_encoder = Encoder(enc_config).to(device)
    target_encoder = copy.deepcopy(context_encoder)  # Start with same weights
    predictor = Predictor(pred_config).to(device)

    if args.compile:
        chunk_encoder = torch.compile(chunk_encoder)
        context_encoder = torch.compile(context_encoder)
        target_encoder = torch.compile(target_encoder)
        predictor = torch.compile(predictor)

    if is_distributed:
        chunk_encoder = DDP(chunk_encoder, device_ids=[local_rank])
        context_encoder = DDP(context_encoder, device_ids=[local_rank])
        predictor = DDP(predictor, device_ids=[local_rank])
        # Note: target_encoder is not wrapped in DDP as it doesn't need gradients

    def get_module(model):
        """Get the underlying module from DDP wrapper if needed."""
        return model.module if hasattr(model, 'module') else model

    # See https://arxiv.org/abs/2507.07101 for beta2 scaling.
    beta2 = (0.95)**(1.0/(512/args.batch_size))
    chunk_enc_optimizers = get_module(chunk_encoder).configure_optimizers(
        wd=0.1, adam_lr=args.learning_rate, adam_betas=(0.9, beta2)
    )
    context_enc_optimizer = get_module(context_encoder).configure_optimizers(
        wd=0.1, adam_lr=args.learning_rate, adam_betas=(0.9, beta2)
    )
    predictor_optimizer = get_module(predictor).configure_optimizers(
        wd=0.1, adam_lr=args.learning_rate, adam_betas=(0.9, beta2)
    )

    optimizers = [chunk_enc_optimizers, context_enc_optimizer, predictor_optimizer]

    schedulers = []
    for optimizer in optimizers:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.num_steps - args.warmup_steps,
        )
        schedulers.append(scheduler)

    # EMA decay for target encoder updates

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

    for step in range(args.num_steps):
        batch_start_time = time.perf_counter()
        batch = train_loader.next_batch()

        # Apply warmup or cosine schedule
        if step < args.warmup_steps:
            # Linear warmup
            lr_scale = (step + 1) / args.warmup_steps
            for opt_idx, optimizer in enumerate(optimizers):
                for group_idx, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = initial_lrs[opt_idx][group_idx] * lr_scale

        loss = train_step(chunk_encoder, context_encoder, target_encoder, predictor, batch, optimizers)
        batch_time = time.perf_counter() - batch_start_time

        # Update target encoder with EMA of context encoder
        # Handle DDP wrapper if present
        with torch.no_grad():
            context_model = get_module(context_encoder)
            for target_param, context_param in zip(target_encoder.parameters(), context_model.parameters()):
                target_param.data.mul_(args.ema_decay).add_(context_param.data, alpha=1 - args.ema_decay)

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

        if step % args.val_loss_every == 0 and step > 0:
            val_loss = validate(
                chunk_encoder, context_encoder, target_encoder, predictor,
                val_loader
            )

            if rank == 0:
                print(f'Step {step} | Validation Loss: {val_loss:.4f}')
                if experiment:
                    experiment.log_metric('val_loss', val_loss, step=step)

        if rank == 0 and step % 5000 == 0 and step > 0:
            checkpoint = {
                'step': step,
                'chunk_encoder': get_module(chunk_encoder).state_dict(),
                'context_encoder': get_module(context_encoder).state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'predictor': get_module(predictor).state_dict(),
                'optimizers': [opt.state_dict() for opt in optimizers],
                'schedulers': [sched.state_dict() for sched in schedulers],
                'ema_decay': args.ema_decay,
                'args': args,
            }
            torch.save(checkpoint, f'checkpoint_step_{step}.pt')
            print(f'Saved checkpoint at step {step}')

    if rank == 0:
        checkpoint = {
            'step': args.num_steps,
            'chunk_encoder': get_module(chunk_encoder).state_dict(),
            'context_encoder': get_module(context_encoder).state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'predictor': get_module(predictor).state_dict(),
            'optimizers': [opt.state_dict() for opt in optimizers],
            'schedulers': [sched.state_dict() for sched in schedulers],
            'args': args,
        }
        torch.save(checkpoint, 'checkpoint_final.pt')
        print('Training complete! Saved final checkpoint.')

    cleanup_distributed()
    if experiment:
        experiment.end()

if __name__ == '__main__':
    main()