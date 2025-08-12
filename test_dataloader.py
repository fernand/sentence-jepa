import torch
from dataloader import DataLoader

def test_dataloader():
    # Configuration matching the JEPA architecture
    batch_size = 4
    chunk_size = 16  # Including CLS token
    n_chunks = 8
    mask_ratio = 0.25
    device = 'cpu'  # Use CPU for testing
    
    # Create dataloader
    dataloader = DataLoader(
        filename_pattern='data/fineweb_10B/fineweb-edu_val_*.bin',
        B=batch_size,
        chunk_size=chunk_size,
        n_chunks=n_chunks,
        mask_ratio=mask_ratio,
        device=device
    )
    
    print(f"\nDataLoader Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Number of chunks: {n_chunks}")
    print(f"  Tokens per chunk: {dataloader.tokens_per_chunk}")
    print(f"  Total tokens per sample: {dataloader.T}")
    print(f"  Mask ratio: {mask_ratio}")
    
    # Get a batch
    batch = dataloader.next_batch()
    
    print(f"\nBatch contents:")
    print(f"  tokens shape: {batch['tokens'].shape}")
    print(f"  chunk_mask shape: {batch['chunk_mask'].shape}")
    print(f"  target_positions shape: {batch['target_positions'].shape}")
    print(f"  n_chunks: {batch['n_chunks']}")
    
    # Check masking statistics
    masks_per_sample = batch['chunk_mask'].sum(dim=1)
    print(f"\nMasking statistics:")
    print(f"  Masked chunks per sample: {masks_per_sample.tolist()}")
    print(f"  Average mask ratio: {masks_per_sample.float().mean() / n_chunks:.3f}")
    
    # Verify target positions match masked positions
    for i in range(batch_size):
        mask = batch['chunk_mask'][i]
        targets = batch['target_positions'][i]
        masked_positions = torch.where(mask)[0]
        n_masked = masked_positions.shape[0]
        
        # Check that first n_masked positions in targets match masked_positions
        targets_trimmed = targets[:n_masked]
        assert torch.equal(targets_trimmed, masked_positions), \
            f"Sample {i}: Target positions don't match masked positions"
    
    print("\n✓ DataLoader test passed!")
    print("\nThe DataLoader returns everything needed for JEPA training:")
    print("  - tokens: for ChunkEncoder input")
    print("  - chunk_mask: for masking in Encoder")
    print("  - target_positions: for Predictor to know which positions to predict")

if __name__ == "__main__":
    test_dataloader()