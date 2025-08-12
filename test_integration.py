"""
Integration test showing how all JEPA components work with the DataLoader.
"""
import torch
import torch.nn.functional as F
from dataloader import DataLoader
from model import (
    ChunkEncoder, ChunkEncoderConfig,
    Encoder, EncoderConfig,
    Predictor, PredictorConfig
)

def test_jepa_integration():
    # Configuration
    device = torch.device('cpu')  # Use CPU for testing
    batch_size = 2
    vocab_size = 65024  # Falcon tokenizer vocab size
    chunk_size = 16
    n_chunks = 8
    mask_ratio = 0.25
    n_embd = 512  # Embedding dimension
    
    # Model configs
    chunk_enc_config = ChunkEncoderConfig(
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        n_layer=4,
        n_head=8,
        n_embd=n_embd
    )
    
    enc_config = EncoderConfig(
        max_chunks=32,
        n_layer=4,
        n_head=8,
        n_embd=n_embd
    )
    
    pred_config = PredictorConfig(
        max_chunks=32,
        n_layer=2,
        n_head=8,
        n_embd=384,  # Can be smaller
        n_encoder_embd=n_embd  # Output dimension matching encoder
    )
    
    # Initialize models
    chunk_encoder = ChunkEncoder(chunk_enc_config).to(device)
    context_encoder = Encoder(enc_config).to(device)
    target_encoder = Encoder(enc_config).to(device)  # Separate or EMA in practice
    predictor = Predictor(pred_config).to(device)
    
    # Create dataloader
    dataloader = DataLoader(
        filename_pattern='data/fineweb_10B/fineweb-edu_val_*.bin',
        B=batch_size,
        chunk_size=chunk_size,
        n_chunks=n_chunks,
        mask_ratio=mask_ratio,
        device=device
    )
    
    # Get a batch
    batch = dataloader.next_batch()
    tokens = batch['tokens']
    chunk_mask = batch['chunk_mask']
    target_positions = batch['target_positions']
    
    print("=" * 60)
    print("JEPA Training Integration Test")
    print("=" * 60)
    
    # Step 1: Encode tokens to chunks
    print("\n1. ChunkEncoder: tokens → chunk embeddings")
    chunk_embeddings = chunk_encoder(tokens)
    print(f"   Input: {tokens.shape} → Output: {chunk_embeddings.shape}")
    
    # Step 2: Context path (with masking)
    print("\n2. Context Encoder: chunk embeddings + mask → context")
    context_embeddings = context_encoder(chunk_embeddings, chunk_mask)
    print(f"   Input: {chunk_embeddings.shape} + mask {chunk_mask.shape}")
    print(f"   Output: {context_embeddings.shape}")
    
    # Step 3: Target path (no masking)
    print("\n3. Target Encoder: chunk embeddings → targets")
    with torch.no_grad():  # Targets don't get gradients
        target_embeddings = target_encoder(chunk_embeddings, chunk_mask=None)
    print(f"   Input: {chunk_embeddings.shape} (no mask)")
    print(f"   Output: {target_embeddings.shape}")
    
    # Step 4: Predict masked chunks
    print("\n4. Predictor: context → predicted masked chunks")
    predicted_embeddings = predictor(context_embeddings, target_positions)
    print(f"   Context input: {context_embeddings.shape}")
    print(f"   Target positions: {target_positions.shape}")
    print(f"   Predictions: {predicted_embeddings.shape}")
    
    # Step 5: Compute loss (L2 in latent space)
    print("\n5. Loss Computation")
    
    # Gather target embeddings for masked positions
    target_for_loss = []
    for b in range(batch_size):
        n_masked = (target_positions[b] != 0).sum() if target_positions[b].sum() > 0 else 1
        positions = target_positions[b, :n_masked]
        targets = target_embeddings[b, positions]
        target_for_loss.append(targets)
    
    # Stack targets (need to handle variable lengths in practice)
    max_len = max(t.shape[0] for t in target_for_loss)
    padded_targets = torch.zeros(batch_size, max_len, n_embd, device=device)
    padded_predictions = torch.zeros(batch_size, max_len, n_embd, device=device)
    
    for b in range(batch_size):
        n = target_for_loss[b].shape[0]
        padded_targets[b, :n] = target_for_loss[b]
        padded_predictions[b, :n] = predicted_embeddings[b, :n]
    
    # Compute L2 loss
    loss = F.mse_loss(padded_predictions, padded_targets)
    print(f"   MSE Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Integration test successful!")
    print("=" * 60)
    print("\nTraining loop would:")
    print("1. Get batch from DataLoader")
    print("2. Forward pass through all components")
    print("3. Compute L2/L1 loss between predictions and targets")
    print("4. Backpropagate through context encoder and predictor")
    print("5. Update target encoder with EMA or stop-gradient")

if __name__ == "__main__":
    test_jepa_integration()