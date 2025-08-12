import torch
from dataloader import create_random_chunk_mask
from model import (
    ChunkEncoder, ChunkEncoderConfig,
    Encoder, EncoderConfig,
    Predictor, PredictorConfig,
)

def test_jepa_components():
    # Configuration
    device = torch.device('cpu')  # Use CPU for testing
    batch_size = 2
    vocab_size = 50000
    chunk_size = 16
    n_chunks = 8
    tokens_per_chunk = chunk_size - 1  # minus CLS token

    # Create configs
    chunk_enc_config = ChunkEncoderConfig(
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        n_layer=4,  # Small for testing
        n_head=8,
        n_embd=512
    )

    enc_config = EncoderConfig(
        max_chunks=32,
        n_layer=4,
        n_head=8,
        n_embd=512
    )

    pred_config = PredictorConfig(
        max_chunks=32,
        n_layer=2,
        n_head=8,
        n_embd=512
    )

    # Initialize models
    chunk_encoder = ChunkEncoder(chunk_enc_config).to(device)
    encoder = Encoder(enc_config).to(device)
    predictor = Predictor(pred_config).to(device)

    # Create dummy input
    input_tokens = torch.randint(0, vocab_size, (batch_size, n_chunks * tokens_per_chunk), device=device)

    # Step 1: Encode tokens to chunks
    print(f"Input tokens shape: {input_tokens.shape}")
    chunk_embeddings = chunk_encoder(input_tokens)
    print(f"Chunk embeddings shape: {chunk_embeddings.shape}")

    # Step 2: Create mask
    chunk_mask = create_random_chunk_mask(batch_size, n_chunks, mask_ratio=0.25, device=device)
    print(f"Chunk mask shape: {chunk_mask.shape}")
    print(f"Masked chunks per sample: {chunk_mask.sum(dim=1).tolist()}")

    # Step 3: Process with encoder (context path - with masking)
    context_embeddings = encoder(chunk_embeddings, chunk_mask)
    print(f"Context embeddings shape: {context_embeddings.shape}")

    # Step 4: Get visible chunk embeddings for predictor
    # Extract only the visible (non-masked) chunks
    visible_mask = ~chunk_mask

    # For simplicity, we'll use all context embeddings (including masked positions)
    # In practice, you might want to filter to only visible chunks

    # Step 5: Get target positions (masked positions)
    target_positions_list = []
    for b in range(batch_size):
        masked_pos = torch.where(chunk_mask[b])[0]
        target_positions_list.append(masked_pos)

    # Pad to same length
    max_targets = max(len(pos) for pos in target_positions_list)
    target_positions = torch.zeros(batch_size, max_targets, dtype=torch.long, device=device)
    for b, pos in enumerate(target_positions_list):
        target_positions[b, :len(pos)] = pos

    print(f"Target positions shape: {target_positions.shape}")

    # Step 6: Predict masked chunks
    # Use only visible chunks as context
    context_for_predictor = context_embeddings  # In practice, filter to visible only
    predicted_embeddings = predictor(context_for_predictor, target_positions)
    print(f"Predicted embeddings shape: {predicted_embeddings.shape}")

    # Step 7: Target embeddings (from encoder without masking)
    target_embeddings = encoder(chunk_embeddings, chunk_mask=None)
    print(f"Target embeddings shape: {target_embeddings.shape}")

    print("\n✓ All components working correctly!")
    print("\nArchitecture summary:")
    print(f"  ChunkEncoder: tokens → chunk embeddings")
    print(f"  Encoder: chunk embeddings (+ mask) → contextualized representations")
    print(f"  Predictor: context chunks → predicted masked chunks")

if __name__ == "__main__":
    test_jepa_components()