import torch
from model import Rotary

# Test chunked rotary
def test_chunked_rotary():
    batch_size = 2
    n_chunks = 4
    chunk_size = 8
    n_heads = 12
    head_dim = 64
    seq_len = n_chunks * chunk_size
    
    # Create chunked rotary
    rotary_chunked = Rotary(dim=head_dim, max_seq_len=chunk_size, chunked=True)
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    # Apply chunked rotary
    y = rotary_chunked(x)
    
    # Verify output shape
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    
    # Verify that positions repeat across chunks
    # Extract positional encodings by passing a tensor of ones
    ones = torch.ones_like(x)
    encoded = rotary_chunked(ones)
    
    # Check that first chunk pattern repeats
    for chunk_idx in range(1, n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = (chunk_idx + 1) * chunk_size
        
        # Compare with first chunk
        first_chunk = encoded[:, :chunk_size, :, :]
        current_chunk = encoded[:, chunk_start:chunk_end, :, :]
        
        # They should be approximately equal (allowing for numerical precision)
        assert torch.allclose(first_chunk, current_chunk, atol=1e-6), \
            f"Chunk {chunk_idx} doesn't match first chunk"
    
    print("✓ Chunked rotary test passed!")

# Test regular rotary still works
def test_regular_rotary():
    batch_size = 2
    seq_len = 32
    n_heads = 12
    head_dim = 64
    
    # Create regular rotary
    rotary_regular = Rotary(dim=head_dim, max_seq_len=seq_len, chunked=False)
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    # Apply regular rotary
    y = rotary_regular(x)
    
    # Verify output shape
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    
    # Verify that positions are different (not repeating)
    ones = torch.ones_like(x)
    encoded = rotary_regular(ones)
    
    # Check that different positions have different encodings
    pos1 = encoded[:, 0, :, :]
    pos2 = encoded[:, 8, :, :]
    pos3 = encoded[:, 16, :, :]
    
    assert not torch.allclose(pos1, pos2, atol=1e-6), "Positions should have different encodings"
    assert not torch.allclose(pos2, pos3, atol=1e-6), "Positions should have different encodings"
    
    print("✓ Regular rotary test passed!")

if __name__ == "__main__":
    test_chunked_rotary()
    test_regular_rotary()
    print("\nAll tests passed! The chunked rotary implementation is working correctly.")