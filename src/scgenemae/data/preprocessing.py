import pytest
import torch
from scgenemae.models.mae import SingleCellMAE

def test_single_cell_mae_forward():
    batch_size = 4
    num_genes = 100
    d_model = 64
    
    model = SingleCellMAE(
        num_genes=num_genes,
        d_model=d_model,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128
    )
    
    # Mock inputs
    masked_expression = torch.randn(batch_size, num_genes)
    mask = torch.rand(batch_size, num_genes) < 0.3
    
    # Forward pass
    reconstructed = model(masked_expression, mask)
    
    # Check output shape matches input shape (batch_size, num_genes)
    assert reconstructed.shape == (batch_size, num_genes)
