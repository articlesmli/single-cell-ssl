import pytest
import torch
import anndata as ad
import numpy as np
from scgenemae.data.dataset import SingleCellDataset

@pytest.fixture
def dummy_h5ad(tmp_path):
    """Creates a small mock AnnData file for testing."""
    file_path = tmp_path / "test_adata.h5ad"
    
    # 50 cells, 100 genes
    X = np.random.poisson(lam=2.0, size=(50, 100)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.write_h5ad(file_path)
    
    return str(file_path)

def test_single_cell_dataset(dummy_h5ad):
    mask_ratio = 0.3
    dataset = SingleCellDataset(dummy_h5ad, mask_ratio=mask_ratio)
    
    assert len(dataset) == 50
    
    sample = dataset[0]
    assert "expression" in sample
    assert "mask" in sample
    assert "masked_expression" in sample
    
    # Check tensor shapes
    assert sample["expression"].shape == (100,)
    assert sample["mask"].shape == (100,)
    assert sample["masked_expression"].shape == (100,)
    
    # Check approximate masking ratio
    masked_count = sample["mask"].sum().item()
    # Allowing variance, but roughly 30% should be masked
    assert 10 <= masked_count <= 50
