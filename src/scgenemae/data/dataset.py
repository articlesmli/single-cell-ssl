import torch
from torch.utils.data import Dataset
import numpy as np
import anndata as ad
from typing import Tuple, Optional

class SingleCellDataset(Dataset):
    """
    PyTorch Dataset for single-cell RNA-seq data.
    Handles loading AnnData (.h5ad) matrices, normalization, 
    and generating random gene masks for Self-Supervised Learning (MAE).
    """
    def __init__(
        self, 
        adata_path: str, 
        mask_ratio: float = 0.4, 
        is_logged: bool = True
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        # Load AnnData file (lazy load or in-memory)
        self.adata = ad.read_h5ad(adata_path)
        
        # Ensure expression matrix is dense numpy array for tensor conversion
        # In production, sparse matrices (scipy.sparse) can be used for memory efficiency
        if hasattr(self.adata.X, "toarray"):
            self.expression_matrix = self.adata.X.toarray()
        else:
            self.expression_matrix = np.array(self.adata.X)
            
        self.num_cells, self.num_genes = self.expression_matrix.shape

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dictionary containing:
        - 'expression': Original gene expression vector for the cell
        - 'mask': Boolean mask indicating which genes are masked (True) vs visible (False)
        - 'masked_expression': Expression vector with masked positions zeroed/replaced
        """
        # Extract single cell profile (1D array of gene expressions)
        cell_expression = torch.tensor(self.expression_matrix[idx], dtype=torch.float32)
        
        # Generate random mask for self-supervised learning (MAE style)
        # True = masked (to be predicted), False = visible to encoder
        mask = torch.rand(self.num_genes) < self.mask_ratio
        
        # Create the corrupted/masked input for the encoder
        masked_expression = cell_expression.clone()
        masked_expression[mask] = 0.0  # Zero out masked gene tokens
        
        return {
            "expression": cell_expression,
            "mask": mask,
            "masked_expression": masked_expression
        }
