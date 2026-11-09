import torch
import torch.nn as nn
from typing import Dict

class SingleCellMAE(nn.Module):
    """
    Masked Autoencoder (MAE) for Single-Cell RNA-seq using a Transformer architecture.
    """
    def __init__(
        self,
        num_genes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_genes = num_genes
        self.d_model = d_model
        
        # 1. Embedding layer: maps continuous gene expression values to dense vectors
        self.value_embedding = nn.Linear(1, d_model)
        
        # Learnable [MASK] token representation
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Projection head to map back to original gene expression space
        self.prediction_head = nn.Linear(d_model, 1)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, masked_expression: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Single-Cell MAE.
        
        Args:
            masked_expression: Tensor of shape (batch_size, num_genes) with 0s at masked spots
            mask: Boolean tensor of shape (batch_size, num_genes) where True = masked
            
        Returns:
            reconstructed_expression: Tensor of shape (batch_size, num_genes)
        >"""
        batch_size, num_genes = masked_expression.shape
        
        # Reshape to (batch_size, num_genes, 1) for linear embedding
        x = masked_expression.unsqueeze(-1)
        x = self.value_embedding(x)  # Shape: (batch_size, num_genes, d_model)
        
        # Replace masked positions with learnable mask tokens
        # Expand mask_token to match batch size and masked indices
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.d_model)
        x = torch.where(mask_expanded, self.mask_token.expand(batch_size, num_genes, -1), x)
        
        # Pass through Encoder
        encoded = self.encoder(x)  # Shape: (batch_size, num_genes, d_model)
        
        # Pass through Decoder (using encoded representation as both memory and target for simplicity)
        decoded = self.decoder(encoded, encoded)
        
        # Project back to scalar gene expression values
        reconstructed = self.prediction_head(decoded).squeeze(-1)  # Shape: (batch_size, num_genes)
        
        return reconstructed
