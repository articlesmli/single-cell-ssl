import torch
import numpy as np
from typing import Tuple

def extract_embeddings(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts dense cellular embeddings from the pretrained Transformer encoder.
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            expression = batch["expression"].to(device)
            # Create dummy mask of all False (no masking during evaluation)
            mask = torch.zeros_like(expression, dtype=torch.bool)
            
            # Forward pass through value embedding and encoder
            x = expression.unsqueeze(-1)
            x = model.value_embedding(x)
            encoded = model.encoder(x)
            
            # Mean pool across gene tokens to get a single cell representation vector
            cell_embedding = encoded.mean(dim=1)
            
            embeddings.append(cell_embedding.cpu().numpy())
            if "labels" in batch:
                labels.append(batch["labels"].numpy())
                
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0) if labels else None
    return embeddings, labels
