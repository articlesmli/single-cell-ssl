import torch
from pathlib import Path
from typing import Dict, Any

def save_checkpoint(
    state: Dict[str, Any], 
    filepath: str = "checkpoints/checkpoint.pt"
) -> None:
    """Saves model state and optimizer states to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint successfully saved to {path}")

def load_checkpoint(
    filepath: str, 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer = None
) -> Dict[str, Any]:
    """Loads a saved checkpoint into the model and optional optimizer."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
        
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    print(f"Checkpoint loaded from {path}")
    return checkpoint
