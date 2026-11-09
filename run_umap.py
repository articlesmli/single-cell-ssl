import os
import torch
import scanpy as sc
import matplotlib.pyplot as plt
import mlflow
from scgenemae.utils.mlflow_logger import MLflowTracker
from scgenemae.models.encoder import MAEEncoder

def evaluate_and_log_embeddings(adata, model, tracker: MLflowTracker):
    """Generates UMAP clusters and logs the visualization plot as an MLflow artifact."""
    
    # 1. Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    fig_path = "figures/umap_leiden_clusters.png"

    # 2. Run Scanpy neighborhood graph, UMAP, and Leiden clustering
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8)

    # 3. Plot and save the figure locally
    fig, ax = plt.subplots(figsize=(8, 8))
    sc.pl.umap(adata, color="leiden", ax=ax, show=False, title="MAE Latent Space - Leiden Clusters")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # 4. Log the plot as an artifact in MLflow
    tracker.log_artifact(fig_path)
    print(f"Logged UMAP figure artifact to MLflow from: {fig_path}")

def main():
    # 1. Initialize MLflow tracker pointing to your local tracking server
    tracker = MLflowTracker(
        experiment_name="single-cell-mae-evaluation", 
        tracking_uri="http://localhost:5000"
    )

    # 2. Load your pretrained model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAEEncoder(num_features=1000, d_model=128)
    checkpoint_path = "checkpoints/mae_encoder_best.pt"
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # 3. Load your single-cell AnnData dataset (.h5ad)
    adata = sc.read_h5ad("data/processed/single_cell_subset.h5ad")

    # 4. Start MLflow run and execute evaluation & artifact logging
    with tracker.start_run(run_name="umap-leiden-evaluation-run"):
        tracker.log_hyperparameters({
            "checkpoint": checkpoint_path,
            "n_neighbors": 15,
            "leiden_resolution": 0.8
        })

        # Run the embedding evaluation and log the figure to MLflow
        evaluate_and_log_embeddings(adata, model, tracker)
        
        print("Evaluation completed and artifacts successfully logged to MLflow.")

if __name__ == "__main__":
    main()