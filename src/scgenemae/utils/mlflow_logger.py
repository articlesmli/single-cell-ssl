import os
import mlflow
import mlflow.pytorch
import torch

class MLflowTracker:
    def __init__(self, experiment_name="single-cell-mae-pretrain", tracking_uri="http://localhost:5000"):
        """Initializes MLflow tracking URI and experiment namespace."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.active_run = None

    def start_run(self, run_name=None):
        """Starts a new MLflow run block."""
        self.active_run = mlflow.start_run(run_name=run_name)
        return self.active_run

    def log_hyperparameters(self, params: dict):
        """Logs configuration settings, model shapes, and training hyperparameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int):
        """Logs training/validation losses or evaluation metrics at a specific epoch/step."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str):
        """Logs custom non-model files (e.g., UMAP figures, evaluation plots)."""
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path)

    def log_model_checkpoint(self, model: torch.nn.Module, artifact_path: str = "model_checkpoint"):
        """Saves and logs the PyTorch model state and architecture inside MLflow."""
        mlflow.pytorch.log_model(model, artifact_path)

    def end_run(self):
        """Ends the active MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
