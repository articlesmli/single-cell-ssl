import torch
from scgenemae.utils.mlflow_logger import MLflowTracker

def train_mae_model(model, dataloader, optimizer, criterion, config):
    # 1. Initialize MLflow Tracker (points to local Docker container)
    tracker = MLflowTracker(
        experiment_name="single-cell-mae-pretrain", 
        tracking_uri="http://localhost:5000"
    )

    # 2. Define training configuration/hyperparameters
    hyperparams = {
        "architecture": "TransformerMAE",
        "learning_rate": config.get("lr", 1e-4),
        "batch_size": dataloader.batch_size,
        "mask_ratio": config.get("mask_ratio", 0.15),
        "d_model": config.get("d_model", 128),
        "epochs": config.get("epochs", 10)
    }

    with tracker.start_run(run_name="mae-pretrain-run"):
        tracker.log_hyperparameters(hyperparams)
        
        model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for epoch in range(hyperparams["epochs"]):
            epoch_loss = 0.0

            for batch_idx, (expressions, masks) in enumerate(dataloader):
                expressions = expressions.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                
                # Forward pass & reconstruction loss
                reconstructed = model(expressions, masks)
                loss = criterion(reconstructed, expressions)
                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            
            # 3. Log metrics per epoch
            tracker.log_metrics({"train_loss": avg_epoch_loss}, step=epoch)
            print(f"Epoch [{epoch+1}/{hyperparams['epochs']}] | Loss: {avg_epoch_loss:.4f}")

        # 4. Save final trained model artifact to MLflow
        tracker.log_model_checkpoint(model, artifact_path="mae_encoder_model")
        print("Pretraining complete. Model logged to MLflow successfully.")