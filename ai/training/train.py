import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from tqdm import tqdm

from ai.configs.config import (
    TRAIN_DATASET,
    VALIDATION_DATASET
)

from ai.data.preprocessing.dataloader import create_dataloaders

from ai.models.multimodal_model import PulmoAIModel

from ai.training.losses import MultiLabelLoss

from ai.training.metrics import MetricsCalculator

# ==========================================================
# Configuration
# ==========================================================

BATCH_SIZE = 16

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EPOCHS = 20

NUM_WORKERS = 0          # Windows Safe

PATIENCE = 5             # Early Stopping

SEED = 42

CHECKPOINT_DIR = "ai/checkpoints"

CHECKPOINT_FILE = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)
LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "last_model.pth"
)
# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)

print("PulmoAI Training")

print("=" * 60)

print(f"PyTorch Version : {torch.__version__}")

if torch.cuda.is_available():

    print(f"CUDA Version    : {torch.version.cuda}")

    print(f"GPU             : {torch.cuda.get_device_name(0)}")

else:

    print("Running on CPU")

print(f"Device          : {DEVICE}")

print("=" * 60)

# ==========================================================
# Data Loaders
# ==========================================================

print("\nLoading Datasets...")

train_loader = create_dataloaders(
    train_csv=TRAIN_DATASET,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

validation_loader = create_dataloaders(
    train_csv=VALIDATION_DATASET,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

print("\nDataset Information")

print("-" * 60)

print(f"Training Samples      : {len(train_loader.dataset)}")

print(f"Validation Samples    : {len(validation_loader.dataset)}")

print(f"Batch Size            : {BATCH_SIZE}")

print(f"Training Batches      : {len(train_loader)}")

print(f"Validation Batches    : {len(validation_loader)}")

print("-" * 60)

# ==========================================================
# Model Initialization
# ==========================================================

print("\nInitializing PulmoAI Model...")

model = PulmoAIModel().to(DEVICE)

criterion = MultiLabelLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

metric_calculator = MetricsCalculator()

# ==========================================================
# Model Summary
# ==========================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nModel Information")

print("-" * 60)

print(f"Model                 : PulmoAIModel")

print(f"Image Encoder         : DenseNet121")

print(f"Metadata Features     : 3")

print(f"Number of Diseases    : 15")

print(f"Loss Function         : BCEWithLogitsLoss")

print(f"Optimizer             : AdamW")

print(f"Learning Rate         : {LEARNING_RATE}")

print(f"Weight Decay          : {WEIGHT_DECAY}")

print(f"Total Parameters      : {total_params:,}")

print(f"Trainable Parameters  : {trainable_params:,}")

print("-" * 60)

# ==========================================================
# Checkpoint Configuration
# ==========================================================

os.makedirs(CHECKPOINT_DIR,exist_ok=True)

start_epoch = 0

best_val_loss = float("inf")
best_epoch = 0
# ==========================================================
# Load Existing Checkpoint (Resume Training)
# ==========================================================

if os.path.exists(CHECKPOINT_FILE):

    print("\nLoading Existing Checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    best_val_loss = checkpoint["best_val_loss"]

    best_epoch = checkpoint.get("best_epoch",start_epoch)

    print("\nCheckpoint Loaded Successfully")

    print("-" * 60)

    print(f"Resume Epoch        : {start_epoch}")

    print(f"Best Validation Loss: {best_val_loss:.4f}")

    print(f"Checkpoint File     : {CHECKPOINT_FILE}")

    print("-" * 60)

else:

    print("\nNo Existing Checkpoint Found.")

    print("Training will start from Epoch 1.")

    print(f"Checkpoint Directory: {CHECKPOINT_DIR}")

# ==========================================================
# Training
# ==========================================================

print("\nStarting Training...\n")

epochs_without_improvement = 0

training_start_time = time.time()

# ==========================================================
# Training History
# ==========================================================

history = {
    "epoch": [],
    "train_loss": [],
    "validation_loss": [],
    "train_accuracy": [],
    "validation_accuracy": [],
    "train_precision": [],
    "validation_precision": [],
    "train_recall": [],
    "validation_recall": [],
    "train_f1": [],
    "validation_f1": [],
    "train_roc_auc": [],
    "validation_roc_auc": []
}

for epoch in range(start_epoch, EPOCHS):

    print("=" * 70)

    print(f"Epoch [{epoch + 1}/{EPOCHS}]")

    print("=" * 70)

    epoch_start_time = time.time()

    # ------------------------------------------------------
    # Training Mode
    # ------------------------------------------------------

    model.train()

    running_train_loss = 0.0

    train_predictions = []

    train_targets = []

    train_progress = tqdm(
    train_loader,
    desc=f"Epoch {epoch + 1}/{EPOCHS}",
    leave=True
)

    for images, metadata, labels in train_progress:

        images = images.to(DEVICE)

        metadata = metadata.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            images,
            metadata
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
        )
        optimizer.step()

        running_train_loss += loss.item()

        train_predictions.append(outputs.detach().cpu())

        train_targets.append(labels.detach().cpu())

        train_progress.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_train_loss / len(train_loader)

    train_predictions = torch.cat(
        train_predictions
    )

    train_targets = torch.cat(
        train_targets
    )

    train_metrics = metric_calculator.calculate(
        train_predictions,
        train_targets
    )
     # ------------------------------------------------------
    # Validation Mode
    # ------------------------------------------------------

    model.eval()

    running_val_loss = 0.0

    val_predictions = []

    val_targets = []

    validation_progress = tqdm(
        validation_loader,
        desc=f"Validation {epoch + 1}/{EPOCHS}",
        leave=True
    )

    with torch.no_grad():

        for images, metadata, labels in validation_progress:

            images = images.to(DEVICE)

            metadata = metadata.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(
                images,
                metadata
            )

            loss = criterion(
                outputs,
                labels
            )

            running_val_loss += loss.item()

            val_predictions.append(
                outputs.detach().cpu()
            )

            val_targets.append(
                labels.detach().cpu()
            )

            validation_progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    validation_loss = running_val_loss / len(validation_loader)

    val_predictions = torch.cat(
        val_predictions
    )

    val_targets = torch.cat(
        val_targets
    )

    validation_metrics = metric_calculator.calculate(
        val_predictions,
        val_targets
    )

    # ------------------------------------------------------
    # Epoch Time
    # ------------------------------------------------------

    epoch_time = time.time() - epoch_start_time

    # ------------------------------------------------------
    # Print Epoch Summary
    # ------------------------------------------------------

    print("\n" + "=" * 70)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] Summary")

    print("=" * 70)

    print(f"Train Loss           : {train_loss:.4f}")

    print(f"Validation Loss      : {validation_loss:.4f}")

    print()

    print("Training Metrics")

    print(f"Accuracy             : {train_metrics['accuracy']:.4f}")

    print(f"Precision            : {train_metrics['precision']:.4f}")

    print(f"Recall               : {train_metrics['recall']:.4f}")

    print(f"F1 Score             : {train_metrics['f1_score']:.4f}")

    print(f"ROC-AUC              : {train_metrics['roc_auc']:.4f}")

    print()

    print("Validation Metrics")

    print(f"Accuracy             : {validation_metrics['accuracy']:.4f}")

    print(f"Precision            : {validation_metrics['precision']:.4f}")

    print(f"Recall               : {validation_metrics['recall']:.4f}")

    print(f"F1 Score             : {validation_metrics['f1_score']:.4f}")

    print(f"ROC-AUC              : {validation_metrics['roc_auc']:.4f}")

    print()

    print(f"Epoch Time           : {epoch_time:.2f} sec")

    print("=" * 70)
    history["epoch"].append(epoch + 1)

    history["train_loss"].append(train_loss)
    history["validation_loss"].append(validation_loss)

    history["train_accuracy"].append(train_metrics["accuracy"])
    history["validation_accuracy"].append(validation_metrics["accuracy"])

    history["train_precision"].append(train_metrics["precision"])
    history["validation_precision"].append(validation_metrics["precision"])

    history["train_recall"].append(train_metrics["recall"])
    history["validation_recall"].append(validation_metrics["recall"])

    history["train_f1"].append(train_metrics["f1_score"])
    history["validation_f1"].append(validation_metrics["f1_score"])

    history["train_roc_auc"].append(train_metrics["roc_auc"])
    history["validation_roc_auc"].append(validation_metrics["roc_auc"])
    # ------------------------------------------------------
    # Save Best Model
    # ------------------------------------------------------

    if validation_loss < best_val_loss:

        best_val_loss = validation_loss
        best_epoch=epoch+1
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            },
            CHECKPOINT_FILE
        )
        

        print("\nBest Model Saved Successfully.\n")

    else:

        epochs_without_improvement += 1

        print(
            f"\nValidation loss did not improve "
            f"({epochs_without_improvement}/{PATIENCE})"
        )
    torch.save(
    {
        "epoch": epoch + 1,
        "best_epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    },
    LAST_CHECKPOINT
    )
    # ------------------------------------------------------
    # Early Stopping
    # ------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly Stopping Triggered.")

        break
    # ==========================================================
# Training Complete
# ==========================================================

total_training_time = time.time() - training_start_time

print("\n" + "=" * 70)

print("Training Completed Successfully")

print("=" * 70)

print(f"Best Validation Loss : {best_val_loss:.4f}")
print(f"Best Epoch           : {best_epoch}")
print(f"Total Training Time  : {total_training_time/60:.2f} minutes")

print(f"Checkpoint Saved At  : {CHECKPOINT_FILE}")
os.makedirs(
    "ai/results",
    exist_ok=True
)

history_df = pd.DataFrame(history)

history_df.to_csv(
    "ai/results/history.csv",
    index=False
)
print("Training History Saved : ai/results/history.csv")
print("=" * 70)

