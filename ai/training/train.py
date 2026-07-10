import os
import torch
import torch.optim as optim
from ai.data.preprocessing.dataloader import create_dataloaders
from ai.models.multimodal_model import PulmoAIModel
from ai.training.losses import MultiLabelLoss
from ai.training.metrics import MetricsCalculator
from ai.configs.config import TRAINING_DATASET

# ==========================================================
# Configuration
# ==========================================================

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Device :", DEVICE)
print("=" * 60)


# ==========================================================
# Dataset
# ==========================================================

train_loader = create_dataloaders(
    train_csv=TRAINING_DATASET,
    batch_size=BATCH_SIZE,
    num_workers=4
)

print("Training Samples :", len(train_loader.dataset))


# ==========================================================
# Model
# ==========================================================

model = PulmoAIModel().to(DEVICE)

criterion = MultiLabelLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

metric_calculator = MetricsCalculator()


# ==========================================================
# Save Directory
# ==========================================================

SAVE_DIR = "ai/checkpoints"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

BEST_MODEL_PATH = os.path.join(
    SAVE_DIR,
    "best_model.pth"
)

print("Checkpoint Folder :", SAVE_DIR)

# ==========================================================
# Training Loop
# ==========================================================

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    for images, metadata, labels in train_loader:

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

        optimizer.step()

        running_loss += loss.item()

        all_predictions.append(outputs.detach().cpu())

        all_labels.append(labels.detach().cpu())

    epoch_loss = running_loss / len(train_loader)

    predictions = torch.cat(all_predictions)

    targets = torch.cat(all_labels)

    metrics = metric_calculator.calculate(
        predictions,
        targets
    )

    print("-" * 60)

    print(f"Loss      : {epoch_loss:.4f}")

    print(f"Accuracy  : {metrics['accuracy']:.4f}")

    print(f"Precision : {metrics['precision']:.4f}")

    print(f"Recall    : {metrics['recall']:.4f}")

    print(f"F1 Score  : {metrics['f1_score']:.4f}")

    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("\nBest model updated.")

print("\nTraining Complete.")
print(f"Best Loss : {best_loss:.4f}")