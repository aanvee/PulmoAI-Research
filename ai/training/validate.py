import torch

from ai.configs.config import VALIDATION_DATASET
from ai.data.preprocessing.dataloader import create_dataloaders
from ai.models.multimodal_model import PulmoAIModel
from ai.training.losses import MultiLabelLoss
from ai.training.metrics import MetricsCalculator

# ==========================================================
# Configuration
# ==========================================================

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = "ai/checkpoints/best_model.pth"

print("=" * 60)
print("VALIDATION")
print("=" * 60)
print("Device :", DEVICE)

# ==========================================================
# DataLoader
# ==========================================================

validation_loader = create_dataloaders(
    train_csv=VALIDATION_DATASET,
    batch_size=BATCH_SIZE,
    num_workers=0          # Windows safe
)

print("Validation Samples :", len(validation_loader.dataset))

# ==========================================================
# Load Model
# ==========================================================

model = PulmoAIModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model.eval()

criterion = MultiLabelLoss()

metric_calculator = MetricsCalculator()

# ==========================================================
# Validation Loop
# ==========================================================

running_loss = 0.0

predictions = []

targets = []

print("\nRunning Validation...\n")

with torch.no_grad():

    for images, metadata, labels in validation_loader:

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

        running_loss += loss.item()

        predictions.append(
            outputs.cpu()
        )

        targets.append(
            labels.cpu()
        )

# ==========================================================
# Metrics
# ==========================================================

predictions = torch.cat(predictions)

targets = torch.cat(targets)

metrics = metric_calculator.calculate(
    predictions,
    targets
)

validation_loss = running_loss / len(validation_loader)

# ==========================================================
# Results
# ==========================================================

print("=" * 60)
print("Validation Results")
print("=" * 60)

print(f"Loss       : {validation_loss:.4f}")
print(f"Accuracy   : {metrics['accuracy']:.4f}")
print(f"Precision  : {metrics['precision']:.4f}")
print(f"Recall     : {metrics['recall']:.4f}")
print(f"F1 Score   : {metrics['f1_score']:.4f}")
print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")

print("=" * 60)

print("\nValidation Completed Successfully.")