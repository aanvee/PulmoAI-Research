import torch
import os
from ai.configs.config import VALIDATION_DATASET
from ai.data.preprocessing.dataloader import create_dataloaders
from ai.models.multimodal_model import PulmoAIModel
from ai.training.losses import MultiLabelLoss
from ai.training.metrics import MetricsCalculator
from tqdm import tqdm
import os
import pandas as pd
import time
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

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )
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

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

criterion = MultiLabelLoss()

metric_calculator = MetricsCalculator()
print("Loaded Best Model")
print(f"Best Epoch : {checkpoint.get('best_epoch', 'N/A')}")
print(f"Best Validation Loss : {checkpoint.get('best_val_loss', 'N/A')}")
# ==========================================================
# Validation Loop
# ==========================================================


start_time = time.time()
running_loss = 0.0

predictions = []

targets = []
metadata_list = []

print("\nRunning Validation...\n")

with torch.no_grad():

    for images, metadata, labels in tqdm(validation_loader,desc="Validation"):

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

        predictions.append(torch.sigmoid(outputs).cpu())

        targets.append(labels.cpu())

        metadata_list.append(metadata.cpu())

# ==========================================================
# Metrics
# ==========================================================

os.makedirs(
    "ai/results",
    exist_ok=True
)

predictions = torch.cat(predictions)
targets = torch.cat(targets)
metadata = torch.cat(metadata_list)

prediction_df = pd.DataFrame(
    predictions.numpy(),
    columns=[
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "No Finding",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax"
    ]
)

prediction_df["gender"] = metadata[:,0].numpy()
prediction_df["view_position"] = metadata[:,1].numpy()
prediction_df["age"] = metadata[:,2].numpy()

prediction_df.to_csv(
    "ai/results/validation_predictions.csv",
    index=False
)

metrics = metric_calculator.calculate(
    predictions,
    targets
)

validation_loss = running_loss / len(validation_loader)
validation_time = time.time() - start_time
print(f"Validation Time : {validation_time:.2f} sec")
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
print("Predictions Saved : ai/results/validation_predictions.csv")