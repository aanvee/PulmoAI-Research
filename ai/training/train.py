import os
import torch
import torch.optim as optim
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Device :", DEVICE)
print("=" * 60)


# ==========================================================
# Dataset
# ==========================================================

TRAIN_CSV = "ai/data/processed/training_dataset.csv"

train_loader = create_dataloaders(
    train_csv=TRAIN_CSV,
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