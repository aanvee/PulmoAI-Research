import os

import torch
from PIL import Image

from ai.models.multimodal_model import PulmoAIModel
from ai.data.preprocessing.transforms import valid_transform


# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = "ai/checkpoints/best_model.pth"

THRESHOLD = 0.5


# ==========================================================
# Disease Labels
# ==========================================================

DISEASES = [
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


# ==========================================================
# Load Model
# ==========================================================

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found : {CHECKPOINT}"
    )

model = PulmoAIModel().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ==========================================================
# Prediction Function
# ==========================================================

def predict(
    image_path,
    metadata
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = valid_transform(
        image
    ).unsqueeze(0)

    image = image.to(DEVICE)

    metadata = torch.tensor(
        metadata,
        dtype=torch.float32
    ).unsqueeze(0)

    metadata = metadata.to(DEVICE)

    with torch.no_grad():

        outputs = model(
            image,
            metadata
        )

        probabilities = torch.sigmoid(
            outputs
        ).squeeze(0)

    results = {}

    for disease, probability in zip(
        DISEASES,
        probabilities
    ):

        results[disease] = {
            "probability": float(probability),
            "prediction": bool(
                probability >= THRESHOLD
            )
        }

    return results