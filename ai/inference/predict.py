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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = PROJECT_ROOT / "ai" / "checkpoints" / "best_model.pth"

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

"""
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

"""
# ==========================================================
# Load Model
# ==========================================================

_model = None


def get_model():

    global _model

    if _model is None:

        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT}"
            )

        _model = PulmoAIModel().to(DEVICE)

        checkpoint = torch.load(
            CHECKPOINT,
            map_location=DEVICE
        )

        _model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        _model.eval()

    return _model
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
    model = get_model()
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
# ==========================================================
# Test Prediction
# ==========================================================

if __name__ == "__main__":

    IMAGE_PATH = r"D:\archive\images_001\images\00000003_000.png"

    metadata = [
    0,                  # Female
    0,                  # AP View
    0.1937046004842615  # Normalized age
]

    results = predict(
        IMAGE_PATH,
        metadata
    )

    print("\nPrediction Results\n")

    for disease, info in results.items():

        print(
            f"{disease:20s}"
            f" Probability: {info['probability']:.4f}"
            f" Prediction: {info['prediction']}"
        )