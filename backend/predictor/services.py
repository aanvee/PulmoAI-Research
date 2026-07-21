import os
import tempfile
import joblib
import torch
from django.conf import settings
from ai.inference.predict import predict,DISEASES,get_model
from ai.explainability.gradcam import GradCAM
from ai.explainability.shap import MetadataSHAP
from ai.models.multimodal_model import PulmoAIModel
from ai.data.preprocessing.transforms import valid_transform

from PIL import Image


# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = PROJECT_ROOT / "ai" / "checkpoints" / "best_model.pth"

SCALER_PATH = PROJECT_ROOT / "ai" / "data" / "reports" / "age_scaler.pkl"

age_scaler = joblib.load(SCALER_PATH)
# ==========================================================
# Utility Functions
# ==========================================================
def normalize_age(age):
    return float(
        age_scaler.transform([[age]])[0][0]
    )
def encode_gender(gender):

    return 1.0 if gender.upper() == "M" else 0.0


def encode_view_position(view_position):

    return 1.0 if view_position.upper() == "PA" else 0.0


# ==========================================================
# Prediction Service
# ==========================================================

def predict_service(
    image_file,
    age,
    gender,
    view_position
):

    metadata = [
        encode_gender(gender),
        encode_view_position(view_position),
        normalize_age(age)
    ]

    results = predict(
        image_file,
        metadata
    )

    # Sort diseases by probability (highest first)
    top_predictions = sorted(
        [
            {
                "disease": disease,
                "probability": round(info["probability"]*100,2)
            }
            for disease, info in results.items()
        ],
        key=lambda x: x["probability"],
        reverse=True
    )[:3]

    return {
        "predictions": results,
        "top_predictions": top_predictions
    }


# ==========================================================
# Grad-CAM Service
# ==========================================================

def gradcam_service(
    image_file,
    age,
    gender,
    view_position,
    target_class
):

    if target_class not in DISEASES:
        raise ValueError(
            f"Invalid disease name: {target_class}"
        )

    target_index = DISEASES.index(target_class)

    metadata = torch.tensor(

        [[

            encode_gender(gender),

            encode_view_position(view_position),

            normalize_age(age)

        ]],

        dtype=torch.float32

    ).to(DEVICE)

    model = get_model()

    image = Image.open(

        image_file

    ).convert("RGB")

    image_tensor = valid_transform(

        image

    ).unsqueeze(0).to(DEVICE)

    target_layer = model.image_encoder.backbone.features

    gradcam = GradCAM(

        model,

        target_layer

    )
    heatmap = gradcam.generate_heatmap(

        image_tensor,

        metadata,

        target_index

    )
    gradcam_dir = os.path.join(
        settings.MEDIA_ROOT,
        "gradcam"
    )

    os.makedirs(gradcam_dir,exist_ok=True)

    output_path = os.path.join(

        gradcam_dir,

        "gradcam_output.png"

    )

    overlay = gradcam.overlay_heatmap(

        image_file,

        heatmap

    )

    gradcam.save_visualization(

        overlay,

        output_path

    )

    gradcam.remove_hooks()

    return "/media/gradcam/gradcam_output.png"


# ==========================================================
# SHAP Service
# ==========================================================

def shap_service(
    age,
    gender,
    view_position
):

    explainer = MetadataSHAP()

    background = [

        [1, 0, 45],

        [0, 1, 60],

        [1, 1, 35],

        [0, 0, 50]

    ]

    explainer.create_explainer(

        background

    )

    sample = [[

        encode_gender(gender),

        encode_view_position(view_position),

        normalize_age(age)

    ]]

    shap_values = explainer.compute_shap_values(

        sample

    )

    shap_dir = os.path.join(
        settings.MEDIA_ROOT,
        "shap"
    )

    os.makedirs(shap_dir, exist_ok=True)

    output_path = os.path.join(

        shap_dir,

        "shap_summary.png"

    )

    explainer.plot_feature_importance(

        shap_values,

        feature_names=[

            "Gender",

            "View Position",

            "Age"

        ],

        save_path=output_path

    )

    return "/media/shap/shap_summary.png"