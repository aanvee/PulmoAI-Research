import os
from pathlib import Path

import shap
import torch
import numpy as np

from ai.models.multimodal_model import PulmoAIModel


# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = PROJECT_ROOT / "ai" / "checkpoints" / "best_model.pth"


# ==========================================================
# SHAP Explainer
# ==========================================================

class MetadataSHAP:

    def __init__(self):

        self.model = PulmoAIModel().to(DEVICE)
        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT}"
            )
            
        checkpoint = torch.load(
            str(CHECKPOINT),
            map_location=DEVICE
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

    # ======================================================
    # Prediction Wrapper
    # ======================================================

    def predict(self, metadata):

        metadata = torch.tensor(
            metadata,
            dtype=torch.float32
        ).to(DEVICE)

        batch_size = metadata.shape[0]

        dummy_image = torch.zeros(
            (
                batch_size,
                3,
                224,
                224
            ),
            dtype=torch.float32
        ).to(DEVICE)

        with torch.no_grad():

            outputs = self.model(
                dummy_image,
                metadata
            )

        probabilities = torch.sigmoid(
            outputs
        )

        return probabilities.cpu().numpy()
    # ======================================================
    # Create SHAP Explainer
    # ======================================================

    def create_explainer(self, background_data):

        background_data = np.asarray(
            background_data,
            dtype=np.float32
        )

        self.explainer = shap.Explainer(
            self.predict,
            shap.maskers.Independent(background_data)
        )

    # ======================================================
    # Compute SHAP Values
    # ======================================================

    def compute_shap_values(
        self,
        metadata
    ):

        if not hasattr(self, "explainer"):

            raise RuntimeError(
                "Create the SHAP explainer first using create_explainer()."
            )

        metadata = np.asarray(metadata)

        shap_values = self.explainer(metadata)

        return shap_values

    # ======================================================
    # Plot Feature Importance
    # ======================================================

    def plot_feature_importance(
    self,
    shap_values,
    feature_names,
    save_path="ai/results/shap_summary.png"):

        import matplotlib.pyplot as plt

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        plt.figure(figsize=(6,4))

        # Explain one disease (Hernia = index 7)
        explanation = shap.Explanation(
            values=shap_values.values[:, :, 7],
            base_values=shap_values.base_values[:, 7],
            data=shap_values.data,
            feature_names=feature_names)

        shap.plots.bar(explanation,show=False)

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"SHAP Summary Plot Saved : {save_path}")
        
    # def plot_feature_importance(
    #     self,
    #     shap_values,
    #     feature_names,
    #     save_path="ai/results/shap_summary.png"
    # ):

    #     import matplotlib.pyplot as plt

    #     os.makedirs(
    #         os.path.dirname(save_path),
    #         exist_ok=True
    #     )

    #     plt.figure(
    #         figsize=(8, 6)
    #     )

    #     shap.summary_plot(
    #         shap_values.values[:, :, 7],
    #         features=shap_values.data,
    #         feature_names=feature_names,
    #         show=False
    #     )

    #     plt.tight_layout()

    #     plt.savefig(
    #         save_path,
    #         dpi=300,
    #         bbox_inches="tight"
    #     )

    #     plt.close()

    #     print(
    #         f"SHAP Summary Plot Saved : {save_path}"
    #     )


# ==========================================================
# Test SHAP
# ==========================================================

if __name__ == "__main__":

    explainer = MetadataSHAP()

    # Background samples (same distribution as training metadata)
    background = np.array([
    [0, 0, 0.18],
    [1, 0, 0.22],
    [0, 1, 0.35],
    [1, 1, 0.41]
    ], dtype=np.float32)

    sample = np.array([
        [0, 0, 0.19]
    ], dtype=np.float32)

    explainer.create_explainer(background)

    

    shap_values = explainer.compute_shap_values(sample)

    print(type(shap_values))
    print("Values :", shap_values.values.shape)
    print("Data :", shap_values.data.shape)
    print("Base :", np.array(shap_values.base_values).shape)

    explainer.plot_feature_importance(
        shap_values,
        feature_names=[
            "Gender",
            "View Position",
            "Age"
        ]
    )

    print("SHAP Test Completed Successfully.")
        