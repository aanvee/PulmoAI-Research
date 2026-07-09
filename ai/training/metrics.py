import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class MetricsCalculator:
    """
    Computes evaluation metrics for multi-label classification.
    """

    def __init__(self, threshold=0.5):

        self.threshold = threshold

    def calculate(self, logits, targets):

        # Convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        # Convert probabilities to binary predictions
        predictions = (
            probabilities >= self.threshold
        ).float()

        y_true = targets.cpu().numpy()

        y_pred = predictions.cpu().numpy()

        y_prob = probabilities.cpu().numpy()

        metrics = {}

        metrics["accuracy"] = accuracy_score(
            y_true,
            y_pred
        )

        metrics["precision"] = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        metrics["recall"] = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        metrics["f1_score"] = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        try:

            metrics["roc_auc"] = roc_auc_score(
                y_true,
                y_prob,
                average="macro"
            )

        except ValueError:

            metrics["roc_auc"] = 0.0

        return metrics


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    calculator = MetricsCalculator()

    logits = torch.randn(8, 15)

    labels = torch.randint(
        0,
        2,
        (8, 15)
    ).float()

    results = calculator.calculate(
        logits,
        labels
    )

    print("=" * 50)

    for key, value in results.items():

        print(f"{key:12}: {value:.4f}")