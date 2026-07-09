import torch.nn as nn


class MultiLabelLoss(nn.Module):
    """
    Multi-label classification loss
    """

    def __init__(self):

        super().__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):

        return self.loss_fn(
            predictions,
            targets
        )


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    import torch

    criterion = MultiLabelLoss()

    predictions = torch.randn(4, 15)

    labels = torch.randint(
        0,
        2,
        (4, 15)
    ).float()

    loss = criterion(
        predictions,
        labels
    )

    print("=" * 50)
    print("Loss :", loss.item())