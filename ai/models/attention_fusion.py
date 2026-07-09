import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    """
    Attention-Based Feature Fusion

    Inputs:
        image_features    -> (B,1024)
        metadata_features -> (B,128)

    Output:
        fused_features -> (B,256)
    """

    def __init__(self):

        super().__init__()

        # Reduce image features
        self.image_projection = nn.Linear(
            1024,
            128
        )

        # Learn attention weights
        self.attention = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                256
            ),

            nn.Sigmoid()

        )

    def forward(
        self,
        image_features,
        metadata_features
    ):

        image_features = self.image_projection(
            image_features
        )

        fused = torch.cat(
            [
                image_features,
                metadata_features
            ],
            dim=1
        )

        weights = self.attention(
            fused
        )

        fused = fused * weights

        return fused

if __name__ == "__main__":

    model = AttentionFusion()

    image = torch.randn(
        4,
        1024
    )

    metadata = torch.randn(
        4,
        128
    )

    output = model(
        image,
        metadata
    )

    print("=" * 50)
    print("Image :", image.shape)
    print("Metadata :", metadata.shape)
    print("Output :", output.shape)