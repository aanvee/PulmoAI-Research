import torch
import torch.nn as nn

from ai.models.image_encoder import ImageEncoder
from ai.models.metadata_encoder import MetadataEncoder
from ai.models.attention_fusion import AttentionFusion


class PulmoAIModel(nn.Module):
    """
    Complete Multimodal Lung Disease Detection Model

    Inputs:
        image      -> (B,3,224,224)
        metadata   -> (B,3)

    Output:
        logits     -> (B,15)
    """

    def __init__(self):

        super().__init__()

        self.image_encoder = ImageEncoder()

        self.metadata_encoder = MetadataEncoder()

        self.fusion = AttentionFusion()

        self.classifier = nn.Sequential(

            nn.Linear(256,128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128,15)

        )

    def forward(
        self,
        image,
        metadata
    ):

        image_features = self.image_encoder(image)

        metadata_features = self.metadata_encoder(metadata)

        fused_features = self.fusion(
            image_features,
            metadata_features
        )

        logits = self.classifier(
            fused_features
        )

        return logits


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    model = PulmoAIModel()

    image = torch.randn(
        4,
        3,
        224,
        224
    )

    metadata = torch.randn(
        4,
        3
    )

    output = model(
        image,
        metadata
    )

    print("="*60)

    print("Image Shape      :", image.shape)

    print("Metadata Shape   :", metadata.shape)

    print("Output Shape     :", output.shape)

    print("="*60)