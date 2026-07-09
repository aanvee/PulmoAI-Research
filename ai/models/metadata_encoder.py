import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    """
    Metadata Encoder

    Input:
        (B, 3)

    Output:
        (B, 128)
    """

    def __init__(self):

        super(MetadataEncoder, self).__init__()

        self.encoder = nn.Sequential(

            nn.Linear(3, 64),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, 128),

            nn.BatchNorm1d(128),

            nn.ReLU()

        )

    def forward(self, x):

        return self.encoder(x)


# ==========================================================
# Test
# ==========================================================

#if __name__ == "__main__":

#   model = MetadataEncoder()

#    dummy = torch.randn(4, 3)

#   output = model(dummy)

#    print("=" * 50)
#    print("Input Shape :", dummy.shape)
#    print("Output Shape:", output.shape)