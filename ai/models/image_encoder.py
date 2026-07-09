import torch
import torch.nn as nn
from torchvision.models import (
    densenet121,
    DenseNet121_Weights
)


class ImageEncoder(nn.Module):
    """
    DenseNet121 based Image Encoder

    Input:
        (B, 3, 224, 224)

    Output:
        (B, 1024)
    """

    def __init__(self, pretrained=True):

        super(ImageEncoder, self).__init__()

        if pretrained:
            weights = DenseNet121_Weights.DEFAULT
        else:
            weights = None

        self.backbone = densenet121(weights=weights)

        # Remove classifier layer
        self.backbone.classifier = nn.Identity()

    def forward(self, x):

        features = self.backbone(x)

        return features
#to be deleted later
###
#if __name__ == "__main__":

#        model = ImageEncoder(pretrained=True)

#        dummy = torch.randn(4, 3, 224, 224)

#        output = model(dummy)

#        print("=" * 50)
#       print("Input Shape :", dummy.shape)
#        print("Output Shape:", output.shape)
#