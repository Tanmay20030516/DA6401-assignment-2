"""Classification components"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_bn: bool = True,
    ):
        super(VGG11Classifier, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_p = dropout_p
        self.use_bn = use_bn
        def bn1d(c):
            return nn.BatchNorm1d(c) if self.use_bn else nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = CustomDropout(p=self.dropout_p)
        self.encoder = VGG11Encoder(in_channels=in_channels, use_bn=use_bn)
        # we need to make sure that output is 512 x 7 x 7
        self.fc1 = nn.Linear(in_features=512 * 7 * 7, out_features=4096)
        self.bn1 = bn1d(4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.bn2 = bn1d(4096)
        self.out = nn.Linear(in_features=4096, out_features=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification model.
        Returns:
            Classification logits [B, num_classes].
        """
        n, _, _, _ = x.shape
        x = self.encoder(x)

        x = self.fc1(x.reshape(n, -1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.out(x)

        return x
