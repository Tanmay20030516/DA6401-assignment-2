"""Localization modules"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super(VGG11Localizer, self).__init__()
        self.in_channels = in_channels
        self.dropout_p = dropout_p

        self.relu = nn.ReLU()
        self.dropout = CustomDropout(p=self.dropout_p)
        self.encoder = VGG11Encoder(in_channels=in_channels)
        # we need to make sure that output is 512 x 7 x 7
        self.fc1 = nn.Linear(in_features=512 * 7 * 7, out_features=1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.out = nn.Linear(in_features=512, out_features=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.out(x)
        return x
