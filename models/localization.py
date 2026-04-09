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
        # self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # to avoid parameter explosion (we get a 512 dim vector)
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format
            in original image pixel space (not normalized values).
        """
        x = self.encoder(x)
        # x = self.gap(x).flatten(1) # convert b x 512 x 1 x 1 -> b x 512)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.out(x)
        return x
