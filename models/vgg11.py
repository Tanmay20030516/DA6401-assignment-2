"""VGG11 encoder"""

from typing import Dict, Tuple, Union

import torch


class VGG11Encoder(torch.nn.Module):
    """VGG11-style encoder with optional intermediate feature returns."""

    def __init__(
        self,
        in_channels: int = 3,
        use_bn: bool = True,
    ):
        super(VGG11Encoder, self).__init__()
        self.in_channels = in_channels
        self.use_bn = use_bn

        def bn2d(c):
            return torch.nn.BatchNorm2d(c) if self.use_bn else torch.nn.Identity()

        self.relu = torch.nn.ReLU(inplace=True)

        self.conv1 = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = bn2d(64)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn2 = bn2d(128)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = bn2d(256)
        self.conv4 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = bn2d(256)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = bn2d(512)
        self.conv6 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.bn6 = bn2d(512)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.bn7 = bn2d(512)
        self.conv8 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = bn2d(512)
        self.maxpool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            bottleneck (if return_features=False) or (bottleneck, features_dict) tuple (if return_features=True).
        """
        features = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if return_features:
            features["skip1"] = x
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if return_features:
            features["skip2"] = x
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        if return_features:
            features["skip3"] = x
        x = self.maxpool3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        if return_features:
            features["skip4"] = x
        x = self.maxpool4(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        if return_features:
            features["skip5"] = x
        x = self.maxpool5(x)  # our bottleneck features

        if return_features:
            return (x, features)
        return x

