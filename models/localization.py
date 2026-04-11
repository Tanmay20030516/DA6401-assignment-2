import torch
# import torch.nn.functional as F

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(torch.nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, image_size: int = 224):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super(VGG11Localizer, self).__init__()
        self.in_channels = in_channels
        self.dropout_p = dropout_p
        self.image_size = image_size

        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = CustomDropout(p=self.dropout_p)
        self.encoder = VGG11Encoder(in_channels=in_channels)
        # we bring the output to 512 x 7 x 7
        # image resized to 224x224, so the final dims are 512x7x7 (we can used adaptive pooling to avoid hardcoding this)
        # self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc1 = torch.nn.Linear(512 * 7 * 7, 1024)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.fc2 = torch.nn.Linear(1024, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.out = torch.nn.Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format
            in original image pixel space (not normalized values).
        """
        x = self.encoder(x)
        # x = self.gap(x).flatten(1) # convert bs x 512 x 5 x 5 -> bs x 12800
        x = x.flatten(1)  # convert b x 512 x 7 x 7 -> b x 25088
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.out(x)
        x = torch.sigmoid(x) * self.image_size  # scale output to image pixel space
        return x
