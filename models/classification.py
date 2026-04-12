import torch

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(torch.nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead"""

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
            return torch.nn.BatchNorm1d(c) if self.use_bn else torch.nn.Identity()
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = CustomDropout(p=self.dropout_p)
        self.encoder = VGG11Encoder(in_channels=in_channels, use_bn=use_bn)
        # we bring the output to 512 x 7 x 7
        # image resized to 224x224, so the final dims are 512x7x7 (we can used adaptive pooling to avoid hardcoding this)
        # self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc1 = torch.nn.Linear(in_features=512*7*7, out_features=1024)
        self.bn1 = bn1d(1024)
        self.fc2 = torch.nn.Linear(in_features=1024, out_features=256)
        self.bn2 = bn1d(256)
        self.out = torch.nn.Linear(in_features=256, out_features=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification model.
        Returns:
            Classification logits [B, num_classes].
        """
        n, _, _, _ = x.shape
        x = self.encoder(x)

        # x = self.gap(x).flatten(1) # convert b x 512 x 7 x 7 -> b x 25088
        x = x.flatten(1) # convert b x 512 x 7 x 7 -> b x 25088
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.out(x)

        return x
