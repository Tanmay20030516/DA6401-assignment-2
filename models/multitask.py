"""Unified multi-task model"""

import torch
import torch.nn as nn

from .localization import VGG11Localizer
from .classification import VGG11Classifier
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super(MultiTaskPerceptionModel, self).__init__()
        import gdown
        # classifier.pth = https://drive.google.com/file/d/1KpZNYwPLJW20yFWukZrjhxJjg7C5_4lo/view?usp=sharing
        gdown.download(id="1KpZNYwPLJW20yFWukZrjhxJjg7C5_4lo", output=classifier_path, quiet=False)
        gdown.download(id="1KpZNYwPLJW20yFWukZrjhxJjg7C5_4lo", output=localizer_path, quiet=False)
        gdown.download(id="1KpZNYwPLJW20yFWukZrjhxJjg7C5_4lo", output=unet_path, quiet=False)
        self.num_breeds = num_breeds
        self.seg_classes = seg_classes
        self.in_channels = in_channels
        self.classifier = VGG11Classifier(in_channels=in_channels, num_classes=num_breeds, dropout_p=0.5)
        self.localizer = VGG11Localizer(in_channels=in_channels, dropout_p=0.5)
        self.unet = VGG11UNet(in_channels=in_channels, num_classes=seg_classes, dropout_p=0.5)
        self.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu", weights_only=True))
        self.localizer.load_state_dict(torch.load(localizer_path, map_location="cpu", weights_only=True))
        self.unet.load_state_dict(torch.load(unet_path, map_location="cpu", weights_only=True))


    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """

        class_logits = self.classifier(x)
        bbox_coords = self.localizer(x)
        seg_logits = self.unet(x)
        
        return {
            "classification": class_logits,
            "localization": bbox_coords,
            "segmentation": seg_logits,
        }
