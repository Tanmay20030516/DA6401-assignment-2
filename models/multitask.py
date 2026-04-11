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
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
        dropout_p: float = 0.5,
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
        import os
        # classifier.pth = https://drive.google.com/file/d/120wlTu9B6m_5TmN9dQC2_qZF0DNIIEDY/view?usp=sharing
        # localizer.pth = https://drive.google.com/file/d/1clMC6BqRpcngseWbx4RO4jaXhM4uFNl0/view?usp=sharing
        # unet.pth = https://drive.google.com/file/d/1DWF7iQuoa9KFXTvI_wC4L5H8v6gN2BxB/view?usp=sharing
        def _maybe_download(file_id: str, output: str):
            # avoid overwriting existing checkpoints if they are already present (e.g. from a previous run or manual download)
            if not os.path.exists(output):
                gdown.download(id=file_id, output=output, quiet=False)
            else:
                print(f"[checkpoint] using existing: {output}")
        _maybe_download(file_id="120wlTu9B6m_5TmN9dQC2_qZF0DNIIEDY", output=classifier_path)
        _maybe_download(file_id="1clMC6BqRpcngseWbx4RO4jaXhM4uFNl0", output=localizer_path)
        _maybe_download(file_id="1DWF7iQuoa9KFXTvI_wC4L5H8v6gN2BxB", output=unet_path)
        self.num_breeds = num_breeds
        self.seg_classes = seg_classes
        self.in_channels = in_channels
        self.classifier = VGG11Classifier(in_channels=in_channels, num_classes=num_breeds, dropout_p=dropout_p)
        self.localizer = VGG11Localizer(in_channels=in_channels, dropout_p=dropout_p)
        self.unet = VGG11UNet(in_channels=in_channels, num_classes=seg_classes, dropout_p=dropout_p)
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.localizer.load_state_dict(torch.load(localizer_path))
        self.unet.load_state_dict(torch.load(unet_path))


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
        seg_logits = seg_logits[:, [1, 0, 2], :, :] # because we did a remapping in pets_dataset.py
        
        return {
            "classification": class_logits,
            "localization": bbox_coords,
            "segmentation": seg_logits,
        }
