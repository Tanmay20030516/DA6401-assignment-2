"""Unified multi-task model"""

import os

import torch
import torch.nn as nn

from .layers import CustomDropout



# Shared encoder (mirrors VGG11Encoder exactly)


class _SharedEncoder(nn.Module):
    """VGG11-style encoder shared across all three task heads."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)

        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d(2, stride=2)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = {}

        x = self.relu(self.bn1(self.conv1(x)))
        if return_features:
            features["skip1"] = x
        x = self.maxpool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        if return_features:
            features["skip2"] = x
        x = self.maxpool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        if return_features:
            features["skip3"] = x
        x = self.maxpool3(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        if return_features:
            features["skip4"] = x
        x = self.maxpool4(x)

        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        if return_features:
            features["skip5"] = x
        x = self.maxpool5(x)

        if return_features:
            return x, features
        return x



# Classification head (mirrors VGG11Classifier minus encoder)


class _ClassificationHead(nn.Module):
    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = CustomDropout(p=dropout_p)
        self.gap = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(512 * 5 * 5, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x).flatten(1)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        return self.out(x)



# Localization head (mirrors VGG11Localizer minus encoder)


class _LocalizationHead(nn.Module):
    def __init__(self, dropout_p: float = 0.5, image_size: int = 224):
        super().__init__()
        self.image_size = image_size
        self.relu = nn.ReLU(inplace=True)
        self.dropout = CustomDropout(p=dropout_p)
        self.gap = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(512 * 5 * 5, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x).flatten(1)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        return torch.sigmoid(self.out(x)) * self.image_size



# Segmentation bottleneck + decoder (mirrors VGG11UNet minus encoder)


class _SegmentationDecoder(nn.Module):
    def __init__(self, num_classes: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = CustomDropout(dropout_p)

        # bottleneck
        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        # decoder stage 1 — mirrors skip5
        self.trconv1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv11_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(512)
        self.conv11_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)

        # decoder stage 2 — mirrors skip4
        self.trconv2 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv22_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv22_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)

        # decoder stage 3 — mirrors skip3
        self.trconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv33_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.conv33_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(256)

        # decoder stage 4 — mirrors skip2
        self.trconv4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv44 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        # decoder stage 5 — mirrors skip1
        self.trconv5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv55 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, skip_features: dict) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.trconv1(x)
        x = torch.cat([x, skip_features["skip5"]], dim=1)
        x = self.relu(self.bn3_1(self.conv11_1(x)))
        x = self.relu(self.bn3_2(self.conv11_2(x)))

        x = self.trconv2(x)
        x = torch.cat([x, skip_features["skip4"]], dim=1)
        x = self.relu(self.bn4_1(self.conv22_1(x)))
        x = self.relu(self.bn4_2(self.conv22_2(x)))

        x = self.trconv3(x)
        x = torch.cat([x, skip_features["skip3"]], dim=1)
        x = self.relu(self.bn5_1(self.conv33_1(x)))
        x = self.relu(self.bn5_2(self.conv33_2(x)))

        x = self.trconv4(x)
        x = torch.cat([x, skip_features["skip2"]], dim=1)
        x = self.relu(self.bn6(self.conv44(x)))

        x = self.trconv5(x)
        x = torch.cat([x, skip_features["skip1"]], dim=1)
        x = self.relu(self.bn7(self.conv55(x)))

        x = self.dropout(x)
        return self.out(x)



# Unified multi-task model


class MultiTaskPerceptionModel(nn.Module):
    """
    Shared-backbone multi-task model.

    The encoder is physically constructed once and shared across all three
    task heads. Weights are surgically loaded from the individual task
    checkpoints:
      - encoder weights      <- backbone_source checkpoint  (default: "segmentation")
      - classification head  <- classifier.pth
      - localization head    <- localizer.pth
      - segmentation decoder <- unet.pth
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
        dropout_p: float = 0.4,
        backbone_source: str = "segmentation",  # "classification" | "localization" | "segmentation"
    ):
        super().__init__()

        # optionally download checkpoints from Google Drive

        # classifier.pth = https://drive.google.com/file/d/13ImU_U6x4MdUmt8m4wCO91MQH1sb0POA/view?usp=sharing
        # localizer.pth  = https://drive.google.com/file/d/1odlFguFatDi76B-DvGjWt6aRaEBwr4zT/view?usp=sharing
        # unet.pth       = https://drive.google.com/file/d/1KUz3zPi1-zn7uxjMhMGrUcS5U1qFWWTV/view?usp=sharing

        def _maybe_download(file_id: str, output: str) -> None:
            if not os.path.exists(output):
                import gdown
                gdown.download(id=file_id, output=output, quiet=False)
            else:
                print(f"[checkpoint] using existing: {output}")

        _maybe_download("13ImU_U6x4MdUmt8m4wCO91MQH1sb0POA", classifier_path)
        _maybe_download("1odlFguFatDi76B-DvGjWt6aRaEBwr4zT", localizer_path)
        _maybe_download("1KUz3zPi1-zn7uxjMhMGrUcS5U1qFWWTV", unet_path)


        # build the shared skeleton
        self.encoder     = _SharedEncoder(in_channels=in_channels)
        self.cls_head    = _ClassificationHead(num_classes=num_breeds, dropout_p=dropout_p)
        self.loc_head    = _LocalizationHead(dropout_p=dropout_p)
        self.seg_decoder = _SegmentationDecoder(num_classes=seg_classes, dropout_p=dropout_p)


        # load weights from individual checkpoints

        cls_sd = torch.load(classifier_path, map_location="cpu", weights_only=True)
        loc_sd = torch.load(localizer_path,  map_location="cpu", weights_only=True)
        seg_sd = torch.load(unet_path,       map_location="cpu", weights_only=True)

        _source_map = {
            "classification": cls_sd,
            "localization":   loc_sd,
            "segmentation":   seg_sd,
        }
        if backbone_source not in _source_map:
            raise ValueError(
                f"backbone_source must be one of {list(_source_map)}; got '{backbone_source}'"
            )

        self._load_encoder(self.encoder, _source_map[backbone_source])
        self._load_head(self.cls_head, cls_sd)
        self._load_head(self.loc_head, loc_sd)
        self._load_seg_decoder(self.seg_decoder, seg_sd)


    # weight-loading helpers


    @staticmethod
    def _load_encoder(encoder: _SharedEncoder, state_dict: dict) -> None:
        enc_sd = {
            k[len("encoder."):]: v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        missing, unexpected = encoder.load_state_dict(enc_sd, strict=True)
        if missing:
            print(f"[multitask] encoder — missing keys: {missing}")
        if unexpected:
            print(f"[multitask] encoder — unexpected keys: {unexpected}")

    @staticmethod
    def _load_head(head: nn.Module, state_dict: dict) -> None:
        head_sd = {k: v for k, v in state_dict.items() if not k.startswith("encoder.")}
        missing, unexpected = head.load_state_dict(head_sd, strict=True)
        if missing:
            print(f"[multitask] head — missing keys: {missing}")
        if unexpected:
            print(f"[multitask] head — unexpected keys: {unexpected}")

    @staticmethod
    def _load_seg_decoder(decoder: _SegmentationDecoder, state_dict: dict) -> None:
        dec_sd = {k: v for k, v in state_dict.items() if not k.startswith("encoder.")}
        missing, unexpected = decoder.load_state_dict(dec_sd, strict=True)
        if missing:
            print(f"[multitask] seg decoder — missing keys: {missing}")
        if unexpected:
            print(f"[multitask] seg decoder — unexpected keys: {unexpected}")


    # forward


    def forward(self, x: torch.Tensor) -> dict:
        """Single forward pass through the shared encoder and all three heads.

        Args:
            x: [B, in_channels, H, W]

        Returns:
            dict with keys:
              'classification' — [B, num_breeds] logits
              'localization'   — [B, 4] bbox in (cx, cy, w, h) pixel space
              'segmentation'   — [B, seg_classes, H, W] logits
        """
        bottleneck, skip_features = self.encoder(x, return_features=True)

        class_logits = self.cls_head(bottleneck)
        bbox_coords  = self.loc_head(bottleneck)
        seg_logits   = self.seg_decoder(bottleneck, skip_features)

        return {
            "classification": class_logits,
            "localization":   bbox_coords,
            "segmentation":   seg_logits,
        }