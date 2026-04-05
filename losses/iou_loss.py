"""Custom IoU loss"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression."""

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")

    @staticmethod
    def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        x_center, y_center, width, height = boxes.unbind(dim=1)
        half_width = width / 2.0
        half_height = height / 2.0
        return torch.stack(
            (
                x_center - half_width,
                y_center - half_height,
                x_center + half_width,
                y_center + half_height,
            ),
            dim=1,
        )

    def forward(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        pred_xyxy = self.cxcywh_to_xyxy(pred_boxes)
        target_xyxy = self.cxcywh_to_xyxy(target_boxes)

        top_left = torch.maximum(pred_xyxy[:, :2], target_xyxy[:, :2])
        bottom_right = torch.minimum(pred_xyxy[:, 2:], target_xyxy[:, 2:])
        intersection_wh = (bottom_right - top_left).clamp(min=0.0)
        intersection = intersection_wh[:, 0] * intersection_wh[:, 1]

        pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0.0) * (
            pred_xyxy[:, 3] - pred_xyxy[:, 1]
        ).clamp(min=0.0)
        target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0.0) * (
            target_xyxy[:, 3] - target_xyxy[:, 1]
        ).clamp(min=0.0)
        union = pred_area + target_area - intersection

        iou = intersection / union.clamp(min=self.eps)
        loss = 1.0 - iou

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
