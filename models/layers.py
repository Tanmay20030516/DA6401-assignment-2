"""Reusable custom layers"""

import torch


class CustomDropout(torch.nn.Module):
    """Custom Dropout layer."""

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super(CustomDropout, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        if (self.training):  # if used in training mode, we do inverse scaling by multiplying by 1/1-p
            # inverse scaling is done so that expected value of activations match during training and inference
            mask = (
                torch.rand(x.shape, device=x.device) > self.p
            ).float()  # this is the keep mask
            return (mask * x) / (1.0 - self.p)

        return x
