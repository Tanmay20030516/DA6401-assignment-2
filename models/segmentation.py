import torch

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11UNet(torch.nn.Module):
    """U-Net style segmentation network"""

    def __init__(self, num_classes: int = 3, 
                 in_channels: int = 3, dropout_p: float = 0.5):
        """Initialize the VGG11UNet model
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            dropout_p: Dropout probability for the segmentation head
        """
        super(VGG11UNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_p = dropout_p

        self.relu = torch.nn.ReLU()
        self.dropout = CustomDropout(dropout_p)
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # expanding the encoder bottleneck features to 1024 channels for the decoder
        self.conv1 = torch.nn.Conv2d(512, 1024, 3, padding=1)
        # we can reduce the params by using 1x1 convs
        # self.conv1 = torch.nn.Conv2d(512, 1024, 1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=1024)
        self.conv2 = torch.nn.Conv2d(1024, 512, 3, padding=1)
        # self.conv2 = torch.nn.Conv2d(1024, 512, 1)  # we can reduce the params
        self.bn2 = torch.nn.BatchNorm2d(num_features=512)

        # decoder layers
        self.trconv1 = torch.nn.ConvTranspose2d(512, 512, 2, stride=2)
        # conv after concatenating skip connection
        self.conv11_1 = torch.nn.Conv2d(1024, 512, 3, padding=1)
        self.bn3_1 = torch.nn.BatchNorm2d(num_features=512)
        self.conv11_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.bn3_2 = torch.nn.BatchNorm2d(num_features=512)

        self.trconv2 = torch.nn.ConvTranspose2d(512, 512, 2, stride=2)
        # conv after concatenating skip connection
        self.conv22_1 = torch.nn.Conv2d(1024, 512, 3, padding=1)
        self.bn4_1 = torch.nn.BatchNorm2d(num_features=512)
        self.conv22_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = torch.nn.BatchNorm2d(num_features=512)

        self.trconv3 = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        # conv after concatenating skip connection
        self.conv33_1 = torch.nn.Conv2d(512, 256, 3, padding=1)
        self.bn5_1 = torch.nn.BatchNorm2d(num_features=256)
        self.conv33_2 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.bn5_2 = torch.nn.BatchNorm2d(num_features=256)

        self.trconv4 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv44 = torch.nn.Conv2d(
            256, 128, 3, padding=1
        )  # conv after concatenating skip connection
        self.bn6 = torch.nn.BatchNorm2d(num_features=128)

        self.trconv5 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv55 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn7 = torch.nn.BatchNorm2d(num_features=64)

        self.out = torch.nn.Conv2d(64, self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model
        Args:
            x: Input tensor of shape [B, in_channels, H, W]
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        x, skip_features = self.encoder.forward(x, return_features=True)
        
        # x is of shape [b, 512, 7, 7]
        x = self.relu(self.bn1(self.conv1(x))) # [b, 1024, 7, 7]
        x = self.relu(self.bn2(self.conv2(x))) # [b, 512, 7, 7]

        # let us do the upsampling and begin with decoder part
        
        # decoder stage 1 (mirrors skip5)
        x = self.trconv1(x) # [b, 512, 14, 14]
        # concatenate skip connection
        skip5 = skip_features["skip5"] # [b, 512, 14, 14]
        x = torch.cat([x, skip5], dim=1) # [b, 1024, 14, 14]
        x = self.relu(self.bn3_1(self.conv11_1(x))) # [b, 512, 14, 14]
        x = self.relu(self.bn3_2(self.conv11_2(x))) # [b, 512, 14, 14]

        # decoder stage 2 (mirrors skip4)
        x = self.trconv2(x) # [b, 512, 28, 28]
        # concatenate skip connection
        skip4 = skip_features["skip4"] # [b, 512, 28, 28]
        x = torch.cat([x, skip4], dim=1) # [b, 1024, 28, 28]
        x = self.relu(self.bn4_1(self.conv22_1(x))) # [b, 512, 28, 28]
        x = self.relu(self.bn4_2(self.conv22_2(x))) # [b, 512, 28, 28]

        # decoder stage 3 (mirrors skip3)
        x = self.trconv3(x) # [b, 256, 56, 56]
        # concatenate skip connection
        skip3 = skip_features["skip3"] # [b, 256, 56, 56]
        x = torch.cat([x, skip3], dim=1) # [b, 512, 56, 56]
        x = self.relu(self.bn5_1(self.conv33_1(x))) # [b, 256, 56, 56]
        x = self.relu(self.bn5_2(self.conv33_2(x))) # [b, 256, 56, 56]

        # decoder stage 4 (mirrors skip2)
        x = self.trconv4(x) # [b, 128, 112, 112]
        # concatenate skip connection
        skip2 = skip_features["skip2"] # [b, 128, 112, 112]
        x = torch.cat([x, skip2], dim=1) # [b, 256, 112, 112]
        x = self.relu(self.bn6(self.conv44(x))) # [b, 128, 112, 112]

        # decoder stage 5 (mirrors skip1)
        x = self.trconv5(x) # [b, 64, 224, 224]
        # concatenate skip connection
        skip1 = skip_features["skip1"] # [b, 64, 224, 224]
        x = torch.cat([x, skip1], dim=1) # [b, 128, 224, 224]
        x = self.relu(self.bn7(self.conv55(x))) # [b, 64, 224, 224]

        x = self.dropout(x)

        # final conv layer to get desired number of output channels
        out = self.out(x) # [b, num_classes, 224, 224]
        # for prediction mask, we take argmax across the channel dim
        
        return out
