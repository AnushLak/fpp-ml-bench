"""
UNet for Single-Shot Fringe Projection Profilometry
Input: 960x960 grayscale fringe pattern
Output: 960x960 depth map
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3×3 convolutions with InstanceNorm + ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    """Downsampling block: DoubleConv + MaxPool."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return skip, x


class UpSample(nn.Module):
    """Upsampling block: ConvTranspose + Concat + DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetFPP(nn.Module):
    """
    UNet for Fringe Projection Profilometry
    Input:  (B, 1, 960, 960)
    Output: (B, 1, 960, 960)
    """

    def __init__(self, in_channels=1, out_channels=1, dropout_rate=0.5):
        super().__init__()

        # ENCODER (original widths)
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        # BOTTLENECK
        self.bottleneck = DoubleConv(512, 1024)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # DECODER
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        # OUTPUT
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """He initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.bottleneck(x)
        x = self.dropout(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.out(x)
