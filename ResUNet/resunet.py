import torch
import torch.nn as nn
from resunet_parts import *

class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        """
        ResNet for Fringe Projection Profilometry
        Input: 960x960 fringe image
        Output: 960x960 depth map

        Architecture:
        - Initial block: 960x960 (no downsampling)
        - Down1: 960x960 -> 480x480
        - Down2: 480x480 -> 240x240
        - Down3: 240x240 -> 120x120
        - Bottleneck: 120x120 (with dropout)
        - Up1: 120x120 -> 240x240 (with skip from Down2)
        - Up2: 240x240 -> 480x480 (with skip from Down1)
        - Up3: 480x480 -> 960x960 (with skip from Initial)
        - Output: 960x960
        """
        super().__init__()

        # ENCODER PATH
        # Initial block (no downsampling, maintains 960x960)
        self.initial = InitialBlock(in_channels, 120, num_blocks=2)

        # Downsampling blocks
        self.down1 = DownBlock(120, 240, num_blocks=2)   # 960x960 -> 480x480
        self.down2 = DownBlock(240, 480, num_blocks=2)   # 480x480 -> 240x240
        self.down3 = DownBlock(480, 960, num_blocks=2)   # 240x240 -> 120x120

        # BOTTLENECK with Dropout
        self.bottleneck = nn.Sequential(
            ResidualBlock(960, 960, stride=1),
            ResidualBlock(960, 960, stride=1)
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # DECODER PATH
        self.up1 = UpBlock(960, 480, num_blocks=2)       # 120x120 -> 240x240
        self.up2 = UpBlock(480, 240, num_blocks=2)       # 240x240 -> 480x480
        self.up3 = UpBlock(240, 120, num_blocks=2)       # 480x480 -> 960x960

        # OUTPUT LAYER
        self.out = nn.Conv2d(120, out_channels, kernel_size=1)

    def forward(self, x):
        # ENCODER WITH SKIP CONNECTIONS
        skip1 = self.initial(x)          # 960x960x120
        skip2 = self.down1(skip1)        # 480x480x240
        skip3 = self.down2(skip2)        # 240x240x480
        x = self.down3(skip3)            # 120x120x960

        # BOTTLENECK with Dropout
        x = self.bottleneck(x)           # 120x120x960
        x = self.dropout(x)

        # DECODER with skip connections
        x = self.up1(x, skip3)           # 240x240x480
        x = self.up2(x, skip2)           # 480x480x240
        x = self.up3(x, skip1)           # 960x960x120

        # OUTPUT
        x = self.out(x)                  # 960x960x1

        return x
