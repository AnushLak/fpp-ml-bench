import torch
import torch.nn as nn
from unet_parts import *

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()

        # ENCODER PATH
        # 960x960 -> 480x480
        self.down1 = DownSample(in_channels, 120)
        self.down2 = DownSample(120, 240)
        self.down3 = DownSample(240, 480)

        # BOTTLENECK with Dropout
        self.bottleneck = DoubleConv(480, 960)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # DECODER PATH
        self.up1 = UpSample(960, 480)
        self.up2 = UpSample(480, 240)
        self.up3 = UpSample(240, 120)

        # OUTPUT LAYER
        self.out = nn.Conv2d(120, out_channels, kernel_size=1)

    def forward(self, x):
        # ENCODER WITH SKIP CONNECTIONS
        skip1, x = self.down1(x)      # skip: 960x960x120, x: 480x480x120
        skip2, x = self.down2(x)      # skip: 480x480x240, x: 240x240x240
        skip3, x = self.down3(x)      # skip: 240x240x480, x: 120x120x480

        # BOTTLENECK with Dropout
        x = self.bottleneck(x)        # 120x120x960
        x = self.dropout(x)

        # DECODER with skip connections
        x = self.up1(x, skip3)        # 240x240x480
        x = self.up2(x, skip2)        # 480x480x240
        x = self.up3(x, skip1)        # 960x960x120

        # OUTPUT
        x = self.out(x)               # 960x960x1

        return x