import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Simplified HRNet-W18 backbone for Hformer
Generates multi-scale features with channels [18, 36, 72, 144]
"""

class BasicBlock(nn.Module):
    """Basic residual block for HRNet"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HRNetBackbone(nn.Module):
    """
    Simplified HRNet-W18 backbone
    Outputs 4 feature maps with channels [18, 36, 72, 144]
    """
    def __init__(self, in_channels=1):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.layer1 = self._make_layer(BasicBlock, 64, 18, 4)

        # Transition 1: 1 branch -> 2 branches
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(18, 18, 3, 1, 1, bias=False),
                nn.BatchNorm2d(18),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(18, 36, 3, 2, 1, bias=False),
                nn.BatchNorm2d(36),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 2
        self.stage2 = nn.ModuleList([
            self._make_layer(BasicBlock, 18, 18, 4),
            self._make_layer(BasicBlock, 36, 36, 4)
        ])

        # Transition 2: 2 branches -> 3 branches
        self.transition2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(18, 18, 3, 1, 1, bias=False),
                nn.BatchNorm2d(18),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(36, 36, 3, 1, 1, bias=False),
                nn.BatchNorm2d(36),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(36, 72, 3, 2, 1, bias=False),
                nn.BatchNorm2d(72),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 3
        self.stage3 = nn.ModuleList([
            self._make_layer(BasicBlock, 18, 18, 4),
            self._make_layer(BasicBlock, 36, 36, 4),
            self._make_layer(BasicBlock, 72, 72, 4)
        ])

        # Transition 3: 3 branches -> 4 branches
        self.transition3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(18, 18, 3, 1, 1, bias=False),
                nn.BatchNorm2d(18),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(36, 36, 3, 1, 1, bias=False),
                nn.BatchNorm2d(36),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(72, 72, 3, 1, 1, bias=False),
                nn.BatchNorm2d(72),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(72, 144, 3, 2, 1, bias=False),
                nn.BatchNorm2d(144),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 4
        self.stage4 = nn.ModuleList([
            self._make_layer(BasicBlock, 18, 18, 3),
            self._make_layer(BasicBlock, 36, 36, 3),
            self._make_layer(BasicBlock, 72, 72, 3),
            self._make_layer(BasicBlock, 144, 144, 3)
        ])

    def _make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        layers.append(block(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Stage 1
        x = self.layer1(x)

        # Transition 1
        x_list = [trans(x) for trans in self.transition1]

        # Stage 2
        x_list = [stage(x) for stage, x in zip(self.stage2, x_list)]

        # Transition 2
        x_new = []
        for i, trans in enumerate(self.transition2):
            if i < len(x_list):
                x_new.append(trans(x_list[i]))
            else:
                x_new.append(trans(x_list[-1]))
        x_list = x_new

        # Stage 3
        x_list = [stage(x) for stage, x in zip(self.stage3, x_list)]

        # Transition 3
        x_new = []
        for i, trans in enumerate(self.transition3):
            if i < len(x_list):
                x_new.append(trans(x_list[i]))
            else:
                x_new.append(trans(x_list[-1]))
        x_list = x_new

        # Stage 4
        x_list = [stage(x) for stage, x in zip(self.stage4, x_list)]

        return x_list  # Returns [b1, b2, b3, b4] with shapes matching HRNet-W18
