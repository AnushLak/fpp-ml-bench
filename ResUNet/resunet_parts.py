import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual Block with two convolutions and skip connection
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip_connection(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Skip connection
        out = self.relu(out)

        return out

class DownBlock(nn.Module):
    """
    Downsampling block with residual blocks
    """
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()

        layers = []
        # First block with stride=2 for downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))

        # Additional residual blocks
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)

class UpBlock(nn.Module):
    """
    Upsampling block with residual blocks and skip connections
    """
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()

        # Upsampling
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                          kernel_size=2, stride=2)

        # Residual blocks after concatenation
        layers = []
        # First block takes concatenated channels (out_channels from upsample + out_channels from skip)
        layers.append(ResidualBlock(out_channels * 2, out_channels, stride=1))

        # Additional residual blocks
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
        x = self.blocks(x)
        return x

class InitialBlock(nn.Module):
    """
    Initial convolution block (no downsampling)
    """
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()

        layers = []
        # First block
        layers.append(ResidualBlock(in_channels, out_channels, stride=1))

        # Additional residual blocks
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)
