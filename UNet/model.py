# src/model_torch.py
# PyTorch port of src/model.py (TF/Keras) with depth adjusted to:
# 960 -> 480 -> 240 -> 120 -> 240 -> 480 -> 960

from __future__ import annotations
import torch
import torch.nn as nn


def _conv5x5(in_ch: int, out_ch: int) -> nn.Conv2d:
    # "same" padding for k=5 in PyTorch (NCHW)
    return nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2, bias=False)


class ConvBNLReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = _conv5x5(in_ch, out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UNetFPP(nn.Module):
    """
    Input:  (B, 1, 960, 960)
    Output: (B, 1, 960, 960)

    Matches the repo's TF model style:
    - 5x5 convs
    - BatchNorm + LeakyReLU after each conv
    - Dropout p=0.1 after each pooling and after each decoder stage
    - Transposed conv upsampling with k=5, stride=2, "same-ish" geometry
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, p_drop: float = 0.1):
        super().__init__()

        # Encoder
        self.conv1_1 = ConvBNLReLU(in_ch, 32)
        self.conv1_2 = ConvBNLReLU(32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=p_drop)

        self.conv2_1 = ConvBNLReLU(32, 64)
        self.conv2_2 = ConvBNLReLU(64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(p=p_drop)

        self.conv3_1 = ConvBNLReLU(64, 128)
        self.conv3_2 = ConvBNLReLU(128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(p=p_drop)

        # Bottleneck at 120x120 (this corresponds to "conv4" block in the TF model,
        # but without the extra pool to 60x60).
        self.conv4_1 = ConvBNLReLU(128, 256)
        self.conv4_2 = ConvBNLReLU(256, 256)
        self.drop4 = nn.Dropout2d(p=p_drop)

        # Decoder
        # ConvTranspose2d sizing:
        # out = (in-1)*2 -2*pad + k + out_pad
        # choose k=5, pad=2, out_pad=1 => out = 2*in
        self.up5 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv5_1 = ConvBNLReLU(128, 128)          # "pre-merge" conv (like TF decoder blocks)
        self.conv5_2 = ConvBNLReLU(128 + 128, 128)    # after concat with skip
        self.conv5_3 = ConvBNLReLU(128, 128)
        self.drop5 = nn.Dropout2d(p=p_drop)

        self.up6 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv6_1 = ConvBNLReLU(64, 64)
        self.conv6_2 = ConvBNLReLU(64 + 64, 64)
        self.conv6_3 = ConvBNLReLU(64, 64)
        self.drop6 = nn.Dropout2d(p=p_drop)

        self.up7 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv7_1 = ConvBNLReLU(32, 32)
        self.conv7_2 = ConvBNLReLU(32 + 32, 32)
        self.conv7_3 = ConvBNLReLU(32, 32)
        self.drop7 = nn.Dropout2d(p=p_drop)

        # Output (linear)
        self.out = nn.Conv2d(32, out_ch, kernel_size=5, padding=2, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        # TF used glorot_normal; approximate with Xavier normal
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.conv1_2(self.conv1_1(x))          # (B,32,960,960)
        p1 = self.drop1(self.pool1(x1))             # (B,32,480,480)

        x2 = self.conv2_2(self.conv2_1(p1))         # (B,64,480,480)
        p2 = self.drop2(self.pool2(x2))             # (B,64,240,240)

        x3 = self.conv3_2(self.conv3_1(p2))         # (B,128,240,240)
        p3 = self.drop3(self.pool3(x3))             # (B,128,120,120)

        xb = self.drop4(self.conv4_2(self.conv4_1(p3)))  # (B,256,120,120)

        # Decoder
        u5 = self.up5(xb)                           # (B,128,240,240)
        c5_1 = self.conv5_1(u5)                     # (B,128,240,240)
        m5 = torch.cat([x3, c5_1], dim=1)           # (B,256,240,240)
        x5 = self.drop5(self.conv5_3(self.conv5_2(m5)))  # (B,128,240,240)

        u6 = self.up6(x5)                           # (B,64,480,480)
        c6_1 = self.conv6_1(u6)                     # (B,64,480,480)
        m6 = torch.cat([x2, c6_1], dim=1)           # (B,128,480,480)
        x6 = self.drop6(self.conv6_3(self.conv6_2(m6)))  # (B,64,480,480)

        u7 = self.up7(x6)                           # (B,32,960,960)
        c7_1 = self.conv7_1(u7)                     # (B,32,960,960)
        m7 = torch.cat([x1, c7_1], dim=1)           # (B,64,960,960)
        x7 = self.drop7(self.conv7_3(self.conv7_2(m7)))  # (B,32,960,960)

        return self.out(x7)


def masked_rmse(target_y: torch.Tensor, pred_y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    PyTorch version of the repo's TF loss() in src/model.py.
    Expected:
      target_y: (B, 2, H, W) where target_y[:,0] = depth, target_y[:,1] = mask (0=background)
      pred_y:   (B, 1, H, W)
    Returns:
      scalar RMSE over masked pixels
    """
    mask = target_y[:, 1:2, :, :]  # keep channel dim
    truth = target_y[:, 0:1, :, :]
    diff2 = (truth - pred_y) ** 2
    num = (mask * diff2).sum()
    den = mask.sum().clamp_min(eps)
    return torch.sqrt(num / den)


if __name__ == "__main__":
    import torch
    # from UNet.model import UNetFPP

    m = UNetFPP().eval()
    x = torch.randn(2, 1, 960, 960)
    y = m(x)
    print(y.shape)  # torch.Size([2, 1, 960, 960])
