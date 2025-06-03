import torch
import torch.nn as nn
from config import config

class Discriminator(nn.Module):
    def __init__(self, in_channels=None, base_channels=64):
        super(Discriminator, self).__init__()
        C = in_channels or config.img_channels
        self.model = nn.Sequential(
            # kernel_size 改为 5，padding 对称
            nn.Conv2d(C, base_channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels * 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)
        return out.view(-1)


class Generator(nn.Module):
    def __init__(self, latent_dim=None, out_channels=None, base_channels=64):
        super(Generator, self).__init__()
        Z = latent_dim or config.latent_dim
        C = out_channels or config.img_channels

        self.model = nn.Sequential(
            # 从 1×1 特征图开始上采样
            nn.ConvTranspose2d(Z, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ELU(inplace=True),

            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels, C, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        # z: [B, Z, 1, 1] -> 输出 [B, C, H, W]
        return self.model(z)
