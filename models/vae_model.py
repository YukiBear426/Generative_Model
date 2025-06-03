import torch
import torch.nn as nn

# 编码器：将图像编码为潜在空间的均值和对数方差
class Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 3x64x64 -> 64x32x32
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # 64x32x32 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),# 128x16x16 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),# 256x8x8 -> 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(512*4*4, latent_dim)
        self.fc_logvar = nn.Linear(512*4*4, latent_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# 解码器：将潜在向量解码为图像
class Decoder(nn.Module):
    def __init__(self, latent_dim=100):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512*4*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 512x4x4 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 256x8x8 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 128x16x16 -> 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 64x32x32 -> 3x64x64
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 512, 4, 4)
        x_recon = self.deconv(h)
        return x_recon

# 完整VAE模型，包含重参数采样
class VAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def decode(self, z):
        return self.decoder(z)
