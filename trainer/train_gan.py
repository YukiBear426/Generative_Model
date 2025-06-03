import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.gan_model import Generator, Discriminator
from utils import *
from config import config
from tqdm import tqdm
import numpy as np

# 设置设备
device = torch.device(config.device)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 加载数据集
dataset = datasets.ImageFolder(root=config.data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)

# 实例化模型
G = Generator().to(device)
D = Discriminator().to(device)

# 优化器与损失
optimizer_G = optim.Adam(G.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
optimizer_D = optim.Adam(D.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
criterion = nn.BCELoss()

# 固定噪声用于可视化
fixed_noise = torch.randn(25, config.latent_dim, 1, 1, device=device)

g_losses, d_losses = [], []

def train():
    print(f"熊宇琦2022090048, Generator 参数量：{count_params(G)/1e6:.3f} M")
    print(f"熊宇琦2022090048, Discriminator 参数量：{count_params(D)/1e6:.3f} M")
    for epoch in range(1, config.num_epochs + 1):
        G.train()
        D.train()
        g_total, d_total = 0.0, 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")
        for real_imgs, _ in pbar:
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)
            valid = torch.ones(bs, device=device)
            fake = torch.zeros(bs, device=device)

            # 1) 训练 Discriminator
            noise = torch.randn(bs, config.latent_dim, 1, 1, device=device)
            gen_imgs = G(noise)

            real_loss = criterion(D(real_imgs), valid)
            fake_loss = criterion(D(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) * 0.5

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # 2) 训练 Generator
            g_loss = criterion(D(gen_imgs), valid)
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            g_total += g_loss.item()
            d_total += d_loss.item()
            pbar.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

        avg_g = g_total / len(dataloader)
        avg_d = d_total / len(dataloader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)
        print(f"熊宇琦2022090048, epoch {epoch} G Loss: {avg_g:.4f}, D Loss: {avg_d:.4f}")

        # 保存生成样本
        with torch.no_grad():
            G.eval()
            samples = G(fixed_noise).cpu().clamp(-1, 1)
            epoch_dir = os.path.join(config.output_root, "GAN", f"epoch_{epoch:03d}")
            ensure_dir(epoch_dir)
            save_image(samples, f"{epoch_dir}/samples.png", nrow=5, normalize=True)

    # 绘制并保存损失曲线
    ensure_dir(config.log_dir)
    plot_loss_curve(g_losses, f"{config.log_dir}/gan_g_loss.png")
    plot_loss_curve(d_losses, f"{config.log_dir}/gan_d_loss.png")

    # 测量推理时间
    print("开始测量单张图像的推理时间...")
    G.eval()
    with torch.no_grad():
        noise = torch.randn(1, config.latent_dim, 1, 1, device=device)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        _ = G(noise)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        average_time = (end - start) * 1000
    print(f"熊宇琦2022090048, gan平均单张图像推理时间: {average_time:.3f} ms")


if __name__ == '__main__':
    train()
