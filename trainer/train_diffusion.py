import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F
from config import config
from utils import *
from models.diffusion_model import UNet, Diffusion


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


def train():
    # 初始化模型和扩散过程
    model = UNet(in_channels=3).to(device)
    print(f"熊宇琦2022090048, diffusion模型参数量: {count_params(model)/1e6:.2f} M")

    diffusion = Diffusion(timesteps=config.T, device=device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    ensure_dir(config.output_root)
    ensure_dir(config.log_dir)

    losses = []

    total_time = 0.0

    for epoch in range(0, config.num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        epoch_loss = 0

        for x, _ in pbar:
            x = x.to(device)
            bsz = x.size(0)
            t = torch.randint(0, diffusion.T, (bsz,), device=device).long()
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)
            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"熊宇琦2022090048, epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # 保存采样图像
        model.eval()
        with torch.no_grad():
            samples = diffusion.sample_loop(
                model, (25, 3, config.image_size, config.image_size)
            )
            samples = (samples.clamp(-1, 1) + 1) / 2
            epoch_dir = os.path.join(config.output_root, f"Diffusion/epoch_{epoch+1:03d}")
            ensure_dir(epoch_dir)
            save_images(samples, os.path.join(epoch_dir, "samples.png"))

        # 测试单张图像推理时间
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, config.image_size, config.image_size).to(device)
            dummy_t = torch.tensor([config.T - 1], dtype=torch.long).to(device)

            # 预热
            _ = model(dummy_image, dummy_t)
            torch.cuda.synchronize()
            start_time = time.time()
            _ = model(dummy_image, dummy_t)
            torch.cuda.synchronize()
            end_time = time.time()

            inference_time = (end_time - start_time) * 1000
            print(f"熊宇琦2022090048, 单张图像推理时间：{inference_time:.3f} ms")

            total_time += inference_time

    # 绘制损失曲线
    plot_loss_curve(losses, os.path.join(config.log_dir, "diffusion_loss_curve.png"))

    average_time = total_time / config.num_epochs
    print(f"熊宇琦2022090048, diffusion平均单张图像推理时间：{average_time:.3f} ms")


if __name__ == '__main__':
    train()
