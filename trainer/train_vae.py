import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import config
from models.vae_model import VAE
from utils import *
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

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

# 模型
dim = config.latent_dim
vae = VAE(latent_dim=dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=config.lr)

loss_list = []
fixed_noise = torch.randn(25, dim).to(device)

# 训练
def train():
    print(f"熊宇琦2022090048, vae模型参数量：{count_params(vae)/1e6:.3f} M")
    total_time = 0.0
    for epoch in range(1, config.num_epochs + 1):
        vae.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            recon, mu, logvar = vae(imgs)

            recon_loss = F.mse_loss(recon, imgs, reduction='sum') / imgs.size(0)
            kl_div = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / imgs.size(0)
            loss = recon_loss + 0.5*kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader.dataset)
        loss_list.append(avg_loss)
        print(f"熊宇琦2022090048, epoch {epoch}/{config.num_epochs}, VAE Loss: {avg_loss:.4f}")

        # 生成图像 + 测试单张图像推理时间
        vae.eval()
        with torch.no_grad():
            # 测试推理时间
            warmup_runs = 3  # 先跑几次预热
            test_runs = 10
            for _ in range(warmup_runs):
                _ = vae.decode(fixed_noise)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            for _ in range(test_runs):
                samples = vae.decode(fixed_noise)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()

            avg_infer_time = ((end_time - start_time) / test_runs ) * 1000
            print(f"熊宇琦2022090048, 单张推理时间：{avg_infer_time:.3f} ms")
            total_time += avg_infer_time

            samples = samples.cpu()
            epoch_dir = f"{config.output_root}/VAE/epoch_{epoch:03d}"
            ensure_dir(epoch_dir)
            save_images(samples, f"{epoch_dir}/samples.png")

    # 保存损失曲线
    ensure_dir(config.log_dir)
    plot_loss_curve(loss_list, f"{config.log_dir}/vae_loss.png")

    average_time = total_time / config.num_epochs
    print(f"熊宇琦2022090048, vae平均单张推理时间：{average_time:.3f} ms")

if __name__ == '__main__':
    train()
