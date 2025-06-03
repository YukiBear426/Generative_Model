import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import config
from models.flow_model import FlowModel
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

# 模型与优化器
flow = FlowModel().to(device)
optimizer = optim.Adam(flow.parameters(),
                       lr=config.lr,
                       betas=(config.beta1, config.beta2))

loss_list = []

# 固定样本用于可视化
fixed_imgs, _ = next(iter(dataloader))
fixed_imgs = fixed_imgs[:25].to(device)

# 训练函数
def train():
    print(f"熊宇琦2022090048, flow模型参数量：{count_params(flow)/1e6:.3f} M")
    total_time = 0.0
    for epoch in range(1, config.num_epochs + 1):
        flow.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            recon, _ = flow(imgs)
            loss = F.mse_loss(recon, imgs, reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader.dataset)
        loss_list.append(avg_loss)
        print(f"熊宇琦2022090048, epoch {epoch}/{config.num_epochs}, Flow MSE: {avg_loss:.6f}")

        # 推理及可视化
        flow.eval()
        with torch.no_grad():
            # 预热
            for _ in range(3):
                _ = flow(fixed_imgs)

            torch.cuda.synchronize() if device.type=='cuda' else None
            start = time.time()
            for _ in range(10):
                samples, _ = flow(fixed_imgs)
            torch.cuda.synchronize() if device.type=='cuda' else None
            end = time.time()

            avg_ms = (end - start) / 10 * 1000
            print(f"熊宇琦2022090048, 单张推理时间：{avg_ms:.3f} ms")
            total_time += avg_ms

            samples = samples.cpu().clamp(0,1)
            epoch_dir = f"{config.output_root}/Flow/epoch_{epoch:03d}"
            ensure_dir(epoch_dir)
            save_images(samples, f"{epoch_dir}/samples.png", nrow=5)

    # 保存损失曲线
    ensure_dir(config.log_dir)
    plot_loss_curve(loss_list, f"{config.log_dir}/flow_loss.png")

    avg_time = total_time / config.num_epochs
    print(f"熊宇琦2022090048, flow平均单张推理时间：{avg_time:.3f} ms")

if __name__ == '__main__':
    train()
