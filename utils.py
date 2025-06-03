import os
import torch
import random
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import matplotlib


def set_seed(seed):
    """
    设置随机种子，确保实验结果具有可复现性。
    参数: seed (int): 随机数种子。
    """
    torch.manual_seed(seed)                # 设置 PyTorch 的 CPU 随机种子
    np.random.seed(seed)                   # 设置 NumPy 的随机种子
    random.seed(seed)                      # 设置 Python 原生 random 模块的种子
    torch.cuda.manual_seed_all(seed)       # 设置所有 GPU 的随机种子（用于多 GPU）

def ensure_dir(path):
    """
    确保目录存在，如果目录不存在则创建它。
    参数: path (str): 目标目录路径。
    """
    if not os.path.exists(path):           # 如果路径不存在
        os.makedirs(path)                  # 创建目录及所有中间子目录

def save_images(tensor, save_path, nrow=5):
    """
    将一批图像保存为一张拼接图像。
    参数:
        tensor (Tensor): 图像张量 (N, C, H, W)，只保存前25张。
        save_path (str): 图像保存路径。
        nrow (int): 每行显示图像的数量。
    """
    grid = make_grid(tensor[:25],          # 取前 25 张图像组成网格
                     nrow=nrow,            # 每行 nrow 张图
                     normalize=True,       # 将像素值归一化到 [0, 1]
                     pad_value=255)        # 白色填充图像间距
    save_image(grid, save_path)            # 保存拼接后的图像到磁盘

def plot_loss_curve(loss_list, save_path):
    """
    绘制并保存损失曲线图。
    参数:
        loss_list (List[float]): 每个 epoch 的损失值列表。
        save_path (str): 曲线图的保存路径。
    """
    plt.figure()                                # 新建图像
    plt.plot(loss_list, label="Loss")           # 画出损失曲线
    plt.xlabel("Epoch")                         # X 轴标签：训练轮数
    plt.ylabel("Loss")                          # Y 轴标签：损失值
    plt.title("Training Loss (Xiong Yuqi 2022090048)")  # 图标题，带姓名学号
    plt.legend()                                # 显示图例
    plt.savefig(save_path)                      # 保存图像到指定路径
    plt.close()                                 # 关闭图像，释放内存

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)