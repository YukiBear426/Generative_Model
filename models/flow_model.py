import torch
import torch.nn as nn
from config import config

class FlowModel(nn.Module):
    def __init__(self):
        super(FlowModel, self).__init__()
        # 展平后维度：C*H*W
        self.input_dim = config.img_channels * config.image_size * config.image_size
        # 隐藏层维度，可按需调整
        self.hidden_dim1 = config.base_channels * config.base_channels
        self.hidden_dim2 = config.base_channels // 2 * config.base_channels

        # 四段变换，增加深度
        self.transform1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim1),
            nn.LayerNorm(self.hidden_dim1),
            nn.Tanh()
        )
        self.transform2 = nn.Sequential(
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.BatchNorm1d(self.hidden_dim2),
            nn.ReLU(True)
        )
        self.transform3 = nn.Sequential(
            nn.Linear(self.hidden_dim2, self.hidden_dim1),
            nn.LayerNorm(self.hidden_dim1),
            nn.Sigmoid()
        )
        self.transform4 = nn.Sequential(
            nn.Linear(self.hidden_dim1, self.input_dim),
            nn.BatchNorm1d(self.input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        B = x.size(0)
        x_flat = x.view(B, -1)
        # 正向多层变换
        z1 = self.transform1(x_flat)
        z2 = self.transform2(z1)
        z3 = self.transform3(z2)
        x_hat_flat = self.transform4(z3)
        x_hat = x_hat_flat.view(B, config.img_channels, config.image_size, config.image_size)
        return x_hat, z3