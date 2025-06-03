import torch
import torch.nn as nn
import torch.nn.functional as F


# 生成时间步嵌入（位置编码）
def get_timestep_embedding(timesteps: torch.LongTensor, embedding_dim: int):
    half_dim = embedding_dim // 2
    device = timesteps.device
    emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # 若嵌入维度为奇数，补 0 保持维度一致
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb  # 返回 (B, embedding_dim) 形状的时间步嵌入

# 残差块，包含时间嵌入加权
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)  # 用于时间嵌入
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # 若输入输出通道不同，使用1x1卷积变换，否则恒等映射
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # 添加时间步嵌入
        t = self.time_mlp(F.silu(t_emb))  # (B, out_ch)
        h = h + t[:, :, None, None]       # 广播到图像维度相加

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.res_conv(x)       # 残差连接

# 下采样模块，使用 stride=2 的卷积实现尺寸减半
class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.net(x)

# 上采样模块，使用反卷积实现尺寸扩大
class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.net(x)

# U-Net 主体结构
class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,               # 输入通道数（RGB为3）
        base_channels=64,            # 基础通道数
        channel_mults=(1, 2, 4, 8),  # 每层的通道倍数
        time_emb_dim=256,           # 时间步嵌入维度
    ):
        super().__init__()
        # 时间嵌入 MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # 编码器（下采样部分）
        downs = []
        in_chs = [base_channels] + [base_channels * m for m in channel_mults[:-1]]
        out_chs = [base_channels * m for m in channel_mults]
        for in_ch, out_ch in zip(in_chs, out_chs):
            downs.append(
                nn.ModuleList([
                    ResidualBlock(in_ch, out_ch, time_emb_dim),
                    Downsample(out_ch),
                ])
            )
        self.downs = nn.ModuleList(downs)

        # 中间层（两个残差块）
        self.mid1 = ResidualBlock(out_chs[-1], out_chs[-1], time_emb_dim)
        self.mid2 = ResidualBlock(out_chs[-1], out_chs[-1], time_emb_dim)

        # 解码器（上采样部分）
        rev_out_chs = list(reversed(out_chs))            # 解码器输出通道顺序
        prev_chs   = [out_chs[-1]] + rev_out_chs[:-1]     # 上一层输出通道
        ups = []
        for in_ch, skip_ch, out_ch in zip(prev_chs, rev_out_chs, rev_out_chs):
            ups.append(
                nn.ModuleList([
                    ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim),  # skip connection 拼接
                    Upsample(in_ch),  # 上采样
                ])
            )
        self.ups = nn.ModuleList(ups)

        # 最后归一化与卷积，恢复为原图通道数
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, t):
        """
        x: 输入图像 (B, C, H, W)
        t: 时间步 (B,) long 类型
        """
        # 时间步嵌入
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        # 编码器路径
        h = self.init_conv(x)
        hs = [h]
        for block, down in self.downs:
            h = block(h, t_emb)
            hs.append(h)
            h = down(h)

        # 中间残差块
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # 解码器路径（包含 skip connection）
        for (block, up), skip in zip(self.ups, reversed(hs)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)  # 拼接 encoder 输出
            h = block(h, t_emb)

        h = self.final_norm(h)
        h = F.silu(h)
        return self.final_conv(h)

# ---------------------- Diffusion 扩散核心调度 ----------------------

class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device='cpu'):
        self.device = device
        self.T = timesteps  # 总时间步数
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_acp = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def q_sample(self, x0, t, noise=None):
        """
        根据前向扩散公式 q(x_t | x_0) 给原始图像添加噪声
        """
        if noise is None:
            noise = torch.randn_like(x0)  # 默认生成高斯噪声
        return (
            self.sqrt_alphas_cumprod[t][:, None, None, None] * x0 +
            self.sqrt_one_minus_acp[t][:, None, None, None] * noise
        )

    @torch.no_grad()
    def p_sample(self, model, x, t):
        """
        反向采样 p(x_{t-1} | x_t)
        """
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_acp_t = self.sqrt_one_minus_acp[t][:, None, None, None]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t][:, None, None, None]

        # 由模型预测噪声
        eps_pred = model(x, t)
        # 计算预测均值
        model_mean = sqrt_recip_alpha_t * (
            x - betas_t / sqrt_one_minus_acp_t * eps_pred
        )
        if t[0] > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(betas_t)
            return model_mean + sigma_t * noise  # 添加噪声
        else:
            return model_mean  # 最后一轮不加噪

    @torch.no_grad()
    def sample_loop(self, model, shape):
        """
        从纯噪声开始，逐步去噪生成图像
        """
        img = torch.randn(*shape, device=self.device)  # 初始为标准高斯噪声
        for step in reversed(range(self.T)):  # 从 T-1 到 0 逐步采样
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            img = self.p_sample(model, img, t)
        return img
