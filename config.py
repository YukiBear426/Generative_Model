class Config:
    # ===== 通用参数 =====
    image_size = 64           # 输入图像尺寸（宽 = 高），统一缩放为 64x64
    img_channels = 3          # 图像通道数，3 表示 RGB 彩色图像
    base_channels = 64        # 网络中最基本的通道数，用于定义 Generator/Discriminator 的通道基准
    batch_size = 64           # 每个训练 batch 的图像数量
    latent_dim = 128          # 潜在空间维度，用于生成模型中的随机噪声输入（z 向量）
    num_epochs = 100          # 训练总轮数
    lr = 1e-4                 # 学习率（learning rate），用于优化器
    beta1 = 0.5               # Adam 优化器的 beta1 参数（动量项）
    beta2 = 0.999             # Adam 优化器的 beta2 参数
    num_workers = 4           # DataLoader 使用的子线程数，加快数据加载速度
    seed = 42                 # 随机数种子，用于结果复现

    # ===== Diffusion 参数 =====
    T = 1000                  # Diffusion 模型中的时间步数（即多少步逐步添加噪声）
    beta_start = 1e-4         # 线性 beta 调度的起始值（每一步添加的噪声量初始值）
    beta_end = 0.02           # 线性 beta 调度的终止值（最终添加的噪声量）

    # ===== 路径配置 =====
    data_root = './data'      # 图像数据集的根目录
    output_root = './output'  # 生成图像及模型输出的保存目录
    log_dir = './logs'        # 训练过程日志与图表保存目录

    # ===== 设备配置 =====
    device = 'cuda:6'         # 计算设备

# 实例化配置对象
config = Config()
