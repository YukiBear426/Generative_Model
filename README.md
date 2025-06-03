
# Face_Generation 项目说明

# 本项目基于深度生成模型实现卡通头像图像的生成，支持多种主流生成框架，包括：gan、vae、flow、diffusion

# 项目结构
```bash
Face_generation/ 
├── data/                     # 原始卡通头像数据集  
├── models/                   # 模型定义  
│   ├── gan_model.py          # GAN 模型
│   ├── flow_model.py         # flow 模型
│   ├── vae_model.py          # VAE 模型 
│   └── diffusion_model.py    # diffusion 模型
├── trainer/                  # 模型定义   
├   ├── train_gan.py          # GAN 模型训练脚本  
├   ├── train_gan.py          # flow 模型训练脚本 
├   ├── train_vae.py          # VAE 模型训练脚本  
├   ├── train_diffusion.py    # diffusion 模型训练脚本  
├── config.py                 # 超参数
├── utils.py                  # 辅助函数 
├── train.py                  # 训练总接口
├── visualization.py          # 生成gif可视化文件
├── output/                   # 输出图像目录  
├── logs/                     # 输出损失曲线 
└── README.md                 # 项目说明文档  
```

# 数据集说明：请下载好数据及后放入.data/中

# 安装依赖项：

```bash
cd Face_generation
conda activate # [your env]
pip install torch torchvision matplotlib tqdm pillow 
```

# 训练
```bash
python train.py --model diffusion  # 可选：vae、flow、gan、diffusion
```


