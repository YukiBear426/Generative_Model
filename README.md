## Project Description
This project implements the generation of cartoon avatar images based on deep generative models, supporting multiple mainstream generative frameworks, including GAN, VAE, Flow, and Diffusion.


## Project Structure
```bash
Generative_Model/
├── data/                     # Raw cartoon avatar dataset
├── models/                   # Model definitions
│   ├── gan_model.py          # GAN model
│   ├── flow_model.py         # Flow model
│   ├── vae_model.py          # VAE model
│   └── diffusion_model.py    # Diffusion model
├── trainer/                  # Training scripts
│   ├── train_gan.py          # GAN training script
│   ├── train_flow.py         # Flow training script
│   ├── train_vae.py          # VAE training script
│   └── train_diffusion.py    # Diffusion training script
├── config.py                 # Hyperparameters
├── utils.py                  # Utility functions
├── train.py                  # Unified training interface
├── visualization.py          # Visualization (e.g., GIF generation)
├── output/                   # Output images
├── logs/                     # Training loss logs
└── README.md                 # Project description
  
```

## Dataset Description
Please download the dataset and place it into the `data/` folder before training or testing.

## Install Dependencies

```bash
cd Generative_Model
conda activate # [your env]
pip install torch torchvision matplotlib tqdm pillow 
```

## Training and Testing
```bash
python train.py --model diffusion  # you can choose：vae、flow、gan、diffusion
```


