
import argparse
from trainer import train_gan
from trainer import train_vae
from trainer import train_diffusion
from trainer import train_flow

model_map = {
    'gan': train_gan.train,
    'vae': train_vae.train,
    'diffusion': train_diffusion.train,
    'flow': train_flow.train
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['gan', 'vae', 'diffusion', 'flow'], help='选择训练模型')
    args = parser.parse_args()

    train_fn = model_map[args.model]
    train_fn()

if __name__ == '__main__':
    main()



