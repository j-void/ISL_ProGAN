import utils as util
from model import ProGAN_Generator
import torch
import os
import config
from torchvision.utils import save_image
from math import log2

save_path = "results/"
IMG_SIZE = 4

def main():
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if not os.path.exists(os.path.join(save_path, config.name)):
        os.makedirs(os.path.join(save_path, config.name))
    
    gen = ProGAN_Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.device)
    util.load_checkpoint("netG.pth", gen, None, config.LEARNING_RATE,)
    gen.eval()
    alpha = 1.0
    step = int(log2(IMG_SIZE / 4))
    for i in range(100):
        with torch.no_grad():
            noise = torch.randn(1, config.Z_DIM, 1, 2).to(config.device).float()
            img = gen(noise, alpha, step)
            print(f'Saving {IMG_SIZE}x{IMG_SIZE*2} image : {i}')
            save_image(img*0.5+0.5, os.path.join(save_path, config.name, f'output_{IMG_SIZE}x{IMG_SIZE*2}_'+'{:0>12}'.format(i)+'.png'))

if __name__ == "__main__":
    main()