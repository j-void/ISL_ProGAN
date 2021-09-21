import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import utils as util
from model import Discriminator, Generator
from math import log2
import config
import os

torch.backends.cudnn.benchmarks = True

def main():
    
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    
    if not os.path.exists(os.path.join(config.checkpoint_dir, config.name)):
        os.makedirs(os.path.join(config.checkpoint_dir, config.name))
    
    gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.device)
    
    critic = Discriminator(config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.device)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # for tensorboard plotting
    writer = SummaryWriter(os.path.join(config.checkpoint_dir, config.name, "logs"))

    if config.LOAD_MODEL:
        util.load_checkpoint(
            "netG.pth", gen, opt_gen, config.LEARNING_RATE,
        )
        util.load_checkpoint(
            "netD.pth", critic, opt_critic, config.LEARNING_RATE,
        )

    gen.train()
    critic.train()
    epoch_iter = 0
    tensorboard_step = 0
    # start at step that corresponds to img size that we set in config
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5  # start with very low alpha
        image_size = 4 * 2 ** step
        batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
        dataset = util.get_dataset(config.image_dir, config.label_dir, batch_size, config.num_workers, image_size, isTrain=True)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
        print(f"Current image size: {image_size}x{image_size*2}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            for batch_idx, data in enumerate(dataset):
                real = data["image"].to(config.device)
                cur_batch_size = real.shape[0]
                epoch_iter += cur_batch_size
                noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 2).to(config.device)
                with torch.cuda.amp.autocast():
                    fake = gen(noise, alpha, step)
                    critic_real = critic(real, alpha, step)
                    critic_fake = critic(fake.detach(), alpha, step)
                    gp = util.gradient_penalty(critic, real, fake, alpha, step, device=config.device)
                    loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake))
                        + config.LAMBDA_GP * gp
                        + (0.001 * torch.mean(critic_real ** 2))
                    )

                opt_critic.zero_grad()
                scaler_critic.scale(loss_critic).backward()
                scaler_critic.step(opt_critic)
                scaler_critic.update()

                # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
                with torch.cuda.amp.autocast():
                    gen_fake = critic(fake, alpha, step)
                    loss_gen = -torch.mean(gen_fake)

                opt_gen.zero_grad()
                scaler_gen.scale(loss_gen).backward()
                scaler_gen.step(opt_gen)
                scaler_gen.update()

                # Update alpha and ensure less than 1
                alpha += cur_batch_size / (
                    (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
                )
                alpha = min(alpha, 1)

                if batch_idx % 500 == 0:
                    with torch.no_grad():
                        fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
                    util.plot_to_tensorboard(writer, loss_critic.item(), loss_gen.item(), real.detach(), fixed_fakes.detach(), tensorboard_step)
                    tensorboard_step += 1
                
                if batch_idx % 500 == 0:
                    util.save_checkpoint(gen, opt_gen, filename="netG.pth")
                    util.save_checkpoint(critic, opt_critic, filename="netD.pth")
                    util.save_train_state(image_size, epoch+1, num_epochs, batch_idx)
                
            
            util.save_checkpoint(gen, opt_gen, filename="netG.pth")
            util.save_checkpoint(critic, opt_critic, filename="netD.pth")
            util.save_train_state(image_size, epoch+1, num_epochs, batch_idx)

        step += 1  # progress to the next img size


if __name__ == "__main__":
    main()