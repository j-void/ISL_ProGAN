import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from style2_networks import Generator, Discriminator

from torch.utils.tensorboard import SummaryWriter

from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment

from utils import get_dataset

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

if __name__ == "__main__":
    device = 'cuda'
    
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--name", type=str, help="name of training module")
    parser.add_argument("--checkpoint_dir", type=str, help="path checkpoint dir")
    parser.add_argument("--image_dir", type=str, help="path to the image")
    parser.add_argument("--label_dir", type=str, help="path to the label")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers for loading data")
    parser.add_argument("--load_pretrain", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
    parser.add_argument("--n_sample", type=int, default=4, help="number of the samples generated during training")
    parser.add_argument("--size", type=int, default=512, help="image sizes for the model")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
    parser.add_argument("--path_batch_shrink", type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory consumption)")
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization")
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument("--augment_p", type=float, default=0, help="probability of applying augmentation. 0 = use adaptive augmentation")
    parser.add_argument("--ada_target", type=float, default=0.6, help="target augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_length", type=int, default=500 * 1000, help="target duraing to reach augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_every", type=int, default=256, help="probability update interval of the adaptive augmentation")

    args = parser.parse_args()
    
    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    if not os.path.exists(os.path.join(args.checkpoint_dir, args.name)):
        os.makedirs(os.path.join(args.checkpoint_dir, args.name))
        
    _model_path = os.path.join(args.checkpoint_dir, args.name)
    
    writer = SummaryWriter(os.path.join(_model_path, "logs"))
    
    if not os.path.exists(os.path.join(_model_path, "sample")):
        os.makedirs(os.path.join(_model_path, "sample"))
    _sample_path = os.path.join(_model_path, "sample")
    
    generator = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    
    g_ema.eval()
    accumulate(g_ema, generator, 0)
    start_epoch = 0
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    
    g_optim = optim.Adam(generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))
    
    if args.load_pretrain is not None:
        print("load model:", args.load_pretrain)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.load_pretrain)
            _ckpt_name = os.path.splitext(ckpt_name)[0]
            start_epoch = int(_ckpt_name.split("_")[0])
            #args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])
        
    dataset = get_dataset(args.image_dir, args.label_dir, args.batch, args.num_workers, 512, isTrain=True)
    dataset_size = len(dataset)
    print("Training images = ", dataset_size)
    
    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}
    
    g_module = generator
    d_module = discriminator
    
    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0
    
    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    
    num_epochs = args.epochs
    for epoch in range(start_epoch, num_epochs):
        for batch_idx, data in enumerate(dataset):
            i = epoch * dataset_size + batch_idx
            real_img = data["image"].to(device)
            requires_grad(generator, False)
            requires_grad(discriminator, True)

            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img, _ = generator(noise)

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)
                fake_img, _ = augment(fake_img, ada_aug_p)
            else:
                real_img_aug = real_img
    
            fake_pred = discriminator(fake_img)
            real_pred = discriminator(real_img_aug)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            loss_dict["d"] = d_loss
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()

            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()

            if args.augment and args.augment_p == 0:
                ada_aug_p = ada_augment.tune(real_pred)
                r_t_stat = ada_augment.r_t_stat

            d_regularize = i % args.d_reg_every == 0
            
            if d_regularize:
                real_img.requires_grad = True

                if args.augment:
                    real_img_aug, _ = augment(real_img, ada_aug_p)

                else:
                    real_img_aug = real_img

                real_pred = discriminator(real_img_aug)
                r1_loss = d_r1_loss(real_pred, real_img)

                discriminator.zero_grad()
                (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

                d_optim.step()

            loss_dict["r1"] = r1_loss

            requires_grad(generator, True)
            requires_grad(discriminator, False)
            
            requires_grad(generator, True)
            requires_grad(discriminator, False)

            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img, _ = generator(noise)

            if args.augment:
                fake_img, _ = augment(fake_img, ada_aug_p)

            fake_pred = discriminator(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred)

            loss_dict["g"] = g_loss

            generator.zero_grad()
            g_loss.backward()
            g_optim.step()

            g_regularize = i % args.g_reg_every == 0
            
            if g_regularize:
                path_batch_size = max(1, args.batch // args.path_batch_shrink)
                noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
                fake_img, latents = generator(noise, return_latents=True)

                path_loss, mean_path_length, path_lengths = g_path_regularize(
                    fake_img, latents, mean_path_length
                )

                generator.zero_grad()
                weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

                if args.path_batch_shrink:
                    weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

                weighted_path_loss.backward()

                g_optim.step()

                mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
                )

            loss_dict["path"] = path_loss
            loss_dict["path_length"] = path_lengths.mean()

            accumulate(g_ema, g_module, accum)

            loss_reduced = reduce_loss_dict(loss_dict)

            d_loss_val = loss_reduced["d"].mean().item()
            g_loss_val = loss_reduced["g"].mean().item()
            r1_val = loss_reduced["r1"].mean().item()
            path_loss_val = loss_reduced["path"].mean().item()
            real_score_val = loss_reduced["real_score"].mean().item()
            fake_score_val = loss_reduced["fake_score"].mean().item()
            path_length_val = loss_reduced["path_length"].mean().item()
                
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}, Batch_Idx: {batch_idx}, Total_Iter: {i}, G_Loss: {g_loss_val:.4f}, D_Loss: {d_loss_val:.4f},\
                    R1_Loss: {r1_val:.4f}, Path: {path_loss_val:.4f}, Mean_Path: {mean_path_length_avg:.4f}, Augment: {ada_aug_p:.4f}")
                writer.add_scalar("d_loss_val", d_loss_val, i)
                writer.add_scalar("g_loss_val", g_loss_val, i)
                writer.add_scalar("r1_val", r1_val, i)
                writer.add_scalar("path_loss_val", path_loss_val, i)
                writer.add_scalar("real_score_val", real_score_val, i)
                writer.add_scalar("fake_score_val", fake_score_val, i)
                writer.add_scalar("path_length_val", path_length_val, i)
                
                
            if batch_idx % 1000 == 0:
                g_ema.eval()
                sample, _ = g_ema([sample_z])
                utils.save_image(sample, os.path.join(_sample_path, f"e_{str(epoch).zfill(3)}_idx_{str(batch_idx).zfill(6)}.png") , nrow=int(args.n_sample ** 0.5), normalize=True, range=(-1, 1))
            
            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    os.path.join(_model_path, f"{str(epoch).zfill(3)}_{str(batch_idx).zfill(6)}.pt"),
                )
                
    
    