import os
import torch
import argparse


name = "slv07_progan"
checkpoint_dir = "checkpoints/"
device = "cuda" if torch.cuda.is_available() else 'cpu'
# Path to image dir
image_dir = "../edn_data/slv07_rz/train/train_img/" 
# Path to label dir
label_dir = "../edn_data/slv07_rz/train/train_label"
# Number of epochs
num_epochs = 100
# Number of workers
num_workers = 3
batch_size=1
# ProGAN specific parameters
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 8, 4, 2, 1]
CHANNELS_IMG = 3
Z_DIM = 512  # should be 512 in original paper
IN_CHANNELS = 512  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [100] * len(BATCH_SIZES) # [30, 40, 50, 60, 70, 80, 90, 100] # [30] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(1, Z_DIM, 1, 2).to(device)
START_TRAIN_AT_IMG_SIZE = 4
LOAD_MODEL = False

