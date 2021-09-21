import time
from collections import OrderedDict
import numpy as np
import torch
import config as cfg
import utils as util

isTrain = True

dataset = util.get_dataset(cfg.image_dir, cfg.label_dir, cfg.batch_size, cfg.num_workers, True)




