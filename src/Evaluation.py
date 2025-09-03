GPU_NUM = 0
GPU_NUM = str(GPU_NUM)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM

import gc
import glob
import time
import h5py
import random
import shutil
import natsort
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations
from IPython.display import clear_output

warnings.filterwarnings(action='ignore')
# plt.style.use('seaborn-v0_8-white')
plt.rcParams['image.cmap'] = 'gray'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader, Dataset

print(torch.cuda.is_available(), ': ', torch.cuda.get_device_name(0))

from utils.utils import *
from networks.networks import *
from options.hyper_parameters import *

plot_history('Guhong_MoGLo')

inference(HP_Guhong('Guhong_MoGLo', seq=5), epoch=1000, batch_size_inference=8)

list_model = ['Guhong_MoGLo']
print_total_v2(list_model, [1000])