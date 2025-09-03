GPU_NUM = 0
GPU_NUM = str(GPU_NUM)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM

import shutil
import warnings
import contextlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations
from IPython.display import clear_output

warnings.filterwarnings(action='ignore')
plt.style.use('seaborn-v0_8-white')
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

hp = HP_Guhong(name='Guhong_MoGLo', device='cuda')
hp.epoch_load = False
torch.save(hp, f'../res/{hp.name}/options.pt')
hp.__dict__

transforms = Transforms_Bundle(seq=hp.seq, device=hp.device)
train_set = Dataset_guhong(hp, phase='train', type_X=hp.type_X, type_y=hp.type_y)
valid_set = Dataset_guhong(hp, phase='valid', type_X=hp.type_X, type_y=hp.type_y)
train_loader = DataLoader(dataset=train_set, shuffle=True , batch_size=hp.batch_size, 
                          num_workers=4, pin_memory=True)
valid_loader = DataLoader(dataset=valid_set, shuffle=False, batch_size=hp.batch_size, 
                          num_workers=4, pin_memory=True)
data_loader = [train_loader, valid_loader]
model = MoGLo_Net(dim_in=1, dim_base=hp.dim_base, c_att=True, gl_att=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=hp.optimizer_lr, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.scheduler_step, gamma=hp.scheduler_gamma)

model = model.to(hp.device)
print(f'Model: {hp.name}')

# Loss & Rate(Metric)
MME_func = MMAE(alpha=1, smooth=2)
COR_func = Corr_loss(alpha=1)
TRI_func = Triplet_Loss(alpha=0.005, margin=0.1, n_sample=None, dist='cos')
MAE_func = MAE()

loss = {'MME':MME_func, 
        'COR':COR_func, 
        'TRI':TRI_func}
rate = {'MAE':MAE_func}

loss_keys = ['MME', 'COR', 'TRI']
rate_keys = ['MAE']

# Continual Learning
epoch_s, epoch_e = 1, hp.epochs+1
scaler = torch.cuda.amp.GradScaler()
loss_epoch = torch.zeros([epoch_e, 2, len(loss_keys)])
rate_epoch = torch.zeros([epoch_e, 2, len(rate_keys)])
if hp.epoch_load:
    last_point = torch.load(f'../res/{hp.name}/model/last_point.pt')
    model.load_state_dict(last_point['model'])
    optimizer.load_state_dict(last_point['optimizer'])
    scaler.load_state_dict(last_point['scaler'])
    scheduler.load_state_dict(last_point['scheduler'])
    epoch_s = last_point['epoch']
    history = last_point['history']
    loss_epoch[:epoch_s] = history['loss'][:epoch_s]
    rate_epoch[:epoch_s] = history['rate'][:epoch_s]
    print(f'※ Continual Learning: {epoch_s-1} Epoch ※')


# Fit
scaler = torch.cuda.amp.GradScaler()
loss_epoch = torch.zeros([epoch_e, 2, len(loss_keys)])
rate_epoch = torch.zeros([epoch_e, 2, len(rate_keys)])

for epoch in range(epoch_s, epoch_e):
    for i, phase in enumerate(['Train', 'Valid']):
        if phase=='Train':
            model.train()
            context_manager = contextlib.nullcontext()
        if phase=='Valid':
            model.eval()
            context_manager = torch.no_grad()
        
        loss_batch = torch.zeros([len(data_loader[i]), len(loss_keys)])
        rate_batch = torch.zeros([len(data_loader[i]), len(rate_keys)])
        with context_manager:
            for k, data in enumerate(data_loader[i]):
                B, y = data[0].to(hp.device), data[1].to(hp.device)
                if len(B)<=1: continue
                
                # Forward
                y = transforms.y_scaling(y)
                
                with torch.cuda.amp.autocast():
                    p, p_emb, at_score = model(B, torch.zeros([0]))
                    
                    MME_res = (MME_func(p[0], y) + MME_func(p[1], y))/2
                    COR_res = (COR_func(p[0], y) + COR_func(p[1], y))/2
                    TRI_res = TRI_func(p_emb, y)
                    loss_final = MME_res+COR_res+TRI_res
                    
                    loss_batch[k, 0] = MME_res.item()
                    loss_batch[k, 1] = COR_res.item()
                    loss_batch[k, 2] = TRI_res.item()
                
                # Backward
                if phase=='Train':
                    optimizer.zero_grad()
                    scaler.scale(loss_final).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                # Rate
                with torch.no_grad():
                    pred = transforms.y_scaling_inv(p[1])
                    true = transforms.y_scaling_inv(y)
                    pred[:, :, :3] *= 10
                    true[:, :, :3] *= 10
                    MAE_res  =  MAE_func(pred[:, -1, :], true[:, -1, :])

                    rate_batch[k, 0] = MAE_res.item()
        
        loss_epoch[epoch, i] = torch.mean(loss_batch.cpu(), axis=0)
        rate_epoch[epoch, i] = torch.mean(rate_batch.cpu(), axis=0)
    
    # Scheduler
    if scheduler is not None: scheduler.step()
    
    # Monitoring
    if epoch==1:
        print(f'===== Loss Monitoring =====')
        print(f'Loss: {loss_keys}', end=' ')
        print(f'rate: {rate_keys}', end=' ')
        print(f'(Train, Valid)')
    if epoch%hp.monitoring_cycle==0:
        print(f'{epoch:5.0f}/{hp.epochs:5.0f}', end=' ')
        for l in range(len(loss_keys)):
            loss_train = loss_epoch[epoch, 0, l]
            loss_valid = loss_epoch[epoch, 1, l]
            loss_ratio = (loss_train/loss_valid)*100
            print(f'({loss_train:6.4f}, {loss_valid:6.4f})', end=f' | ')
        print('*', end=' ')
        for l in range(len(rate_keys)):
            rate_train = rate_epoch[epoch, 0, l]
            rate_valid = rate_epoch[epoch, 1, l]
            rate_ratio = (rate_train/rate_valid)*100
            print(f'({rate_train:6.4f}, {rate_valid:6.4f})', end=f' | ')
        print()
        
        # Save
        history = {'loss':loss_epoch, 
                   'rate':rate_epoch, 
                   'loss_keys':loss_keys, 
                   'rate_keys':rate_keys}
        torch.save(history, f'{hp.path_model}/history.pt')
        
        if epoch%hp.save_cycle==0:
            torch.save(model.state_dict(), f'{hp.path_model}/model_{epoch}.pt')
            last_point = {'epoch':epoch, 
                          'history':history, 
                          'model':model.state_dict(), 
                          'optimizer':optimizer.state_dict(), 
                          'scaler':scaler.state_dict(), 
                          'scheduler':scheduler.state_dict()}
            torch.save(last_point, f'{hp.path_model}/last_point.pt')
    
torch.cuda.empty_cache()