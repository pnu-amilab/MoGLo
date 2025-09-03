import os
import gc
import glob
import h5py
import math
import random
import natsort
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations
from IPython.display import clear_output
from scipy.spatial.transform import Rotation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset

from utils.utils_recon import *
from networks.networks import *

def listdir(path):
    return natsort.natsorted(os.listdir(path))

def globsort(path):
    return natsort.natsorted(glob.glob(path))

def make_dir(path):
    if not os.path.exists(path): 
        os.mkdir(path)
        os.chmod(path, 0o777)

def print_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{"Params #":<30}: {total_params:,}')

def img_interp(img, scale_factor):
    expand = nn.Upsample(scale_factor=(scale_factor, 1, 1), mode='trilinear', align_corners=True)
    img = img.unsqueeze(0).unsqueeze(0)
    img = expand(img)
    
    return img.squeeze()

def lee_filter(image, size=3, epsilon=1e-5):
    mean = F.avg_pool2d(image, kernel_size=size, stride=1, padding=size//2)
    mean_square = F.avg_pool2d(image ** 2, kernel_size=size, stride=1, padding=size//2)
    variance = mean_square - mean ** 2

    overall_variance = variance.mean()
    weights = variance / (variance + overall_variance + epsilon)

    result = mean + weights * (image - mean)

    return result

def get_y_scale():
    list_patient = listdir('../../US_Data/Forearm_Main')
    list_scan = listdir('../../US_Data/Forearm_Main/001')

    y = []
    for patient in tqdm(list_patient):
        for scan in list_scan:
            data = h5py.File(f'../../US_Data/Forearm_Main/{patient}/{scan}', 'r')
            y_ = np.array(data['emm_y'][:])
            data.close()
            y.append(y_)
            
    y = np.array(y)
    dy = np.concatenate(np.array([y[:, i+1] - y[:, i] for i in range(y.shape[1]-1)]), axis=0)
    my, My = np.zeros([6]), np.zeros([6])
    plt.figure(figsize=(14, 14))
    for i in range(6):
        plt.subplot(4, 3, i+1)
        plt.plot(dy[:, i], lw=0.7, color='r', alpha=0.5)
        my[i] = np.percentile(-np.abs(dy[:, i]), 2)
        My[i] = np.percentile(+np.abs(dy[:, i]), 98)
        plt.title(f'({my[i]:0.4f}, {My[i]:0.4f})')
        plt.axhline(-1, color='black', ls='--', lw=0.6)
        plt.axhline(+1, color='black', ls='--', lw=0.6)

    y_scale = {'My':My, 
               'my':my}
    torch.save(y_scale, './utils/y_scale.pt')
    print(My)

    dy = dy/My
    for i in range(6):
        plt.subplot(4, 3, i+1+6)
        plt.plot(dy[:, i], lw=0.7, color='r', alpha=0.5)
        my[i] = np.percentile(-np.abs(dy[:, i]), 2)
        My[i] = np.percentile(+np.abs(dy[:, i]), 98)
        plt.title(f'({my[i]:0.4f}, {My[i]:0.4f})')
        plt.axhline(-1, color='black', ls='--', lw=0.6)
        plt.axhline(+1, color='black', ls='--', lw=0.6)

def aug_stop(data, k, seq=5, type_X='img_B', type_y='emm_dy'):
    N = seq
    list_idx_X = [i for i in range(k, k+N)]
    list_idx_y = [i for i in range(k, k+N-1)]

    l = np.random.randint(1, N)
    # l = N-1
    list_idx_c = natsort.natsorted(random.sample([i for i in range(1, N)], l))
    for i in list_idx_c:
        list_idx_X = list_idx_X[:i] + list_idx_X[i-1:i] + list_idx_X[i:-1]
        list_idx_y = list_idx_y[:i-1] + [-1] + list_idx_y[i-1:-1]
    
    X = np.zeros([N, 1, 256, 256], dtype=np.float32)
    y = np.zeros([N-1, 6], dtype=np.float32)
    for n in range(N):
        X[n] = data[type_X][list_idx_X[n]]
        if (n!=N-1): 
            if (list_idx_y[n]!=-1):
                y[n] = data[type_y][list_idx_y[n]]
    
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    
    return X, y

def aug_skip(data, k, seq=5, s=1, type_X='img_B', type_y='emm_dy'):
    if k>(950-5*(s+1)): k -= 5*(s+1)
    N = seq
    list_idx_X = np.array([i for i in range(k, k+N)])
    list_idx_y = np.array([i for i in range(k, k+N-1)])
    
    l = np.random.randint(1, N)
    l = N-1
    list_idx_c = natsort.natsorted(random.sample([i for i in range(1, N)], l))
    for i in list_idx_c:
        list_idx_X[i:] += s
        list_idx_y[i:] += s
    
    X = np.zeros([N, 1, 256, 256], dtype=np.float32)
    y = np.zeros([N-1, 6], dtype=np.float32)
    for n in range(N):
        X[n] = data[type_X][list_idx_X[n]]
        if (n!=N-1):
            if n+1 not in list_idx_c:
                y_temp = data[type_y][list_idx_y[n]]
                y[n] = y_temp
            else:
                y_temp = torch.from_numpy(data[type_y][list_idx_y[n]:list_idx_y[n]+s+1])
                y_temp = transform_cum(y_temp)[-1]
                y[n] = y_temp
    
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    
    return X, y

class Dataset_guhong(Dataset):
    def __init__(self, hp, phase='train', type_X='img_B', type_y='y_rel'):
        self.hp = hp
        self.phase = phase
        self.seq = hp.seq
        self.type_X = type_X
        self.type_y = type_y
        
        list_patient = [str(i).zfill(3) for i in hp.split_patient[phase]]
        list_type_scan = hp.split_type_scan[phase]
        list_path_scan = []
        for patient in list_patient:
            for LR in ['LH', 'RH']:
                for type_scan in list_type_scan:
                    list_path_scan.append(f'{hp.path_dataset}/{patient}/{LR}_{type_scan}_{hp.type_angle}.h5')
        
        self.list_data = [h5py.File(path, 'r') for path in list_path_scan]
        self.list_len_data = [len(self.list_data[i]['img_B']) for i in range(len(self.list_data))]
        
        self.n = len(self.list_data)
        self.N = len(self.list_data)*hp.alpha
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        idx = idx % self.n
        data = self.list_data[idx]
        k = np.random.randint(0, self.list_len_data[idx]-self.seq)
        
        img_B = torch.from_numpy(data[self.type_X][k:k+self.seq])/255
        emt_y = torch.from_numpy(data[self.type_y][k:k+self.seq-1])
        
        return img_B, emt_y
    
class Transforms_Bundle():
    def __init__(self, seq=5, device='cuda', y_scaling=True):
        self.seq = seq
        self.y_scale = torch.load(f'{os.getcwd()}/utils/y_scale_Forearm_Main.pt')
        self.My = torch.tensor(self.y_scale['M_emp_dy'], dtype=torch.float32).to(device).reshape(1, 1, 6)
        if y_scaling==False:
            self.My = torch.ones_like(self.My)
    
    def y_scaling(self, y_rel):
        return y_rel/self.My
    
    def y_scaling_inv(self, y_rel):
        return y_rel*self.My

# Loss & Metric
class MAE(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, p, y, mse=False):
        p = p.flatten()
        y = y.flatten()
        
        if mse==True:
            res = torch.mean((p-y)**2)
        else:
            res = torch.mean(torch.abs(p-y))
        
        return res*self.alpha

# Motion-based MAE
class MMAE(nn.Module):
    def __init__(self, alpha=1, beta=1.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, p, y, w=None, per=False, mse=False):
        res = torch.abs(p-y)
        if mse: res = res**2
        if per: res = res/(torch.abs(y)+1e-5)
        if w is not None:
            if w.shape[1]!=y.shape[1]: w = torch.concat([torch.zeros([y.shape[0], 1, 6]).to(y.device), w], axis=1)
            res =  res*((torch.abs(w)+self.smooth)**self.beta)
        
        res = res.mean()
        
        return res*self.alpha

class Corr_loss(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, p, y):
        corr = F.cosine_similarity(p.flatten(), y.flatten(), dim=0)
        res = 1-corr

        return res*self.alpha
    
class Triplet_Loss(nn.Module):
    def __init__(self, alpha=1, margin=0.10, n_sample=2, dist='cos'):
        super().__init__()
        '''
        p: Sampling %
        bs: Batch*Seq
        p_bs: sampling number
        n_sample: [None, num]
        '''
        self.alpha = alpha
        self.margin = margin
        self.n_sample = n_sample
        if dist=='cos':
            self.dist = self._cossim
        if dist=='norm':
            self.dist = self._norm
    
    def _norm(self, a, b):
        return torch.linalg.norm(a-b, axis=-1)
    
    def _cossim(self, a, b):
        return 1-F.cosine_similarity(a, b, axis=-1)
    
    def forward(self, v, y):
        B, S = y.shape[0], y.shape[1]
        
        v = v.reshape(B, S, -1)
        v = v[:, -1, :]
        y = y[:, -1, :]
        
        combs = list(combinations([i for i in range(v.shape[0])], 3))
        if len(combs)==0: return torch.tensor(0.0).to(v.dtype)
        
        if self.n_sample!=None:
            if len(combs)<self.n_sample:
                return torch.tensor(0.0).to(v.dtype)
            else:
                combs = random.sample(combs, self.n_sample)
            
        Y = torch.stack([y[[combs[i]]] for i in range(len(combs))])
        V = torch.stack([v[[combs[i]]] for i in range(len(combs))])
        
        D1 = self.dist(Y[:, 0], Y[:, 1])
        D2 = self.dist(Y[:, 0], Y[:, 2])
        PN = torch.sign(D2-D1)
        
        res = PN*(self._norm(V[:, 0], V[:, 1]) - self._norm(V[:, 0], V[:, 2])) + self.margin
        res = torch.mean(F.relu(res))
        
        return res*self.alpha

class Corr(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, p, y):
        corr = F.cosine_similarity(p, y, axis=1).mean()

        return corr

class Recon_Frame(nn.Module):
    def __init__(self, alpha=1, size=256, skip=8):
        super().__init__()
        self.alpha = alpha
        self.size = size
        self.skip = skip
        
    def forward(self, y):
        y = torch.tensor(y)
        y_mat = transform_a2m(y).to(y.dtype)
        points = torch.tensor([[i, j, 0] for i in range(0, self.size, self.skip) for j in range(0, self.size, self.skip)], dtype=torch.float32)
        points = points*(38/self.size)
        calib_vec = torch.tensor([20, 0, 7], dtype=torch.float32)
        points = points - calib_vec
        
        y_recon = torch.matmul(y_mat, points.T)
        y_recon = y_recon + y[:, :3].unsqueeze(-1)
        
        return y_recon
    
class Frame_Error(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.size = 256
        self.skip = 4
        
    def forward(self, p, y):
        p_mat = transform_a2m(p)
        y_mat = transform_a2m(y)
        
        points = torch.tensor([[i, j, 0] for i in range(0, self.size, self.skip) for j in range(0, self.size, self.skip)], dtype=torch.float32)
        points = points*(3.8/self.size)
        calib_vec = torch.tensor([2, 0, 0.7], dtype=torch.float32)
        points = points - calib_vec
        
        p_recon = torch.matmul(p_mat, points.T) + p[:, :3].unsqueeze(-1)
        y_recon = torch.matmul(y_mat, points.T) + y[:, :3].unsqueeze(-1)
        error = torch.sum((p_recon-y_recon)**2, axis=1)**0.5
        
        error = error
        
        error = torch.mean(error, axis=1)*self.alpha
        error_T = torch.mean(error)*self.alpha
        
        return error_T, error

class ACC(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha 
        
    def forward(self, y_pred, y_true):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        
        y_pred = (y_pred>self.alpha).to(torch.float32)
        y_true = (y_true>self.alpha).to(torch.float32)
        
        correct = (y_pred==y_true).sum()
        acc = correct / len(y_true)
        
        return acc

def get_ylim(loss, title, alpha=2, margin=0.1):
    m = loss.min()
    M = loss.max()
    a = loss.mean()
    
    if M-a>(a*alpha):
        M = a*alpha
        
    m = m*(1-margin)
    M = M*(1+margin)
    
    l = len(loss)//2
    
    if torch.sum(loss[0:10][0])>torch.sum(loss[-10:][0]): 
        plt.axhline(loss[:, 1].min(), color='black', linewidth=0.7, linestyle='--', alpha=0.7)
        epoch = torch.argmin(loss[:, 1])
        plt.axvline(epoch, color='r', linewidth=0.7, linestyle='--', alpha=0.7)
        plt.title(f'{title}: {loss[:, 1].min():0.4f} ({loss[l:, 1].mean():0.4f})')
    else:
        plt.axhline(loss[:, 1].max(), color='black', linewidth=0.7, linestyle='--', alpha=0.7)
        epoch = torch.argmax(loss[:, 1])
        plt.axvline(epoch, color='r', linewidth=0.7, linestyle='--', alpha=0.7)
        plt.title(f'{title}: {loss[:, 1].max():0.4f} ({loss[l:, 1].mean():0.4f})')

    return m, M

def plot_history(name, epoch=0, loss_titles=None, rate_titles=None, figsize=[14, 5]):
    hp = torch.load(f'../res/{name}/options.pt')
    history = torch.load(f'../res/{hp.name}/model/history.pt')
    loss_keys = history['loss_keys'][:2]
    rate_keys = history['rate_keys']
    if 'Loss_Final' in loss_keys: loss_keys.remove('Loss_Final')
    
    loss = history['loss'][1:]
    rate = history['rate'][1:]
    k = len(loss[:, 1, 0][loss[:, 1, 0]>0])
    loss = loss[:k, :, :]
    rate = rate[:k, :, :]
    
    if loss_titles==None:
        loss_titles = loss_keys
    if rate_titles==None:
        rate_titles = rate_keys
        
    N = len(loss_keys) + len(rate_keys)
    n, m = N//3+1, 3
    
    plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle(f'{hp.name} ({len(loss)})')
    for i in range(len(loss_titles)):
        plt.subplot(n, m, i+1)
        plt.plot(loss[:, 0, i], color='b', linewidth=0.7, alpha=0.7, label='Train')
        plt.plot(loss[:, 1, i], color='r', linewidth=0.7, alpha=0.7, label='Valid')
        plt.ylim(get_ylim(loss[:, :, i], loss_titles[i], alpha=1.5))
        if i==2: plt.legend(loc='upper right')

    for j in range(len(rate_titles)):
        plt.subplot(n, m, len(loss_titles)+j+1)
        plt.plot(rate[:, 0, j], color='b', linewidth=0.7, alpha=0.7)
        plt.plot(rate[:, 1, j], color='r', linewidth=0.7, alpha=0.7)
        plt.ylim(get_ylim(rate[:, :, j], rate_titles[j], alpha=1.5))
        
class Dataset_guhong_inference(Dataset):
    def __init__(self, hp, imgs):
        self.hp = hp
        self.imgs = imgs
        self.seq_inference = hp.seq_inference
        
    def __len__(self):
        return len(self.imgs)-self.seq_inference+1
    
    def __getitem__(self, idx):
        seq = self.imgs[idx:idx+self.seq_inference]/255
        
        return seq

def inference(hp, epoch, batch_size_inference=4, clear=False, y_scaling=True):
    make_dir(f'{hp.path_inference}/{epoch}')
    make_dir(f'{hp.path_evaluation}/{epoch}')
    make_dir(f'{hp.path_fig}/{epoch}')
    model = MoGLo_Net(dim_base=hp.dim_base)
    model.load_state_dict(torch.load(f'{hp.path_model}/model_{epoch}.pt', map_location='cpu'))
    model = model.eval().cuda()
    recon_frame = Recon_Frame(skip=16)
    
    transforms = Transforms_Bundle(seq=hp.seq, device=hp.device, y_scaling=y_scaling)
    list_patient = [str(i).zfill(3) for i in hp.split_patient['test']]
    list_type_scan = hp.split_type_scan['test']
    
    list_scan = []
    for patient in list_patient:
        for LR in ['LH', 'RH']:
            for type_scan in list_type_scan:
                list_scan.append(f'{patient}/{LR}_{type_scan}_{hp.type_angle}.h5')
    
    print(f'{hp.name:<36} {"rAE":>8} {"aAE":>8} {"rFE":>8} {"aFE":>8} {"FDR":>8} {"Corr":>8}')
    Res = []
    list_rAE_T = []
    list_aAE_T = []
    list_rFE = []
    list_aFE = []
    list_FD = []
    list_FDR = []
    list_corr_recon_AT = []
    for n, scan in enumerate(list_scan):
        print(f'[{n+1:2.0f}/{len(list_scan):2.0f}]: {scan} ', end='[')
        path = f'{hp.path_dataset}/{scan}'
        name = f'{path[-18:-15]}_{path[-14:-3]}'
        data = h5py.File(path, 'r')
        img_B = torch.tensor(data['img_B'][:], dtype=torch.float32).squeeze()
        y_rel = torch.tensor(data['y_rel'][:], dtype=torch.float32)
        y_abs = torch.tensor(data['y_abs'][:], dtype=torch.float32)
        data.close()
        
        pad = img_B[0:1].repeat(hp.seq_inference-2, 1, 1)
        img_B = torch.concat([pad, img_B], axis=0).unsqueeze(1)
        
        test_set = Dataset_guhong_inference(hp, img_B)
        test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=batch_size_inference, 
                                 num_workers=4, pin_memory=True)
        
        print(f'P', end='')
        p_rel = []
        with torch.no_grad():
            for x in test_loader:
                p, emb, at_score = model(x.to(hp.device))
                p1, p2 = p[0], p[1]
                p = (p1+p2)/2
                p = transforms.y_scaling_inv(p)
                p_rel.append(p.cpu())
        p_rel = torch.concat(p_rel, axis=0)
        p_rel = p_rel[:, -1]
        p_abs = transform_acum(p_rel.unsqueeze(0)).squeeze()
        
        y_rel[:, :3] *= 10
        p_rel[:, :3] *= 10
        y_abs[:, :3] *= 10
        p_abs[:, :3] *= 10
        
        y_recon_R = recon_frame(y_rel)
        p_recon_R = recon_frame(p_rel)
        y_recon_A = recon_frame(y_abs)
        p_recon_A = recon_frame(p_abs)
        
        res = {'y_rel':y_rel, 
               'p_rel':p_rel, 
               'y_abs':y_abs, 
               'p_abs':p_abs, 
               'y_recon_R':y_recon_R, 
               'p_recon_R':p_recon_R, 
               'y_recon_A':y_recon_A, 
               'p_recon_A':p_recon_A} 
        torch.save(res, f'{hp.path_inference}/{epoch}/{name}.pt')
        
        print(f'E', end='')
        rAE = torch.mean(torch.abs(p_rel-y_rel), axis=0)
        aAE = torch.mean(torch.abs(p_abs-y_abs), axis=0)
        rAE_T = torch.mean(rAE)
        aAE_T = torch.mean(aAE)

        corr_R = F.cosine_similarity(p_rel, y_rel, axis=0)
        corr_A = F.cosine_similarity(p_abs, y_abs, axis=0)
        corr_recon_R = F.cosine_similarity(p_recon_R, y_recon_R, axis=0)
        corr_recon_A = F.cosine_similarity(p_recon_A, y_recon_A, axis=0)
        corr_RT = torch.mean(corr_R)
        corr_AT = torch.mean(corr_A)
        corr_recon_RT = torch.mean(corr_recon_R)
        corr_recon_AT = torch.mean(corr_recon_A)
        
        rFE = torch.mean(torch.sum((p_recon_R-y_recon_R)**2, axis=1)**0.5, axis=-1)
        aFE = torch.mean(torch.sum((p_recon_A-y_recon_A)**2, axis=1)**0.5, axis=-1)

        FD = aFE[-1]
        rFE = torch.mean(rFE)
        aFE = torch.mean(aFE)
        y_temp = y_recon_A[:, :, y_recon_A.shape[-1]//2]
        dist = torch.sum((y_temp[0]-y_temp[-1])**2)**0.5
        FDR = (FD/dist)*100
        res = {'rAE':rAE, 
               'aAE':aAE, 
               'rAE_T':rAE_T, 
               'aAE_T':aAE_T, 
               'corr_R':corr_R, 
               'corr_A':corr_A, 
               'corr_recon_R':corr_recon_R, 
               'corr_recon_A':corr_recon_A, 
               'corr_RT':corr_RT, 
               'corr_AT':corr_AT, 
               'corr_recon_RT':corr_recon_RT, 
               'corr_recon_AT':corr_recon_AT, 
               'rFE':rFE, 
               'aFE':aFE, 
               'FD':FD, 
               'FDR':FDR}
        torch.save(res, f'{hp.path_evaluation}/{epoch}/{name}.pt')
        
        print(f'F] || ', end='')
        dict_ylim = {'rAE':0.1, 
                     'aAE':10, 
                     'rFE':0.5, 
                     'aFE':50, 
                     'Corr':1.0, 
                     'FD':100}
        plt.figure(figsize=(14, 14))
        plt.suptitle(f'{name}: {rAE_T:1.4f}  {aAE_T:1.4f}  {rFE:1.4f}  {aFE:2.4f}  {FDR:2.4f}  {corr_recon_AT:8.4f}')
        for i in range(6):
            plt.subplot(4, 3, i+1)
            plt.plot(y_rel[:, i], color='b', lw=0.8, label='true')
            plt.plot(p_rel[:, i], color='r', lw=0.8, label='pred')
            M = torch.abs(torch.concat([y_rel[:, i], p_rel[:, i]])).max()
            plt.ylim(-M, M)
            plt.axhline(0, color='black', lw=0.8, ls='--', alpha=0.8)
            if i==0: plt.legend(loc='upper right')
        for i in range(6):
            plt.subplot(4, 3, i+7)
            plt.plot(y_abs[:, i], color='b', lw=0.8, label='true')
            plt.plot(p_abs[:, i], color='r', lw=0.8, label='pred')
            M = torch.abs(torch.concat([y_abs[:, i], p_abs[:, i]])).max()
            plt.ylim(-M, M)
            plt.axhline(0, color='black', lw=0.8, ls='--', alpha=0.8)
        plt.savefig(f'{hp.path_fig}/{epoch}/{name}.png')
        plt.close()
        
        print(f'{rAE_T:8.4f} {aAE_T:8.4f} {rFE:8.4f} {aFE:8.4f} {FDR:8.4f} {corr_recon_AT:8.4f}')
        Res.append(res)
        list_rAE_T.append(rAE_T)
        list_aAE_T.append(aAE_T)
        list_rFE.append(rFE)
        list_aFE.append(aFE)
        list_FD.append(FD)
        list_FDR.append(FDR)
        list_corr_recon_AT.append(corr_recon_AT)
    list_rAE_T = np.array(list_rAE_T)
    list_aAE_T = np.array(list_aAE_T)
    list_rFE = np.array(list_rFE)
    list_aFE = np.array(list_aFE)
    list_FD = np.array(list_FD)
    list_FDR = np.array(list_FDR)
    list_corr_recon_AT = np.array(list_corr_recon_AT)
    print('='*99)
    print(f'{"@ Total @":<37}', end='')
    print(f'{list_rAE_T.mean():8.4f} {list_aAE_T.mean():8.4f} {list_rFE.mean():8.4f} {list_aFE.mean():8.4f} {list_FDR.mean():8.4f} {list_corr_recon_AT.mean():8.4f}')
    
    plt.figure(figsize=(14, 9))
    plt.suptitle(f'{hp.name}[{epoch}] | FDR: {np.mean(list_FDR):0.4f}')
    titles = ['rAE', 'aAE', 'rFE', 'aFE', 'Corr', 'FD']
    for i, key in enumerate(['rAE_T', 'aAE_T', 'rFE', 'aFE', 'corr_AT', 'FD']):
        plt.subplot(3, 2, i+1)
        res = []
        for j in range(len(Res)):
            res.append(Res[j][key])
        plt.stem(res)
        plt.title(f'{titles[i]}: {np.mean(res):0.4f}')
        plt.axhline(np.mean(res), color='r', lw=0.8, ls='--', alpha=0.8)
        plt.ylim(0, dict_ylim[titles[i]])
    plt.savefig(f'{hp.path_Res}/Fig/{hp.name}_{epoch}.png')
    plt.close()
    print()
    torch.cuda.empty_cache()
    if clear: clear_output() 
    
def print_total(list_model=None, list_epoch=None, all=False):
    if list_model==None:
        list_model = listdir('../res')
        list_model.remove('Fig')
        list_model.remove('Fig_Recon')

    n = 0
    for model in list_model:
        if model=='--': 
            print('-'*82)
            continue
        
        if list_epoch==None:
            list_epoch = listdir(f'../res/{model}/evaluation')
        
        for epoch in list_epoch:
            n += 1
            rAE, aAE = [], []
            rAE_T, aAE_T = [], []
            corr_R, corr_A, corr_recon_R, corr_recon_A = [], [], [], []
            corr_RT, corr_AT, corr_recon_RT, corr_recon_AT = [], [], [], []
            rFE, aFE, FD, FDR = [], [], [], []
            list_eval = listdir(f'../res/{model}/evaluation/{epoch}')
            
            for eval in list_eval:
                res = torch.load(f'../res/{model}/evaluation/{epoch}/{eval}')
                rAE.append(res['rAE'])
                aAE.append(res['aAE'])
                rAE_T.append(res['rAE_T'])
                aAE_T.append(res['aAE_T'])
                corr_R.append(res['corr_R'])
                corr_A.append(res['corr_A'])
                corr_recon_R.append(res['corr_recon_R'])
                corr_recon_A.append(res['corr_recon_A'])
                corr_RT.append(res['corr_RT'])
                corr_AT.append(res['corr_AT'])
                corr_recon_RT.append(res['corr_recon_RT'])
                corr_recon_AT.append(res['corr_recon_AT'])
                rFE.append(res['rFE'])
                aFE.append(res['aFE'])
                FD.append(res['FD'])
                FDR.append(res['FDR'])
            
            rAE = torch.stack(rAE, axis=0)
            aAE = torch.stack(aAE, axis=0)
            rAE_T = torch.stack(rAE_T, axis=0)
            aAE_T = torch.stack(aAE_T, axis=0)
            corr_R = torch.stack(corr_R, axis=0)
            corr_A = torch.stack(corr_A, axis=0)
            corr_recon_R = torch.stack(corr_recon_R, axis=0)
            corr_recon_A = torch.stack(corr_recon_A, axis=0)
            corr_RT = torch.stack(corr_RT, axis=0)
            corr_AT = torch.stack(corr_AT, axis=0)
            corr_recon_RT = torch.stack(corr_recon_RT, axis=0)
            corr_recon_AT = torch.stack(corr_recon_AT, axis=0)
            rFE = torch.stack(rFE, axis=0)
            aFE = torch.stack(aFE, axis=0)
            FD = torch.stack(FD, axis=0)
            FDR = torch.stack(FDR, axis=0)
            
            rAE = torch.mean(rAE, axis=0)
            aAE = torch.mean(aAE, axis=0)
            rAE_T = torch.mean(rAE_T)
            aAE_T = torch.mean(aAE_T)
            corr_R = torch.mean(corr_R, axis=0)
            corr_A = torch.mean(corr_A, axis=0)
            corr_recon_R = torch.mean(corr_recon_R, axis=0)
            corr_recon_A = torch.mean(corr_recon_A, axis=0)
            corr_RT = torch.mean(corr_RT)
            corr_AT = torch.mean(corr_AT)
            corr_recon_RT = torch.mean(corr_recon_RT)
            corr_recon_AT = torch.mean(corr_recon_AT)
            rFE = torch.mean(rFE)
            aFE = torch.mean(aFE)
            FD = torch.mean(FD)
            FDR = torch.mean(FDR)
            
            if (n==1) & (all==False):
                print(f'[{"epoch":>5}] {"model":<20} {"rAE":>8} {"aAE":>8} {"rFE":>8} {"aFE":>8} {"FDR":>8} {"Corr":>8} {"FD":>8}')
            if all==False:
                print(f'[{epoch:>5}] {model:<20} {rAE_T:8.4f} {aAE_T:8.4f} {rFE:8.4f} {aFE:8.4f} {FDR:8.4f} {corr_recon_AT:8.4f} {FD:8.4f}')

            if n==1 & all==True:
                print(f'[{"epoch":>5}] {"model":<20}', end= ' ')
                print(f'{"rAE_x":>8} {"rAE_y":>8} {"rAE_z":>8}', end=' ' )
                print(f'{"rAE_a":>8} {"rAE_b":>8} {"rAE_c":>8}', end=' ' )
                print(f'{"aAE_x":>8} {"aAE_y":>8} {"aAE_z":>8}', end=' ' )
                print(f'{"aAE_a":>8} {"aAE_b":>8} {"aAE_c":>8}', end=' ' )
                print(f'{"corr_R_x":>8} {"corr_R_y":>8} {"corr_R_z":>8}', end=' ' )
                print(f'{"corr_R_a":>8} {"corr_R_b":>8} {"corr_R_c":>8}', end=' ' )
                print(f'{"corr_A_x":>8} {"corr_A_y":>8} {"corr_A_z":>8}', end=' ' )
                print(f'{"corr_A_a":>8} {"corr_A_b":>8} {"corr_A_c":>8}', end=' ' )
                print()
            if all==True:
                print(f'[{epoch:>5}] {model:<20}', end= ' ')
                print(f'{rAE[0]:8.4f} {rAE[1]:8.4f} {rAE[2]:8.4f}', end=' ' )
                print(f'{rAE[3]:8.4f} {rAE[4]:8.4f} {rAE[5]:8.4f}', end=' ' )
                print(f'{aAE[0]:8.4f} {aAE[1]:8.4f} {aAE[2]:8.4f}', end=' ' )
                print(f'{aAE[3]:8.4f} {aAE[4]:8.4f} {aAE[5]:8.4f}', end=' ' )
                print(f'{corr_R[0]:8.4f} {corr_R[1]:8.4f} {corr_R[2]:8.4f}', end=' ' )
                print(f'{corr_R[3]:8.4f} {corr_R[4]:8.4f} {corr_R[5]:8.4f}', end=' ' )
                print(f'{corr_A[0]:8.4f} {corr_A[1]:8.4f} {corr_A[2]:8.4f}', end=' ' )
                print(f'{corr_A[3]:8.4f} {corr_A[4]:8.4f} {corr_A[5]:8.4f}', end=' ' )
                print()
                
def print_total_v2(list_model=None, list_epoch=None, all=False):
    if list_model==None:
        list_model = listdir('../res')
        list_model.remove('Fig')
        list_model.remove('Fig_Recon')

    n = 0
    for model in list_model:
        if model=='--': 
            print('-'*82)
            continue
        
        if list_epoch==None:
            list_epoch = listdir(f'../res/{model}/evaluation')
        
        for epoch in list_epoch:
            n += 1
            rAE, aAE = [], []
            rAE_T, aAE_T = [], []
            corr_R, corr_A, corr_recon_R, corr_recon_A = [], [], [], []
            corr_RT, corr_AT, corr_recon_RT, corr_recon_AT = [], [], [], []
            rFE, aFE, FD, FDR = [], [], [], []
            list_eval = [i for i in listdir(f'../res/{model}/evaluation/{epoch}') if '@' not in i]
            
            for eval in list_eval:
                res = torch.load(f'../res/{model}/evaluation/{epoch}/{eval}')
                rAE.append(res['rAE'])
                aAE.append(res['aAE'])
                rAE_T.append(res['rAE_T'])
                aAE_T.append(res['aAE_T'])
                corr_R.append(res['corr_R'])
                corr_A.append(res['corr_A'])
                corr_recon_R.append(res['corr_recon_R'])
                corr_recon_A.append(res['corr_recon_A'])
                corr_RT.append(res['corr_RT'])
                corr_AT.append(res['corr_AT'])
                corr_recon_RT.append(res['corr_recon_RT'])
                corr_recon_AT.append(res['corr_recon_AT'])
                rFE.append(res['rFE'])
                aFE.append(res['aFE'])
                FD.append(res['FD'])
                FDR.append(res['FDR'])
            
            rAE = torch.stack(rAE, axis=0)
            aAE = torch.stack(aAE, axis=0)
            rAE_T = torch.stack(rAE_T, axis=0)
            aAE_T = torch.stack(aAE_T, axis=0)
            corr_R = torch.stack(corr_R, axis=0)
            corr_A = torch.stack(corr_A, axis=0)
            corr_recon_R = torch.stack(corr_recon_R, axis=0)
            corr_recon_A = torch.stack(corr_recon_A, axis=0)
            corr_RT = torch.stack(corr_RT, axis=0)
            corr_AT = torch.stack(corr_AT, axis=0)
            corr_recon_RT = torch.stack(corr_recon_RT, axis=0)
            corr_recon_AT = torch.stack(corr_recon_AT, axis=0)
            rFE = torch.stack(rFE, axis=0)
            aFE = torch.stack(aFE, axis=0)
            FD = torch.stack(FD, axis=0)
            FDR = torch.stack(FDR, axis=0)
            
            rAE_S, rAE_S_std = rAE[:, :3].mean(), rAE[:, :3].std()
            rAE_A, rAE_A_std = rAE[:, 3:].mean(), rAE[:, 3:].std()
            aAE_S, aAE_S_std = aAE[:, :3].mean(), aAE[:, :3].std()
            aAE_A, aAE_A_std = aAE[:, 3:].mean(), aAE[:, 3:].std()
            rAE = torch.mean(rAE, axis=0)
            aAE = torch.mean(aAE, axis=0)
            rAE_T = torch.mean(rAE_T)
            aAE_T = torch.mean(aAE_T)
            
            corr_R = torch.mean(corr_R, axis=0)
            corr_A = torch.mean(corr_A, axis=0)
            corr_recon_R = torch.mean(corr_recon_R, axis=0)
            corr_recon_A = torch.mean(corr_recon_A, axis=0)
            corr_RT = torch.mean(corr_RT)
            corr_AT = torch.mean(corr_AT)
            corr_recon_RT, corr_recon_RT_std = torch.mean(corr_recon_RT), corr_recon_RT.std()
            corr_recon_AT, corr_recon_AT_std = torch.mean(corr_recon_AT), corr_recon_AT.std()
            rFE, rFE_std = torch.mean(rFE), rFE.std()
            aFE, aFE_std = torch.mean(aFE), aFE.std()
            FD, FD_std = torch.mean(FD), FD.std()
            FDR, FDR_std = torch.mean(FDR), FDR.std()
            
            if n==1:
                print(f'[{"epoch":>5}] {"model":<20} {"rAE_S":>15} {"rAE_A":>15} {"aAE_S":>15} {"aAE_A":>15} {"rFE":>15} {"aFE":>15} {"Corr":>15} {"FDR":>15} {"FD":>15}')
            print(f'[{epoch:>5}] {model:<20} {rAE_S:8.4f}±{rAE_S_std:6.4f} {rAE_A:8.4f}±{rAE_A_std:6.4f} {aAE_S:8.4f}±{aAE_S_std:6.4f} {aAE_A:8.4f}±{aAE_A_std:6.4f} {rFE:8.4f}±{rFE_std:6.4f} {aFE:8.4f}±{aFE_std:6.4f} {corr_recon_AT:8.4f}±{corr_recon_AT_std:6.4f} {FDR:8.4f}±{FDR_std:6.4f} {FD:8.4f}±{FD_std:6.4f}')
            