import cv2
import torch
import numpy as np
import skimage.measure as measure
import matplotlib.pyplot as plt

from copy import deepcopy
from skimage import morphology as mp
from skimage import filters
from scipy import ndimage
from scipy.signal import savgol_filter

class PA_Processing():
    def __init__(self, T=190, shift=7, r1=3, r2=7, nr=False):
        self.T = T
        self.nr = nr
        self.shift = shift
        # self.kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r1, r1))
        self.kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r2, r2))
        self.list_idx_skin = []
    
    def thresholding(self, img):
        img = deepcopy(img)
        # img[120:, 50:-50] = 0
        
        return np.array(img>self.T, dtype=np.int8)
        
    def large_connected_domain(self, img):
        cd, num = measure.label(img, return_num=True, connectivity=1)
        volume = np.zeros([num])
        for k in range(num):
            volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
        volume_sort = np.argsort(volume)
        img = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
        img = ndimage.binary_fill_holes(img)>0
        img = img.astype(np.uint8)
        
        return img
        
    def opening(self, img):
        img = img.astype(np.float32)
        
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, self.kernel_opening)

    def get_skin_idx(self, img, t=20, window=50):
        idx = np.argmax(img, axis=0)-self.shift
        idx = np.clip(idx, 0, 1e+5).astype(np.int16)
        l = len(idx)//2
        for i in range(l-1):
            if np.abs(idx[l+(i+1)]-idx[l+(i)])>=t: idx[l+(i+1)]=idx[l+(i)]
            if np.abs(idx[l-(i+1)]-idx[l-(i)])>=t: idx[l-(i+1)]=idx[l-(i)]
        idx[0] = idx[1]
        idx[-1] = idx[-2]
        idx = savgol_filter(idx, window_length=window, polyorder=3)
        idx = np.ceil(idx).astype(np.int16)
        
        return idx
    
    def get_skin_mask(self, img, idx):
        res = np.zeros_like(img, dtype=np.uint8)
        # res[idx, np.linspace(0, 255, 256, dtype=np.uint8)] = 1
        res[np.arange(img.shape[0])[:, None]<idx] = 1
        
        return res
    
    def dilation(self, img):
        
        return cv2.morphologyEx(img, cv2.MORPH_DILATE, self.kernel_dilation)
    
    def reverse_mask(self, mask):
        
        return (mask.astype(np.int8)-1)*(-1)
    
    def noise_reduction(self, img, idx_skin): 
        idx_skin = int(np.mean(idx_skin))
        M = img[idx_skin:idx_skin+50].max()
        a = img[idx_skin:idx_skin+50].mean()
        T = a+(M-a)*0.3
        mask = img>T
        mask = mp.dilation(mask, mp.disk(1))
        img = mask*img
        
        return img
    
    def skin_rejection_(self, img_B, img_P):
        # Thresholding
        self.img_T = self.thresholding(img_B)
        
        # Opening
        self.img_O = self.opening(self.img_T)
        
        # LCD
        self.img_L = self.large_connected_domain(self.img_O)
        
        # Skin Idx
        self.idx_skin = self.get_skin_idx(self.img_O)
        self.list_idx_skin.append(self.idx_skin)
        
        # Skin Line
        self.img_S = self.get_skin_mask(self.img_O, self.idx_skin)
        
        # Reverse
        self.img_R = self.reverse_mask(self.img_S)
        
        # Skin Rejection
        self.res = img_P*self.img_R
        
        # Noise Reduction
        if self.nr: self.res = self.noise_reduction(self.res, self.idx_skin)
        
        return self.res, self.img_R
    
    def skin_rejection(self, imgs_B, imgs_P):
        Res = [self.skin_rejection_(imgs_B[i], imgs_P[i]) for i in range(len(imgs_B))]
        Res = np.stack(Res).astype(np.float32)
        self.list_idx_skin = np.array(self.list_idx_skin)
        
        return Res

def depth_compensation(imgs, list_idx_skin, s=10, margin=20, US=False):
    list_idx = np.mean(list_idx_skin, axis=-1)
    list_ddx = np.array([list_idx[i+1]-list_idx[i] for i in range(len(list_idx)-1)])
    T = list_ddx.mean()*30
    
    for i in range(s, len(list_idx)-1):
        if np.abs(list_idx[i+1]-list_idx[i])>T:
            list_idx[i+1] = list_idx[i]
    
    list_idx = list_idx.astype(np.int16)
    
    if US==False:
        for i in range(len(list_idx)):
            pad1 = np.zeros([margin, imgs.shape[-1]], dtype=np.float32)
            pad2 = np.zeros([list_idx[i], imgs.shape[-1]], dtype=np.float32)
            img = imgs[i][list_idx[i]:-margin, :]
            img = np.concatenate([pad1, img, pad2], axis=0)
            imgs[i] = img
    else:
        for i in range(len(list_idx)):
            pad2 = np.zeros([list_idx[i]+margin, imgs.shape[-1]], dtype=np.float32)
            # pad2 = np.ones([list_idx[i]+margin, imgs.shape[-1]], dtype=np.float32)*(imgs[i].mean())
            img = imgs[i][list_idx[i]-margin:-margin*2, :]
            img = np.concatenate([img, pad2], axis=0)
            imgs[i] = img
        
    return imgs

def projection_DE(imgs, idx_m, idx_M, alpha=3.5, beta=1, cmap='jet', flip=True):
    imgs = imgs.permute(1, 2, 0)
    imgs = imgs[idx_m:idx_M]
    D, W, L = imgs.shape[0], imgs.shape[1], imgs.shape[2]
    imgs_RGB = torch.zeros([W*L, 3])
    print(f'{"Image Size (D, H, L)":<30}: {imgs.shape}')
    print(f'{"RoI":<30}: ({idx_m}, {idx_M})')
    
    cmap = plt.get_cmap(cmap)
    cmap = cmap(np.arange(cmap.N))[:, :3]
    
    imgs_v = imgs.reshape(D, W*L)
    idx_M = torch.argmax(imgs_v, dim=0)
    idx_M = (idx_M-idx_M.min())/(idx_M.max()-idx_M.min())
    idx_M = np.array(idx_M*255, dtype=np.int16)
    
    imgs_a = torch.max(imgs_v, dim=0)[0]
    imgs_a = imgs_a.reshape(W, L)
    # imgs_a = sigmoid(imgs_a, T=0.2, a=10)
    imgs_a = imgs_a*alpha
    imgs_a = imgs_a**beta
    imgs_a = torch.clip(imgs_a, 0, 1)
    # imgs_a[imgs_a<0.2] = 0
    
    res = cmap[idx_M]
    res = res.reshape(W, L, 3)
    pad = np.ones([imgs_a.shape[0], imgs_a.shape[1], 1], dtype=np.int16)
    pad = pad*(imgs_a.unsqueeze(-1).numpy())
    res = np.concatenate([res, pad], axis=-1)
    if flip: res = res[::-1]
    
    return res
    
def projection_AP(imgs_R, idx_m=0, idx_M=-1, alpha=1.5, cmap='hot', flip=True):
    res = torch.max(imgs_R[:, idx_m:idx_M, :], dim=1)[0].T
    # res = torch.mean(imgs_R[:, idx_m:idx_M, :], dim=1).T
    res = res*alpha
    res = torch.clip(res, 0, 1)
    if flip: res = torch.flip(res, dims=[0])
    
    return res

def minMaxScaling_frame(imgs):
    d1, d2, d3 = imgs.shape
    imgs = imgs.reshape(d1, -1)
    M = torch.max(imgs, dim=1, keepdim=True)[0]
    m = torch.min(imgs, dim=1, keepdim=True)[0]
    imgs = (imgs-m)/(M-m)
    imgs = imgs.reshape(d1, d2, d3)
    
    return imgs

def get_roi(imgs):
    res = torch.mean(imgs, axis=2)
    res = torch.mean(res, axis=0)
    for i in range(len(res)-1):
        diff = res[i+1]-res[i]
        if diff>0: idx_m = i; break
    for i in range(1, len(res)-1):
        diff = res[len(res)-i]-res[len(res)-(i+1)]
        if diff<0: idx_M = len(res)-i; break
        
    return idx_m, idx_M    

def minmax_scaling(imgs):
    return (imgs-imgs.min())/(imgs.max()-imgs.min())

def averaging(imgs, k=3, US=False):
    l = k//2
    temp = []
    for i in range(l, len(imgs)-l): 
        if US==False: temp.append(np.mean(imgs[i-l:i+l+1], axis=0))
        if US==True: temp.append(imgs[l])
    
    return np.stack(temp)

def rejection(img, T):
    return img*(img>T)