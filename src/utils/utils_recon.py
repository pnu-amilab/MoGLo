import os
import h5py
import glob
import natsort
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy
from IPython.display import clear_output
from matplotlib.colors import ListedColormap
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
try:
    from pytorch3d.transforms import euler_angles_to_matrix
    from pytorch3d.transforms import matrix_to_euler_angles
except ImportError:
    euler3d_to_matrix = matrix3d_to_euler = None

def listdir(path):
    return natsort.natsorted(os.listdir(path))

def globsort(path):
    return natsort.natsorted(glob.glob(path))

def make_dir(path):
    if not os.path.exists(path): 
        os.mkdir(path)
        os.chmod(path, 0o777)

def transform_a2m(y, degrees=True, convention='YZX'):
    """
    Convert 3D vector [angle_z, angle_y, angle_x] to 3x3 transformation matrix
    y: Tensor of shape [B, T, 3]
    R: Tensor of shape [B, T, 3, 3]
    """
    a = y[..., 3:].clone()  # [..., 3]
    if degrees: a *= (torch.pi/180)

    R = euler_angles_to_matrix(a, convention=convention)  # [..., 3, 3]

    return R

def transform_v2m(y, degrees=True, convention='YZX'):
    """
    Convert 6D vector [tx, ty, tz, angle_z, angle_y, angle_x] to 4x4 transformation matrix
    y: Tensor of shape [B, T, 6]
    T: Tensor of shape [B, T, 4, 4]
    """
    a = y[..., 3:].clone()  # [..., 3]
    t = y[..., :3].clone()   # [..., 3]
    if degrees: a *= (torch.pi/180)

    R = euler_angles_to_matrix(a, convention=convention)  # [..., 3, 3]

    # Identity matrix of shape [B, T, 4, 4]
    B, T_len = y.shape[:2]
    T = torch.eye(4, dtype=y.dtype, device=y.device).expand(B, T_len, 4, 4).clone()  # [B, T, 4, 4]

    T[..., :3, :3] = R
    T[..., :3,  3] = t

    return T

def transform_m2v(T, degrees=True, convention='YZX'):
    """
    Convert 4x4 transformation matrix to 6D vector [tx, ty, tz, angle_z, angle_y, angle_x]
    T: Tensor of shape [N, 4, 4]
    y: Tensor of shape [N, 6]
    """
    R = T[..., :3, :3].clone()
    t = T[..., :3,  3].clone()
    a = matrix_to_euler_angles(R, convention=convention)  # [N, 3]
    if degrees: a *= (180/torch.pi)

    return torch.cat([t, a], dim=-1)

def transform_acum(pose, source='param', target='param', degrees=True, convention='YZX'):
    """
    Accumulate relative motion vectors into absolute poses
    y_rel: [N-1, 6] relative motion vectors (in YZX + translation order)
    y_abs: [N, 6] absolute poses
    """
    if source=='param':
        T_rel = transform_v2m(pose, degrees=degrees, convention=convention)  # [N-1, 4, 4]
    else: 
        T_rel = pose  # [N-1, 4, 4]

    B, T, _, _ = T_rel.shape
    T_abs = torch.zeros((B, T + 1, 4, 4), dtype=T_rel.dtype, device=T_rel.device)  # [B, T+1, 4, 4]
    T_abs[:, 0] = torch.eye(4, dtype=T_rel.dtype, device=T_rel.device)

    for t in range(1, T + 1):
        T_abs[:, t] = torch.matmul(T_abs[:, t - 1], T_rel[:, t - 1])

    if target == 'param':
        return transform_m2v(T_abs, convention=convention)  # [B, T+1, 6]
    else:
        return T_abs  # [B, T+1, 4, 4]

def transform_diff(pose, source='param', target='param', degrees=True, convention='YZX'):
    """
    Convert absolute pose sequence into relative motion vectors

    pose: [B, T+1, 6] if source == 'param'
          [B, T+1, 4, 4] if source == 'matrix'

    Returns:
    - [B, T, 6] if target == 'param'
    - [B, T, 4, 4] if target == 'matrix'
    """
    if source == 'param':
        T_abs = transform_v2m(pose, degrees=degrees, convention=convention)  # [B, T+1, 4, 4]
    else:
        T_abs = pose  # [B, T+1, 4, 4]

    B, T1, _, _ = T_abs.shape
    T = T1 - 1

    T_rel = torch.zeros((B, T, 4, 4), dtype=T_abs.dtype, device=T_abs.device)

    for t in range(T):
        T_prev_inv = torch.linalg.inv(T_abs[:, t])       # [B, 4, 4]
        T_rel[:, t] = torch.matmul(T_prev_inv, T_abs[:, t + 1])  # T_rel = T_prev⁻¹ × T_curr

    if target == 'param':
        return transform_m2v(T_rel, degrees=degrees, convention=convention)  # [B, T, 6]
    else:
        return T_rel  # [B, T, 4, 4]

def transform_acum_sub(list_p_abs, degrees=True, convention='YZX'):
    list_T_abs = transform_v2m(list_p_abs, degrees=degrees, convention=convention)  # [N, seq, 4, 4]
    T_abs = torch.zeros_like(list_T_abs)
    base = torch.eye(4, device=list_T_abs.device)

    for i in range(len(list_T_abs)):
        T_abs_sub = list_T_abs[i] @ base
        T_abs[i] = T_abs_sub
        base = base @ list_T_abs[i][1]
    p_abs = transform_m2v(T_abs, convention=convention)
    
    return p_abs

### Scipy
def euler_angles_to_matrix_local(a, convention='YZX'):
    B, S, _ = a.shape
    R_list = []
    for b in range(B):
        R_b = R.from_euler(convention.lower(), a[b], degrees=False).as_matrix()
        R_list.append(R_b)
    return torch.tensor(np.stack(R_list), dtype=a.dtype)

def matrix_to_euler_angles_local(R_mat, convention='YZX'):
    B, S = R_mat.shape[:2]
    angle_list = []
    for b in range(B):
        angles = R.from_matrix(R_mat[b]).as_euler(convention.lower(), degrees=False)
        angle_list.append(angles)
    return torch.tensor(np.stack(angle_list), dtype=R_mat.dtype)

def transform_a2m_local(y, degrees=True, convention='YZX'):
    a = y[..., 3:].clone()  # [B, S, 3]
    if degrees:
        a *= (torch.pi / 180)
    return euler_angles_to_matrix_local(a, convention)

def transform_v2m_local(y, degrees=True, convention='YZX'):
    t = y[..., :3].clone()   # [B, S, 3]
    a = y[..., 3:].clone()   # [B, S, 3]
    if degrees:
        a *= (torch.pi / 180)
    R_mat = euler_angles_to_matrix_local(a, convention)

    B, S = y.shape[:2]
    T = torch.eye(4, dtype=y.dtype).expand(B, S, 4, 4).clone()
    T[..., :3, :3] = R_mat
    T[..., :3, 3] = t
    return T

def transform_m2v_local(T, degrees=True, convention='YZX'):
    R_mat = T[..., :3, :3].clone()  # [B, S, 3, 3]
    t = T[..., :3, 3].clone()       # [B, S, 3]
    a = matrix_to_euler_angles_local(R_mat, convention)
    if degrees:
        a *= (180 / torch.pi)
    return torch.cat([t, a], dim=-1)  # [B, S, 6]

def transform_acum_local(pose, source='param', target='param', degrees=True, convention='YZX'):
    if source == 'param':
        T_rel = transform_v2m_local(pose, degrees=degrees, convention=convention)
    else:
        T_rel = pose

    B, S, _, _ = T_rel.shape
    T_abs = torch.zeros((B, S + 1, 4, 4), dtype=T_rel.dtype)
    for b in range(B):
        T_abs[b, 0] = torch.eye(4, dtype=T_rel.dtype)
        for s in range(1, S + 1):
            T_abs[b, s] = T_abs[b, s - 1] @ T_rel[b, s - 1]

    if target == 'param':
        return transform_m2v_local(T_abs, degrees=degrees, convention=convention)
    else:
        return T_abs

def transform_diff_local(pose, source='param', target='param', degrees=True, convention='YZX'):
    if source == 'param':
        T_abs = transform_v2m_local(pose, degrees=degrees, convention=convention)
    else:
        T_abs = pose

    B, S1, _, _ = T_abs.shape
    S = S1 - 1
    T_rel = torch.zeros((B, S, 4, 4), dtype=T_abs.dtype)

    for b in range(B):
        for s in range(S):
            T_inv = torch.inverse(T_abs[b, s])
            T_rel[b, s] = T_inv @ T_abs[b, s + 1]

    if target == 'param':
        return transform_m2v_local(T_rel, degrees=degrees, convention=convention)
    else:
        return T_rel
    
# Utils
def imshow_hist(img, alpha=4):
    img = deepcopy(img)**alpha
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.subplot(1, 2, 2)
    plt.hist(img.flatten().numpy(), bins=300)
    plt.axvline(0.5, c='r', ls='--')
    plt.show()

def imshow_proj(s, cmap='gray'):
    titles = ['Longi_1', 'Longi_2', 'Trans']
    plt.figure(figsize=(20, 7))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(titles[i], fontsize=20)
        plt.imshow(torch.amax(s, dim=i), cmap=cmap, interpolation='bicubic')
        
def plot_proj_total(model, epoch=2000, n=[5, 6, 7], m=[0, 1, 2, 3, 4, 5], size=64):
    make_dir('../res/Fig_Proj')
    for n_ in n:
        for m_ in m:
            imgs, list_p_acum = get_res([model], epoch, n_, m_)
            s, scale = get_shape(list_p_acum, size)
            scale = [scale['xm'], scale['ym'], scale['zm']]
            s = recon_3d(s, imgs, list_p_acum[-1], size, scale, edge=0, c=255, skip=1, floor=0) # GT
            s = recon_3d(s, imgs, list_p_acum[+0], size, scale, edge=4, c=255, skip=1, floor=0) # Ours
            imshow_proj(s)
            plt.suptitle(f'{model}_{n_}_{m_}', fontsize=24)
            plt.savefig(f'../res/Fig_Proj/{model}_{n_}_{m_}.png')
            plt.close()

def plot_total(list_model, epoch=3000, list_n=[5, 6, 7], list_m=[0, 1, 2, 3, 4, 5], close=True):
    for n in tqdm(list_n):
        for m in list_m:
            list_patient = listdir('../../US_Data/Forearm_Main')
            list_dataset = listdir('../../US_Data/Forearm_Main/001')
            patient = list_patient[n]
            dataset = list_dataset[m]

            score = torch.load(f'../res/{list_model[0]}/evaluation/{epoch}/{patient}_{dataset[:-3]}.pt')
            rAE = score['rAE_T']
            aAE = score['aAE_T']
            rFE = score['rFE']
            aFE = score['aFE']
            fdr = score['FDR']
            corr = score['corr_recon_RT']
            score_keys = f'{"rAE":<7}, {"aAE":<7}, {"rFE":<7}, {"aFE":<7}, {"FDR":<7}, {"corr":<7}'
            score = f'{rAE:0.4f}, {aAE:0.4f}, {rFE:0.4f}, {aFE:0.4f}, {fdr:0.4f}, {corr:0.4f}'
            
            list_p_recon = []
            list_p_acum = []
            list_p_data = []
            for model in list_model:
                res = torch.load(f'../res/{model}/inference/{epoch}/{patient}_{dataset[:-3]}.pt')
                list_p_recon.append(res['p_recon_A'][:, :, 0])
                list_p_acum.append(res['p_acum'])
                list_p_data.append(res['p_data'])
            list_p_recon.append(res['y_recon_A'][:, :,0])
            list_p_acum.append(res['y_acum'])
            list_p_data.append(res['y_data'])
            
            list_p_acum = torch.stack(list_p_acum)
            list_p_acum[:, :, [0, 2]] *= -1
            list_p_acum[:, :, :3] -= torch.min(torch.min(list_p_acum[:, :, :3], axis=0)[0], axis=0)[0]

            title = f'({n}, {m}): {patient}_{dataset}\n{score_keys}\n{score}'
            list_color = ['red', 'blue', 'blue', 'green', 'green', 'black']
            list_ls = ['-', '-', '--', '-', '--', '--']
            plt.figure(figsize=(20, 20))
            for i in range(3):
                plt.subplot(5, 3, i+1)
                plt.axhline(0, color='black', ls='--', lw=0.6, alpha=0.7)
                plt.plot(list_p_recon[-1][:, i], color='black', label='True')
                for j in range(len(list_model)):
                    plt.plot(list_p_recon[j][:, i], color=list_color[j], ls=list_ls[j], lw=1, label=list_model[j])
                if i==2: plt.legend()
            
            plt.suptitle(title, fontsize=20)
            for i in range(6):
                plt.subplot(5, 3, i+1+3)
                plt.axhline(0, color='black', ls='--', lw=0.6, alpha=0.7)
                plt.plot(list_p_acum[-1][:, i], color='black', label='True')
                for j in range(len(list_model)):
                    plt.plot(list_p_acum[j][:, i], color=list_color[j], ls=list_ls[j], lw=1, label=list_model[j])
                if i==2: plt.legend()
            
            for i in range(6):
                plt.subplot(5, 3, i+1+9)
                plt.axhline(0, color='black', ls='--', lw=0.6, alpha=0.7)
                plt.plot(list_p_data[-1][:, i], color='black', label='True')
                for j in range(len(list_model)):
                    alpha = 0.3 if j!=0 else 1
                    plt.plot(list_p_data[j][:, i], color=list_color[j], ls=list_ls[j], lw=1, alpha=alpha, label=list_model[j])
                if i==2: plt.legend()
                
            plt.savefig(f'../res/Fig_Recon/{patient}_{dataset[:-3]}.png')
            if close: plt.close()

def get_res(list_model, epoch, n, m, supress=False):
    patient = listdir('../../US_Data/Forearm_Main')[n]
    dataset = listdir('../../US_Data/Forearm_Main/001')[m]
    data = h5py.File(f'../../US_Data/Forearm_Main/{patient}/{dataset}', 'r')
    imgs = data['img_B'][:].squeeze()
    imgs = (imgs-imgs.min())/(imgs.max()-imgs.min())
    imgs = torch.from_numpy(imgs)
    data.close()
    
    y_acum = torch.load(f'../res/{list_model[0]}/inference/{epoch}/{patient}_{dataset[:-3]}.pt')['y_acum']
    list_p_acum = []
    for model in list_model:
        res = torch.load(f'../res/{model}/inference/{epoch}/{patient}_{dataset[:-3]}.pt')
        list_p_acum.append(res['p_acum'])

    list_p_acum.append(y_acum)
    list_p_acum = torch.stack(list_p_acum)
    if supress:
        list_p_acum[:, :, 0] *= 0.1
    # list_p_acum[:, :, [0, 2]] *= -1
    # list_p_acum[:, :, :3] -= torch.min(torch.min(list_p_acum[:, :, :3], axis=0)[0], axis=0)[0]
    
    return imgs, list_p_acum

def custom_cmap(base_cmap='binary_r', T=100):
    cmap = plt.cm.get_cmap(base_cmap)
    cmap = cmap(np.arange(cmap.N))
    cmap[:T, -1] = 0
    cmap[-1] = np.array([1, 0, 0, 1]) # 빨강 255
    cmap[-2] = np.array([0, 1, 0, 1]) # 초록 254
    cmap[-3] = np.array([0, 0, 1, 1]) # 파랑 253
    cmap[-4] = np.array([1, 1, 0, 1]) # 노랑 252
    cmap[-5] = np.array([1, 0, 1, 1]) # 핑크 251
    cmap[-6] = np.array([0, 1, 1, 1]) # 하늘 250
    cmap[-7] = np.array([0, 0, 0, 1]) # 검정 249
    cmap = ListedColormap(cmap)
    
    return cmap

def cumsum_reconstruction(p_data):
    p_acum = torch.concat([torch.zeros([1, 6], dtype=torch.float32), p_data], axis=0)
    
    return p_acum

def pair_wise_initialization(emt_y):
    emp_dy = []
    for i in range(len(emt_y)-1):
        emp_dy_ = np.zeros([6], dtype=np.float32)
        emt_y1, emt_y2 = emt_y[i+0], emt_y[i+1]
        emt_t1, emt_t2 = emt_y1[:3], emt_y2[:3]
        emt_a1, emt_a2 = emt_y1[3:], emt_y2[3:]
        M = R.from_euler('yzx', [emt_a1[0], emt_a1[1], emt_a1[2]], degrees=True).as_matrix().T
        emt_z1, emt_z2 = np.matmul(M, emt_t1), np.matmul(M, emt_t2)
        emp_dy_[:3] = emt_z2-emt_z1
        emp_dy_[3:] = emt_a2-emt_a1
        emp_dy.append(emp_dy_)
        
    emp_dy = np.stack(emp_dy)
    
    return emp_dy

def pair_wise_reconstruction(emp_dy):
    emp_dy = np.array(emp_dy)
    emt_y = [np.zeros([6], dtype=np.float32)]

    for i in range(len(emp_dy)):
        emp_y1 = emt_y[-1]
        M = R.from_euler('yzx', [emp_y1[3], emp_y1[4], emp_y1[5]], degrees=True).as_matrix()
        
        emp_dt = np.matmul(M, emp_dy[i, :3])
        emp_t = emp_y1[:3] + emp_dt
        emp_a = emp_y1[3:] + emp_dy[i, 3:]
        
        emt_y.append(np.concatenate([emp_t, emp_a]))

    emt_y = np.array(emt_y)
    emt_y = torch.tensor(emt_y, dtype=torch.float32)
    
    return emt_y

# Visualization
def get_shape(list_p_acum_, size, calib_vec=[20, 0, 7]):
    list_p_acum = deepcopy(list_p_acum_)
    calib_vec = torch.tensor(calib_vec)*(size/256)
    list_point = torch.tensor([[i, j, 0] for i in range(0, size+1, 16) for j in range(0, size+1, 16)]).to(torch.float32)
    list_point -= calib_vec
    
    list_p_acum = list_p_acum.reshape(-1, 6)
    list_p_acum[:, :3] *= size/38
    
    list_R = transform_a2m_local(list_p_acum.unsqueeze(0)).squeeze()
    
    res = torch.matmul(list_R, list_point.T)
    res = res + list_p_acum[:, :3].unsqueeze(-1)
    res = res.permute(0, 2, 1)
    
    xM, xm = res[..., 0].max().to(torch.int64), res[..., 0].min().to(torch.int64)
    yM, ym = res[..., 1].max().to(torch.int64), res[..., 1].min().to(torch.int64)
    zM, zm = res[..., 2].max().to(torch.int64), res[..., 2].min().to(torch.int64)
    
    margin = int(size*0.1)
    xl = (xM-xm+margin).to(torch.int64)
    yl = (yM-ym+margin).to(torch.int64)
    zl = (zM-zm+margin).to(torch.int64)
    
    s = torch.zeros([xl, yl, zl], dtype=torch.float32)
    scale = {'xM':xM, 
             'xm':xm, 
             'yM':yM, 
             'ym':ym, 
             'zM':zM, 
             'zm':zm}
    
    return s, scale

def recon_3d(s, imgs, y_acum_, size, scale=[0, 0, 0], alpha=3.5, edge=0, edge_last=1, c=255, 
             calib_vec=[20, 0, 7], skip=1, floor=False, cut=0.15, del_edge=0):
    '''
    imgs: B-mode [n, 256, 256]
    y_acum: Trajectory
    size: img resize
    edge: render only edge
    c: edge color
    calib_vec: calibration vector with respect to the pos of EMT
    skip: frame interval
    floor: floor color
    '''
    # Image Processing
    Imgs = deepcopy(imgs)
    Imgs = Imgs**alpha
    Imgs = Imgs*255
    Imgs = Imgs.flip(dims=[0])
    cut = round(size*cut)
    
    # cut
    Imgs = Imgs[:, cut:-cut, cut:-cut]
    model = nn.Upsample(size=(size, size), mode='bicubic', align_corners=True)
    Imgs = model(Imgs.unsqueeze(1)).squeeze()
    Imgs = torch.clip(Imgs, 0, 255)
    imgs = deepcopy(Imgs)

    # Calibration
    y_acum = deepcopy(y_acum_)
    calib_vec = torch.tensor(calib_vec)*(size/256)
    
    # idx for 2D imgs (I4I)
    if edge==0:
        list_I4I = torch.tensor([[i, j] for i in range(0, size) for j in range(0, size)])
    else:
        imgs = torch.ones_like(Imgs)*c
        list_e0 = torch.tensor([[       i,        j] for i in range(0, edge) for j in range(0, edge)])
        list_e1 = torch.tensor([[       i, size-1-j] for i in range(0, edge) for j in range(0, edge)])
        list_e2 = torch.tensor([[size-1-i, size-1-j] for i in range(0, edge) for j in range(0, edge)])
        list_e3 = torch.tensor([[size-1-i,        j] for i in range(0, edge) for j in range(0, edge)])
        list_idx = [0, 1, 2, 3]
        list_e = torch.stack([list_e0, list_e1, list_e2, list_e3])
        for i in del_edge: list_idx.remove(i)
        list_e = list_e[list_idx]
        list_I4I = list_e.reshape(-1, 2)
        
    # idx for 3D vol (I4V)
    list_point = torch.concat([list_I4I, torch.zeros([len(list_I4I), 1])], axis=1)
    list_point -= calib_vec # Calibration
    list_point = list_point.to(torch.float32)
    y_acum[:, :3] *= size/38 # Converson
    y_acum = y_acum[::skip]
    
    # Rotation & Shift
    list_R = transform_a2m_local(y_acum.unsqueeze(0)).squeeze()
    list_I4V = torch.matmul(list_R, list_point.T)
    list_I4V += y_acum[:, :3].unsqueeze(-1)
    list_I4V = list_I4V.permute(0, 2, 1)
    
    # base
    list_I4V[:, :, 0]  -= scale[0]
    list_I4V[:, :, 1]  -= scale[1]
    list_I4V[:, :, 2]  -= scale[2]
    
    # Type
    list_I4V = list_I4V.to(torch.int64)
    list_I4I = list_I4I.to(torch.int64)
    
    # idx: imgs
    idx = torch.linspace(0, len(imgs)-0.5, len(list_I4V)).to(torch.int64)
    imgs_ = imgs[idx]
        
    # Reconstruction
    s[list_I4V[:, :, 0], list_I4V[:, :, 1], list_I4V[:, :, 2]] = imgs_[:, list_I4I[:, 0], list_I4I[:, 1]]
    
    # Start/End
    if edge!=0:
        del_edge = [3, 2]
        imgs = torch.ones_like(Imgs)*c
        list_e0 = torch.tensor([[       i,        j] for i in range(0, edge) for j in range(0, size)])
        list_e1 = torch.tensor([[       i, size-1-j] for i in range(0, size) for j in range(0, edge)])
        list_e2 = torch.tensor([[size-1-i,        j] for i in range(0, edge) for j in range(0, size)])
        list_e3 = torch.tensor([[       i,        j] for i in range(0, size) for j in range(0, edge)])
        list_e = [list_e0, list_e1, list_e2, list_e3]
        list_idx = [0, 1, 2, 3]
        if (0 in del_edge) and (3 in list_idx): list_idx.remove(3)
        if (0 in del_edge) and (0 in list_idx): list_idx.remove(0)        
        if (1 in del_edge) and (0 in list_idx): list_idx.remove(0)
        if (1 in del_edge) and (1 in list_idx): list_idx.remove(1)        
        if (2 in del_edge) and (1 in list_idx): list_idx.remove(1)
        if (2 in del_edge) and (2 in list_idx): list_idx.remove(2)        
        if (3 in del_edge) and (2 in list_idx): list_idx.remove(2)
        if (3 in del_edge) and (3 in list_idx): list_idx.remove(3)
        
        list_I4I_e = torch.concat([list_e[i] for i in list_idx], axis=0)
        
        list_point_e = torch.concat([list_I4I_e, torch.zeros([len(list_I4I_e), 1])], axis=1)
        list_point_e -= calib_vec # Calibration
        list_point_e = list_point_e.to(torch.float32)
        
        list_R_e = torch.concat([list_R[:edge]], axis=0)
        y_acum_e = torch.concat([y_acum[:edge]], axis=0)

        list_I4V_e = torch.matmul(list_R_e, list_point_e.T)
        list_I4V_e += y_acum_e[:, :3].unsqueeze(-1)
        list_I4V_e = list_I4V_e.permute(0, 2, 1)
        
        imgs = torch.concat([imgs[:edge]], axis=0)
        list_I4V_e[:, :, 0]  -= scale[0]
        list_I4V_e[:, :, 1]  -= scale[1]
        list_I4V_e[:, :, 2]  -= scale[2]
        
        list_I4I_e = list_I4I_e.to(torch.int64)
        list_I4V_e = list_I4V_e.to(torch.int64)
        
        s[list_I4V_e[:, :, 0], list_I4V_e[:, :, 1], list_I4V_e[:, :, 2]] = imgs[:, list_I4I_e[:, 0], list_I4I_e[:, 1]]
    
    if edge!=0:
        imgs = torch.ones_like(Imgs)*c
        list_e0 = torch.tensor([[       i,        j] for i in range(0, edge) for j in range(0, size)])
        list_e1 = torch.tensor([[       i, size-1-j] for i in range(0, size) for j in range(0, edge)])
        list_e2 = torch.tensor([[size-1-i,        j] for i in range(0, edge) for j in range(0, size)])
        list_e3 = torch.tensor([[       i,        j] for i in range(0, size) for j in range(0, edge)])
        list_e = [list_e0, list_e1, list_e2, list_e3]
        list_idx = [0, 1, 2, 3]
        
        list_I4I_e = torch.concat([list_e[i] for i in list_idx], axis=0)
        
        list_point_e = torch.concat([list_I4I_e, torch.zeros([len(list_I4I_e), 1])], axis=1)
        list_point_e -= calib_vec # Calibration
        list_point_e = list_point_e.to(torch.float32)
        
        list_R_e = torch.concat([list_R[-edge*edge_last:]], axis=0)
        y_acum_e = torch.concat([y_acum[-edge*edge_last:]], axis=0)

        list_I4V_e = torch.matmul(list_R_e, list_point_e.T)
        list_I4V_e += y_acum_e[:, :3].unsqueeze(-1)
        list_I4V_e = list_I4V_e.permute(0, 2, 1)
        
        imgs = torch.concat([imgs[-edge*edge_last:]], axis=0)
        list_I4V_e[:, :, 0]  -= scale[0]
        list_I4V_e[:, :, 1]  -= scale[1]
        list_I4V_e[:, :, 2]  -= scale[2]
        
        list_I4I_e = list_I4I_e.to(torch.int64)
        list_I4V_e = list_I4V_e.to(torch.int64)
        
        s[list_I4V_e[:, :, 0], list_I4V_e[:, :, 1], list_I4V_e[:, :, 2]] = imgs[:, list_I4I_e[:, 0], list_I4I_e[:, 1]]
    
    # Floor
    if floor>0:
        s[:, :, 0] = floor
        s[:, 0, :] = floor
        s[0, :, :] = floor
        
    return s

def get_current_view(plotter):
    position = np.array(plotter.camera.position)
    focal_point = np.array(plotter.camera.focal_point)
    viewup = np.array(plotter.camera.up)

    view_vector = focal_point - position
    view_vector = view_vector / np.linalg.norm(view_vector)
    view_vector *= -1

    return view_vector, viewup

def rendering(volume, cmap='gray', alpha=1, background='white', 
              view_vector=(0, 1, 0), viewup=(0, 1, 0)):
    opacity = [0, alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha, 1]
    volume = np.float32(volume.numpy())

    grid = pv.ImageData(dimensions=volume.shape)
    grid["values"] = volume.flatten(order='F')

    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.set_background(background)
    plotter.add_volume(grid, scalars='values', opacity=opacity, cmap=cmap, clim=(0, 255))
    plotter.view_vector(vector=view_vector, viewup=viewup)
    plotter.show()

    view_vector, viewup = get_current_view(plotter)
    print("View Vector    :", view_vector)
    print("View Up Vector :", viewup)

    return view_vector, viewup