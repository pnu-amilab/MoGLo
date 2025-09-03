import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def parallel_run(layer, Xs):
    B = Xs[0].shape[0]
    x_temp = layer(torch.concat(Xs, axis=0))
    
    return [x_temp[i*B:(i+1)*B] for i in range(x_temp.shape[0]//B)]

# Print Functions for Debugging
def printt(x, t=False):
    if t: print(x)

class xshape(nn.Module):
    def __init__(self, name='name'):
        super().__init__()
        self.name = name
        
    def forward(self, x):
        print(self.name, ': ', x.shape)
        
        return x

class NoneBlock(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

# Simple Distance Measure Function between two Features
class Distance_Module(nn.Module):
    def __init__(self, mode='cos'):
        super().__init__()
        '''
        mode = [cos, l2]
        '''
        self.mode = mode
        
    def forward(self, x1, x2):
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)
        
        if self.mode=='cos':
            res = (F.cosine_similarity(x1, x2, dim=1)+1)/2
        if self.mode=='l2':
            res = torch.mean((x1-x2)**2, axis=1)

        return res

# Conventional Channel Attention Module (CBAM)
class CA_module(nn.Module):
    def __init__(self, in_channels, r=16, return_score=False):
        super().__init__()
        self.return_score = return_score
        self.squeeze_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.squeeze_max = nn.AdaptiveMaxPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels//r), 
            nn.SiLU(), 
            nn.Linear(in_channels//r, in_channels), 
            nn.Sigmoid())
        
    def forward(self, x):
        xa = self.squeeze_avg(x)
        xm = self.squeeze_max(x)
        xa = xa.view(xa.shape[0], -1)
        xm = xm.view(xm.shape[0], -1)
        
        xa = self.excitation(xa)
        xm = self.excitation(xm)
        
        xc = xa+xm
        xc = xc.reshape(x.shape[0], x.shape[1], 1, 1)
        
        if self.return_score:
            return xc
        else:
            return xc*x

# Conventional Spatial Attention Module (CBAM)
class SA_module(nn.Module):
    def __init__(self, k=3, return_score=False):
        super().__init__()
        self.return_score = return_score
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, (k, k), padding=1, bias=False), 
            nn.BatchNorm2d(1), 
            nn.Sigmoid())
            
    def forward(self, x):
        xa = torch.unsqueeze(torch.mean(x, axis=1), 1)
        xm = torch.unsqueeze(torch.max(x, axis=1)[0], 1)
        xs = torch.concat((xa, xm), axis=1)
        xs = self.conv(xs)
        
        if self.return_score:
            return xs
        else:
            return x*xs

# Global-Local Attention Module
class GL_module(nn.Module):
    def __init__(self, dim_in_L, dim_in_G, margin=[0, 0, 0, 0], 
                 r_projection_size=1/1, Ss=(64, 4), 
                 simple_att_l=True, simple_att_g=True, gl_att=True, 
                 mode='cos', final_projection=False, sm=False):
        super().__init__()
        '''
        dim_in_L: dim of the local latent
        dim_in_G: dim of the global latent
        margin: region exclusion of the local feature
        r_projection_size: for computational efficency, dim can be controlled by this factor
        ★ Ss: size of local and global latent, It should be set correctly according to the feature size
        simple_att_l: conventional attnetion for local latent
        simple_att_g: conventional attention for global latent
        gl_all: global-local attention
        mode: distance (similarity) measure function 
        final_projection: for computational efficency
        sm: use softmax for normalization
        '''
        self.simple_att_g = simple_att_g
        self.simple_att_l = simple_att_l
        self.gl_att = gl_att
        self.mode = mode
        self.sm = sm
        self.final_projection = final_projection
        
        self.r_projection_size = r_projection_size
        self.dim_down = int(dim_in_L*r_projection_size)
        
        self.Size = Ss[0]
        self.size = Ss[1]
        self.margin = margin
        self.N = int(self.Size/self.size)
        self.rn = (self.N-(self.margin[0]+self.margin[1]))*(self.N-(self.margin[2]+self.margin[3]))
        
        self.ca_block_g = CA_module(dim_in_G, return_score=True)
        self.sa_block_g = SA_module(k=3, return_score=True)
        self.ca_block_l = CA_module(dim_in_L, return_score=True)
        self.ca_block_r = CA_module(self.rn, return_score=True)
        
        self.projection_g = nn.Sequential(
            nn.Conv2d(dim_in_G, self.dim_down, (1, 1), bias=False), 
            nn.BatchNorm2d(self.dim_down), 
            nn.SiLU())
        if dim_in_L!=self.dim_down:
            self.projection_l = nn.Sequential(
                nn.Conv2d(dim_in_L, self.dim_down, (1, 1), bias=False), 
                nn.BatchNorm2d(self.dim_down), 
                nn.SiLU())
        else:
            self.projection_l = nn.Sequential()
        
        if final_projection:
            self.projection_g_final = nn.Sequential(
                nn.Conv2d(dim_in_G, int(dim_in_G/2), (1, 1), bias=False), 
                nn.BatchNorm2d(int(dim_in_G/2)), 
                nn.SiLU())
            self.projection_l_final = nn.Sequential(
                nn.Conv3d(self.rn, int((dim_in_G/dim_in_L)/2), (1, 1, 1), bias=False), 
                nn.BatchNorm3d(int((dim_in_G/dim_in_L)/2)), 
                nn.SiLU())
        else:
            self.projection_l_final = nn.Sequential(
                nn.Conv3d(self.rn, int((dim_in_G/dim_in_L)/1), (1, 1, 1), bias=False), 
                nn.BatchNorm3d(int((dim_in_G/dim_in_L)/1)), 
                nn.SiLU())
        
        self.distance = Distance_Module(mode=mode)
        
    def forward(self, L, G):
        B = L.shape[0]
        rn = self.rn
        
        # Patchnization
        p = torch.stack([L[:, :, i*self.size:(i+1)*self.size, j*self.size:(j+1)*self.size] \
                         for i in range(self.margin[0], self.N-self.margin[1]) \
                         for j in range(self.margin[2], self.N-self.margin[3])], axis=1)
        g = G
        '''
        p: [B, 256, 128, 4, 4]
        g: [B, 512, 4, 4]
        '''
        # Conventional Attention
        if self.simple_att_g:
            att_g = self.ca_block_g(g)
            g = att_g*g
            att_g = self.sa_block_g(g)
            g = att_g*g
        
        if self.simple_att_l:
            att_p = torch.mean(p, axis=1)
            att_p = self.ca_block_l(att_p).unsqueeze(1)
            p = att_p*p
        
        # Global-Local Attention
        if self.gl_att:
            p_ = p.reshape(B*rn, p.shape[2], p.shape[3], p.shape[4])
            p_ = self.projection_l(p_)
            g_ = self.projection_g(g)
            
            g_ = torch.unsqueeze(g_, axis=1).repeat(1, rn, 1, 1, 1)
            g_ = g_.reshape(B*rn, g_.shape[2], g_.shape[3], g_.shape[4])
        
            at_score = self.distance(p_, g_)
            at_score = at_score.reshape(B, rn).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            if self.mode=='l2': at_score = F.softmax(at_score, dim=1)
            if self.mode=='cos' and self.sm: at_score = F.softmax(at_score, dim=1)
            at_score = at_score.reshape(B, rn, 1, 1, 1)
        
            p = p*at_score
        
        # Projection
        p = self.projection_l_final(p)
        if self.final_projection:
            g = self.projection_g_final(g)
        p = p.reshape(B, p.shape[1]*p.shape[2], p.shape[3], p.shape[4])
        res = [p, g]
        
        meta = {'margin':self.margin, 
                'Size':self.Size, 
                'size':self.size, 
                'rn':rn}
        
        return res, meta, at_score

#  Resblock
class Res_block(nn.Module):
    def __init__(self, dim_in, dim_out, k=3, dr=1, down=1, act=nn.SiLU()):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.down = down
        self.act = act
        self.dr = dr
        dim_down = dim_out//dr
        stride = 2 if self.down==2 else 1

        if dr>1: # Bottle Neck
            self.encoder = nn.Sequential(
                nn.Conv2d(dim_in, dim_down, (1, 1), stride=1, bias=False), 
                nn.BatchNorm2d(dim_down), 
                self.act, 
                
                nn.Conv2d(dim_down, dim_down, (k, k), stride=stride, padding=k//2, bias=False), 
                nn.BatchNorm2d(dim_down), 
                self.act, 
                
                nn.Conv2d(dim_down, dim_out, (1, 1), stride=1, bias=False), 
                nn.BatchNorm2d(dim_out))
        else: # Res Block
            self.encoder = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, (k, k), stride=stride, padding=k//2, bias=False), 
                nn.BatchNorm2d(dim_out), 
                self.act, 
                
                nn.Conv2d(dim_out, dim_out, (k, k), stride=1, padding=k//2, bias=False), 
                nn.BatchNorm2d(dim_out))
        
        self.eq_channel = nn.Conv2d(dim_in, dim_out, (1, 1), stride=1)
        self.eq_size_up = nn.Upsample(scale_factor=1/down, mode='bilinear', align_corners=True)
        self.eq_size_down = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        y = self.encoder(x)
        if y.shape[1]!=x.shape[1]: 
            x = self.eq_channel(x)
        if self.down==2: 
            x = self.eq_size_down(x)
        if self.down==0.5: 
            x = self.eq_size_up(x)
            y = self.eq_size_up(y)
        y = self.act(y+x)
        
        return y

# Correlatino Volume
def corr_operation(fmap1, fmap2, window=1, stride=5, radius=8, f16=True):
    '''
    window: kernel radius
    radius: ROI radius, redius > window
    stride: stride of ROI
    '''
    dt = fmap1.dtype
    no_grad = True
    k = window*2+1
    if no_grad: context_manager = torch.no_grad()    
    else: context_manager = contextlib.nullcontext()
    
    with context_manager:
        fmap_B = fmap1.shape[0]
        fmap_C = fmap1.shape[1]
        fmap_Sx = fmap1.shape[2]
        fmap_Sy = fmap1.shape[3]
        padding = (radius, radius, radius, radius)
        fmap1 = F.pad(fmap1, mode='constant', pad=padding, value=0)
        fmap2 = F.pad(fmap2, mode='constant', pad=padding, value=0)
        if f16: fmap1, fmap2 = fmap1.to(torch.float16), fmap2.to(torch.float16)
        
        fmap1 = fmap1[:, :, radius-1:radius-1+fmap_Sx, radius-1:radius-1+fmap_Sy].unfold(2, k, stride).unfold(3, k, stride)
        fmap1 = fmap1.reshape(fmap_B, fmap_C, -1, k, k)
        fmap1 = fmap1.permute(2, 0, 1, 3, 4)
        fmap1 = torch.swapaxes(fmap1, 0, 1)
        
        mask1 = torch.norm(fmap1, dim=[3, 4])
        mask1 = torch.norm(mask1, dim=2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fmap1 = fmap1 / (mask1 + 1e-6)
        fmap1 = fmap1.unsqueeze(2)
        
        fmap2 = fmap2.unfold(2, radius*2+1, stride).unfold(3, radius*2+1, stride)
        fmap2 = fmap2.reshape(fmap_B, fmap_C, -1, radius*2+1, radius*2+1)
        fmap2 = fmap2.permute(2, 0, 1, 3, 4)
        fmap2 = torch.swapaxes(fmap2, 0, 1)
        
        fmap2 = fmap2.unfold(3, k, 1).unfold(4, k, 1)
        fmap2 = fmap2.reshape(fmap_B, fmap2.shape[1], fmap_C, -1, k, k)
        fmap2 = fmap2.permute(3, 0, 1, 2, 4, 5)
        fmap2 = torch.swapaxes(torch.swapaxes(fmap2, 0, 1), 1, 2)
        
        mask2 = torch.norm(fmap2, dim=[4, 5])
        mask2 = torch.norm(mask2, dim=3).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fmap2 = fmap2 / (mask2 + 1e-6)
        
        res = fmap1 * fmap2
        res = torch.sum(res, dim=[3, 4, 5])
        B, C, S = res.shape[0], res.shape[1], int(res.shape[2]**0.5)
        res = res.reshape(B, C, S, S)
        res = torch.swapaxes(res, -1, -2)
        res = res.to(dt)
        
    return res

def corr_operation_F(fmap1, fmap2, window=1, stride=5, radius=8, f16=True):
    '''
    window: kernel radius
    radius: ROI radius, redius > window
    stride: stride of ROI
    '''
    dt = fmap1.dtype
    no_grad = True
    if no_grad: context_manager = torch.no_grad()    
    else: context_manager = contextlib.nullcontext()
    
    with context_manager:
        fmap_B = fmap1.shape[0]
        fmap_C = fmap1.shape[1]
        fmap_S = fmap1.shape[2]
        padding = (radius, radius, radius, radius)
        fmap1 = F.pad(fmap1, mode='constant', pad=padding, value=0)
        fmap2 = F.pad(fmap2, mode='constant', pad=padding, value=0)
        if f16: fmap1, fmap2 = fmap1.to(torch.float16), fmap2.to(torch.float16)
        
        fmap1 = torch.stack([fmap1[:, :, x-window:x+window+1, y-window:y+window+1] \
                                for y in range(radius, radius+fmap_S, stride) \
                                for x in range(radius, radius+fmap_S, stride)])
        fmap1 = torch.swapaxes(fmap1, 0, 1)
                
        fmap2 = torch.stack([fmap2[:, :, x-radius:x+radius+1, y-radius:y+radius+1] \
                                for y in range(radius, radius+fmap_S, stride) \
                                for x in range(radius, radius+fmap_S, stride)])
        fmap2 = torch.swapaxes(fmap2, 0, 1)
        fmap2 = torch.stack([fmap2[:, :, :, x-window:x+window+1, y-window:y+window+1] \
                                for y in range(window, -window+radius*2+1) \
                                for x in range(window, -window+radius*2+1)])
        fmap2 = torch.swapaxes(torch.swapaxes(fmap2, 0, 1), 1, 2)
        
        fmap1 = fmap1.unsqueeze(2)
        fmap1 = fmap1.repeat(1, 1, fmap2.shape[2], 1, 1, 1)
        fmap1 = fmap1.reshape(fmap1.shape[0], fmap1.shape[1], fmap1.shape[2], -1)
        fmap2 = fmap2.reshape(fmap2.shape[0], fmap2.shape[1], fmap2.shape[2], -1)
        res = F.cosine_similarity(fmap1, fmap2, dim=-1)
        res = res.reshape(res.shape[0], res.shape[1], int(res.shape[2]**0.5), int(res.shape[2]**0.5))
        res = res.to(dt)
        
    return res

class Corr_block(nn.Module):
    def __init__(self, window=1, stride=2, radius=8, f16=True, scale_factor=16/15):
        super().__init__()
        self.window = window
        self.stride = stride
        self.radius = radius
        self.f16 = f16
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        
    def forward(self, fmap1, fmap2):
        cv = corr_operation(fmap1, fmap2, 
                            window=self.window, stride=self.stride, radius=self.radius, f16=self.f16)
        cv = self.upsample(cv)

        return cv
    
### Models
class MoGLo_Net(nn.Module):
    def __init__(self, dim_in=1, dim_base=64, shape=False, c_att=True, gl_att=True):
        super().__init__()
        '''
        ★ input shape: [Batch, Seq, C, 256, 256]
        dim_in: input dim
        dim_base: dim controlling factor
        shape: print shape for debugging
        c_att: conventional attention
        gl_att: glocal-local attention
        '''
        self.shape = shape
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_base*1, (7, 7), stride=2, padding=7//2), 
            nn.BatchNorm2d(dim_base*1), 
            nn.ReLU(), # 128
            
            Res_block(dim_base*1, dim_base*1), 
            Res_block(dim_base*1, dim_base*1), 
            Res_block(dim_base*1, dim_base*1, down=2)) # 64
            
        self.encoder_2 = nn.Sequential(
            Res_block(dim_base*2, dim_base*2), 
            Res_block(dim_base*2, dim_base*2), 
            Res_block(dim_base*2, dim_base*2), 
            Res_block(dim_base*2, dim_base*2, down=1)) # 64
        
        self.encoder_3 = nn.Sequential(
            Res_block(dim_base*2, dim_base*4, down=2), # 32
            Res_block(dim_base*4, dim_base*4), 
            Res_block(dim_base*4, dim_base*4), 
            Res_block(dim_base*4, dim_base*4), 
            Res_block(dim_base*4, dim_base*4), 
            Res_block(dim_base*4, dim_base*4, down=2)) # 16
        
        self.encoder_4 = nn.Sequential(
            Res_block(dim_base*4+256, dim_base*8), 
            Res_block(dim_base*8   , dim_base*8),
            Res_block(dim_base*8   , dim_base*8, down=2)) # 8
            
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # 4
        self.pool_adap = nn.AdaptiveAvgPool2d((1, 1)) # 1
        
        self.corr = Corr_block(window=1, stride=4, radius=8, scale_factor=16/15)
        self.att = GL_module(dim_in_L=dim_base*2, dim_in_G=dim_base*8, 
                             margin=[0, 0, 0, 0], Ss=(64, 4), 
                             simple_att_l=c_att, simple_att_g=c_att, gl_att=gl_att)
        
        self.lstm_L = nn.LSTM(dim_base*8, dim_base*8, num_layers=1, batch_first=True)
        self.lstm_G = nn.LSTM(dim_base*8, dim_base*8, num_layers=1, batch_first=True)
        self.fc_L = nn.Linear(dim_base*8, 6)
        self.fc_G = nn.Linear(dim_base*8, 6)
        
    def forward(self, x, o=None):
        # Encoder
        B, S = x.shape[0], x.shape[1] - 1
        printt(f'{"B_data":<30}: {x.shape}', self.shape)
        x1 = x[:,   :-1, :, :, :]
        x2 = x[:,  1:  , :, :, :]
        printt(f'{"x1":<30}: {x1.shape}', self.shape)
        printt(f'{"x2":<30}: {x2.shape}', self.shape)
        printt(f'{"=="*35}', t=self.shape)
        
        printt(f'{"@ Encoded Fmap 1"}', t=self.shape)
        x1 = x1.reshape(B*S, x1.shape[2], x1.shape[3], x1.shape[4])
        x2 = x2.reshape(B*S, x2.shape[2], x2.shape[3], x2.shape[4])
        x1, x2 = parallel_run(self.encoder_1, [x1, x2])
        printt(f'{"x1":<30}: {x1.shape}', self.shape)
        printt(f'{"x2":<30}: {x2.shape}', self.shape)
        printt(f'{"=="*35}', t=self.shape)
        
        printt(f'{"@ Corr Operation"}', t=self.shape)
        cv = self.corr(x1, x2)
        printt(f'{"cv":<30}: {cv.shape}', t=self.shape)
        printt(f'{"=="*35}', t=self.shape)
        
        printt(f'{"@ Encoded Fmap 2"}', t=self.shape)
        x = torch.concat([x1, x2], axis=1)
        printt(f'{"x1+x2 (Concat)":<30}: {x.shape}', self.shape)
        x = self.encoder_2(x)
        xl = x
        printt(f'{"x (Local)":<30}: {x.shape}', self.shape)
        printt(f'{"=="*35}', t=self.shape)
        
        printt(f'{"@ Encoded Fmap 3"}', t=self.shape)
        x = self.encoder_3(x)
        printt(f'{"x":<30}: {x.shape}', self.shape)
        printt(f'{"=="*35}', t=self.shape)
        
        printt(f'{"@ Encoded Fmap 4"}', t=self.shape)
        x = torch.concat([x, cv], axis=1)
        printt(f'{"x+cv (concat)":<30}: {x.shape}', self.shape)
        x = self.encoder_4(x)
        printt(f'{"x":<30}: {x.shape}', self.shape)
        printt(f'{"=="*35}', t=self.shape)
        
        
        printt(f'{"@ Attention"}', t=self.shape)
        x = self.pool(x)
        printt(f'{"x (Avg Pool)":<30}: {x.shape}', self.shape)
        x_emb, meta, at_score = self.att(xl, x)
        L, G = x_emb[0], x_emb[1]
        x_emb = torch.concat(x_emb, axis=1)
        
        printt(f'{"L":<30}: {L.shape}', self.shape)
        printt(f'{"G":<30}: {G.shape}', self.shape)
        
        L = self.pool_adap(L)
        G = self.pool_adap(G)
        printt(f'{"G (Adap Pool)":<30}: {G.shape}', self.shape)
        printt(f'{"L (Adap Pool)":<30}: {L.shape}', self.shape)
        L = L.reshape(B, S, -1)
        G = G.reshape(B, S, -1)
        printt(f'{"G (Reshape)":<30}: {G.shape}', self.shape)
        printt(f'{"L (Reshape)":<30}: {L.shape}', self.shape)
        printt(f'{"=="*35}', t=self.shape)

        printt(f'{"@ Estimator"}', t=self.shape)
        L = self.lstm_L(L)[0]
        G = self.lstm_G(G)[0]
        printt(f'{"L (LSTM)":<30}: {L.shape}', self.shape)
        printt(f'{"G (LSTM)":<30}: {G.shape}', self.shape)
        L = self.fc_L(L)
        G = self.fc_G(G)
        printt(f'{"L (Output)":<30}: {L.shape}', self.shape)
        printt(f'{"G (Output)":<30}: {G.shape}', self.shape)
        
        return [L, G], x_emb, at_score