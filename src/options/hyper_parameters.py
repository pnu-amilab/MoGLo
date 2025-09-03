import os
import torch
from utils.utils import make_dir

class HP_Guhong():
    def __init__(self, name, device='cuda', seq=5, alpha=10):
        self.name = name
        self.device = device
        
        self.path_dataset = f'{os.path.dirname(os.getcwd())}/data/'
        self.split_type_scan = {'train':['uS1'], 
                                'valid':['uS1'], 
                                 'test':['uS1']}
        self.split_patient = {'train':[i for i in range(1, 2)], 
                              'valid':[i for i in range(1, 2)], 
                               'test':[i for i in range(1, 2)]}
        self.type_X = 'img_B'
        self.type_y = 'y_rel'
        self.type_angle = 'an31'
        
        self.seq = seq
        self.seq_total_mean = 500
        
        self.alpha = alpha
        self.epoch_load = False
        self.epochs = 1001
        self.save_cycle = 100
        self.monitoring_cycle = 1
        
        self.dim_base = 64
        self.batch_size = 14
        self.optimizer_lr = 1e-4
        self.scheduler_step = 100
        self.scheduler_gamma = 0.8
        
        self.path_res = f'{os.path.dirname(os.getcwd())}/res/{self.name}'
        self.path_Res = f'{os.path.dirname(os.getcwd())}/res'
        
        self.path_model = f'{self.path_res}/model'
        self.path_inference = f'{self.path_res}/inference'
        self.path_evaluation = f'{self.path_res}/evaluation'
        self.path_fig = f'{self.path_res}/fig'
        self.path_temp = f'{self.path_res}/temp'
        
        make_dir(self.path_Res)
        make_dir(f'{self.path_Res}/Fig')
        make_dir(self.path_res)
        make_dir(self.path_model)
        make_dir(self.path_inference)
        make_dir(self.path_evaluation)
        make_dir(self.path_fig)
        make_dir(self.path_temp)
        
        self.batch_size_inference = 24
        self.seq_inference = self.seq
