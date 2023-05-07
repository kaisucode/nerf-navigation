import torch
import os
import numpy as np
from models.networks import NGP
from utils import load_ckpt


class ngp_model:
    def __init__(self, scale, ckpt_path):
        self.model = NGP(scale=scale).cuda()
        load_ckpt(self.model, ckpt_path)      
    
    # xyz: (N, 3)
    def get_density(self, xyz):
        sigma = self.model.density(xyz)
        return sigma
    

if __name__=="__main__":
    scale = 0.5
    ckpt_path = "../../../spot_data/ckpts/spot_online/Spot/2_slim.ckpt"

    xyz = torch.tensor([[0., 0., 0.], 
                        [0.6, 0.7, 0.8], 
                        [-0.1, 0, 0.8], 
                        [0.3, 0.3, 0.3]]).cuda()

    ngp_model = ngp_model(scale, ckpt_path)
    sigma = ngp_model.get_density(xyz)
    print(sigma)
