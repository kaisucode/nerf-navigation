import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions, rotateAxis
from .color_utils import read_image

from .base import BaseDataset

from scipy.spatial.transform import Rotation

def frame_path(frame):
  return frame['file_path']

class SpotDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
            meta = json.load(f)
        # choose one camera
        # assume the intrinsic matrix are sharable
        # meta = meta['frames'][0]

        w = int(meta['w']*self.downsample)
        h = int(meta['h']*self.downsample)

        fx = 0.5*meta['w']/np.tan(0.5*meta['camera_angle_x'])*self.downsample
        fy = 0.5*meta['h']/np.tan(0.5*meta['camera_angle_y'])*self.downsample

        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)
    
    def get_all_poses(self):
        return self.poses


    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'train':
            with open(os.path.join(self.root_dir, f"transforms.json"), 'r') as f:
                frames = json.load(f)["frames"]
                frames.sort(key=frame_path)
        elif split == 'test':
            with open(os.path.join(self.root_dir, f"transforms.json"), 'r') as f:
                frames = json.load(f)["frames"]
                frames.sort(key=frame_path)


        pos = np.array([0., 0., 0.])
        print(f'Loading {len(frames)} {split} images ...')
        for frame in tqdm(frames):
            c2w = np.array(frame['transform_matrix'])[:3, :4]

            c2w[:, 1:3] *= -1
            
            self.poses += [c2w]
            pos += c2w[:, 3]            

            img_path = os.path.join(self.root_dir, f"{frame['file_path']}")
            #print(img_path)
            img = read_image(img_path, self.img_wh)
            self.rays += [img]
        
        # center pos to [0, 0, 0]
        # and bound it by [-0.5, 0.5]
        for i in range(len(self.poses)):
            self.poses[i][:, 3] /= 20.       
       
        if len(self.rays)>0:
            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(np.array(self.poses)) # (N_images, 3, 4)
