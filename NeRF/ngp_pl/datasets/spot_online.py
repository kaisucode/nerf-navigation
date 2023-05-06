import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import get_image

from .base import BaseDataset


class SpotOnlineDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, images=None, colmap_poses=None, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.colmap_poses = colmap_poses
        self.images = images
        self.read_intrinsics()
        
        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        meta = self.colmap_poses

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

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'trainval':
            frames = self.colmap_poses["frames"]
        else:
            frames = self.colmap_poses["frames"]
        pos = np.array([0., 0., 0.])
        print(f'Loading {len(frames)} {split} images ...')
        for frame in tqdm(frames):
            img_id = int(frame['file_path'].split(os.sep)[-1][:-4])
            #print(img_id)
            c2w = np.array(frame['transform_matrix'])[:3, :4]
            c2w[:, 1:3] *= -1 # [right up back] to [right down front]
            self.poses += [c2w]
            pos += c2w[:, 3]
            
            img_path = os.path.join(self.root_dir, f"{frame['file_path']}")
            img = get_image(self.images[img_id], self.img_wh)
            self.rays += [img]

        for i in range(len(self.poses)):
            self.poses[i][:, 3] /= 20.
       
        if len(self.rays)>0:
            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
