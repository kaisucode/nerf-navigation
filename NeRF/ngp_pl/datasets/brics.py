import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset


class BRICSDataset(BaseDataset):
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

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'trainval':
            with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
                frames = json.load(f)["frames"]
            with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
                frames+= json.load(f)["frames"]
        else:
            with open(os.path.join(self.root_dir, f"transforms.json"), 'r') as f:
                frames = json.load(f)["frames"]
        pos = np.array([0., 0., 0.])
        print(f'Loading {len(frames)} {split} images ...')
        for frame in tqdm(frames):
            c2w = np.array(frame['transform_matrix'])[:3, :4]

            # determine scale
            if 'Jrender_Dataset' in self.root_dir:
                c2w[:, :2] *= -1 # [left up front] to [right down front]
                folder = self.root_dir.split('/')
                scene = folder[-1] if folder[-1] != '' else folder[-2]
                if scene=='Easyship':
                    pose_radius_scale = 1.2
                elif scene=='Scar':
                    pose_radius_scale = 1.8
                elif scene=='Coffee':
                    pose_radius_scale = 2.5
                elif scene=='Car':
                    pose_radius_scale = 0.8
                else:
                    pose_radius_scale = 1.5
            else:
                c2w[:, 1:3] *= -1 # [right up back] to [right down front]
                #pose_radius_scale = 1.1 #1.5
            #c2w[:, 3] /= np.linalg.norm(c2w[:, 3])/pose_radius_scale
            
            
            # add shift
            if 'Jrender_Dataset' in self.root_dir:
                if scene=='Coffee':
                    c2w[1, 3] -= 0.4465
                elif scene=='Car':
                    c2w[0, 3] -= 0.7
            self.poses += [c2w]
            pos += c2w[:, 3]            

            try:
                img_path = os.path.join(self.root_dir, f"{frame['file_path']}")
                img = read_image(img_path, self.img_wh)
                # print(img.shape)
                self.rays += [img]
            except: pass

        # center pos to [0, 0, 0]
        # and bound it by [-0.5, 0.5]
        for i in range(len(self.poses)):
            # self.poses[i][:, 3] -= np.array([-0.10901881, -0.06427684,  0.27593734])
            #print(pos/len(self.poses))
            self.poses[i][:, 3] /= 20.
            #print(self.poses[i][:, 3])        
       
        if len(self.rays)>0:
            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(np.array(self.poses)) # (N_images, 3, 4)


if __name__=="__main__":
    
    root_dir = "/home/rahulsajnani/Education/Brown/1_sem2/52-O/project/data/spot_data_colmap/spot_data_0/"
    dataloader = BRICSDataset(root_dir=root_dir)