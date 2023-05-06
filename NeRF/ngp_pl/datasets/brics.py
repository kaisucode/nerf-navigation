import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset
from .spot import debug_poses, SpotDataset, visualize_poses
import open3d as o3d

from odom_to_ngp import read_camera_json_sorted
from utils import frame_path, align_odom_to_colmap


def visualize_poses_dual(poses_1, poses_2):
    """_summary_

    Args:
        poses (_type_): N x 4 x 4 - w2c frame
    """
    

    origin = np.array([0.0, 0.0, 0.0, 1.0])[..., None] # 4, 1
    mesh_list = []
    for p_idx in range(poses_1.shape[0]):
        cam_pose = poses_1[p_idx] @ origin
        cam_pose_2 = poses_2[p_idx] @ origin

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1.0 / 20.0)
        mesh_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1.0 / 20.0)

        mesh.rotate(poses_1[p_idx, :3, :3])
        mesh.translate(cam_pose[:3, -1])

        mesh_2.rotate(poses_2[p_idx, :3, :3])
        mesh_2.translate(cam_pose_2[:3, -1])

        mesh_list.append(mesh)
        mesh_list.append(mesh_2)
    
    o3d.visualization.draw_geometries(mesh_list)



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
    
    def get_all_poses(self):
        return self.poses


    def read_meta(self, split):
        self.rays = []
        self.poses = []
        
        if split == 'trainval':
            with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
                frames = json.load(f)["frames"]
                frames.sort(key=frame_path)
            with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
                frames+= json.load(f)["frames"]
                frames.sort(key=frame_path)

        else:
            with open(os.path.join(self.root_dir, f"transforms.json"), 'r') as f:
                # frames = ["frames"]
                frames = json.load(f)["frames"]
                frames.sort(key=frame_path)

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
        print(self.poses.shape)

if __name__=="__main__":
    
    # root_dir = "/home/rahulsajnani/Education/Brown/1_sem2/52-O/project/data/spot_data_colmap/spot_data_0/"
    root_dir = "../../../data/spot_data_0/"
    root_dir = os.path.join(root_dir,"")
    dataloader = BRICSDataset(root_dir=root_dir, split = "val")

    # poses = read_camera_json_sorted(root_dir)
    # poses[:, 1:3] *= -1

    root_dir_odom = "/home/rahul/Education/Brown/1_sem2/CSCI2952-O/Project/data/spot_data_numpy-20230430T235508Z-001/spot_data_numpy/04282023/spot_data_0.npz"

    dataloader_odom = SpotDataset(root_dir_odom, split = "train", downsample = 1.0)
    poses_colmap = dataloader.get_all_poses()
    poses_odom = dataloader_odom.get_all_poses()
    
    # debug_poses(None, scale = 1.0, end_idx = -1, poses_in = poses_colmap)
    # debug_poses(None, scale = 1.0, end_idx = -1, poses_in = poses_odom)
    
    # debug_poses(None, scale = 20.0, end_idx = -1, poses_in = poses)

    # scalar, 3x3, 3
    # N, 3, 4
    c, R, t = align_odom_to_colmap(poses_colmap, poses_odom)

    R_orient = c * R 
    # t_orient = t[..., None]
    
    bottom = np.array([[0, 0, 0, 1.]])
    T = np.vstack([np.hstack([R_orient, t[..., None]]), bottom]) # 4, 4

    bottom_torch = torch.tensor(bottom)[None].repeat(poses_odom.shape[0], 1, 1)

    poses_odom = torch.concat([poses_odom, bottom_torch], 1)[:, :3]

    print(poses_odom.shape, T.shape, poses_colmap.shape, bottom_torch.shape)

    poses_new_odom = (torch.tensor(T)[None] @ torch.concat([poses_odom, bottom_torch], 1))[:, :3]
    # poses_odom = torch.tensor(T) @ poses_odom

    visualize_poses(poses_colmap)
    visualize_poses(poses_odom)
    visualize_poses_dual(poses_colmap, poses_odom)
    visualize_poses_dual(poses_colmap, poses_new_odom)



