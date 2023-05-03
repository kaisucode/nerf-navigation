import torch
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from .ray_utils import get_ray_directions, rotateAxis, translateMatrix
from .color_utils import read_image, preprocess_image

from .base import BaseDataset
from scipy.spatial.transform import Rotation
import cv2
import open3d as o3d



def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def visualize_poses(poses):
    """_summary_

    Args:
        poses (_type_): N x 4 x 4 - w2c frame
    """
    

    origin = np.array([0.0, 0.0, 0.0, 1.0])[..., None] # 4, 1
    mesh_list = []
    for p_idx in range(poses.shape[0]):
        cam_pose = poses[p_idx] @ origin

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1.0 / 20.0)
        mesh.rotate(poses[p_idx, :3, :3])
        mesh.translate(cam_pose[:3, -1])
        mesh_list.append(mesh)
    
    o3d.visualization.draw_geometries(mesh_list)



    


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, pose_avg


def debug_poses(data_path, start_idx = 0, end_idx = 20, scale = 20.0, poses_in = None):


    if poses_in is None:
        dataset_array = np.load(data_path)


        ts = dataset_array["arr_2"]
        qs = dataset_array["arr_3"]

    frame_change = rotateAxis(-90, 2) @ rotateAxis(-90, 0)
    poses = []
    bottom = np.array([[0, 0, 0, 1.]])
    if end_idx < 0.0:
        if poses_in is None:
            end_idx = ts.shape[0]
        else:
            end_idx = poses_in.shape[0]


        
    for i in range(start_idx, end_idx):
        
        if poses_in is None:

            t = ts[i] # 3
            q = qs[i] # 4
            R = Rotation.from_quat(np.array([q[1], q[2], q[3], q[0]])).as_matrix()
            T = np.vstack([np.hstack([R, t[..., None]]), bottom]) @ frame_change # 4, 4 
        else:
            T = poses_in[i]

        poses.append(T)

    poses = (np.stack(poses, 0)[:, :3]) # N, 4, 4
    poses[:, :3, -1] = poses[:, :3, -1] / scale
    print(poses.shape)

    visualize_poses(poses)

class SpotDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        self.start_idx = 0
        self.end_idx = -1

        self.read_intrinsics()
        self.read_data()

        # if kwargs.get('read_meta', True):
        #     self.read_meta(split)

    def read_intrinsics(self):
        
        h = int(480.0 * self.downsample)
        w = int(640.0 * self.downsample)
        fx = 538.64 * self.downsample
        fy = 521.76 * self.downsample

        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])

        self.K = K
        # self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)
        # self.directions = get_ray_directions(h, w, self.K)

        k1 =  0.10407162296473346
        k2 = -0.17195930869305776
        k3 =  0
        k4 =  0
        p1 = -0.001478066445083941
        p2 = 0.000508493977714055
        self.distortion_params = np.array([k1, k2, p1, p2])
        '''
        "camera_angle_x": 1.0721156444887512,
        "camera_angle_y": 0.8622471142946126,
        "fl_x": 538.6447623568952,
        "fl_y": 521.7599966228122,
        "k1": 0.10407162296473346,
        "k2": -0.17195930869305776,
        "k3": 0,
        "k4": 0,
        "p1": -0.001478066445083941,
        "p2": 0.000508493977714055,
        "is_fisheye": false,
        "cx": 320.60462050513354,
        "cy": 238.34617146194952,
        "w": 640.0,
        "h": 480.0,
        "aabb_scale": 32,
        '''


    def read_data(self):
        # self.root_dir = os.path.join(self.root_dir, "") 
        
        scale = 20.0
        self.dataset_array = np.load(self.root_dir)

        # x- front, y-left, z-up

        frame_change = rotateAxis(-90, 2) @ rotateAxis(-90, 0)


        images = self.dataset_array["arr_0"]
        depths = self.dataset_array["arr_1"]
        ts = self.dataset_array["arr_2"]
        qs = self.dataset_array["arr_3"]
        times = self.dataset_array["arr_4"]

        self.rays = []
        self.poses = []
        self.depths = []

        bottom = np.array([[0, 0, 0, 1.]])
        
        if self.end_idx < 0:
            self.end_idx = images.shape[0]
        for i in range(self.start_idx, self.end_idx):
            
            im = cv2.cvtColor(images[i].astype(np.float32) / 255.0, cv2.COLOR_RGB2BGR)# H, W, 3
            

            new_K, roi = cv2.getOptimalNewCameraMatrix(self.K, self.distortion_params, self.img_wh, 1, self.img_wh)

            # im = cv2.undistort(im, self.K, self.distortion_params, None, new_K)
            t = ts[i] # 3
            q = qs[i] # 4
            d = (depths[i] / 1000.0) / scale # / (depths[i].max() / 1000.0 + 1e-6) # H, W, 1
            # print(d.max())
            d = cv2.undistort(d, self.K, self.distortion_params, None, new_K)[..., None]
            # print(d.max())
            d_meters = d #(d / 1000.0) / (d.max() + 1e-6) # Converting to meters
            # print()

            # print(im.shape, d.shape)

            R = Rotation.from_quat(np.array([q[1], q[2], q[3], q[0]])).as_matrix()
            T = np.vstack([np.hstack([R, t[..., None]]), bottom]) @ frame_change # 4, 4
            

            # print(T)
            # print()
            # plt.imshow(d_meters)
            # plt.show()

            #  ************** undistort images!!!!! *******
            im_rays = (preprocess_image(im, self.img_wh, True)) # (h  w), 3


            depth_rays = preprocess_image(d_meters, self.img_wh, True) # (h w), 1
            
            # depth_rays_meters
            


            self.rays.append(im_rays)
            self.depths.append(depth_rays)
            self.poses.append(T)

        
        # self.K = torch.FloatTensor(new_K)
        self.K = torch.FloatTensor(self.K)

        self.directions = torch.FloatTensor(get_ray_directions(self.img_wh[1], self.img_wh[0], self.K))
        # Center poses!
        self.poses = torch.FloatTensor(np.stack(self.poses, 0)[:, :3]) # N, 4, 4
        self.poses[:, :3, -1] = self.poses[:, :3, -1] / scale

        # visualize_poses(self.poses[:20])

        # print(self.poses[:, :3, -1].min(), self.poses[:, :3, -1].max())
        self.rays = torch.FloatTensor(np.stack(self.rays, 0)) # N, (h w), c
        self.depths = torch.FloatTensor(np.stack(self.depths, 0).astype(np.float32)) #
        print(self.poses.shape, self.rays.shape, "dimensions")
        print(self.rays.max(), self.rays.min())
        # Convert to torch

        # print("done")
        
    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            rays = self.rays[img_idxs, pix_idxs]
            depths = self.depths[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3], "depth": depths}
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample




if __name__== "__main__":

    # root_dir = "/home/rahul/Education/Brown/1_sem2/CSCI2952-O/Project/data"
    # root_dir = "/home/rahul/Education/Brown/1_sem2/CSCI2952-O/Project/data/spot_data_numpy-20230430T235508Z-001/spot_data_numpy/04282023/spot_data_0.npz"

    # dataloader = SpotDataset(root_dir, split = "train", downsample = 1.0)
    # root_dir = "/home/rahul/Education/Brown/1_sem2/CSCI2952-O/Project/data"
    # root_dir = "/home/rahul/Education/Brown/1_sem2/CSCI2952-O/Project/data/spot_data_numpy-20230430T235508Z-001/spot_data_numpy/04282023/spot_data_0.npz"
    root_dir = "/home/rahulsajnani/Education/Brown/1_sem2/52-O/project/data/04282023-20230501T003435Z-001/04282023/spot_data_0.npz"
    # dataloader = SpotDataset(root_dir, split = "train", downsample = 1.0)

    debug_poses(root_dir, end_idx = -1, scale = 5.0)


    pass
