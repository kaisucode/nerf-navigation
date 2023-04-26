import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image, preprocess_image

from .base import BaseDataset
from scipy.spatial.transform import Rotation



def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


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

class SpotDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        self.read_data()
        # if kwargs.get('read_meta', True):
        #     self.read_meta(split)

    def read_intrinsics(self):
        
        h = 480.0 * self.downsample
        w = 640.0 * self.downsample
        fx = 538.64 * self.downsample
        fy = 521.76 * self.downsample

        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)
        self.directions = get_ray_directions(h, w, self.K)
        self.distortion_params = [0.10407162296473346, -0.17195930869305776]
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
        self.root_dir = os.path.join(self.root_dir, "") 
        self.dataset_array = np.load(self.root_dir + "spot_data.npz")

        images = self.dataset_array["arr_0"]
        depths = self.dataset_array["arr_1"]
        ts = self.dataset_array["arr_2"]
        qs = self.dataset_array["arr_3"]
        times = self.dataset_array["arr_4"]

        self.rays = []
        self.poses = []
        self.depths = []

        bottom = np.array([[0, 0, 0, 1.]])
        for i in range(images.shape[0]):
            
            im = images[i].astype(np.float32) / 255.0 # H, W, 3
            t = ts[i] # 3
            q = qs[i] # 4
            d = depths[i] # H, W, 1

            R = Rotation.from_quat(np.array([q[1], q[2], q[3], q[0]])).as_matrix()
            T = np.vstack([np.hstack(R, t[..., None]), bottom[None]]) # 4, 4
            
            #  ************** undistort images!!!!! *******
            im_rays = (preprocess_image(im, self.img_wh, True)) # (h  w), 3
            depth_rays = preprocess_image(d, self.img_wh, True) # (h w), 1

            self.rays.append(im_rays)
            self.depths.append(depth_rays)
            self.poses.append(T)

        
        # Center poses!

        self.poses = np.stack(self.poses, 0) # N, 4, 4
        
        self.rays = np.stack(self.rays, 0) # N, (h w), c
        # Convert to torch





if __name__== "__main__":
    pass
