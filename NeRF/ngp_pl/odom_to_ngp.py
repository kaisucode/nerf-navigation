import json
import os
import numpy as np
from scipy.spatial.transform import Rotation
from utils import load_transform_json, align_odom_to_colmap


def rotateAxis(degrees, axis):
    '''
    Function to rotate around given axis
    Input:
        degrees - scalar - Angle in degrees
        
        axis - scalar - options:
            0 - around x axis
            1 - around y axis
            2 - around z axis  
    
    Returns:
        Homogeneous rotation matrix
    '''

    radians = np.radians(degrees)

    if axis == 2: # z - axis

        rotation_mat = np.array([[np.cos(radians), -np.sin(radians),           0,          0],
                                 [np.sin(radians),  np.cos(radians),           0,          0],
                                 [              0,                0,           1,          0],
                                 [              0,                0,           0,          1]])

    elif axis == 1: # y - axis

        rotation_mat = np.array([[np.cos(radians),                0,  np.sin(radians),          0],
                                 [              0,                1,                0,          0],
                                 [-np.sin(radians),               0, np.cos(radians),          0],
                                 [              0,                0,                0,          1]])

    elif axis == 0: # x - axis


        rotation_mat = np.array([[             1,                0,                0,          0],
                                [              0,  np.cos(radians), -np.sin(radians),          0],
                                [              0,  np.sin(radians),  np.cos(radians),          0], 
                                [              0,                0,                0,          1]])
    
    return rotation_mat

def translateMatrix(x, y, z):
    
    translation_matrix = np.eye(4)
    translation_matrix[0,3] += x
    translation_matrix[1,3] += y
    translation_matrix[2,3] += z

    return translation_matrix

def save_as_transform_json(parent, poses):
    data = load_transform_json(parent, 0, -1)
    for i in range(len(data['frames'])):
        data['frames'][i]['transform_matrix'] = poses[i].tolist()
    d = json.dumps(data)
    with open(os.path.join(parent, "transforms_odom.json"), "w") as out:
        out.write(d)

def read_camera_json_sorted(path, start = 0, end = -1):
    data = load_transform_json(path, start, end)
    
    print(len(data["frames"]))
    poses = []

    frame_change = rotateAxis(-90, 2) @ rotateAxis(-90, 0)
    # bottom = np.array([[0, 0, 0, 1.]]) 

    for cam_idx in range(len(data["frames"])):
        T_c = data['frames'][cam_idx]["transform_matrix"] #   # 4 x 4
        poses.append(T_c)
    
    poses = np.array(poses)
    #print(poses.shape)
    return poses

def get_odom_to_nerf_matrix(parent, ts, qs, scale=1):
    data = load_transform_json(parent, 0, -1)

    frame_change = rotateAxis(-90, 2) @ rotateAxis(-90, 0) #@ rotateAxis(180, 0)
    bottom = np.array([[0, 0, 0, 1.]]) 
    
    poses_colmap = []
    poses_odom = []
    for i in range(len(data['frames'])):
        cam_id = int(data['frames'][i]['file_path'].split(os.sep)[-1][:-4])
        poses_colmap.append(data['frames'][i]['transform_matrix'])
        
        # quaternion to rotation matrix
        R = Rotation.from_quat(np.array([qs[cam_id][1], qs[cam_id][2], 
                                         qs[cam_id][3], qs[cam_id][0]])).as_matrix()
        # to 4 x 4 matrix
        c2w = np.vstack([np.hstack([R, ts[cam_id][..., None]]), bottom]) @ frame_change
        c2w[:3, 3] *= scale
        #c2w[:3, 3] += np.random.normal(0, 1, 3)*0.00001

        poses_odom.append(c2w)

    # N x 4 x 4
    # N x 4 x 4
    poses_colmap = np.array(poses_colmap)
    poses_odom = np.array(poses_odom)

    c, R, t = align_odom_to_colmap(poses_colmap[:, :3], poses_odom[:, :3])
    #print(c, R, t)
    return c*R, t

def odom_to_nerf(parent, ts, qs, o2n, offset, scale=1, to_ngp=False):
    frame_change = rotateAxis(-90, 2) @ rotateAxis(-90, 0)# @ rotateAxis(180, 0)
    bottom = np.array([[0, 0, 0, 1.]]) 

    poses = []
    for i in range(len(ts)):
        # quaternion to rotation matrix
        R = Rotation.from_quat(np.array([qs[i][1], qs[i][2], qs[i][3], qs[i][0]])).as_matrix()
        # to nerf coordinate
        P = np.vstack([np.hstack([R, ts[i][..., None]]), bottom]) @ frame_change # 4, 4     
        #P[:3, 3] *= scale
        P[:3, 3] = o2n @ P[:3, 3]
        P[:3, 3] += offset
        # we rescale scene by 20 in ngp dataloader
        if to_ngp:
            P[:, 1:3] *= -1
            P[:3, 3] /= 20
        poses.append(P)
    return np.array(poses)


if __name__=="__main__":
    parent = "../../../spot_data/"
    # read T and Q from odom
    ts = np.load(os.path.join(parent, "arr_2.npy"))
    qs = np.load(os.path.join(parent, "arr_3.npy"))
    
    o2n, offset = get_odom_to_nerf_matrix(parent, ts, qs, 1)
    #print(o2n)
    #print(offset)

    poses = odom_to_nerf(parent, ts, qs, o2n, offset, scale=1, to_ngp=True)
    print(poses.shape)

    #poses = odom_to_nerf(parent, ts, qs, o2n, offset, scale=1, to_ngp=True)
    #save_as_transform_json(parent, poses)
