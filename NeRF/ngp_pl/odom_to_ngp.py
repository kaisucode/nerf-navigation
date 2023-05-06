import json
import os
import numpy as np
from scipy.spatial.transform import Rotation
from datasets.ray_utils import rotateAxis
from utils import load_transform_json


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

    frame_change = rotateAxis(-90, 2) @ rotateAxis(-90, 0) @ rotateAxis(180, 0)
    # bottom = np.array([[0, 0, 0, 1.]]) 

    for cam_idx in range(len(data["frames"])):
        T_c = data['frames'][cam_idx]["transform_matrix"] #   # 4 x 4
        poses.append(T_c)
    
    poses = np.array(poses)
    print(poses.shape)
    return poses

    

def get_odom_to_nerf_matrix(parent, ts, qs, scale=3):
    data = load_transform_json(parent, 0, -1)
    cam_id = int(data['frames'][0]['file_path'].split(os.sep)[-1][:-4])

    frame_change = rotateAxis(-90, 2) @ rotateAxis(-90, 0) @ rotateAxis(180, 0)
    bottom = np.array([[0, 0, 0, 1.]]) 

    # quaternion to rotation matrix
    R = Rotation.from_quat(np.array([qs[cam_id][1], qs[cam_id][2], 
                                     qs[cam_id][3], qs[cam_id][0]])).as_matrix()
    # to nerf coordinate
    c2w = np.vstack([np.hstack([R, ts[cam_id][..., None]]), bottom]) @ frame_change
    c2w[:3, 3] *= scale
    # find o2n
    o2n = np.array(data['frames'][0]['transform_matrix'])[:3, :3]@np.linalg.inv(c2w[:3, :3])
    offset = np.array(data['frames'][0]['transform_matrix'])[:3, 3]-o2n@c2w[:3, 3]

    return o2n, offset

def odom_to_nerf(parent, ts, qs, o2n, offset, scale=3, to_ngp=False):
    # not sure why the orientation is reversed with rotateAxis(180, 0)
    frame_change = rotateAxis(-90, 2) @ rotateAxis(-90, 0) #@ rotateAxis(180, 0)
    bottom = np.array([[0, 0, 0, 1.]]) 

    poses = []
    for i in range(len(ts)):
        # quaternion to rotation matrix
        R = Rotation.from_quat(np.array([qs[i][1], qs[i][2], qs[i][3], qs[i][0]])).as_matrix()
        # to nerf coordinate
        P = np.vstack([np.hstack([R, ts[i][..., None]]), bottom]) @ frame_change # 4, 4     
        P[:3, 3] *= scale
        P[:3, 3] = o2n @ P[:3, 3]
        P[:3, 3] += offset
        poses.append(P)
        # we rescale scene by 20 in ngp dataloader
        if to_ngp:
            poses[:3, 3] /= 20
    return np.array(poses)


if __name__=="__main__":
    parent = "../../../../robot/spot_data_/spot_data_0/"
    # read T and Q from odom
    ts = np.load(os.path.join(parent, "arr_2.npy"))
    qs = np.load(os.path.join(parent, "arr_3.npy"))
    
    o2n, offset = get_odom_to_nerf_matrix(parent, ts, qs, 3)
    print(o2n)
    print(offset)

    poses = odom_to_nerf(parent, ts, qs, o2n, offset, scale=3, to_ngp=False)
    print(poses.shape)

    save_as_transform_json(parent, poses)
