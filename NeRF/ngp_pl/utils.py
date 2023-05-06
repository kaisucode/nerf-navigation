import torch
import json
import os
import numpy as np


def umeyama(X, Y):
    """Rigid alignment of two sets of points in k-dimensional Euclidean space.
    Given two sets of points in correspondence, this function computes the
    scaling, rotation, and translation that define the transform TR that
    minimizes the sum of squared errors between TR(X) and its corresponding
    points in Y.  This routine takes O(n k^3)-time.
    Parameters:
        X:
            A ``k x n`` matrix whose columns are points.
        Y:
            A ``k x n`` matrix whose columns are points that correspond to the
            points in X
    Returns:
        c,R,t:
            The  scaling, rotation matrix, and translation vector defining the
            linear map TR as ::
                       TR(x) = c * R * x + t
             such that the average norm of ``TR(X(:, i) - Y(:, i))`` is
             minimized.
    Copyright: Carlo Nicolini, 2013
    Code adapted from the Mark Paskin Matlab version from
    http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m
    See paper by Umeyama (1991)
    """

    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)

    Xc = X - np.tile(mx, (n, 1)).T
    Yc = Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc * Xc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U, D, V = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = V.T.copy()

    r = np.linalg.matrix_rank(Sxy)
    S = np.eye(m)

    if r < (m - 1):
        raise ValueError('not enough independent measurements')

    if (np.linalg.det(Sxy) < 0):
        S[-1, -1] = -1
    elif (r == m - 1):
        if (np.linalg.det(U) * np.linalg.det(V) < 0):
            S[-1, -1] = -1

    R = np.dot(np.dot(U, S), V.T)
    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return c, R, t

def frame_path(frame):
  return frame['file_path']

def load_transform_json(parent, start = 0, end= -1):
    # transforms.json
    with open(os.path.join(parent, "transforms.json"), 'r') as f:
        poses = json.load(f)
    poses['frames'].sort(key=frame_path)


    # print(len(poses["frames"]), start, end)
    if end < 0:
        end = len(poses["frames"])
    poses['frames'] = poses['frames'][start:end]
    # print(len(poses["frames"]))
    return poses


def align_odom_to_colmap(poses_colmap, poses_odom):


    origin = np.array([0.0, 0.0, 0.0, 1.0])[..., None] # 4, 1
    # mesh_list = []
    points_colmap = []
    points_odom = []
    for p_idx in range(poses_colmap.shape[0]):
        cam_pose_colmap = poses_colmap[p_idx] @ origin
        cam_pose_odom = poses_odom[p_idx] @ origin
        
        points_colmap.append(cam_pose_colmap[..., 0])
        points_odom.append(cam_pose_odom[..., 0])
        # print(cam_pose_colmap.shape)

    points_colmap = np.vstack(points_colmap)
    points_odom = np.vstack(points_odom)
    # print(points_colmap.shape)


    c, R, t = umeyama(points_odom.T, points_colmap.T)
    # print(c, R, t)
    # poses_colmap[:, :, ]
    
    # points_c_odom = c * R @ points_odom.T + t[..., None]

    # print(np.mean(np.abs(points_colmap - points_c_odom.T)))

    return c, R, t
        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1.0 / 20.0)
        # mesh.rotate(poses[p_idx, :3, :3])
        # mesh.translate(cam_pose[:3, -1])
        # mesh_list.append(mesh)


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path: return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def slim_ckpt(ckpt_path, save_poses=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # pop unused parameters
    keys_to_pop = ['directions', 'model.density_grid', 'model.grid_coords']
    if not save_poses: keys_to_pop += ['poses']
    for k in ckpt['state_dict']:
        if k.startswith('val_lpips'):
            keys_to_pop += [k]
    for k in keys_to_pop:
        ckpt['state_dict'].pop(k, None)
    return ckpt['state_dict']
