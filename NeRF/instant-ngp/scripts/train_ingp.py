# This file is modified from I-NGP run.py
import os
import commentjson as json

import numpy as np

import shutil
import time
import sys
sys.path.append("../build")
sys.path.append(".")

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp
import cv2

def frame_path(frame):
  return frame['file_path']

def run_npg(name, start, end, parent, scene_train_json, network, aabb_scale, gui, total_n_steps):
    
    # load transform_*.json file
    f = open(scene_train_json)
    params = json.load(f)
    params['frames'].sort(key=frame_path)
    params['frames'] = params['frames'][start:end]
    f.close()

    # python binding
    testbed = ngp.Testbed()
    # setup root dir
    testbed.root_dir = parent
    # initialize a network
    testbed.reload_network_from_file(network)
    # initialize a dataset
    testbed.create_empty_nerf_dataset(n_images=len(params['frames']), aabb_scale=aabb_scale)
    # setup dataset
    for idx, param in enumerate(params['frames']):
        img_path = os.path.join(parent, param['file_path'])
        #print(img_path, idx)
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        img = img.astype(np.float32)
        img /= 255.0
        # set_image should only accept linear color
        img = srgb_to_linear(img)
        # premultiply
        img[..., :3] *= img[..., 3:4]
        height, width = img.shape[:2]
        assert (height==params['h'])
        assert (width==params['w'])
        depth_img = np.zeros((img.shape[0], img.shape[1]))
        testbed.nerf.training.set_image(idx, img, depth_img)
        testbed.nerf.training.set_camera_extrinsics(idx, param['transform_matrix'][:3], convert_to_ngp=True)
        testbed.nerf.training.set_camera_intrinsics(
            idx,
            fx=params["fl_x"], fy=params["fl_y"],
            cx=params["cx"], cy=params["cy"],
            k1=params["k1"] if "k1" in params.keys() else 0, 
            k2=params["k2"] if "k2" in params.keys() else 0,
            p1=params["p1"] if "p1" in params.keys() else 0, 
            p2=params["p2"] if "p2" in params.keys() else 0
        )

    if gui:
        sw = width 
        sh = height 
        testbed.init_window(sw, sh)
        testbed.nerf.visualize_cameras = True

    print(f"Optimizing Camera Parameters !!")
    print("--------------------------------------------")
    testbed.nerf.training.optimize_extrinsics = True
    testbed.nerf.training.optimize_distortion = True

    save_path = os.path.join(parent, 'I-NGP', 'train', f'{name:08d}')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Training
    testbed.nerf.training.n_images_for_training = len(params['frames'])
    testbed.shall_train = True
    training_step = 0
    tqdm_last_update = 0
    # near plane
    testbed.nerf.training.near_distance = 0.1
    # Start training
    if total_n_steps > 0:
        with tqdm(desc="Training", total=total_n_steps, unit="step") as t:
            while testbed.frame():
                # What will happen when training is done?
                if testbed.training_step >= total_n_steps:
                    break

                # Update progress bar
                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - training_step)
                    t.set_postfix(loss=testbed.loss)
                    training_step = testbed.training_step
                    tqdm_last_update = now  

    testbed.save_snapshot(os.path.join(save_path, "model.ingp"), False)
 

if __name__ == "__main__":
    parent = "../../../../../robot/spot_data_/spot_data_0/"
    scene_train_json = "../../../../../robot/spot_data_/spot_data_01/transforms_odom.json"
    network = "../configs/nerf/base.json"
    aabb_scale = 16
    gui = True
    total_n_steps = 2000
    # first frame
    name = 0
    start = 0
    end = 40
    run_npg(name, start, end, parent, scene_train_json, network, aabb_scale, gui, total_n_steps)
    # more frames
    name = 1
    start = 0
    end = 80
    run_npg(name, start, end, parent, scene_train_json, network, aabb_scale, gui, total_n_steps)
