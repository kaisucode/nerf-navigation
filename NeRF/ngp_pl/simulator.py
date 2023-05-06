import argparse
import shutil
import os
import numpy as np
from PIL import Image
from colmap2nerf import parse_args as colmap_parse_args
from colmap2nerf import start_colmap
from train_online import train_ngp
from utils import load_transform_json
from opt import get_opts
import argparse

if __name__ == "__main__":
    # simulator

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default="../../../spot_data_0/", help="Binary flag for train versus inference.")

    args = parser.parse_args()

        # parent = os.path.join(args.dataset, "")
    if os.path.exists(os.path.join(os.path.join(args.dataset, ""), "arr_0.npy")):
        parent = os.path.join(args.dataset, "")
        data = np.load(os.path.join(os.path.join(args.dataset, ""), "arr_0.npy"))
    else:
        parent = os.path.dirname(args.dataset)
        data = np.load(args.dataset)["arr_0"]

    print(data.shape)
    
    # 
    # data = np.load(os.path.join(parent, "arr_0.npy"))

    
    # mapping every 40 steps
    counter = 0
    step = 40
    last_step = 0
    # clean image folder for colmap
    try:
        shutil.rmtree(os.path.join(parent, "images"))
    except:
        pass
    # clean checkpoint folder
    try:
        shutil.rmtree(os.path.join(".", "ckpts"))
    except:
        pass

    # main loop
    while True:
        counter += 1
        
        # start mapping
        if counter%step==0:
            # save image
            for i in range(last_step, last_step+step, 1):
                #print(i)
                im = Image.fromarray(data[i])
                path = os.path.join(parent, "images")
                if not os.path.exists(path):
                    os.makedirs(path)
                im.save(os.path.join(path, f"{i:08d}.png"))
            last_step += step
            
            # colmap argument
            colmap_args = colmap_parse_args()
            colmap_args.colmap_matcher = "exhaustive"
            colmap_args.run_colmap = True
            colmap_args.aabb_scale = 32
            colmap_args.images = os.path.join(parent, "images")
            colmap_args.text = os.path.join(parent, "text")
            colmap_args.out = os.path.join(parent, "transforms.json")
            colmap_args.overwrite = True
            # start_colmap
            start_colmap(colmap_args)
            
            # I-NGP parameters
            hparams = get_opts()
            hparams.root_dir = parent
            hparams.exp_name = "Spot"
            hparams.dataset_name = "spot_online"
            hparams.epoch = 1
            hparams.scale = 0.5
            hparams.batch_size = 20000
            # arr_0.npy
            imgs = {"imgs": data[:last_step]}
            # transforms.json
            colmap_poses = {"colmap_poses": load_transform_json(parent, 0, last_step)}
            # train_ngp
            name = int((last_step+0.5)/step)
            print(name)
            train_ngp(hparams=hparams, name=name, imgs=imgs, colmap_poses=colmap_poses)

        if counter>=len(data):
            break
