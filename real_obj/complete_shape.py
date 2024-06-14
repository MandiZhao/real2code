import os
import json
import argparse
from os.path import join 
from glob import glob
from natsort import natsorted
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import trimesh
import pickle 
import open3d as o3d
from PIL import Image
from einops import rearrange
from shape_complete import ShapeCompletionModel, get_handcraft_obb, remove_mesh_clusters, chamfer_distance


def load_shape_model(args):
    model = ShapeCompletionModel()
    model_fname = join(args.shape_model_dir, args.shape_run_name, f"step_{args.shape_load_step}", f"model_{args.shape_load_step}.pth")
    assert os.path.exists(model_fname), f"Model file {model_fname} does not exist" 
    model.load_state_dict(torch.load(model_fname))
    model = model.cuda()
    model.eval()
    return model

def main(args):
    model = load_shape_model(args)
    pcd_fnames = natsorted(glob(join(args.data_dir, args.folder, 'sam', '*.npz')))
    saved_meshes = []
    for fname in pcd_fnames:
        pcd = np.load(fname)['points']
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
        o3d_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.8)
        input_pcd = o3d_pcd.farthest_point_down_sample(2048)

    return 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='use dust3r results to generate SAM prompt points')
    parser.add_argument('--folder', '-f', type=str, default='1', help='Input folder')
    parser.add_argument('--data_dir', default='/home/mandi/real_rgb/', type=str, help='Data directory')
    parser.add_argument('--num_points', default=10, type=int, help='Number of points to generate')
    parser.add_argument('--num_rgbs', '-n', default=12, type=int, help='Number of RGB images to use')
    parser.add_argument('--min_valid', default=9, type=int, help='Minimum number of images a point should be valid in')
    
    parser.add_argument("--shape_model_dir", type=str, default="/local/real/mandi/shape_models")
    parser.add_argument("--shape_input_size", type=int, default=2048)
    parser.add_argument("--shape_run_name", "-shrn", type=str, default="v4_shiftidx_query12000_inp2048_qr0.25")
    parser.add_argument("--shape_load_step", type=int, default=160000)
    
    args = parser.parse_args()
    main(args)



