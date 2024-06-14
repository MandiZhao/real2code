"""
Manually select and group the multi-view seg. masks into segmented pcd
Note the RGB image is 7.5 times bigger than depth

Input: obj_type/obj_folder/0_3d/*.pkl, contains masks
Output: obj_type/obj_folder/0_3d/filled_pcd_x.npz, which can then be used to run shape completion models 
"""

import open3d as o3d
import os
from os.path import join 
import numpy as np
import argparse
from glob import glob
from natsort import natsorted
import pickle
from image_seg import get_filled_img
from image_seg.sam_to_pcd import load_blender_data
from image_seg.test_sam import get_one_tsdf
import cv2
import trimesh
from PIL import Image

SELCT_FILES = {
    "1": dict(
        rgb_idxs=[32], # really bad
    ),
    "2": dict(
    rgb_idxs=[7, 19, 57, 127, 146, 158, 179, 212, 337 ], 
    pcd_idxs={
        "0": [(7, 2), (19,2), (57, 2), (127, 1), (146, 2), (158, 3), (179, 4), (212,3)], # bottom drawer
        "1": [(7,1), (19,1), (57,1), (127,2), (146, 1), (158, 4), (179, 3), (212,2)], #middle drawer
        "2": [(179, 1)], # (127,4), (158,1) (19, 3)
        "3": [(127,3), (146, 3), (158, 2), (179, 2), (212, 1)], # body
    }
    ),
    "3": dict(
        rgb_idxs=[19, 25, 57, 70, 103, 116, 134, 146, 172, 222, 285],  
        pcd_idxs={
            "0": [(19, 2), (25,1), (57,1), (70, 1), (103, 2), (116, 1), (134, 2), (146, 2), (172, 1), (285, 1)], # bottom biggest drawer
            "1": [(70,2), (103, 1), (116, 2),  (146, 1), (134, 1), (172,2), (285, 2)], # middle drawer
            "2": [(134, 4), ], # top drawer
            "3": [(19, 1), (25, 2), (57, 2), (70, 3), (134, 3), (172, 3), (144,3) ] # body
        
        }
    ),
    "4": dict(
        rgb_idxs=[38, 44, 50, 56, 62, 74, 86, 99, 137, 163, 177],
        pcd_idxs={
            '0': [(38,1), (44,1), (50,1), (56,1), (62,1), (74,1), (86,1), (99,1), (163,1),],
            '1': [(38,2), (44,2), (50,2), (56,2), (62,2), (74,2), (86,2), (99,2), (137,2), (163,2), (177,2)],
            }
        ),
    "5": dict(
        rgb_idxs=[18, 24, 55, 67, 73],
        pcd_idxs={
            '0': [(18, 1), (24, 2), (55, 1), (67, 1), (73, 1)],
            '1': [(18, 2), (24, 1), (55, 2), (67, 2), (73, 2)],
        }
    ),
    "6": dict(
        rgb_idxs=[13, 32, 44, 50, 56, 69, 88, 94],
        pcd_idxs={
            "0": [(13, 1), (32, 1), (44, 1), (50, 1), (56, 1), (69, 2), (88, 1), (94, 1)],
            "1": [(13, 2), (32, 2), (44, 2), (50, 2), (56, 2), (69, 1), (88, 2), (94, 2)]
        }
    ), 
    "13": dict(
        rgb_idxs=[6, 19, 61, 88, 239, 245, 282], 
        pcd_idxs={
            "0": [(6, 1), (19, 1), (61, 1), (88,1), (239, 2), (245, 1), (282, 2)], # drawer
            "1": [(6, 2), (19, 2), (61, 2), (239, 1), (245, 2), (282, 1)] # body
        }
    ),
    "14": dict(
        rgb_idxs=[38, 57, 77, 102, 135,],
        pcd_idxs={
            "0": [(38, 1), (57, 1), (77,1), (102, 1), (135, 1)],  # bot drawer
            "1": [(38, 2), (77, 2) ],  # mid drawer
            "2": [(38, 3), (77,3),],  # body
        }
    ),
    "15": dict(
        rgb_idxs=[6, 30, 36, 42, 74, 86, 124, 163],
        pcd_idxs={
            "0": [(6, 1), (30, 1), (36, 1), (42, 1), (74, 1), (124, 1), (163, 1)], # lower drawer
            "1": [(6, 2),  ], # upper drawer
            "2": [(6, 3), (30, 2), (42, 2) ] # body
        }
    ),
}

def merge_pcd_manually(pkl_fnames, obj_dir, data_dir):
    pcd_idxs = SELCT_FILES[args.obj_folder]['pcd_idxs']
    for pcd_idx, toselect in pcd_idxs.items():
        out_fname = join(obj_dir, f'filled_pcd_{pcd_idx}.npz')
        merged_pcd = []
        for fname in pkl_fnames:
            idx = int(fname.split('/')[-1].split('.')[0])
            pairs= [x for x in toselect if x[0] == idx]
            if len(pairs) == 0:
                continue 
            seg_id = pairs[0][1]
            pcd_fname = fname.replace('.pkl', f'filled_pcd_{seg_id}.npz')
            assert os.path.exists(pcd_fname), f"{pcd_fname} not found"
            points = np.load(pcd_fname)['points']
            merged_pcd.append(points)
        merged_pcd = np.concatenate(merged_pcd, axis=0)
        # use o3d to remove outliers
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_pcd)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.7)
        merged_pcd = np.array(pcd.points)
        np.savez(out_fname, points=merged_pcd)
        print(f"Saved {out_fname}")
    return

            
def run(args):
    obj_dir = glob(join(args.input_dir, args.obj_type, args.obj_folder, '0_3d'))
 
    assert len(obj_dir) == 1, f"{args.obj_type} {args.obj_folder} not found: {obj_dir}"
   
    obj_dir = obj_dir[0]
    obj_type = obj_dir.split('/')[-3]
    obj_folder = obj_dir.split('/')[-2]
    data_dir = join(args.dataset_dir, obj_type, obj_folder, 'loop_0')
    pkl_fnames = natsorted(glob(join(obj_dir, '*.pkl')))
    if len(pkl_fnames) == 0:
        print(f"No pkl files found in {obj_dir}")
        return
    toselect = SELCT_FILES[args.obj_folder]['rgb_idxs']
    filtered_pkl_fnames = []
    for fname in pkl_fnames:
        idx = int(fname.split('/')[-1].split('.')[0])
        if idx in toselect:
            filtered_pkl_fnames.append(fname)
    print(f"Selected {len(filtered_pkl_fnames)} pkl files")
    if args.merge_pcd:
        merge_pcd_manually(filtered_pkl_fnames, obj_dir, data_dir)
        return 
    # remove the existing pcds
    if args.overwrite:
        for oldfname in glob(join(obj_dir, f'*filled_pcd_*.npz')) + glob(join(obj_dir, f'*filled_mesh_*.obj')):
            os.remove(oldfname)
    all_tsdfs = dict()
    for fname in filtered_pkl_fnames:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        masks = data['nms_filtered']
        pkl_id = fname.split('/')[-1].split('.')[0]
        
        h5_fname = join(data_dir, f"{fname.split('/')[-1].replace('.pkl', '.hdf5')}")
        assert os.path.exists(h5_fname), f"{h5_fname} not found"
        data = load_blender_data(h5_fname)
       
        filled_img, used_masks, used_idxs = get_filled_img(
            masks, threshold=0.999, min_new_pixels=80000)
        Image.fromarray(filled_img).save(fname.replace('.pkl', '_filled.png'))
       
        rgb = data['rgb'].copy()
        vertical_rgb_shape = rgb.shape[:2] # (1920, 1440) or (1440, 1920)
        need_rotate_back = data['need_rotate_back']
        if need_rotate_back: 
            rgb = cv2.rotate(rgb,  cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        depth = data['depth']
        camera_intrinsics = np.array(data['cam_intrinsics'])
        camera_pose = np.array(data['cam_pose']) 
        extrinsics = np.linalg.inv(camera_pose)
        extrinsics[2, :] *= -1
        extrinsics[1, :] *= -1
        # requires scaling the SAM outputs to smaller (256, 192) size
        depth_shape = depth.shape
        # resize rgb to be same as depth
        rgb = cv2.resize(
            rgb, (depth_shape[1], depth_shape[0])
        )
        
        rot_resized_masks = [] # assert depth_shape == (h, w), f"Got depth shape {depth_shape} but expected {(h, w)}"
        for mask in used_masks:
            if mask.shape == (1920, 1920):
                # the mask was padded with 240 on each side, need to crop it to fit vertical_rgb_shape 
                if vertical_rgb_shape[0] == 1440:
                    mask = mask[240:1680, :]
                else:
                    mask = mask[:, 240:1680] 

            resized_mask = np.array(mask, dtype=np.float32) 
            if need_rotate_back:
                resized_mask = cv2.rotate(resized_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)  
            resized_mask = cv2.resize(
                resized_mask, (depth_shape[1], depth_shape[0])
            )
            rot_resized_masks.append(resized_mask)

        # get the tsdf
        volumes = [] 
        for m, mask in enumerate(rot_resized_masks):
            if m == 0:
                # skipping bg mask
                continue
            masked_rgb = rgb.copy()
            masked_rgb[mask == 0] = 0
            masked_depth = depth.copy()
            masked_depth[mask == 0] = -1
            tsdf, o3d_rgbd, o3d_intrinsics, o3d_extrinsics = get_one_tsdf(
                [masked_rgb], [masked_depth], cam_intrinsics=[camera_intrinsics], extrinsics=[extrinsics],return_info=True,
                )
            volumes.append(tsdf)
            pcd = tsdf.extract_point_cloud()
            pcd_fname = fname.replace('.pkl', f'filled_pcd_{m}.npz')
            np.savez(pcd_fname, points=np.array(pcd.points)) 
            # mesh = tsdf.extract_triangle_mesh()
            # mesh_fname = fname.replace('.pkl', f'filled_mesh_{m}.obj')
            # tri_mesh = trimesh.Trimesh(vertices=np.array(mesh.vertices), faces=np.array(mesh.triangles))
            # tri_mesh.export(mesh_fname)

            tsdf_name = f"{pkl_id}_{m}"
            tsdf_info = dict(
                tsdf=tsdf,
                rgbd=o3d_rgbd,
                intrinsics=o3d_intrinsics,
                extrinsics=o3d_extrinsics,
            )
            all_tsdfs[tsdf_name] = tsdf_info
        
        # use pcd_idx to merge all the tsdfs
        
    pcd_idxs = SELCT_FILES[args.obj_folder]['pcd_idxs'] 
    for pcd_idx, toselect in pcd_idxs.items():
        grouped_tsdf = []
        for idx, seg_id in toselect: 
            tsdf_name = f"{idx}_{seg_id}"
            print(tsdf_name)
            if tsdf_name in all_tsdfs:
                grouped_tsdf.append(all_tsdfs[tsdf_name]) 
     
        print(f"Grouped {len(grouped_tsdf)} tsdfs")
        if len(grouped_tsdf) == 0:
            continue
        vol = grouped_tsdf[0]['tsdf']
        for tsdf_info in grouped_tsdf[1:]:
            vol.integrate(
                tsdf_info['rgbd'],
                tsdf_info['intrinsics'],
                tsdf_info['extrinsics'],
            )
        mesh = vol.extract_triangle_mesh()
        # o3d smooth 
        mesh = mesh.filter_smooth_laplacian(50, 0.01)
        mesh_fname = join(obj_dir, f"filled_mesh_{pcd_idx}.obj")
        tri_mesh = trimesh.Trimesh(vertices=np.array(mesh.vertices), faces=np.array(mesh.triangles))
        tri_mesh.export(mesh_fname)
        print(f"Saved {mesh_fname}")
        pcd = vol.extract_point_cloud()
        # remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.7)
        pcd_fname = join(obj_dir, f"filled_pcd_{pcd_idx}.npz")
        np.savez(pcd_fname, points=np.array(pcd.points))


    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="/home/mandi/eval_real2code/v4_pointsTrue_lr0.0003_bs24_ac12_02-21_11-45/ckpt_epoch_11/", type=str, help='input folder containing sam-2d data')
    parser.add_argument('--dataset_dir', default="/local/real/mandi/scanner_dataset/test", type=str, help='input folder containing sam-2d data')
    parser.add_argument('--output_dir', type=str, default="/local/real/mandi/scanner_dataset", help='output folder for h5 files')
    parser.add_argument('--obj_type', type=str, default="*", help='object type')
    parser.add_argument('--obj_folder', type=str, default="4", help='object folder')
    parser.add_argument('--overwrite', '-o', action='store_true', help='overwrite existing files')
    parser.add_argument('--merge_pcd', '-mp', action='store_true', help='merge the pcds')
    args = parser.parse_args()
    run(args)
