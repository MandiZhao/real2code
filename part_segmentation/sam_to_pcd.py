import os
from os.path import join
from glob import glob
from natsort import natsorted
import wandb
import json
from tqdm import tqdm
import cv2
import numpy as np   
import h5py
from collections import defaultdict 
import open3d  
from matplotlib import pyplot as plt
from functools import partial
import pickle
import scipy 
from scipy import spatial, interpolate
from scipy.ndimage import zoom
import argparse   
import shutil
import itertools
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree as KDTree
import open3d as o3d
import torch
from kaolin.ops.conversions import pointclouds_to_voxelgrids
"""
Note that the lookup obj types and folders must be a specific value, not * like other scripts

TODO: look at why loop_id 1/2 are not working


e.g. 
RUN=sam_v2_pointsTrue_lr0.001_bs8_ac16_01-19_23-56;  STEP=30000
python sam_to_pcd.py --run_name $RUN --load_step ${STEP} --load_obj_type Box --load_loop_id 0
"""
WIDTH, HEIGHT = 512, 512 # NEW for v2 data! 640, 480

def get_voxel_iou(points1, points2, resolution=96):
    pcd1 = torch.tensor(points1).cuda()
    pcd2 = torch.tensor(points2).cuda()
    origin = np.min(np.concatenate([points1, points2], axis=0), axis=0) 
    origin = np.array([-2, -2, -2])
    origin = torch.tensor(origin).cuda()[None]
    scale = np.max(np.concatenate([points1, points2], axis=0), axis=0) 
    scale = np.array([1, 1, 1])
    scale = (torch.tensor(scale).cuda() - origin)[None]

    grid1 = pointclouds_to_voxelgrids(pcd1[None], origin=origin, resolution=resolution, scale=scale)
    grid2 = pointclouds_to_voxelgrids(pcd2[None], origin=origin, resolution=resolution, scale=scale) # shape 1, 84, 84, 84

    union = (grid1 + grid2).clamp(0, 1)
    intersection = (grid1 * grid2).clamp(0, 1)
    return intersection.sum(), union.sum(), grid1[0], grid2[0]
    
def compute_chamfer_distance(points1, points2, dist_type="center", voxel=False, voxel_size=0.01, same_size=False):
    """ compute dist w/o considering masks """
    if len(points1) == 0 and len(points2) == 0:
        return 0.0
    if len(points1) == 0 or len(points2) == 0:
        return 1000
    # compute iou
    def voxelgrid_to_set(voxel_grid):
        occupied_voxels = set()
        for voxel in voxel_grid.get_voxels():
            # Convert voxel grid coordinates to a tuple and add to the set
            voxel_coords = (voxel.grid_index[0], voxel.grid_index[1], voxel.grid_index[2])
            occupied_voxels.add(voxel_coords)
        return occupied_voxels
    if dist_type == "chamfer": 
        if voxel:
        # create voxel grid then compare distance
            o3d_pcd1 = open3d.geometry.PointCloud()
            o3d_pcd1.points = open3d.utility.Vector3dVector(points1)
            o3d_pcd1 = o3d_pcd1.voxel_down_sample(voxel_size=voxel_size) 
            # o3d_pcd1.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.95)
            
            o3d_pcd2 = open3d.geometry.PointCloud()
            o3d_pcd2.points = open3d.utility.Vector3dVector(points2)
            o3d_pcd2 = o3d_pcd2.voxel_down_sample(voxel_size=voxel_size)  
            # o3d_pcd2.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.95)
            
            intersection, union, grid1, grid2 = get_voxel_iou(
                np.array(o3d_pcd1.points), np.array(o3d_pcd2.points), resolution=88
            )
            if abs(1 - intersection / grid1.sum()) <= 0.15 or abs(1 - intersection / grid2.sum()) <= 0.15:
                print("One is completely inside the other")
                return 0.0 
            # print("Intersection", intersection, grid1.sum(), grid2.sum()) 
            iou = intersection / union
            if iou > 0.8:
                print(f"High IoU, {iou} > 0.8")
                return 0.0
            if same_size:
                # resize to the same size
                min_size = min(len(o3d_pcd1.points), len(o3d_pcd2.points))
                o3d_pcd1 = o3d_pcd1.farthest_point_down_sample(min_size)
                o3d_pcd2 = o3d_pcd2.farthest_point_down_sample(min_size)
             
    
        tree1 = KDTree(points1)
        one_distances, one_vertex_ids = tree1.query(points2)
        dist1 = np.mean(np.square(one_distances))

        # other direction
        tree2 = KDTree(points2)
        two_distances, two_vertex_ids = tree2.query(points1) 
        dist2 = np.mean(np.square(two_distances))
        # weight by distance shape
        dist1_weight = two_distances.shape[0] / (one_distances.shape[0] + two_distances.shape[0])
        dist2_weight = one_distances.shape[0] / (one_distances.shape[0] + two_distances.shape[0])
        dist1 *= dist1_weight
        dist2 *= dist2_weight
        chamfer_dist = dist1 + dist2 
        return chamfer_dist
    
    elif dist_type == "iou":
        # compute 3D IoU
        if voxel:
            # o3d_pcd1 and ocd_pcd2 are already voxelized
            # convert to voxel grid
            intersection, union, grid1, grid2 = get_voxel_iou(points1, points2)
            # check if one points is completely inside the other
            if intersection == grid1.sum() or intersection == grid2.sum():
                print("One is completely inside the other")
                return 0.0
            # Calculate IoU
            return 1 - iou
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
       
def load_blender_data(fname, mask_thres=10): 
    """ convert the keys from hdf5 files """
    lookup_keys = ['cam_id', 'cam_pose', 'cam_fov', 'cam_intrinsics', 'binary_masks', 'colors', 'depth']
    scanner_data = ( 'scanner' in fname)
    if scanner_data:
        print(f"Loading scanner data from {fname}, uses special processing")
        lookup_keys.append('need_rotate_back')
    with h5py.File(fname, "r") as f: 
        data = dict()
        for key in lookup_keys:
            data_key = key 
            if key == 'colors':
                data_key = 'rgb' 
 
            data[data_key] = np.array(f[key][:]) 
            if key == 'cam_fov': 
                data['fov'] = data['cam_fov'][0] # assume square
            if key == 'cam_pose': 
                data['pos'] = data['cam_pose'][:3, -1]
                data['rot_mat'] = data['cam_pose'][:3,:3]
            if key == 'binary_masks':
                data['segment_body'] = {i: mask for i, mask in enumerate(data['binary_masks'])}
            data['fov_metric'] = 'rad'
    mask = data['binary_masks'][0] # background mask, use to mask depth 
    if not scanner_data:
        depth = data['depth'] 
        depth[mask] = mask_thres
        depth[depth > mask_thres] = mask_thres # object could still have holes
        data['depth'] = depth
    data['is_scanner'] = scanner_data
    return data

def find_neighbor_mask(seg_masks, kd_tree, query_pt, init_r=0.02, ignore_ids=[], min_count=1):
    """ take vote from nearby mask ids """
    nearby_mask_ids = defaultdict(int) 
    neighbor_count = 0
    while neighbor_count < min_count:
        nearby_pts = kd_tree.query_ball_point(query_pt, r=init_r)
        init_r += 0.005
        for pt in nearby_pts: 
            for _id, mask in seg_masks.items():
                if (not _id in ignore_ids) and mask[pt] > 0:
                    nearby_mask_ids[_id] += 1
                    neighbor_count += 1
    # take vote
    pairs = sorted(nearby_mask_ids.items(), key=lambda x: x[1], reverse=True)
    seg_id = pairs[0][0]
    return seg_id

def post_process_merged(merged_xyzs, merged_rgbs, merged_segs, ignore_bg=True, smooth_r=0.01, stat_remove=False): 
    # for all the points in xyz_pts, find if it was assigned to a mask, if not, find the closest mask and assign it to that mask
    tree = spatial.cKDTree(merged_xyzs)
    if ignore_bg:
        # find the background mask
        mask_len = [(k, sum(mask>0)) for k, mask in merged_segs.items()]
        mask_len = sorted(mask_len, key=lambda x: x[1])
        bg_id = -1
        if mask_len[0][0] < 300:
            bg_id = mask_len[0][0]
            # assume no pixel on the object should be assigned to bg
            bg_mask = merged_segs[bg_id] > 0
            # re-assign these pixels to the closest mask
            pos_idxs = np.where(bg_mask)[0]
            pos_pts = merged_xyzs[pos_idxs]

            for idx, xyz_pt in zip(pos_idxs, pos_pts):
                seg_id = find_neighbor_mask(merged_segs, tree, xyz_pt, init_r=0.005, ignore_ids=[bg_id], min_count=5)
                for other_id in merged_segs.keys():
                    merged_segs[other_id][idx] = 0 
                merged_segs[seg_id][idx] = 1
    
    # print('begin filling empty pixels and de-duplicating pixels')
    # for i, xyz_pt in enumerate(merged_xyzs):
    #     # has_mask = [(_id, mask[i]) for _id, mask in merged_segs.items()]
    #     # if sum([mask for _id, mask in has_mask]) > 1 or sum([mask for _id, mask in has_mask]) == 0:
    #     seg_id = find_neighbor_mask(merged_segs, tree, xyz_pt, init_r=smooth_r, ignore_ids=[bg_id], min_count=5)
    #     for _id, mask in merged_segs.items():
    #         if _id != seg_id:
    #             merged_segs[_id][i] = 0  
    # check mask is cont. valued
     
    print('begin cleaning outlier pixels')
    # process start from large masks to small masks
    mask_len = [(k, sum(mask), mask) for k, mask in merged_segs.items()]
    for _id, _, mask in sorted(mask_len, key=lambda x: x[1], reverse=True):
        pos_idxs = np.where(mask)[0]
        pos_pts = merged_xyzs[pos_idxs]
        for idx, xyz_pt in zip(pos_idxs, pos_pts):
            # dist_to_centers = [(_id, np.linalg.norm(xyz_pt - center)) for _id, center in mask_centers.items()]
            # dist_to_centers = sorted(dist_to_centers, key=lambda x: x[1])
            # re-assign only if this pixel is also different from many its neighbors
            neighbors_mask_id = find_neighbor_mask(merged_segs, tree, xyz_pt, init_r=smooth_r, ignore_ids=[bg_id], min_count=5)
            if neighbors_mask_id != _id:
                new_id = neighbors_mask_id# dist_to_centers[0][0] 
                for other_id in merged_segs.keys():
                    merged_segs[other_id][idx] = 0
                merged_segs[new_id][idx] = 1

    if stat_remove:
        print('try using statistical outlier removal')
        new_xyzs, new_rgbs, new_segs = [], [], dict()
        for _id, mask in merged_segs.items():
            seg_pcd = o3d.geometry.PointCloud() 
            seg_pcd.points = open3d.utility.Vector3dVector(merged_xyzs[mask])
            seg_pcd.colors = open3d.utility.Vector3dVector(merged_rgbs[mask])
            seg_pcd, idxs = seg_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.95) 
            new_xyzs.append(np.array(seg_pcd.points))
            new_rgbs.append(np.array(seg_pcd.colors)) 
            print('before stat remove', sum(mask))
            print('after stat remove', np.array(seg_pcd.points).shape)

        for i, _id in enumerate(merged_segs.keys()):
            new_segs[_id] = np.zeros((np.concatenate(new_xyzs).shape[0], ))
            masked = new_xyzs[i]
            prev_idx = 0 if i == 0 else new_xyzs[i-1].shape[0]
            new_segs[_id][prev_idx:prev_idx+masked.shape[0]] = 1
            # cast to int 
            new_segs[_id] = (new_segs[_id] > 0)

        print(merged_xyzs.shape, merged_rgbs.shape, merged_segs[0].shape)
        merged_xyzs = np.concatenate(new_xyzs)
        merged_rgbs = np.concatenate(new_rgbs)
        merged_segs = new_segs  
        print('after removal', merged_xyzs.shape, merged_rgbs.shape, merged_segs[0].shape)

    return merged_xyzs, merged_rgbs, merged_segs

def merge_pcds_iteratively(pcds, dist_fn, dist_thres=0.2, voxel=False, show_interm=False):
    """ DEPRECATED, use tsdf iterate through pcds and merge 3D segmentations based on distance"""
    assert len(pcds) > 0, "No pcds to merge"
    num_pcds = len(pcds)
    mid_idx = max(0, int(num_pcds/2) - 1)
    merged_pcds = [pcds[mid_idx]]
    for i, pcd in enumerate(pcds[:mid_idx] + pcds[mid_idx+1:]):
        _dists = []
        points = np.asarray(pcd.points)
        for j, ref_pcd in enumerate(merged_pcds):
            points_ref = np.asarray(ref_pcd.points)
            cd = dist_fn(points, points_ref)
            _dists.append((cd, j))
        _dists = sorted(_dists, key=lambda x: x[0]) 
        if _dists[0][0] < dist_thres:
            # print(f"pcd {i}: merging into pcd {j}")
            merge_idx = _dists[0][1]
            merged_pcds[merge_idx] += pcd
        else:
            print(f"pcd {i}: adding new pcd")
            merged_pcds.append(pcd)
    return merged_pcds

def merge_tsdfs(tsdf_ls, dist_fn, score_sort=True, dist_thres = 0.2, voxel=False, show_interm=False):
    assert len(tsdf_ls) > 0, "No pcds to merge"
    if score_sort:
        print("Sorting tsdfs by their scores before iteratively merging")
        tsdf_ls = sorted(tsdf_ls, key=lambda x: x['score'], reverse=True) 
        print("Sorted tsdfs by their scores: ", [(x['cam_name'], x['score']) for x in tsdf_ls])
        merged_tsdfs = [tsdf_ls[0]]
    else:
        print("No sorting, using the middle tsdf as the reference")
        mid_idx = max(0, int(len(tsdf_ls)/2) - 1)
        merged_tsdfs = [tsdf_ls[mid_idx]]
        tsdf_ls = tsdf_ls[:mid_idx][::-1] + tsdf_ls[mid_idx+1:][::-1]
        # tsdf_ls = sorted(tsdf_ls[:mid_idx][::-1], key=lambda x: x['score'], reverse=True) + sorted(tsdf_ls[mid_idx+1:], key=lambda x: x['score'], reverse=True)
    for i, tsdf_info in enumerate(tsdf_ls):
        tsdf = tsdf_info['tsdf']  
        pcd = tsdf_info['pcd']
        if len(pcd.points) < 100:
            continue
        _dists = [] 
        points = np.asarray(pcd.points)
        for j, ref_tsdf in enumerate(merged_tsdfs):
            points_ref = np.asarray(ref_tsdf['pcd'].points)
            cd = dist_fn(points, points_ref)
            _dists.append((cd, j))
        _dists = sorted(_dists, key=lambda x: x[0]) 
        if _dists[0][0] < dist_thres:
            print(f"pcd {i}: merging into pcd {j}: {_dists[0][0]}")
            merge_idx = _dists[0][1]
            new_rgbd = tsdf_info['rgbd']
            # new_mask = tsdf_info['mask']
            # old_mask = merged_tsdfs[merge_idx]['mask']
            # # only keep the new additional pixels 
            # new_mask = (new_mask > 0) & (old_mask == 0)
            # new_rgbd.depth[new_mask] = -1
            merged_tsdfs[merge_idx]['tsdf'].integrate(
                    new_rgbd,
                    tsdf_info['intrinsics'],
                    tsdf_info['extrinsics'],
                    ) 
            new_pcd = merged_tsdfs[merge_idx]['tsdf'].extract_point_cloud()  
            merged_tsdfs[merge_idx]['pcd'] = new_pcd
            # merged_tsdfs[merge_idx]['mask'] = (old_mask > 0) or (new_mask > 0)
        else:
            print(f"TSDF {i}: adding new tsdf: {_dists[0][0]}")
            merged_tsdfs.append(tsdf_info)
        # print("Min dist", _dists[0][0])
    return merged_tsdfs

def get_tsdfs(outputs, vlength=0.15, sdf_trunc=0.3, depth_trunc=10, return_imgs=False): 
    all_volumes = dict()
    for cam_name, data in outputs.items(): 
        binary_masks = data['used_masks']
        weighted_scores = data['weighted_scores']
        assert len(binary_masks) == len(weighted_scores), "Mismatched length"
        volumes = [
            o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=vlength, 
                sdf_trunc=sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                )
            for _ in binary_masks
        ]
        volume_infos = [dict() for _ in binary_masks]
        rgb = data['rgb']
        depth = data['depth']
        camera_intrinsics = np.array(data['cam_intrinsics'])
        camera_pose = np.array(data['cam_pose'])  
            
        intrinsics = o3d.camera.PinholeCameraIntrinsic() 
        intrinsics.set_intrinsics(
            width=WIDTH, height=HEIGHT, fx=camera_intrinsics[0, 0], fy=camera_intrinsics[1, 1], cx=(WIDTH / 2), cy=(HEIGHT / 2))
        extrinsics = np.linalg.inv(camera_pose)
        extrinsics[2,:] *= -1
        extrinsics[1,:] *= -1 
        depth[depth>depth_trunc] = -1 # depth_trunc
        for idx, mask in enumerate(binary_masks): 
            rgb_copy = rgb.copy()
            # make the background black 
            depth_copy = depth.copy()
            rgb_copy[mask == 0] = 0
            depth_copy[mask == 0] = -1    
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb_copy), 
                    o3d.geometry.Image(depth_copy),
                    depth_trunc=depth_trunc, 
                    depth_scale=1,
                    convert_rgb_to_intensity=False,
                    )
            # print(f"Integrating {h5_file} to volume {idx}")
            volumes[idx].integrate(
                    rgbd,
                    intrinsics,
                    extrinsics,
                    )
            volume_infos[idx] = dict(
                    tsdf=volumes[idx],
                    rgbd=rgbd,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    pcd=volumes[idx].extract_point_cloud(),
                    score=np.mean(weighted_scores), # weighted_scores[idx],
                    mask=mask,
                    cam_name=cam_name,
                    mask_idx=idx,
                    )
        if return_imgs:
            all_volumes[cam_name] = volume_infos
        else:
            all_volumes[cam_name] = volumes

    return all_volumes

def mesh_distance(mesh1, mesh2, sample_points=10000):
    # sample points on each and compute chamfer distance
    pcd1 = mesh1.sample_points_uniformly(number_of_points=sample_points)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=sample_points)
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    # dist1 = np.linalg.norm(points1[:, None] - points2[None], axis=-1).min(axis=-1).mean()
    # dist2 = np.linalg.norm(points1[:, None] - points2[None], axis=-1).min(axis=-1).mean()
    # chamfer_dist = (dist1 + dist2) / 2.0
    # return chamfer_dist
    # one direction
    points1_kd_tree = KDTree(points1)
    one_distances, one_vertex_ids = points1_kd_tree.query(points2)
    dist1 = np.mean(np.square(one_distances))
    # use L1:
    # dist1 = np.mean(np.abs(one_distances))

    # other direction
    points2_kd_tree = KDTree(points2)
    two_distances, two_vertex_ids = points2_kd_tree.query(points1)
    dist2 = np.mean(np.square(two_distances))
    # use L1:
    # dist2 = np.mean(np.abs(two_distances))
    return (dist1 + dist2 )/2

def get_mesh_distances(mesh_ls1, mesh_ls2):
    """ 
    - sample 10k points on each mesh and compute chamfer distance
    - compute the total loss for every possible permutation
    """
    if len(mesh_ls1) == 0 or len(mesh_ls2) == 0:
        print("Empty mesh list")
        return 0.0, []
    if len(mesh_ls1) != len(mesh_ls2):
        print("Meshes are not the same length")
        return 0.0, []

    all_possible_perms = list(itertools.permutations(range(len(mesh_ls1))))
    all_chamfer_dists = []
    for perm_idxs in all_possible_perms:
        chamfer_dist = []
        mesh_ls1_cp = [mesh_ls1[idx] for idx in perm_idxs]
        for mesh1, mesh2 in zip(mesh_ls1_cp, mesh_ls2):            
            chamfer_dist.append(mesh_distance(mesh1, mesh2, sample_points=10000))
        all_chamfer_dists.append(
            (chamfer_dist, perm_idxs)
            )
    all_chamfer_dists = sorted(all_chamfer_dists, key=lambda x: np.mean(x[0]))
    best_perm_idxs = all_chamfer_dists[0][1]
    best_chamfer_dist = np.mean(all_chamfer_dists[0][0]) 
    print("All dists", all_chamfer_dists)
    return best_chamfer_dist, best_perm_idxs
    
def process_one_sam_image(sam_data_fname, h5_fname, mask_thres=10):
    cam_id = h5_fname.split("/")[-1].split(".hdf5")[0]
    gt_data = load_blender_data(h5_fname, mask_thres)
    with open(sam_data_fname, "rb") as f:
        data = pickle.load(f) 
    used_masks = data['used_masks'] 
    gt_data['used_masks'] = used_masks
    gt_data['weighted_scores'] = data['scores']['weighted']
    gt_data['segment_body'] = {i: mask for i, mask in enumerate(used_masks)} 
    cam_name = f"cam_{cam_id}"
    return gt_data

def get_tsdfs_and_group(
    outputs, vlength, sdf_trunc, depth_trunc, dist_thres, chamfer_voxel_size=0.05, return_imgs=True, size_thres=800
    ):
    """ Process only one object + loop and all its images"""
    all_volumes = get_tsdfs(outputs, vlength=vlength, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc, return_imgs=return_imgs)
    print('done getting tsdfs') 
    dist_type = "chamfer" 
    voxel = True 
    dist_fn = partial(compute_chamfer_distance, dist_type=dist_type, voxel=voxel, voxel_size=chamfer_voxel_size)
    
    tsdf_ls = []
    for cam_name, volumes in all_volumes.items(): 
        tsdf_ls.extend(volumes)
    merged_tsdfs = merge_tsdfs(tsdf_ls, dist_fn, dist_thres=dist_thres)
    print(f"Merged TSDFs: {len(merged_tsdfs)}")
    # examine the size of each tsdf
    size_filtered = []
    for i, tsdf_info in enumerate(merged_tsdfs):
        pcd = tsdf_info['tsdf'].extract_point_cloud()
        print(f"Merged TSDF {i}: {len(pcd.points)} points")
        if len(pcd.points) > size_thres:
            size_filtered.append(tsdf_info)
    print(f"After filtering with >={size_thres} points", len(size_filtered))

    # filter again:
    size_filtered = merge_tsdfs(size_filtered, dist_fn, dist_thres=dist_thres)
    return size_filtered


def main(args):
    ckpt_name = f"ckpt_step_{args.load_steps}.pth"
    lookup_dir = join(
        args.output_dir, args.run_name, ckpt_name, f"loop_{args.load_loop_id}", args.load_obj_type, args.load_obj_folder, "*")
    folders = natsorted(glob(lookup_dir))
    datas = [join(folder, "data.pkl") for folder in folders]
    datas = natsorted(datas)
    if args.subsample_camera_views > 0 and len(datas) >= 24:
        datas = datas[::args.subsample_camera_views]
    if len(datas) == 0:
        print(f"Cannot find data for {lookup_dir}")
        exit()
    
    # copy the GT meshes to the same folder
    gt_dir = join(args.data_dir, args.split, args.load_obj_type, args.load_obj_folder, "blender_meshes")
    target_dir = join(
        args.output_dir, "mesh_extracts", args.run_name, ckpt_name, args.load_obj_type, args.load_obj_folder, "gt_meshes")
    gt_mesh_fnames = natsorted(glob(join(gt_dir, "*")))
    assert len(gt_mesh_fnames) > 0, f"Cannot find gt meshes for {gt_dir}"
    os.makedirs(target_dir, exist_ok=True)
    
    outputs = dict()
    data_fname = datas[0]
    obj_type = data_fname.split("/")[-4]
    obj_folder = data_fname.split("/")[-3]
    # TODO: load gt meshes
    for data_fname in datas: 
        cam_id = data_fname.split("/")[-2]
        h5_fname = glob(
            join(args.data_dir, args.split, obj_type, obj_folder, f"loop_{args.load_loop_id}", f"{cam_id}.hdf5")
            ) 
        assert len(h5_fname) == 1, f"Cannot find h5_fname for {data_fname}"
        h5_fname = h5_fname[0]
        cam_name = f"cam_{cam_id}"
        proccessed_data = process_one_sam_image(data_fname, h5_fname, mask_thres=10)
        outputs[cam_name] = proccessed_data   
               
    filtered_tsdfs = get_tsdfs_and_group(
        outputs, args.vlength, args.sdf_trunc, args.depth_trunc, args.dist_thres, 
        return_imgs=True, size_thres=800)

    # export to mesh 
    mesh_output_dir = join(
        args.output_dir, "mesh_extracts", args.run_name, ckpt_name, args.load_obj_type, args.load_obj_folder, f"loop_{args.load_loop_id}", f"meshes_vlength{args.vlength}_sdf{args.sdf_trunc}_dist{args.dist_thres}")
    os.makedirs(mesh_output_dir, exist_ok=True)

    pred_meshes = []
    for i, tsdf_info in enumerate(filtered_tsdfs):
        tsdf = tsdf_info['tsdf']
        mesh = tsdf.extract_triangle_mesh()
        mesh = mesh.compute_vertex_normals()
        # export to obj file
        obj_fname = join(mesh_output_dir, f"mesh_{i}.obj")
        
        # pred_meshes.append(mesh)
        lap_smoothed = mesh.filter_smooth_laplacian(number_of_iterations=5, lambda_filter=0.8)
        # mesh_smp = lap_smoothed.simplify_vertex_clustering(voxel_size=0.02, contraction=open3d.geometry.SimplificationContraction.Average)
        # mesh_smp = mesh_smp.compute_vertex_normals()
        pred_meshes.append(lap_smoothed)
        o3d.io.write_triangle_mesh(obj_fname, lap_smoothed)
        print(f"Exported mesh to {obj_fname}")

    gt_meshes = []
    for fname in gt_mesh_fnames:
        target_fname = join(target_dir, fname.split("/")[-1])
        shutil.copy(fname, target_fname)
        gt_meshes.append(o3d.io.read_triangle_mesh(fname))

    # compute chamfer distance
    print("Computing mesh distance")
    chamfer_dist, perm_idxs = get_mesh_distances(pred_meshes, gt_meshes)
    print(f"Chamfer distance to GT meshes: {chamfer_dist}")
    breakpoint()
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/local/real/mandi/blender_dataset_v3") 
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_dir", type=str, default="/local/real/mandi/sam_models")
    parser.add_argument("--output_dir", type=str, default="/local/real/mandi/sam_eval") # save eval outputs
    parser.add_argument("--sam_type", default="default", type=str)
    parser.add_argument("--skip_load","-sl", action="store_true")
    parser.add_argument("--loop_id", default=-1, type=int)
    parser.add_argument("--subsample_camera_views", "-sub", type=int, default=-1)
    parser.add_argument("--run_name", "-rn", default="use_pts_pointsTrue_lr0.0003_bs8_gradstep16_10-10_18-43", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--grad_accum_steps", "-ac", default=16, type=int)
    parser.add_argument("--pcd_downsample", default=0.01, type=float)
    parser.add_argument("--load_epoch", default=-1, type=int)
    parser.add_argument("--load_steps", default=-1, type=int) 
    parser.add_argument("--input_xml_fname", default="merged_v1.xml", type=str)
    parser.add_argument("--num_render_cameras", default=10, type=int)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=480, type=int)
    parser.add_argument("--overwrite", "-o", action="store_true") # if True, overwrite existing output folder
    parser.add_argument("--num_grid_points", "-ng", default=32, type=int) # number of grid points to sample
    parser.add_argument("--load_eval", action="store_true") 
    parser.add_argument("--load_obj_type", default="*", type=str) 
    parser.add_argument("--load_obj_folder", default="*", type=str) 
    parser.add_argument("--load_loop_id", default="0", type=str) 
    parser.add_argument("--vlength", default=0.005, type=float) 
    parser.add_argument("--sdf_trunc", default=0.02, type=float) 
    parser.add_argument("--depth_trunc", default=10, type=float) 
    parser.add_argument("--dist_thres", default=0.2, type=float)  
    args = parser.parse_args()
    main(args)

