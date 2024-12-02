"""
Run the 2D to 3D to 2D prompting scheme to evaluate a fine-tuned SAM model 

NOTE: assume the input images are all square
"""
import os
from os.path import join
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import cv2
import numpy as np
import torch 
from torch.utils.data import DataLoader
import argparse

import open3d as o3d
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

from part_segmentation.test_sam_utils import load_sam_model, get_one_tsdf, project_3D_to_2D, eval_model_on_points

from part_segmentation.finetune_sam import get_image_transform
from part_segmentation.sam_datasets import SamH5Dataset
from part_segmentation.sam_to_pcd import load_blender_data

ORIGINAL_IMG_SIZE=(512, 512)
SAM_MASK_SIZE=(1920, 1920)

def load_eval_dataset(
    data_dir, obj_type, obj_folder, loop_id, subsample_cameras
):
    """ Because SAMH5Dataset returns image-by-image, we create one dataset+loader for each object at a time """
    
    dataset_kwargs = {
        "root_dir": data_dir,
        "loop_id": loop_id,
        "subsample_camera_views": subsample_cameras,
        "transform": get_image_transform(1024, jitter=False, random_crop=False, center_crop=0, pad=0), 
        "is_train": False,
        "point_mode": True,
        "original_img_size": ORIGINAL_IMG_SIZE,
        "prompts_per_mask": 1, # NOT actually needed 
        "return_gt_masks": True,
        "obj_type": obj_type,
        "obj_folder": obj_folder,
        "topcrop": 0,
        "bottomcrop": 0,
    }
    val_dataset = SamH5Dataset(**dataset_kwargs)
    # only load one object at a time
    val_loader = DataLoader(
        val_dataset, batch_size=len(val_dataset),
        shuffle=False, num_workers=0
        ) 
    return val_loader, val_dataset

def prepare_img_camera_data(batch, model, device):
    """ 
    Load camera params from h5 files.
    TODO: need to add back processing for background mask for real world data
    Synthetic data is okay for now because the depth outside of object is large enough that the pcd is already object only
    """
    images = batch["image"] 
    h5_fnames = batch["filename"]
    h5_datas = [load_blender_data(fname) for fname in h5_fnames] 
    sam_rgb_size = h5_data[0]['rgb'].shape[:2] # potentially vertical 
    sam_mask_size = SAM_MASK_SIZE
    all_data = []
    for img, h5_data in zip(images, h5_datas): 
        inp_rgb = model.preprocess(img[None])
        embedding = model.image_encoder(inp_rgb) 
        raw_rgb = h5_data['rgb'].copy()
        depth = h5_data['depth'].copy() 
        data = {
            "image": img,
            "raw_rgb": raw_rgb,
            "embedding": embedding,
            "depth": depth,
            "intrinsics": np.array(h5_data['cam_intrinsics']),
            "extrinsics": np.array(h5_data['cam_pose']),
        } 
        all_data.append(data)
    return all_data

def visualize_sam_results():
    return 

def get_prompt_points(tsdf, all_cam_data, num_sample_points):
    # First sample 3D _candidate_ points from the volume
    pcd = tsdf.extract_point_cloud()
    tot_pcd_size = len(np.array(pcd.points))
    mesh = tsdf.extract_triangle_mesh()
    # first sample more points and take only these that have a large number of valid corresponding 2D points on the multi-view RGB images
    candidate_points = mesh.sample_points_uniformly(number_of_points=num_sample_points * 2) 
    num_views_valid = np.zeros(len(candidate_points)) # for each point, count how many views have a valid 2D projection
    all_2d_points = []
    for cam_data in all_cam_data:
        depth = cam_data['depth']
        h, w = depth.shape
        p_image = project_3D_to_2D(
            candidate_points, cam_data['extrinsics'], cam_data['intrinsics'],
            h, w, depth, 
        ) # shape (num_points, 2)
        valid_idxs = np.where(p_image[:, 0] > -1)[0]
        num_views_valid[valid_idxs] += 1
        all_2d_points.append(p_image)

    sorted_idxs = np.argsort(num_views_valid)[::-1][:num_sample_points] # now it's shorter length
    sample_3d_points = candidate_points[sorted_idxs]
    sample_2d_points = [points_2d[sorted_idxs] for points_2d in all_2d_points] # all the projected 2D points for each camera
    return sample_3d_points, sample_2d_points

def filter_sort_all_camera_results(init_values):
    return 

def evaluate_one_object(batch, model, device, forward_fn, save_dir, num_sample_3d_points=10, minibatch_size=128):
    """
    First get a background mask from coarsely sample query points, then, sample 3D points from the foreground masked object.
    Project each 3D points to 2D on each camera view image, then group the mask predictions across all 2D images for that point.
    """
    all_data = prepare_img_camera_data(batch, model, device)
    tsdf = get_one_tsdf(
        rgbs=[data['raw_rgb'] for data in all_data],
        depths=[data['depth'] for data in all_data],
        cam_intrinsics=[data['intrinsics'] for data in all_data],
        extrinsics=[data['extrinsics'] for data in all_data],
    )
    sample_3d_points, sample_2d_points = get_prompt_points(tsdf, all_data, num_sample_3d_points)
    num_cameras = len(all_data)
    num_points = len(sample_3d_points)
    sam_mask_size = SAM_MASK_SIZE
    init_values = dict( 
        masks=np.zeros((num_points, num_cameras, sam_mask_size[0], sam_mask_size[1])),
        repaired_masks=np.zeros((num_points, num_cameras, sam_mask_size[0], sam_mask_size[1])), 
        gt_ious=np.zeros((num_points, num_cameras)),
        stability_scores=np.zeros((num_points, num_cameras)), 
        iou_predictions=np.zeros((num_points, num_cameras)),
        coords_2d=np.zeros((num_points, num_cameras, 2)),
    )
    labels = batch["gt_masks"].detach().cpu().numpy()
    for idx, (point_3d, points_2d, cam_data) in enumerate(zip(sample_3d_points, sample_2d_points, all_data)):
        valid_idxs = np.where(points_2d[:, 0] > -1)[0]
        valid_points = points_2d[valid_idxs]
        mini_batch = int(np.ceil(valid_points.shape[0] / minibatch_size))
        seg_results = []
        embedding = cam_data['embedding']
        image = cam_data['raw_rgb']
        input_size = tuple(image.shape[1:]) 
        for i in range(mini_batch):
            img_label = labels[idx][None]
            point_coords = valid_points[i*minibatch_size:(i+1)*minibatch_size] 
            # flip the xy coords  
            coords_flip = np.zeros_like(point_coords)
            coords_flip[:, 0] = point_coords[:, 1]
            coords_flip[:, 1] = point_coords[:, 0] 
            batch_results = eval_model_on_points(
                coords_flip, 
                img_label,
                embedding, 
                model, 
                sam_mask_size, 
                device, 
                input_size, 
                pad_mask_size=None,
            )
            # these mask results are again sometimes vertical 
            batch_results["coords_2d"] = coords_flip
            seg_results.append(batch_results)
            
        if len(seg_results) == 0:
            seg_results = dict()
        else:
            seg_results = {k: np.concatenate([b[k] for b in seg_results], axis=0) for k in seg_results[0].keys()}
        for k, v in seg_results.items(): 
            if len(v) > 0:
                init_values[k][valid_idxs, idx] = v 
 

def main():
    args = parser.parse_args()

    obj_lookup_type = args.obj_type # might be "*"!
    obj_lookup_folder = args.obj_folder
    loop_id = args.loop_id
    data_dir = args.data_dir
    assert os.path.exists(data_dir), f"Data dir {data_dir} does not exist"
    subsample_cameras = args.subsample_cameras
    if subsample_cameras > 1:
        print(f"WARNING - Subsampling cameras by {subsample_cameras}")

    # lookup all the objects
    unique_objects = natsorted(glob(join(args.data_dir, "test", obj_lookup_type, obj_lookup_folder, f"loop_{loop_id}")))
    print(f"Found {len(unique_objects)} object+rendering loops")

    model, device, forward_fn, load_model_fname = load_sam_model(
        zero_shot=args.zero_shot_sam,
        run_name=args.sam_run_name,
        load_epoch=args.sam_load_epoch,
        load_steps=args.sam_load_steps,
        sam_type=args.sam_type,
        points=True,
        model_dir=args.sam_model_dir, 
        skip_load=args.skip_load_sam,
    )
    save_dir = join(args.output_dir, load_model_fname.split('/')[-2], load_model_fname.split("/")[-1].split(".pth")[0])
    os.makedirs(save_dir, exist_ok=True)
    for obj_loop_dir in tqdm(unique_objects):
        # e.g. /store/real/mandi/real2code_dataset_v0/test/Scissors/11111/loop_0
        obj_type, obj_folder, loop_id = obj_loop_dir.split("/")[-3:]
        obj_save_path = join(save_dir, obj_type, obj_folder, loop_id)
        os.makedirs(obj_save_path, exist_ok=True)
        # if there's already files in the directory, check if args.overwrite is set
        if len(glob(join(obj_save_path, "*"))) > 0 and not args.overwrite:
            print(f"Skipping {obj_save_path} as it already has files")
            continue
        loader, dataset = load_eval_dataset(
            data_dir, obj_type, obj_folder, loop_id, subsample_cameras
        ) 
        batch = next(iter(loader)) # one batch should be all the images for one object!
        
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/store/real/mandi/real2code_dataset_v0")
    parser.add_argument("--output_dir", type=str, default="/store/real/mandi/real2code_eval_v0")

    parser.add_argument("--obj_type", type=str, default="*")
    parser.add_argument("--obj_folder", type=str, default="*")
    parser.add_argument("--loop_id", type=str, default="0")
    parser.add_argument("--sam_run_name", type=str, default="scissors_eyeglasses_only_pointsTrue_lr0.001_bs24_ac24_11-30_00-23")
    parser.add_argument("--sam_load_epoch", type=int, default=None) ## epoch takes precedence over steps
    parser.add_argument("--sam_load_steps", type=int, default=None)
    parser.add_argument("--sam_type", type=str, default="default")
    parser.add_argument("--sam_model_dir", type=str, default="/store/real/mandi/sam_models")
    parser.add_argument("--skip_load_sam", action="store_true") 
    parser.add_argument("--zero_shot_sam", action="store_true")
    parser.add_argument("--subsample_cameras", "-sub", type=int, default=1)
    parser.add_argument("--overwrite", '-o', action="store_true")
    parser.add_argument("--num_3d_points", type=int, default=10)
    main()


