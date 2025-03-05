from __future__ import annotations
# import mujoco
# from dm_control import mjcf 
# from mujoco import viewer
# from dm_control import mujoco as dm_mujoco 
import os
from os.path import join
from glob import glob
from natsort import natsorted
import wandb
import pickle
import json
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import argparse
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling.mask_decoder import MLP as MaskDecoderMLP
from segment_anything.utils.amg import calculate_stability_score, remove_small_regions
from torch.optim import Adam
from monai.losses import DiceCELoss, FocalLoss
from PIL import Image
import torch 
import torch.nn as nn
import numpy as np
from copy import deepcopy
import torchvision
import random 
from torchvision.transforms.functional import crop as tvf_crop
from torchvision.transforms.functional import resize as tvf_resize
from torchvision.transforms import Compose, ColorJitter, ToTensor, Normalize, RandomCrop 
from datetime import datetime
from part_segmentation.sam_to_pcd import load_blender_data
from part_segmentation.finetune_sam import SamH5Dataset, get_image_transform, forward_sam, forward_sam_points, get_wandb_table
import seaborn as sns
import open3d as o3d
CKPT_PATH="/home/mandi/sam_vit_h_4b8939.pth"
"""
RUN=cached_bg4_uniform_16prompt_pointsTrue_lr0.0003_bs8_ac32_10-20_17-42; STEP=58000
python test_sam.py --points --blender --run_name $RUN --load_step ${STEP} --num_grid_points 32 --data_dir /local/real/mandi/blender_dataset_v1/

RUN=sam_v2_pointsTrue_lr0.001_bs8_ac16_01-19_23-56;  STEP=30000
python test_sam.py --points --blender --run_name $RUN --load_step ${STEP} --num_grid_points 32 --data_dir /local/real/mandi/blender_dataset_v3/ --loop_id 0 -sub 2


eval zero-shot model:
python test_sam.py --points --blender --run_name zero_shot --load_step 0 --num_grid_points 32 --data_dir /local/real/mandi/blender_dataset_v1/ --zero_shot
"""
def load_sam_model(
        zero_shot=False, 
        run_name="sam",
        load_epoch=-1,
        load_steps=-1,
        sam_type="default",
        points=True,
        model_dir="/local/real/mandi/sam_models",
        skip_load=False,
        device=None, 
    ):
    if not zero_shot: 
        if len(run_name.split("/")) > 1:
            run_name = run_name.split("/")[-1]
        load_run_dir = join(model_dir, run_name) # same key as training but now use it for loading instead
        ckpts = natsorted(glob(join(load_run_dir, "ckpt_*pth")))
        print(f"Loading from dir: {load_run_dir}")
        if load_epoch == -1:
            ckpt_name = ckpts[-1]
            if load_steps > 0:
                ckpt_name = [c for c in ckpts if str(load_steps) in c.split('/')[-1]][0]
        else:
            ckpts = natsorted(glob(join(load_run_dir, "ckpt_*pth")))
            ckpt_name = [c for c in ckpts if str(load_epoch) in c.split('/')[-1]][0]
        if skip_load:
            return None, None, None, ckpt_name
    else:
        ckpt_name = "ckpt_zero_shot.pth"
    model = sam_model_registry[sam_type](checkpoint=CKPT_PATH)
    print("Original SAM Model loaded.")
    forward_fn = forward_sam_points if points else forward_sam 
    
    if not zero_shot:
        loaded_decoder = torch.load(ckpt_name, map_location="cpu")
        model.mask_decoder.load_state_dict(loaded_decoder)
        print(f"Decoder loaded from {ckpt_name}")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("Model moved to device, params frozen, set to eval mode")
    return model, device, forward_fn, ckpt_name

def compute_iou(pred, gt):
    intersection = np.sum(pred * gt > 0)
    union = np.sum(pred + gt > 0)
    if union == 0:
        return 1
    return intersection / union

def points_nms_filter(masks, iou_thres=0.7):
    """ filter masks using NMS on points """
    num_masks = len(masks)
    b_pool = [mask for mask in masks]
    b_pool_idxs = list(range(num_masks))
    d_pool = []
    d_pool_idxs = []
    while len(b_pool) > 0:
        mask = b_pool.pop(0)
        d_pool.append(mask)
        d_pool_idxs.append(b_pool_idxs.pop(0))
        topop = []
        for i in range(len(b_pool)):
            other_mask = b_pool[i]
            intersection = np.sum(mask * other_mask > 0)
            union = np.sum(mask + other_mask > 0)
            iou = intersection / union
            if iou > iou_thres:
                topop.append(i)
        b_pool = [b_pool[i] for i in range(len(b_pool)) if i not in topop]
        b_pool_idxs = [b_pool_idxs[i] for i in range(len(b_pool_idxs)) if i not in topop]
    return d_pool, d_pool_idxs

def find_mask_for_point(point, masks):
    """ find the GT mask that contains the point, use for eval """
    pos_masks = []
    x, y = int(point[0]), int(point[1])
    for mask in masks: 
        if mask[x, y] > 0:
            pos_masks.append(mask)
    if len(pos_masks) == 0:
        # print('WARNING: no mask found for point')
        return np.zeros(masks[0].shape)
    # if len(pos_masks) > 1:
    #     print(f'WARNING: {len(pos_masks)} masks found for point {point}')
    return pos_masks[0].copy()

def draw_coord_on_mask(binary_mask, coord, color=(255, 0, 0)):
    """ draw a point on a mask """
    rgb_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3))
    x, y = int(coord[0]), int(coord[1])
    rgb_mask[binary_mask > 0] = [255, 255, 255]
    rgb_mask[x-3:x+3, y-3:y+3] = color
    return rgb_mask

def eval_model_on_points(
        point_coords, labels, image_embeddings, model, original_size, device, 
        input_size=(1024, 1024), pad_mask_size=None,
    ):
    gt_masks_for_points = [] 
    if len(labels) > 0:
        for point in point_coords:
            gt_mask = find_mask_for_point(point, masks=labels[0]) # skip batch dim
            gt_masks_for_points.append(gt_mask)
 
    point_coords = torch.from_numpy(point_coords).float().to(device) # shape (num_mask, 2)
    point_coords = point_coords.unsqueeze(1) # shape (num_mask, num_points=1, 2)
    point_labels = torch.ones(
        (point_coords.shape[0], 1), 
        dtype=torch.float32
        ).to(device)
    points = (point_coords, point_labels)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=points, boxes=None, masks=None
    )
    low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings, # shape 1,3,1024,1024
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
    
    pred = model.postprocess_masks(
        low_res_masks,
        input_size=input_size,
        original_size=original_size,
    ) 
    stability_scores = calculate_stability_score(
        pred[:,0,:,:], mask_threshold=0, threshold_offset=0.8
        ).detach().cpu().numpy() # shape (num_masks_per_batch, )
    
    pred = pred[:,0,:,:].detach().cpu().numpy() # shape (num_masks_per_batch, 640, 640)
    masks = pred 
    binary_masks = masks > 0
    bboxes = []
    repaired_masks = []
    for mask in masks:
        repaired_mask = (mask > 0).astype(np.uint8) 
        repaired_mask, filled_hole = remove_small_regions(repaired_mask, area_thresh=100, mode="holes")
        repaired_mask, filled_hole = remove_small_regions(repaired_mask, area_thresh=100, mode="islands")
        pos_mask = np.where(repaired_mask > 0)  
        repaired_masks.append(repaired_mask) # this changes from 0-1 to True/False
        
        if len(pos_mask[0]) == 0:
            # print("Got all zero mask")
            # bboxes.append([0, 0, h, w])
            continue
        y1, y2 = np.min(pos_mask[0]), np.max(pos_mask[0])
        x1, x2 = np.min(pos_mask[1]), np.max(pos_mask[1])
        bboxes.append([x1, y1, x2, y2]) 
    # compute iou for each point-prediction and its gt_mask
    gt_ious = []
    for mask, gt in zip(binary_masks, gt_masks_for_points):
        intersection = np.sum(mask * gt > 0)
        union = np.sum(mask + gt > 0)
        iou = intersection / union 
        gt_ious.append(iou)
    
    if pad_mask_size is not None:
        # pad the 1440x1440 mask to 1920x1440 or 1440x1920
        assert pad_mask_size[0] == 1920 or pad_mask_size[1] == 1920, f"Invalid pad_mask_size {pad_mask_size}"
        new_pred = []
        for mask in pred:
            new_mask = np.zeros(pad_mask_size)
            if pad_mask_size[0] == 1920:
                new_mask[240:-240, :] = mask
            else:
                new_mask[:, 240:-240] = mask
            new_pred.append(new_mask)
        pred = np.stack(new_pred, axis=0)
        new_repaired = []
        for mask in repaired_masks:
            new_mask = np.zeros(pad_mask_size)
            if pad_mask_size[0] == 1920:
                new_mask[240:-240, :] = mask
            else:
                new_mask[:, 240:-240] = mask
            new_repaired.append(new_mask)
        repaired_masks = new_repaired

    return dict( 
        masks=pred, # cont. mask
        # binary_masks=binary_masks,
        repaired_masks=np.stack(repaired_masks, axis=0),
        # gt_masks=gt_masks_for_points,
        gt_ious=gt_ious,
        stability_scores=stability_scores,
        # bboxes=bboxes, 
        iou_predictions=iou_predictions.squeeze(1).detach().cpu().numpy(),
    )

def sort_mask_by_score(masks, stability_score, iou_preds, stability_weight=0.5):
    """ sort masks by stability score and iou predictions """
    overall_scores = []
    for i in range(len(masks)):
        overall_scores.append(
            np.clip(stability_score[i], 0, 1) * stability_weight + np.clip(iou_preds[i], 0, 1) * (1 - stability_weight)
        )
    sorted_idxs = np.argsort(overall_scores)[::-1]
    sorted_masks = [masks[i] for i in sorted_idxs] 
    return sorted_masks, sorted_idxs

def get_background_mask(
        model, device, image_embeddings, labels, grid_points, original_size, input_size=(1024, 1024), pad_mask_size=None):

    batch_results = eval_model_on_points(
        grid_points, labels, image_embeddings, model, device=device, original_size=original_size, input_size=input_size, pad_mask_size=pad_mask_size
    )  
    sorted_masks, sorted_idxs = sort_mask_by_score(
        batch_results["repaired_masks"], 
        batch_results["stability_scores"],
        batch_results["iou_predictions"], 
    )
    bg_mask = sorted_masks[0]
    return batch_results, bg_mask

def sample_points_eval( 
    model, 
    device, 
    image, 
    labels, # should contain ALL the GT masks loaded from validation dataset
    save_path,
    num_grid_points=32, 
    num_masks_per_batch=32,
    original_size=(640, 640),
    sample_foreground=False, # if True, first get the bg mask, then only sample grid points from the segmented foreground
    pad_mask_size=None,
    sample_nonzero=False, 
):
    """ sample a grid of points on image and group masks """
    assert len(image.shape) == 4 and image.shape[1] == 3, f"image shape {image.shape} is not (batch, 3, H, W)"
    h, w = original_size
    grid_size = int(h/num_grid_points)
    input_size = (image.shape[2], image.shape[3])
    rgb = model.preprocess(image)
    image_embeddings = model.image_encoder(rgb)
    bs = rgb.shape[0]
    all_outputs = []

    grid_coords = np.mgrid[0:h:grid_size, 0:w:grid_size].reshape(2, -1).T # shape (n, 2) 
    if sample_foreground:
        corner_grids = grid_coords[:3] # sample from the top left corner
        batch_results, bg_mask = get_background_mask(
            model, device, image_embeddings, labels, corner_grids, 
            original_size=original_size, input_size=input_size, pad_mask_size=pad_mask_size
            )
        all_outputs.append(batch_results)
        # now, sample grid points from the foreground mask!
        fg_mask = (1 - bg_mask > 0) > 0
        fg_mask_coords = np.where(fg_mask > 0)
        fg_coord_idxs = np.random.choice(fg_mask_coords[0].shape[0], size=num_grid_points**2, replace=True)
        fg_coords = np.stack([fg_mask_coords[0][fg_coord_idxs], fg_mask_coords[1][fg_coord_idxs]], axis=1)
        grid_coords = fg_coords
    
    # visualize the grid and save:
    img = image[0].permute(1,2,0).detach().cpu().numpy() # shape (H, W, 3) NOTE: batch['image'] is already resized into 1024x1024
    img = img.astype(np.uint8) 
    if save_path is not None:
        Image.fromarray(img).save(join(save_path, "rgb_grid.png"))  
    if sample_nonzero:
        # sample only from the non-zero region of the RGB image
        sum_img = np.sum(img, axis=2) # shape (H, W)
        non_zero_coords = np.where(sum_img > 0)
        non_zero_idxs = np.random.choice(non_zero_coords[0].shape[0], size=num_grid_points**2, replace=True)
        non_zero_coords = np.stack([non_zero_coords[0][non_zero_idxs], non_zero_coords[1][non_zero_idxs]], axis=1)
        grid_coords = non_zero_coords

    num_batchs = int(np.ceil(grid_coords.shape[0] / num_masks_per_batch))
    print(f"Sampling {grid_coords.shape[0]} points, will split into {num_batchs} batches")
    for i in range(num_batchs):
        point_coords = grid_coords[i*num_masks_per_batch:(i+1)*num_masks_per_batch]
        batch_results = eval_model_on_points(
            point_coords, labels, image_embeddings, model, 
            original_size=original_size, 
            device=device, 
            pad_mask_size=pad_mask_size,
        ) 
        all_outputs.append(
            batch_results
        )
        # TODO: reduce size!!  
    # concatenate all the outputs
    final_outputs = dict()
    for k in all_outputs[0].keys():
        final_outputs[k] = np.concatenate([o[k] for o in all_outputs], axis=0) 

    if save_path is not None:
        with open(join(save_path, "eval_outputs.pkl"), "wb") as f: 
            pickle.dump(final_outputs, f)
        print(f"Saved results to {save_path}") 
 
    return final_outputs

def multimask_nms_filter(masks_ls, iou_thres=0.7):
    """ 
    masks_ls is a list of multimask outputs, each corresponds to one 3D point and multiple 2D projected masks, shaped (num_masks, h, w) 
    for each pair of points, get iou across pairware 2D masks, total iou is the mean of all pairwise ious
    """
    def get_iou(mask, other_mask):
        intersection = np.sum(mask * other_mask > 0)
        union = np.sum(mask + other_mask > 0)
        if union == 0: # both masks are empty
            return 1 
        return intersection / union
    num_points = len(masks_ls)
    b_pool = [masks for masks in masks_ls]
    b_pool_idxs = list(range(num_points))
    d_pool = []
    d_pool_idxs = []
    while len(b_pool) > 0:
        masks = b_pool.pop(0)
        d_pool.append(masks)
        d_pool_idxs.append(b_pool_idxs.pop(0))
        topop = []
        for i in range(len(b_pool)):
            other_masks = b_pool[i]
            all_ious = []
            for mask, other_mask in zip(masks, other_masks): 
                all_ious.append(get_iou(mask, other_mask))
            
            if np.mean(all_ious) > iou_thres:
                topop.append(i)
        b_pool = [b_pool[i] for i in range(len(b_pool)) if i not in topop]
        b_pool_idxs = [b_pool_idxs[i] for i in range(len(b_pool_idxs)) if i not in topop]
    return d_pool, d_pool_idxs

def get_one_tsdf(
        rgbs, depths, cam_intrinsics, extrinsics, 
        h=100, w=100, vlength=0.01, trunc=0.02,
        return_info=False,
        ):
    """ 
    Integrate one rgbd image into the volume, then extract the point cloud and mesh
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=vlength, 
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
    h, w = depths[0].shape
    for i, (rgb, depth, cam_intr, extri) in enumerate(zip(rgbs, depths, cam_intrinsics, extrinsics)):
        if rgb.shape[0] != h or rgb.shape[1] != w:
            # scanner data has bigger rgb than depth!
            rgb = cv2.resize(rgb, (w, h))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(depth), 
            depth_trunc=10, # set this to high!!!
            depth_scale=1, convert_rgb_to_intensity=False,
        )
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(
            width=w, height=h, cx=w/2, cy=h/2, fx=cam_intr[0,0], fy=cam_intr[1,1]
        )
        volume.integrate(rgbd, intrinsics, extri)
    if return_info:
        return volume, rgbd, intrinsics, extri
    return volume

def sample_3d_eval(
        model, device, images, h5_fnames, labels, save_path, 
        num_points=10, original_size=(512, 512), num_masks_per_batch=128, overwrite=False
    ):
    """ 
    First get a background mask from coarsely sample query points, then, sample 3D points from the foreground masked object.
    Project each 3D points to 2D on each camera view image, then group the mask predictions across all 2D images for that point.
    
    NOTE: for scanner data, the images from SAMH5Dataset is already center-cropped to be 1440x1440, need to paste it into a 1920x1440 raw image
    """
    assert len(images) == len(h5_fnames), f"Got {len(images)} images but {len(h5_fnames)} h5 files"
    
    # take first image for bg mask
    bg_masks = []
    h, w = original_size
    grid_size = int(h/24)
    corner_grids = np.mgrid[0:h:grid_size, 0:w:grid_size].reshape(2, -1).T[:10]
    # now, project all the 2D images into 3D foreground object 
    print('Loading cam h5 data')
    h5_data = [load_blender_data(fname) for fname in h5_fnames] 
    is_scanner = 'scanner' in h5_fnames[0]
    sam_rgb_size = h5_data[0]['rgb'].shape[:2] # potentially vertical 
    sam_mask_size = (1920, 1920) #(1440, 1440)
    pad_mask_size = None # sam_rgb_size if is_scanner else None
    print('Loaded cam h5 data, RGB size:', sam_rgb_size)
 
    sorted_results = dict()
    # remove files inside save_path:
    if os.path.exists(join(save_path, "nms_filtered_masks.pkl")) and not overwrite:
        print(f"Found existing nms_filtered_masks.pkl in {save_path}, skipping")
        sorted_results = pickle.load(open(join(save_path, "nms_filtered_masks.pkl"), "rb"))
        bg_masks = sorted_results["bg_masks"]
    else:
        for f in glob(join(save_path, "*")):
            os.remove(f) 
        for i, img in enumerate(images):
            input_size = tuple(img.shape[1:]) # 3, 1024, 768
            rgb = model.preprocess(img[None])
            image_embeddings = model.image_encoder(rgb) 
            bg_labels = [] if is_scanner else labels
            batch_results, bg_mask = get_background_mask(
                model, device, image_embeddings, bg_labels, corner_grids, sam_mask_size, input_size, pad_mask_size=pad_mask_size
                )
            bg_masks.append(bg_mask) 
        print(f"SAM model input size: {input_size}")
    needs_rotate = sam_rgb_size[0] != h or sam_rgb_size[1] != w
    cam_data = []
    for i, (bg_mask, data) in enumerate(zip(bg_masks, h5_data)):
        rgb = data['rgb'].copy()
        depth = data['depth']
        camera_intrinsics = np.array(data['cam_intrinsics'])
        camera_pose = np.array(data['cam_pose']) 
        rgb[bg_mask > 0] = 0
        
        if data['is_scanner']:
            # requires scaling the SAM outputs to smaller (256, 192) size
            depth_shape = depth.shape
            # assert depth_shape == (h, w), f"Got depth shape {depth_shape} but expected {(h, w)}"
            mask = np.array(data['binary_masks'][0], dtype=np.float32)
            bg_mask_resized = np.array(bg_mask.copy(), dtype=np.float32) #cv2.resize(bg_mask, (w, h))
            rgb[mask > 0] = 0
            if np.sum(data['need_rotate_back']):
                assert needs_rotate, "Got need_rotate_back but not needs_rotate"
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                bg_mask_resized = cv2.rotate(bg_mask_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rgb = cv2.rotate(rgb,  cv2.ROTATE_90_COUNTERCLOCKWISE)
            rot_resized_mask = cv2.resize(
                mask, (depth_shape[1], depth_shape[0])
            )
            bg_mask_resized = cv2.resize(
                bg_mask_resized, (depth_shape[1], depth_shape[0])
            )
            depth[rot_resized_mask > 0] = -1
            depth[bg_mask_resized > 0] = -1
            depth = np.array(depth).astype(np.float32)
            # rgb = cv2.resize(rgb, (depth_shape[1], depth_shape[0]))
            # Image.fromarray(rgb).save(f"test_rgb_{i}.png")
            # breakpoint()
        else:
            depth[bg_mask > 0] = -1
        extrinsics = np.linalg.inv(camera_pose)
        extrinsics[2, :] *= -1
        extrinsics[1, :] *= -1
        # NOTE: the rgb are rotated but NOT resized! depth is still small and but masked
        cam_data.append((camera_intrinsics, extrinsics, depth, rgb)) 
    volume = get_one_tsdf(
        rgbs=[d[3] for d in cam_data], 
        depths=[d[2] for d in cam_data], 
        cam_intrinsics=[d[0] for d in cam_data], 
        extrinsics=[d[1] for d in cam_data], 
        # h=h, 
        # w=w
    )
    # now, sample 3D points from the volume
    pcd = volume.extract_point_cloud()
    tot_pcd_size = len(np.array(pcd.points))
    mesh = volume.extract_triangle_mesh()
    surface_points = mesh.sample_points_uniformly(number_of_points=num_points) 
    sample_points = np.array(surface_points.points)
    # np.savez("surface_points.npz", points=np.array(pcd.points))
    # np.savez("sample_points.npz", points=sample_points)
    # breakpoint()
    colors = sns.color_palette("colorblind", num_points * 2)
    colors = [np.array(np.array(c) * 255, dtype=np.uint8) for c in colors] 
    # sample twice as many points as needed, then take the points that have the most valid projections
    def scatter_points(img, points, colors):
        for i, point in enumerate(points):
            if point[0] > -1:
                x, y = int(point[0]), int(point[1]) 
                img[y: y+5, x:x+5] = colors[i]
        return img
    if len(sorted_results) == 0: 
        candidate_points = np.array(
            mesh.sample_points_uniformly(number_of_points=num_points*2).points # so annoying!
            )
        # np.savez("cand_points.npz", points=candidate_points)
        num_valid = np.zeros(num_points*3)
        # test if the projection is correct:
        for idx, (image, _data) in enumerate(zip(images, cam_data)): 
            camera_intrinsics, extrinsics, depth, rgb = _data
            p_image = project_3D_to_2D(candidate_points, extrinsics, camera_intrinsics, h, w, depth)
            valid_idxs = np.where(p_image[:, 0] > -1)[0]
            num_valid[valid_idxs] += 1
            # # save test img
            # blank = rgb.copy()
            # for i, point in enumerate(p_image):
            #     # if point[0] > -1:
            #     x, y = int(point[0]), int(point[1]) 
            #     blank[y-5:y+5, x-5:x+5] = colors[i]
            #     Image.fromarray(blank.astype(np.uint8)).save(f"test_img_{idx}.png") 
        # now, take the top num_points from the candidate points
        sorted_idxs = np.argsort(num_valid)[::-1]
        sample_points = candidate_points[sorted_idxs[:num_points]]
        # np.savez("sample_points.npz", points=sample_points)
         
        # now, project each 3D point into 2D on each camera view image
        points_results = []
        num_cameras = len(images)
        init_values = dict( 
            masks=np.zeros((num_points, num_cameras, sam_mask_size[0], sam_mask_size[1])),
            repaired_masks=np.zeros((num_points, num_cameras, sam_mask_size[0], sam_mask_size[1])), 
            gt_ious=np.zeros((num_points, num_cameras)),
            stability_scores=np.zeros((num_points, num_cameras)), 
            iou_predictions=np.zeros((num_points, num_cameras)),
            coords_2d=np.zeros((num_points, num_cameras, 2)),
        )
        if is_scanner:
            init_values["masks"] = np.zeros((num_points, num_cameras, pad_mask_size[0], pad_mask_size[1]))
            init_values["repaired_masks"] = np.zeros((num_points, num_cameras, pad_mask_size[0], pad_mask_size[1]))

        def rotate_2d_points(points, before_rot_shape=(1440, 1920)):
            idx_img = np.zeros(before_rot_shape)
            for i, pt in enumerate(points):
                x, y = pt
                idx_img[y, x] = i+1
            rotated_idx = cv2.rotate(idx_img, cv2.ROTATE_90_CLOCKWISE)
            rotated_pts = []
            for i in range(len(points)):
                if len(np.where(np.ones(1) == 2)[0]) > 0:
                    x, y = np.where(rotated_idx == i+1)[0][0], np.where(rotated_idx == i+1)[1][0]
                else:
                    x, y = 0, 0
                rotated_pts.append((y,x))
            return np.array(rotated_pts)

        for idx, (image, _data) in enumerate(zip(images, cam_data)):
            img_label = labels[idx][None]  
            if is_scanner:
                img_label = [] # not using 

            input_size = tuple(image.shape[1:]) 
            camera_intrinsics, extrinsics, depth, rgb = _data
            p_image = project_3D_to_2D(sample_points, extrinsics, camera_intrinsics, h, w, depth)
            # save test img
            # blank = rgb.copy()
            # blank = scatter_points(blank, p_image, colors) 
            # Image.fromarray(blank.astype(np.uint8)).save(f"test_img_{idx}.png") 

            # # NOTE: the RGB in cam_data are alreay rotated to horizontal, but for SAM, the prompting RGB
            # images are sometimes vertical, hence the prompting points also need to be rotated back check the rgb can be roated back
            
            valid_idxs = np.where(p_image[:, 0] > -1)[0]
            valid_points = p_image[valid_idxs]
            
            if needs_rotate:
                # these 3D->2D points are always horizontal but the SAM input are sometimes vertical
                print("Rotating prompt points before SAM:")
                valid_points = rotate_2d_points(valid_points, before_rot_shape=(1440, 1920)) 
            
            mini_batch = int(np.ceil(valid_points.shape[0] / num_masks_per_batch))
            seg_results = []
            image_embeddings = model.image_encoder(model.preprocess(image[None])) 
            for i in range(mini_batch):
                point_coords = valid_points[i*num_masks_per_batch:(i+1)*num_masks_per_batch] 
                # flip the xy coords  
                coords_flip = np.zeros_like(point_coords)
                coords_flip[:, 0] = point_coords[:, 1]
                coords_flip[:, 1] = point_coords[:, 0] 
                batch_results = eval_model_on_points(
                    coords_flip, img_label, image_embeddings, 
                    model, sam_mask_size, device, input_size, 
                    pad_mask_size=pad_mask_size
                )
                # these mask results are again sometimes vertical 
                batch_results["coords_2d"] = coords_flip
                seg_results.append(batch_results)
                
            if len(seg_results) == 0:
                seg_results = dict()
            else:
                seg_results = {k: np.concatenate([b[k] for b in seg_results], axis=0) for k in seg_results[0].keys()}
            for k, v in seg_results.items():
                # if v.shape[0] != valid_idxs.shape[0]:
                #     breakpoint()
                if len(v) > 0:
                    init_values[k][valid_idxs, idx] = v 

        # only take non-zero values to compute avg. 
        # now, get the pooled-stats for each point across all image views
        stability_weight = 0.8
        weighted_scores = []
        for i in range(num_points):
            s_score = np.clip(init_values["stability_scores"][i], 0, 1)
            if np.sum(s_score) == 0:
                s_score = 0
            else:
                s_score = np.mean(s_score) #[s_score > 0])
            iou_score = np.clip(init_values["iou_predictions"][i], 0, 1)
            if np.sum(iou_score) == 0:
                iou_score = 0
            else:
                iou_score = np.mean(iou_score) #[iou_score > 0])
            weighted_score = s_score * stability_weight + iou_score * (1 - stability_weight)
            weighted_scores.append(weighted_score)
        weighted_scores = np.array(weighted_scores) 
        sorted_idxs = np.argsort(weighted_scores)[::-1] 
        weighted_scores = weighted_scores[sorted_idxs]
        min_score = 0.1
        if np.sum(weighted_scores > min_score) > 0:
            sorted_idxs = sorted_idxs[weighted_scores >= min_score]
        sorted_results = {k: init_values[k][sorted_idxs] for k in init_values.keys()}
        
        # run NMS on all the masks
        nms_filtered, nms_idxs = multimask_nms_filter(sorted_results["repaired_masks"], iou_thres=0.6)
 
        if needs_rotate:
            # NOTE: need to rotate the vertical images back to horizontal
            print("Rotating the vertical images back to horizontal")
            rotated_nms = []
            for n in range(len(nms_filtered)):
                rotated_masks = np.array([cv2.rotate(m, cv2.ROTATE_90_CLOCKWISE) for m in nms_filtered[n]])
                rotated_nms.append(rotated_masks)
            nms_filtered = np.array(rotated_nms) 
        print(f"Filtered {len(nms_filtered)} masks")
        for masks, idx in zip(nms_filtered, nms_idxs):
            coords_2d = sorted_results["coords_2d"][idx]
            rgb_masks = []
            for binary_mask, coords in zip(masks, coords_2d):
                x, y = int(coords[0]), int(coords[1])
                rgb_mask = np.zeros((sam_mask_size[0], sam_mask_size[1], 3))
                if is_scanner:
                    rgb_mask = np.zeros((pad_mask_size[0], pad_mask_size[1], 3))
                rgb_mask[binary_mask > 0] = [255, 255, 255]
                rgb_mask[x:x+5, y:y+5] = [255, 0, 0]
                rgb_masks.append(rgb_mask)
            concat_masks = np.concatenate(rgb_masks, axis=1)
            Image.fromarray(concat_masks.astype(np.uint8)).save(
                    join(save_path, f"nms_mask_{idx}.png")
                )
        breakpoint()
        # save all the nms-filtered masks to pkl
        tosave = {k: sorted_results[k][nms_idxs] for k in sorted_results.keys()}
        tosave['bg_masks'] = bg_masks 
        with open(join(save_path, "nms_filtered_masks.pkl"), "wb") as f:
            pickle.dump(tosave, f)
    
    else:
        nms_filtered = sorted_results["repaired_masks"]
    
    # remove existing npz
    for f in glob(join(save_path, "filled_pcd*.npz")):
        os.remove(f)
    for f in glob(join(save_path, "filled_mesh*.obj")):
        os.remove(f)
    
    # Now, project the masks to a pcd and fill the entire obj.  
    filled_pcds, filled_volumes = [] , []
    used_masks = []
    for pt_idx, masks in enumerate(nms_filtered):
        rgbs = [_data[3].copy() for _data in cam_data]
        depths = [_data[2].copy() for _data in cam_data]
        # use previously used mask to get only the new pixels provided by each new mask
        new_masks = masks
        for mask_ls in used_masks:
            for i, old_mask in enumerate(mask_ls):
                new_mask = new_masks[i] * (old_mask == 0)
                new_masks[i] = new_mask
        # mask out the current mask 
        for i, mask in enumerate(new_masks):
            rgbs[i][mask < 1] = 0
            depths[i][mask < 1] = -1
        volume = get_one_tsdf(rgbs, depths, [d[0] for d in cam_data], [d[1] for d in cam_data], h, w)
        pcd = volume.extract_point_cloud()
        if len(pcd.points) < 900:
            print(f"Skipping point {pt_idx} with {len(pcd.points)} points")            
            continue
         
        filled_pcds.append(pcd)
        filled_volumes.append(volume)
        used_masks.append(new_masks)
        filled_pcd_size = sum([len(np.array(pcd.points)) for pcd in filled_pcds])
        
        print(f"Filled {filled_pcd_size} points, at {pt_idx} point idx")
        # np.savez(join(save_path, f"filled_pcd_{len(filled_pcds)-1}.npz"), points=np.array(pcd.points))
        # breakpoint()
        if filled_pcd_size / tot_pcd_size > 0.9:
            break
    # merge the final volumes iteratively
    merged_volumes = filled_volumes 
    # now save the filled volume as meshes   
    for i, volume in enumerate(merged_volumes):
        mesh = volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(
            join(save_path, f"filled_mesh_{i}.obj"), mesh
            )
        pcd = np.array(volume.extract_point_cloud().points)
        np.savez(join(save_path, f"filled_pcd_{i}.npz"), points=pcd)
    return 

def project_3D_to_2D(points, extrinsics, intrinsics, h, w, depth_image, tolerance=0.01):
    # points: shape N,3
    # h, w = depth_image.shape
    num_points = points.shape[0]
    homo_points = np.concatenate([points, np.ones((num_points, 1))], axis=1) # shape N, 4
    p_camera = np.dot(homo_points, extrinsics.T) # N, 4
    p_image_homo = np.dot(p_camera[:, :3], intrinsics.T)
    
    p_image = p_image_homo[:, :2] / p_image_homo[:, 2][:, np.newaxis]
    # filter the points that go beyond the image frame
    filtered = p_image.copy()
    outside_indices = (p_image[:, 0] < 0) | (p_image[:, 0] >= w) | (p_image[:, 1] < 0) | (p_image[:, 1] >= h)
    # Assign -1 to points outside the image frame
    filtered[outside_indices] = -1
    # also make sure the project point is valid depth
    p_image_rounded = np.round(p_image).astype(int)
    
    # Initialize filtered array with -1 to indicate invalid points
    filtered = np.full(p_image.shape, -1, dtype=int)
    
    # Filter points outside the image frame
    inside_indices = (p_image_rounded[:, 0] >= 0) & (p_image_rounded[:, 0] < w) & \
                     (p_image_rounded[:, 1] >= 0) & (p_image_rounded[:, 1] < h)
    
    valid_points = p_image_rounded[inside_indices]
    valid_points_on_depth = valid_points
    projected_depth = p_camera[:, 2]
    projected_depth_valid = projected_depth[inside_indices]

    # NOTE: when the depth and RGB image are not the same shape, need to resize the projected point to smaller depth image shape
    if h > depth_image.shape[0]:
        valid_points_on_depth = valid_points * (depth_image.shape[0] / h)
        valid_points_on_depth = np.round(valid_points_on_depth).astype(int)
    # Fetch depth values from the depth image for valid points
    depth_image_values = depth_image[valid_points_on_depth[:, 1], valid_points_on_depth[:, 0]]
    
    # Filter based on depth comparison
    depth_valid = projected_depth_valid <= (depth_image_values + tolerance)
    
    # Only keep points that pass the depth comparison
    filtered[inside_indices] = np.where(depth_valid[:, np.newaxis], valid_points, -1) 
    
    return filtered

def get_filled_img(sorted_masks, threshold=0.999, min_new_pixels=800):
    """ use a list of sorted masks to fill an image in order """
    sorted_masks = deepcopy(sorted_masks)  #take the top 20 masks
    # sort again by shape from small to big
    # sorted_masks = sorted_masks[:1] + sorted(sorted_masks, key=lambda x: np.sum(x)) # first one is bg mask, stable but biggest
    # print(f"Given {len(sorted_masks)} masks to fill image with threshold {threshold} and min_new_pixels {min_new_pixels}")
    init_mask = np.zeros(sorted_masks[0].shape)
    threshold_size = int(init_mask.shape[0] * threshold * init_mask.shape[1] * threshold)
    background_color = np.array([255, 255, 255])
    
    mask_idx = 1
    used_idx = 0
    used_masks = []
    used_idxs = []
    while np.sum(init_mask > 0) < threshold_size and len(sorted_masks) > 0:
        curr_mask = sorted_masks.pop(0) > 0
        # compare this new mask with current init_mask, and only take the pixels that are currently 0
        new_mask = curr_mask * (init_mask == 0)
        if np.sum(new_mask) < min_new_pixels:
            continue
        print(f"New mask size: {np.sum(new_mask)}")
        # print(f"current mask size: {np.sum(curr_mask)}, new mask size: {np.sum(new_mask)}")
        used_masks.append(new_mask) #curr_mask)
        used_idxs.append(used_idx)
        used_idx += 1
        # give new_mask a mask idx
        new_mask = new_mask * mask_idx
        # add new_mask to init_mask
        init_mask = init_mask + new_mask 
        mask_idx += 1
    if len(used_masks) <= 13:
        mask_colors = sns.color_palette("colorblind", 12) # has only 12 unique colors
    else:
        mask_colors = sns.color_palette("mako", len(used_masks)*3) # this provides unique colors q
        mask_colors = mask_colors[::3]
    mask_colors = [background_color] + [np.array(np.array(c) * 255, dtype=np.uint8) for c in mask_colors]
    init_rgb = np.zeros((init_mask.shape[0], init_mask.shape[1], 3))
    for i, mask in enumerate(used_masks):
        mask_color = mask_colors[i]  
        rgb_idxs = np.where(mask > 0)
        init_rgb[rgb_idxs[0], rgb_idxs[1], :] = mask_color
    # Image.fromarray(init_rgb.astype(np.uint8)).save('test_filled.png')
    print(f"Filled image with {len(used_masks)} masks")
    return init_rgb.astype(np.uint8), used_masks, used_idxs

def draw_bbox(mask, bbox, bbox_edge_size=3):
    """ draw a bbox on an image """
    x1, y1, x2, y2 = bbox
    mask[y1:y1+bbox_edge_size, x1:x2] = 1
    mask[y2-bbox_edge_size:y2, x1:x2] = 1
    mask[y1:y2, x1:x1+bbox_edge_size] = 1
    mask[y1:y2, x2-bbox_edge_size:x2] = 1
    return mask

def process_eval_outputs(outputs, iou_thres=0.7):
    """ filter the masks and get stats from batched model outputs"""
    for key in ["repaired_masks", "iou_predictions", "stability_scores"]:
        assert key in outputs, f"Key {key} not found in outputs"
    masks = outputs["repaired_masks"] > 0
    # labels = outputs["gt_masks"]
    iou_preds = outputs["iou_predictions"]
    stability_scores = outputs["stability_scores"]
    gt_ious = outputs["gt_ious"]
    # sort all the masks
    overall_scores = []
    stability_weight = 0.7
    iou_weight = 1 - stability_weight
    
    for i in range(len(masks)):
        overall_scores.append(
            np.clip(stability_scores[i], 0, 1) * stability_weight + np.clip(iou_preds[i], 0, 1) * iou_weight
        )
    sorted_idxs = np.argsort(overall_scores)[::-1]
    sorted_scores = [overall_scores[i] for i in sorted_idxs]
    sorted_stability_scores = [stability_scores[i] for i in sorted_idxs]
    sorted_iou_preds = [iou_preds[i] for i in sorted_idxs]
    
    # cutoff all the scores below 0.5
    cutoff_idx = len(sorted_idxs) 
    if np.sum(np.array(sorted_scores) < 0.5) > 0:
        cutoff_idx = np.where(np.array(sorted_scores) < 0.5)[0][0]
    sorted_masks = [masks[i] for i in sorted_idxs][:cutoff_idx]
    sorted_gt_ious = [gt_ious[i] for i in sorted_idxs][:cutoff_idx]
    sorted_scores = sorted_scores[:cutoff_idx]
    sorted_stability_scores = sorted_stability_scores[:cutoff_idx]
    sorted_iou_preds = sorted_iou_preds[:cutoff_idx]

    # print(f"Sorted {len(sorted_masks)} masks") -> on scale ot 4, 5 hundred
    
    # then, use NMS to filter the masks
    nms_filtered, nms_idxs = points_nms_filter(sorted_masks, iou_thres) 
    nms_scores = [sorted_scores[i] for i in nms_idxs]
    nms_stability_scores = [sorted_stability_scores[i] for i in nms_idxs]
    nms_iou_preds = [sorted_iou_preds[i] for i in nms_idxs]
    # use the filtered masks to fill an image
    filled_img, used_masks, used_idxs_on_nms = get_filled_img(nms_filtered, threshold=0.999, min_new_pixels=1000)
    used_idxs = [nms_idxs[i] for i in used_idxs_on_nms] 
    used_scores = [nms_scores[i] for i in used_idxs_on_nms]
    continous_masks = outputs["masks"][sorted_idxs]
    used_cont_masks = [continous_masks[i] for i in used_idxs]

    # filled_labels, used_labels, used_label_idxs = get_filled_img([l for l in labels]) # labels are shaped (1, num_masks, 640, 640)
    all_scores = dict(
        weighted=[np.round(x, 4) for x in used_scores],
        stability=[np.round(nms_stability_scores[i], 4) for i in used_idxs_on_nms],
        iou_pred=[np.round(float(nms_iou_preds[i]), 4) for i in used_idxs_on_nms],
        stability_weight=[stability_weight],
    )
    all_scores = {k: [float(x) for x in v] for k, v in all_scores.items()}
    
    return dict(
        used_masks=used_masks,
        used_cont_masks=used_cont_masks,
        nms_filtered=nms_filtered,
        filled_img=filled_img,
        nms_scores=nms_scores,
        scores=all_scores,
        # filled_labels=filled_labels,
        iou_stats=dict(
            iou_all=np.mean(gt_ious),
            iou_filtered=np.mean([sorted_gt_ious[i] for i in nms_idxs]),
            iou_used=np.mean([sorted_gt_ious[i] for i in used_idxs]),
        )
    )

def load_eval_data(args, load_model_fname):
    """ post-process the eval data """
    data_path = join(args.output_dir, args.run_name, load_model_fname.split("/")[-1])
    batches = natsorted(glob(join(data_path, "batch_*")))
    if len(batches) == 0:
        print(f"No batches found in {data_path}")
        exit()
    iou_thresholds = [0.7, 0.9]
    stability_thresholds = [0.7, 0.9]
    col_names = ["rgb", "fill_img", "fill_label", "after_nms"]
    thres_tuples = []
    for iou_thres in iou_thresholds:
        for score in stability_thresholds:
            col_names.append(f"iou{iou_thres:.2f}_score{score:.2f}")
            thres_tuples.append((iou_thres, score))
    table = wandb.Table(columns=col_names)
    print(f"Found {len(batches)} eval batches")
     
    iou_stats = {k: [] for k in ["all", "nms_filtered", "used"]}

    for j, batch in enumerate(batches):
        if j % 25 == 0:
            print(f"Processing batch {j}")
        rgb = wandb.Image(join(batch, "rgb_grid.png"))
        # nms_filtered = wandb.Image(join(batch, "nms_filtered_masks.png"))
        with open(join(batch, "eval_outputs.pkl"), "rb") as f:
            outputs = pickle.load(f) 

        processed_outputs = process_eval_outputs(outputs)
        used_masks = processed_outputs["used_masks"]        
        with open(join(batch, "used_masks.pkl"), "wb") as f:
            pickle.dump(used_masks, f)
         
        used_cont_masks = processed_outputs["used_cont_masks"]
        with open(join(batch, "used_cont_masks.pkl"), "wb") as f:
            pickle.dump(used_cont_masks, f) 

        iou_stats["all"].append(
            processed_outputs["iou_stats"]["iou_all"]
        )
        iou_stats["nms_filtered"].append(
            processed_outputs["iou_stats"]["iou_filtered"]
        )
        iou_stats["used"].append(
            processed_outputs["iou_stats"]["iou_used"]
        )
        filled_img = processed_outputs["filled_img"]
        Image.fromarray(filled_img).save(join(batch, "merged_preds.png"))
        filled_labels = processed_outputs["filled_labels"]
        Image.fromarray(filled_labels).save(join(batch, "merged_labels.png"))
            
        masks = outputs["repaired_masks"] > 0
        all_masks = [] 
        for idx, (iou_thres, score) in enumerate(thres_tuples):
            concat_masks = [] 
            for i in range(len(masks)):
                if outputs["iou_predictions"][i] > iou_thres and outputs["stability_scores"][i] > score:
                    concat_masks.append(masks[i])  
            concat_masks = np.concatenate(concat_masks, axis=1)
            concat_img = Image.fromarray(concat_masks > 0)
            all_masks.append(wandb.Image(concat_img)) 
        nms_concat = np.concatenate(processed_outputs["nms_filtered"], axis=1)
        row_data = [rgb, wandb.Image(filled_img), wandb.Image(filled_labels), nms_concat] + all_masks
        if args.wandb:
            table.add_data(*row_data) # BUG
        if j == 8:
            exit() # tmp! 
    print('All data ready')
    # compute iou stats
    final_iou_stats = {} 
    for k in iou_stats.keys(): 
        final_iou_stats[k+"_mean"] = np.mean(iou_stats[k])
        final_iou_stats[k+"_std"] = np.std(iou_stats[k])
    print(final_iou_stats)
    with open(join(data_path, "iou_stats.pkl"), "wb") as f:
        pickle.dump(iou_stats, f)
        print(f"Saved iou stats to {join(data_path, 'iou_stats.pkl')}")
    if args.wandb:
        run = wandb.init(project="real2code", name=f"Eval_{data_path}", group="eval")
        wandb.log({"eval_table": table})
    return

def load_3d(args, load_model_fname, filenames):
    data_path = join(args.output_dir, args.run_name, load_model_fname.split("/")[-1])
    batches = natsorted(glob(join(data_path, "batch_*")))
    if len(batches) == 0:
        print(f"No batches found in {data_path}")
        exit()
    assert len(batches) == len(filenames), f"Found {len(batches)} batches but {len(filenames)} filenames"
    print(f"Found {len(batches)} eval batches")  

    outputs = dict()
    for j, batch in enumerate(batches): 
        if j % 8 == 0:
            break
            # outputs = dict() 
        with open(join(batch, "used_masks.pkl"), "rb") as f:
            used_masks = pickle.load(f)
        used_masks = [m.astype(np.uint8) for m in used_masks]
        fname = filenames[j]

        with h5py.File(fname, "r") as f: 
            data = dict()
            for key in ['cam_id', 'cam_pose', 'cam_fov', 'cam_intrinsics', 'binary_masks', 'colors', 'depth']:
                data_key = key 
                if key == 'colors':
                    data_key = 'rgb'  
                data[data_key] = np.array(f[key]) 
                if key == 'cam_fov': 
                    data['fov'] = data['cam_fov'][0] # assume square
                if key == 'cam_pose': 
                    data['pos'] = data['cam_pose'][:3, -1]
                    data['rot_mat'] = data['cam_pose'][:3,:3]
                if key == 'binary_masks':
                    data['segment_body'] = {i: mask for i, mask in enumerate(used_masks)}
                data['fov_metric'] = 'rad'
        # gt_mask = data['binary_masks'][0] # background mask, use to mask depth 
        depth = data['depth'] 
        # depth[mask] = 10
        depth[depth > 10] = 10 # object could still have holes 
        data['depth'] = depth
        cam_name = f"cam_{data['cam_id'][0]}"
        outputs[cam_name] = data
    
    pcds = []
    for cam_name, data in outputs.items():
        pcd = get_merged_pointcloud(
            camera_keys=[cam_name],
            outputs=outputs,
            height=640,
            width=640,
            lower_bound=[-2, -2, -2],
            upper_bound=[2, 2, 2],
        )
        # pcds.append(pcd)
        smoll = pcd.subsample(grid_size=15)
        pcds.append(smoll)
        print('cam_name', cam_name)


def main(args): 
    if args.load_eval:
        print("Loading eval result data")
        _, _, _, load_model_fname = load_sam_model(
            zero_shot=args.zero_shot,
            run_name=args.run_name,
            load_epoch=args.load_epoch,
            load_steps=args.load_steps,
            sam_type=args.sam_type,
            points=args.points,
            model_dir=args.model_dir, 
            skip_load=True
            )
        load_eval_data(args, load_model_fname)
        exit() 
    model, device, forward_fn, load_model_fname = load_sam_model(
            zero_shot=args.zero_shot,
            run_name=args.run_name,
            load_epoch=args.load_epoch,
            load_steps=args.load_steps,
            sam_type=args.sam_type,
            points=args.points,
            model_dir=args.model_dir, 
            skip_load=True
            )
    # load dataset
    original_size = (512, 512) 
    val_dataset = SamH5Dataset(
        args.data_dir,
        loop_id=args.loop_id,
        subsample_camera_views=args.subsample_camera_views,
        transform=get_image_transform(1024, jitter=False, random_crop=False), 
        is_train=0,
        point_mode=True,
        original_img_size=original_size,
        prompts_per_mask=16,
        max_background_masks=2,
        return_gt_masks=True,
        ) 
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
        ) 

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):   
            image = batch["image"].to(device)
            save_path = join(args.output_dir, args.run_name,  load_model_fname.split("/")[-1], f"batch_{i}")
            os.makedirs(save_path, exist_ok=True) 
            outputs = sample_points_eval(
                model=model, 
                device=device, 
                image=image, 
                labels=batch["gt_masks"].detach().cpu().numpy(),
                save_path=save_path,
                num_grid_points=args.num_grid_points,
                original_size=original_size,
                )
            # if i == 8:
            #     break  
    breakpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/store/real/mandi/real2code_dataset_v0") 
    parser.add_argument("--model_dir", type=str, default="/store/real/mandi/sam_models")
    parser.add_argument("--output_dir", type=str, default="/store/real/mandi/sam_eval") # save eval outputs
    parser.add_argument("--sam_type", default="default", type=str) # from the original SAMv1 release
    parser.add_argument("--skip_load","-sl", action="store_true") # skip loading the model for faster testing
    parser.add_argument("--loop_id", default=-1, type=int)
    parser.add_argument("--subsample_camera_views", "-sub", type=int, default=-1)
    parser.add_argument("--run_name", "-rn", default="use_pts_pointsTrue_lr0.0003_bs8_gradstep16_10-10_18-43", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--grad_accum_steps", "-ac", default=16, type=int)
    parser.add_argument("--log_freq", default=50, type=int)
    parser.add_argument("--vis_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--load_epoch", default=-1, type=int)
    parser.add_argument("--load_steps", default=-1, type=int)   
    parser.add_argument("--overwrite", "-o", action="store_true") # if True, overwrite existing output folder
    parser.add_argument("--num_grid_points", "-ng", default=32, type=int) # number of grid points to sample
    parser.add_argument("--load_eval", action="store_true") # if True, load eval data and save to wandb
    parser.add_argument("--zero_shot", action="store_true") # if True, don't load model, just run eval
    args = parser.parse_args()
    main(args)

