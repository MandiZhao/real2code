from __future__ import annotations
import mujoco
from dm_control import mjcf 
from mujoco import viewer
from dm_control import mujoco as dm_mujoco 
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
from tune_sam import SamH5Dataset, get_image_transform, forward_sam, forward_sam_points, get_wandb_table, reset_decoder_head
from render_utils import *
from render_dataset import save_2d_visuals
from render_sam_data import set_cameras, set_max_joint
import seaborn as sns

CKPT_PATH="/home/mandi/sam_vit_h_4b8939.pth"
"""
Assumes blender dataset and points based prompting!

 
RUN=sam_v2_pointsTrue_lr0.001_bs8_ac16_01-19_23-56;  STEP=30000
python sam_test.py --points --run_name $RUN --load_step ${STEP} --num_grid_points 32 --data_dir /local/real/mandi/blender_dataset_v2/ --loop_id 0 -sub 4

- Pool stats
RUN=sam_v2_pointsTrue_lr0.001_bs8_ac16_01-19_23-56;  STEP=30000
python sam_test.py --run_name $RUN --load_step ${STEP} --load_eval --load_obj_type StorageFurniture --load_obj_folder "*" --load_loop_id "*"
"""
def load_sam_model(args, skip_load=False):
    if not args.zero_shot: 
        if len(args.run_name.split("/")) > 1:
            args.run_name = args.run_name.split("/")[-1]
        load_run_dir = join(args.model_dir, args.run_name) # same key as training but now use it for loading instead
        ckpts = natsorted(glob(join(load_run_dir, "ckpt_*pth")))
        print(f"Loading from dir: {load_run_dir}")
        if args.load_epoch == -1:
            ckpt_name = ckpts[-1]
            if args.load_steps > 0:
                ckpt_name = [c for c in ckpts if str(args.load_steps) in c][0]
        else:
            ckpts = natsorted(glob(join(load_run_dir, "ckpt_*pth")))
            ckpt_name = [c for c in ckpts if str(args.load_epoch) in c][0]
        if skip_load:
            return None, None, None, ckpt_name
    else:
        ckpt_name = "ckpt_zero_shot.pth"
    model = sam_model_registry[args.sam_type](checkpoint=CKPT_PATH)
    print("Original SAM Model loaded.")
    forward_fn = forward_sam_points if args.points else forward_sam
    if not args.points:
        print("Resetting decoder head to 9 classes")
        reset_decoder_head(model, new_size=9)
    
    if not args.zero_shot:
        loaded_decoder = torch.load(ckpt_name, map_location="cpu")
        model.mask_decoder.load_state_dict(loaded_decoder)
        print(f"Decoder loaded from {ckpt_name}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("Model moved to device, params frozen, set to eval mode")
    return model, device, forward_fn, ckpt_name

def points_nms_filter(masks, ious, iou_thres=0.7):
    """ filter masks using NMS on points """
    num_masks = len(masks)
    assert len(ious) == num_masks
    b_pool = [(mask, iou) for mask, iou in zip(masks, ious)]
    d_pool = [] 
    while len(b_pool) > 0:
        mask, gt_iou = b_pool.pop(0)
        d_pool.append((mask, gt_iou))
        topop = []
        for i in range(len(b_pool)):
            other_mask = b_pool[i][0]
            intersection = np.sum(mask * other_mask > 0)
            union = np.sum(mask + other_mask > 0)
            iou = intersection / union
            if iou > iou_thres:
                topop.append(i)
        b_pool = [b_pool[i] for i in range(len(b_pool)) if i not in topop] 
    nms_masks = [d[0] for d in d_pool]
    nms_ious = [d[1] for d in d_pool]
    return nms_masks, nms_ious

def find_mask_for_point(point, masks):
    """ find the GT mask that contains the point, use for eval """
    pos_masks = [] 
    for mask in masks:
        if mask[point[0], point[1]] > 0:
            pos_masks.append(mask)
    if len(pos_masks) == 0:
        print('WARNING: no mask found for point')
        return np.zeros(masks[0].shape)
    if len(pos_masks) > 1:
        print(f'WARNING: {len(pos_masks)} masks found for point {point}')
    return pos_masks[0].copy()

def compute_iou(pred, gt):
    """ compute iou between two binary masks """
    intersection = np.sum(pred * gt > 0)
    union = np.sum(pred + gt > 0)
    iou = intersection / union
    return iou

def sample_points_eval( 
    model, 
    device, 
    image, 
    labels, # should contain ALL the GT masks loaded from validation dataset
    save_path,
    num_grid_points=32, 
    num_masks_per_batch=32,
    original_size=(640, 640),
    save_data=False,
):
    """ sample a grid of points on image and group masks """
    assert len(image.shape) == 4 and image.shape[1] == 3, f"image shape {image.shape} is not (batch, 3, H, W)"
    h, w = original_size
    grid_size = int(h/num_grid_points)
    grid_coords = np.mgrid[0:h:grid_size, 0:w:grid_size].reshape(2, -1).T # shape (n, 2) 

    # visualize the grid and save:
    img = image[0].permute(1,2,0).detach().cpu().numpy() # shape (H, W, 3) NOTE: batch['image'] is already resized into 1024x1024
    img = img.astype(np.uint8) 
    Image.fromarray(img).save(join(save_path, "rgb_grid.png"))  

    rgb = model.preprocess(image)
    image_embeddings = model.image_encoder(rgb)
    bs = rgb.shape[0]
    all_outputs = []
    num_batchs = int(np.ceil(grid_coords.shape[0] / num_masks_per_batch))

    for i in range(num_batchs):
        point_coords = grid_coords[i*num_masks_per_batch:(i+1)*num_masks_per_batch]
        gt_masks_for_points = []
        for point in point_coords:
            gt_mask = find_mask_for_point(point, masks=labels) # skip batch dim
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
            input_size=(1024, 1024),
            original_size=original_size,
        ) 
        stability_scores = calculate_stability_score(
            pred[:,0,:,:], mask_threshold=0, threshold_offset=0.6).detach().cpu().numpy() # shape (num_masks_per_batch, )
        pred = pred[:,0,:,:].detach().cpu().numpy() # shape (num_masks_per_batch, H, W)
        masks = pred 
        binary_masks = masks > 0
        bboxes = []
        repaired_masks = []
        for mask in masks:
            mask = (mask > 0).astype(np.uint8) 
            repaired_mask, filled_hole = remove_small_regions(mask, area_thresh=200, mode="holes")
            repaired_mask, filled_hole = remove_small_regions(repaired_mask, area_thresh=200, mode="islands")
            repaired_masks.append(repaired_mask) # this changes from 0-1 to True/False
            pos_mask = np.where(repaired_mask > 0)  
            if len(pos_mask[0]) == 0:
                # print("Got all zero mask")
                bboxes.append([0, 0, h, w])
                continue
            y1, y2 = np.min(pos_mask[0]), np.max(pos_mask[0])
            x1, x2 = np.min(pos_mask[1]), np.max(pos_mask[1])
            bboxes.append([x1, y1, x2, y2]) 
        # compute iou for each point-prediction and its gt_mask
        gt_ious = []
        for mask, gt in zip(binary_masks, gt_masks_for_points):
            gt_ious.append(compute_iou(mask, gt))
        all_outputs.append(dict(
            masks=pred,
            binary_masks=binary_masks,
            repaired_masks=np.stack(repaired_masks, axis=0),
            gt_masks=gt_masks_for_points,
            gt_ious=gt_ious,
            stability_scores=stability_scores,
            bboxes=bboxes, 
            point_coords=point_coords.detach().cpu().numpy(),
            point_labels=point_labels.detach().cpu().numpy(),
            iou_predictions=iou_predictions.squeeze(1).detach().cpu().numpy(),
            low_res_logits=low_res_masks.detach().cpu().numpy(),
        ))
        # TODO: reduce size!! 
    
    # concatenate all the outputs
    final_outputs = dict()
    for k in all_outputs[0].keys():
        final_outputs[k] = np.concatenate([o[k] for o in all_outputs], axis=0) 
    if save_data:
        with open(join(save_path, "eval_outputs.pkl"), "wb") as f: 
            pickle.dump(final_outputs, f)
        print(f"Saved results to {save_path}")
    final_outputs["unique_gt_masks"] = labels # (n, H, W)
    return final_outputs

def get_filled_img(sorted_masks, sorted_ious, threshold=0.999, min_new_pixels=200):
    """ use a list of sorted masks to fill an image in order """
    sorted_masks = deepcopy(sorted_masks)
    sorted_ious = deepcopy(sorted_ious)
    init_mask = np.zeros(sorted_masks[0].shape)
    init_rgb = np.zeros((init_mask.shape[0], init_mask.shape[1], 3))
    threshold_size = int(init_mask.shape[0] * threshold * init_mask.shape[1] * threshold)
    background_color = np.array([255, 255, 255])
    mask_colors = sns.color_palette("colorblind", len(sorted_masks))  
    mask_colors = [np.array(np.array(c) * 255, dtype=np.uint8) for c in mask_colors]
    
    mask_idx = 1
    used_idx = 0
    used_masks = []
    used_idxs = []
    used_ious = []
    while np.sum(init_mask > 0) < threshold_size and len(sorted_masks) > 0:
        curr_mask = sorted_masks.pop(0) > 0
        curr_iou = sorted_ious.pop(0)
        # compare this new mask with current init_mask, and only take the pixels that are currently 0
        new_mask = curr_mask * (init_mask == 0)
        if np.sum(new_mask) < min_new_pixels:
            continue
        used_masks.append(curr_mask)
        used_ious.append(curr_iou)
        used_idxs.append(used_idx)
        used_idx += 1
        # give new_mask a mask idx
        new_mask = new_mask * mask_idx
        # add new_mask to init_mask
        init_mask = init_mask + new_mask
        mask_color = mask_colors[mask_idx - 1]
        # fill the rgb image with the mask color
        rgb_idxs = np.where(new_mask > 0)
        init_rgb[rgb_idxs[0], rgb_idxs[1], :] = mask_color
        mask_idx += 1
    # Image.fromarray(init_rgb.astype(np.uint8)).save('test_filled.png') 
    return init_rgb.astype(np.uint8), used_masks, used_idxs, used_ious

def draw_bbox(mask, bbox, bbox_edge_size=3):
    """ draw a bbox on an image """
    x1, y1, x2, y2 = bbox
    mask[y1:y1+bbox_edge_size, x1:x2] = 1
    mask[y2-bbox_edge_size:y2, x1:x2] = 1
    mask[y1:y2, x1:x1+bbox_edge_size] = 1
    mask[y1:y2, x2-bbox_edge_size:x2] = 1
    return mask

def compute_stats(outputs, iou_thres=0.7, match_iou_thres=0.8):
    masks = outputs["repaired_masks"] #outputs["binary_masks"]
    labels = outputs["gt_masks"]
    unique_gt_masks = outputs["unique_gt_masks"]
    iou_preds = outputs["iou_predictions"]
    stability_scores = outputs["stability_scores"]
    # sort all the masks
    overall_scores = []
    for i in range(len(masks)):
        overall_scores.append(
            np.clip(stability_scores[i], 0, 1) + np.clip(iou_preds[i], 0, 1)
        )
    sorted_idxs = np.argsort(overall_scores)[::-1]
    sorted_masks = [masks[i] for i in sorted_idxs] 
    sorted_gt_ious = [outputs["gt_ious"][i] for i in sorted_idxs]

    # then, use NMS to filter the masks
    nms_filtered, nms_ious = points_nms_filter(sorted_masks, sorted_gt_ious, iou_thres=iou_thres)
    # use the filtered masks to fill an image
    filled_img, used_masks, used_idxs_on_nms, used_ious = get_filled_img(nms_filtered, nms_ious) 

    # match the final used masks to the unique GT masks 
    num_intersect = 0
    for used_mask in used_masks: 
        for gt_mask in unique_gt_masks:
            match_mask = compute_iou(used_mask, gt_mask) > match_iou_thres
            if match_mask:
                num_intersect += 1
                break
    num_union = len(used_masks) + len(unique_gt_masks) - num_intersect
    match_rate = num_intersect / num_union
    stats = dict(
        uniform_iou=np.mean(outputs["gt_ious"]),
        filtered_iou=np.mean(nms_ious),
        used_iou=np.mean(used_ious),
        match_rate=match_rate,
    )
    data = dict(
        # pred_masks=outputs["masks"],
        binary_masks=outputs["binary_masks"],
        used_masks=used_masks, 
    )

    return stats, data
        
def pool_all_stats(args):
    ckpt_name = f"ckpt_step_{args.load_steps}.pth"
    lookup_dir = join(
        args.output_dir, args.run_name, ckpt_name, f"loop_{args.load_loop_id}", args.load_obj_type, args.load_obj_folder, "*", "stats.json")
    stats_fnames = natsorted(glob(lookup_dir))
    print(f"Found {len(stats_fnames)} stats files")
    all_stats = defaultdict(list)
    for fname in stats_fnames:
        with open(fname, "r") as f:
            stats = json.load(f)
        for k, v in stats.items():
            all_stats[k].append(v)
    final_stats = dict()
    for k, v in all_stats.items():
        final_stats[k+"_mean"] = np.mean(v)
        final_stats[k+"_err"] = np.std(v) / np.sqrt(len(v))
    print("Lookup path:", lookup_dir)
    print("Final stats:")
    for k, v in final_stats.items():
        print(f"{k}: {v}")
    return all_stats
        

def main(args): 
    if args.load_eval:
        all_stats = pool_all_stats(args)
        exit() 
    model, device, forward_fn, load_model_fname = load_sam_model(args)
    # load dataset
    original_size = (640, 640)
    if "_v2" in args.data_dir:
        original_size = (512, 512)
    val_dataset = SamH5Dataset(
        args.data_dir,
        loop_id=args.loop_id,
        subsample_camera_views=args.subsample_camera_views,
        transform=get_image_transform(1024, jitter=False, random_crop=False), 
        is_train=0,
        point_mode=args.points,
        original_img_size=original_size,
        prompts_per_mask=16,
        max_background_masks=2,
        return_gt_masks=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,

    ) 
    save_dir = join(args.output_dir, args.run_name, load_model_fname.split("/")[-1], f"loop_{args.loop_id}")
    os.makedirs(save_dir, exist_ok=True)

    all_stats = defaultdict(list)
    for i, batch in enumerate(tqdm(val_loader)):  
        fname = batch["filename"][0] # e.g. /local/real/mandi/blender_dataset_v2/test/Box/100664/loop_2/0.hdf5
        cam_id = fname.split("/")[-1].split(".")[0] # e.g. 0
        obj_type = fname.split("/")[-4] # e.g. Box
        obj_id = fname.split("/")[-3]
        save_path = join(save_dir, obj_type, obj_id, cam_id)
        stats_fname = join(save_path, "stats.json")
        data_fname = join(save_path, "data.pkl")
        if not args.overwrite and os.path.exists(stats_fname) and os.path.exists(data_fname):
            print(f"Skipping batch {i}")
            continue
        with torch.no_grad():  
            image = batch["image"].to(device) 
            os.makedirs(save_path, exist_ok=True)  
            unique_gt_masks = batch["gt_masks"].detach().cpu().numpy()[0]
            outputs = sample_points_eval(
                model=model, 
                device=device, 
                image=image, 
                labels=unique_gt_masks,
                save_path=save_path,
                num_grid_points=args.num_grid_points,
                original_size=original_size,
                )
        stats, data = compute_stats(outputs)
        
        # save stats
        with open(stats_fname, "w") as f:
            json.dump(stats, f)
        with open(data_fname, "wb") as f:
            pickle.dump(data, f)
        # concat the masks 
        concat_used = np.concatenate(data["used_masks"], axis=1)
        Image.fromarray(concat_used).save(join(save_path, "used_masks.png"))
        
        concat_gt = np.concatenate(unique_gt_masks, axis=1) > 0 
        Image.fromarray(concat_gt).save(join(save_path, "gt_masks.png")) 
        
        for k, v in stats.items():
            all_stats[k].append(v) 

    final_stats = dict()
    for k, v in all_stats.items():
        final_stats[k+"_mean"] = np.mean(v)
        final_stats[k+"_err"] = np.std(v) / np.sqrt(len(v))
    # save all_stats
    with open(join(save_dir, "all_stats.json"), "w") as f:
        json.dump(final_stats, f)
    print("Final stats:", final_stats)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/local/real/mandi/blender_dataset_v2")
    parser.add_argument("--xml_dir", type=str, default="/local/real/mandi/mobility_dataset_v1")
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
    parser.add_argument("--log_freq", default=50, type=int)
    parser.add_argument("--vis_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--load_epoch", default=-1, type=int)
    parser.add_argument("--load_steps", default=-1, type=int)
    parser.add_argument("--points", default=True, action="store_true") 
    parser.add_argument("--seg_3d", action="store_true") # if True, load a single object and run SAM to project
    parser.add_argument("--object_folder", "-of", default="StorageFurniture/41003", type=str)
    parser.add_argument("--input_xml_fname", default="merged_v1.xml", type=str)
    parser.add_argument("--num_render_cameras", default=10, type=int)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=480, type=int)
    parser.add_argument("--overwrite", "-o", action="store_true") # if True, overwrite existing output folder
    parser.add_argument("--num_grid_points", "-ng", default=32, type=int) # number of grid points to sample
    parser.add_argument("--load_eval", action="store_true") # if True, load eval data and save to wandb
    parser.add_argument("--load_obj_type", default="*", type=str) # if True, load eval data and save to wandb
    parser.add_argument("--load_obj_folder", default="*", type=str) # if True, load eval data and save to wandb
    parser.add_argument("--load_loop_id", default="0", type=str) # if True, load eval data and save to wandb
    parser.add_argument("--zero_shot", action="store_true") # if True, don't load model, just run eval

    parser.add_argument("--iou_thres", default=0.7, type=float)
    parser.add_argument("--match_iou_thres", default=0.8, type=float)
    args = parser.parse_args()
    main(args)

