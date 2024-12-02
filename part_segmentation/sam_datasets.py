import os
from os.path import join
from glob import glob
from natsort import natsorted
import wandb 
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, BatchSampler 
import argparse
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling.mask_decoder import MLP as MaskDecoderMLP
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
import torch.nn.functional as F
from einops import rearrange, repeat
import h5py
from collections import defaultdict

class SamH5Dataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        loop_id=-1,
        obj_type="*",
        obj_folder="*",
        transform=None, 
        is_train=True, 
        split=0.9, 
        num_masks=9, 
        point_mode=True, 
        prompts_per_mask=8, 
        seed=42, 
        subsample_camera_views=-1, 
        grid_size=1, 
        original_img_size=(640, 640),
        input_img_size=(1024, 1024),
        max_background_masks=-1, # if > 0, cap the number of background masks in a batch
        use_cache=False,
        return_gt_masks=False,
        filter_masks=True,
        topcrop=0,
        bottomcrop=0,
        rebuttal_objects_only=False, # if True, only use Scissors Eyeglasses
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.return_gt_masks = return_gt_masks
        self.split = split 
        self.filter_masks = filter_masks
        self.filenames, self.object_ids, self.filename_to_nmask, self.obj_to_nmask = \
            self.load_filenames(
                root_dir, is_train, split, subsample_camera_views, obj_type, obj_folder, loop_id,
                rebuttal_objects_only=rebuttal_objects_only
                )
        self.num_masks = num_masks
        self.point_mode = point_mode
        self.prompts_per_mask = prompts_per_mask
        self.seed = seed
        self.np_random = np.random.RandomState(seed) 
        self.original_img_size = original_img_size
        self.topcrop = topcrop
        self.bottomcrop = bottomcrop
        self.input_img_size = input_img_size
        self.set_grid_size(grid_size)
        self.max_background_masks = max_background_masks
        self.use_cache = use_cache 
        self.cache = dict() # cache the loaded images and masks
        self.total_num_masks = 0
        self.idx_to_obj = []
        
        if self.use_cache:
            print('Caching images and masks...')
            for idx in range(len(self.filenames)):
                filename = self.filenames[idx] 
                # OLD: load all data, bad for num_workers > 1
                # with h5py.File(filename, "r") as f:
                #     rgb = np.array(f['colors']).astype(np.float32)
                #     masks = np.array(f['binary_masks']).astype(np.float32) 
                # self.cache[idx] = dict(
                #     rgb=rgb,
                #     masks=masks,
                # )
                # only load num_masks!                
                # n_masks = masks.shape[0] # at least 2, one for background, one for object
                # assert n_masks > 1, "Must have at least 2 masks!"
                n_masks = self.filename_to_nmask[filename]
                self.total_num_masks += n_masks - 1
                # new sampling strategy: each rgb appears n_mask-1 times in the dataset
                self.idx_to_obj.extend([idx] * (n_masks-1))
            assert len(self.idx_to_obj) == self.total_num_masks, "Something wrong with the caching!"
            print(f"Done caching images and masks! change dataset length to {self.total_num_masks}")

    def set_grid_size(self, grid_size):
        h, w = self.original_img_size
        self.grid_size = grid_size 
        self.grid_coords = np.mgrid[0:h:grid_size, 0:w:grid_size].transpose(1,2,0).reshape(2, -1).T # shape (n, 2)
        self.grid_map = np.zeros((w, h), dtype=np.uint8) 
        # self.grid_map[self.grid_coords[:,0], self.grid_coords[:,1]] = 1 # NOTE not used
        return 

    def load_filenames(
            self, 
            root_dir, 
            is_train=True, 
            split=0.9, 
            subsample_camera_views=-1, 
            lookup_obj_type="*", 
            lookup_obj_folder="*", 
            loop_id=-1,
            rebuttal_objects_only=False
        ):
        """ 
        load each loop's hdf5 files, assume data ataset_v0)
            - StorageFurniture
                - folder_id (e.g. 35058)
                    - loop_id (e.g. 0)
                        - 0.hdf5 # contains 1 rgb + all masks
                    - loop_id (e.g. 1)
                - folder_id (e.g. 35059)
            - Table
            - ...   
        """
        filenames = []
        filename_to_nmask = dict()
        obj_to_nmask = dict()
        object_ids = dict()
        total_objs = 0
        is_old = 'blender_dataset_v0' in root_dir  or "blender_dataset_v1" in root_dir 
        if is_train and not is_old:
            root_dir = join(root_dir, 'train')
        elif not is_train and not is_old:
            root_dir = join(root_dir, 'test')
        for obj_type in natsorted(os.listdir(root_dir)):
            if lookup_obj_type != "*" and obj_type != lookup_obj_type:
                continue 
            if rebuttal_objects_only and obj_type not in ['Scissors', 'Eyeglasses']:
                continue
            obj_filenames = defaultdict(list)
            for obj_folder in natsorted(os.listdir(join(root_dir, obj_type))):
                if lookup_obj_folder != "*" and obj_folder != lookup_obj_folder:
                    continue
                obj_folder_dir = join(root_dir, obj_type, obj_folder)
                if loop_id == -1:
                    lookup_path = join(obj_folder_dir, 'loop_*')
                else:
                    lookup_path = join(obj_folder_dir, f'loop_{loop_id}')
                for _id in natsorted(glob(lookup_path)):
                    loop_dir = join(obj_folder_dir, _id) 
                    txt_fname = join(loop_dir, "num_masks.txt")
                    hdf5_fnames = natsorted(glob(join(loop_dir, "*.hdf5")))
                    if len(hdf5_fnames) > 0:
                        assert os.path.exists(txt_fname), f"Missing {txt_fname}"
                        with open(txt_fname, "r") as f:
                            # each line records number of masks in each hdf5 file f"{fname} {num_masks}"
                            txt_lines = f.readlines()
                            assert len(txt_lines) == len(hdf5_fnames), f"Number of txt lines {len(txt_lines)} != number of hdf5 files {len(hdf5_fnames)}"
                            for line in txt_lines:
                                fname, num_masks = line.strip().split()
                                num_masks = int(num_masks)
                                assert num_masks >= 1, "Must have at least 1 masks!"
                                filename_to_nmask[fname] = num_masks 
                        if subsample_camera_views > 0:
                            hdf5_fnames = hdf5_fnames[::subsample_camera_views]
                        obj_filenames[obj_folder].extend(hdf5_fnames)
                        obj_to_nmask[obj_folder] = num_masks
                        # if num_masks > 5 and is_train:
                        #     # HACK! add this object one more time to make it more likely to be sampled
                        #     obj_filenames[f"{obj_folder}01"].extend(hdf5_fnames)

            # determiniscally shuffle filenames
            random.seed(42)
            obj_folder_keys = list(obj_filenames.keys())
            random.shuffle(obj_folder_keys) 
            if len(obj_folder_keys) > 0:
                if 'blender_dataset_v1' in root_dir:
                    print(f"Using blender_dataset_v1 which contains only unseen object. Setting split to 1 {split}")
                    split = 1
                    is_train = True
                
                num_objs = int(len(obj_folder_keys) * split)
                num_objs = max(num_objs, 1)
                num_objs = min(num_objs, len(obj_folder_keys)) 
                if is_train:
                    obj_folders = obj_folder_keys[:num_objs]
                else:
                    obj_folders = obj_folder_keys[num_objs:]
                
                if not is_old: # dataset already split into train/test
                    obj_folders = obj_folder_keys
                

                for obj_folder in obj_folders:
                    filenames.extend(obj_filenames[obj_folder])
                total_objs += len(obj_folders)
                object_ids[obj_type] = obj_folders
        # TODO: fix this so the val set contains only novel objects!!! 
        print(f"Train Set? {is_train} - Found {total_objs} distinct objects, in total {len(filenames)} image-all_its_mask pairs") 
        return filenames, object_ids, filename_to_nmask, obj_to_nmask
    
    def __len__(self):
        if self.use_cache:
            return self.total_num_masks
        return len(self.filenames)
    
    def compute_num_masks(self, save_histogram=False):
        num_masks = []
        for filename in self.filenames:
            with h5py.File(filename, "r") as f:
                masks = np.array(f['binary_masks'])
                num_masks.append(masks.shape[0])
                # if masks.shape[0] > 10:
                #     print(filename)
        if save_histogram:
            import matplotlib.pyplot as plt
            plt.hist(num_masks, bins=20)
            # set x-axis labels and ticks
            plt.xlabel('Number of masks')
            plt.xticks(range(1, 20))
            # set y label on top of each bar:
            for i in range(1, 20):
                plt.text(i, num_masks.count(i)+1, str(num_masks.count(i)))
            plt.savefig(f"{'train' if save_histogram else 'val'}_num_masks_historgram.png")
        return num_masks 

    def center_crop(self, rgb, masks):
        if self.topcrop > 0 or self.bottomcrop > 0:
            h, w = rgb.shape[:2]
            rgb = rgb[:, self.topcrop: w-self.bottomcrop, :]
            masks = masks[:, :, self.topcrop:w-self.bottomcrop]
        return rgb, masks
    
    def __getitem__(self, idx):
        if self.use_cache:
            # rgb_idx = self.idx_to_obj[idx]
            # data = self.cache[rgb_idx]
            # rgb = data['rgb']
            # masks = data['masks']
            idx = self.idx_to_obj[idx] # get the idx of the rgb image
        filename = self.filenames[idx] 
        with h5py.File(filename, "r") as f:
            rgb = np.array(f['colors']).astype(np.float32)
            masks = np.array(f['binary_masks']).astype(np.float32) 
            
        rgb, masks = self.center_crop(rgb, masks)
        gt_masks = masks.copy()

        if not self.point_mode:
            masks = masks[:self.num_masks] # cap the number of masks    
        if len(masks) < self.num_masks and not self.point_mode:
            masks = np.concatenate(
                [masks, np.zeros((self.num_masks - len(masks), masks.shape[1], masks.shape[2]))]
            ) # pad with zeros
        

        rgb, masks, original_size = self.transform(rgb, masks) # mask is torch float32 now
        if self.filter_masks:
            # filter out near empty masks 
            masks = torch.stack([mask for mask in masks if torch.sum(mask) > 0], axis=0)
        if self.point_mode:
            point_coords, point_labels = [], [] 
            if self.max_background_masks > 0:
                num_bg_masks = self.np_random.choice(self.max_background_masks, size=1, replace=False)[0]
                mask_idxs = self.np_random.choice( 
                    range(1, len(masks)), size=self.prompts_per_mask - num_bg_masks, replace=True).tolist()
                mask_idxs = [0] * num_bg_masks + mask_idxs
            else:
                mask_idxs = self.np_random.choice(len(masks), size=self.prompts_per_mask, replace=True)  
            # count number of unique masks, and how many points to sample from them
            unique_idxs, unique_counts = np.unique(mask_idxs, return_counts=True)
            tostack_masks = []
            for _idx, _count in zip(unique_idxs, unique_counts):
                mask = masks[_idx] 
                if np.where(mask > 0)[0].shape[0] == 0:
                    breakpoint()
                coords, labels = self.sample_points(
                    mask, 
                    num_points=_count,
                    mode="grid" if self.grid_size > 1 else "uniform"
                )
                point_coords.append(coords)
                point_labels.append(labels)
                tostack_masks.extend(
                    [mask.clone() for _ in range(_count)]
                )
            point_coords = np.concatenate(point_coords, axis=0)
            point_labels = np.concatenate(point_labels, axis=0)
            masks = torch.stack(tostack_masks, axis=0) # stack the repeated masks  
        
            output = dict(
                image=rgb,
                masks=masks,
                point_coords=point_coords,
                point_labels=point_labels,
            )

        else:
            output = dict(
                image=rgb,
                masks=masks,
                # original_size=original_size,
            )
        if self.return_gt_masks:
            output['gt_masks'] = gt_masks # strict same # as GT masks
            output['filename'] = filename
        return output
    
    def sample_grid_points(self, mask, num_points=1):
        assert self.grid_size > 1, "Must use this in grid mode!" 
        mask_arr = mask.numpy()
        # compute intersection with self.grid_map:
        intersect = self.grid_map * mask_arr
        pos_points = np.where(intersect > 0)
        # if no points, sample from the whole image
        if len(pos_points[0]) == 0:
            intersect = self.grid_map * (mask_arr >= 0)
            pos_points = np.where(intersect > 0)
        tosample_coords = np.stack([pos_points[0], pos_points[1]], axis=1) # shape num_points, 2

        sampled_idxs = self.np_random.choice(len(tosample_coords), size=num_points, replace=(len(tosample_coords) < num_points))
        coords = tosample_coords[sampled_idxs] # shape num_points, 2
        
        labels = mask_arr[coords[:,0], coords[:,1]] # shape num_points
        coords = coords[:, None, :] # shape num_points, 1, 2
        labels = labels[:, None] # shape num_points, 1
        return coords, labels

    def sample_points(self, mask, num_points, mode="uniform"):
        """ instead of grid, just sample uniformly from positive points on the mask"""
        mask_arr = mask.numpy()
        if mode ==  "uniform":
            pos_points = np.where(mask_arr > 0)
            # if no points, sample from the whole image
            if len(pos_points[0]) == 0:
                if self.filter_masks:
                    raise ValueError("Shouldn't have empty masks if filter_masks is True!")
                pos_points = np.where(mask_arr >= 0)
        elif mode == "grid":
            assert self.grid_size > 1, "Must use this in grid mode!"
            # compute intersection with self.grid_map:
            intersect = self.grid_map * mask_arr
            pos_points = np.where(intersect > 0)
            if len(pos_points[0]) == 0:
                # intersect = self.grid_map * (mask_arr >= 0)
                # pos_points = np.where(intersect > 0) 
                pos_points = np.where(mask_arr >= 0) # directly sample from mask! doesn't work well if just sample from negative grid points
        else:
            raise NotImplementedError
        
        tosample_coords = np.stack([pos_points[0], pos_points[1]], axis=1) # shape num_points, 2
        sampled_idxs = self.np_random.choice(len(tosample_coords), size=num_points, replace=(len(tosample_coords) < num_points))
        coords = tosample_coords[sampled_idxs] # shape num_points, 2
        labels = mask_arr[coords[:,0], coords[:,1]] # shape num_points
        
        # # visualize the sampled points
        # mask_cp = np.array(deepcopy(mask).numpy()[:, :, None].repeat(3, axis=2) * 255, dtype=np.uint8)
        # for coord in coords:
        #     x = int(coord[0].item())
        #     y = int(coord[1].item())
        #     mask_cp[x:x+8, y:y+8] = np.array([255,0,0])
        # Image.fromarray(mask_cp).save(f"test_sample_points.png")

        coords = coords[:, None, :] # shape num_points, 1, 2
        labels = labels[:, None] # shape num_points, 1
        return coords, labels

    def sample_one_point(self, mask):
        pos_points = np.where(mask > 0)  
        # if no points, sample from the whole image
        if len(pos_points[0]) == 0:
            pos_points = np.where(mask >= 0)  
        pos_coords = np.stack([pos_points[0], pos_points[1]], axis=1) # shape num_points, 2 
        if self.grid_size > 1:
            pos_grid_bbox = np.array([np.min(pos_coords, axis=0), np.max(pos_coords, axis=0)]) # shape 2, 2
            x1, y1, x2, y2 = pos_grid_bbox[0,0], pos_grid_bbox[0,1], pos_grid_bbox[1,0], pos_grid_bbox[1,1]
            pos_grid_coords = self.grid_coords[(self.grid_coords[:,0] >= x1) & (self.grid_coords[:,0] <= x2) & (self.grid_coords[:,1] >= y1) & (self.grid_coords[:,1] <= y2)]
            # only sample from the pre-defined grid of points  TOO SLOW
            # pos_grid_coords = [coord for coord in self.grid_coords.tolist() if coord in pos_coords.tolist()]
             
            if len(pos_grid_coords) > 0:
                point_idx = self.np_random.choice(len(pos_grid_coords), size=1, replace=False)[0]
                coord = pos_grid_coords[point_idx] 
                coord = np.array(coord)[None, :] # shape 1, 2
            else: # sample from mask
                point_idxs = self.np_random.choice(len(pos_points[0]), size=1, replace=False)
                coord = np.stack([pos_points[0][point_idxs], pos_points[1][point_idxs]], axis=1) # shape 1, 2   

        else:
            point_idxs = self.np_random.choice(len(pos_points[0]), size=1, replace=False)
            coord = np.stack([pos_points[0][point_idxs], pos_points[1][point_idxs]], axis=1) # shape 1, 2   
        label = mask[coord[:,0], coord[:,1]]
        return coord, label

    def get_obj_weights(self):
        """ weight sampling probability by the number of masks of each object"""
        num_masks = [v for v in self.filename_to_nmask.values()] 
        obj_weights = np.array(num_masks) / np.sum(num_masks)
        return obj_weights
    
# if __name__ == "__main__":
    # from finetune_sam import get_image_transform, forward_sam, forward_sam_points, get_loss_fn, get_wandb_table, reset_decoder_head
    # transform_fn = get_image_transform(1024, jitter=False, random_crop=False)
    # dataset = SamH5Dataset(
    #     root_dir="/local/real/mandi/blender_dataset_v4", 
    #     transform=transform_fn,
    #     is_train=1,
    #     num_masks=8,
    #     point_mode=1,
    #     prompts_per_mask=8,
    #     grid_size=32,
    #     split=0.9, 
    #     max_background_masks=3,
    #     use_cache=1,
    #     filter_masks=True,
    #     original_img_size=(512,512)
    # )
    # for i in range(1000):
    #     batch = dataset[i] 
    # dataset = SamH5Dataset(
    #     root_dir="/local/real/mandi/scanner_dataset",
    #     loop_id=0,
    #     subsample_camera_views=1,
    #     transform=get_image_transform(1024, jitter=False, random_crop=False), 
    #     is_train=0,
    #     point_mode=True,
    #     original_img_size=(1440, 1440),
    #     prompts_per_mask=16,
    #     max_background_masks=2,
    #     return_gt_masks=True,
    #     obj_type='cab',
    #     obj_folder='1',
    #     topcrop=240,
    #     bottomcrop=240,
    # )
    # breakpoint()

#     from tune_sam import MobilityDataset, get_image_transform, forward_sam, forward_sam_points, get_loss_fn, get_wandb_table, reset_decoder_head
#     transform_fn = get_image_transform(1024, jitter=False, random_crop=False)
#     for _ in range(3):
#         dataset = SamH5Dataset(
#             root_dir="/local/real/mandi/blender_dataset_v0", 
#             transform=transform_fn,
#             is_train=0,
#             num_masks=9,
#             point_mode=1,
#             prompts_per_mask=16,
#             grid_size=32,
#             split=0.9, 
#             max_background_masks=3,
#             use_cache=1,
#         )
#         obj_ids = dataset.object_ids
#         for k, v in obj_ids.items():
#             print(k, natsorted(v))

 
#     breakpoint()
#     for i in range(3):
#         batch = dataset[i] 
#         coords = batch['point_coords']
#         masks = batch['masks'].numpy()
#         for j, (coord, mask) in enumerate(zip(coords, masks)):
#             mask_cp = deepcopy(mask)
#             # convert to 3-channel rgb::
         
#             mask_cp = np.array(mask_cp[:, :, None].repeat(3, axis=2) * 255, dtype=np.uint8)
#             x = int(coord[0,0].item())
#             y = int(coord[0,1].item())
#             mask_cp[x:x+8, y:y+8] = np.array([255,0,0])
#             Image.fromarray(mask_cp).save(f"test_{i}_{j}.png")
#         # get all the value in the mask that's not 0 or 1 
#         # vals = masks[(masks != 0) & (masks != 1)]
#         # print(np.unique(vals))
#         # print(f"num of non-binary values: {len(vals)}")
#         # concat_masks = np.concatenate(masks, axis=0)
#         # Image.fromarray(concat_masks > 0).save('test_masks.png')
#     breakpoint()
#     num_masks = dataset.compute_num_masks(save_histogram=0)
#     print(np.mean(num_masks), np.std(num_masks), np.min(num_masks), np.max(num_masks))
#     exit()
#     loss_fn = get_loss_fn()
#     loss_fn_min = get_loss_fn(min_loss=1)
#     loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
#     model = sam_model_registry['default'](checkpoint="/home/mandi/sam_vit_h_4b8939.pth")
#     # freeze!
#     for param in model.parameters():
#         param.requires_grad = False
#     print("Model loaded")
#     reset_decoder_head(model, new_size=9)
#     print("Decoder reset")
   
#     device = torch.device("cuda")
#     model = model.to(device)
#     for batch in loader:
#         outputs = forward_sam(model, batch, device, (640, 640))
#         total_loss, fc_loss, dc_loss, iou_loss = loss_fn(
#             outputs["pred"], outputs["iou_predictions"], batch["masks"].to(device)
#             )
#         print(f"without min: total_loss: {total_loss}, fc_loss: {fc_loss}, dc_loss: {dc_loss}, iou_loss: {iou_loss}")
        
#         total_loss, fc_loss, dc_loss, iou_loss = loss_fn_min(
#             outputs["pred"], outputs["iou_predictions"], batch["masks"].to(device)
#             )
#         print(f"with min: total_loss: {total_loss}, fc_loss: {fc_loss}, dc_loss: {dc_loss}, iou_loss: {iou_loss}")
#         breakpoint()
#     # dataset = SamPointsDataset(
#     #     "/local/real/mandi/sam_dataset_v2", transform=transform_fn, 
#     #     is_train=1, split=0.9, filter_masks=True
#     # )
#     # data = dataset[0]
#     # dataset_no_filter = SamPointsDataset(
#     #     "/local/real/mandi/sam_dataset_v1", transform=transform_fn, 
#     #     is_train=1, split=0.9, filter_masks=False
#     # )
#     breakpoint()
#     exit()

#     loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
#     device = torch.device("cuda")
#     model = model.to(device)
#     loss_fn = get_loss_fn()
#     for i, batch in enumerate(loader):
#         outputs = forward_sam_points(model, batch, device)
#         preds = outputs["pred"]
#         iou_preds = outputs["iou_predictions"]
#         labels = batch["masks"].to(device)
#         total_loss, fc_loss, dc_loss, iou_loss = loss_fn(preds, iou_preds, labels)
#         print(total_loss, fc_loss, dc_loss, iou_loss)
#         table = get_wandb_table(batch, preds, mask_threshold=0)
#         breakpoint()