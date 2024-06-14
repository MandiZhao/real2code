import os 
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image
from glob import glob
from natsort import natsorted
import argparse
import h5py
import json 
from os.path import join
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import calculate_stability_score, remove_small_regions
import imageio 
"""
convert raw data from 3D iPhone scanner app to h5 format matching the rendered sim dataset!
NOTE: depth image is much smaller, i.e. after segmentation, need to resize the RGB image 
NOTE: cv2.imread(cv2.IMREAD_UNCHANGED) reads depth image as the raw shape, but cv2.imread reads RGB gets the desired rotation, need to handle this 
inp: depth, rgb images, json files containing camera info
out: h5 file with depth, rgb, camera info, color img should be center-cropped to square (1440,1440) and depth to (192, 192)
"""
DEPTH_SCALE=192
RGB_SCALE=1440
SELECT_IDXS={
    "1_cab": dict(
        idxs=[0, 32, 44, 56, 70, 82, 100, 124],
        kwargs=dict(upper=1, lower=1, left=1, margin=180, stepsize=200),
        crop=dict(top=200, bottom=0, left=50), 
    ),
    "2_cab": dict(
        idxs=[7, 19, 38, 57, 127, 146, 158, 179, 212, 327],
        kwargs=dict(left=1,right=1, margin=250, stepsize=50),
        crop=dict(top=10, bot=1400, left=100),
    ),
    "3_cab": dict(
        idxs=[19, 25, 57, 70, 103, 116, 134, 146, 172, 222,285],
        kwargs=dict(right=1, lower=1, upper=1, margin=300, stepsize=250),
        crop=dict(top=20, bot=1400, left=20),
    ),
    "4_laptop": dict(
        idxs=[ 38, 44, 50, 56, 62, 74, 86, 99, 137, 163,177 ],
        kwargs=dict(left=1, upper=1, lower=1, margin=100, stepsize=250),
        crop=dict(top=100, left=20, right=1400),
    ),
    "5_laptop": dict(
        idxs=[0, 18, 24, 30, 42, 55, 67, 73, 79, 85, 92],
        kwargs=dict(left=1,right=1, lower=1, margin=150, stepsize=600), 
        crop=dict(top=150, left=20, right=1370),
    ),
    "6_laptop": dict(
        idxs=[13, 32, 44, 50, 56, 69, 88, 94],
        kwargs=dict(left=1,right=1, lower=1, margin=150, stepsize=600), 
        crop=dict(top=150, left=20, right=1400),
    ),
    "7_fridge": dict(
        idxs=[20, 44, 57, 63, 76, 94, 106, 127, 145, 157, 194],
        kwargs=dict(left=1, right=1, upper=1, lower=1, margin=250, stepsize=300),
        crop=dict(top=400, bot=1500, left=150, right=1250),
    ),
    "8_fridge": dict(
        idxs=[21, 52, 58, 77, 84, 90, 96, 109, 121, 140, 164, 202],
        kwargs=dict(right=1, upper=1, lower=1, margin=250, stepsize=350),
        crop=dict(top=200, bot=1700, left=50, right=1400),
    ),
    "9_cab": dict(
        idxs=[0, 13, 33, 52, 64, 78, 90, 102, 126, 145,158,  266 ],
        kwargs=dict(upper=1, lower=1, margin=250, stepsize=200),
        crop=dict(top=200, bot=1700, left=50, right=1400),
    ),
    "10_cab": dict(
        idxs=[37, 49, 55, 62, 68, 80, 93, 112, 131, 149, 168, ],
        kwargs=dict(upper=1, lower=1, margin=250, stepsize=250),
        crop=dict(top=200, bot=1700, left=50, right=1400),
    ),
    "11_cab": dict(
        idxs=[12, 18, 37, 56, 102, 159, 172, 223, 236, 256,],
        kwargs=dict(upper=1, lower=1, margin=250, stepsize=300),
        crop=dict(top=300, bot=1700, left=50, right=1400),
    ),
    # obj_id = '13_cab'
# idxs = [6, 19, 31, 61, 75, 88, 136, 174, 208, 239, 245, 282, 315 ]

# obj_id = '14_stand'
# idxs = [38, 57, 77, 102, 115, 135, 155, 162, 180,  186, 220, 246 ]

# obj_id = '15_dresser'
# idxs = [6, 30, 36, 42, 74, 86, 96, 124, 144, 163 ]
    "13_cab": dict(
        idxs=[6, 19, 31, 61, 75, 88, 136, 174, 208, 239, 245, 282, 315],
        kwargs=dict(upper=1, lower=1, margin=200, stepsize=200),
        crop=dict(top=80, bot=1700, left=50, right=1880),
    ),
    "14_stand": dict(
        idxs=[38, 57, 77, 102, 115, 135, 155, 162, 180,  186, 220, 246],
        kwargs=dict(upper=1, lower=1, margin=250, stepsize=200),
        crop=dict(top=200, bot=1700, left=200, right=1400),
    ),
    "15_dresser": dict(
        idxs=[6, 30, 36, 42, 74, 86, 96, 124, 144, 163],
        kwargs=dict(upper=1, lower=1, margin=150, stepsize=200),
        crop=dict(top=300, bot=1600, left=50, right=1400),
    ),
}

def load_sam(sam_checkpoint="/home/mandi/sam_vit_h_4b8939.pth"):
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("Loaded SAM model zero shot")
    predictor = SamPredictor(sam)
    return predictor

def read_rgb(fname):
    image_raw = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    image = cv2.imread(fname)
    need_rotate_back = (image.shape != image_raw.shape) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    return image, need_rotate_back

def predict_process_masks(predictor, image, prompt, label, topcrop=0, botcrop=0, leftcrop=0, rightcrop=0):
    predictor.set_image(image)
    input_point = np.array(prompt) # N, 2
    input_label = np.array(label) # N, 1
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    mask = masks[0]
    mask, _ = remove_small_regions(mask, area_thresh=800, mode='holes')
    mask, _ = remove_small_regions(mask, area_thresh=800, mode='islands')
    if topcrop > 0:
        new_image = np.zeros(image.shape).astype(image.dtype)
        new_image[topcrop:, :, :] = image[topcrop:, :, :].copy()
        image = new_image
        new_mask = np.ones(mask.shape).astype(mask.dtype)
        new_mask[topcrop:, :] = mask[topcrop:, :].copy()
        mask = new_mask
    if botcrop > 0:
        new_image = np.zeros(image.shape).astype(image.dtype)
        new_image[:botcrop, :, :] = image[:botcrop, :, :].copy()
        image = new_image
        new_mask = np.ones(mask.shape).astype(mask.dtype)
        new_mask[:botcrop, :] = mask[:botcrop, :].copy()
        mask = new_mask    
    if leftcrop > 0:
        new_image = np.zeros(image.shape).astype(image.dtype)
        new_image[:, leftcrop:, :] = image[:, leftcrop:, :].copy()
        image = new_image
        new_mask = np.ones(mask.shape).astype(mask.dtype)
        new_mask[:, leftcrop:] = mask[:, leftcrop:].copy()
        mask = new_mask
    if rightcrop > 0:
        new_image = np.zeros(image.shape).astype(image.dtype)
        new_image[:, :rightcrop, :] = image[:, :rightcrop, :].copy()
        image = new_image
        new_mask = np.ones(mask.shape).astype(mask.dtype)
        new_mask[:, :rightcrop] = mask[:, :rightcrop].copy()
        mask = new_mask
 
    masked_image = image.copy()
    masked_image[mask > 0] = 0 
    
    return image, mask, masked_image

def sample_bg_prompts(
    image_shape, stepsize=200, upper=False, lower=False, left=False, right=False, margin=150
):
    """ sample randomly along the 4 edges of the image """
    h, w = image_shape[:2]
    prompt = [] 
    for i in range(10, w, stepsize):
        if upper:
            off = np.random.randint(2, margin) 
            prompt.append([i, off])
        if lower:
            off = np.random.randint(2, margin)
            prompt.append([i, h-off])
    for i in range(10, h, stepsize):
        if left:
            off = np.random.randint(2, margin)
            prompt.append([off, i])
        if right:
            off = np.random.randint(2, margin)
            prompt.append([w-off, i])
    prompt = np.array(prompt)
    labels = np.ones(prompt.shape[0]) # NOTE this should be 1 for foreground!!
    return prompt, labels

def show_plt_result(image, mask, masked_image, prompt):
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(mask)
    axs[1].imshow(image)
    axs[1].scatter(prompt[:, 0], prompt[:, 1], c='r', s=80)
    axs[2].imshow(masked_image)
    plt.show()         

def get_bg_mask(predictor, obj_dir, params): 
    idxs = params["idxs"]
    kwargs = params["kwargs"]
    crop = params.get("crop", dict())
    topcrop = crop.get("top", 0)
    botcrop = crop.get("bot", 0)
    transpose = params.get("transpose", False)
    
    rgb_files = natsorted(glob(os.path.join(obj_dir, f'frame_*.jpg')))
    filtered_rgb_files = []
    for fname in rgb_files:
        idx = int(fname.split('_')[-1].split('.')[0])
        if idx in idxs:
            filtered_rgb_files.append(fname)
    rgb_files = filtered_rgb_files
    print(f"Found {len(rgb_files)} images")
    all_rgb_masks = dict()
    for fname in rgb_files:
        image, need_rotate_back = read_rgb(fname) # (1920, 1440, 3)  
        prompt, label = sample_bg_prompts(image.shape, **kwargs)
        image, mask, masked_image = predict_process_masks(
            predictor, image, prompt, label, topcrop=topcrop, botcrop=botcrop, 
            leftcrop=crop.get("left", 0), rightcrop=crop.get("right", 0)
        )
        # save masks
        all_rgb_masks[fname] = (image, mask, masked_image, need_rotate_back)
    return all_rgb_masks

def load_depth_camera(transpose, rgb_fname):
    depth_fname = rgb_fname.replace("frame", "depth")
    depth_fname = depth_fname.replace(".jpg", ".png")
    depth = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000.0  
    json_file = json.load(open(rgb_fname.replace(".jpg", ".json")))
    intrinsics = np.asarray(json_file['intrinsics']).reshape([3, 3])
    projectionMatrix = np.asarray(json_file['projectionMatrix']).reshape([4, 4])
    cameraPoseARFrame = np.asarray(json_file['cameraPoseARFrame']).reshape([4, 4])
    
    return depth, intrinsics, projectionMatrix, cameraPoseARFrame

def center_crop_images(depth, image, crop_size_depth=(192,192), crop_size_rgb=(1440,1440)):
    """ crop (256,192) or (192,256) into (192,192); crop (1920,1440) or (1440, 1920) into (1440,1440)"""
    if depth.shape[0] > depth.shape[1]:
        assert image.shape[0] > image.shape[1], f"For depth {depth.shape}, image should be (1920, 1440), got {image.shape}"
        new_depth = depth[32:224, :]
        new_image = image[240:1680, :, :]
    else:
        assert image.shape[0] < image.shape[1], f"For depth {depth.shape}, image should be (1440, 1920), got {image.shape}"
        new_depth = depth[:, 32:224]
        new_image = image[:, 240:1680, :]
    assert new_depth.shape == crop_size_depth, f"Expected {crop_size_depth}, got {new_depth.shape}"
    assert new_image.shape == crop_size_rgb, f"Expected {crop_size_rgb}, got {new_image.shape}"
    return new_depth, new_image
    
def run(args):
    obj_dirs = natsorted(glob(join(args.input_dir, args.obj_lookup)))
    print(f"Found {len(obj_dirs)} objects")
    if len(obj_dirs) == 0:
        print(f"Object {args.obj_lookup} not found in {args.input_dir}")
        return
    predictor = load_sam(sam_checkpoint="/home/mandi/sam_vit_h_4b8939.pth")
    for obj_dir in obj_dirs:
        obj_name = os.path.basename(obj_dir)
        obj_type = obj_name.split('_')[1]
        obj_id = obj_name.split('_')[0]
        print(f"Processing {obj_id}")
        params = SELECT_IDXS.get(obj_name, None)
        assert params is not None, f"Object {obj_name} not found in SELECT_IDXS"
        out_dir = join(args.output_dir, 'test', obj_type, obj_id, 'loop_0')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        elif args.overwrite:
            print(f"Overwriting {out_dir}")
            os.system(f"rm -r {out_dir}")
            os.makedirs(out_dir)
        else:
            print(f"Skipping {out_dir}")
            continue
        all_rgb_masks = get_bg_mask(predictor, obj_dir, params)
        transpose = params.get("transpose", False)
        txt_towrite = []
        for rgb_fname, (image, mask, masked_image, need_rotate_back) in all_rgb_masks.items():
            depth, intrinsics, projectionMatrix, cameraPoseARFrame = load_depth_camera(transpose, rgb_fname)
            # scale depth!
            intrinsics *= DEPTH_SCALE / RGB_SCALE 
            
            img_id = int(rgb_fname.split('_')[-1].split('.')[0])
            h5_fname = join(out_dir, f"{img_id}.hdf5")
            
            cam_fov = np.array([intrinsics[0, 0], intrinsics[1, 1]]) # (2,)
            cam_id = np.array([img_id]) # (1,) 
            frame_data = {
                'colors': masked_image, # NOTE this is background-masked image! should have black bg
                'depth': depth,
                'binary_masks': np.stack([mask, mask], axis=0), # dummy mask!
                'cam_fov': cam_fov,
                'cam_id': cam_id, 
                'cam_pose': cameraPoseARFrame,
                'cam_intrinsics': intrinsics,
                'need_rotate_back': np.array( (int(need_rotate_back), )) ,
            } 
            with h5py.File(h5_fname, 'w') as hfile:
                for key, data in frame_data.items():
                    hfile.create_dataset(key, data=data)
            # try saving and export rgb image
            rgb_fname = join(out_dir, f"{img_id}.jpg")
            
            with h5py.File(h5_fname, 'r') as hfile:
                img = hfile['colors'][:]
                Image.fromarray(img).save(rgb_fname)
                print(f"Saved {rgb_fname}")
            txt_towrite.append(f"{h5_fname} 1")
        txt_fname = join(out_dir, "num_masks.txt") # dummy file!!
        with open(txt_fname, "w") as f:
            f.write("\n".join(txt_towrite))
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="/local/real/mandi/scanner_data", type=str, help='input folder containing raw data')
    parser.add_argument('--output_dir', type=str, default="/local/real/mandi/scanner_dataset", help='output folder for h5 files')
    parser.add_argument('--obj_lookup', type=str, default="*", help='object id')
    parser.add_argument('--overwrite', '-o', action='store_true', help='overwrite existing files')
    args = parser.parse_args()
    run(args)