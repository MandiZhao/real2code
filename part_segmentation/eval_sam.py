"""
Run the 2D to 3D to 2D prompting scheme to evaluate a fine-tuned SAM model 

NOTE: assume the input images are all square

Example command for fast debugging:
python part_segmentation/eval_sam.py --obj_folder 103177 --num_3d_points 100 -o 

FULLRUN=rebuttal_full_pointsTrue_lr0.001_bs21_ac12_12-01_19-52
STEPS=24000
python part_segmentation/eval_sam.py --obj_folder 7236 --sam_run_name $FULLRUN --sam_load_steps $STEPS --num_3d_points 100 -o 

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
from PIL import Image
import pickle
import seaborn as sns
import open3d as o3d 

from part_segmentation.test_sam_utils import load_sam_model, get_one_tsdf, project_3D_to_2D, eval_model_on_points, multimask_nms_filter
from part_segmentation.finetune_sam import get_image_transform
from part_segmentation.sam_datasets import SamH5Dataset
from part_segmentation.sam_to_pcd import load_blender_data

ORIGINAL_IMG_SIZE=(512, 512)
SAM_MASK_SIZE=(1920, 1920)
STABILITY_WEIGHT=0.8
IOU_THRESHOLD=0.6
MAX_NMS_FILTERED=15

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
    assert len(val_dataset) > 0, f"Did not find any object that matches obj_type{obj_type}, obj_folder{obj_folder}, loop_id{loop_id}"
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
    all_data = []
    for img, h5_data in zip(images, h5_datas): 
        inp_rgb = model.preprocess(img[None].to(device))
        embedding = model.image_encoder(inp_rgb) 
        raw_rgb = h5_data['rgb'].copy()
        depth = h5_data['depth'].copy() 
        cam_pose = np.array(h5_data['cam_pose'])
        extrinsics = np.linalg.inv(cam_pose)
        extrinsics[2, :] *= -1
        extrinsics[1, :] *= -1
        data = {
            "image": img.to(device),
            "raw_rgb": raw_rgb,
            "embedding": embedding,
            "depth": depth,
            "intrinsics": np.array(h5_data['cam_intrinsics']),
            "extrinsics": extrinsics,
            "cam_pose": cam_pose,
        } 
        all_data.append(data)
    return all_data

def get_prompt_points(mesh, all_cam_data, num_sample_points):
    # First sample 3D _candidate_ points from the volume  
    # first sample more points and take only these that have a large number of valid corresponding 2D points on the multi-view RGB images
    candidates = mesh.sample_points_uniformly(number_of_points=num_sample_points * 2) 
    candidate_points = np.array(candidates.points)
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
    stability_weight = 0.8 
    weighted_scores = []
    num_points = init_values["stability_scores"].shape[0]
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
    nms_filtered, nms_idxs = multimask_nms_filter(sorted_results["repaired_masks"], iou_thres=0.1)
    print(f"Filtered {len(nms_filtered)} masks, truncated to {MAX_NMS_FILTERED} masks")
    # save the masks
    # h, w = ORIGINAL_IMG_SIZE
    nms_filtered = nms_filtered[:MAX_NMS_FILTERED]
    nms_idxs = nms_idxs[:MAX_NMS_FILTERED]

    mask_images = []
    for masks, idx in zip(nms_filtered, nms_idxs):
        coords_2d = sorted_results["coords_2d"][idx]
        rgb_masks = []
        for binary_mask, coords in zip(masks, coords_2d):
            x, y = int(coords[0]), int(coords[1])
            rgb_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3))
            rgb_mask[binary_mask > 0] = [255, 255, 255]
            rgb_mask[x:x+5, y:y+5] = [255, 0, 0]
            rgb_masks.append(rgb_mask)
        concat_masks = np.concatenate(rgb_masks, axis=1)
        mask_images.append(
            Image.fromarray(concat_masks.astype(np.uint8))
        ) 
    
    return nms_filtered, nms_idxs, mask_images

def aggregate_mask_to_pcd(nms_filtered, all_cam_data, max_pcd_size):
    filled_pcds, filled_volumes = [] , []
    used_masks = [] 
    extrinsics = [d['extrinsics'] for d in all_cam_data]
    intrinsics = [d['intrinsics'] for d in all_cam_data]
    # start by filling the smaller-sized masks first
    sorted_by_size = sorted(nms_filtered, key=lambda x: np.sum(x > 0))
    for pt_idx, masks in enumerate(sorted_by_size):
        rgbs = [data['raw_rgb'].copy() for data in all_cam_data]
        depths = [data['depth'].copy() for data in all_cam_data]
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
        h, w = rgbs[0].shape[:2]
        volume = get_one_tsdf(
            rgbs, depths, intrinsics, extrinsics,
            h=h, w=w,
            vlength=0.005,
            trunc=0.01,
        )
        pcd = volume.extract_point_cloud()
        pcd, _ = pcd.remove_statistical_outlier(50, 0.8)
        if len(pcd.points) < 100 and len(used_masks) > 0:
            print(f"Skipping point {pt_idx} with {len(pcd.points)} points")            
            continue
         
        filled_pcds.append(pcd)
        filled_volumes.append(volume)
        used_masks.append(new_masks)
        filled_pcd_size = sum([len(np.array(pcd.points)) for pcd in filled_pcds])
        
        print(f"Filled {filled_pcd_size} points, at {pt_idx} point idx")
        # np.savez(join(save_path, f"filled_pcd_{len(filled_pcds)-1}.npz"), points=np.array(pcd.points))
        # breakpoint()
        if filled_pcd_size / max_pcd_size > 0.95:
            break 
    return filled_pcds, used_masks

def visualize_merged_masks(used_masks, raw_rgbs=None):
    """ 
    Make a color-coded visualization of the masks
    Input: a list of N mask-groups, each has shape (num_cameras, 512, 512) 
    Output: a concatenated RGB image of shape (512, 512*num_cameras, 3) -> each camera view gets merged color masks
    """
    num_masks = len(used_masks)
    colors = sns.color_palette("colorblind", num_masks)
    colors = [np.array(np.array(c) * 255, dtype=np.uint8) for c in colors] 
    h,w = used_masks[0][0].shape
    num_cameras = used_masks[0].shape[0]
    out_image = np.zeros((h, w * num_cameras, 3), dtype=np.uint8)
    for i, mask_group in enumerate(used_masks):
        for j, mask in enumerate(mask_group):
            rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
            # rgb_mask[mask > 0] = colors[i]
            out_image[:, j*w:(j+1)*w][mask > 0] = colors[i]
    if raw_rgbs is not None:
        # concat to make it shape (512, 512*num_cameras, 3)
        raw_rgbs = np.concatenate(raw_rgbs, axis=1) # concat with mask!
        out_image = np.concatenate([raw_rgbs, out_image], axis=0) # shape (1024, 512*num_cameras, 3)
    return out_image

def evaluate_one_object(batch, model, device, save_dir, num_sample_3d_points=10, minibatch_size=128):
    """
    First get a background mask from coarsely sample query points, then, sample 3D points from the foreground masked object.
    Project each 3D points to 2D on each camera view image, then group the mask predictions across all 2D images for that point.
    """
    all_data = prepare_img_camera_data(batch, model, device)
    h = all_data[0]['raw_rgb'].shape[0]
    w = all_data[0]['raw_rgb'].shape[1]
    tsdf = get_one_tsdf(
        rgbs=[data['raw_rgb'] for data in all_data],
        depths=[data['depth'] for data in all_data],
        cam_intrinsics=[data['intrinsics'] for data in all_data],
        extrinsics=[data['extrinsics'] for data in all_data],
        h=h,
        w=w,
        vlength=0.005,
        trunc=0.01,
    )
    pcd = tsdf.extract_point_cloud()
    # debugging:write this pcd to file
    fname = join(save_dir, "partial_pcd.ply")
    o3d.io.write_point_cloud(fname, pcd)

    max_pcd_size = len(np.array(pcd.points))
    mesh = tsdf.extract_triangle_mesh()
    sample_3d_points, sample_2d_points = get_prompt_points(
        mesh, all_data, num_sample_3d_points)
    num_cameras = len(all_data)
    num_points = len(sample_3d_points)
    sam_mask_size = ORIGINAL_IMG_SIZE #SAM_MASK_SIZE
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
        # input_size = tuple(image.shape[1:]) 
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
                input_size=(1024, 1024), 
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
    
    nms_filtered, nms_idxs, mask_images = \
        filter_sort_all_camera_results(init_values)
     
    # these are a long array of masks:
    for i, mask in enumerate(mask_images):
        mask.save(join(save_dir, f"nms_mask_{i}.png"))

    # save the data needed for aggregation to pcd: 1) list of masks, 2) list of camera data
    nms_fname = join(save_dir, "nms_masks.pkl")
    with open(nms_fname, "wb") as f:
        pickle.dump(nms_filtered, f)
    tosave_cam_keys = ["intrinsics", "extrinsics", "raw_rgb", "depth"]
    cam_data_to_save = [{k: data[k] for k in tosave_cam_keys} for data in all_data]
    raw_rgbs = [data['raw_rgb'] for data in all_data]
    cam_fname = join(save_dir, "cam_data.pkl")
    with open(cam_fname, "wb") as f:
        pickle.dump(cam_data_to_save, f)

    filled_pcds, used_masks = aggregate_mask_to_pcd(
        nms_filtered, cam_data_to_save, max_pcd_size
    )
    # also save used masks
    used_masks_fname = join(save_dir, "used_masks.pkl")
    with open(used_masks_fname, "wb") as f:
        pickle.dump(used_masks, f)
    for i, pcd in enumerate(filled_pcds): 
        pcd_fname = join(save_dir, f"filled_pcd_{i}.ply") 
        o3d.io.write_point_cloud(pcd_fname, pcd)
    used_rgb = visualize_merged_masks(used_masks, raw_rgbs)
    out_image_fname = join(save_dir, "merged_used_masks.png")
    Image.fromarray(used_rgb).save(out_image_fname)
    return

def load_and_merge_pcds(obj_type, obj_folder, loop, save_dir):

    # get pcs again
    h, w = 512, 512
    cam_data_fname = join(save_dir, "cam_data.pkl")
    with open(cam_data_fname, "rb") as f:
        cam_data = pickle.load(f)
    tsdf = get_one_tsdf(
        rgbs=[data['raw_rgb'] for data in cam_data],
        depths=[data['depth'] for data in cam_data],
        cam_intrinsics=[data['intrinsics'] for data in cam_data],
        extrinsics=[data['extrinsics'] for data in cam_data],
        h=h,
        w=w,
        vlength=0.005,
        trunc=0.01,
    )
    partial_pcd = tsdf.extract_point_cloud() 
    partial_pcd_fname = join(save_dir, "partial_pcd.ply")
    
    # write 
    o3d.io.write_point_cloud(partial_pcd_fname, partial_pcd)
    # breakpoint()
    # partial_pcd = o3d.io.read_point_cloud(partial_pcd_fname)
    nms_fname = join(save_dir, "nms_masks.pkl")
    with open(nms_fname, "rb") as f:
        nms_masks = pickle.load(f)
    print(f"Loaded {len(nms_masks)} masks from {nms_fname}")
    cam_fname = join(save_dir, "cam_data.pkl")
    with open(cam_fname, "rb") as f:
        cam_data = pickle.load(f)
    max_pcd_size = len(np.array(partial_pcd.points))
    print(f"Max pcd size is {max_pcd_size}")
    filled_pcds, used_masks = aggregate_mask_to_pcd(
        [masks for masks in nms_masks], 
        cam_data, 
        max_pcd_size
    )
    for i, pcd in enumerate(filled_pcds): 
        pcd_fname = join(save_dir, f"filled_pcd_{i}.ply") 
        o3d.io.write_point_cloud(pcd_fname, pcd)
    raw_rgbs = [data['raw_rgb'] for data in cam_data]
    out_image = visualize_merged_masks(used_masks, raw_rgbs)
    out_image_fname = join(save_dir, "merged_used_masks.png")
    Image.fromarray(out_image).save(out_image_fname)
    return

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
    
    ## use this only for looking up model checkpoint names
    _, _, _, load_model_fname = load_sam_model(
        zero_shot=args.zero_shot_sam,
        run_name=args.sam_run_name,
        load_epoch=args.sam_load_epoch,
        load_steps=args.sam_load_steps,
        sam_type=args.sam_type,
        points=True,
        model_dir=args.sam_model_dir, 
        skip_load=True,
    )
    if args.zero_shot_sam:
        print("Zero shot model loaded")
        save_dir = join(args.output_dir, "zero_shot")
    else:
        save_dir = join(args.output_dir, load_model_fname.split('/')[-2], load_model_fname.split("/")[-1].split(".pth")[0])
    os.makedirs(save_dir, exist_ok=True)
    
    if args.merge_pcd_only:
        print("Merging PCDs only")
        for obj_loop_dir in tqdm(unique_objects):
            # e.g. /store/real/mandi/real2code_dataset_v0/test/Scissors/11111/loop_0
            obj_type, obj_folder, loop = obj_loop_dir.split("/")[-3:]
            loop_id = loop.split("_")[-1]
            obj_save_path = join(save_dir, obj_type, obj_folder, loop)
            if not os.path.exists(obj_save_path):
                print(f"Path {obj_save_path} does not exist - skipping")
                continue
            load_and_merge_pcds(obj_type, obj_folder, loop, obj_save_path)
        print("Merging PCDs complete")
        return

    if args.vis_mask_only:
        print("Visualizing masks only")
        for obj_loop_dir in tqdm(unique_objects):
            obj_type, obj_folder, loop = obj_loop_dir.split("/")[-3:]
            loop_id = loop.split("_")[-1]
            obj_save_path = join(save_dir, obj_type, obj_folder, loop)
            used_fname = join(obj_save_path, "used_masks.pkl")
            if not os.path.exists(used_fname):
                print(f"Path {used_fname} does not exist - skipping")
                continue
            used_masks = pickle.load(open(used_fname, "rb"))
            
            
            cam_fname = join(obj_save_path, "cam_data.pkl")
            with open(cam_fname, "rb") as f:
                cam_data = pickle.load(f)
            raw_rgbs = [d['raw_rgb'] for d in cam_data]
            used_rgb = visualize_merged_masks(used_masks, raw_rgbs)
            
            out_image_fname = join(obj_save_path, "merged_used_masks.png")
            Image.fromarray(used_rgb).save(out_image_fname)
        return

    model, device, forward_fn, load_model_fname = load_sam_model(
        zero_shot=args.zero_shot_sam,
        run_name=args.sam_run_name,
        load_epoch=args.sam_load_epoch,
        load_steps=args.sam_load_steps,
        sam_type=args.sam_type,
        points=True,
        model_dir=args.sam_model_dir, 
        skip_load=False,
    )

    for obj_loop_dir in tqdm(unique_objects):
        # e.g. /store/real/mandi/real2code_dataset_v0/test/Scissors/11111/loop_0
        obj_type, obj_folder, loop = obj_loop_dir.split("/")[-3:]
        loop_id = loop.split("_")[-1]
        obj_save_path = join(save_dir, obj_type, obj_folder, loop)
        os.makedirs(obj_save_path, exist_ok=True)
        # if there's already files in the directory, check if args.overwrite is set
        if len(glob(join(obj_save_path, "*"))) > 0 and not args.overwrite:
            print(f"Skipping {obj_save_path} as it already has files")
            continue
        loader, dataset = load_eval_dataset(
            data_dir, obj_type, obj_folder, loop_id, subsample_cameras
        ) 
        batch = next(iter(loader)) # one batch should be all the images for one object!
        
        evaluate_one_object(batch, model, device, obj_save_path, num_sample_3d_points=args.num_3d_points)
        print(f"Saved results to {obj_save_path}")
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/store/real/mandi/real2code_dataset_v0")
    parser.add_argument("--output_dir", type=str, default="/store/real/mandi/real2code_eval_v0")

    parser.add_argument("--obj_type", type=str, default="*")
    parser.add_argument("--obj_folder", type=str, default="*")
    parser.add_argument("--loop_id", type=str, default="0")
    parser.add_argument("--sam_run_name", type=str, default="scissors_eyeglasses_only_pointsTrue_lr0.001_bs24_ac24_11-30_00-23")
    parser.add_argument("--sam_load_epoch", type=int, default=110) ## epoch takes precedence over steps
    parser.add_argument("--sam_load_steps", type=int, default=-1)
    parser.add_argument("--sam_type", type=str, default="default")
    parser.add_argument("--sam_model_dir", type=str, default="/store/real/mandi/sam_models") 
    parser.add_argument("--zero_shot_sam", action="store_true")
    parser.add_argument("--subsample_cameras", "-sub", type=int, default=1)
    parser.add_argument("--overwrite", '-o', action="store_true")
    parser.add_argument("--num_3d_points", type=int, default=1000)

    parser.add_argument("--merge_pcd_only", "-mp", action="store_true")
    parser.add_argument("--vis_mask_only", "-vm", action="store_true")
    main()


