import argparse 
import os
from os.path import join
import numpy as np
import tempfile
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation

import seaborn as sns
from glob import glob 
from PIL import Image
from natsort import natsorted
from scipy.spatial import cKDTree as KDTree
import open3d as o3d
from image_seg import load_sam_model, get_image_transform, eval_model_on_points, get_background_mask
from image_seg.test_sam import multimask_nms_filter, get_filled_img, compute_iou

RGB_SIZE = (512, 512)

def prompt_sam(model, device, all_prompts, all_img_data, batch_size=128, use_fg_mask=True):
    transform = get_image_transform(1024, jitter=False, random_crop=False)  
    num_points = len(all_prompts[0])
    num_cameras = len(all_img_data)
    sam_mask_size = RGB_SIZE
    init_values = dict( 
        masks=np.zeros((num_points, num_cameras, sam_mask_size[0], sam_mask_size[1])),
        repaired_masks=np.zeros((num_points, num_cameras, sam_mask_size[0], sam_mask_size[1])),  
        stability_scores=np.zeros((num_points, num_cameras)), 
        iou_predictions=np.zeros((num_points, num_cameras)),
        coords_2d=np.zeros((num_points, num_cameras, 2)),
    )
    for i, (rgb, pts3d, fg_mask) in enumerate(all_img_data):
        point_coords = all_prompts[i]
        labels = np.ones(len(point_coords))
        num_batches = int(np.ceil(len(point_coords) / batch_size))
        image = copy.deepcopy(rgb)
        if use_fg_mask:
            image[fg_mask == 0] = 0 
        bg_mask = np.zeros_like(fg_mask)
        bg_mask[fg_mask == 0] = 1
        
        image, _, _ = transform(image, np.ones((1,) + RGB_SIZE)) # dummy mask
        
        image = model.preprocess(image.to(device))
        image_embeddings = model.image_encoder(image.unsqueeze(0))
        image_outputs = []
        for j in range(num_batches):
            prompts = point_coords[j*batch_size : (j+1)*batch_size]
            batch_results = eval_model_on_points(
                prompts, [], image_embeddings, model, original_size=RGB_SIZE, device=device,  
            ) 
            batch_results["coords_2d"] = prompts
            pred_masks = batch_results["repaired_masks"]
            for n, mask in enumerate(pred_masks):
                if compute_iou(mask, bg_mask) > 0.7:
                    # remove the predicted mask if it's bg
                    batch_results["repaired_masks"][n] = np.zeros_like(mask)
                    batch_results["masks"][n] = np.zeros_like(mask)
                    batch_results["stability_scores"][n] = 0
                    batch_results["iou_predictions"][n] = 0

            image_outputs.append(batch_results)
        concat_outputs = {k: np.concatenate([x[k] for x in image_outputs], axis=0) 
                          for k in image_outputs[0].keys() if k != "gt_ious"}

        for k, v in concat_outputs.items():
            init_values[k][:, i] = v
        
    return init_values

def postprocess_sam_results(init_values, save_path, sam_mask_size=RGB_SIZE):
    num_points = len(init_values["masks"])
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
    num_saved = 0
    for masks, idx in zip(nms_filtered, nms_idxs):
        coords_2d = sorted_results["coords_2d"][idx]
        rgb_masks = []
        for binary_mask, coords in zip(masks, coords_2d):
            x, y = int(coords[0]), int(coords[1])
            rgb_mask = np.zeros((sam_mask_size[0], sam_mask_size[1], 3))
            rgb_mask[binary_mask > 0] = [255, 255, 255]
            rgb_mask[x:x+10, y:y+10] = [255, 0, 0]
            rgb_masks.append(rgb_mask)
        concat_masks = np.concatenate(rgb_masks, axis=1)
        Image.fromarray(concat_masks.astype(np.uint8)).save(
                join(save_path, f"nms_mask_{idx}.png")
            )
        num_saved += 1
        if num_saved > 10:
            break
    return sorted_results, nms_filtered

def merge_seg_pcd(nms_filtered, all_img_data, size_thres=2000):
    filled_pcds, used_masks = [] , [] 
    tot_pcd_size = []
    for rgb, pts3d, mask in all_img_data:
        tot_pcd_size.append(pts3d[mask > 0].reshape(-1, 3).shape[0])
    tot_pcd_size = sum(tot_pcd_size)

    for pt_idx, masks in enumerate(nms_filtered): 
        # use previously used mask to get only the new pixels provided by each new mask
        new_masks = masks
        for mask_ls in used_masks:
            for i, old_mask in enumerate(mask_ls):
                new_mask = new_masks[i] * (old_mask == 0)
                new_masks[i] = new_mask
        # mask out the current mask 
        seg_pcd = []
        for i, mask in enumerate(new_masks):
            masked_pts3d = all_img_data[i][1].copy() 
            fg_mask = all_img_data[i][2]
            masked_pts3d[fg_mask == 0] = 0
            masked_pts3d[mask == 0] = 0
            seg_pcd.append(masked_pts3d[mask > 0].reshape(-1, 3))
        seg_pcd = np.concatenate(seg_pcd, axis=0)
        if len(seg_pcd) < size_thres:
            print(f"Skipping point {pt_idx} with {len(seg_pcd)} points")
            continue
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(seg_pcd)
        # remove outliers
        cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_pcd, alpha=0.1)
        seg_pcd = np.asarray(cl.points)
        filled_pcds.append(seg_pcd)
        used_masks.append(new_masks)
        filled_pcd_size = sum([pcd.shape[0] for pcd in filled_pcds])
        print(f"Filled {filled_pcd_size} {filled_pcd_size/tot_pcd_size} points, at {pt_idx} point idx")
        # np.savez(join(save_path, f"filled_pcd_{len(filled_pcds)-1}.npz"), points=np.array(pcd.points))
        # breakpoint()
        if filled_pcd_size / tot_pcd_size > 0.9:
            break
    return filled_pcds, used_masks

def find_correspondance(query_3d, target_3d, dist_thres=0.01):
    """ given 3d pts from source image, find corresponding 3d pts in target image and their 2d indices """
    H, W, _ = target_3d.shape
    tree = KDTree(target_3d.reshape(-1, 3))
    dists, ind = tree.query(query_3d, k=1) # both are shape (N, k)
    ind_2d = np.unravel_index(ind, (H, W)) 
    ind_2d = np.stack(ind_2d, axis=-1)
    
    # shape (N, 2)
    mask = dists > dist_thres # shape (N,) 
    # replace invalid dist indices with -1
     
    ind_2d[mask] = [-1, -1]
    return dists, ind_2d    

def sample_points(rgb, pts3d, mask, num_points=10):
    # sample 2d pts from the mask=True area then get the 3d pts
    H, W, _ = rgb.shape
    pos_pts = np.where(mask > 0)
    tosample = np.stack(pos_pts, axis=-1) # shape (N, 2)
    sampled_idxs = np.random.choice(tosample.shape[0], size=num_points, replace=False)
    sampled_2d = tosample[sampled_idxs] # shape (num_points, 2)
    sampled_3d = pts3d[sampled_2d[:, 0], sampled_2d[:, 1]] # shape (num_points, 3)
    return sampled_2d, sampled_3d

def generate_prompt_points(rgb_fnames, num_points=10, min_valid=5):
    all_img_data = []
    for fname in rgb_fnames:
        img = Image.open(fname)
        rgb = np.array(img)
        scene_fname = fname.replace('.jpg', '_scene.npz')
        pts3d = np.load(scene_fname)['pts3d']
        mask_fname = fname.replace('.jpg', '_mask.png')
        mask = np.array(Image.open(mask_fname))
        all_img_data.append((rgb, pts3d, mask))
    num_rgb = len(all_img_data)
    min_valid = min(min_valid, num_rgb)
    num_points_per_image = int(num_points * num_rgb)
    
    # sample num_points points from each 2D image, then find the corresponding 3D points on all other images 
    all_prompts = np.zeros(
        (num_rgb, num_points_per_image, 2), dtype=np.int32
        )
    for i, (rgb, pts3d, mask) in enumerate(all_img_data):  
        sampled_2d, sampled_3d = sample_points(rgb, pts3d, mask, num_points)
        all_prompts[i][i*num_points : (i+1)*num_points] = sampled_2d
        for j, (rgb_j, pts3d_j, mask_j) in enumerate(all_img_data):
            if i == j:
                continue
            dists, ind_2d = find_correspondance(sampled_3d, pts3d_j, dist_thres=0.005) 
            all_prompts[j][i*num_points : (i+1)*num_points] = ind_2d 
    # filter out points that are not valid in at least min_valid images
    valid_points = []
    for i in range(num_points_per_image):
        valid_images = np.where(np.all(all_prompts[:, i] != -1, axis=1))[0]
        if len(valid_images) >= min_valid:
            valid_points.append(i)
    print(f"Valid points: {len(valid_points)}")
    all_prompts = all_prompts[:, valid_points]
    return all_prompts, all_img_data

def draw_points_on_image(rgb, points, colors):
    ncolors = len(colors)
    rgb = copy.deepcopy(rgb)
    for i, point in enumerate(points): 
        x, y = point
        color = colors[int(i % ncolors)] 
        rgb[x:x+5, y:y+5, :] = color
    return rgb

def main(args):

    inp_folder = join(args.data_dir, args.folder)
    objdir = inp_folder
    lookup = f"{objdir}/*.jpg"
    # some images don't have corresponding mask, skip those
    inputfiles = natsorted(glob(lookup)) 
    fnames = []
    for fname in inputfiles:
        if os.path.exists(fname.replace('.jpg', '_scene.npz')) and os.path.exists(fname.replace('.jpg', '_mask.png')):
            fnames.append(fname)
    np_random = np.random.RandomState(0)
    sampled_idxs = np_random.choice(len(fnames), size=min(args.num_rgbs, len(fnames)), replace=False)
    fnames = [fnames[i] for i in sampled_idxs]
    num_rgb = len(fnames)
    print(f"Processing {objdir} with {len(fnames)} images")
    colors = sns.color_palette("colorblind", 20)
    colors = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
    all_prompts, all_img_data = generate_prompt_points(
        fnames, num_points=args.num_points, min_valid=args.min_valid
    )
    # for i, (rgb, pts3d, mask) in enumerate(all_img_data):
    #     points = all_prompts[i]
    #     rgb = draw_points_on_image(rgb, points, colors)
    #     Image.fromarray(rgb).save(f"prompt_points_{i}.jpg")
    dust3r_msks = []
    for fname in fnames:
        msk = np.load(fname.replace('.jpg', '_scene.npz'))['msk']
        dust3r_msks.append(msk)
    
    print(f"Loading SAM model:")
    model, device, forward_fn, ckpt_name = load_sam_model(
        run_name="v4_pointsTrue_lr0.0003_bs24_ac12_02-21_11-45",
        load_epoch=11, skip_load=False
    )
    init_values = prompt_sam(model, device, all_prompts, all_img_data, 128)
    save_path = join(objdir, "sam")
    os.makedirs(save_path, exist_ok=True)
    # remove existing files
    for fname in glob(join(save_path, "*")):
        os.remove(fname)
    sorted_results, nms_filtered = postprocess_sam_results(init_values, save_path)
    
    # using dust3r masks removes the noise 
    msk_masks = []
    for masks in nms_filtered:
        row = [mask.copy() * dust3r_msks[n] for n, mask in enumerate(masks)]
        msk_masks.append(row)
    msk_masks = np.array(msk_masks)

    merged_pcds, used_masks = merge_seg_pcd(msk_masks, all_img_data)
    for n in range(num_rgb): 
        filled_img, final_masks, final_idxs = get_filled_img(
            [masks[n] for masks in used_masks], min_new_pixels=1000)
        Image.fromarray(filled_img).save(join(save_path, f"filled_img_{n}.png"))
    for i, pcd in enumerate(merged_pcds):
        np.savez(join(save_path, f"filled_pcd_{i}.npz"), points=pcd)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='use dust3r results to generate SAM prompt points')
    parser.add_argument('--folder', '-f', type=str, default='1', help='Input folder')
    parser.add_argument('--data_dir', default='/home/mandi/real_rgb/', type=str, help='Data directory')
    parser.add_argument('--num_points', default=10, type=int, help='Number of points to generate')
    parser.add_argument('--num_rgbs', '-n', default=12, type=int, help='Number of RGB images to use')
    parser.add_argument('--min_valid', default=9, type=int, help='Minimum number of images a point should be valid in')
    args = parser.parse_args()
    main(args)