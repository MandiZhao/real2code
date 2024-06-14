import os
import numpy as np
import h5py
from PIL import Image
from os.path import join
from glob import glob
from natsort import natsorted
import seaborn as sns
import json 
from copy import deepcopy
from data_utils.obb_utils import show_obb_pyplot, translate_dict_to_code
import base64
from io import BytesIO
import webdataset as wds
import uuid 

"""
Convert data to shard, used for LLM fine-tuning
"""
CAM_IDS=[2,4,8,10]
def convert_h5_to_dict(folder_dir, camera_ids=CAM_IDS, save_mask=False):
    """ Under one loop's folder, select a subset of hdf5 files and load RGB images to use """
    image_infos = []
    loop_id = folder_dir.split('/')[-1].split('_')[-1] # e.g. loop_0
    h5_fnames = natsorted(glob(join(folder_dir, "*.hdf5")))
    for hd5f_path in h5_fnames:
        cam_id = hd5f_path.split('/')[-1].split('.')[0] # e.g. 0.hdf5
        if int(cam_id) not in camera_ids:
            continue
        with h5py.File(hd5f_path, 'r') as hfile:
            if save_mask:
                id_mask = hfile['class_segmaps'] # shape H,W, need to convert the category-id here to rgb values
                rgb_mask = np.zeros((id_mask.shape[0], id_mask.shape[1], 3), dtype=np.uint8)
                binary_masks = np.array(hfile['binary_masks']) 
                colors = sns.color_palette("muted", binary_masks.shape[0]) # color has dtype float
                for i, mask in enumerate(binary_masks): 
                    xs, ys = np.where(mask == 1)
                    rgb_mask[xs, ys] = (np.array(colors[i]) * 255).astype(np.uint8)
                # save the rgb mask
                rgb_mask = Image.fromarray(rgb_mask)
                mask_fname = join(folder_dir, f'mask_{cam_id}.png')
                rgb_mask.save(mask_fname) # NOTE: this is merged mask!!
                # print(f"Saving mask to {mask_fname}")
                image_infos.append(
                    {
                        "name": f"loop_{loop_id}_mask_{cam_id}",
                        "path": mask_fname,
                        "type": "mask",
                        "camera_name": f"loop_{loop_id}_cam_{cam_id}",
                    }
                )

            rgb = hfile['colors']
            rgb_fname = join(folder_dir, f'rgb_{cam_id}.png')
            if not os.path.exists(rgb_fname):
                rgb = Image.fromarray(rgb)
                rgb.save(rgb_fname)
            image_infos.append(
                {
                    "name": f"loop_{loop_id}_rgb_{cam_id}",
                    "path": rgb_fname,
                    "type": "rgb",
                    "camera_name": f"loop_{loop_id}_cam_{cam_id}",
                }
            )
    return image_infos  

def rot_around_z(theta):
    """ rotation matrix for rotating around z-axis """
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def augment_obbs(bboxes, label_code, mode="obb_rot", center_margin=0.3, angle_margin=np.pi):
    """ 
    Augment the OBBs by rotating around z-axis and translating them.
    If the mode is absolute, need to also offset and rotate the joints
    """
    aug_obbs = dict()
    rand_angle = np.random.uniform(-angle_margin, angle_margin)
    rot_matrix = rot_around_z(rand_angle)
    def rotate_around_z(center, R, rot_matrix):
        center_rot = rot_matrix @ center
        R_rot = rot_matrix @ R
        return center_rot, R_rot
    offset = np.random.uniform(-center_margin, center_margin, 3)
    # get the rot matrix for rotating around z-axis for this angle
    for bbox_id, bbox in bboxes.items():
        center = np.array(bbox["center"])
        rot_R = np.array(bbox["R"]) # 3x3 
        
        rot_center, new_R = rotate_around_z(center, rot_R, rot_matrix)
        new_center = rot_center + offset
        aug_obbs[bbox_id] = dict(
            center=np.round(new_center, 2).tolist(),
            R=np.round(new_R, 2).tolist(),
            extent=bbox["extent"].copy(),
        )
    aug_joints = deepcopy(label_code)
    if mode == "absolute": 
        aug_lines = []
        for line in label_code.split("\n"):
            newline = deepcopy(line)
            if "pos=" in line:
                _pos = line.split("pos=[")[1].split("]")[0]
                pos = np.array([float(v) for v in _pos.split(",")])
                pos_rot = rot_matrix @ pos + offset
                newline = newline.replace(_pos, f"{pos_rot[0]:.2f},{pos_rot[1]:.2f},{pos_rot[2]:.2f}")
            if 'axis=' in line:
                _axis = line.split("axis=[")[1].split("]")[0]
                axis = np.array([float(v) for v in _axis.split(",")])
                axis_rot = rot_matrix @ axis
                newline = newline.replace(_axis, f"{axis_rot[0]:.2f},{axis_rot[1]:.2f},{axis_rot[2]:.2f}")
            aug_lines.append(newline)
        aug_joints = "\n".join(aug_lines)
    return aug_obbs, aug_joints, rot_matrix, offset

def get_train_data_from_info(
        obj_folder, 
        modes=["absolute"], 
        loop_id_lookup="*",
        overwrite=False, 
        num_augs=3,
        aug_center_margin=0.3,
        aug_angle_margin=np.pi,
        ):
    """
    For each info_loop_x.json, generate data_loop_x.json where the OBBs are augmented num_augs times
    data_loop_x.json contains:
        - image_info: fnames for RGB images from this loop
        - aug_obb_code: a list of augmented OBB lines 
        - label_code: copied from info_loop_x.json, each version of aug_obbs share this same label_code 
    """
    
    all_saved_jsons = dict()
    all_saved_fnames = dict()
    loop_folders = natsorted(glob(join(obj_folder, f"loop_{loop_id_lookup}")))
    image_infos = dict()
    for loop_folder in loop_folders:
        _id = loop_folder.split('/')[-1].split('_')[-1] # e.g. loop_0
        image_infos[_id] = convert_h5_to_dict(loop_folder)
    for mode in modes: 
        mode_folder = join(obj_folder, mode)
        info_jsons = natsorted(glob(join(mode_folder, f"info_loop_{loop_id_lookup}.json")))
        if len(info_jsons) == 0:
            print(f"Warning! No info json files found in {mode_folder}")
            continue 
        loop_data, loop_fnames = [], []
        for json_fname in info_jsons:
            data_fname = json_fname.replace("info_", "data_")
            # print(f"Processing mode {mode}", json_fname, data_fname)
            if not overwrite and os.path.exists(data_fname):
                with open(data_fname, "r") as f:
                    loaded_data = json.load(f)
                loop_data.append(loaded_data)
                loop_fnames.append(data_fname)
                continue

            with open(json_fname, "r") as f:
                info_dict = json.load(f)
            loop_id = info_dict["loop_id"] # single digit
            image_info = image_infos[loop_id]
            bboxes = info_dict["bboxes"]
            label_code = info_dict["label_code"]
            aug_obbs, aug_labels, aug_rot, aug_offset = [], [], [],[]

            for i in range(num_augs):
                aug_obb, aug_label, rot_matrix, offset = augment_obbs(
                    deepcopy(bboxes), label_code, mode, aug_center_margin, aug_angle_margin
                    )
                # for DEBUG:
                # show_obb_pyplot(
                #     [v for v in aug.values()], joints=[], save_img_fname=f"aug_{i}.png", skip_show=True) 
                aug_code = translate_dict_to_code(aug_obb)
                aug_obbs.append("\n".join(aug_code))
                aug_labels.append(aug_label)
                aug_rot.append(rot_matrix.tolist())
                aug_offset.append(offset.tolist())
            if num_augs == 0:
                aug_code = translate_dict_to_code(bboxes)
                aug_obbs.append("\n".join(aug_code))
                aug_labels.append(label_code)
                aug_rot.append(np.eye(3).tolist())
                aug_offset.append(np.zeros(3).tolist())

            tosave_data = dict(
                image_info=image_info,
                aug_obbs=aug_obbs,
                aug_labels=aug_labels,
                label_code=label_code,
                aug_rot=aug_rot,
                aug_offset=aug_offset
            )
            loop_data.append(tosave_data)
            loop_fnames.append(data_fname)
            with open(data_fname, "w") as f:
                json.dump(tosave_data, f, indent=4)
        all_saved_jsons[mode] = loop_data
        all_saved_fnames[mode] = loop_fnames
    return all_saved_jsons, all_saved_fnames

def get_img_str(pil_img):
    # try save with json
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8") 
    return img_str

def load_images(image_info, base64=False, save_mask=False, num_images_per_sample=1):
    images = dict()
    paths = dict()
    for info in image_info[:num_images_per_sample]:
        _name = info["name"]
        _path = info["path"]
        if not save_mask and "mask" in _name:
            continue
        assert os.path.exists(_path), f"Image path {_path} does not exist."
        img = Image.open(_path)
        if base64:
            img = get_img_str(img)
        images[_name] = img
        paths[_name] = _path

    return images, paths 

def compose_text(
        label_codes, aug_obbs, np_random, num_images=0, num_augs=3):
    all_prompts = []
    if len(label_codes) == 1:
        for aug_code in aug_obbs[:num_augs]: 
            prompt_text = ["<image>" for _ in range(num_images)] + [aug_code, label_codes[0]]
            all_prompts.append(
                "\n".join(prompt_text)
            )
    else:
        assert len(label_codes) == len(aug_obbs), "Different number of label codes and aug obbs"
        for label_code, aug_code in zip(label_codes, aug_obbs):
            prompt_text = ["<image>" for _ in range(num_images)] + [aug_code, label_code]
            all_prompts.append(
                "\n".join(prompt_text)
            )
            if len(all_prompts) == num_augs:
                break
    return all_prompts

def write_to_shard(
        output_dir, data_dicts: list, data_fnames: list, 
        np_random, split="train", num_images_per_sample=0,
        num_files_per_shard=9000, num_augs=3,
    ):
    """ Takes a list of dicts, agnostic to loop id or object ids, write to a shard"""
    num_written = 0
    os.makedirs(join(output_dir, split), exist_ok=True)
    _dir = join(output_dir, split, "%04d.tar")
    with wds.ShardWriter(_dir) as sink:
        # fname e.g. real2code_dataset_v0/train/Box/100064/absolute/data_0.json
        for fname, sample_data in zip(data_fnames, data_dicts):
            obj_type = fname.split('/')[-4]
            obj_folder = fname.split('/')[-3]
            obj_dir = "/".join(fname.split('/')[:-2])
            image_info = sample_data["image_info"]
            images, paths = load_images(
                image_info, base64=True, save_mask=False, 
                num_images_per_sample=num_images_per_sample
                )
            processed_data = dict( 
                obj_type=obj_type,
                obj_folder=obj_folder,
                obj_dir=obj_dir,
                image_names=list(images.keys()),
                images=images
            )
           
            texts = compose_text(
                # sample_data["label_code"], 
                sample_data["aug_labels"],
                sample_data["aug_obbs"], 
                np_random=np_random,
                num_images=num_images_per_sample,
                num_augs=num_augs
            )
            for text in texts:  
                # save a separate copy per random text! 
                key = uuid.uuid4().hex
                processed_data["text"] = text 
                towrite = dict(
                    __key__=key,
                    json=processed_data
                ) 
                sink.write(towrite)
                num_written += 1
            if (num_written + 1) % 100 == 0:
                print(f"Wrote {num_written} samples")

            if (num_written + 1) % num_files_per_shard == 0:
                sink.next_stream() 
    print(f"Finished writing {num_written} samples")
    return _dir