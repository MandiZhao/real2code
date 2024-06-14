import os 
from os.path import join
import copy
import numpy as np
from copy import deepcopy
import dm_control
import mujoco
from dm_control import mujoco as dm_mujoco
from dm_control import mjcf
import trimesh
import argparse
import lxml
from lxml import etree
from collections import defaultdict
from glob import glob  
from natsort import natsorted 
from data_utils import raw_urdf_to_merged_xml, repaired_xml_to_mjcf, get_offset_xml, get_relative_joint_code, visualize_obj_sim
from data_utils import get_train_data_from_info, write_to_shard

"""
Data preparation pipeline

Input: obj folders used for Blender rendering, each should contain a mobility_repaired.urdf file
Output: 
1. repaired.xml & mjcf_code.py: stand-alone mujoco files for the object, better for being loaded & visualized in mujoco than the original urdf files
2. offsetted.xml: geom-shifted version of the object, where:
    - add refquat for each mesh asset, make all body and geom positioned at origin
    - move joint pos to (rotated) new non-origin position
3. different versions of re-parameterizing the joint parameters, using OBBs extracted from each mesh as prompt header
    1) absolute/ - global absolute params: doesn't use OBB for joints
    2) obb_rot/ - OBB edge for axes, position is relative to OBB center 
    3) obb_rel/ - OBB edge for both axes and pos 
    
    Each folder contains: info_loop_x.json, one for each rendering loop
    
4. the json files can the be used by write_shard for generating the code dataset

Example commands:
# render test-time single loop data:
python preprocess_data.py --loop_id 0 --folder "*" --split test --num_augs 3  
# write out shard files:
python preprocess_data.py --loop_id 0 --folder "*" --split test --write_augs 1 -sh 
"""

REPAIRD_XML = "repaired.xml"
MJCF_FNAME = "mjcf_code.py"
MESH_FOLDER = "blender_meshes"
OFFSET_XML = "offsetted.xml"

SKIP_IDS=[47585]

def lookup_folders(args):
    lookup = join(args.data_dir, args.split, args.obj_type, args.folder)
    folders = natsorted(glob(join(lookup)))
    folders = [f for f in folders if int(f.split("/")[-1]) not in SKIP_IDS]
    return folders


def process_object(args, object_path):
    obj_type = object_path.split("/")[-2]
    obj_id = object_path.split("/")[-1]
    mesh_path = join(object_path, MESH_FOLDER) # use global mesh folder
    urdf_fname = join(object_path, "mobility_repaired.urdf")
    assert os.path.exists(urdf_fname), f"urdf file not found: {urdf_fname}"
    #### 1. repaired.xml & mjcf_code.py  #### 
    repaired_xml = raw_urdf_to_merged_xml(
            urdf_fname,
            output_fname=REPAIRD_XML,
            overwrite=args.overwrite_xml,        
        )
    assert repaired_xml is not None, f"Failed to repair the urdf file: {urdf_fname}"
    saved_mjcf = repaired_xml_to_mjcf(
        repaired_xml, output_fname=MJCF_FNAME, mesh_folder=mesh_path)
    assert os.path.exists(saved_mjcf), f"Failed to save the mjcf file: {saved_mjcf}"
    # print(f"Successfully saved mjcf file: {saved_mjcf}")

    #### 2. offsetted.xml ####
    offset_xml = get_offset_xml(repaired_xml, output_fname=OFFSET_XML, overwrite=args.overwrite_xml, try_load=True,)
    assert os.path.exists(offset_xml), f"Failed to save the offsetted xml file: {offset_xml}"

    #### 3. re-parameterizing the joint parameters ####
    saved_infos = get_relative_joint_code(
        obj_folder=object_path, loop_lookup=args.loop_id,
        mesh_folder=mesh_path, input_fname=OFFSET_XML, 
        overwrite=args.overwrite_obb, save_vis=True,
    )
    # should be dict(loop_x={absolute: dict, obb_rot: dict, obb_rel: dict})
    # print(f"Successfully saved the joint parameterization info: {saved_infos.keys()}")
    if args.try_vis:
        for mode in ["absolute", "obb_rot", "obb_rel"]:
            vis_succ = visualize_obj_sim(
                saved_infos['loop_0'][mode]['full_code'],
                img_fname=f"data_utils/test_{mode}.png",
                save_code_fname=f"data_utils/test_{mode}.py")
            if not vis_succ:
                print(f"Failed to visualize the object in mode: {mode}")
    
    # now, use the GT OBB-relative code to generate data for training LLM, where the OBBs are rotation * center augmented
    saved_data_jsons, all_saved_fnames = get_train_data_from_info(
        object_path, 
        loop_id_lookup=args.loop_id,
        modes=["absolute", "obb_rel", "obb_rot"], 
        overwrite=args.overwrite_info, num_augs=args.num_augs,
        aug_center_margin=0.3, aug_angle_margin=np.pi,
    )
 
    return saved_data_jsons, all_saved_fnames

def collect_data_to_shard(args):
    folders = lookup_folders(args)
    all_data, all_fnames = defaultdict(list), defaultdict(list)
    obj_count = 0
    for object_path in folders:
        saved_data_jsons, all_saved_fnames = get_train_data_from_info(
            object_path, 
            loop_id_lookup=args.loop_id,
            modes=["absolute", "obb_rel", "obb_rot"], 
            overwrite=False, # collect data only
            num_augs=args.write_augs, 
        )
        for mode in saved_data_jsons.keys():
            all_data[mode].extend(saved_data_jsons[mode])
            all_fnames[mode].extend(all_saved_fnames[mode])
        obj_count += 1
    print(f"Processed {obj_count} objects in {args.data_dir}/{args.split}/{args.obj_type}/{args.folder}")
    return all_data, all_fnames

def run(args):
    folders = lookup_folders(args)
    if len(folders) == 0:
        print(f"No folders found in {args.data_dir}/{args.split}/{args.obj_type}/{args.folder}")
        return
    if args.shard_only:
        all_data, all_fnames = collect_data_to_shard(args)
         # a list per every mode
        print(f"Writing to shard:")
        for mode in ["absolute", "obb_rel", "obb_rot"]:
            _data, _fnames = all_data[mode], all_fnames[mode]
            shard_dir = join(args.shard_output_dir, mode)
            os.makedirs(shard_dir, exist_ok=True)
            written_dir = write_to_shard(
                output_dir=shard_dir,
                data_dicts=_data,
                data_fnames=_fnames, np_random=np.random.RandomState(0),
                split=args.split,
                num_augs=args.write_augs,
                num_images_per_sample=args.num_images,
            )
        print(f"Successfully written shards to {written_dir}")
    else:
        all_data, all_fnames = defaultdict(list), defaultdict(list)
        obj_count = 0
        for folder in folders:
            data_jsons, saved_fnames = process_object(args, folder)
            for mode in data_jsons.keys():
                all_data[mode].extend(data_jsons[mode])
                all_fnames[mode].extend(saved_fnames[mode])
            obj_count += 1
        print(f"Processed {obj_count} objects in {args.data_dir}/{args.split}/{args.obj_type}/{args.folder}")
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/local/real/mandi/real2code_dataset_v0/")
    parser.add_argument("--shard_output_dir", type=str, default="/local/real/mandi/real2code_shards_v0")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--obj_type", type=str, default="*")
    parser.add_argument("--loop_id", type=str, default="*")
    parser.add_argument("--folder", type=str, default="*") 
    # parser.add_argument("--input_folder", type=str, default="blender_meshes")   
    parser.add_argument("--merged_xml_name", "-m", type=str, default="merged.xml", help="name of the merged xml")
    parser.add_argument("--skip_collision", "-sc", action="store_true", help="skip collision geoms") # TODO: need to add try-loading to the merged collision meshes, keep enabling this until then
    parser.add_argument("--overwrite_xml", "-o", action="store_true", help="overwrite the merged xml if it exists" )
    parser.add_argument("--overwrite_obb", "-ob", action="store_true", help="overwrite the obb info if it exists" )
    parser.add_argument("--overwrite_info", "-oi", action="store_true", help="overwrite the data info json if it exists" )

    parser.add_argument("--try_vis", "-vis", action="store_true", help="try to visualize the object in mujoco")
    parser.add_argument("--shard_only", "-sh", action="store_true", help="only write to shard")
    parser.add_argument("--num_augs", type=int, default=5, help="number of augmentations to generate")
    parser.add_argument("--write_augs", type=int, default=1, help="number of augmentations to write to shard")
    parser.add_argument("--num_images", type=int, default=1, help="number of images to save per object")
    args = parser.parse_args()
    run(args)

