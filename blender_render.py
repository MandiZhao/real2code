import blenderproc as bproc
import numpy as np
from os.path import join
import lxml
from lxml import etree
import os
from glob import glob 
import argparse
from blenderproc.python.types.BoneUtility import get_constraint
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.writer.WriterUtility import _WriterUtility
import h5py
import bpy
from bpy import context
from PIL import Image
import trimesh
import json 
""" 
Latest: 
- for each loop: 
    2. save each link's get_visual_local2world_mats() -> use this to align GT mesh with the rendered mesh pcd, this also avoids adjusting refquat 
    3. save each joint's set value 
    4. to get OBB: use the saved GT global mesh and transforms to rotate each part mesh first, then get the OBB
- todo for obb-based code-gen: random rotate the object along z axis 

for OBJ in StorageFurniture Table; do
export MB_DATADIR=/local/real/mandi/mobility_dataset_v1/${OBJ}
for FOLDER in ${MB_DATADIR}/*; do 
    printf "   processing object#   : $FOLDER"  
    blenderproc run blender_render.py --folder ${FOLDER}   -o
done
done

Test set objects:
for FOLDER in 19855 20985 22241 22301 22433 25493 26387 26652; do blenderproc run blender_render.py --folder ${FOLDER} --num_loops -o; done

for FOLDER in 40417 44781 44817 44853 44962 45243 45248 45271 45332 45423 45505 45523 45662 45693 45694 45699 45747 45779 45922 45940 46172 46762 46787 46955 47419 47742 47976 48263 48271 48356 48467 48855 49025 49133; do blenderproc run blender_render.py --folder ${FOLDER} --num_loops 5 -o; done
"""

NUM_LIGHTS = 6
CAM_DISTANCE = 4 #5 
CAM_HEIGHT = 0.1 # 0.7 # NOTE: manually set this because bproc.object.compute_poi() seems to always get a lower height
# POI_RANGE_LOW = np.array([-0.2, -0.2, 0])
# POI_RANGE_HIGH = np.array([-0.1, -0.1, CAM_HEIGHT]) # shift it back because many drawers are open forward!

CAM_ROTATION_MAX = 3*np.pi/2 - np.pi/6
CAM_ROTATION_MIN = np.pi/2 + np.pi/8
# render full circle:
# CAM_ROTATION_MIN = 0
# CAM_ROTATION_MAX = 2*np.pi
RENDER_WIDTH = 512
RENDER_HEIGHT = 512
RENDER_DEPTH = True
DEPTH_KINECT_NOISE = False
RENDER_NORMALS = False

CUSTOM_CAMERA_DIST_HEIGHT={
    "Eyeglasses": [2.8, 0.85],
    "Scissors": [3.5, 1.5],
    "Microwave": [4.5, 0.5],
    "Door": [4.2, 0.5],
}
  
from mathutils import Euler
import bpy
from blenderproc.python.utility.Utility import Utility
from mathutils import Euler

import bpy
from typing import Union
import os
import numpy as np
from blenderproc.python.utility.Utility import Utility


def clean_up_urdf(fname, np_random, open_drawer=True, margin=0.8):
    # print(f"Warning!! Opening all the drawers fully now")
    parsed = etree.parse(fname)
    root = parsed.getroot()
    # add effort and velocity limits to all joints
    for joint in root.findall('joint'):
        limit_elem = joint.find('limit')
        if limit_elem is None:
            limit_elem = etree.SubElement(joint, 'limit')
        limit_elem.attrib['effort'] = '100'
        limit_elem.attrib['velocity'] = '100'
        if open_drawer and joint.attrib['type'] == 'prismatic': # drawers
            _upper = limit_elem.attrib['upper']
            _upper = np_random.uniform(float(_upper) * margin, float(_upper))
            # _upper = float(_upper)# NOTE: open the drawer fully!
            _axis = joint.find('axis').attrib['xyz']
            _axis = np.array(_axis.split(' ')).astype(np.float)
            _offset = _axis * _upper
            child = joint.find('child')
            # print(_offset)
            if child is not None:
                child_link = child.attrib['link']
                child_link_elem = [link_elem for link_elem in root.findall('link') if link_elem.attrib['name'] == child_link]
                
                if len(child_link_elem) > 0:
                    _elem = child_link_elem[0]
                    # this doesn't work:
                    #  origin_elem = etree.SubElement(_elem, 'origin')
                    # origin_elem.attrib['xyz'] = ' '.join([str(x) for x in _offset])
                    visuals = _elem.findall('visual')
                    for visual in visuals:
                        ori = visual.find('origin') 
                        if ori is not None:
                            xyz = ori.attrib['xyz'].split(' ')
                            xyz = np.array(xyz).astype(np.float) + _offset
                            ori.attrib['xyz'] = ' '.join([str(x) for x in xyz])
                    
    tmp_fname = fname.replace('.urdf', '_tmp.urdf')
    parsed.write(tmp_fname, pretty_print=True)
    return tmp_fname

def simplify_urdf(fname, mesh_dir="blender_meshes"):
    """ (after all the meshes were fixed) replace the complex meshes with 1 merged mesh per object """
    parsed = etree.parse(fname)
    root = parsed.getroot()
    # add mujoco: <mujoco> <compiler discardvisual="false" meshdir="blender_meshes"/></mujoco>
    mujoco_elem = etree.SubElement(root, 'mujoco')
    compiler_elem = etree.SubElement(mujoco_elem, 'compiler', discardvisual="false", meshdir=mesh_dir)
    # add effort and velocity limits to all joints
    for joint in root.findall('joint'):
        limit_elem = joint.find('limit')
        if limit_elem is None:
            limit_elem = etree.SubElement(joint, 'limit')
        limit_elem.attrib['effort'] = '100'
        limit_elem.attrib['velocity'] = '100'
    # remove all the meshes for each link
    for i, link_elem in enumerate(root.findall('link')):
        link_name = link_elem.attrib['name']
        if link_name is None:
            link_name = f"link_{i}"
        visuals = link_elem.findall('visual')
        visual_origins = []
        # remove all visual elems
        for visual in visuals:
            if visual.find('origin') is not None:
                xyz = visual.find('origin').attrib['xyz'] # string
                visual_origins.append(
                    np.array([float(x) for x in xyz.split(' ')])
                    )
            link_elem.remove(visual)
        
        collisions = link_elem.findall('collision')
        # remove all collision elems
        for collision in collisions:
            link_elem.remove(collision)

        # add a new visual elem
        if len(visuals) > 0: # skip if there are no visuals
            visual_elem = etree.SubElement(link_elem, 'collision')
            visual_elem.attrib['name'] = link_name + "_collision"
            if len(visual_origins) == 0:
                new_xyz = [0, 0, 0]
            else:
                new_xyz = np.mean(np.array(visual_origins), axis=0) # average all the origins 
            origin_elem = etree.SubElement(visual_elem, 'origin', xyz=' '.join([f"{x:.2f}" for x in new_xyz]))
            geometry_elem = etree.SubElement(visual_elem, 'geometry')
            mesh_elem = etree.SubElement(geometry_elem, 'mesh', filename=f"{mesh_dir}/{link_name}.obj")
    return parsed

def set_hinge_joints(obj, np_random, margin=0.4, randomize_jnt=True, high_margin=None):
    """ only handles revolute joints, prismatic joints are handled in urdf for now"""
    links = obj.links 
    joint_rots = dict()
    for l, link in enumerate(links): 
        link_name = link.get_name()
        if link.joint_type is None or link.joint_type == 'fixed':
            joint_rots[link_name] = None
            continue
        bone = link.fk_bone
        c = get_constraint(bone=bone, constraint_name='Limit Rotation') 
        axis = link._determine_rotation_axis(bone=bone)
        if c is not None:
            max_value = {"x": c.max_x, "y": c.max_y, "z": c.max_z}[axis.lower()]
            min_value = {"x": c.min_x, "y": c.min_y, "z": c.min_z}[axis.lower()]
            # set rotation NOTE possible that max_value == 0, min_value is negative 
            if randomize_jnt:
                if high_margin is not None:
                    max_value *= high_margin
                value = np_random.uniform(min_value + (max_value - min_value) * margin, max_value)
                link.set_rotation_euler_fk(rotation_euler=value)
            else:
                value = min_value
                link.set_rotation_euler_fk(rotation_euler=value)
            joint_rots[link_name] = value 
        else:
            joint_rots[link_name] = None
    return joint_rots

def resample_lights(num_lights, init_lights=[], np_random=np.random.RandomState(0)):
    lights = init_lights
    # # add one overhead fixed light
    # light = bproc.types.Light()
    # light.set_type("SUN")
    # light.set_location(np.array([0, 0, 4]) + np_random.uniform(-0.5, 0.5))
    # light.set_energy(500)
    # light.set_color([1, 1, 1])
    # lights.append(light)
    # return lights
    if len(lights) == 0:   
        for _ in range(num_lights):
            light = bproc.types.Light()
            # light.set_type("POINT")
            light.set_type("AREA")
            lights.append(light)
        # add one overhead fixed light
        light = bproc.types.Light()
        light.set_type("SUN")
        light.set_location(np.array([0, 0, 4.5]) + np_random.uniform(-0.5, 0.5))
        light.set_energy(20)
        light.set_color([1, 1, 1])
        lights.append(light)
    light_locations = np.array([
        [-0.1, CAM_DISTANCE, CAM_DISTANCE], 
        [0, -CAM_DISTANCE, CAM_DISTANCE], 
        [-0.5, -CAM_DISTANCE, -CAM_DISTANCE], 
        [0, CAM_DISTANCE, -CAM_DISTANCE]]
    )
    for j, light in enumerate(lights[:-1]): 
        # random location on a sphere segment
        light.set_location(bproc.sampler.shell(
            center=light_locations[j % 4],
            # radius_min=CAM_DISTANCE*0.75,
            # radius_max=CAM_DISTANCE*1.25,
            radius_min=CAM_DISTANCE*2,
            radius_max=CAM_DISTANCE*3,
            elevation_min=50,
            elevation_max=60,
        ))
        # random energy
        light.set_energy(np_random.uniform(10, 20))
        # random color
        # light.set_color(np_random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
        
    return lights 

def save_render_data(data, camera_infos, output_path, save_mask_png=False, try_load=True):
    """ regroup data by frame, then save individually """
    regroup = []
    num_frames = len(data["colors"])
    all_instance_attribute_maps = data.get("instance_attribute_maps", None)
    all_unique_ids = None 
    if all_instance_attribute_maps is not None:
        all_unique_ids = set()
        for i in range(num_frames):
            _maps = all_instance_attribute_maps[i]
            category_ids = [_map['category_id'] for _map in _maps]
            all_unique_ids.update(category_ids)

    for i in range(num_frames):
        frame_data = {k: v[i] for k, v in data.items()}
        camera_info = camera_infos[i]
        for key, val in camera_info.items():
            frame_data[key] = val.copy()
        if 'instance_attribute_maps' in frame_data:
            # _maps = frame_data['instance_attribute_maps']
            # category_ids = [_map['category_id'] for _map in _maps] # contains repeat!
            # category_ids = sorted(list(set(category_ids)))
            _mask = frame_data['class_segmaps'] # shape (h, w) for now
            binary_masks = []
            mask_ids = []
            for _id in all_unique_ids:
                binary_masks.append((_mask == _id).astype(np.uint8))
                mask_ids.append(_id)
            # frame_data['num_masks'] = len(binary_masks)
            frame_data['binary_masks'] = np.stack(binary_masks, axis=0) # shape (n, h, w)
            frame_data['mask_ids'] = np.array(mask_ids)
            # print("Saved {} masks".format(len(binary_masks)))
             
        regroup.append(frame_data)   
    txt_towrite = []
    # breakpoint()
    # -- WRITE --  
    print('hdf5 saving to:', output_path)
    for i, frame_data in enumerate(regroup):
        hdf5_path = join(output_path, f"{i}.hdf5")
        with h5py.File(hdf5_path, "w") as hfile:
            for key, data in frame_data.items():
                _WriterUtility.write_to_hdf_file(hfile, key, data) 

        if try_load:
            with h5py.File(hdf5_path, "r") as hfile:
                # print(hfile.keys())
                # print(hfile['colors'].shape) (h, w, 3)
                # print(hfile['binary_masks'].shape) (num_masks, h, w)
                num_masks = hfile['binary_masks'].shape[0]
                txt_towrite.append(
                    f"{hdf5_path} {num_masks}"
                )
                img = np.array(hfile['colors'])
                Image.fromarray(img).save(join(output_path, f"rgb_{i}.png"))
                if save_mask_png:
                    masks = np.array(hfile['binary_masks'])
                    for j, mask in enumerate(masks):
                        Image.fromarray(mask*255).save(join(output_path, f"frame_{i}_mask_{j}.png"))
    txt_fname = join(output_path, "num_masks.txt")
    with open(txt_fname, "w") as f:
        f.write("\n".join(txt_towrite))
    
    return 

def resample_cameras(
        num_frames, full_circle=False, np_random=np.random.RandomState(0), cam_distance=CAM_DISTANCE, 
        cam_height=CAM_HEIGHT, obj_center=np.array([0,0,0]), rotation_min=CAM_ROTATION_MIN, rotation_max=CAM_ROTATION_MAX
        ):
    # camera will look towards this point of interest
    # poi = bproc.object.compute_poi(obj.links[1].get_visuals()) # seems unreliable
    # poi = np_random.uniform(POI_RANGE_LOW, POI_RANGE_HIGH)
    poi = obj_center #+ np_random.uniform(-0.2, -0.1, 3)

    cam_azimuth = np.linspace(
        rotation_min + np_random.uniform(0, np.pi/6),
        rotation_max + np_random.uniform(-1*np.pi/6, 0),
        num_frames
        )
    if full_circle:
        cam_azimuth = np.linspace(0+ np_random.uniform(0, np.pi/6), 2*np.pi + np_random.uniform(-1*np.pi/6, 0), num_frames)

    cam_location = np.array([np.cos(cam_azimuth) * cam_distance,
                            np.sin(cam_azimuth) * cam_distance,
                            np.ones_like(cam_azimuth) * cam_height]).T

    # translational random walk (added as offset to poi)
    step_magnitude = np_random.uniform(0.04, 0.08)
    interval_low = np_random.uniform(-0.5, 0)
    interval_high = np_random.uniform(0, 0.5)
    poi_drift = bproc.sampler.random_walk(
        total_length=num_frames, dims=3, step_magnitude=step_magnitude, window_size=5, interval=[interval_low, interval_high], distribution='uniform'
        )
    frames = []
    for i in range(num_frames):
        # offset poi (look-at location) and rotate camera towards this new poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi + poi_drift[i] - cam_location[i])
        # get transformation matrix
        cam2world_matrix = bproc.math.build_transformation_mat(cam_location[i], rotation_matrix) 
        new_frame = bproc.camera.add_camera_pose(cam2world_matrix, frame=i)
        cam_fov = bproc.camera.get_fov()
        intrinsics = bproc.camera.get_intrinsics_as_K_matrix()
        frames.append(
                dict(
                    cam_id=np.array([new_frame]),
                    cam_fov=np.array([cam_fov[0], cam_fov[1]]),
                    cam_pose=cam2world_matrix,
                    cam_intrinsics=np.array(intrinsics),
                )
            ) 
    return frames

def merge_export_meshs(args, obj, fname, output_path, merged_mesh_folder = "blender_meshes"):
    """ merge all the meshes in each link into one mesh, remove the material, and save as .obj""" 
    os.makedirs(join(output_path, merged_mesh_folder), exist_ok=True)
    num_merged = 0
    for l, link in enumerate(obj.links):
        link_name = link.get_name()
        link_visuals = link.get_visuals() # [MeshObject]
        tojoin_objs = [link_obj.blender_obj for link_obj in link_visuals]
        print(f"Link #{link_name}, {len(tojoin_objs)} objects")
        if len(tojoin_objs) > 0:
            merged_name = link_name
            merged_mesh = trimesh.Trimesh()
            with context.temp_override(
                active_object=tojoin_objs[0], 
                # selected_editable_objects=tojoin_objs,
                selected_objects=tojoin_objs,
                ):
                # 
                for obj in tojoin_objs:
                    # bpy.context.view_layer.objects.active = obj
                    # bpy.ops.object.mode_set(mode='OBJECT')   
                    # Make a copy of the object and select it
                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                    # Add and apply the solidify modifier
                    bpy.ops.object.modifier_add(type='SOLIDIFY')
                    bpy.context.object.modifiers["Solidify"].thickness = 0.001  # Adjust thickness as needed
                    bpy.ops.object.modifier_apply(modifier="Solidify")

                    # Switch to OBJECT mode if not already
                    bpy.ops.object.mode_set(mode='OBJECT') 
 
                    mesh = bpy.context.object.data
                    verts = np.array([v.co for v in mesh.vertices])
                    # faces = np.array([p.vertices for p in mesh.polygons])
                    faces = []
                    for poly in mesh.polygons:
                        if len(poly.vertices) == 3:
                            faces.append(poly.vertices)
                        else:
                            for i in range(1, len(poly.vertices) - 1):
                                faces.append([poly.vertices[0], poly.vertices[i], poly.vertices[i + 1]]) 
   
                    # Create a Trimesh from the Blender mesh data
                    trimesh_obj = trimesh.Trimesh(vertices=verts, faces=faces) 
                    # Combine the Trimesh objects
                    merged_mesh = trimesh.util.concatenate(merged_mesh, trimesh_obj)
            # save the merged mesh as .obj:
            print('Watertight?:', merged_mesh.is_watertight)
            merged_mesh.export(join(output_path, merged_mesh_folder, f"{merged_name}.obj"), file_type="obj")
            num_merged += 1

    simplified = simplify_urdf(fname, mesh_dir=merged_mesh_folder)
    output_fname = join(output_path, "mobility_repaired.urdf")
    with open(output_fname, 'wb') as f:
        f.write(etree.tostring(simplified, pretty_print=True))
    print(f"saved {num_merged} merged meshes and repaired urdf to {output_fname}")
    return num_merged

def process_folder(args, folder):
    # bproc.clean_up()
    if len(folder.split('/')) > 1:
        folder = folder.split('/')[-1]
    folder_id = int(folder)
    lookup = join(args.data_dir, args.split, "*", folder, args.input_urdf)
    fname = glob(
        lookup
    )
    if len(fname) == 0:
        print(f'file {lookup} does not exist')
        return  
    fname = fname[0]
    obj_type = fname.split('/')[-3]
    
    np_random = np.random.RandomState(folder_id)
    output_path = join(args.out_dir, args.split, obj_type, folder)
    os.makedirs(output_path, exist_ok=True)
    if args.overwrite:
        for f in glob(join(output_path, 'loop_*', '*')):
            os.remove(f)
    render_loops = []
    for loop in range(args.num_loops):
        # save
        output_folder = join(output_path, f"loop_{loop}")
        if (not args.overwrite) and os.path.exists(output_folder):
            if len(glob(join(output_folder, '*.png'))) > 0:
                print(f"skipping {output_folder}") 
                continue
        render_loops.append(loop)
    if len(render_loops) == 0:
        print(f"skipping {output_path}")
        return
    
    open_drawer = True
    tmp_fname = clean_up_urdf(fname, np_random, open_drawer=open_drawer, margin=0.2) 
    obj = bproc.loader.load_urdf(urdf_file=tmp_fname)
    # get the size of obj     
    children_bbox = [] # each bound_box is (8, 3)
    for child in obj.get_children():
        mat = child.get_local2world_mat()
        global_corner = np.array(child.get_bound_box()) @ mat[:3, :3].T + mat[:3, 3]
        children_bbox.append(global_corner)
    children_bbox = np.array(children_bbox)
    min_bound = np.min(children_bbox[:, 0, :], axis=0)
    max_bound = np.max(children_bbox[:, 1, :], axis=0)
    obj_center = (min_bound + max_bound) / 2
    obj_size = max_bound - min_bound
    # breakpoint() 
    obj_volume = np.prod(obj_size)
    if obj_volume < 2:
        cam_dist, cam_height = 3.6, 0.3
    elif obj_volume < 3.5:
        cam_dist, cam_height = 3.9, 0.35
    elif obj_volume < 5:
        cam_dist, cam_height = 3.9, 0.6
    else: 
        cam_dist, cam_height = 3.7, 0.7

    if obj_type in CUSTOM_CAMERA_DIST_HEIGHT:
        print(f"Using custom camera dist and height for {obj_type}")
        cam_dist, cam_height = CUSTOM_CAMERA_DIST_HEIGHT[obj_type]
    
    rotation_min, rotation_max = CAM_ROTATION_MIN, CAM_ROTATION_MAX
    jnt_margin = 0.1
    joint_high = None
    if args.folder == '30666':
        cam_dist, cam_height = 3.6, 0.9
    if args.folder == "22367":
        cam_dist, cam_height = 3.3, 0.7 
        rotation_min, rotation_max = np.pi/2 + np.pi/3, 3*np.pi/2 - np.pi/5
    if args.folder == "25493":
        cam_dist, cam_height = 3.4, 0.8
    if args.folder == "26608":
        cam_dist, cam_height = 3.2, 0.5
    if args.folder == "45332":
        cam_dist, cam_height = 4.0, 0.9
        jnt_margin = 0.99
        rotation_min, rotation_max = np.pi/2 + np.pi/4, 3*np.pi/2 - np.pi/4
    if args.folder == "45662":
        jnt_margin = 0.9
    if obj_type == "Eyeglasses":
        print("Not bending the glasses joints too much")
        jnt_margin = 0.1
        joint_high = 0.2
    # breakpoint()
    print(f"\nobj center: {obj_center} obj volume: {obj_volume:.2f}, cam_dist: {cam_dist:.2f}, cam_height: {cam_height:.2f}\n")
    # breakpoint()
    # remove temporary file
    os.remove(tmp_fname)
    # add link id as 'category_id' for all objects belonging to that link
    obj.set_ascending_category_ids() 
    fix_meshes = []
    fix_blender_objs = []
    for l, link in enumerate(obj.links):
        for link_obj in link.get_visuals():
            if link_obj.blender_obj.data.validate():
                # print(f'had to fix {link_obj.blender_obj.name}')
                fix_meshes.append(link_obj.blender_obj.name) 
                fix_blender_objs.append(link_obj.blender_obj)
                # TODO: also resample materials here 
    # print(f"fixed {len(fix_meshes)} meshes")

    num_merged = merge_export_meshs(args, obj, fname, output_path)
    if args.merge_mesh_only: # skip rendering
        return

    # -- LIGHT --
    init_lights = resample_lights(NUM_LIGHTS, init_lights=[], np_random=np_random)
    # camera settings
    bproc.camera.set_resolution(RENDER_WIDTH, RENDER_HEIGHT)   
    bproc.renderer.enable_segmentation_output(map_by=["class"]) 
    if RENDER_DEPTH:
        # bproc.renderer.enable_depth_output(True, antialiasing_distance_max=10) -> gives bugs
        bproc.renderer.enable_depth_output(False) # NOTE this should be False! True gives bad depth maps
    if RENDER_NORMALS:
        bproc.renderer.enable_normals_output() 
     
    for loop in render_loops:
        haven_hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(args.haven_path) 
        # bproc.renderer.set_output_format(enable_transparency=True)
        if not args.render_bg:
            bpy.context.scene.render.film_transparent = True
        # bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)  # RGBA for white

        bproc.world.set_world_background_hdr_img(haven_hdri_path, strength=2.0)
        # save
        output_folder = join(output_path, f"loop_{loop}")
        if (not args.overwrite) and os.path.exists(output_folder):
            print(f"skipping {output_folder}")
            continue
        
        joint_rots = set_hinge_joints(obj, np_random, margin=jnt_margin, high_margin=joint_high)
        # resample lights
        init_lights = resample_lights(NUM_LIGHTS, init_lights=init_lights, np_random=np_random)
        # resample cameras
        center = (obj_center + np_random.uniform(-0.3, 0, 3))
        if args.full_circle:
            center = obj_center 
        cam_frames = resample_cameras(
            args.num_frames, args.full_circle, np_random=np_random, 
            cam_distance=cam_dist, cam_height=cam_height, obj_center=center,
            rotation_min=rotation_min, rotation_max=rotation_max
        )
        # render
        render_data = bproc.renderer.render() 
        os.makedirs(output_folder, exist_ok=True) 
        joint_rots_fname = join(output_folder, "joint_info.json")
        with open(joint_rots_fname, "w") as f:
            json.dump(joint_rots, f, indent=4)

        link_matrices = dict()
        for link in obj.links:
            link_name = link.get_name()
            matrix = link.get_visual_local2world_mats()
            if matrix is None:
                matrix = np.eye(4)
            else:
                matrix = np.array(matrix)
                matrix = np.round(matrix, 4)
            link_matrices[link_name] = matrix.tolist()
        mesh_transform_fname = join(output_folder, "mesh_transforms.json")
        with open(mesh_transform_fname, "w") as f:
            json.dump(link_matrices, f, indent=4)
        save_render_data(render_data, cam_frames, output_folder, save_mask_png=args.save_mask_png)
        
    print(f"=== saved {args.num_loops} loops to {output_path} ===")
    return 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/store/real/mandi/mobility_dataset_v2")
    parser.add_argument('--out_dir', type=str, default="/local/real/mandi/blender_dataset_v5")
    parser.add_argument('--split', type=str, default="test") 
    parser.add_argument('--folder', type=str, default="46172") 
    parser.add_argument('--overwrite', "-o", action="store_true")
    parser.add_argument('--num_loops', type=int, default=1)
    parser.add_argument('--save_mask_png', action="store_true")
    parser.add_argument('--folder_idx_left', '-l', type=int, default=0)
    parser.add_argument('--folder_idx_right', '-r', type=int, default=-1)
    parser.add_argument('--merge_mesh_only', '-m', action="store_true")
    parser.add_argument('--input_urdf', type=str, default="mobility.urdf") 
    parser.add_argument('--num_frames', type=int, default=2)
    parser.add_argument('--full_circle', action="store_true")
    parser.add_argument('--haven_path', nargs='?', default="/local/real/mandi/", help="The folder where the `hdri` folder can be found, to load an world environment")
    parser.add_argument('--render_bg', action="store_true")
    args = parser.parse_args()
    bproc.init()
    process_folder(args, args.folder)
    exit()
 
