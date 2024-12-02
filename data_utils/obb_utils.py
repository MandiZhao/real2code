import os 
import open3d as o3d
import numpy as np
from os.path import join
import json
from glob import glob 
from natsort import natsorted 
import h5py 
from matplotlib import pyplot as plt
from lxml import etree
from pyquaternion import Quaternion
import argparse
from copy import deepcopy
from data_utils.mjcf_utils import translate_asset, translate_body
from data_utils.obb_helper_functions import GET_HELPER_CODE

def obb_from_axis(points: np.ndarray, axis_idx: int):
    """get the oriented bounding box from a set of points and a pre-defined axis"""
    # Compute the centroid, points shape: (N, 3)
    centroid = np.mean(points, axis=0)
    # Align points with the fixed axis idx ([1, 0, 0]), so ignore x-coordinates
    if axis_idx == 0:
        points_aligned = points[:, 1:]
        axis_1 = np.array([1, 0, 0])
    elif axis_idx == 1:
        points_aligned = points[:, [0, 2]]
        axis_1 = np.array([0, 1, 0])
    elif axis_idx == 2:
        points_aligned = points[:, :2]
        axis_1 = np.array([0, 0, 1])
    else:  
        raise ValueError(f"axis_idx {axis_idx} not supported!") 

    # Compute PCA on the aligned points
    points_centered = points_aligned - np.mean(points_aligned, axis=0)  
    cov = np.cov(points_centered.T)
    _, vh = np.linalg.eig(cov)
    axis_2, axis_3 = vh[:, 0], vh[:, 1] # 2D!!
    # axis_2, axis_3 = vh[0], vh[1] # 2D!! 
    axis_2, axis_3 = np.round(axis_2, 1), np.round(axis_3, 1)  
    x2, y2 = axis_2
    x3, y3 = axis_3 
    
    if sum(axis_2 < 0) == 2 or (sum(axis_2 < 0) == 1 and sum(axis_2 == 0) == 1):
        axis_2 = -axis_2
    if sum(axis_3 < 0) == 2 or (sum(axis_3 < 0) == 1 and sum(axis_3 == 0) == 1):
        axis_3 = -axis_3

    # remove -0
    axis_2 = np.array([0. if x == -0. else x for x in axis_2])
    axis_3 = np.array([0. if x == -0. else x for x in axis_3]) 
    if axis_idx == 0:
        evec = np.array([
            axis_1,
            [0, axis_2[0], axis_2[1]],
            [0, axis_3[0], axis_3[1]]
            ]).T
    elif axis_idx == 1:
        evec = np.array([
            [axis_2[0], 0, axis_2[1]],
            axis_1,
            [axis_3[0], 0, axis_3[1]]
            ]).T 
    elif axis_idx == 2:
        evec = np.array([
            [axis_2[0], axis_2[1], 0],
            [axis_3[0], axis_3[1], 0],
            axis_1,
            ]).T 
    # Use these axes to find the extents of the OBB
    # # Project points onto these axes 
    all_centered = points - centroid # (N, 3)
    projection = all_centered @ evec # (N, 3) @ (3, 3) -> (N, 3)

    # Find min and max projections to get the extents
    _min = np.min(projection, axis=0)
    _max = np.max(projection, axis=0)
    extent = (_max - _min) # / 2 -> o3d takes full length
    # Construct the OBB using the centroid, axes, and extents 
 
    return dict(center=centroid, R=evec, extent=extent)

def get_tight_obb(mesh, z_weight=1.5):
    all_obbs = []
    if isinstance(mesh, np.ndarray):
        vertices = mesh    
    else:
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_faces() 
        vertices = np.array(mesh.vertices) 
    if len(vertices) == 0:
        return dict(center=np.zeros(3), R=np.eye(3), extent=np.ones(3))
    for axis_idx in range(3):
        obb_dict = obb_from_axis(vertices, axis_idx)
        all_obbs.append(obb_dict)

    # select obb with smallest volume, but prioritize axis z 
    bbox_sizes = [np.prod(x['extent']) for x in all_obbs] 
    bbox_sizes[2] /= z_weight # prioritize z axis 
    min_size_idx  = np.argmin(bbox_sizes)
    obb_dict = all_obbs[min_size_idx]
    return obb_dict
  
def get_obb_from_loop(loop_dir, width=512, height=512, depth_trunc=10, vlength=0.01, z_weight=1.5):
    """ Load RGBD data from loop_dir and get per-part OBBs"""
    h5s = natsorted(glob(join(loop_dir, "*hdf5")))
    volumes = [] 
    num_masks = []
    for h5_file in h5s: 
        with h5py.File(h5_file, "r") as f:  
            binary_masks = f["binary_masks"][:]
        num_masks.append(len(binary_masks))
    max_num_masks = max(num_masks)  
    # NOTE in the latest rendering code, there should still be an empty mask even if the part is not visible from that angle
    for h5_file in h5s: 
        with h5py.File(h5_file, "r") as f: 
            rgb = f["colors"][:]
            depth = f["depth"][:]
            binary_masks = f["binary_masks"][:]
            camera_intrinsics = np.array(f['cam_intrinsics'])
            camera_pose = np.array(f['cam_pose'])
        if len(binary_masks) < max_num_masks:
            # need to skip because mask order is messed up 
            continue
        obj_masks = binary_masks[1:] # skip the first background mask!
        if len(volumes) == 0:
            volumes = [
                o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=vlength, 
                    sdf_trunc=(2 * vlength),
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                    )
                for _ in obj_masks
            ]
        
        intrinsics = o3d.camera.PinholeCameraIntrinsic() 
        intrinsics.set_intrinsics(
            width=width, height=height, 
            fx=camera_intrinsics[0, 0], 
            fy=camera_intrinsics[1, 1], cx=(width / 2), cy=(height / 2))
        extrinsics = np.linalg.inv(camera_pose)
        # NOTE requires inverting here!!
        extrinsics[2,:] *= -1
        extrinsics[1,:] *= -1 
        depth[depth>depth_trunc] = depth_trunc 

        for idx, mask in enumerate(obj_masks): 
            rgb_copy = rgb.copy()
            depth_copy = depth.copy()
            rgb_copy[mask == 0] = 0
            depth_copy[mask == 0] = 0  
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb_copy), 
                    o3d.geometry.Image(depth_copy),
                    depth_trunc=depth_trunc, 
                    depth_scale=1,
                    convert_rgb_to_intensity=False,
            )
            volumes[idx].integrate(
                    rgbd,
                    intrinsics,
                    extrinsics,
                    )
    
    all_obbs = []
    o3d_vis = []
    for volume in volumes:
        pcd = volume.extract_point_cloud()
        _pcd = pcd.voxel_down_sample(0.02)
        # _pcd, _ = pcd.remove_statistical_outlier(100, 1)  
        obb_dict = get_tight_obb(np.array(_pcd.points), z_weight=z_weight)
        bbox = o3d.geometry.OrientedBoundingBox(**obb_dict)
        
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
        all_obbs.append(
            dict(
                center=np.array(bbox.get_center()), 
                box_points=np.array(bbox.get_box_points()), 
                extent=np.array(bbox.extent),
                R=np.array(bbox.R),
                lines=np.array(lineset.lines),
                )
            ) 
        o3d_vis.extend([_pcd, bbox])  
    return volumes, all_obbs, o3d_vis    

def find_all_joints(root):
    all_part_bodys = root.find("worldbody").find("body").findall("body")
    all_joint_elems = []
    for part_body in all_part_bodys:
        all_joint_elems.extend(part_body.findall('joint'))
    all_joints = dict()
    for j, joint_elem in enumerate(all_joint_elems):
        name = joint_elem.attrib.get("name", f"{j}_joint") 
        axis = joint_elem.attrib.get("axis", "1 0 0")
        pos = joint_elem.attrib.get("pos", "0 0 0")
        all_joints[name] = dict(
            type=joint_elem.attrib.get("type", "hinge"),
            axis=np.array([float(x) for x in axis.split(" ")]),
            pos=np.array([float(x) for x in pos.split(" ")]),
            elem=joint_elem,
        )
    return all_joints 

def show_obb_pyplot(
    obbs, 
    joints, 
    show_edge=False, 
    show_joints=False, 
    save_img_fname='test.png',
    view_angles=dict(azim=150, elev=5),
    skip_show=False,
):
    # fig = plt.figure(figsize=(4,4))
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(680*px, 680*px))
    ax = fig.add_subplot(111, projection='3d') 
    ax.view_init(**view_angles)
    colors = 'rgb'
    axis = np.eye(3)
    
    if 'box_points' not in obbs[0]:
        # convert into o3d_obbs first
        o3d_obbs = []
        for obb in obbs:
            o3d_obb = o3d.geometry.OrientedBoundingBox(
                center=obb['center'], 
                extent=obb['extent'],
                R=obb['R']
            )
            o3d_obbs.append(
                dict(
                    center=np.array(o3d_obb.get_center()),
                    extent=np.array(o3d_obb.extent),
                    R=np.array(o3d_obb.R),
                    box_points=np.array(o3d_obb.get_box_points()),
                    lines=np.array(o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_obb).lines),
                )
            )
        obbs = o3d_obbs
    for link_id, obb in enumerate(obbs):
        center = np.array(obb["center"])
        half_lengths = np.array(obb["extent"]) / 2
        rot = np.array(obb["R"])
        box_points = obb["box_points"]
        ax.scatter(center[0], center[1], center[2], c='g') 
        # draw half-lengths 
        for i in range(3): 
            x0, y0, z0 = center
            rot_axis = rot[:, i]
            x1, y1, z1 = center + rot_axis * half_lengths[i]
            ax.plot([x0, x1], [y0, y1], [z0, z1], c=colors[i]) 

        # # draw the box 
        # ax.scatter(box_points[:,0], box_points[:,1], box_points[:,2], c='r')
        lines = obb["lines"]
        for line in lines:
            x0, y0, z0 = box_points[line[0]]
            x1, y1, z1 = box_points[line[1]]
            ax.plot([x0, x1], [y0, y1], [z0, z1], c='b', alpha=0.5)
 
        if len(joints) > 0 and joints[link_id] is not None and show_joints:
            info = joints[link_id]
            gt_pos = info['pos']
            gt_axis = info['axis']
            x0, y0, z0 = gt_pos
            x1, y1, z1 = gt_pos + gt_axis * 0.5
            ax.plot([x0, x1], [y0, y1], [z0, z1], c='orange', linewidth=3)

            bbox_edge, bbox_axis = info['obb_edge'], info['obb_axis']
            axis_idx = info['axes_idx']
            if show_edge:
                # scatter all tbe edge positions
                edges = [np.insert(np.array(edge), axis_idx, 0) for edge in [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]] 
                for edge in edges: 
                    edge_pos = center + rot @ (edge * half_lengths)
                    ax.scatter(edge_pos[0], edge_pos[1], edge_pos[2], c='r')
            # draw the axis and posv 
            edge_pos = center + rot @ (bbox_edge * half_lengths)
            # print("edge_pos", edge_pos, "bbox_axis", bbox_axis, "half_lengths", half_lengths)
            x0, y0, z0 = edge_pos
            x1, y1, z1 = edge_pos + bbox_axis * half_lengths[axis_idx]
            ax.plot([x0, x1], [y0, y1], [z0, z1], c='black', linewidth=3)
    # set ticks on each axis to be the same
    ticks = np.arange(-1, 1.1, 0.5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    # set limits 
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    # plt.tight_layout()
    if save_img_fname is not None:
        plt.savefig(save_img_fname) # dpi=300)
    if not skip_show:
        plt.show()
        breakpoint()
    return

def angle_between_vectors(v1, v2):
    """angle between two vectors in radians"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
 
def relate_joint_to_obb(obb_center, obb_axes, obb_half_lengths, joint_pos, joint_axis):
    """
    Given absolute joint params, find params relative to the given OBB
    return:
        rot_idx: index of the axis parallel to the joint axis
        rot_axis: the select 1 axis in parallel to the joint axis
        edge: the 1 out of 4 edges closest to the joint pos
        sign: the sign of the joint axis
        rel_pos: the 2D joint pos relative to the obb center
    Given 3 obb axes & 4 edges along each axis, find the axis parallel to joint axis and the edge closest to the joint pos """
    angles = []
    for i in range(3):
        rot_axis = obb_axes[:, i]
        for sign in [1, -1]:
            # angle between joint axis and obb axis
            pos_angle = angle_between_vectors(joint_axis, sign * rot_axis)
            angles.append((pos_angle, i, sign))
    min_angle, rot_idx, sign = min(angles, key=lambda x: x[0]) 
    edges = [np.insert(np.array(edge), rot_idx, 0) for edge in [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]]
    edge_dists = []
    for edge in edges: 
        edge_pos = obb_center + obb_axes @ (edge * obb_half_lengths)
        edge_dists.append( (np.linalg.norm(edge_pos - joint_pos), edge) )
    min_dist, min_edge = min(edge_dists, key=lambda x: x[0])
    # find the 2D pos of the absolute joint pos along the selected obb rotation axis
    rel_3d_pos = joint_pos - obb_center
    rel_pos = np.delete(rel_3d_pos, rot_idx)
    return rot_idx, obb_axes[:, rot_idx], min_edge, sign, rel_pos

def get_mjcf_import_lines(mesh_dir):
    return [
    "import numpy as np",
    "from dm_control import mjcf",  
    "model = mjcf.RootElement(model='object')",
    f"model.compiler.meshdir = '{mesh_dir}'",
    "model.compiler.angle = 'radian'",
    "model.compiler.autolimits = 'true'"
    ]

def process_one_loop_obb_joints(xml_root, loop_folder):
    """ 
    Find the OBBs in the current loop and relate joints to them
    Output: 
    - relative_joints: dict of joints, with info on how they are related to the OBBs
    """
    global_joints = find_all_joints(xml_root)
    volumes, all_obbs, o3d_vis = get_obb_from_loop(loop_folder)
    relative_joints = dict() 
    bbox_listed = dict()
    for link_id, obb in enumerate(all_obbs):
        joint_name = f"joint_{link_id}"
        bbox_name = f"bbox_{link_id}"
        if joint_name in global_joints:
            joint_info = global_joints[joint_name]
            rot_idx, obb_axis, edge, sign, rel_pos = relate_joint_to_obb(
                obb['center'], obb['R'], obb['extent']/2, joint_info['pos'], joint_info['axis']
            )
            relative_joints[joint_name] = dict(
                bbox_name=bbox_name,
                link_id=link_id,
                type=joint_info['type'],
                axis=joint_info['axis'],
                pos=joint_info['pos'],
                rot_idx=rot_idx, 
                obb_axis=obb_axis, 
                edge=edge, 
                sign=sign, 
                xy_pos=rel_pos, 
            )
        listed = dict(
            center=np.round(obb['center'], 2).tolist(), 
            R=np.round(obb['R'], 2).tolist(),
            extent=np.round(obb['extent'], 2).tolist()
        )  
        bbox_listed[link_id] = listed 
    return relative_joints, all_obbs, bbox_listed

def parse_joint_line(line):
    vals = dict(pos=None, axis=None, type=None, range=None)
    for key in ["pos", "axis", "type", "range"]:
        if key+"=" in line:
            if key in ['range', 'axis']:
                string = line.split(f"{key}=[")[1].split("]")[0]
                vals[key] = f"[{string}]"
            else:    
                vals[key] = line.split(f"{key}=")[1].split(",")[0]
    vals['body'] = line.split(".add")[0] 
    vals['name'] = line.split("name=")[1].split(",")[0]
    return vals

def rewrite_helper_line(
        joint_pos, joint_axis, joint_type, 
        bbox_id, axis_idx, bbox_edge, sign, xy_pos, mode="obb_rel"
    ):
    global_pos, global_axis = np.round(joint_pos, 2).tolist(), np.round(joint_axis, 1).tolist()    
    xy_pos = np.round(xy_pos, 2).tolist()
    if mode == "absolute":
        global_pos = f"[{global_pos[0]},{global_pos[1]},{global_pos[2]}]"
        global_axis = f"[{global_axis[0]},{global_axis[1]},{global_axis[2]}]"
        new_line = f"dict(box={bbox_id},type={joint_type},pos={global_pos},axis={global_axis}),"
    elif mode == "obb_rot":
        new_line = f"dict(box={bbox_id},type={joint_type},idx={int(axis_idx)},edge={xy_pos},sign={sign}),"
    elif mode == "obb_rel":
        new_line = f"dict(box={bbox_id},type={joint_type},idx={int(axis_idx)},edge={bbox_edge},sign={sign}),"
    else:
        raise ValueError(f"mode {mode} not supported!")
    return new_line

def translate_dict_to_code(bbox_dict):
    bbox_lines = ["bboxes={"]
    for bbox_id, info_listed in bbox_dict.items():
        line = str(info_listed)
        line = line.replace(" ", "") 
        new_line = f"{bbox_id}:" + line + ","
        bbox_lines.append(new_line)
    bbox_lines.append("}")
    return bbox_lines

def rewrite_obb_code(xml_root, relative_joints, bbox_listed, mesh_dir, mode="absolute"):
    """
    Takes in original offsetted.xml root and re-write the joints based on OBB info
    Output: {
        'bbox_code': GT values of OBB header, need to be augmented (i.e. rotated) for LLM train/fine-tuning
        'label_code': GT code useful during testing,
        'full_code': stand-alone script that can be executed directly, inclues import lines, body_root lines, inertial, and define and run helper function line
    } 
    """
    assert mode in ["absolute", "obb_rot", "obb_rel"], f"mode {mode} not supported"
    import_lines = get_mjcf_import_lines(mesh_dir)
    asset_lines = translate_asset(xml_root)
    body_root_lines = [
        "body_root = model.worldbody.add('body', name='root')",
        "body_root.add('inertial', mass=1, pos=[0, 0, 0], diaginertia=[1, 1, 1])"
        ]
    bbox_lines = translate_dict_to_code(bbox_listed)
    
    header_lines = import_lines + asset_lines + body_root_lines + bbox_lines 

    body_strs = translate_body(
        xml_root.find('worldbody'), 
        parent="model.worldbody", 
        body_bboxs=dict(),
        include_all_geom=True,
        )
    # find the geom used for root body
    root_geom_line = [l for l in body_strs if "add('geom'" in l][0]
    root_geom_id = root_geom_line.split("mesh='link_")[1].split("'")[0]

    adjusted_lines = [
        f"root_geom = {root_geom_id}",
        "child_joints = ["
        ]
    # now, rewrite the joint lines with obb info
    for line in body_strs:
        if "add('joint'" in line: # skips all inertial lines
            joint_name = line.split("name='")[1].split("'")[0]
            vals = parse_joint_line(line)
            if joint_name in relative_joints:
                joint_info = relative_joints[joint_name]
                bbox_name = joint_info["bbox_name"]  
                bbox_edge = joint_info["edge"].tolist()  
                rot_idx = joint_info["rot_idx"]
                # make the edge 2-dim
                bbox_edge = [val for i, val in enumerate(bbox_edge) if i != rot_idx]
                bbox_edge = f"[{bbox_edge[0]},{bbox_edge[1]}]"
                
                joint_type = vals.get('type', None)
                joint_type = "'hinge'" if joint_type is None else joint_type
                
                assert joint_type in ["'hinge'", "'slide'"], f"joint type {joint_type} not supported!"
                if joint_type == "'slide'": 
                    bbox_edge = "[+1,+1]"
                # remove range pred. 
                bbox_id = bbox_name.split("bbox_")[1]
                new_line = rewrite_helper_line(
                    joint_info['pos'], joint_info['axis'], joint_type, 
                    bbox_id, rot_idx, bbox_edge, joint_info['sign'], joint_info['xy_pos'], mode=mode
                )
            else:
                continue
            adjusted_lines.append(new_line)
    adjusted_lines.append("]")

    helper_code = GET_HELPER_CODE.get(mode, None)
    assert helper_code is not None, f"mode {mode} not supported!"
    return dict(
        bbox_code="\n".join(bbox_lines),
        label_code="\n".join(adjusted_lines),
        full_code="\n".join(header_lines + adjusted_lines + [helper_code]),
    )

def get_relative_joint_code(
        obj_folder, save_vis=True, 
        loop_lookup="*", mesh_folder="blender_meshes", 
        input_fname="offsetted.xml", overwrite=False,
    ):
    """
    Get OBBs and joint params from the offsetted.xml file
    NOTE: makes strong assumption that joints/object parts are in the same order as the rendered part masks
    """
    assert os.path.exists(join(obj_folder, input_fname)), f"{input_fname} not found in {obj_folder}"
    tree = etree.parse(join(obj_folder, input_fname))
    root = tree.getroot()
    loop_dirs = natsorted(glob(join(obj_folder, f"loop_{loop_lookup}")))
    saved_infos = dict()
    for loop_folder in loop_dirs:
        loop_root = deepcopy(root)  
        loop_id = loop_folder.split("loop_")[-1]
        # rewrite the joint lines with obb info
        mesh_dir = join(loop_folder, mesh_folder)
        code_dicts = dict() 
        for mode in ["absolute", "obb_rot", "obb_rel"]:
            os.makedirs(join(obj_folder, mode), exist_ok=True)
            save_fname = join(obj_folder, mode, f"info_loop_{loop_id}.json")
            if not overwrite and os.path.exists(save_fname):
                with open(save_fname, "r") as f:
                    code_dicts[mode] = json.load(f)
                continue
            relative_joints, all_obbs, bbox_listed = process_one_loop_obb_joints(loop_root, loop_folder)
            code_dicts[mode] = rewrite_obb_code(loop_root, relative_joints, bbox_listed, mesh_dir, mode=mode)
            code_dicts[mode].update(dict(loop_id=loop_id, bboxes=bbox_listed))
            with open(save_fname, "w") as f:
                json.dump(code_dicts[mode], f, indent=4) 
                
            if save_vis and mode == "absolute":
                show_obb_pyplot(
                        all_obbs, 
                        [relative_joints.get(f"joint_{i}", None) for i in range(len(all_obbs))], 
                        save_img_fname=f"{obj_folder}/obb_loop_{loop_id}.png", # this is optimized for view 2!
                        skip_show=True
                )
        saved_infos[f"loop_{loop_id}"] = code_dicts

    return saved_infos