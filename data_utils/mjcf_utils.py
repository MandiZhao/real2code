"""
Given object xml, reverse-generate the corresponding mjcf code

desired output:

model = mjcf.RootElement(name="35059")
link_0 = model.worldbody.add('body', name='link_0', pos='0 0 0', quat='1 0 0 0')
segment_body_geoms(link_0, segmented_images)
link_1 = link_0.add('body', name='link_1', pos='0 0 0', quat='1 0 0 0')
"""

import os 
from os.path import join
import json
import argparse
from dm_control import mjcf
from dm_control import mujoco as dm_mujoco
import mujoco 
import lxml
from lxml import etree
import numpy as np
from pyquaternion import Quaternion

def compose_add_str(parent, child, elem, keys, child_name=None):
    attrib_str = []
    for key in keys:
        val = elem.attrib.get(key) 
        if val is not None:
            if key in ["name", "type", "file", "mesh", "mass"]:
                val = f"'{val}'" 
            else:
                assert key in ['pos', 'quat', 'axis', 'range', 'size', 'rgba', 'diaginertia'], f"key {key} not implemented"
                if child == 'inertial': # simplify the numbers 
                    val = [f"{int(v)}" for v in val.split(" ")] 
                else:
                    val = [f"{float(v):.2f}" for v in val.split(" ")] 
                val = "[" + ", ".join(val) + "]" # this converts to [1.00, 0.00, 0.00], avoids ['1.00', '0.00', '0.00']  
            attrib_str.append(f"{key}={val}")
    attrib_str = ", ".join(attrib_str)
    code_str = f"{parent}.add('{child}', {attrib_str})"
    if child_name is not None:
        code_str = f"{child_name} = {code_str}"
    return code_str

def joint_elem_to_mjcf(elem, parent):
    keys = ['name', 'pos', 'axis', 'type', 'range']
    return compose_add_str(parent, "joint", elem, keys)

def geom_elem_to_mjcf(elem, parent):
    keys = ['name', 'type', 'pos', 'quat', 'size', 'rgba', 'mesh']
    return compose_add_str(parent, "geom", elem, keys)

def translate_body(
    parent_elem, 
    parent="model.worldbody",
    body_bboxs=dict(),
    include_all_geom=False,
    ):
    """recursively map the body elems into mjcf code strings, do joint, geom, childbody """
    code_strs = []
    if parent_elem.find('inertial') is not None:
        inert_elem = parent_elem.find('inertial')
        keys = ['pos', 'mass', 'diaginertia']
        code_strs.append( 
            compose_add_str(parent, "inertial", inert_elem, keys)
        )
    if len(parent_elem.findall("joint")) > 0:
        for joint_elem in parent_elem.findall("joint"):
            code_strs.append(
                joint_elem_to_mjcf(joint_elem, parent)
                )
    if include_all_geom and len(parent_elem.findall("geom")) > 0:
        for geom_elem in parent_elem.findall("geom"):
            code_strs.append(
                geom_elem_to_mjcf(geom_elem, parent)
                )
    

    body_name = parent_elem.attrib.get("name", None) 
    if body_name is not None: #and 'link' in parent_elem.attrib.get("name"):
        bbox = body_bboxs.get(body_name, None) # xyz, xyz
        if bbox is not None:
            pos = np.mean(bbox, axis=0)
            parent_pos = parent_elem.attrib.get("pos", "0 0 0")
            parent_pos = [float(v) for v in parent_pos.split(" ")]
            parent_quat = parent_elem.attrib.get("quat", "1 0 0 0")
            parent_quat = [float(v) for v in parent_quat.split(" ")]


            rel_pos = [pos[i] - parent_pos[i] for i in range(3)]
            _pos = Quaternion(parent_quat).inverse.rotate(rel_pos)
            x, y, z = _pos

            size = (np.max(bbox, axis=0) - np.min(bbox, axis=0)) / 2 # NOTE: mujoco uses half-size for box geoms
            h, w, l = size 
            
            _quat = Quaternion([1,0,0,0]) * Quaternion(parent_quat).inverse.elements
            code_strs.append(
                f"{parent}.add('geom', name='{body_name}_geom_approx', type='box', pos=[{x:.2f}, {y:.2f}, {z:.2f}], size=[{h:.2f}, {w:.2f}, {l:.2f}], quat=[{_quat[0]:.2f}, {_quat[1]:.2f}, {_quat[2]:.2f}, {_quat[3]:.2f}], rgba=[1, 1, 1, 1])"
            ) 
            
    if len(parent_elem.findall("body")) > 0:
        for body_elem in parent_elem.findall("body"):
            body_keys = ['name', 'pos', 'quat']
            body_name = body_elem.attrib.get("name", None)
            
            body_name = "body_" + body_name
            code_strs.append("") # leave one empty line per body
            code_strs.append(
                compose_add_str(parent, "body", body_elem, body_keys, child_name=body_name)
            )
            code_strs.extend(
                translate_body(
                    parent_elem=body_elem, 
                    parent=body_name, 
                    body_bboxs=body_bboxs, 
                    include_all_geom=include_all_geom,
                ))
    return code_strs

def translate_asset(root, parent="model"):
    """ map the asset elems into mjcf code strings """
    asset_elem = root.find("asset")
    if asset_elem is None:
        return []
    code_strs = []
    for mesh_elem in asset_elem.findall("mesh"):
        mesh_fname = mesh_elem.attrib["file"]
        mesh_name = mesh_elem.attrib["name"]
        if "refquat" in mesh_elem.attrib:
            code_str = f"{parent}.asset.add('mesh', name='{mesh_name}', file='{mesh_fname}', refquat='{mesh_elem.attrib['refquat']}')"
        else:
            code_str = f"{parent}.asset.add('mesh', name='{mesh_name}', file='{mesh_fname}')"
        code_strs.append(code_str)
    return code_strs


def translate_xml_to_mjcf(
    data_dir,
    folder, 
    input_fname, 
    output_fname,
    body_bboxs=dict(),
    include_all_geom=False,
    ):
    input_fname = join(data_dir, folder, input_fname)
    assert os.path.exists(input_fname)
    print(f"processing {input_fname}")
    # obj_id = "object_" + folder
    obj_id = "object"
    mesh_dir = join(data_dir, folder, "merged_objs")
    asset_fname = join(data_dir, folder, "assets.xml")
    etree_parsed = etree.parse(input_fname)
    root = etree_parsed.getroot()

    physics = dm_mujoco.Physics.from_xml_path(input_fname)
    
    tosave = {}
    mjcf_lines = [
        "from dm_control import mjcf",
        # "from mujoco import viewer",
        f"model = mjcf.RootElement(model='{obj_id}')",
        # f"model.compiler.meshdir = '{mesh_dir}'",
        # f"model.compiler.angle = 'radian'",
        # f"model.compiler.autolimits = 'true'"
        # f"model.add('include', file = '{asset_fname}')", BUG: donno how to add include
        ]
    tosave["mjcf_header"] = "\n".join(mjcf_lines)
    
    asset_strs = []
    if include_all_geom:
        asset_strs = translate_asset(root, parent="model")
    tosave["mjcf_asset"] = "\n".join(asset_strs)

    body_strs = translate_body(
        root.find('worldbody'), 
        parent="model.worldbody", 
        body_bboxs=body_bboxs,
        include_all_geom=False,
        )
    tosave["mjcf_body"] = "\n".join(body_strs)

    mjcf_lines.extend(asset_strs)
    mjcf_lines.extend(body_strs)
    mjcf_code = "\n".join(mjcf_lines)

    # remove all the folder IDs 
    mjcf_code = mjcf_code.replace(str(folder), "object")
    tosave["mjcf_code"] = mjcf_code

    # breakpoint()
    visualize_code = f"""
physics = mjcf.Physics.from_mjcf_model(model) 
_mjmodel = physics.model._model 
viewer.launch(_mjmodel)
"""
    mjcf_code = mjcf_code + "\n" + visualize_code
    tosave["full_code"] = mjcf_code
    # print(mjcf_code)
    # breakpoint()
    # try exporting the mjcf code into python:
    # exec(mjcf_code)

    output_fname = join(data_dir, folder, output_fname)
    # save as json 
    with open(output_fname, "w") as f:
        json.dump(tosave, f, indent=4)
    return tosave
