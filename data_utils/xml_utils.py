import os 
from os.path import join
import lxml
from lxml import etree
import dm_control
import mujoco
from dm_control import mujoco as dm_mujoco
from dm_control import mjcf
from copy import deepcopy
import numpy as np
from pyquaternion import Quaternion
from data_utils.mjcf_utils import translate_asset, translate_body

def raw_urdf_to_merged_xml(
        urdf_fname="data/mobility_repaired.urdf", 
        overwrite=False,
        output_fname="repaired.xml",
        try_load=True,
    ):
    object_path = os.path.dirname(urdf_fname)
    xml_fname = urdf_fname.replace(".urdf", ".xml") 
    if not os.path.exists(xml_fname):
        try:
            model = mujoco.MjModel.from_xml_path(urdf_fname)
            # print(f"Successfully loaded {urdf_fname} with dm_mujoco")
            # export to xml
            mujoco.mj_saveLastXML(xml_fname, model)
        except Exception as e:
            print(f"Failed to load {urdf_fname} with mujoco, error:")
            print(e)
            return None
    model = mujoco.MjModel.from_xml_path(xml_fname)
    parsed_xml = etree.parse(xml_fname)
    new_fname = join(object_path, output_fname)
    if not os.path.exists(new_fname) or overwrite: 
        root = parsed_xml.getroot()
        old_worldbody = root.find("worldbody") 
        new_root = etree.Element("mujoco", model=f"object") # NOTE: remove obj id to prevent memorization
        for _type in ['compiler', 'asset']:
            _elem = root.find(_type)
            if _elem is not None:
                new_root.append(deepcopy(_elem))  
        worldbody = etree.SubElement(new_root, "worldbody")            
        obj_body = etree.SubElement(worldbody, "body", name="root") # object_id)
        # add inertia <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        inert_elem = etree.SubElement(obj_body, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")

        geoms = old_worldbody.findall("geom")
        for _geom in geoms: 
            obj_body.append(_geom)
        
        for body in old_worldbody.findall("body"):
            new_child_body = etree.SubElement(obj_body, "body")
            inert_elem = etree.SubElement(new_child_body, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
            # copy all the attributes:
            for key, val in body.attrib.items():
                if 'gravcomp' in key:
                    # this causes loading bug from mujoco
                    continue
                new_child_body.attrib[key] = val
            
            for _geom in body.findall("geom"):  
                new_child_body.append(_geom)
            for joint in body.findall("joint"):
                new_child_body.append(joint)
        
        new_xml = etree.ElementTree(new_root)
        etree.indent(new_xml, space="  ")
        new_xml.write(new_fname, pretty_print=True)
    
    if try_load:
        try:
            model = mujoco.MjModel.from_xml_path(new_fname)
            physics = dm_mujoco.Physics.from_xml_path(new_fname)
            # print(f"Successfully saved and loaded {new_fname} with dm_mujoco") 
        except Exception as e:
            print(f"Failed to load {new_fname} with dm_mujoco, error:")
            print(e)
            return None
    return new_fname

def repaired_xml_to_mjcf(
        xml_fname="data/repaired.xml", 
        output_fname="mjcf_code.py", 
        mesh_folder="blender_meshes",
        overwrite=False,
    ):
    obj_path = os.path.dirname(xml_fname)
    new_fname = join(obj_path, output_fname)
    if not overwrite and os.path.exists(new_fname):
        return new_fname

    new_root = etree.parse(xml_fname).getroot()
    mjcf_lines = [
        "from dm_control import mjcf", 
        "from mujoco import viewer",
        f"model = mjcf.RootElement(model='object')",
        f"model.compiler.meshdir = '{mesh_folder}'",
        f"model.compiler.angle = 'radian'",
        f"model.compiler.autolimits = 'true'"
    ]
    asset_strs = translate_asset(new_root, parent="model")
    mjcf_lines.extend(asset_strs)
    body_strs = translate_body(
        new_root.find('worldbody'), 
        parent="model.worldbody", 
        body_bboxs=dict(),
        include_all_geom=True,
        )
    mjcf_lines.extend(body_strs)
    mjcf_lines.extend([
        "physics = mjcf.Physics.from_mjcf_model(model)",
        "_mjmodel = physics.model._model",
        "viewer.launch(_mjmodel)"
    ])
    with open(new_fname, "w") as f:
        f.write("\n".join(mjcf_lines))
    return new_fname

def get_offset_xml(input_fname, output_fname="offsetted.xml", overwrite=False, try_load=True):
    """move all the body and geom mesh to origin and offset+rotate joints"""
    obj_path = os.path.dirname(input_fname)
    offset_xml_fname = join(obj_path, output_fname)
    if os.path.exists(offset_xml_fname) and (not overwrite):
        return offset_xml_fname

    parsed_xml = etree.parse(input_fname)
    root = parsed_xml.getroot()
    asset_elems = root.findall("asset")
    for asset_elem in asset_elems:
        mesh_elems = asset_elem.findall("mesh")
        for mesh_elem in mesh_elems:
            mesh_elem.attrib["refquat"] = "0.5 -0.5 0.5 0.5"
    worldbody_elem = root.find("worldbody")
    obj_body = worldbody_elem.find("body")
    for geom_elem in obj_body.findall("geom"):
        # remove the quat attrib
        geom_elem.attrib.pop("quat", None)

    for part_body in obj_body.findall("body"):
        old_body_pos = part_body.attrib.get("pos", "0 0 0") 
        old_body_quat = part_body.attrib.get("quat", "1 0 0 0") 
        old_quat = Quaternion(
            np.array([float(x) for x in old_body_quat.split(" ")])
        )
        
        # move all the geoms to origin
        for geom_elem in part_body.findall("geom"):
            geom_elem.attrib.pop("pos", None)
            
        # offset all the joints 
        for joint_elem in part_body.findall("joint"):
            joint_pos = joint_elem.attrib.get("pos", "0 0 0")
            assert joint_pos == "0 0 0", f"joint pos: {joint_pos} not at origin!"
            joint_axis = joint_elem.attrib.get("axis", "1 0 0")
            joint_axis = np.array([float(x) for x in joint_axis.split(" ")])
            new_axis = old_quat.rotate(joint_axis)
            new_axis = np.array(new_axis, dtype=np.int8) # not uint8!! 
            # if all zero: try rounding
            if np.all(new_axis == 0):
                new_axis = np.round(joint_axis).astype(np.int8)
            idxs, = np.where(new_axis != 0) 
            assert len(idxs) == 1, f"joint axis: {new_axis} has more than one non-zero element!"

            
            # get the new joint pos
            joint_elem.attrib["pos"] = old_body_pos # TODO: handle ball joints?
            
            # for hinge joint, set the position along the non-zero axis to 0
            if joint_elem.attrib.get("type", "hinge") == "hinge":
                new_joint_pos = old_body_pos.split(" ")
                new_joint_pos[idxs[0]] = "0"
                joint_elem.attrib["pos"] = " ".join(new_joint_pos) 

            joint_elem.attrib["axis"] = " ".join([str(x) for x in new_axis])
            
            if joint_elem.attrib.get("type", "hinge") == "slide":
                joint_elem.attrib.pop("pos", None) # slide joint doesn't need pos
        part_body.attrib.pop("pos", None) 
        part_body.attrib.pop("quat", None)

    new_xml = etree.ElementTree(root)
    etree.indent(new_xml, space="  ")
    obj_path = os.path.dirname(input_fname)
    offset_xml_fname = join(obj_path, output_fname)
    new_xml.write(offset_xml_fname, pretty_print=True)
    
    if try_load:
        try:
            model = mujoco.MjModel.from_xml_path(offset_xml_fname)
            physics = dm_mujoco.Physics.from_xml_path(offset_xml_fname)
            # print(f"Successfully saved and loaded {offset_xml_fname} with dm_mujoco")
        except Exception as e:
            print(f"Failed to load {offset_xml_fname} with dm_mujoco, error:")
            print(e)  
    return offset_xml_fname
