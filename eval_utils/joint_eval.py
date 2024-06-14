import os 
from os.path import join
import copy
from copy import deepcopy
import dm_control
import mujoco
from dm_control import mujoco as dm_mujoco
from dm_control import mjcf 
from collections import defaultdict
from glob import glob
import json 
import base64
import time 
import openai
import requests
import numpy as np
from natsort import natsorted
from itertools import permutations

def match_joints(joint1, joint2):
    # joints are matched if they have the same body and parent, bodys are considered same if the mesh names are same
    def match_bodies(body1, body2):
        if (body1 is None and body2 is not None) or (body1 is not None and body2 is None):
            return False
        if body1 is None and body2 is None:
            return True
        if body1['geomnum'] != body2['geomnum']:
            return False
        mesh1, mesh2 = body1['mesh_names'], body2['mesh_names']
        if len(mesh1) != len(mesh2):
            return False
        for m1 in mesh1:
            if m1 not in mesh2:
                return False
        for m2 in mesh2:
            if m2 not in mesh1:
                return False
        return True

    body1 = joint1['body']
    body2 = joint2['body']
    if match_bodies(body1, body2) is False:
        return False
    parent1, parent2 = joint1['parent'], joint2['parent']
    if match_bodies(parent1, parent2) is False:
        return False
    return True


def compute_joint_error(pred_joint, gt_joint):
    def norm(a, axis=-1):
        return np.sqrt(np.sum(a ** 2, axis=axis))
    def get_angle(axis1, axis2):
        return np.arccos(np.dot(axis1, axis2) / (norm(axis1) * norm(axis2))) * 180 / np.pi
    # return gt type, type acc, rot axis error,  pos. error (if both are hinge joints)
    type_acc = 1 
    gt_type = gt_joint['type']  
    if pred_joint['type'] != gt_type:
        return gt_type, dict(type=0, rot=None, pos=None)
     
    pred_axis = np.array(pred_joint['axis'])
    pred_pos = np.array(pred_joint['pos'])
    gt_axis = np.array(gt_joint['axis'])
    gt_pos = np.array(gt_joint['pos'])

    # type 2 is slide, 3 is hinge joint
    pos_err = None
    if gt_type == 3:   
        # use np.cross to compute the distance between two lines
        # Compute the cross product of V1 and V2
        W = np.cross(pred_axis, gt_axis)
        if np.allclose(W, 0):
            # The lines are parallel or collinear.
            # Create a vector perpendicular to V1 (and V2 since they are parallel)
            # You can choose any vector that is not a scalar multiple of V1
            W = np.cross(pred_axis, [1, 0, 0]) if not np.allclose(pred_axis, [1, 0, 0]) else np.cross(gt_axis, [0, 1, 0])
        P = gt_pos - pred_pos
        # Compute the shortest distance
        pos_err = np.abs(np.dot(P, W)) / np.linalg.norm(W) 
        # pos_err = np.linalg.norm(pred_pos - gt_pos)   
        # scale = np.sqrt(12) # don't do this!
        pos_err = pos_err  

    # get the rotation error measured in degree
    rot_err = get_angle(pred_axis, gt_axis)
    # rotate the pred axis and compute again
    err_2 = get_angle(-pred_axis, gt_axis)
    rot_err = min(rot_err, err_2)

    return gt_type, dict(type=type_acc, rot=rot_err, pos=pos_err)

def permute_joint_error(pred_joints, gt_joints):
    """ Use for full eval pipeline, compute error between all permutations and take the lowest"""
    if len(pred_joints) == 0 and len(gt_joints) > 0:
        # consider all type failures 
        min_errs = [dict(type=0, rot=None, pos=None, gt_type=gt_joint['type']) for gt_joint in gt_joints]
        return min_errs, []
    all_perm_errs = []
    all_perms = []
    avg_errs = []
    for perm in permutations(pred_joints):
        avg_perm_errs = defaultdict(list)
        per_joint_errs = []
        for perm_joint, gt_joint in zip(perm, gt_joints):
            gt_type, errors = compute_joint_error(perm_joint, gt_joint)
            for k, v in errors.items():
                avg_perm_errs[k].append(v)
            avg_perm_errs['gt_type'].append(gt_type)  
            errors["gt_type"] = gt_type
            per_joint_errs.append(errors)
        all_perms.append(perm)
        min_err = []
        for k, v in avg_perm_errs.items():
            if k == "gt_type":
                continue
            if k == "type":
                # flip the type error since higher is better
                v = [1 - vv for vv in v]
            v = [vv for vv in v if vv is not None]
            min_err.append(np.mean(v) if len(v) > 0 else 0)
        avg_errs.append(np.mean(min_err))
        all_perm_errs.append(per_joint_errs)
    # find the permutation with the lowest error 
    min_err_idx = np.argmin(avg_errs)
    min_err_perm = all_perms[min_err_idx]
    min_errs = all_perm_errs[min_err_idx] 
    return min_errs, min_err_perm

def adjust_prediction(prediction: str):
    """ Parse and clean the LLM prediction string """
    preds = prediction.split("\n")
    adjusted_preds = []
    if "child_joints = " not in prediction:
        return adjusted_preds
    for line in preds:
        # get int value
        if len(line.strip()) <= 2 and line.strip().isdigit(): # handles single or double digits
            adjusted_preds.append(line) 
        if line.startswith("child_joints = ") or line.startswith("dict(box="):
            adjusted_preds.append(line)
        if line.startswith("]"):
            adjusted_preds.append(line.split("]")[0] + "]") 
        if 'child_joints = [' in adjusted_preds and "]" in adjusted_preds:
            break # early stop
    return "\n".join(adjusted_preds)

def get_joint_code(save_fname="'joints.json'"):
    if '.json' not in save_fname:
        save_fname = save_fname + ".json"
    return f"""
import json
import numpy as np
def get_body_info(mjmodel, bodyid): 
    if isinstance(bodyid, np.ndarray):
        bodyid = bodyid[0]
    jnt_body = mjmodel.body(bodyid)
    geomadr = jnt_body.geomadr
    mesh_names = []
    for adr in geomadr:
        geom = mjmodel.geom(adr)
        if int(geom.type[0]) == 7: # means mesh
            try:
                dataid = int(geom.dataid[0])
                mesh_names.append(mjmodel.mesh(dataid).name)
            except:
                pass
    info = dict(name=jnt_body.name, id=jnt_body.id, rootid=int(jnt_body.rootid[0]), geomnum=int(jnt_body.geomnum[0]), mesh_names=mesh_names)
    return info
joints = []
mjmodel = physics.model
for i in range(mjmodel.njnt): 
    joint = mjmodel.jnt(i)
    jnt_info = dict(name=joint.name, axis=joint.axis.tolist(), pos=joint.pos.tolist(), range=joint.range.tolist(), type=int(joint.type[0]))
    jnt_info['body'] = get_body_info(mjmodel, joint.bodyid)
    jnt_parent = joint.parentid
    if jnt_parent != -1:
        jnt_info['parent'] = get_body_info(mjmodel, jnt_parent)
    else:
        jnt_info['parent'] = None
    joints.append(jnt_info)
# save to json file 
with open('{save_fname}', "w") as f:
    json.dump(joints, f, indent=4)
"""


