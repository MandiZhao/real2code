"""
Hard-coded helper functions for adding joints based on OBB
"""
# use OBB edge as both joint axis and position
OBB_REL_HELPER=""" 
def add_body_and_joint(parent_body, mesh_id, obb, obb_axis_idx, obb_edge, obb_sign, joint_type):  
    center = np.array(obb['center'])
    obb_R = np.array(obb['R'])
    obb_half_lengths = np.array(obb['extent']) / 2
    obb_edge = np.insert(np.array(obb_edge), obb_axis_idx, 0)
    joint_pos = center + obb_R @ (obb_edge * obb_half_lengths)
    obb_axis = obb_R[:, obb_axis_idx]
    joint_axis = np.array(obb_axis) * obb_sign  
    
    link_body = parent_body.add('body', name=f'link_{mesh_id}')
    link_body.add('geom', type='mesh', name=f'link_{mesh_id}_geom', mesh=f'link_{mesh_id}')
    link_body.add('inertial', mass=1, pos=[0, 0, 0], diaginertia=[1, 1, 1])
    link_body.add('joint', name=f'joint_{mesh_id}', pos=joint_pos, axis=joint_axis, type=joint_type, range=[0, 1])
    return link_body
for joint in child_joints:
    box_id = joint['box']
    add_body_and_joint(body_root, box_id, bboxes[box_id], joint['idx'], joint['edge'], joint['sign'], joint['type'])
# add root body's geom:
body_root.add('geom', type='mesh', name='root_geom', mesh=f'link_{root_geom}')
"""

# use OBB rot as joint axis, position is relative to OBB center
OBB_ROT_HELPER=""" 
def add_body_and_joint(parent_body, mesh_id, obb, obb_axis_idx, joint_xy_pos, obb_sign, joint_type):
    center = np.array(obb['center'])
    obb_R = np.array(obb['R'])  
    obb_pos = np.insert(np.array(joint_xy_pos), obb_axis_idx, 0)
    joint_pos = center + obb_pos
    obb_axis = obb_R[:, obb_axis_idx]
    joint_axis = np.array(obb_axis) * obb_sign  
    
    link_body = parent_body.add('body', name=f'link_{mesh_id}')
    link_body.add('geom', type='mesh', name=f'link_{mesh_id}_geom', mesh=f'link_{mesh_id}')
    link_body.add('inertial', mass=1, pos=[0, 0, 0], diaginertia=[1, 1, 1])
    link_body.add('joint', name=f'joint_{mesh_id}', pos=joint_pos, axis=joint_axis, type=joint_type, range=[0, 1])
    return link_body
for joint in child_joints:
    box_id = joint['box']
    add_body_and_joint(body_root, box_id, bboxes[box_id], joint['idx'], joint['edge'], joint['sign'], joint['type'])
# add root body's geom:
body_root.add('geom', type='mesh', name='root_geom', mesh=f'link_{root_geom}')
"""

GLOBAL_JOINT_HELPER="""
def add_body_and_joint(parent_body, mesh_id, joint_pos, joint_axis, joint_type):  
    link_body = parent_body.add('body', name=f'link_{mesh_id}')
    link_body.add('geom', type='mesh', name=f'link_{mesh_id}_geom', mesh=f'link_{mesh_id}')
    link_body.add('inertial', mass=1, pos=[0, 0, 0], diaginertia=[1, 1, 1])
    link_body.add('joint', name=f'joint_{mesh_id}', pos=joint_pos, axis=joint_axis, type=joint_type, range=[0, 1])
    return link_body
for joint in child_joints:
    box_id = joint['box']
    add_body_and_joint(body_root, box_id, joint['pos'], joint['axis'], joint['type']) 
# add root body's geom:
body_root.add('geom', type='mesh', name='root_geom', mesh=f'link_{root_geom}')
"""

GET_HELPER_CODE={
    "obb_rel": OBB_REL_HELPER,
    "obb_rot": OBB_ROT_HELPER,
    "absolute": GLOBAL_JOINT_HELPER,
}