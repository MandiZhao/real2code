import os
import numpy as np
from glob import glob
from os.path import join 
from natsort import natsorted
import argparse
import shutil
import json
import trimesh 
import open3d as o3d
import kaolin as kal 
from kaolin.ops.mesh import check_sign, index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance, uniform_laplacian_smoothing
from kaolin.io.obj import import_mesh
from kaolin.ops.conversions import trianglemeshes_to_voxelgrids, sdf_to_voxelgrids, voxelgrids_to_trianglemeshes
from matplotlib import pyplot as plt
import pyquaternion
import h5py
import torch
from copy import deepcopy

from data_utils import get_tight_obb as get_handcraft_obb

"""
Generate input and target complete pcds 
1. Run Manifold on merged meshes
2. rotate the mesh according to loop
3. Get GT obb on each mesh and save to json
4. get pointclouds from RGBDs

conda activate mesh2sdf
python datagen.py --split test
"""
QUAT_ARR = np.array([0.5, -0.5, 0.5, 0.5])
WIDTH, HEIGHT = 512, 512
RAW_MESH_ORIGIN = torch.tensor([-1,-1,-1])[None]
RAW_MESH_SCALE = torch.tensor([2,2,2])[None]

SPECIAL_IDS=[46859,46874,]
   
def get_pcds(loop_dir, np_random, num_use_cameras=8, pcd_size=8000, remove_outlier=True, depth_trunc=10, vlength=0.01):
    """ return all the part-level PCDs """
    h5s = natsorted(glob(join(loop_dir, "*hdf5"))) 
    # NOTE: some masks don't have all the parts due to partial occlusion!!
    num_masks = []
    for h5_file in h5s:
        # print(f"Processing {h5_file}")
        with h5py.File(h5_file, "r") as f:  
            binary_masks = f["binary_masks"][:]
        num_masks.append(len(binary_masks))
    max_num_masks = max(num_masks)  

    # randomly shuffle the h5s
    volumes = [
            o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=vlength, 
                sdf_trunc=(2 * vlength),
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                )
            for _ in range(max_num_masks - 1) # first mask is the background
        ]
    np_random.shuffle(h5s)
    used_cameras = 0
    for h5_file in h5s:
        # print(f"Processing {h5_file}")
        with h5py.File(h5_file, "r") as f: 
            rgb = f["colors"][:]
            depth = f["depth"][:]
            binary_masks = f["binary_masks"][:]
            camera_intrinsics = np.array(f['cam_intrinsics'])
            camera_pose = np.array(f['cam_pose'])
        if len(binary_masks) < max_num_masks:
            # need to skip because mask order is messed up 
            continue
          
        intrinsics = o3d.camera.PinholeCameraIntrinsic() 
        intrinsics.set_intrinsics(
            width=WIDTH, height=HEIGHT, fx=camera_intrinsics[0, 0], fy=camera_intrinsics[1, 1], cx=(WIDTH / 2), cy=(HEIGHT / 2))
        extrinsics = np.linalg.inv(camera_pose)
        extrinsics[2,:] *= -1
        extrinsics[1,:] *= -1 
        depth[depth>depth_trunc] = depth_trunc 
        
        for idx, mask in enumerate(binary_masks[1:]): 
            rgb_copy = rgb.copy()
            depth_copy = depth.copy().astype(np.float32)
            rgb_copy[mask == 0] = 0
            depth_copy[mask == 0] = 0  
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb_copy), 
                    o3d.geometry.Image(depth_copy),
                    depth_trunc=depth_trunc, 
                    depth_scale=1,
                    convert_rgb_to_intensity=False,
                    )
            # print(f"Integrating {h5_file} to volume {idx}")
            volumes[idx].integrate(
                    rgbd,
                    intrinsics,
                    extrinsics,
                    )
        used_cameras += 1
        if used_cameras == num_use_cameras:
            break
    pcds = []
    for volume in volumes:
        pcd = volume.extract_point_cloud()
        if remove_outlier:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
        if pcd_size > -1 and len(pcd.points) > pcd_size:
            # print(f"Subsampling {len(pcd.points)} to {pcd_size}")
            # subsampled_idxs = np_random.choice(len(pcd.points), pcd_size, replace=(len(pcd.points) < pcd_size))
            # new_points = np.array(pcd.points)[subsampled_idxs]
            # new_colors = np.array(pcd.colors)[subsampled_idxs]
            # pcd.points = o3d.utility.Vector3dVector(new_points)
            # pcd.colors = o3d.utility.Vector3dVector(new_colors)
            # sample farther points
            pcd = pcd.farthest_point_down_sample(pcd_size) 

        pcds.append(pcd)
    
    return pcds

def show_obb_pyplot(
    obb_dicts,   
    save_img_fname='test.png',
    view_angles=dict(azim=150, elev=5), 
):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(680*px, 680*px))
    ax = fig.add_subplot(111, projection='3d') 
    ax.view_init(**view_angles)
    colors = 'rgb'
    origin = np.array([0, 0, 0])
    # plot unit axis 
    ax.plot([origin[0], 1], [origin[1], 0], [origin[2], 0], color=colors[0], linewidth=1)
    ax.plot([origin[0], 0], [origin[1], 1], [origin[2], 0], color=colors[1], linewidth=1)
    ax.plot([origin[0], 0], [origin[1], 0], [origin[2], 1], color=colors[2], linewidth=1)

    for i, obb_dict in enumerate(obb_dicts):
        center = obb_dict['center']
        extents = obb_dict['extent']
        transform = obb_dict['R']
        o3d_obb = o3d.geometry.OrientedBoundingBox(
            center=center, R=np.array(transform)[:3, :3], extent=extents)
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_obb)
        lines = np.array(lineset.lines)
        box_points = np.array(o3d_obb.get_box_points())
        cidx = i % len(colors)
        for line in lines:
            x0, y0, z0 = box_points[line[0]]
            x1, y1, z1 = box_points[line[1]]
        
            ax.plot([x0, x1], [y0, y1], [z0, z1], color=colors[cidx], linewidth=1)
    if save_img_fname is not None:
        plt.savefig(save_img_fname) # dpi=300)
    # close figure 
    plt.close(fig)
    return 

def save_voxel_grid(args, voxel_size=96):
    """ Load all the rotated mesh from shape dataset and save the voxel grid"""
    if args.skip_extent:
        print(f"Skipping extent normalization when saving voxel grid!")
    lookup_path = join(args.data_dir, args.split, args.obj_type, args.obj_folder)
    obj_folders = natsorted(glob(lookup_path))
    print(f"Found {len(obj_folders)} objects for {args.obj_type} in {args.split} split.")
    np_random = np.random.RandomState(args.np_seed)
    for obj_folder in obj_folders:
        print(f"Processing {obj_folder}")
        obj_type = obj_folder.split("/")[-2]
        obj_id = obj_folder.split("/")[-1]
        out_path = join(args.out_dir, args.split, obj_type, obj_id)
        loop_dirs = natsorted(glob(join(obj_folder, f"loop_{args.loop_id}")))
        if len(loop_dirs) == 0:
            print(f"No loop found for {obj_folder}")
            continue

        for loop_dir in loop_dirs:
            loop_id = loop_dir.split("/")[-1]
            out_dir = join(args.out_dir, args.split, obj_type, obj_id, loop_id)
            
            # rotate all the meshes
            transforms_fname = join(loop_dir, "mesh_transforms.json")
            assert os.path.exists(transforms_fname), f"{transforms_fname} does not exist"
            mesh_transforms = json.load(open(transforms_fname, "r"))
            
            rotated_mesh_fnames = natsorted(glob(join(out_dir, "*_rot.obj")))
            obb_fnames = natsorted(glob(join(out_dir, "obb_*.json"))) 
            
            for obj_fname in rotated_mesh_fnames:
                link_name = obj_fname.split("/")[-1].split("_rot.obj")[0]
                obb_fname = join(out_dir, f"obb_{link_name}.json")
                obb_dict = json.load(open(obb_fname, "r"))
                center = obb_dict['center']
                extent = obb_dict['extent']
                R = obb_dict['R']

                center = torch.tensor(center)
                extent = torch.tensor(extent)
                R = torch.tensor(R)

                mesh = import_mesh(obj_fname)
                vertices, faces = mesh.vertices, mesh.faces
                vertices = (vertices - center) @ R 
                if not args.skip_extent:
                    vertices = vertices / extent
                    mesh_origin = torch.tensor([-0.5,-0.5,-0.5])[None].cuda()
                    mesh_scale = torch.tensor([1,1,1])[None].cuda()
                else:
                    mesh_origin = RAW_MESH_ORIGIN.cuda()
                    mesh_scale = RAW_MESH_SCALE.cuda()
                voxelgrid = trianglemeshes_to_voxelgrids(
                    vertices[None].cuda(), 
                    faces.cuda(), voxel_size, origin=mesh_origin, scale=mesh_scale)
                grid_fname = join(out_dir, f"{link_name}_grid_norm_{voxel_size}.pt")
                if args.skip_extent:
                    grid_fname = join(out_dir, f"{link_name}_grid_norm_{voxel_size}_noextent.pt")
                torch.save(voxelgrid, grid_fname)
    return

def save_occupancy(args, voxel_size=96, use_aabb=False):
    """ Load all the rotated mesh from shape dataset and get occupancy via sdf"""
    lookup_path = join(args.data_dir, args.split, args.obj_type, args.obj_folder)
    obj_folders = natsorted(glob(lookup_path))
    print(f"Found {len(obj_folders)} objects for {args.obj_type} in {args.split} split.")
    np_random = np.random.RandomState(args.np_seed)
    if args.skip_extent:
        print(f"Skipping extent normalization when saving occupancy grid!")
    for obj_folder in obj_folders:
        print(f"Processing {obj_folder}")
        obj_type = obj_folder.split("/")[-2]
        obj_id = obj_folder.split("/")[-1]
        out_path = join(args.out_dir, args.split, obj_type, obj_id)
        loop_dirs = natsorted(glob(join(obj_folder, f"loop_{args.loop_id}")))
        if len(loop_dirs) == 0:
            print(f"No loop found for {obj_folder}")
            continue 
        for loop_dir in loop_dirs:
            loop_id = loop_dir.split("/")[-1]
            out_dir = join(args.out_dir, args.split, obj_type, obj_id, loop_id)
            vis_out_dir = join(args.vis_dir, args.split, obj_type, obj_id, loop_id)
            os.makedirs(vis_out_dir, exist_ok=True)
            # rotate all the meshes
            transforms_fname = join(loop_dir, "mesh_transforms.json")
            assert os.path.exists(transforms_fname), f"{transforms_fname} does not exist"
            mesh_transforms = json.load(open(transforms_fname, "r"))
            
            rotated_mesh_fnames = natsorted(glob(join(out_dir, "*_rot.obj")))
            obb_fnames = natsorted(glob(join(out_dir, "obb_*.json"))) 
            
            for obj_fname in rotated_mesh_fnames:
                link_name = obj_fname.split("/")[-1].split("_rot.obj")[0] 
                grid_fname = join(out_dir, f"{link_name}_occ_norm_{voxel_size}.pt")
                if args.skip_extent:
                    grid_fname = join(out_dir, f"{link_name}_occ_norm_{voxel_size}_noextent.pt")
                if use_aabb:
                    grid_fname = join(out_dir, f"{link_name}_occ_norm_{voxel_size}_aabb.pt")
                if os.path.exists(grid_fname):
                    print(f"Skipping {grid_fname}")
                    continue
                
                mesh = import_mesh(obj_fname) # Table 26652 is bug
                # mesh = trimesh.load(obj_fname, process=False)
                vertices, faces = mesh.vertices.cuda(), mesh.faces.cuda()
                if use_aabb:
                    # use aabb for mesh and save aabb json 
                    center = torch.mean(vertices, dim=0)
                    extent = torch.max(vertices, dim=0).values - torch.min(vertices, dim=0).values
                    R = torch.eye(3)
                    aabb_fname = join(out_dir, f"aabb_{link_name}.json")
                    with open(aabb_fname, "w") as f:
                        json.dump(dict(center=center.cpu().numpy().tolist(), extent=extent.cpu().numpy().tolist()), f, indent=4)
                else:
                    obb_fname = join(out_dir, f"obb_{link_name}.json")
                    obb_dict = json.load(open(obb_fname, "r"))
                    center = obb_dict['center']
                    extent = obb_dict['extent'] 
                    R = obb_dict['R']
                center = torch.tensor(center).cuda()
                extent = torch.tensor(extent).cuda() 
                R = torch.tensor(R).cuda()
                vertices = (vertices - center) @ R 
                if not args.skip_extent:
                    vertices = vertices / extent
                    mesh_origin = torch.tensor([-0.5,-0.5,-0.5])[None].cuda()
                    mesh_scale = torch.tensor([1,1,1])[None].cuda()
                else:
                    mesh_origin = RAW_MESH_ORIGIN.cuda()
                    mesh_scale = RAW_MESH_SCALE.cuda()

                voxelgrid = trianglemeshes_to_voxelgrids(
                    vertices[None].cuda(), faces.cuda(), voxel_size, origin=mesh_origin, scale=mesh_scale)
                query_resolution = voxel_size
                query_locations = torch.stack(torch.meshgrid(
                    torch.linspace(0, 1, query_resolution),
                    torch.linspace(0, 1, query_resolution),
                    torch.linspace(0, 1, query_resolution),
                ), dim=-1).view(-1, 3).cuda()
                query_points = (query_locations * mesh_scale + mesh_origin).cuda()
                is_inside = check_sign(vertices[None], faces.long(), query_points[None])
                occupancy_grid = voxelgrid 
                # print(f'num points on mesh: {torch.sum(voxelgrid)}')
                is_inside = is_inside.view(1, query_resolution, query_resolution, query_resolution)
                occupancy_grid[is_inside] = 1
                # print(f'num points occupying mesh: {torch.sum(occupancy_grid)}')
                # face_verts = index_vertices_by_faces(vertices[None], faces.long()).cuda()
                
                # query_resolution = 96
                # query_locations = torch.stack(torch.meshgrid(
                #     torch.linspace(0, 1, query_resolution),
                #     torch.linspace(0, 1, query_resolution),
                #     torch.linspace(0, 1, query_resolution),
                # ), dim=-1).view(-1, 3).cuda()
                # query_points = (query_locations * mesh_scale + mesh_origin).cuda()
                # def sdf(point):
                #     # shape should be shape (N, 3)
                #     pd, face_indices, distance_type = point_to_mesh_distance(point[None], face_verts)
                #     is_inside = check_sign(vertices[None], faces.long(), point[None])
                #     sdf = torch.where(is_inside, -pd, pd) # shape torch.Size([1, N])
                #     return sdf[0], is_inside[0], pd[0]
                # sdf_grid, is_inside, pd = sdf(query_points) 
                # breakpoint()
                # sdf_grid = sdf_grid.view(query_resolution, query_resolution, query_resolution) 
                # occupancy_grid = is_inside.view(query_resolution, query_resolution, query_resolution).float()

                occ_verts, occ_faces = voxelgrids_to_trianglemeshes(occupancy_grid.cuda())
                occ_mesh = trimesh.Trimesh(vertices=occ_verts[0].cpu().numpy(), faces=occ_faces[0].cpu().numpy())
                occ_mesh.export(join(vis_out_dir, f"{link_name}_occ_{voxel_size}.obj")) 
                torch.save(occupancy_grid, grid_fname)
    return

def save_pcds(args):
    lookup_path = join(args.data_dir, args.split, args.obj_type, args.obj_folder)
    obj_folders = natsorted(glob(lookup_path))
    print(f"Found {len(obj_folders)} objects for {args.obj_type} in {args.split} split.")
    np_random = np.random.RandomState(args.np_seed)
    for obj_folder in obj_folders:
        obj_type = obj_folder.split("/")[-2]
        obj_id = obj_folder.split("/")[-1]
        out_path = join(args.out_dir, args.split, obj_type, obj_id)
        os.makedirs(out_path, exist_ok=True) 

        loop_dirs = natsorted(glob(join(obj_folder, f"loop_{args.loop_id}")))
        if len(loop_dirs) == 0:
            print(f"No loop found for {obj_folder}")
            continue
        print(f"Processing {obj_folder}")
        for loop_dir in loop_dirs:
            loop_id = loop_dir.split("/")[-1]
            out_dir = join(args.out_dir, args.split, obj_type, obj_id, loop_id)
            os.makedirs(out_dir, exist_ok=True)
            pcds = get_pcds(loop_dir, np_random, num_use_cameras=args.num_use_cameras, pcd_size=args.pcd_size)
            for idx, pcd in enumerate(pcds):
                link_name = f"link_{idx}"
                pcd_fname = join(out_dir, f"{link_name}_{args.pcd_size}.npz")
                np.savez(pcd_fname, points=np.array(pcd.points), colors=np.array(pcd.colors))
                
        

def main(args):
    lookup_path = join(args.data_dir, args.split, args.obj_type, args.obj_folder)
    obj_folders = natsorted(glob(lookup_path))
    print(f"Found {len(obj_folders)} objects for {args.obj_type} in {args.split} split.")
    np_random = np.random.RandomState(args.np_seed)
    for obj_folder in obj_folders:
        print(f"Processing {obj_folder}")
        obj_type = obj_folder.split("/")[-2]
        obj_id = obj_folder.split("/")[-1]
        out_path = join(args.out_dir, args.split, obj_type, obj_id)
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(join(out_path, "blender_meshes"), exist_ok=True)

        loop_dirs = natsorted(glob(join(obj_folder, f"loop_{args.loop_id}")))
        if len(loop_dirs) == 0:
            print(f"No loop found for {obj_folder}")
            continue
        mesh_fnames = natsorted(glob(join(obj_folder, "blender_meshes", "*.obj")))
        new_meshes = dict()
        for obj_fname in mesh_fnames:
            _name = obj_fname.split("/")[-1]
            out_fname = join(out_path, "blender_meshes", _name)
            if not os.path.exists(out_fname): #or args.overwrite:
                command = f"/home/mandi/ManifoldPlus/build/manifold --input {obj_fname} --output {out_fname} --depth 8\n"
                os.system(command)
            link_name = _name.split(".")[0]
            new_meshes[link_name] = trimesh.load(out_fname, process=True)

        for loop_dir in loop_dirs:
            loop_id = loop_dir.split("/")[-1]
            out_dir = join(args.out_dir, args.split, obj_type, obj_id, loop_id)
            os.makedirs(out_dir, exist_ok=True)
            # rotate all the meshes
            transforms_fname = join(loop_dir, "mesh_transforms.json")
            if not os.path.exists(transforms_fname):
                print(f"WARNING: {transforms_fname} does not exist")
                continue
            # if all([os.path.exists(join(out_dir, f"{link_name}_rot.obj")) for link_name in new_meshes]) \
            #     and all([os.path.exists(join(out_dir, f"obb_{link_name}.json")) for link_name in new_meshes]) and not args.overwrite: 
            #     continue
            mesh_transforms = json.load(open(transforms_fname, "r"))
            
            rotated_obbs = dict() 
            rotated_meshes = dict()
        
            for link_name, mesh in new_meshes.items(): 
                new_fname = join(out_dir, link_name + "_rot.obj")
                obb_fname = join(out_dir, f"obb_{link_name}.json")
                if os.path.exists(new_fname) and os.path.exists(obb_fname) and not args.overwrite:
                    rotated_obbs[link_name] = json.load(open(obb_fname, "r"))
                    rotated_meshes[link_name] = new_fname
                else:
                    mesh = deepcopy(mesh)
                    if link_name in mesh_transforms:
                        mesh.apply_transform(np.array(mesh_transforms[link_name])[0])
                    mesh.export(new_fname) 
                    rotated_meshes[link_name] = new_fname
                    obb_dict = get_tight_obb(mesh) 
                    obb_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obb_dict.items()}
                    rotated_obbs[link_name] = obb_dict 
                    
                    with open(obb_fname, "w") as f:
                        json.dump(obb_dict, f, indent=4)
            voxel_size = 96
            grid_fnames = [join(out_dir, f"{link_name}_occ_partial_{voxel_size}.pt") for link_name in rotated_meshes]
            if all([os.path.exists(x) for x in grid_fnames]) and not args.overwrite:
                continue

            img_fname = join(out_dir, "gt_boxes.jpg")
            show_obb_pyplot(list(rotated_obbs.values()), save_img_fname=img_fname)
            
            pcd_fnames = natsorted(glob(join(out_dir, "link*.npz"))) 
            if len(pcd_fnames) == len(new_meshes) and not args.overwrite:
                # print('Loading existing PCDs')
                pcd_points = []
                for pcd_fname in pcd_fnames:
                    pcd = np.load(pcd_fname)
                    pcd_points.append(pcd['points'])
            else:
                pcds = get_pcds(loop_dir, np_random, num_use_cameras=args.num_use_cameras, pcd_size=args.pcd_size)
                assert len(pcds) == len(rotated_obbs), f"len(pcds) {len(pcds)} != len(rotated_obbs) {len(rotated_obbs)}"
                pcd_points = []
                for idx, pcd in enumerate(pcds):
                    link_name = f"link_{idx}"
                    pcd_fname = join(out_dir, f"{link_name}.npz")
                    np.savez(pcd_fname, points=np.array(pcd.points), colors=np.array(pcd.colors))
                    pcd_points.append(np.array(pcd.points))

            partial_obbs = dict()  
            for idx, points in enumerate(pcd_points):
                link_name = f"link_{idx}"
                partial_obb = get_handcraft_obb(points)
                partial_obb['extent'] *= np.random.uniform(1.1, 1.5, 3)
                partial_fname = join(out_dir, f"partial_{link_name}.json")
                with open(partial_fname, "w") as f:
                    json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in partial_obb.items()}, f, indent=4)
                partial_obbs[link_name] = partial_obb
            img_fname = join(out_dir, "partial_boxes.jpg")
            show_obb_pyplot(list(partial_obbs.values()), save_img_fname=img_fname)
            
            # save occupancy grid:
            for link_name, mesh_fname in rotated_meshes.items():
                # obb_dict = rotated_obbs[link_name]
                obb_dict = partial_obbs[link_name] 
                # print(f"Normalizing with partial obb!")
                grid_fname = join(out_dir, f"{link_name}_occ_partial_{voxel_size}.pt")
                if args.skip_extent:
                    grid_fname = join(out_dir, f"{link_name}_occ_norm_{voxel_size}_noextent.pt")
                if os.path.exists(grid_fname) and not args.overwrite:
                    print(f"Skipping {grid_fname}")
                    continue
                mesh = import_mesh(mesh_fname)
                vertices, faces = mesh.vertices.cuda(), mesh.faces.cuda()
                # subsample mesh
                # if len(vertices) > 250000:
                #     print(f"Subsampling {len(vertices)} to 200,000")
                #     sub_mesh = trimesh.load(mesh_fname, process=False)
                #     sub_mesh = sub_mesh.simplify_quadric_decimation(200000)
                #     vertices, faces = sub_mesh.vertices, sub_mesh.faces
                #     vertices = torch.tensor(vertices).cuda()
                #     faces = torch.tensor(faces).cuda()
                center = torch.tensor(obb_dict['center']).cuda()
                extent = torch.tensor(obb_dict['extent']).cuda()
                R = torch.tensor(obb_dict['R']).cuda()
                vertices = (vertices - center) @ R
                if not args.skip_extent:
                    vertices = vertices / extent
                    mesh_origin = torch.tensor([-0.5,-0.5,-0.5])[None].cuda()
                    mesh_scale = torch.tensor([1,1,1])[None].cuda()
                else:
                    mesh_origin = RAW_MESH_ORIGIN.cuda()
                    mesh_scale = RAW_MESH_SCALE.cuda()
                voxelgrid = trianglemeshes_to_voxelgrids(
                    vertices[None].cuda(), faces.cuda(), voxel_size, origin=mesh_origin, scale=mesh_scale)
                query_resolution = voxel_size
                query_locations = torch.stack(torch.meshgrid(
                    torch.linspace(0, 1, query_resolution),
                    torch.linspace(0, 1, query_resolution),
                    torch.linspace(0, 1, query_resolution),
                ), dim=-1).view(-1, 3).cuda()
                query_points = (query_locations * mesh_scale + mesh_origin).cuda()
                is_inside = check_sign(vertices[None], faces.long(), query_points[None])
                occupancy_grid = voxelgrid
                is_inside = is_inside.view(1, query_resolution, query_resolution, query_resolution)
                occupancy_grid[is_inside] = 1
                torch.save(occupancy_grid, grid_fname)
                # skip saving GT label converted to mesh
                occ_verts, occ_faces = voxelgrids_to_trianglemeshes(occupancy_grid.cuda())
                occ_mesh = trimesh.Trimesh(vertices=occ_verts[0].cpu().numpy(), faces=occ_faces[0].cpu().numpy())
                occ_mesh.export(join(out_dir, f"{link_name}_occ_partial_{voxel_size}.obj"))

 
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/local/real/mandi/blender_dataset_v4/")
    parser.add_argument("--obj_type", type=str, default="*")
    parser.add_argument("--obj_folder", type=str, default="*")
    parser.add_argument("--loop_id", type=str, default="*")
    parser.add_argument("--np_seed", type=int, default=0)
    parser.add_argument("--pcd_size", type=int, default=2048)
    parser.add_argument("--num_use_cameras", type=int, default=24)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out_dir", type=str, default="/local/real/mandi/shape_dataset_v4/")
    parser.add_argument("--vis_dir", type=str, default="/local/real/mandi/shape_dataset_v4/vis")
    parser.add_argument("--overwrite", "-o", action="store_true")
    parser.add_argument("--pcd_only", action="store_true")
    parser.add_argument("--voxel_grid", action="store_true")
    parser.add_argument("--skip_extent", action="store_true")
    parser.add_argument("--occupancy", "-occ", action="store_true")
    parser.add_argument("--use_aabb", action="store_true")
    args = parser.parse_args()
    if args.voxel_grid:
        save_voxel_grid(args, 96)
        exit()
    if args.occupancy:
        save_occupancy(args, 96, use_aabb=args.use_aabb)
        exit()
    if args.pcd_only:
        save_pcds(args)
        exit()

    main(args)