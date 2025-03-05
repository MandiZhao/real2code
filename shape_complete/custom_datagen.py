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
from PIL import Image   
from shape_complete.datagen import *

"""
also saves masked RGB images using each mask
for 1 object with N parts x K camera views, saves NxK RGB images, N raw pcds (grouped across cameras), N voxel grids (normalized to 1x1x1)

OBJ=StorageFurniture 
FOLDER=44781
SPLIT=test
python shape_complete/custom_datagen.py --data_dir /store/real/mandi/real2code_dataset_v0/ \
     --obj_type $OBJ --obj_folder $FOLDER --split $SPLIT --out_dir /store/real/mandi/r2c2r_data/
"""

WIDTH, HEIGHT = 512, 512
RAW_MESH_ORIGIN = torch.tensor([-1,-1,-1])[None]
RAW_MESH_SCALE = torch.tensor([2,2,2])[None]


def main(args):
    lookup_path = join(args.data_dir, args.split, args.obj_type, args.obj_folder)
    obj_folders = natsorted(glob(lookup_path))
    print(f"Found {len(obj_folders)} objects for {args.obj_type} in {args.split} split.")
    np_random = np.random.RandomState(args.np_seed)
    voxel_size = int(args.voxel_grid_size)
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
                    obb_dict = get_handcraft_obb(mesh) 
                    obb_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obb_dict.items()}
                    rotated_obbs[link_name] = obb_dict 
                    
                    with open(obb_fname, "w") as f:
                        json.dump(obb_dict, f, indent=4)

            grid_fnames = [join(out_dir, f"{link_name}_occ_partial_{voxel_size}.pt") for link_name in rotated_meshes]
            if all([os.path.exists(x) for x in grid_fnames]) and not args.overwrite:
                continue
            
            pcd_fnames = natsorted(glob(join(out_dir, "link*.ply"))) 
            if len(pcd_fnames) == len(new_meshes) and not args.overwrite:
                # print('Loading existing PCDs')
                pcd_points = []
                for pcd_fname in pcd_fnames:
                    pcd = np.load(pcd_fname)
                    pcd_points.append(pcd['points'])
            else:
                pcds, masked_rgbs, masked_depths = get_pcds(
                    loop_dir, np_random, num_use_cameras=args.num_use_cameras, 
                    pcd_size=args.pcd_size,
                    return_masked_rgb=True,
                    )
                assert len(pcds) == len(rotated_obbs), f"len(pcds) {len(pcds)} != len(rotated_obbs) {len(rotated_obbs)}"
                pcd_points = []
                for idx, pcd in enumerate(pcds):
                    link_name = f"link_{idx}"
                    pcd_fname = join(out_dir, f"{link_name}.ply")
                    # save ply file!! each pcd is already an open3d point cloud
                    o3d.io.write_point_cloud(pcd_fname, pcd)
                    pcd_points.append(np.array(pcd.points))

                    # also save the masked rgbds
                    img_out_dir = join(out_dir, f"{link_name}_rgb")
                    os.makedirs(img_out_dir, exist_ok=True)
                    for cam_i, rgb_i in enumerate(masked_rgbs[idx]):
                        rgb_fname = join(img_out_dir, f"rgb_{cam_i}.png")
                        Image.fromarray(rgb_i).save(rgb_fname) 
                        # depth_fname = join(img_out_dir, f"depth_{cam_i}.png")
                        # depth_i = masked_depths[idx][cam_i]
                        # Image.fromarray(depth_i).save(depth_fname)

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

            # save occupancy grid:
            for link_name, mesh_fname in rotated_meshes.items():
                # obb_dict = rotated_obbs[link_name]
                obb_dict = partial_obbs[link_name] 
                # print(f"Normalizing with partial obb!")
                grid_fname = join(out_dir, f"{link_name}_occ_partial_{voxel_size}.pt") 
                if os.path.exists(grid_fname) and not args.overwrite:
                    print(f"Skipping {grid_fname}")
                    continue
                mesh = import_mesh(mesh_fname)
                vertices, faces = mesh.vertices.cuda(), mesh.faces.cuda()

                center = torch.tensor(obb_dict['center']).cuda()
                extent = torch.tensor(obb_dict['extent']).cuda()
                R = torch.tensor(obb_dict['R']).cuda()
                vertices = (vertices - center) @ R

                mesh_origin = RAW_MESH_ORIGIN.cuda()
                mesh_scale = RAW_MESH_SCALE.cuda()
                voxelgrid = trianglemeshes_to_voxelgrids(
                    vertices[None].cuda(), faces.cuda(), voxel_size, origin=mesh_origin, scale=mesh_scale
                    )
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

                # try loading it
                # fname = "/store/real/mandi/r2c2r_data/test/StorageFurniture/45693/loop_0/link_0_occ_partial_64.pt"
                # occupancy_grid = torch.load(fname)

                # SKIP: save mesh
                # occ_verts, occ_faces = voxelgrids_to_trianglemeshes(occupancy_grid.cuda())
                # occ_mesh = trimesh.Trimesh(vertices=occ_verts[0].cpu().numpy(), faces=occ_faces[0].cpu().numpy())
                # occ_mesh.export(join(out_dir, f"{link_name}_occ_partial_{voxel_size}.obj"))

    return 

if __name__ == "__main__":

    # fname = "/store/real/mandi/r2c2r_data/test/StorageFurniture/45693/loop_0/link_0_occ_partial_64.pt"
    fname = "/store/real/mandi/r2c2r_data/test/StorageFurniture/44781/loop_0/link_3.ply"
    pcd = o3d.io.read_point_cloud(fname)
    # print size of pcdf 
    print(len(pcd.points))
    breakpoint()
    # occupancy_grid = torch.load(fname)
    # # save this as a point cloud to ply file?
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(occupancy_grid.nonzero().cpu().numpy())
    # fname = 'tmp.ply'
    # o3d.io.write_point_cloud(fname, pcd)
    # breakpoint()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/store/real/mandi/real2code_dataset_v0/")
    parser.add_argument("--obj_type", type=str, default="*")
    parser.add_argument("--obj_folder", type=str, default="*")
    parser.add_argument("--loop_id", type=str, default="0")
    parser.add_argument("--np_seed", type=int, default=0)
    parser.add_argument("--pcd_size", type=int, default=10000)
    parser.add_argument("--num_use_cameras", type=int, default=12)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out_dir", type=str, default="/store/real/mandi/r2c2r_data/") 
    parser.add_argument("--overwrite", "-o", action="store_true") 
    parser.add_argument("--voxel_grid_size", '-vg', type=int, default=64)
    args = parser.parse_args() 
    main(args)