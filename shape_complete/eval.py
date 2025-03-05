import os
import torch
import wandb 
import json
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
from einops import rearrange
from shape_complete.models import ShapeCompletionModel
from shape_complete.dataset import ShapeCompletionEvalDataset
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from kaolin.ops.mesh import sample_points
from kaolin.io.obj import import_mesh
from kaolin.ops.conversions import pointclouds_to_voxelgrids, voxelgrids_to_trianglemeshes
import trimesh 
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
from collections import defaultdict
from glob import glob 
from natsort import natsorted
# from pytorch3d.ops import sample_points_from_meshes
# from pytorch3d.loss import chamfer_distance # pytorch3d is bugged
from shape_complete.datagen import get_handcraft_obb
import sys 
sys.path.append("../real2code")

"""
run model on all query points and extract mesh  

RUN=scissors_eyeglasses_query6000_inp1024_qr0.2
LS=28000
python shape_complete/eval.py -r $RUN -ls $LS --save_mesh --obj_folder 10907

# use SAM-generated seg pcd:
RUN=scissors_eyeglasses_query6000_inp1024_qr0.2
LS=28000
python shape_complete/eval.py -r $RUN -ls $LS --save_mesh --obj_folder 101860 --eval_sam

# eval with the older shape completion model
python shape_complete/eval.py  -r v4_shiftidx_query12000_inp2048_qr0.25 -ls 165000 --log_dir /local/real/mandi/shape_models/ --obj_type Microwave
# the SAM model is also different here
python shape_complete/eval.py  -r v4_shiftidx_query12000_inp2048_qr0.25 -ls 165000 --log_dir /local/real/mandi/shape_models/ \
    --obj_type Microwave --eval_sam     --sam_result_dir /store/real/mandi/real2code_eval_v0/rebuttal_full_pointsTrue_lr0.001_bs21_ac12_12-01_19-52/ckpt_step_11000 
    
""" 

VOXEL_SIZE = 96
def get_uniform_query_points(grid_size=VOXEL_SIZE):
    """ sample query points uniformly """
    all_grid_idxs = torch.arange(0, grid_size**3)
    query_points = torch.tensor(np.array(np.unravel_index(all_grid_idxs, (grid_size, grid_size, grid_size))).T) 
    query_points = query_points / grid_size # range [0, 1]
    return query_points

def normalize_pcd(pcd, center, extent, R):
    return ((pcd - center) @ R) / extent

def run_on_seg_data(model, obj_pcd_dir, merge_raw_pcd=False):
    # e.g. /store/real/mandi/real2code_eval_v0/scissors_eyeglasses_only_pointsTrue_lr0.001_bs24_ac24_11-30_00-23/ckpt_epoch_110/Eyeglasses/101845/loop_0/filled_pcd_0.ply
    pcd_fnames = natsorted(glob(join(obj_pcd_dir, "filled*.ply")))
    query_pts = get_uniform_query_points().cuda()[None]
    results = []
    for fname in pcd_fnames:
        pcd = o3d.io.read_point_cloud(fname)
        raw_pcd = np.array(pcd.points)  
        obb = get_handcraft_obb(raw_pcd)  
        extent = obb['extent'] * 1.5 # add margin! 
        center = obb['center']
        R = obb['R']

        raw_pcd = normalize_pcd(raw_pcd, center, extent, R)
        input_pcd = raw_pcd.copy()

         # normalize input pcd: 
        input_pcd = torch.from_numpy(input_pcd).float().cuda()[None]
        # run shape completion
        with torch.no_grad():
            pred = model(input_pcd, query_pts)
        logits = torch.sigmoid(pred).detach()
        mesh_origin = np.array([-0.5, -0.5, -0.5])
        pred_voxelgrid = rearrange(logits, 'b (x y z) -> b x y z', x=VOXEL_SIZE, y=VOXEL_SIZE, z=VOXEL_SIZE)
        # add the input pcd to the voxelgrid??
        if merge_raw_pcd:
            input_voxelgrid = pointclouds_to_voxelgrids(
                input_pcd, resolution=VOXEL_SIZE, 
                origin=torch.tensor(mesh_origin)[None].cuda(), 
                scale=torch.ones(3)[None].cuda()
            )
            pred_voxelgrid += input_voxelgrid
        # extract mesh
        verts, faces = voxelgrids_to_trianglemeshes(pred_voxelgrid, iso_value=0.7)
        vertices = verts[0].cpu().numpy()
        faces = faces[0].cpu().numpy()
        vertices = vertices / VOXEL_SIZE + mesh_origin
        vertices = vertices * extent  
        vertices = vertices @ R.T + center

        save_fname = fname.replace("filled_pcd", "pred_mesh").replace(".ply", ".obj")
        results.append((fname, save_fname, vertices, faces))
    return results


def points_chamfer_distance(points1, points2, num_points=100000):
     # one direction
    tree1 = KDTree(points1)
    one_distances, one_vertex_ids = tree1.query(points2)
    dist1 = np.mean(np.square(one_distances))

    # other direction
    tree2 = KDTree(points2)
    two_distances, two_vertex_ids = tree2.query(points1)
    dist2 = np.mean(np.square(two_distances))
    return dist1 + dist2

def chamfer_distance(mesh1, mesh2, num_points=100000):
    # print('Warning!! Asume first mesh is GT mesh for normnalization')
    points1 = mesh1.sample(num_points)
    points2 = mesh2.sample(num_points)
    chamfer = points_chamfer_distance(points1, points2)
    # then normalize by first points, assume first is GT
    gt_center = np.mean(points1, axis=0)
    gt_scale = np.max(points1, axis=0) - np.min(points1, axis=0)
    points1_norm = (points1 - gt_center) / gt_scale
    points2_norm = (points2 - gt_center) / gt_scale
    chamfer_norm = points_chamfer_distance(points1_norm, points2_norm)
    return chamfer, chamfer_norm

def remove_mesh_clusters(mesh): 
    # mesh = mesh.filter_smooth_laplacian(10, lambda_filter=0.5)
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    # remove small clusters
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles) 
    cluster_area = np.asarray(cluster_area)
    # print(cluster_n_triangles)
    size_thres = np.max([n for n in cluster_n_triangles])
    triangles_to_remove = cluster_n_triangles[triangle_clusters] != size_thres
    mesh.remove_triangles_by_mask(triangles_to_remove)
    return mesh

def create_dataset(args_fname, args):
    with open(args_fname, "r") as f:
        dataset_kwargs = json.load(f)['dataset_kwargs']
    print(f"Loaded dataset kwargs: {dataset_kwargs}")
    # setup dataset and loader 
    dataset_kwargs['data_dir'] = args.data_dir
    dataset_kwargs['split'] = args.split 
    dataset_kwargs['obj_type'] = args.obj_type
    dataset_kwargs['obj_folder'] = args.obj_folder
    dataset_kwargs['query_size'] = args.num_query_points
    dataset_kwargs['query_surface_ratio'] = args.query_surface_ratio
    dataset_kwargs['loop_dir'] = str(args.loop_dir)

    skip_extents = dataset_kwargs.get('skip_extents', False)
    use_max_extents = dataset_kwargs.get('use_max_extents', False)
    val_dataset = ShapeCompletionEvalDataset(**dataset_kwargs)
    return val_dataset, skip_extents, use_max_extents

def get_link_name(mesh_fname): 
    obj_folder = mesh_fname.split("/")[-3]
    obj_type = mesh_fname.split("/")[-4]
    loop_id = mesh_fname.split("/")[-2]
    link_name = mesh_fname.split("/")[-1].split("_rot")[0]
    link_name = f"{obj_type}_{obj_folder}_{loop_id}_{link_name}"
    return link_name, obj_folder, obj_type, loop_id

def run(args):
    args_fname = join(args.log_dir, args.resume, "args.json")
    assert os.path.exists(args_fname), f"Args file {args_fname} does not exist"
    
    model = ShapeCompletionModel(
        agg_args=args.agg_args,
        unet_args=args.unet_args,
        decoder_args=args.decoder_args
        )
    model = model.cuda()
    model_fname = join(args.log_dir, args.resume, f"step_{args.load_step}", f"model_{args.load_step}.pth")
    assert os.path.exists(model_fname), f"Model file {model_fname} does not exist"
    print(f"Loading model from {model_fname}")
    model.load_state_dict(torch.load(model_fname))

    if args.eval_sam:
        print(f"Loading SAM model {args.sam_result_dir}")
        sam_result_dir = join(
            args.eval_output_dir, 
            args.sam_result_dir, 
        )
        assert os.path.exists(sam_result_dir), f"Model file {sam_result_dir} does not exist"
        lookup = join(sam_result_dir, args.obj_type, args.obj_folder, args.loop_dir)
        folders = natsorted(glob(lookup))
        print(f"Found {len(folders)} folders from {lookup}")
        for folder in folders:
            print(f"Running on {folder}")
            results = run_on_seg_data(model, folder, merge_raw_pcd=False)
            for _, save_fname, vertices, faces in results: 
                mesh = trimesh.Trimesh(vertices, faces)
                mesh.export(save_fname)
        return 

    val_dataset, skip_extents, use_max_extents = create_dataset(args_fname, args)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
        ) 
    voxel_size = val_dataset.voxel_size
    if args.wandb:
        run = wandb.init(project="real2code", group="shape", name=f"eval_{args.resume}")
        wandb.config.update(args)
    
    eval_save_dir = join(args.eval_output_dir, args.resume, f"step_{args.load_step}") 
    os.makedirs(eval_save_dir, exist_ok=True)     
    all_chamfer = defaultdict(list)
    for i, data in tqdm(
        enumerate(val_dataloader), 
        total=len(val_dataloader),
        desc="Validation", 
    ):
        input, query, label = data['input'].cuda(), data['query'].cuda(), data['label'].cuda()
        R = data['R'][0].cpu().numpy()
        center = data['center'][0].cpu().numpy()
        extent = data['extent'][0].cpu().numpy()
        
        link_name, obj_folder, obj_type, loop_id = get_link_name(data['mesh_fname'][0])
        obj_save_dir = join(eval_save_dir, obj_type, obj_folder, loop_id)
        os.makedirs(obj_save_dir, exist_ok=True)

        pcd_fname = join(obj_save_dir, f"input_{link_name}.ply")
        if not os.path.exists(pcd_fname):
            pcd = o3d.geometry.PointCloud()  
            pcd.points = o3d.utility.Vector3dVector(input[0].cpu().numpy())
            o3d.io.write_point_cloud(pcd_fname, pcd)
        raw_pcd_fname = join(obj_save_dir, f"pcd_{link_name}.ply")
        if not os.path.exists(raw_pcd_fname):
            pcd = o3d.geometry.PointCloud()
            raw_pcd = data['raw_pcd'][0]
            pcd.points = o3d.utility.Vector3dVector(raw_pcd.cpu().numpy())
            o3d.io.write_point_cloud(raw_pcd_fname, pcd)

        save_fname = join(obj_save_dir, f"pred_{link_name}.obj")
        if os.path.exists(save_fname) and not args.overwrite:
            continue
        pred = model(input, query)
        logits = torch.sigmoid(pred).detach()

        pred_np, logits_np = pred.cpu().detach().numpy(), logits.cpu().detach().numpy()
        label_np = label.cpu().detach().numpy()
        loss = model.compute_loss(pred, label)
        print(f"Loss: {loss.item()}")

        # pred_voxelgrid = torch.tensor(logits > 0.5, dtype=torch.float32)
        pred_voxelgrid = rearrange(logits, 'b (x y z) -> b x y z', x=voxel_size, y=voxel_size, z=voxel_size) 
        
        if not skip_extents:
            mesh_origin = np.array([-0.5, -0.5, -0.5])
            mesh_scale = 1 
        else:
            mesh_origin = np.array([-1, -1, -1])
            mesh_scale = 2
        # add input voxelgrid
        input_voxelgrid = pointclouds_to_voxelgrids(
            input, resolution=voxel_size, 
            origin=torch.tensor(mesh_origin)[None].to(input.device), 
            scale=torch.tensor(mesh_scale)[None].to(input.device)
        )
        pred_voxelgrid += input_voxelgrid
        verts, faces = voxelgrids_to_trianglemeshes(pred_voxelgrid, iso_value=args.iso_value) # ths is ranged (0, 96)
        vertices = verts[0].cpu().numpy()
        faces = faces[0].cpu().numpy()

        vertices = vertices / 96 + mesh_origin
        if not skip_extents and not use_max_extents:
            vertices = vertices * extent 
        if use_max_extents:
            vertices = vertices * np.max(extent)
        vertices = vertices @ R.T + center
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        # smooth
        # o3d_mesh = o3d_mesh.filter_smooth_laplacian(20) 
        o3d_mesh = remove_mesh_clusters(o3d_mesh)
        if args.save_mesh:
            o3d.io.write_triangle_mesh(save_fname, o3d_mesh)
        pred_vertices = np.array(o3d_mesh.vertices, dtype=np.float32)
        pred_faces = np.array(o3d_mesh.triangles, dtype=np.int64) 
        pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces) 

        label_grid = rearrange(label, 'b (x y z) -> b x y z', x=voxel_size, y=voxel_size, z=voxel_size)
        verts, fs = voxelgrids_to_trianglemeshes(label_grid, iso_value=args.iso_value)
        label_vertices = verts[0].cpu().numpy() / voxel_size + mesh_origin
        label_faces = fs[0].cpu().numpy() 
        if not skip_extents and not use_max_extents:
            label_vertices *= extent
        if use_max_extents:
            label_vertices *= np.max(extent)
        label_vertices = label_vertices @ R.T + center
        label_mesh = trimesh.Trimesh(label_vertices, label_faces)
        
        gt_mesh = trimesh.load(data['mesh_fname'][0]) 
        # rotate gt_mesh around z axis by 90:
        # gt_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1]))
        if args.save_mesh:
            label_mesh.export(join(obj_save_dir, f"label_{link_name}.obj"))
            gt_mesh.export(join(obj_save_dir, f"gt_{link_name}.obj")) 
        # compute chamfer distance 
        chamfer, chamfer_norm = chamfer_distance(gt_mesh, pred_mesh) # gt mesh at front!!!
        # print(f"Chamfer distance: {chamfer}") 
        mesh_name = data['mesh_fname'][0]  

        # code name 
        link_id = mesh_name.split("link_")[-1].split("_")[0] 
        obj_dir = "/".join(mesh_name.split("/")[:-2]) 
        code_dir = obj_dir.replace(args.data_dir, args.code_dataset_dir)
        code_fname = join(code_dir, f"obb_info_loop_0.json")
        if not os.path.exists(code_fname):
            print(f"WARNING! Missing {code_fname}")
            is_static = False
        else:
            with open(code_fname, "r") as f:
                code_info = json.load(f)['test_code']
            root_id = code_info.split('root_geom = ')[-1].split('\n')[0]
            is_static = (link_id == root_id)
        
        mesh_stats = {
                "link_id": link_id,
                "chamfer": float(chamfer),
                "chamfer_norm": float(chamfer_norm),
                "is_static": is_static,
            }
        # save with json
        json_fname = join(obj_save_dir, f"stats_{link_name}.json")
        with open(json_fname, "w") as f:
            json.dump(mesh_stats, f)
        mesh_stats.update({
            "gt_mesh": gt_mesh,
            "pred_mesh": pred_mesh, 
        })
        all_chamfer[obj_folder].append(mesh_stats)

    static_chamfer, whole_chamfer, part_chamfer = get_whole_part_chamfer(all_chamfer)
    print_stats(static_chamfer, whole_chamfer, part_chamfer)
    return 
        
def get_whole_part_chamfer(all_chamfer):
    static_chamfer = []
    part_chamfer = []
    whole_chamfer = []
    for obj_id, chamfers in all_chamfer.items():
        all_gt_mesh = []
        all_pred_mesh = []
        for chamfer in chamfers:
            toappend = static_chamfer if chamfer['is_static'] else part_chamfer
            toappend.append((chamfer['chamfer'], chamfer['chamfer_norm']))
            
            all_gt_mesh.append(chamfer['gt_mesh'])
            all_pred_mesh.append(chamfer['pred_mesh'])
        whole_gt_mesh = trimesh.util.concatenate(all_gt_mesh)
        whole_pred_mesh = trimesh.util.concatenate(all_pred_mesh)
        # print(f"num of gt meshes concated: {len(all_gt_mesh)}")
        # whole_gt_mesh.export(f"whole_gt_{obj_id}.obj")
        # whole_pred_mesh.export(f"whole_pred_{obj_id}.obj")

        whole_chamfer.append(
            tuple(chamfer_distance(whole_gt_mesh, whole_pred_mesh))
        )

    return static_chamfer, whole_chamfer, part_chamfer

def print_stats(static_chamfer, whole_chamfer, part_chamfer):
    for name, dists in zip(
        ["static", "whole", "part"],
        [static_chamfer, whole_chamfer, part_chamfer]
        ):
        chamfer = np.mean([d[0] for d in dists])
        chamfer_norm = np.mean([d[1] for d in dists])
        print(f"{name}: chamfer: {np.round(np.mean(chamfer) * 1000, 3)} | chamfer_norm: {np.round(np.mean(chamfer_norm) * 1000, 3)}")
    return 

def pool_stats(args):
    args_fname = join(args.log_dir, args.resume, "args.json")
    assert os.path.exists(args_fname), f"Args file {args_fname} does not exist"
    with open(args_fname, "r") as f:
        dataset_kwargs = json.load(f)['dataset_kwargs']
    val_dataset, skip_extents, use_max_extents = create_dataset(args_fname, args)
    all_stats = defaultdict(list)
    eval_save_dir = join(args.eval_output_dir, args.resume, f"step_{args.load_step}") 
    for i, data in enumerate(val_dataset):
        link_name, obj_folder, obj_type, loop_id = get_link_name(data['mesh_fname'])
        obj_save_dir = join(eval_save_dir, obj_type, obj_folder, loop_id)
        os.makedirs(obj_save_dir, exist_ok=True)
        
        save_fname = join(obj_save_dir, f"pred_{link_name}.obj")
        stats_fname = join(obj_save_dir, f"{args.split}_stats_{link_name}.json")
        if not os.path.exists(save_fname) or not os.path.exists(stats_fname):
            print(f"WARNING! Missing {save_fname}")
            continue

        pred_mesh = trimesh.load(save_fname)
        gt_mesh = trimesh.load(data['mesh_fname'])
        with open(stats_fname, "r") as f:
            mesh_stats = json.load(f)
        link_id, chamfer, is_static = mesh_stats['link_id'], mesh_stats['chamfer'], mesh_stats['is_static']
        
        chamfer, chamfer_norm = chamfer_distance(gt_mesh, pred_mesh)
        mesh_stats = {
            "link_id": link_id,
            "chamfer": float(chamfer),
            "chamfer_norm": float(chamfer_norm),
            "is_static": is_static,
            "gt_mesh": gt_mesh,
            "pred_mesh": pred_mesh,
        }
        if chamfer_norm > 7:
            print(obj_folder, link_name, chamfer_norm*1000)
            continue
        all_stats[obj_folder].append(mesh_stats)
        
    
    static_chamfer, whole_chamfer, part_chamfer = get_whole_part_chamfer(all_stats)
    print_stats(static_chamfer, whole_chamfer, part_chamfer)    
    return all_stats

        

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    # dataset and loader:
    parser.add_argument("--data_dir", type=str, default="/store/real/mandi/real2code_shape_dataset_v0")
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--num_input_points", "-i", type=int, default=1024)
    parser.add_argument("--num_query_points", "-q", type=int, default=8000)
    parser.add_argument("--query_surface_ratio", "-qr", type=float, default=0.4)
    parser.add_argument("--num_workers", "-w", type=int, default=0)
    parser.add_argument("--obj_type", type=str, default="*")    
    parser.add_argument("--obj_folder", type=str, default="*")  
    parser.add_argument("--cache_mesh", action="store_true")
    parser.add_argument("--loop_dir", type=str, default="*")
    parser.add_argument("--load_voxelgrid", action="store_true")
    parser.add_argument("--voxel_size", "-vs", type=int, default=64)
    parser.add_argument("--rot_aug", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    # eval:
    parser.add_argument("--iso_value", "-iso", type=float, default=0.7)
    parser.add_argument("--save_mesh", "-sm", default=True, action="store_true")
    parser.add_argument("--code_dataset_dir", type=str, default="/store/real/mandi/real2code_dataset_v0") # need this for static/mobile part distinction
    parser.add_argument("--eval_output_dir", type=str, default="/store/real/mandi/real2code_eval_v0/")
    # model:
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", "-e", type=int, default=100)
    parser.add_argument("--agg_args", type=dict, default=dict())
    parser.add_argument("--unet_args", type=dict, default=dict(in_channels=128, out_channels=128))
    parser.add_argument("--decoder_args", type=dict, default=dict())
    parser.add_argument("--val_interval", "-vi", type=int, default=100)
    parser.add_argument("--num_val_steps", "-v", type=int, default=100)
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--dp_devices", "-dp", type=int, default=2)
    parser.add_argument("--resume", "-r", type=str, default="v4_shiftidx_query10000_inp2048_qr0.3")
    parser.add_argument("--load_step", "-ls", type=int, default=95000)
    # logging:
    parser.add_argument("--log_dir", "-ld", type=str, default="/store/real/mandi/real2code_shape_models/")
    parser.add_argument("--log_interval", "-log", type=int, default=50)
    parser.add_argument("--save_interval", "-save", type=int, default=5000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", "-rn", type=str, default="test")
    parser.add_argument("--overwrite", "-o", action="store_true") 

    # load sam results
    parser.add_argument('--eval_sam', '-es', action="store_true")
    # contain both run name and epoch
    parser.add_argument('--sam_result_dir', default="scissors_eyeglasses_only_pointsTrue_lr0.001_bs24_ac24_11-30_00-23/ckpt_epoch_110")

    parser.add_argument("--pool_stats", "-p", action="store_true")
    args = parser.parse_args()
    if args.pool_stats:
        all_stats = pool_stats(args)
        # breakpoint()
        exit() 
    run(args)
    print("Done")



