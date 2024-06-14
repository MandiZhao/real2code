import os
from os.path import join
from glob import glob
import numpy as np
import trimesh
import json 
from natsort import natsorted
import kaolin as kal
from kaolin.io.obj import import_mesh
from kaolin.ops.mesh import sample_points, index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.conversions import trianglemeshes_to_voxelgrids, sdf_to_voxelgrids, voxelgrids_to_trianglemeshes
import sys 
sys.path.append("../real2code")
from data_utils import get_tight_obb as obb_from_axis
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from time import time
import wandb
from einops import rearrange
OBJ_FNAME = "*_rot.obj"
RAW_MESH_ORIGIN = torch.tensor([-1,-1,-1])[None]
RAW_MESH_SCALE = torch.tensor([2,2,2])[None]
SKIP_IDS=[46874, 46893, 46922, 46944, 46981]

class ShapeCompletionDataset(Dataset):
    """
    TODO: add z-axis augmentation rotaiton
    """
    def __init__(
            self, 
            data_dir="/local/real/mandi/shape_dataset_v3", 
            split="test", 
            obj_type="*", 
            obj_folder="*", # 
            transform=None,
            input_size=1000,
            query_size=1000,
            query_surface_ratio=0.7, # sample a mixture of surface and volume points
            normalize_with_obb=True,
            skip_extents=False,
            use_max_extents=False,
            voxel_size=96,
            seed=42,
            dist_thresh=0.01, # use to determine if a point is on mesh
            loop_dir="*",
            cache_mesh=False,
            load_voxelgrid=False,
            rot_aug=False,
            load_pcd_size=-1,
            use_aabb=False,
            aug_obb=False,
        ):
        self.data_dir = data_dir
        self.split = split
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        self.obj_type = obj_type
        self.obj_folder = obj_folder
        self.loop_dir = loop_dir
        self.transform = transform
        self.input_size = input_size
        self.query_size = query_size
        self.query_surface_ratio = query_surface_ratio
        self.voxel_size = voxel_size
        self.normalize_with_obb = normalize_with_obb
        self.skip_extents = skip_extents
        self.use_max_extents = use_max_extents
        self.dist_thresh = dist_thresh
        self.cache_mesh = cache_mesh
        self.load_voxelgrid = load_voxelgrid
        self.rot_aug = rot_aug
        self.aug_obb = aug_obb
        if aug_obb:
            assert not load_voxelgrid, "aug_obb and load_voxelgrid cannot be True at the same time"
        self.load_pcd_size = load_pcd_size
        if load_pcd_size > 0:
            assert input_size <= load_pcd_size, "input_size must be <= load_pcd_size"
        self.use_aabb = use_aabb
        self.mesh_info = self.load_data()

        print(f"Loaded {len(self.mesh_info)} meshes")
        torch.manual_seed(seed)
        
    def load_data(self):
        """ Load GT rotated mesh, GT OBB, and partial PCD"""
        lookup_path = os.path.join(self.data_dir, self.split, self.obj_type, self.obj_folder)
        obj_dirs = natsorted(glob(lookup_path))
        all_mesh_info = []  
        for obj_dir in obj_dirs:
            if int(os.path.basename(obj_dir)) in SKIP_IDS:
                continue
            loop_dirs = natsorted(glob(join(obj_dir, f"loop_{self.loop_dir}")))
            for loop_dir in loop_dirs:
                mesh_fnames = natsorted(glob(join(loop_dir, OBJ_FNAME)))
                for fname in mesh_fnames:
                    link_name = os.path.basename(fname).replace("_rot.obj", "")
                    partial_pcd_fname = join(loop_dir, link_name + ".npz")
                    if self.load_pcd_size > 0:
                        partial_pcd_fname = partial_pcd_fname.replace(".npz", f"_{self.load_pcd_size}.npz")
                    assert os.path.exists(partial_pcd_fname), f"Partial PCD {partial_pcd_fname} does not exist"
                    # gt_obb_fname = join(loop_dir, f"obb_{link_name}.json")
                    gt_obb_fname = join(loop_dir, f"partial_{link_name}.json")
                    if self.use_aabb:
                        gt_obb_fname = join(loop_dir, f"aabb_{link_name}.json") 
                    
                    # np.load pcd
                    pcd = np.load(partial_pcd_fname)
                    if pcd['points'].shape[0] < 100:
                        print(f"Partial PCD {partial_pcd_fname} has less than 100 points")
                        continue 
                    gt_obb = json.load(open(gt_obb_fname, "r"))  
                    info = dict(
                        mesh_fname=fname, 
                        pcd_fname=partial_pcd_fname,
                        gt_obb=dict(
                            center=np.array(gt_obb['center']),
                            extent=np.array(gt_obb['extent']),
                            R=np.array(gt_obb.get('R', np.eye(3)))
                        ),
                        mesh=None,
                    )
                    if self.load_voxelgrid:
                        mesh_voxelgrid_name = join(loop_dir, link_name + f"_occ_partial_{self.voxel_size}.pt")
                        if self.skip_extents or self.use_max_extents:
                            mesh_voxelgrid_name = join(loop_dir, link_name + f"_occ_norm_{self.voxel_size}_noextent.pt")
                        if self.use_aabb:
                            mesh_voxelgrid_name = mesh_voxelgrid_name.replace(".pt", "_aabb.pt")
                        assert os.path.exists(mesh_voxelgrid_name), f"Voxelgrid {mesh_voxelgrid_name} does not exist"
                        try:
                            voxelgrid = torch.load(mesh_voxelgrid_name)
                        except:
                            print(f"Error loading {mesh_voxelgrid_name}")
                            continue
                        info['voxelgrid'] = mesh_voxelgrid_name
                    all_mesh_info.append(info) 
        return all_mesh_info
    
    def set_seed(self, seed):
        self.np_random = np.random.RandomState(seed)
        torch.manual_seed(seed)

    def get_input_pcd(self, pcd_fname, center, extents, R):
        pcd = np.load(pcd_fname)['points'] 
        raw_size = pcd.shape[0]
        pcd = torch.tensor(pcd, dtype=torch.float32).cuda()
        if raw_size != self.input_size:
            sampled_idxs = torch.randint(0, raw_size, (self.input_size,)).cuda() 
            pcd = pcd[sampled_idxs]
        if self.normalize_with_obb:
            pcd = pcd - center.to(pcd.device)
            pcd = pcd @ R.to(pcd.device)
            if not self.skip_extents and not self.use_max_extents:
                pcd = pcd / extents.to(pcd.device) 
            if self.use_max_extents:
                max_extent = torch.max(extents)
                pcd = pcd / max_extent
        return pcd
    
    def normalize_mesh_with_obb(self, vertices, faces, center, extents, R): 
        vertices = vertices - center 
        vertices = vertices @ R
        if not self.skip_extents and not self.use_max_extents:
            vertices = vertices / extents
        if self.use_max_extents:
            max_extent = torch.max(extents)
            vertices = vertices / max_extent
        return vertices, faces
    
    def get_query_points(self, voxelgrid):
        """ sample query points from mesh surface and volume """
        all_grid_idxs = torch.arange(0, voxelgrid.shape[0] * voxelgrid.shape[1] * voxelgrid.shape[2]).to(voxelgrid.device)
        nonzero_idxs = torch.nonzero(voxelgrid)
        # randomly add -1, 0, or 1 to shift the idxs as augmentation
        nonzero_idxs += torch.randint(-1, 2, size=nonzero_idxs.shape).to(nonzero_idxs.device)
        # clip `nonzero_idxs` to be within the range of `voxelgrid.shape
        nonzero_idxs = torch.clamp(nonzero_idxs, 0, voxelgrid.shape[0] - 1)
        num_surface = int(self.query_size * self.query_surface_ratio) # sample from surface, i.e. non-zero voxels

        # sampled_idxs = torch.randperm(nonzero_idxs.shape[0])[:num_surface]
        sampled_idxs = torch.randint(0, nonzero_idxs.shape[0], (num_surface,))
        surface_idxs = nonzero_idxs[sampled_idxs] 

        num_volume = int(self.query_size - num_surface) # sample uniformly from cube  
        
        zero_idxs = torch.where(voxelgrid == 0, 1, 0).nonzero(as_tuple=False)  
        
        sampled_idxs = torch.randint(0, zero_idxs.shape[0], (num_volume,)).to(zero_idxs.device)
        volume_idxs = zero_idxs[sampled_idxs] 

        query_points = torch.cat([surface_idxs, volume_idxs], dim=0)    
        labels = voxelgrid[query_points[:, 0], query_points[:, 1], query_points[:, 2]]
        
        query_points = query_points / voxelgrid.shape[0] # range [0, 1]
        return query_points, labels
        
    def __len__(self):
        return len(self.mesh_info)
    
    def get_all_query_points(self, voxelgrid):
        """ sample query points uniformly """
        all_grid_idxs = torch.arange(0, voxelgrid.shape[0] * voxelgrid.shape[1] * voxelgrid.shape[2])
        
        query_points = torch.tensor(np.array(np.unravel_index(all_grid_idxs, voxelgrid.shape)).T) 
        labels = voxelgrid[query_points[:, 0], query_points[:, 1], query_points[:, 2]]
        
        query_points = query_points / voxelgrid.shape[0] # range [0, 1]
        return query_points, labels
    
    def __getitem__(self, idx):
        mesh_info = self.mesh_info[idx]
        gt_obb = mesh_info['gt_obb']
        center, extent, R = gt_obb['center'], gt_obb['extent'], gt_obb['R']
        # extent += 0.03 # add a small margin to extent 
        if self.aug_obb:
            # cannot do absolute margin here, it would quickly compress some thin meshes into nothing
            aug_center = center * self.np_random.uniform(0.9, 1.1, 3)
            aug_extent = extent * self.np_random.uniform(0.9, 1.1, 3) 
            aug_center = torch.tensor(aug_center, dtype=torch.float32).cuda()
            aug_extent = torch.tensor(aug_extent, dtype=torch.float32).cuda()

        center = torch.tensor(center, dtype=torch.float32).cuda()
        extent = torch.tensor(extent, dtype=torch.float32).cuda()
        R = torch.tensor(R, dtype=torch.float32).cuda()
        if self.rot_aug:
            assert not self.load_voxelgrid
            # randomly rotate around z-axis
            angle = self.np_random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            aug_R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
        if self.load_voxelgrid: 
            voxelgrid = torch.load(mesh_info['voxelgrid']).cuda()
        else:
            mesh_info = mesh_info
            # start_time = time()
            mesh = mesh_info['mesh']
            if mesh is None:
                mesh = import_mesh(mesh_info['mesh_fname'])
                if self.cache_mesh:
                    mesh_info['mesh'] = mesh
            # mesh_load_time = time() - start_time
            mesh_vertices, mesh_faces = mesh.vertices.cuda(), mesh.faces.cuda()
            if self.normalize_with_obb:
                if self.aug_obb: 
                    vertices, faces = self.normalize_mesh_with_obb(mesh_vertices, mesh_faces, aug_center, aug_extent, R)
                else:
                    vertices, faces = self.normalize_mesh_with_obb(mesh_vertices, mesh_vertices, center, extent, R)
                if self.skip_extents:
                    mesh_origin = RAW_MESH_ORIGIN
                    mesh_scale = RAW_MESH_SCALE
                else: # this also includes use_max_extents=True
                    mesh_origin = torch.tensor([-0.5,-0.5,-0.5])[None]
                    mesh_scale = torch.tensor([1,1,1])[None]
            else:
                mesh_origin = torch.min(mesh_vertices, dim=0)[0][None]
                mesh_scale = (torch.max(mesh_vertices, dim=0)[0] - mesh_origin)[None] 
            
            if self.rot_aug:
                # rotate mesh
                vertices = vertices @ aug_R.T
            voxelgrid = trianglemeshes_to_voxelgrids(   
                vertices[None].cuda(), faces.cuda(), 
                resolution=self.voxel_size, 
                origin=mesh_origin.cuda(),
                scale=mesh_scale.cuda()
                )
            # check if normalization made it all zero
            if self.aug_obb and torch.sum(voxelgrid) == 0:
                aug_center, aug_extent = center, extent
                vertices.detach()
                faces.detach()
                del vertices, faces
                voxelgrid.detach()
                del voxelgrid 
                vertices, faces = self.normalize_mesh_with_obb(mesh_vertices, mesh_faces, center, extent, R)
                voxelgrid = trianglemeshes_to_voxelgrids(   
                    vertices[None].cuda(), faces.cuda(), 
                    resolution=self.voxel_size, 
                    origin=mesh_origin.cuda(),
                    scale=mesh_scale.cuda()
                )
        if self.aug_obb:
            center, extent = aug_center, aug_extent
        partial_pcd = self.get_input_pcd(mesh_info["pcd_fname"], center, extent, R) 
        if self.rot_aug:
            # rotate pcd
            partial_pcd = partial_pcd @ aug_R.T.to(partial_pcd.device)
        query_points, labels = self.get_query_points(voxelgrid[0]) # shape (Q, 3)  
        data = dict(
            input=partial_pcd,
            label=labels,
            query=query_points, # shape (Q, 3) 
        ) 
        return data
 

class ShapeCompletionEvalDataset(ShapeCompletionDataset):
    """ Use for validation and mesh extraction, query points uniformly """
    def __init__(self, *args, **kwargs):
        super(ShapeCompletionEvalDataset, self).__init__(*args, **kwargs) 
        print(f"Loaded {len(self.mesh_info)} meshes")
    
    def get_partial_obb(self, pcd_points: np.ndarray):
        obbs = []
        for axis_idx in range(3):
            obb_dict = obb_from_axis(pcd_points, axis_idx)
            obbs.append(obb_dict)
        bbox_sizes = [np.prod(obb['extent']) for obb in obbs]
        bbox_sizes[2] /= 1.5 # prioritize z axis 
        min_size_idx  = np.argmin(bbox_sizes) 
        obb_dict = obbs[min_size_idx] # {center, extent, R} 
        return obb_dict

    def get_partial_aabb(self, pcd_points: np.ndarray):
        aabb = {
            "center": np.mean(pcd_points, axis=0),
            "extent": np.max(pcd_points, axis=0) - np.min(pcd_points, axis=0),
            "R": np.eye(3)
        }
        return aabb

    def __getitem__(self, idx):
        """ Use GT OBB for now. Need to get estimate from partial pcd later"""
        mesh_info = self.mesh_info[idx]
        gt_obb = mesh_info['gt_obb']
        if not self.use_aabb:
            partial_obb = self.get_partial_obb(np.load(mesh_info['pcd_fname'])['points'])
            obb = partial_obb
        else:
            obb = self.get_partial_aabb(np.load(mesh_info['pcd_fname'])['points'])
        # obb = gt_obb
        center, extent, R = obb['center'], obb['extent'], obb['R']
        # center, extent, R = gt_obb['center'], gt_obb['extent'], gt_obb['R']
        extent *= 1.5 # add a small margin to extent
        
        center = torch.tensor(center, dtype=torch.float32)
        extent = torch.tensor(extent, dtype=torch.float32)
        R = torch.tensor(R, dtype=torch.float32)
        
        if self.load_voxelgrid:
            voxelgrid = torch.load(mesh_info['voxelgrid']).cuda()
        else:
            mesh_info = mesh_info
            # start_time = time()
            mesh = mesh_info['mesh']
            if mesh is None:
                mesh = import_mesh(mesh_info['mesh_fname'])
                if self.cache_mesh:
                    mesh_info['mesh'] = mesh
            # mesh_load_time = time() - start_time
            vertices, faces = mesh.vertices, mesh.faces
            if self.normalize_with_obb:
                vertices, faces = self.normalize_mesh_with_obb(vertices, faces, center, extent, R)
                mesh_origin = torch.tensor([-0.5,-0.5,-0.5])[None]
                if self.skip_extents:
                    mesh_scale = RAW_MESH_SCALE
                else:
                    mesh_scale = torch.tensor([1,1,1])[None]
            else:
                mesh_origin = torch.min(vertices, dim=0)[0][None]
                mesh_scale = (torch.max(vertices, dim=0)[0] - mesh_origin)[None] 
            
            voxelgrid = trianglemeshes_to_voxelgrids(   
                vertices[None].cuda(), faces.cuda(), 
                resolution=self.voxel_size, 
                origin=mesh_origin.cuda(),
                scale=mesh_scale.cuda()
                )
  
        partial_pcd = self.get_input_pcd(mesh_info["pcd_fname"], center, extent, R) 
         
        query_points, labels = self.get_all_query_points(voxelgrid[0]) # shape (Q, 3) 
 
        data = dict(
            mesh_fname=mesh_info['mesh_fname'],
            input=partial_pcd,
            label=labels,
            query=query_points, # shape (Q, 3)
            R=R,
            center=center,
            extent=extent,
        )
         
        return data


if __name__ == "__main__":
    # import wandb 
    # run = wandb.init(project="real2code", entity="mandi") 
    dataset = ShapeCompletionEvalDataset(
        normalize_with_obb=True, split="test", 
        obj_type="Box", obj_folder="*", loop_dir="0", 
        use_aabb=True,
        cache_mesh=False, query_surface_ratio=0.5, input_size=1024, load_pcd_size=1024,
        query_size=500, 
        load_voxelgrid=True
        )
    for i in range(10):
        data = dataset[i]
        pcd = data['input'][0]
        # print(f"Max xyz: {torch.max(pcd, dim=0)[0]}, Min xyz: {torch.min(pcd, dim=0)[0]}")
        label = data['label'] 
        R = data['R'].cpu().numpy()
        center = data['center'].cpu().numpy()
        extent = data['extent'].cpu().numpy()
        voxel_size = 96
        # print(data['input'].shape, data['query'].shape, data['label'].shape)
        label_grid = rearrange(label, '(x y z) -> x y z', x=voxel_size, y=voxel_size, z=voxel_size)
        verts, faces = voxelgrids_to_trianglemeshes(label_grid[None], iso_value=0.5)
        vertices = verts[0].cpu().numpy()
        faces = faces[0].cpu().numpy() 
        vertices = vertices * (extent + 0.1)
        vertices = vertices @ R.T + center
        _mesh = trimesh.Trimesh(vertices, faces)
        _mesh.export(join(f"aabbs/test_aabb_mesh_{i}.obj")) 
        

    breakpoint()
