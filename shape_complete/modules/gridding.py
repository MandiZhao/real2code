from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
# import torch_scatter


def batch_to_volume(batch, volume_size, reduce='mean'):
    batch_idx = batch.batch
    points = batch.pos
    features = batch.x
    # debug
    # features = torch.arange(-12000, len(points)-12000, device=points.device).repeat(3, 1).T.type(torch.float32)
    # points = features / volume_size

    batch_size = batch.num_graphs
    feature_size = features.shape[1]
    # compute volume index
    volume_idx_flat = None
    with torch.no_grad():
        grid_coord_f = points * volume_size
        grid_coord_i = torch.clamp(grid_coord_f.type(torch.int64), 0, volume_size-1)
        volume_idx_flat = \
            batch_idx * (volume_size ** 3) \
            + grid_coord_i[:,0] * (volume_size ** 2) \
            + grid_coord_i[:,1] * volume_size \
            + grid_coord_i[:,2]
    
    # scatter and aggregate to volume
    # TODO
    features = features.type(torch.float32)
    volume_feature_flat = torch_scatter.scatter(
        src=features.T, index=volume_idx_flat, 
        dim=-1, dim_size=batch_size*volume_size**3,
        reduce=reduce)
    
    # reshape to volume
    volume_feature = volume_feature_flat.reshape(
        (feature_size, batch_size, 
        volume_size, volume_size, volume_size)).permute((1,0,2,3,4))

    return volume_feature


def nocs_grid_sample(feature_volume: torch.Tensor, query_points: torch.Tensor, 
        mode:str='bilinear', padding_mode:str='border', 
        align_corners:bool=True) -> torch.Tensor:
    """
    feature_volume: (N,C,D,H,W) or (N,D,H,W) or (D,H,W)
    query_points: (N,M,3) or (M,3)
    return: (N,M,C) or (M,C)
    """
    # 1. processs query_points
    # normalize query points to (-1, 1), which is 
    # requried by grid_sample
    query_points_normalized = 2.0 * query_points - 1.0

    query_points_shape = None
    if len(query_points.shape) == 2:
        shape = tuple(query_points.shape)
        query_points_shape = (1, shape[0], 1, 1, shape[1])
    elif len(query_points.shape) == 3:
        shape = tuple(query_points.shape)
        query_points_shape = (shape[0], shape[1], 1, 1, shape[2])
    else:
        raise RuntimeError("Invalid query_points shape {}".format(
            str(query_points.shape)))
    query_points_reshaped = query_points_normalized.view(*query_points_shape)
    # sample_features uses zyx convension in coordinate, not xyz
    query_points_reshaped = query_points_reshaped.flip(-1)


    # 2. process feature_volume
    feature_volume_shape = None
    if len(feature_volume.shape) == 5:
        feature_volume_shape = tuple(feature_volume.shape)
    elif len(feature_volume.shape) == 4:
        shape = tuple(feature_volume.shape)
        feature_volume_shape = (shape[0], 1, shape[1], shape[2], shape[3])
    elif len(feature_volume.shape) == 3:
        feature_volume_shape = (1,1) + feature_volume.shape
    feature_volume_reshaped = feature_volume.view(*feature_volume_shape)

    # 3. sample
    # shape (N,C,M,1,1)
    sampled_features = F.grid_sample(
        input=feature_volume_reshaped, grid=query_points_reshaped,
        mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    
    # 4. reshape output
    out_features_shape = None
    if len(query_points.shape) == 2:
        out_features_shape = (query_points.shape[0], sampled_features.shape[1])
    elif len(query_points.shape) == 3:
        out_features_shape = query_points.shape[:2] + (sampled_features.shape[1],)
    out_features = sampled_features.permute(0,2,1,3,4).view(*out_features_shape)

    return out_features


class VirtualGrid:
    def __init__(self,
        lower_corner=(0,0,0), 
        upper_corner=(1,1,1), 
        grid_shape=(32, 32, 32),
        batch_size=8,
        device=torch.device('cpu'),
        int_dtype=torch.int64,
        float_dtype=torch.float32,
        ):
        self.lower_corner = tuple(lower_corner)
        self.upper_corner = tuple(upper_corner)
        self.grid_shape = tuple(grid_shape)
        self.batch_size = int(batch_size)
        self.device = device
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
    
    @property
    def num_grids(self):
        grid_shape = self.grid_shape
        batch_size = self.batch_size
        return int(np.prod((batch_size,) + grid_shape))

    def get_grid_idxs(self, include_batch=True):
        batch_size = self.batch_size
        grid_shape = self.grid_shape
        device = self.device
        int_dtype = self.int_dtype
        dims = grid_shape
        if include_batch:
            dims = (batch_size,) + grid_shape
        axis_coords = [torch.arange(0, x, device=device, dtype=int_dtype) 
            for x in dims]
        coords_per_axis = torch.meshgrid(*axis_coords)
        grid_idxs = torch.stack(coords_per_axis, axis=-1)
        return grid_idxs
    
    def get_grid_points(self, include_batch=True):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        float_dtype = self.float_dtype
        device = self.device
        grid_idxs = self.get_grid_idxs(include_batch=include_batch)
        
        lc = torch.tensor(lower_corner, dtype=float_dtype, device=device)
        uc = torch.tensor(upper_corner, dtype=float_dtype, device=device)
        idx_scale = torch.tensor(grid_shape, 
            dtype=float_dtype, device=device) - 1
        scales = (uc - lc) / idx_scale
        offsets = -lc

        grid_idxs_no_batch = grid_idxs
        if include_batch:
            grid_idxs_no_batch = grid_idxs[:,:,:,:,1:]
        grid_idxs_f = grid_idxs_no_batch.to(float_dtype)
        grid_points = grid_idxs_f * scales + offsets
        return grid_points
    
    def get_points_grid_idxs(self, points, batch_idx=None):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        int_dtype = self.int_dtype
        float_dtype = self.float_dtype
        device = self.device
        lc = torch.tensor(lower_corner, dtype=float_dtype, device=device)
        uc = torch.tensor(upper_corner, dtype=float_dtype, device=device)
        idx_scale = torch.tensor(grid_shape, 
            dtype=float_dtype, device=device) - 1
        offsets = -lc
        scales = idx_scale / (uc - lc)
        points_idxs_f = (points + offsets) * scales
        points_idxs_i = points_idxs_f.to(dtype=int_dtype)
        points_idxs = torch.empty_like(points_idxs_i)
        for i in range(3):
            points_idxs[...,i] = torch.clamp(
                points_idxs_i[...,i], min=0, max=grid_shape[i]-1)
        final_points_idxs = points_idxs
        if batch_idx is not None:
            final_points_idxs = torch.cat(
                [batch_idx.view(*points.shape[:-1], 1).to(
                    dtype=points_idxs.dtype), points_idxs],
                axis=-1)
        return final_points_idxs
        
    def flatten_idxs(self, idxs, keepdim=False):
        grid_shape = self.grid_shape
        batch_size = self.batch_size

        coord_size = idxs.shape[-1]
        target_shape = None
        if coord_size == 4:
            # with batch
            target_shape = (batch_size,) + grid_shape
        elif coord_size == 3:
            # without batch
            target_shape = grid_shape
        else:
            raise RuntimeError("Invalid shape {}".format(str(idxs.shape)))
        target_stride = tuple(np.cumprod(np.array(target_shape)[::-1])[::-1])[1:] + (1,)
        flat_idxs = (idxs * torch.tensor(target_stride, 
            dtype=idxs.dtype, device=idxs.device)).sum(
                axis=-1, keepdim=keepdim, dtype=idxs.dtype)
        return flat_idxs
    
    def unflatten_idxs(self, flat_idxs, include_batch=True):
        grid_shape = self.grid_shape
        batch_size = self.batch_size
        target_shape = grid_shape
        if include_batch:
            target_shape = (batch_size,) + grid_shape
        target_stride = tuple(np.cumprod(np.array(target_shape)[::-1])[::-1])[1:] + (1,)
        
        source_shape = tuple(flat_idxs.shape)
        if source_shape[-1] == 1:
            source_shape = source_shape[:-1]
            flat_idxs = flat_idxs[...,0]
        source_shape += (4,) if include_batch else (3,)

        idxs = torch.empty(size=source_shape, 
            dtype=flat_idxs.dtype, device=flat_idxs.device)
        mod = flat_idxs
        for i in range(source_shape[-1]):
            idxs[...,i] = mod / target_stride[i]
            mod = mod % target_stride[i]
        return idxs

    def idxs_to_points(self, idxs):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        float_dtype = self.float_dtype
        int_dtype = idxs.dtype
        device = idxs.device

        source_shape = idxs.shape
        point_idxs = None
        if source_shape[-1] == 4:
            # has batch idx
            point_idxs = idxs[...,1:]
        elif source_shape[-1] == 3:
            point_idxs = idxs
        else:
            raise RuntimeError("Invalid shape {}".format(tuple(source_shape)))

        lc = torch.tensor(lower_corner, dtype=float_dtype, device=device)
        uc = torch.tensor(upper_corner, dtype=float_dtype, device=device)
        idx_scale = torch.tensor(grid_shape, 
            dtype=float_dtype, device=device) - 1
        offsets = lc
        scales = (uc - lc) / idx_scale
        
        idxs_points = point_idxs * scales + offsets
        return idxs_points


def ceil_div(a, b):
    return -(-a // b)

class ArraySlicer:
    def __init__(self, shape: tuple, chunks: tuple):
        assert(len(chunks) <= len(shape))
        relevent_shape = shape[:len(chunks)]
        chunk_size = tuple(ceil_div(*x) \
            for x in zip(relevent_shape, chunks))
        
        self.relevent_shape = relevent_shape
        self.chunks = chunks
        self.chunk_size = chunk_size
    
    def __len__(self):
        chunk_size = self.chunk_size
        return int(np.prod(chunk_size))
    
    def __getitem__(self, idx):
        relevent_shape = self.relevent_shape
        chunks = self.chunks
        chunk_size = self.chunk_size
        chunk_stride = np.cumprod((chunk_size[1:] + (1,))[::-1])[::-1]
        chunk_idx = list()
        mod = idx
        for x in chunk_stride:
            chunk_idx.append(mod // x)
            mod = mod % x

        slices = list()
        for i in range(len(chunk_idx)):
            start = chunks[i] * chunk_idx[i]
            end = min(relevent_shape[i], 
                chunks[i] * (chunk_idx[i] + 1))
            slices.append(slice(start, end))
        return slices

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
