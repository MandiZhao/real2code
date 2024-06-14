import torch
from torch import nn
import torch.nn.functional as F
from shape_complete.modules.pointnet import PointNetSetAbstraction, PointNetSetAbstractionMsg, PointNetFeaturePropagation
from shape_complete.modules.unet3d import UNet3D, Abstract3DUNet, DoubleConv
from shape_complete.modules.mlp import MLP
from shape_complete.modules.gridding import VirtualGrid
import torch_scatter 
from einops import rearrange
import math
# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html 

class PointNetppEncoder(nn.Module): 
    """ PointNet++ Encoder, returns global feature vector and local features"""
    def __init__(self, use_color=False):
        super(PointNetppEncoder, self).__init__()
        in_channel = 3 if use_color else 0
        self.use_color = use_color
        # self.num_points = num_points
        
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True) 
        
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134+in_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        """
        xyz: shape (B, 6, N) or (B, 3, N)
        """
        B, C, N = xyz.shape
        # assert self.num_points == N, f"Expected {self.num_points} points, got {N}"
        if self.use_color:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz 
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        # x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1) 
        return x # shape (B, d, N)

class VolumeFeatureAggregator(nn.Module):
    def __init__(self,
            nn_channels=[131,1024,128],
            batch_norm=True,
            lower_corner=(-0.5,-0.5,-0.5), 
            upper_corner=(0.5, 0.5, 0.5), 
            grid_shape=(32, 32, 32),
            reduce_method='max',
            include_point_feature=True,
            # include_confidence_feature=False
            ):
        super().__init__()
        # self.save_hyperparameters()
        self.local_nn = MLP(nn_channels, batch_norm=batch_norm)
        self.lower_corner = tuple(lower_corner)
        self.upper_corner = tuple(upper_corner)
        self.grid_shape = tuple(grid_shape)
        self.reduce_method = reduce_method
        self.include_point_feature = include_point_feature
        # self.include_confidence_feature = include_confidence_feature
    
    def forward(self, xyz, point_features): 
        local_nn = self.local_nn
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        reduce_method = self.reduce_method
        batch_size = point_features.shape[0]
        device = point_features.device
        float_dtype = point_features.dtype
        int_dtype = torch.int64

        vg = VirtualGrid(
            lower_corner=lower_corner, 
            upper_corner=upper_corner, 
            grid_shape=grid_shape, 
            batch_size=batch_size,
            device=device,
            int_dtype=int_dtype,
            float_dtype=float_dtype
            )
        
        # get aggregation target index
        reshaped_xyz = rearrange(xyz, 'b c n -> (b n) c')   
        points_grid_idxs = vg.get_points_grid_idxs(points=reshaped_xyz)
        flat_idxs = vg.flatten_idxs(points_grid_idxs, keepdim=False)
        flat_idxs = rearrange(flat_idxs, '(b n) -> b n', b=batch_size) # (B, N)
        
        # get features
        features = torch.cat([xyz, point_features], 1) # (B, 3+d, N) 
        # per-point transform
        features = rearrange(features, 'b d n -> b n d', b=batch_size) # (B, N, d)
        if local_nn is not None: 
            features = local_nn(features)
        
        # scatter
        num_grids = math.prod(grid_shape) # doesn't consider batch size
        volume_feature_flat = torch_scatter.scatter(
            src=features, index=flat_idxs, dim=1, 
            dim_size=num_grids, reduce=reduce_method)
        # shape B, num_grids, d
        x, y, z = grid_shape
        volume_feature = rearrange(volume_feature_flat, 'b (x y z) d -> b d x y z', b=batch_size, x=x, y=y, z=z)
        return volume_feature

class OccupancyDecoder(nn.Module):
    def __init__(self, nn_channels=(128, 512, 512, 1), batch_norm=True):
        super(OccupancyDecoder, self).__init__()
        self.mlp = MLP(nn_channels, batch_norm=batch_norm)

    def forward(self, dense_features, query_points):
        """ 
        For each query point, trilinear interpolate the dense features then decode the occupancy 1/0 value
        - dense_features: (B,C,D,H,W)
        query_points: (B,N,3)
        """
        assert query_points.shape[-1] == 3, "query_points should have last dim=3"
        # assume query points are already normalized to -1 to 1
        query_points_normalized = 2.0 * query_points - 1.0
        
        # shape (B,C,M,1,1)
        query_points_reshaped = rearrange(query_points_normalized, 'b num_points c -> b num_points 1 1 c')
        
        sampled_features = F.grid_sample(
            input=dense_features, 
            grid=query_points_reshaped,
            mode='bilinear', padding_mode='border',
            align_corners=True)
        # shape (B,M,C)
        sampled_features = rearrange(sampled_features, 'b c num_points 1 1 -> b num_points c')        
        # shape (N,M,C)
        out_features = self.mlp(sampled_features) # B, N, out_dim=1 
        return out_features

class ShapeCompletionModel(nn.Module):
    def __init__(
            self, 
            agg_args=dict(),
            unet_args=dict(in_channels=128, out_channels=128),
            decoder_args=dict(),    
        ):
        super(ShapeCompletionModel, self).__init__()
        self.pointnet = PointNetppEncoder()
        self.volume_aggregator = VolumeFeatureAggregator(**agg_args)
        self.unet = UNet3D(**unet_args)
        self.decoder = OccupancyDecoder(**decoder_args)
        

    def forward(self, xyz, query_points):
        assert len(xyz.shape) == 3, "Expected input shape (B, 3, N)"
        if xyz.shape[1] != 3 and xyz.shape[1] != 6:
            xyz = rearrange(xyz, 'b n c -> b c n')
        if query_points.shape[-1] != 3:
            query_points = rearrange(query_points, 'b c n -> b n c')
        pointnet_feats = self.pointnet(xyz)
        agg_feats = self.volume_aggregator(xyz, pointnet_feats)
        unet_feats = self.unet(agg_feats)
        decoder_out = self.decoder(unet_feats, query_points)
        decoder_out = rearrange(decoder_out, 'b n 1 -> b n')
        return decoder_out

    def compute_loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":
    # pointnet = PointNetppEncoder(num_points=6000).to("cuda")
    # volume_aggregator = VolumeFeatureAggregator().to("cuda")
    # unet = UNet3D(in_channels=128, out_channels=128).to("cuda")
    # decoder = OccupancyDecoder().to("cuda")
    # uniform random points from -1 to 1
    # xyz = torch.rand(1, 3, 6000).to("cuda") * 2 - 1
    # pointnet_feats = pointnet(xyz)
    # concat_feats = torch.cat([xyz, pointnet_feats], 1) # (B, 3+d, N)
    # agg_feats = volume_aggregator(xyz, pointnet_feats)
    # unet_feats = unet(agg_feats)
    # query_points = torch.rand(1, 100, 3).to("cuda") * 2 - 1
    # decoder_out = decoder(unet_feats, query_points)
    
    model = ShapeCompletionModel().to("cuda")
    input_points = torch.rand(4, 1200, 3).to("cuda")
    query_points = torch.rand(4, 10, 3).to("cuda")
    pred = model(input_points, query_points)
    labels = torch.randint(0, 2, (4, 10)).float().to("cuda")
    breakpoint()


