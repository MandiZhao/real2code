## Shape Completion 
See `shape_complete/`, we use Blender-rendered RGBD images to generate partially-observable point clouds inputs; `kaolin` for processing ground-truth mesh to generate occupancy label grids. 

### Data Generation

The `datagen.py` script generates training data for our shape completion model: pairs of partially-observed point clouds and each point cloud's corresponding ground-truth part mesh. Note that this is better stored as a separate shape dataset. 