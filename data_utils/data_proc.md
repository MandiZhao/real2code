
## Data Generation & Processing  
1. Download raw data:
- Use the pre-processed version of PartNet-Mobility data from [UMPNet](https://github.com/real-stanford/umpnet). For object categories that UMPNet do not have, download from the [Sapien official website](https://sapien.ucsd.edu/browse) (this website also has a useful web-based interactive visualization for the objects). 
- Note that you may need to manually inspect and de-duplicate the objects: e.g. for "Eyeglasses", we removed 11 repeated objects from the original site, which results in 54 objects in total, from which we selected 5 test objects.


2. Use `blender_render.py` to process and render RGBD images from [PartNet-Mobility](https://sapien.ucsd.edu/browse) data. 
  - If you see error `xcb_connection_has_error() returned true`, try unsetting `DISPLAY` variable (e.g. `export DISPLAY=`).

3. Use `preprocess_data.py` to generate OBB-relative MJCF code data from the raw URDFs for LLM fine-tuning.  
