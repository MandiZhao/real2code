import os

def get_render_code(show_joints=False):
    return f"""
model.worldbody.add('camera', name='track_cam', pos=[-3, 2, 1],mode='targetbody', target='root')
model.asset.add('texture', name='groundplane', type="2d", builtin="checker", mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", markrgb=[0.8, 0.8, 0.8], width=300, height=300)
model.asset.add('texture', type="skybox", builtin="gradient", rgb1="0.3 0.5 0.7", rgb2="0 0 0", width=512, height=3000)
model.asset.add('material', name='groundplane', texture='groundplane', texuniform='true', texrepeat=[2,2], reflectance=0.2)
model.worldbody.add('geom', name='floor', pos=[0,0,-0.5], size=[0,0,0.05], type='plane', material='groundplane')

physics = mjcf.Physics.from_mjcf_model(model)
physics.model._model.vis.global_.offwidth = 800
physics.model._model.vis.global_.offheight = 800
physics.model._model.vis.headlight.ambient = [0.8,0.8,0.8]
physics.model._model.vis.headlight.diffuse = [0.6,0.6,0.6]
physics.model._model.vis.headlight.specular = [0,0,0]
physics.model._model.vis.rgba.haze= [0.15,0.25,0.35,1]
physics.model._model.vis.quality.shadowsize = 4096

from dm_control.mujoco.wrapper.core import MjvOption
from dm_control.mujoco.wrapper.mjbindings import enums
scene_option = MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = {show_joints}
# open all the joints 
for i in range(physics.model._model.njnt):
    jrange = physics.model._model.jnt_range[i]
    jnt_val = (jrange[1] + jrange[0]) / 2
    physics.data.qpos[i] = jnt_val
physics.step() 
img = physics.render(camera_id='track_cam', width=800, height=800, scene_option=scene_option)
from PIL import Image
pil_img = Image.fromarray(img) 
"""

# this leaves the first root geom gray
GEOM_COLOR_CODE="""
# change the geom rgbas
import seaborn as sns
colors = sns.color_palette('colorblind', max(10, len(bboxes)))
for i in range(2, physics.model._model.ngeom):
    physics.model._model.geom_rgba[i] = list(colors[i % len(colors)]) + [1.0]
"""
def get_vis_code(img_fname, show_joints=False):
    render_code = get_render_code(show_joints)
    return "\n".join([
        render_code,
        f"pil_img.save('{img_fname}')"]
        )

def visualize_obj_sim(obj_code: str, img_fname="test.png", show_joints=False, save_code_fname=None):
    render_code = get_render_code(show_joints)
    toexec_code = "\n".join([
        obj_code,
        render_code,
        f"pil_img.save('{img_fname}')"
        ])
    if save_code_fname is not None:
        with open(save_code_fname, "w") as f:
            f.write(toexec_code)
    tmp_fname = "tmp.py"
    assert not os.path.exists(tmp_fname), f"tmp file already exists: {tmp_fname}"
    with open(tmp_fname, "w") as f:
        f.write(toexec_code)
    try:
        os.system(f"python {tmp_fname}")
        
    except Exception as e:
        print(f"Failed to execute the code: ")
        print(e)
        return False
    os.remove(tmp_fname)
    assert os.path.exists(img_fname), f"Failed to save the image: {img_fname}"
    return True 