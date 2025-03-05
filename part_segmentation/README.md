## Kinematics-Aware SAM Fine-tuning 
This sub-folder covers fine-tuning SAM for our part-level segmentation task from RGB inputs. 
### Download the Pre-trained SAM checkpoint

First, follow [the official Segment-Anything repo](https://github.com/facebookresearch/segment-anything) for installing the package. Download the default model checkpoint by following the instructions [here])https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints). Then set the `CKPT_PATH` in `part_segmentation/finetune_sam.py` to your own checkpoint path.

### Model Fine-tuning 
You should be able to refer to the path to your downloaded real2code dataset and use its RGB + segmented masks data for SAM fine-tuning. For example, to run SAM fine-tuning (the `--wandb` flag is optional) on 3 GPUs, use command:
```
DATADIR=xxx # your data path
NAME=test # give a run name
python part_segmentation/finetune_sam.py --blender --run_name $NAME  --wandb --use_dp -dp 3 --batch_size 24 --data_dir $DATADIR
```

### Evaluation
Because we ultimately care about segmenting object parts in 3D from 2D multi-view input images, our evaluation procedure involves a 2D->3D->2D prompting scheme to obtain 2D points to evaluate the fine-tuned SAM model with. 

Below are example commands to evaluate the model on one object with unique folder ID. Note these flags would also depend on your specific setup:
- `--data_dir`: path to the test object's input RGBs 
- `--output_dir`: where to store the evaluation output data
- `--sam_model_dir`: path to your fine-tuned SAM model.
```
# For faster debugging:
python part_segmentation/eval_sam.py --obj_folder 103177 --zero_shot_sam --num_3d_points 100 -o 

# For loading from a tuned model checkpoint:
NAME=rebuttal_full_pointsTrue_lr0.001_bs21_ac12_12-01_19-52 # this should be output folder of the previous run
STEPS=24000 # should be the model's ckpt step
python part_segmentation/eval_sam.py --obj_folder 103177 --sam_run_name $FULLRUN --sam_load_steps $STEPS --num_3d_points 100 -o 

