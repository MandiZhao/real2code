# export CUDA_VISIBLE_DEVICES=4
# python shape_complete/datagen.py --data_dir /store/real/mandi/real2code_dataset_v0/ --obj_type Scissors --out_dir /store/real/mandi/real2code_shape_dataset_v0/ --vis_dir /store/real/mandi/real2code_shape_dataset_v0/vis --split train

export CUDA_VISIBLE_DEVICES=4
OBJ=Eyeglasses
for SPLIT in train test; do
    python shape_complete/datagen.py --data_dir /store/real/mandi/real2code_dataset_v0/ --obj_type $OBJ --out_dir /store/real/mandi/real2code_shape_dataset_v0/ --vis_dir /store/real/mandi/real2code_shape_dataset_v0/vis --split $SPLIT
done