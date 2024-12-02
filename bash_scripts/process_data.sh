# first, use blenderproc to render the images
export CUDA_VISIBLE_DEVICES=0,1,2,3 # specify the GPU to use
LOOPS=5
FRAME=12
OUT_DIR=/local/real/mandi/real2code_dataset_v0
MOBILITY_DATA_DIR=/local/real/mandi/mobility_dataset_v2

OBJ=Eyeglasses 

SPLIT=test
export MB_DATADIR=${MOBILITY_DATA_DIR}/${SPLIT}/${OBJ}
for FOLDER in ${MB_DATADIR}/*; do 
    printf "   processing object#   : $FOLDER"  
    blenderproc run blender_render.py --folder ${FOLDER}  --split $SPLIT --out_dir $OUT_DIR -o --render_bg --num_loops $LOOPS --num_frames $FRAME
done 

SPLIT=train
export MB_DATADIR=${MOBILITY_DATA_DIR}/${SPLIT}/${OBJ}
for FOLDER in ${MB_DATADIR}/*; do 
    printf "   processing object#   : $FOLDER"  
    blenderproc run blender_render.py --folder ${FOLDER}  --split $SPLIT --out_dir $OUT_DIR -o --render_bg --num_loops $LOOPS --num_frames $FRAME
done 

# test Eyeglasses 
# 101844
# 101860 
# 101303
# 103177
# 101845
# for TEST in 101844 101860 101303 103177 101845; do 
#     mv  /local/real/mandi/mobility_dataset_v2/train/Eyeglasses/${TEST} /local/real/mandi/mobility_dataset_v2/test/Eyeglasses/
# done

# # remove these below duplicates!
# for RM in 101300 101335 101839 101840 101842 101843 101848 102612 102568 102569 102570; do 
#     rm -rf /local/real/mandi/mobility_dataset_v2/train/Eyeglasses/${RM}
# done

 

OBJ=Scissors 
# # test Scissors 
# mkdir  /local/real/mandi/mobility_dataset_v2/test/${OBJ}
# for TEST in 10907 11029 10561 11111; do 
#     mv  /local/real/mandi/mobility_dataset_v2/train/${OBJ}/${TEST} /local/real/mandi/mobility_dataset_v2/test/${OBJ}/
# done

OBJ=Scissors  
LOOPS=5
FRAME=12
OUT_DIR=/local/real/mandi/real2code_dataset_v0
MOBILITY_DATA_DIR=/local/real/mandi/mobility_dataset_v2

SPLIT=test
export MB_DATADIR=${MOBILITY_DATA_DIR}/${SPLIT}/${OBJ}
for FOLDER in ${MB_DATADIR}/*; do 
    printf "   processing object#   : $FOLDER"  
    blenderproc run blender_render.py --folder ${FOLDER}  --split $SPLIT --out_dir $OUT_DIR -o --render_bg --num_loops $LOOPS --num_frames $FRAME
done 

SPLIT=train
export MB_DATADIR=${MOBILITY_DATA_DIR}/${SPLIT}/${OBJ}
for FOLDER in ${MB_DATADIR}/*; do 
    printf "   processing object#   : $FOLDER"  
    blenderproc run blender_render.py --folder ${FOLDER}  --split $SPLIT --out_dir $OUT_DIR -o --render_bg --num_loops $LOOPS --num_frames $FRAME
done 