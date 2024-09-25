#!/bin/bash

data_path="/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33/demo_gentex_im128_randcams.hdf5"


# playback the original demo
# python robomimic/scripts/add_obj_to_dataset.py \
#     --dataset ${data_path} \
#     --add_obj_num 0 \
#     --write_video \
#     --use_actions \
#     --n 1


# add 10 object distractors to the original scene.
python robomimic/scripts/add_obj_to_dataset.py \
    --dataset ${data_path} \
    --add_obj_num 10 \
    --write_video --write_first_frame --write_gt_mask \
    --use_actions \
    --save_new_data \
    --n 10
