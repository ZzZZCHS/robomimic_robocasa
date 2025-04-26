#!/bin/bash

data_path=$1

python robomimic/scripts/playback_demos.py \
    --dataset ${data_path} \
    --n 2 \
    --camera_height 512 \
    --camera_width 512 \
    --write_gt_mask \
    --write_video


# python robomimic/scripts/playback_demos.py \
#     --dataset ${data_path} \
#     --save_new_data --save_obs \
#     --n 30 \
#     --use_actions \
#     --camera_height 256 \
#     --camera_width 256 \
#     --write_gt_mask \
#     --write_video 
#     # --write_first_frame \
#     # --skip_replay \
#     # --unique_attr class \
#     # --add_obj_num 6