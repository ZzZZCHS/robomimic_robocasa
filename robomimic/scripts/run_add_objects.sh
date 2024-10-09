#!/bin/bash

# data_path="/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33/demo_gentex_im128_randcams.hdf5"
data_path="/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/mg/2024-05-08-09-20-31/demo_gentex_im128_randcams.hdf5"

# playback原始数据，将成功的数据保存在demo_gentex_im128_randcams_use_actions.mp4中
# 原始数据有3000条，--n参数控制playback多少条数据
# python robomimic/scripts/add_obj_to_dataset.py \
#     --dataset ${data_path} \
#     --add_obj_num 0 \
#     --write_video \
#     --use_actions \
#     --n 1


# 添加新的物体，并将第一帧的图像和GT mask保存在demo_gentex_im128_randcams_addobj_use_actions.mp4中
# --add_obj_num参数控制新加入的物体数量，--n参数控制数据条数
python robomimic/scripts/add_obj_to_dataset.py \
    --dataset ${data_path} \
    --add_obj_num 1 \
    --write_video --write_first_frame --write_gt_mask \
    --use_actions \
    --save_new_data \
    --n 5
