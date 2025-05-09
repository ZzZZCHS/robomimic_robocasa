#!/bin/bash

# pnp
data_path=$1
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-07-12-04-33-29/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/mg/2024-05-04-22-13-21_and_2024-05-07-07-41-17/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/mg/2024-05-04-22-14-26_and_2024-05-07-07-41-42/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-05-04-22-14-20/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/mg/2024-05-04-22-14-40/demo_gentex_im128_randcams.hdf5"


# doors
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/mg/2024-05-04-22-37-39/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/mg/2024-05-04-22-34-56/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/mg/2024-05-04-22-35-53/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/mg/2024-05-04-22-22-42_and_2024-05-08-06-02-36/demo_gentex_im128_randcams.hdf5"


# drawer
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer/mg/2024-05-04-22-38-42/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/mg/2024-05-09-09-32-19/demo_gentex_im128_randcams.hdf5"

# sink
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/mg/2024-05-04-22-17-46/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/mg/2024-05-04-22-17-26/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout/mg/2024-05-09-09-31-12/demo_gentex_im128_randcams.hdf5"

# stove
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/mg/2024-05-08-09-20-31/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOffStove/mg/2024-05-08-09-20-45/demo_gentex_im128_randcams.hdf5"

# microwave
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/mg/2024-05-04-22-40-00/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/mg/2024-05-04-22-39-23/demo_gentex_im128_randcams.hdf5"

# coffee
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/mg/2024-05-04-22-22-13_and_2024-05-08-05-52-13/demo_gentex_im128_randcams.hdf5"
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/mg/2024-05-04-22-21-50/demo_gentex_im128_randcams.hdf5"

# button
# data_path="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/mg/2024-05-04-22-21-32/demo_gentex_im128_randcams.hdf5"

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

# python robomimic/scripts/add_obj_to_dataset.py \
#     --dataset ${data_path} \
#     --add_obj_num 1 \
#     --write_video --write_first_frame --write_gt_mask \
#     --use_actions \
#     --save_new_data \
#     --n 1

python robomimic/scripts/playback_demos.py \
    --dataset ${data_path} \
    --n 2 \
    --camera_height 512 \
    --camera_width 512 \
    --write_gt_mask \
    --write_video
    # --save_new_data --save_obs \
    
    # --write_first_frame \
    # --skip_replay \
    # --unique_attr class \
