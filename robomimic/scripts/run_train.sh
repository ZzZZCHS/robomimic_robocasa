#!/bin/bash

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"
NNODE=1
NUM_GPUS=2
MASTER_NODE='localhost'

config_file=/root/huanghaifeng/tmp/autogen_configs/ril/bc/robocasa/im/07-26-tmp/07-26-24-14-30-55/json/seed_123_ds_human-50.json

torchrun --nnodes=${NNODE} --nproc_per_node=${NUM_GPUS} \
    /root/huanghaifeng/robocasa_exps/robomimic/robomimic/scripts/train.py \
    --config ${config_file}