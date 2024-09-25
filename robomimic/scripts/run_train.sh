#!/bin/bash

#SBATCH --gres=gpu:1 -N 1 -n 6 -p smartbot

module load anaconda/2024.02
source activate robocasa

# export MASTER_PORT=$((12000 + $RANDOM % 20000))
# export OMP_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"
# NNODE=1
# NUM_GPUS=2
# MASTER_NODE='localhost'

config_file=/ailab/user/huanghaifeng/work/robocasa_exps/tmp/autogen_configs/ril/bc/robocasa/im/08-16-train_mg/08-16-24-14-55-06/json/seed_123_ds_mg-3000.json

# torchrun --nnodes=${NNODE} --nproc_per_node=${NUM_GPUS} \
#     /root/huanghaifeng/robocasa_exps/robomimic/robomimic/scripts/train.py \
#     --config ${config_file}

python /ailab/user/huanghaifeng/work/robocasa_exps/robomimic/robomimic/scripts/train.py \
    --config ${config_file}