#!/bin/bash

data_path="/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33/demo_gentex_im128_randcams_cp.hdf5"

TOTAL_DATA=3000
NUM_PROCESSES=48
GPU_NUM=8
PROCESS_PER_GPU=$((NUM_PROCESSES / GPU_NUM))
BASE_CHUNK_SIZE=$((TOTAL_DATA / NUM_PROCESSES))
REMAINDER=$((TOTAL_DATA % NUM_PROCESSES))

START_INDEX=0
START_TIME=$SECONDS

mkdir -p logs
declare -a PIDS

for (( PROCESS_ID=0; PROCESS_ID<NUM_PROCESSES; PROCESS_ID++ ))
do
    CHUNK_SIZE=$BASE_CHUNK_SIZE
    if [ $PROCESS_ID -lt $REMAINDER ]; then
        CHUNK_SIZE=$((CHUNK_SIZE + 1))
    fi

    END_INDEX=$((START_INDEX + CHUNK_SIZE))

    GPU_ID=$((PROCESS_ID / PROCESS_PER_GPU))

    echo "Launching process $PROCESS_ID on GPU $GPU_ID with data indices $START_INDEX to $END_INDEX"

    CUDA_VISIBLE_DEVICES=$GPU_ID python robomimic/scripts/add_obj_to_dataset.py \
        --dataset ${data_path} \
        --write_gt_mask \
        --use_actions \
        --save_new_data \
        --interval_left $START_INDEX \
        --interval_right $END_INDEX \
        --global_process_id $PROCESS_ID \
        > logs/process_$PROCESS_ID.log 2>&1 &

    PID=$!
    PIDS[$PROCESS_ID]=$PID

    # Update the start index for the next process
    START_INDEX=$((END_INDEX))
done

echo "Wait for processes: ${PIDS[@]}"

wait


echo "All processes have completed."

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"  