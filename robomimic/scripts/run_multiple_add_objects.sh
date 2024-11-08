#!/bin/bash

task_dirs=(
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-07-12-04-33-29/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/mg/2024-05-04-22-13-21_and_2024-05-07-07-41-17/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/mg/2024-05-04-22-14-26_and_2024-05-07-07-41-42/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-05-04-22-14-20/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/mg/2024-05-04-22-14-40/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/mg/2024-05-04-22-37-39/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/mg/2024-05-04-22-34-56/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/mg/2024-05-04-22-35-53/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/mg/2024-05-04-22-22-42_and_2024-05-08-06-02-36/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer/mg/2024-05-04-22-38-42/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/mg/2024-05-09-09-32-19/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/mg/2024-05-04-22-17-46/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/mg/2024-05-04-22-17-26/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout/mg/2024-05-09-09-31-12/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/mg/2024-05-08-09-20-31/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOffStove/mg/2024-05-08-09-20-45/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/mg/2024-05-04-22-40-00/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/mg/2024-05-04-22-39-23/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/mg/2024-05-04-22-22-13_and_2024-05-08-05-52-13/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/mg/2024-05-04-22-21-50/"
    "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/mg/2024-05-04-22-21-32/"
)


ST_I=$1
ED_I=$2
GPU_NUM=$3

TOTAL_DATA=3000
NUM_PROCESSES=$((GPU_NUM * 5))
PROCESS_PER_GPU=$((NUM_PROCESSES / GPU_NUM))
BASE_CHUNK_SIZE=$((TOTAL_DATA / NUM_PROCESSES))
REMAINDER=$((TOTAL_DATA % NUM_PROCESSES))

mkdir -p logs
declare -a PIDS


for ((i = ST_I; i < ED_I; ++i))
do
    data_path=${task_dirs[$i]}/demo_gentex_im128_randcams.hdf5
    START_INDEX=0
    START_TIME=$SECONDS

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
            --save_new_data --save_obs \
            --interval_left $START_INDEX \
            --interval_right $END_INDEX \
            --global_process_id $PROCESS_ID \
            --use_actions \
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

    python robomimic/scripts/merge_hdf5_files.py \
        --task_dir ${task_dirs[$i]} \
        --src_filename "demo_gentex_im128_randcams_addobj_use_actions_process*_hhf.hdf5" \
        --tgt_filename "demo_gentex_im128_randcams_addobj_use_actions_hhf.hdf5"
    
    python robomimic/scripts/move_files.py
done