#!/bin/bash

DATASET_PATH="/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33/demo_gentex_im128_randcams_addobj.hdf5"

# Total number of data items to process
TOTAL_DATA=3000

# Number of processes and GPUs
NUM_PROCESSES=8

# Calculate the base size of each data chunk
BASE_CHUNK_SIZE=$((TOTAL_DATA / NUM_PROCESSES))
REMAINDER=$((TOTAL_DATA % NUM_PROCESSES))

# Initialize the start index
START_INDEX=0

# Create logs directory if it doesn't exist
mkdir -p logs
declare -a PIDS

for (( PROCESS_ID=0; PROCESS_ID<NUM_PROCESSES; PROCESS_ID++ ))
do
    # Each process may get an extra data item if there's a remainder
    CHUNK_SIZE=$BASE_CHUNK_SIZE
    if [ $PROCESS_ID -lt $REMAINDER ]; then
        CHUNK_SIZE=$((CHUNK_SIZE + 1))
    fi

    # Calculate the end index
    END_INDEX=$((START_INDEX + CHUNK_SIZE))

    # Assign GPU ID (assuming GPUs are numbered 0 to 7)
    GPU_ID=$PROCESS_ID

    # Display the assignment
    echo "Launching process $PROCESS_ID on GPU $GPU_ID with data indices $START_INDEX to $END_INDEX"

    # Launch the Python process in the background, redirecting output to a log file
    CUDA_VISIBLE_DEVICES=$GPU_ID python /ailab/user/huanghaifeng/work/robocasa_exps/robomimic/robomimic/scripts/dataset_states_to_obs.py \
        --dataset $DATASET_PATH \
        --generative_textures --randomize_cameras --add_objs \
        --interval_left $START_INDEX \
        --interval_right $END_INDEX \
        --global_process_id $PROCESS_ID \
        > logs/process_$PROCESS_ID.log 2>&1 &
    
    PID=$!
    PIDS[$PROCESS_ID]=$PID

    # Update the start index for the next process
    START_INDEX=$((END_INDEX))
done

echo "${PIDS[@]}"

wait

echo "All processes have completed."
