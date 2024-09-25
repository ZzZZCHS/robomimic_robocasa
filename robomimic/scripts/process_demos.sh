#!/bin/bash

#SBATCH --gres=gpu:1 -N 1 -n 6 -p smartbot

# module load anaconda/2024.02
# source activate robocasa

# Define the data root
DATA_ROOT="/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/datasets/v0.1/single_stage"

# Find all the demo.hdf5 files under the root directory
find "$DATA_ROOT" -type f -name "demo.hdf5" | while read -r file; do
    # Get the parent directory of the current file
    PARENT_DIR=$(dirname "$file")
    # Define the target file name
    TARGET_FILE="$PARENT_DIR/demo_gentex_im256_randcams.hdf5"

    # Check if the target file does not exist
    if [ ! -f "$TARGET_FILE" ]; then
        # Run the specified command with the necessary environment variables
        echo $TARGET_FILE
        MUJOCO_GL=egl OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python robomimic/scripts/dataset_states_to_obs.py --camera_width 256 --camera_height 256 --generative_textures --randomize_cameras --dataset "$file"
        exit 0
    fi
done
