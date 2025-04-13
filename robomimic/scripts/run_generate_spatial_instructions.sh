#!/bin/bash

# Define the directory containing the HDF5 files
folder_path="/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_data"

# Loop through each .hdf5 file in the specified folder
for hdf5_path in "$folder_path"/*.hdf5; do
  # Run the Python script with the current file path as an argument
  python robomimic/scripts/generate_spatial_instructions.py --file_path "$hdf5_path"
done
