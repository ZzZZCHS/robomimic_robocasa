import h5py
import glob
import os
from tqdm import tqdm
from robomimic.scripts.conversion.extract_action_dict import extract_action_dict
from robomimic.scripts.filter_dataset_size import filter_dataset_size
import torch
import argparse
import traceback

# task_dir = '/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33'
# src_file_path = os.path.join(task_dir, 'demo_gentex_im128_randcams_cp_addobj_use_actions_process*.hdf5')
# tgt_file_path = os.path.join(task_dir, 'demo_gentex_im128_randcams_cp_addobj_use_actions.hdf5')

def merge_hdf5_files(args):
    tgt_file_path = os.path.join(args.task_dir, args.tgt_filename)
    src_file_path = os.path.join(args.task_dir, args.src_filename)
    if os.path.exists(tgt_file_path):
        os.remove(tgt_file_path)
    tgt_f = h5py.File(tgt_file_path, 'w')
    for filename in tqdm(sorted(glob.glob(src_file_path))):
        try:
            src_f = h5py.File(filename, 'r')
            for name in src_f:
                if name not in tgt_f:
                    src_f.copy(name, tgt_f, name=name)
                else:
                    if name == 'data':
                        for demo_id in src_f[name]:
                            # new_demo_id = f"demo_{int(demo_id.split('_')[1])+st_idx}"
                            src_f.copy(f"{name}/{demo_id}", tgt_f, name=f"{name}/{demo_id}")
            os.remove(filename)
        except Exception as e:
            print(filename)
            print(e)
            print(traceback.format_exc())
            # breakpoint()
            if src_f:
                src_f.close()
    total_len = 0
    for demo_id in tgt_f['data'].keys():
        total_len += tgt_f[f"data/{demo_id}/actions"].shape[0]
    tgt_f['data'].attrs['total'] = total_len
    
    extract_action_dict(dataset=tgt_file_path)
    for num_demos in [10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]:
        filter_dataset_size(
            tgt_file_path,
            num_demos=num_demos,
        )
    
    print("total demo:", len(tgt_f['data']))
    
    tgt_f.close()
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_dir",
        type=str,
        help="directory of the task"
    )
    parser.add_argument(
        "--src_filename",
        type=str,
        help="the source file name"
    )
    parser.add_argument(
        "--tgt_filename",
        type=str,
        help="the target file name"
    )
    args = parser.parse_args()
    merge_hdf5_files(args)
