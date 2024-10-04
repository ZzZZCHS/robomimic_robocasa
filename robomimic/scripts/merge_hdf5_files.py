import h5py
import glob
import os
from tqdm import tqdm
from robomimic.scripts.conversion.extract_action_dict import extract_action_dict
from robomimic.scripts.filter_dataset_size import filter_dataset_size
import torch

task_dir = '/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33'
src_file_path = os.path.join(task_dir, 'demo_gentex_im128_randcams_cp_addobj_use_actions_process')
tgt_file_path = os.path.join(task_dir, 'demo_gentex_im128_randcams_cp_addobj_use_actions.hdf5')
tgt_pt_path = os.path.join(task_dir, 'demo_gentex_im128_randcams_cp_addobj_use_actions.pt')

def merge_hdf5_files(output_filename):
    tgt_f = h5py.File(output_filename, 'w')
    for filename in tqdm(sorted(glob.glob(f'{src_file_path}*.hdf5'))):
        # st_idx = int(filename.split('process')[-1].split('.hdf5')[0]) * 375
        src_f = h5py.File(filename, 'r')
        for name in src_f:
            if name not in tgt_f:
                src_f.copy(name, tgt_f, name=name)
            else:
                if name == 'data':
                    for demo_id in src_f[name]:
                        # new_demo_id = f"demo_{int(demo_id.split('_')[1])+st_idx}"
                        src_f.copy(f"{name}/{demo_id}", tgt_f, name=f"{name}/{demo_id}")
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
    
    tgt_f.close()
    
    
    tot_dict = {}
    for filename in tqdm(sorted(glob.glob(f'{src_file_path}*.pt'))):
        cur_dict = torch.load(filename, map_location='cpu')
        tot_dict.update(cur_dict)
    torch.save(tot_dict, tgt_pt_path)
                    
if __name__ == '__main__':
    merge_hdf5_files(tgt_file_path)
