import h5py
import glob
from tqdm import tqdm
from robomimic.scripts.conversion.extract_action_dict import extract_action_dict
from robomimic.scripts.filter_dataset_size import filter_dataset_size

src_file_path = '/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33/demo_gentex_im128_randcams_addobj_im128_randcams_gentex_process'
tgt_file_path = '/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33/demo_addobj_gentex_im128_randcams.hdf5'

def merge_hdf5_files(output_filename):
    with h5py.File(output_filename, 'w') as tgt_f:
        for filename in tqdm(sorted(glob.glob(f'{src_file_path}*.hdf5'))):
            st_idx = int(filename.split('process')[-1].split('.hdf5')[0]) * 375
            with h5py.File(filename, 'r') as src_f:
                for name in src_f:
                    if name not in tgt_f:
                        src_f.copy(name, tgt_f, name=name)
                    else:
                        if name == 'data':
                            for demo_id in src_f[name]:
                                new_demo_id = f"demo_{int(demo_id.split('_')[1])+st_idx}"
                                src_f.copy(f"{name}/{demo_id}", tgt_f, name=f"{name}/{new_demo_id}")
                        
                    
if __name__ == '__main__':
    merge_hdf5_files(tgt_file_path)
    
    extract_action_dict(dataset=tgt_file_path)
    for num_demos in [10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]:
        filter_dataset_size(
            tgt_file_path,
            num_demos=num_demos,
        )
