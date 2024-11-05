import h5py
import glob
import json
import torch
from tqdm import tqdm
import os
from collections import defaultdict

data_dir = "/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_1105"

train_env_infos = defaultdict(list)
for filepath in tqdm(glob.glob(os.path.join(data_dir, "*.hdf5"))):
    with h5py.File(filepath, 'r') as f:
        env_args = json.loads(f['data'].attrs['env_args'])
        env_name = env_args['env_name']
        for ep_i in f['data']:
            ep_data_grp = f['data'][ep_i]
            train_env_infos[env_name].append({
                "model": ep_data_grp.attrs['model_file'],
                "states": ep_data_grp['states'][0],
                "ep_meta": ep_data_grp.attrs['ep_meta']
            })

torch.save(train_env_infos, os.path.join(data_dir, "train_env_infos.pt"))