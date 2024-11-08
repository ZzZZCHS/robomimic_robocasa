import h5py
from tqdm import tqdm

with h5py.File('/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_1107/PnPCounterToCab_depth.hdf5', 'a') as f:
    for ep in tqdm(f['data']):
        for obs_key in f[f'data/{ep}/obs']:
            if 'depth' in obs_key:
                # print(ep, obs_key)
                del f[f'data/{ep}/obs/{obs_key}']


