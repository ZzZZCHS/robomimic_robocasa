import h5py
from tqdm import tqdm
import os


src_dir = '/ailab/user/huanghaifeng/data/robocasa/raw_data'
tgt_dir = '/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/raw_data'


def copy_group(src_group, tgt_group):
    for attr_name, attr_value in src_group.attrs.items():
        tgt_group.attrs[attr_name] = attr_value
    
    for name, item in src_group.items():
        if name.endswith("_depth"):
            continue
        if isinstance(item, h5py.Group):
            new_group = tgt_group.create_group(name)
            copy_group(item, new_group)
        elif isinstance(item, h5py.Dataset):
            new_dataset = tgt_group.create_dataset(name, data=item[...])
            for attr_name, attr_value in item.attrs.items():
                new_dataset.attrs[attr_name] = attr_value


for filename in tqdm(os.listdir(src_dir)):
    print(filename)
    filepath = os.path.join(src_dir, filename)
    src_f = h5py.File(filepath, 'r')
    tgt_filepath = os.path.join(tgt_dir, filename)
    tgt_f = h5py.File(tgt_filepath, 'w')
    copy_group(src_f, tgt_f)
    

# with h5py.File('/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_1107/PnPCounterToCab_depth.hdf5', 'a') as f:
#     for ep in tqdm(f['data']):
#         for obs_key in f[f'data/{ep}/obs']:
#             if 'depth' in obs_key:
#                 # print(ep, obs_key)
#                 del f[f'data/{ep}/obs/{obs_key}']


