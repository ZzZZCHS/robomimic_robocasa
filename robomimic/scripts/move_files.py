import os
import h5py
import shutil
import argparse

task_dirs = [
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-07-12-04-33-29/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/mg/2024-05-04-22-13-21_and_2024-05-07-07-41-17/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/mg/2024-05-04-22-14-26_and_2024-05-07-07-41-42/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-05-04-22-14-20/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/mg/2024-05-04-22-14-40/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/mg/2024-05-04-22-37-39/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/mg/2024-05-04-22-34-56/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/mg/2024-05-04-22-35-53/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/mg/2024-05-04-22-22-42_and_2024-05-08-06-02-36/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer/mg/2024-05-04-22-38-42/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/mg/2024-05-09-09-32-19/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/mg/2024-05-04-22-17-46/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/mg/2024-05-04-22-17-26/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout/mg/2024-05-09-09-31-12/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/mg/2024-05-08-09-20-31/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOffStove/mg/2024-05-08-09-20-45/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/mg/2024-05-04-22-40-00/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/mg/2024-05-04-22-39-23/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/mg/2024-05-04-22-22-13_and_2024-05-08-05-52-13/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/mg/2024-05-04-22-21-50/",
    "/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/mg/2024-05-04-22-21-32/"
]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tgt_dir",
    type=str,
    default="/ailab/group/pjlab-smartbot/chenxinyi/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0407"
)

parser.add_argument(
    "--file_name",
    type=str,
    default="demo_gentex_im128_randcams_addobj_use_actions_hhf.hdf5"
)

args = parser.parse_args()

tgt_dir = args.tgt_dir
file_name = args.file_name

os.makedirs(tgt_dir, exist_ok=-True)

tot = 0
num = 0
for task_dir in task_dirs:
    task_name = task_dir.split('/')[-4]
    data_path = os.path.join(task_dir, file_name)
    if os.path.exists(data_path):
        f = h5py.File(data_path, 'r')
        try:
            print(task_name, len(f['data']))
            tot += len(f['data'])
            num += 1
            shutil.move(data_path, os.path.join(tgt_dir, f"{task_name}.hdf5"))
        except:
            continue


print(tot, num, tot / (num + 0.0001))
