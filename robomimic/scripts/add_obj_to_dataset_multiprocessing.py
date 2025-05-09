"""
A script to visualize dataset trajectories by loading the simulation states
one by one or loading the first state and playing actions back open-loop.
The script can generate videos as well, by rendering simulation frames
during playback. The videos can also be generated using the image observations
in the dataset (this is useful for real-robot datasets) by using the
--use-obs argument.

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, use the subset of trajectories
        in the file that correspond to this filter key

    n (int): if provided, stop after n trajectories are processed

    use-obs (bool): if flag is provided, visualize trajectories with dataset 
        image observations instead of simulator

    use-actions (bool): if flag is provided, use open-loop action playback 
        instead of loading sim states

    render (bool): if flag is provided, use on-screen rendering during playback
    
    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    render_image_names (str or [str]): camera name(s) / image observation(s) to 
        use for rendering on-screen or to video

    first (bool): if flag is provided, use first frame of each episode for playback
        instead of the entire episode. Useful for visualizing task initializations.

Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_image_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-actions --render_image_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

import os
import json
import h5py
import argparse
import imageio
import numpy as np
import torch
import random
import glob
import shutil
import cv2
import multiprocessing

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType
# from robomimic.utils.ground_utils import GroundUtils

from robosuite.demos.demo_segmentation import segmentation_to_rgb

import xml.etree.ElementTree as ET
from robocasa.models.objects.objects import MJCFObject

import traceback
import copy
import time

# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}

# grounding_model = GroundUtils(device="cuda:1")


# def save_new_data(args, tgt_f, )


def playback_trajectory_with_env(
    env, 
    initial_state, 
    states, 
    args,
    actions=None, 
    render=False, 
    video_writer=None, 
    video_skip=5, 
    camera_names=None,
    first=False,
    ep="",
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert isinstance(env, EnvBase)

    video_count = 0
    
    zero_actions = np.zeros(actions.shape[1])

    # load the initial state
    ob_dict = env.reset_to(initial_state)
    
    save_images = []
    save_masks = []
    
    if args.write_gt_mask:
        seg_sensors = {}
        for cam_name in camera_names:
            seg_sensor, seg_name = env.env._create_segmentation_sensor(cam_name, 512, 512, "instance", "segmentation")
            seg_sensors[cam_name] = seg_sensor
        name2id = {inst: i for i, inst in enumerate(list(env.env.model.instances_to_ids.keys()))}

    traj_len = states.shape[0]
    action_playback = (actions is not None)
    if action_playback:
        assert states.shape[0] == actions.shape[0]
    
    zero_actions = np.zeros(actions.shape[1])
    for i in range(50):
        env.step(zero_actions)

    success = False
    outputs = dict(actions_abs=[], rewards=[], dones=[], states=[])

    frames = []
    
    for i in range(traj_len):
        state = env.get_state()["states"]
        if action_playback:
            obs, r, _, info = env.step(actions[i])
        else:
            env.reset_to({"states" : states[i]})

        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])
            
        if action_playback:
            r = env.get_reward()
        done = success = env.is_success()["task"]
        done = int(done)
        action_abs = env.base_env.convert_rel_to_abs_action(actions[i])
        new_distr_names = [f"new_distr_{i}" for i in range(1, env.env.add_object_num+1)]
        
        # video render
        if not args.write_first_frame and video_count % video_skip == 0 or \
            args.write_first_frame and video_count == 0:
            video_img = []
            for cam_name in camera_names:
                video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
            if len(save_images) == 0:
                save_images.extend(video_img)
            video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
            if args.write_gt_mask:
                seg_img = []
                flag = True
                for cam_name in camera_names:
                    tmp_seg = seg_sensors[cam_name]().squeeze(-1)[::-1]
                    seg_rgb = segmentation_to_rgb(tmp_seg, random_colors=False)
                    ct = 0
                    for distr_name in new_distr_names:
                        if len(tmp_seg == name2id[distr_name] + 1) > 10:
                            ct += 1
                    if ct < 0.8 * len(new_distr_names):
                        flag = False
                        return None, False
                    seg_rgb[tmp_seg != name2id['obj'] + 1] = 0
                    seg_rgb[tmp_seg == name2id['obj'] + 1, 0] = 255
                    seg_rgb[tmp_seg == name2id['obj'] + 1, 1:] = 1
                    seg_img.append(seg_rgb)
                if len(save_masks) == 0:
                    save_masks.extend(seg_img)
                seg_img = np.concatenate(seg_img, axis=1)
                seg_video_img = video_img
                seg_video_img[seg_img != 0] //= 2
                seg_img //= 2
                seg_video_img = seg_video_img + seg_img
                # breakpoint()
                video_img = np.concatenate([video_img, seg_video_img], axis=0)
            frame = video_img.copy()
            text1 = env._ep_lang_str
            position1 = (10, 50)
            color = (255, 0, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            font_scale = 0.6
            cv2.putText(frame, text1, position1, font, font_scale, color, thickness)
            text2 = f"{ep}  Unique Attribute: {env.env.unique_attr}"
            position2 = (10, 100)
            cv2.putText(frame, text2, position2, font, font_scale, color, thickness)
            frames.append(frame)
            video_count += 1

        outputs['rewards'].append(r)
        outputs['dones'].append(done)
        outputs['actions_abs'].append(action_abs)
        outputs['states'].append(state)
    
    # if write_video and success:
    #     for frame in frames:
    #         video_writer.append_data(frame)
    
    outputs.update({
        "images": save_images,
        "masks": save_masks,
        "ep": ep,
        "lang": env._ep_lang_str,
        "class": ' '.join(env.env.target_obj_name.split('_')[:-1]),
        "unique_attr": env.env.unique_attr,
        "target_phrase": env.target_phrase,
        "new_model": env.env.sim.model.get_xml(),
        "new_ep_meta": env.env.get_ep_meta(),
        "frames": frames
    })
    print("Success:", success)
    return outputs, success


def retrieve_new_index(process_num, current_work_array, work_queue, lock):
    with lock:
        if work_queue.empty():
            return -1
        try:
            tmp = work_queue.get(False)
            current_work_array[process_num] = tmp
            return tmp
        except queue.Empty:
            return -1


def playback_demos_process(process_num, current_work_array, work_queue, lock, args, num_finished, mul_queue):
    dummy_spec = dict(
        obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=[],
            ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    # env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False # absolute action space
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=args.write_video)

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)
    
    f = h5py.File(args.dataset, "r")
    
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    elif "data" in f.keys():
        demos = list(f["data"].keys())
    else:
        demos = None

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    
    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]
    
    ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
    while (not work_queue.empty()) and (ind != -1):
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)
        
        ori_model = copy.copy(initial_state['model'])
        
        # supply actions if using open-loop action playback
        actions = None
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]
            # actions = f["data/{}/actions_abs".format(ep)][()] # absolute actions
        
        env.env.add_object_num = args.add_obj_num
        success = False
        for try_idx in range(3):
            try:
                initial_state['model'] = copy.copy(ori_model)
                outputs, success = playback_trajectory_with_env(
                    env=env, 
                    initial_state=initial_state, 
                    states=states, 
                    args=args,
                    actions=actions, 
                    render=args.render, 
                    video_writer=video_writer, 
                    video_skip=args.video_skip,
                    camera_names=args.render_image_names,
                    first=args.first,
                    ep=ep
                )
                if success:
                    break
            except KeyboardInterrupt:
                print('Control C pressed. Closing files and ending.')
            except Exception as e:
                # print(traceback.format_exc())
                pass
        mul_queue.put([ep, outputs, success])
        ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
    
    num_finished.value += 1


def write_new_data_process(args, tgt_dataset_path, total_samples, total_run, num_process, mul_queue, demo_keys):
    video_writer = None
    if args.write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)
    tgt_f = h5py.File(tgt_dataset_path, "r+")
    start_time = time.time()
    num_processed = 0
    success_num = 0
    
    try:
        while((total_run.value < num_process) or not mul_queue.empty()):
            if mul_queue.empty():
                continue
            num_processed = num_processed + 1
            ep, outputs, success = mul_queue.get()
            demo_keys.remove(ep)
            
            if not success or outputs is None:
                if f"data/{ep}" in tgt_f:
                    del tgt_f[f"data/{ep}"]
                continue

            success_num += 1
            
            save_data_info[ep] = {
                "lang": outputs["lang"],
                "class": outputs["class"],
                "unique_attr": outputs["unique_attr"],
                "target_phrase": outputs["target_phrase"],
                "images": outputs["images"],
                "masks": outputs["masks"]
            }
            
            if args.write_video:
                for frame in outputs["frames"]:
                    video_writer.append_data(frame)
            
            new_model = outputs["new_model"]
            new_ep_meta = outputs["new_ep_meta"]
            new_ep_meta["lang"] = outputs["lang"]
            new_ep_meta = json.dumps(new_ep_meta, indent=4)
            
            print('object number:', len(env.env.object_cfgs))
            
            tgt_f["data/{}".format(ep)].attrs["model_file"] = new_model
            tgt_f["data/{}".format(ep)].attrs["ep_meta"] = new_ep_meta
            for item_name in ['states', 'rewards', 'dones', 'actions_abs']:
                item_path = f"data/{ep}/{item_name}"
                if item_path in tgt_f:
                    del tgt_f[item_path]
                tgt_f.create_dataset(item_path, data=outputs[item_name])
            
            total_samples.value += len(outputs["action_abs"])

            print("ep {}: wrote {} transitions to group {} at process {} with {} finished. Datagen rate: {:.2f} sec/demo".format(
                num_processed, ep_data_grp.attrs["num_samples"], ep, process_num, total_run.value, (time.time() - start_time) / num_processed
            ))
    except KeyboardInterrupt:
        print("Control C pressed. Closing File and ending \n\n\n\n")
    
    for left_ep in demo_keys:
        if f"data/{left_ep}" in tgt_f:
            del tgt_f[f"data/{left_ep}"]
    tgt_f["data"].attrs["total"] = total_samples.value
    
    tgt_f.close()
    if args.write_video:
        video_writer.close()
    
    print(f"success: {success_num}/{num_processed}")


def playback_dataset_multiprocessing(args):
    # some arg checking
    write_video = args.write_video #(args.video_path is not None)
    extra_str = ""
    extra_str += "_addobj" if args.add_obj_num > 0 else ""
    extra_str += "_use_actions" if args.use_actions else ""
    if args.video_path is None: 
        args.video_path = args.dataset.split(".hdf5")[0] + extra_str + ".mp4"
    assert not (args.render and write_video) # either on-screen or video but not both
    
    if args.save_new_data:
        tgt_dataset_path = args.dataset.split(".hdf5")[0] + extra_str + ".hdf5"
        tgt_data_info_path = args.dataset.split(".hdf5")[0] + extra_str + ".pt"
        if os.path.exists(tgt_dataset_path):
            os.remove(tgt_dataset_path)
        shutil.copy(args.dataset, tgt_dataset_path)

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1


    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    elif "data" in f.keys():
        demos = list(f["data"].keys())
    else:
        demos = None

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    left_demos = copy.copy(demos)
    
    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]
    
    num_demos = len(demos)
    f.close()
    
    num_process = args.num_process
    index = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()
    total_samples_shared = multiprocessing.Value('i', 0)
    num_finished = multiprocessing.Value('i', 0)
    mul_queue = multiprocessing.Queue()
    work_queue = multiprocessing.Queue()
    for index in range(num_demos):
        work_queue.put(index)
    current_work_array = multiprocessing.Array('i', num_process)
    processes = []
    for i in range(num_process):
        process = multiprocessing.Process(target=playback_demos_process, args=(i, current_work_array, work_queue, lock, args, num_finished, mul_queue))
        processes.append(process)
    
    if args.save_new_data:
        process_write = multiprocessing.Process(target=write_new_data_process, args=(args, tgt_dataset_path, total_samples_shared, num_finished, num_process, mul_queue, left_demos))
        processes.append(process_write)
    
    for process in processes:
        process.start()
    
    for process in processes:
        process.join()
    
    print("Finished Multiprocessing")
        
    save_data_info = {}
    
    
    torch.save(save_data_info, tgt_data_info_path)


def playback_dataset(args):
    # some arg checking
    write_video = args.write_video #(args.video_path is not None)
    extra_str = ""
    extra_str += "_addobj" if args.add_obj_num > 0 else ""
    extra_str += "_use_actions" if args.use_actions else ""
    if args.video_path is None: 
        args.video_path = args.dataset.split(".hdf5")[0] + extra_str + ".mp4"
    assert not (args.render and write_video) # either on-screen or video but not both
    
    if args.save_new_data:
        tgt_dataset_path = args.dataset.split(".hdf5")[0] + extra_str + ".hdf5"
        tgt_data_info_path = args.dataset.split(".hdf5")[0] + extra_str + ".pt"
        if os.path.exists(tgt_dataset_path):
            os.remove(tgt_dataset_path)
        shutil.copy(args.dataset, tgt_dataset_path)

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert not args.use_actions, "playback with observations is offline and does not support action playback"

    # create environment only if not playing back with observations
    if not args.use_obs:
        # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
        # for playback since observations are unused. Pass a dummy spec here.
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        # env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False # absolute action space
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)

        # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")
    if args.save_new_data:
        tgt_f = h5py.File(tgt_dataset_path, "r+")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    elif "data" in f.keys():
        demos = list(f["data"].keys())
    else:
        demos = None

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    left_demos = copy.copy(demos)
    
    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        # if not args.dont_shuffle_demos:
        #     random.shuffle(demos)
        # random.shuffle(demos)
        demos = demos[:args.n]
    
    f.close()

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)
        
    save_data_info = {}
        
    success_num = 0
    for ind in range(len(demos)):
        ep = demos[ind]
        left_demos.remove(ep)
        print("Playing back episode: {}".format(ep))

        # prepare initial state to reload from
        states = tgt_f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = tgt_f["data/{}".format(ep)].attrs["model_file"]
            initial_state["ep_meta"] = tgt_f["data/{}".format(ep)].attrs.get("ep_meta", None)
        
        ori_model = copy.copy(initial_state['model'])
        
        # supply actions if using open-loop action playback
        actions = None
        if args.use_actions:
            actions = tgt_f["data/{}/actions".format(ep)][()]
            # actions = f["data/{}/actions_abs".format(ep)][()] # absolute actions
        
        env.env.add_object_num = args.add_obj_num
        success = False
        for try_idx in range(3):
            try:
                initial_state['model'] = copy.copy(ori_model)
                outputs, success = playback_trajectory_with_env(
                    env=env, 
                    initial_state=initial_state, 
                    states=states, 
                    args=args,
                    actions=actions, 
                    render=args.render, 
                    video_writer=video_writer, 
                    video_skip=args.video_skip,
                    camera_names=args.render_image_names,
                    first=args.first,
                    ep=ep
                )
                if success:
                    break
            except KeyboardInterrupt:
                print('Control C pressed. Closing files and ending.')
                f.close()
                if args.save_new_data:
                    tgt_f.close()
                if write_video:
                    video_writer.close()
            except Exception as e:
                # print("try idx:", try_idx)
                print(traceback.format_exc())
                print(e)
                print("fail to reset env, try again...")
            
        if not success or outputs is None:
            if f"data/{ep}" in tgt_f:
                del tgt_f[f"data/{ep}"]
            continue

        success_num += 1
        
        if args.save_new_data:
            save_data_info[ep] = {
                "lang": outputs["lang"],
                "class": outputs["class"],
                "unique_attr": outputs["unique_attr"],
                "target_phrase": outputs["target_phrase"],
                "images": outputs["images"],
                "masks": outputs["masks"]
            }
            
            new_model = outputs["new_model"]
            new_ep_meta = outputs["new_ep_meta"]
            new_ep_meta['lang'] = outputs["lang"]
            new_ep_meta = json.dumps(new_ep_meta, indent=4)
            
            print('object number:', len(env.env.object_cfgs))
            
            tgt_f["data/{}".format(ep)].attrs["model_file"] = new_model
            tgt_f["data/{}".format(ep)].attrs["ep_meta"] = new_ep_meta
            for item_name in ['states', 'rewards', 'dones', 'actions_abs']:
                item_path = f"data/{ep}/{item_name}"
                if item_path in tgt_f:
                    del tgt_f[item_path]
                tgt_f.create_dataset(item_path, data=outputs[item_name])
    
    print(f"success: {success_num}/{len(demos)}")
    
    for ep in left_demos:
        if f"data/{ep}" in tgt_f:
            del tgt_f[f"data/{ep}"]

    if args.save_new_data:
        tgt_f.close()
    if write_video:
        video_writer.close()
    
    if args.save_new_data:
        torch.save(save_data_info, tgt_data_info_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )
    
    parser.add_argument(
        "--use_actions",
        action="store_true",
        help="use actions to playback demos"
    )
    
    parser.add_argument(
        "--add_obj_num",
        default=0,
        type=int,
        help="number of newly added objects"
    )
    
    parser.add_argument(
        "--write_video",
        action="store_true",
        help="write successful demos to video"
    )
    
    parser.add_argument(
        "--write_first_frame",
        action="store_true",
        help="only write the first frame of each demo"
    )
    
    parser.add_argument(
        "--save_new_data",
        action="store_true",
        help="save successful demos (with added objects)"
    )
    
    parser.add_argument(
        "--write_gt_mask",
        action="store_true",
        help="write the ground truth mask of the target object"
    )
    
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="use multiprocessing to process demos"
    )
    
    parser.add_argument(
        "--num_process",
        default=8,
        type=int,
        help="number of parallel processes"
    )

    args = parser.parse_args()
    if args.multiprocessing:
        playback_dataset_multiprocessing(args)
    else:
        playback_dataset(args)
    
