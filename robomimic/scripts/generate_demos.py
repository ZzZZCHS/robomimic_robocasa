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
from PIL import Image
from collections import defaultdict

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType
# from robomimic.utils.ground_utils import GroundUtils

from robosuite.demos.demo_segmentation import segmentation_to_rgb
from robomimic.scripts.conversion.extract_action_dict import extract_action_dict
from robomimic.scripts.filter_dataset_size import filter_dataset_size

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

# set a number range of new objects for each env
ENV_NAME2RANGE = {
    "PnPCounterToCab": (3, 10),
    "PnPCabToCounter": (3, 6),
    "PnPCounterToSink": (3, 10),
    "PnPSinkToCounter": (2, 4),
    "PnPCounterToMicrowave": (2, 5),
    "PnPMicrowaveToCounter": (2, 4),
    "PnPCounterToStove": (3, 6),
    "PnPStoveToCounter": (1, 3),
    "OpenSingleDoor": (3, 6),
    "CloseSingleDoor": (3, 6),
    "OpenDoubleDoor": (3, 6),
    "CloseDoubleDoor": (3, 6),
    "OpenDrawer": (2, 5),
    "CloseDrawer": (2, 5),
    "TurnOnSinkFaucet": (3, 6),
    "TurnOffSinkFaucet": (3, 6),
    "TurnSinkSpout": (3, 6),
    "TurnOnStove": (3, 6),
    "TurnOffStove": (3, 6),
    "CoffeeSetupMug": (3, 6),
    "CoffeeServeMug": (3, 6),
    "CoffeePressButton": (3, 6),
    "TurnOnMicrowave": (3, 6),
    "TurnOffMicrowave": (3, 6)
}


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
    
    # load the initial state

    obs = env.reset_to(initial_state)
    
    save_images = []
    save_masks = []
    save_obs_dict = defaultdict(list)
    
    # breakpoint()
    
    if args.write_gt_mask:
        target_obj_str = env.env.target_obj_str
        if target_obj_str == "obj":
            target_obj_str += "_main"
        target_place_str = env.env.target_place_str
        
        seg_sensors = {}
        geom2body_id_mapping = {geom_id: body_id for geom_id, body_id in enumerate(env.env.sim.model.geom_bodyid)}
        for cam_name in camera_names:
            seg_sensor, seg_name = env.env._create_segmentation_sensor(cam_name, args.camera_width, args.camera_height, "element", "segmentation", custom_mapping=geom2body_id_mapping)
            seg_sensors[cam_name] = seg_sensor
        name2id = env.env.sim.model._body_name2id

    traj_len = states.shape[0]
    action_playback = (actions is not None)
    if action_playback:
        assert states.shape[0] == actions.shape[0]
    
    if env.env.add_object_num > 0:
        zero_actions = np.zeros(actions.shape[1])
        for i in range(50):
            obs, _, _, _ = env.step(zero_actions)

    success = False
    outputs = defaultdict(list)

    frames = []
    
    obj_infos = {}
    for obj in env.env.objects.values():
        obj_name = obj.name
        obj_qpos = env.env.sim.data.get_joint_qpos(obj.joints[0]).tolist()
        obj_folder = obj.folder
        obj_id = '_'.join([obj_folder.split('/')[-3], obj_folder.split('/')[-1]])
        obj_infos[obj_name] = {
            'qpos': obj_qpos,
            'id': obj_id
        }
    # breakpoint()
    
    for i in range(traj_len):
        if args.save_obs:
            for k in obs.keys():
                save_obs_dict[k].append(obs[k])
            for cam_name in camera_names:
                # image_name = f"{cam_name}_image"
                # save_obs_dict[image_name].append(obs[image_name])
                # save depth image
                # depth_name = f"{cam_name}_depth"
                # _, depth = env.env.sim.render(
                #     camera_name=cam_name,
                #     width=args.camera_width,
                #     height=args.camera_height,
                #     depth=True
                # )
                # depth = np.expand_dims(depth[::-1], axis=-1)
                # save_obs_dict[depth_name].append(depth)
                # Image.fromarray(((depth-depth.min())/(depth.max()-depth.min())*255).astype(np.uint8)).save('tmp.jpg')

                # save segmentation mask
                if video_count == 0 and args.write_gt_mask:
                    mask_name = f"{cam_name}_mask"
                    tmp_seg = seg_sensors[cam_name]().squeeze(-1)[::-1]
                    tmp_mask = np.zeros(tmp_seg.shape, dtype=np.uint8)
                    for tmp_target_obj_str in target_obj_str.split('/'):
                        tmp_mask[tmp_seg == name2id[tmp_target_obj_str] + 1] = 1
                    if target_place_str:
                        tmp_mask[tmp_seg == name2id[target_place_str] + 1] = 2
                        # a special case
                        if (tmp_seg == name2id[target_place_str] + 1).sum() == 0 and target_place_str == "container_main" and None in name2id and name2id[target_place_str] == name2id[None] - 1:
                            tmp_mask[tmp_seg == name2id[None] + 1] = 2
                    save_obs_dict[mask_name].append(np.expand_dims(tmp_mask, axis=-1))
        state = env.get_state()["states"]
        if action_playback:
            obs, r, _, info = env.step(actions[i])
        else:
            obs = env.reset_to({"states" : states[i]})
            r = env.get_reward()
            
        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])
            
        done = success = env.is_success()["task"]
        done = int(done)
        new_distr_names = [f"new_distr_{i}_main" for i in range(1, env.env.add_object_num+1)]
        
        # video render
        if video_writer is not None and (not args.write_first_frame and video_count % video_skip == 0 or args.write_first_frame and video_count == 0):
            video_img = []
            for cam_name in camera_names:
                video_img.append(env.render(mode="rgb_array", height=args.camera_height, width=args.camera_width, camera_name=cam_name))
            if len(save_images) == 0:
                save_images.extend(video_img)
            # breakpoint()
            video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
            if args.write_gt_mask:
                seg_img = []
                seg_img2 = []
                for cam_name in camera_names:
                    tmp_seg = seg_sensors[cam_name]().squeeze(-1)[::-1]
                    # seg_rgb = segmentation_to_rgb(tmp_seg, random_colors=False)
                    seg_rgb = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
                    seg_rgb2 = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
                    for tmp_target_obj_str in target_obj_str.split('/'):
                        seg_rgb[tmp_seg == name2id[tmp_target_obj_str] + 1, 0] = 255
                    if target_place_str:
                        seg_rgb2[tmp_seg == name2id[target_place_str] + 1, 2] = 255
                        # a special case
                        if (tmp_seg == name2id[target_place_str] + 1).sum() == 0 and target_place_str == "container_main" and None in name2id and name2id[target_place_str] == name2id[None] - 1:
                            seg_rgb2[tmp_seg == name2id[None] + 1, 2] = 255
                    seg_img.append(seg_rgb)
                    seg_img2.append(seg_rgb2)
                if len(save_masks) == 0:
                    save_masks.extend(seg_img)
                seg_img = np.concatenate(seg_img, axis=1)
                seg_img2 = np.concatenate(seg_img2, axis=1)
                seg_video_img = video_img
                # alpha = 0.6
                seg_video_img = seg_video_img // 5 * 3 + seg_img // 5 * 2
                seg_video_img2 = video_img // 5 * 3 + seg_img2 // 5 * 2
                # breakpoint()
                video_img = np.concatenate([video_img, seg_video_img, seg_video_img2], axis=0)
            frame = video_img.copy()
            text1 = env._ep_lang_str
            position1 = (10, 50)
            color = (255, 0, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            font_scale = 0.6
            # cv2.putText(frame, text1, position1, font, font_scale, color, thickness)
            text2 = f"{ep}  Unique Attribute: {env.env.unique_attr}"
            position2 = (10, 100)
            # cv2.putText(frame, text2, position2, font, font_scale, color, thickness)
            frames.append(frame)
        video_count += 1

        outputs['rewards'].append(r)
        outputs['dones'].append(done)
        if action_playback:
            outputs['actions_abs'].append(env.base_env.convert_rel_to_abs_action(actions[i]))
        outputs['states'].append(state)
        
        if args.skip_replay:
            success = True
            break
    
    # breakpoint()
    print(len(frames))
    # if write_video and success:
    #     for frame in frames:
    #         video_writer.append_data(frame)
    
    outputs.update({
        # "images": save_images,
        # "masks": save_masks,
        "ep": ep,
        "lang": env._ep_lang_str,
        "unique_attr": env.env.unique_attr,
        "target_obj_phrase": env.env.target_obj_phrase,
        "target_place_phrase": env.env.target_place_phrase,
        "new_model": env.env.sim.model.get_xml(),
        "new_ep_meta": env.env.get_ep_meta(),
        "frames": frames,
        "save_obs_dict": save_obs_dict,
        "obj_infos": obj_infos
    })
    
    if args.skip_replay:
        state = {
            "model": env.env.sim.model.get_xml(),
            "states": np.array(env.env.sim.get_state().flatten()),
            "ep_meta": env.env.get_ep_meta()
        }
        state["ep_meta"].update({
            "lang": env._ep_lang_str,
            "unique_attr": env.env.unique_attr,
            "target_obj_phrase": env.env.target_obj_phrase,
            "target_place_phrase": env.env.target_place_phrase
        })
        state["ep_meta"] = json.dumps(state["ep_meta"], indent=4)
        outputs['state'] = state
        
    print("Success:", success)
    return outputs, success


def playback_dataset(args):
    # some arg checking
    write_video = args.write_video #(args.video_path is not None)
    extra_str = ""
    extra_str += "_addobj"
    extra_str += "_use_actions" if args.use_actions else ""
    extra_str += f"_process{args.global_process_id}" if args.global_process_id else ""
    extra_str += f"_{args.unique_attr}" if args.unique_attr else ""
    # extra_str += "_hhf"
    if write_video and args.video_path is None: 
        args.video_path = args.dataset.split(".hdf5")[0] + extra_str + ".mp4"
    tmp_save_path = ""
    tmp_save_infos = defaultdict(list)
    assert not (args.render and write_video) # either on-screen or video but not both
    
    if args.save_new_data:
        if args.tgt_dataset_path is None:
            tgt_dataset_path = args.dataset.split(".hdf5")[0] + extra_str + ".hdf5"
        else:
            tgt_dataset_path = args.tgt_dataset_path
        # tgt_data_info_path = args.dataset.split(".hdf5")[0] + extra_str + ".pt"
        if os.path.exists(tgt_dataset_path):
            os.remove(tgt_dataset_path)
        # shutil.copy(args.dataset, tgt_dataset_path)

    # Auto-fill camera rendering info if not specified
    if args.camera_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.camera_names = DEFAULT_CAMERAS[env_type]

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

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
        
        # env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=True)
        env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=args.camera_names, 
            camera_height=args.camera_height, 
            camera_width=args.camera_width, 
            reward_shaping=args.shaped,
        )

        # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)
    
    f = h5py.File(args.dataset, "r")
    if args.save_new_data:
        tgt_f = h5py.File(tgt_dataset_path, "w")
        data_grp = tgt_f.create_group("data")
    else:
        tgt_f = None

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
    
    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        # if not args.dont_shuffle_demos:
        #     random.shuffle(demos)
        # random.shuffle(demos)
        demos = demos[:args.n]
    
    demos = demos[args.interval_left:args.interval_right]
    # demos = demos[2:3]
    # add_num_min, add_num_max = ENV_NAME2RANGE[env._env_name]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)
        
    save_data_info = {}
        
    success_num = 0
    total_samples = 0
    # breakpoint()
    for ind in range(len(demos)):
        ep = demos[ind]
        f_ep = f[f"data/{ep}"]
        print(f"Playing back episode: {ep}")
        
        # if 'obj_info' in f_ep.attrs and len(json.loads(f_ep.attrs['obj_infos'])) < 6:
        #     continue
        
        add_num_min, add_num_max = ENV_NAME2RANGE[env._env_name]
        
        # prepare initial state to reload from
        states = f_ep["states"][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f_ep.attrs["model_file"]
            initial_state["ep_meta"] = f_ep.attrs.get("ep_meta", None)
        
        if not args.use_actions:
            tmp_ep_meta = json.loads(initial_state["ep_meta"])
            tmp_ep_meta["unique_attr"] = "class"
            initial_state["ep_meta"] = json.dumps(tmp_ep_meta, indent=4)
        
        if args.unique_attr:
            tmp_ep_meta = json.loads(initial_state["ep_meta"])
            tmp_ep_meta["unique_attr"] = args.unique_attr
            initial_state["ep_meta"] = json.dumps(tmp_ep_meta, indent=4)
        
        ori_model = copy.copy(initial_state['model'])
        
        # supply actions if using open-loop action playback
        actions = f_ep["actions"][()]
        
        success = False
        try_max = 3 
        for try_idx in range(try_max):
            try:
                if not args.use_actions or try_idx == try_max - 1:
                    env.env.add_object_num = 0
                elif args.add_obj_num != -1:
                    env.env.add_object_num = args.add_obj_num
                else:
                    env.env.add_object_num = random.randint(add_num_min, add_num_max)
                initial_state['model'] = copy.copy(ori_model)
                outputs, success = playback_trajectory_with_env(
                    env=env, 
                    initial_state=initial_state, 
                    states=states, 
                    args=args,
                    actions=actions if args.use_actions else None, 
                    render=args.render, 
                    video_writer=video_writer, 
                    video_skip=args.video_skip,
                    camera_names=args.camera_names,
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
                if args.skip_replay:
                    torch.save(tmp_save_infos, tmp_save_path)
                return
            except Exception as e:
                print("try idx:", try_idx)
                print(traceback.format_exc())
                print(e)
                print("fail to reset env, try again...")
                add_num_max = max(add_num_max - 1, 1)
                add_num_min = max(add_num_min - 1, 0)
            
        if not success or outputs is None:
            continue

        if write_video:
            for frame in outputs["frames"]:
                video_writer.append_data(frame)

        success_num += 1
        
        if args.skip_replay:
            tmp_save_infos[env._env_name].append(outputs['state'])
            del outputs['state']
        
        if args.save_new_data:
            new_model = outputs["new_model"]
            new_ep_meta = outputs["new_ep_meta"]
            new_ep_meta["lang"] = outputs["lang"]
            new_ep_meta["unique_attr"] = outputs["unique_attr"]
            new_ep_meta["target_obj_phrase"] = outputs["target_obj_phrase"]
            new_ep_meta["target_place_phrase"] = outputs["target_place_phrase"]
            new_ep_meta = json.dumps(new_ep_meta, indent=4)
            new_obj_infos = json.dumps(outputs['obj_infos'], indent=4)
            
            print('object number:', len(env.env.object_cfgs))
            print('instruction:', outputs['lang'])
            
            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset("actions", data=actions)
            ep_data_grp.create_dataset("states", data=np.array(outputs["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(outputs["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(outputs["dones"]))
            if args.use_actions:
                ep_data_grp.create_dataset("actions_abs", data=np.array(outputs["actions_abs"]))
            else:
                ep_data_grp.create_dataset("actions_abs", data=f_ep["actions_abs"][()])
            for k in f_ep["obs"].keys():
                if k not in outputs["save_obs_dict"]:
                    ep_data_grp.create_dataset(f"obs/{k}", data=np.array(f_ep[f"obs/{k}"][()]), compression="gzip")
            for k in outputs["save_obs_dict"]:
                ep_data_grp.create_dataset(f"obs/{k}", data=np.array(outputs["save_obs_dict"][k]), compression="gzip")
            if "action_dict" in f_ep:
                action_dict = f_ep["action_dict"]
                for k in action_dict:
                    ep_data_grp.create_dataset(f"action_dict/{k}", data=np.array(action_dict[k][()]))
            ep_data_grp.attrs["model_file"] = new_model
            ep_data_grp.attrs["ep_meta"] = new_ep_meta
            ep_data_grp.attrs["obj_infos"] = new_obj_infos
            ep_data_grp.attrs["num_samples"] = actions.shape[0]
            total_samples += actions.shape[0]
    
    print(f"success: {success_num}/{len(demos)}")
    
    if args.skip_replay:
        torch.save(tmp_save_infos, tmp_save_path)
    
    if args.save_new_data:
        if "mask" in f:
            f.copy("mask", tgt_f)
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
        tgt_f.close()
    f.close()
    
    if args.save_new_data:
        extract_action_dict(dataset=tgt_dataset_path)
        for num_demos in [10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]:
            filter_dataset_size(
                tgt_dataset_path,
                num_demos=num_demos,
            )
        
    if write_video:
        video_writer.close()
        print("Video is saved at ", args.video_path)
    
    # if args.save_new_data:
    #     torch.save(save_data_info, tgt_data_info_path)
        

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

    parser.add_argument(
        "--tgt_dataset_path",
        type=str,
        default=None
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
        "--camera_names",
        type=str,
        nargs='+',
        default=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )
    
    parser.add_argument(
        "--camera_height",
        type=int,
        default=256,
        help="height of image observations"
    )
    
    parser.add_argument(
        "--camera_width",
        type=int,
        default=256,
        help="width of image observations"
    )
    
    # flag for reward shaping
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
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
        default=-1,
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
        "--skip_replay",
        action="store_true",
        help="skip the replay of actions, only for write first frames and masks"
    )
    
    parser.add_argument(
        "--save_new_data",
        action="store_true",
        help="save successful demos (with added objects)"
    )
    
    parser.add_argument(
        "--save_obs",
        action="store_true",
        help="save obs and masks"
    )
    
    parser.add_argument(
        "--write_gt_mask",
        action="store_true",
        help="write the ground truth mask of the target object"
    )
    
    parser.add_argument(
        "--interval_left",
        default=0,
        type=int,
        help="only process demos with id >= interval_left"
    )
    
    parser.add_argument(
        "--interval_right",
        default=100000,
        type=int,
        help="only process demos with id < interval_right"
    )
    
    parser.add_argument(
        "--global_process_id",
        default=None
    )
    
    parser.add_argument(
        "--unique_attr",
        type=str,
        default=None
    )

    args = parser.parse_args()
    # breakpoint()
    playback_dataset(args)
    
