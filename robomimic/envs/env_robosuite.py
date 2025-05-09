"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy
import random

import robosuite
try:
    import robocasa
except ImportError:
    pass

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.lang_utils as LangUtils
import robomimic.envs.env_base as EB
from robomimic.macros import LANG_EMB_KEY
from robosuite.utils.errors import RandomizationError
from robocasa.models.objects.kitchen_objects import ALL_OBJ_INFOS
from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCounterToMicrowave, PnPCounterToStove, PnPMicrowaveToCounter

import xml.etree.ElementTree as ET


class EnvRobosuite(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self, 
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True,
        env_lang=None, 
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @env_meta["use_images"] is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).

            lang: TODO add documentation
        """
        self.postprocess_visual_obs = postprocess_visual_obs

        # robosuite version check
        self._is_v1 = (robosuite.__version__.split(".")[0] == "1")
        if self._is_v1:
            assert (int(robosuite.__version__.split(".")[1]) >= 2), "only support robosuite v0.3 and v1.2+"

        kwargs = deepcopy(kwargs)

        # update kwargs based on passed arguments
        update_kwargs = dict(
            has_renderer=render,
            has_offscreen_renderer=(render_offscreen or use_image_obs),
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=use_image_obs,
            camera_depths=False,
        )
        kwargs.update(update_kwargs)

        if self._is_v1:
            if kwargs["has_offscreen_renderer"]:
                """
                remove reliance on egl_probe, may cause issues
                # ensure that we select the correct GPU device for rendering by testing for EGL rendering
                # NOTE: this package should be installed from this link (https://github.com/StanfordVL/egl_probe)
                import egl_probe
                valid_gpu_devices = egl_probe.get_available_devices()
                if len(valid_gpu_devices) > 0:
                    kwargs["render_gpu_device_id"] = valid_gpu_devices[0]
                """
                pass
        else:
            # make sure gripper visualization is turned off (we almost always want this for learning)
            kwargs["gripper_visualization"] = False
            del kwargs["camera_depths"]
            kwargs["camera_depth"] = False # rename kwarg

        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)
        self.env = robosuite.make(self._env_name, **kwargs)
        self.base_env = self.env # for mimicgen
        self.env_lang = env_lang
        self.target_phrase = ""

        if self._is_v1:
            # Make sure joint position observations and eef vel observations are active
            for ob_name in self.env.observation_names:
                if ("joint_pos" in ob_name) or ("eef_vel" in ob_name):
                    self.env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        obs = self.get_observation(obs)
        info["is_success"] = self.is_success()
        return obs, r, self.is_done(), info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        di = self.env.reset()
        
        # keep track of episode language and embedding
        if self.env_lang is not None:
            self._ep_lang_str = self.env_lang
        elif hasattr(self.env, "get_ep_meta"):
            # get ep_meta if applicable
            ep_meta = self.env.get_ep_meta()
            self._ep_lang_str = ep_meta.get("lang", "dummy")

            target_obj_name = self.env.target_obj_name
            unique_attr = self.env.unique_attr
            if unique_attr != 'class' and target_obj_name in ALL_OBJ_INFOS['obj_infos'] and self.env.target_obj_phrase in self._ep_lang_str:
                unique_attrs = ALL_OBJ_INFOS['obj_infos'][target_obj_name][unique_attr]
                if type(unique_attrs) != list:
                    unique_attrs = [unique_attrs]
                target_phrase = unique_attrs[0]
                target_phrase += " object"
                self._ep_lang_str = self._ep_lang_str.replace(self.env.target_obj_phrase, target_phrase)
                self.env.target_obj_phrase = target_phrase
        else:
            self._ep_lang_str = "dummy"
            

        # self._ep_lang_emb = LangUtils.get_lang_emb(self._ep_lang_str)
        
        return self.get_observation(di)
    
    #notifies the environment whether or not the next environemnt testing object should update its category
    def update_env(self, attr, value):
        setattr(self.env, attr, value)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        should_ret = False
        if "model" in state:
            if state.get("ep_meta", None) is not None:
                # set relevant episode information
                ep_meta = json.loads(state["ep_meta"])
            else:
                ep_meta = {}
            if hasattr(self.env, "set_attrs_from_ep_meta"): # older versions had this function
                self.env.set_attrs_from_ep_meta(ep_meta)
            elif hasattr(self.env, "set_ep_meta"): # newer versions
                self.env.set_ep_meta(ep_meta)
            
            target_obj_name = None
            for obj_cfg in ep_meta['object_cfgs']:
                if obj_cfg['name'] != self.env.target_obj_str:
                    continue
                # target_obj_name = info['mjcf_path'].split('/')[-2]
                target_obj_source = obj_cfg['info']['mjcf_path'].split('/')[-4]
                target_obj_id = obj_cfg['info']['mjcf_path'].split('/')[-2]
                target_obj_name = f"{target_obj_source}_{target_obj_id}"
            self.env.target_obj_name = target_obj_name
            
            self.env.no_placement = True
            # this reset is necessary.
            # while the call to env.reset_from_xml_string does call reset,
            # that is only a "soft" reset that doesn't actually reload the model.
            self.reset()
            self.env.no_placement = False
            
            ori_obj_names = [x['name'] for x in ep_meta['object_cfgs']] if 'object_cfgs' in ep_meta else []
            for obj_name, obj in self.env.objects.items():
               if obj_name not in ori_obj_names:
                   state['model'] = self.add_object_model(state['model'], obj.get_xml())
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml
                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                # with open('tmp.xml', 'w') as f:
                #     f.write(state['model'])
                # breakpoint()
                xml = self.env.edit_model_xml(state["model"])
            # breakpoint()
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            if not self._is_v1:
                # hide teleop visualization after restoring from model
                self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
                self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
            if hasattr(self.env, "unset_ep_meta"): # unset the ep meta after reset complete
                self.env.unset_ep_meta()
        if "states" in state:
            # breakpoint()
            self.env.sim.set_state_from_flattened(state["states"])
            self.env.sim.forward()
            should_ret = True
        
        # after loading model and states, sample placements for newly added objects
        if "model" in state:
            # breakpoint()
            placed_objects = {} # object already placed
            # Retrieve the original object
            for obj_name in ori_obj_names:
                obj = self.env.objects[obj_name]
                qpos = self.env.sim.data.get_joint_qpos(obj.joints[0])
                # breakpoint()
                remove_classes = (PnPCounterToMicrowave, PnPCounterToStove, PnPMicrowaveToCounter)
                if obj_name == "obj_container" and isinstance(self.env, remove_classes):
                    # to avoid container's influence when training for PnPCounterToMicrowave\PnPCounterToStove\PnPMicrowaveToCounter, but not PnPStoveToCounter
                    qpos[0] += 0.5
                    print(f"Change obj_container's position to {qpos}.")
                placed_objects[obj_name] = (tuple(qpos[:3].tolist()), qpos[3:], obj)
            # Add fixtures in the environment
            placed_objects.update(self.env.fxtr_placements)
            object_placements = None
            for try_idx in range(2):
                try:
                    # object tend to be placed
                    object_placements = self.env.placement_initializer.sample(placed_objects=placed_objects)
                except RandomizationError as e:
                    print("Randomization error in new object placement. Try #{}".format(try_idx))
                    continue
                break
            if object_placements is None:
                raise RandomizationError
            for obj_pos, obj_quat, obj in object_placements.values():
                self.env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        
        # update state as needed
        if hasattr(self.env, "update_sites"):
            # older versions of environment had update_sites function
            self.env.update_sites()
        if hasattr(self.env, "update_state"):
            # later versions renamed this to update_state
            self.env.update_state()

        if "goal" in state:
            self.set_goal(**state["goal"])
        if should_ret:
            # only return obs if we've done a forward call - otherwise the observations will be garbage
            return self.get_observation()
        return None
    
    def add_object_model(self, model_str, obj_str):
        obj_root = ET.fromstring(obj_str)
        obj_asset = obj_root.find("asset")
        obj_body = obj_root.find("worldbody").find("body").find("body")

        root = ET.fromstring(model_str)
        
        asset = root.find("asset")
        for texture in obj_asset.findall("texture"):
            asset.append(texture)
        for material in obj_asset.findall("material"):
            asset.append(material)
        for mesh in obj_asset.findall("mesh"):
            asset.append(mesh)
        
        worldbody = root.find("worldbody")
        worldbody.append(obj_body)
            
        return ET.tostring(root)

    # def remove_object_model(self, model_str, obj_name):
    #     obj_root = ET.fromstring(obj_str)
    #     obj_asset = obj_root.find("asset")
    #     obj_body = obj_root.find("worldbody").find("body").find("body")

    #     root = ET.fromstring(model_str)
        
    #     asset = root.find("asset")

    def render(self, mode="human", height=None, width=None, camera_name=None, segmentation=False):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        # if camera_name is None, infer from initial env kwargs
        if camera_name is None:
            camera_name = self._init_kwargs.get("camera_names", ["agentview"])[0]

        if mode == "human":
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.viewer.set_camera(cam_id)
            return self.env.render()
        elif mode == "rgb_array":
            return self.env.sim.render(height=height, width=width, camera_name=camera_name, segmentation=segmentation)[::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """
        if di is None:
            di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()
        ret = {}
        for k in di:
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                ret[k] = di[k][::-1]
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)

        # "object" key contains object information
        if "object-state" in di:
            ret["object"] = np.array(di["object-state"])

        if self._is_v1:
            for robot in self.env.robots:
                # add all robot-arm-specific observations. Note the (k not in ret) check
                # ensures that we don't accidentally add robot wrist images a second time
                pf = robot.robot_model.naming_prefix
                for k in di:
                    if k.startswith(pf) and (k not in ret) and \
                            (not k.endswith("proprio-state")):
                        ret[k] = np.array(di[k])
        else:
            # minimal proprioception for older versions of robosuite
            ret["proprio"] = np.array(di["robot-state"])
            ret["eef_pos"] = np.array(di["eef_pos"])
            ret["eef_quat"] = np.array(di["eef_quat"])
            ret["gripper_qpos"] = np.array(di["gripper_qpos"])

        # ret["lang_emb"] = np.array(self._ep_lang_emb)
        return ret

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        xml = self.env.sim.model.get_xml() # model xml file
        state = np.array(self.env.sim.get_state().flatten()) # simulator state
        info = dict(model=xml, states=state)
        if hasattr(self.env, "get_ep_meta"):
            # get ep_meta if applicable
            info["ep_meta"] = json.dumps(self.env.get_ep_meta(), indent=4)
        return info

    def get_reward(self):
        """
        Get current reward.
        """
        return self.env.reward()

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        return self.get_observation(self.env._get_goal())

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        return self.env.set_goal(**kwargs)

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """

        # Robosuite envs always rollout to fixed horizon.
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return { "task" : succ }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_spec[0].shape[0]

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.ROBOSUITE_TYPE

    @property
    def version(self):
        """
        Returns version of robosuite used for this environment, eg. 1.2.0
        """
        return robosuite.__version__

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(
            env_name=self.name,
            env_version=self.version,
            type=self.type,
            env_kwargs=deepcopy(self._init_kwargs)
        )

    @classmethod
    def create_for_data_processing(
        cls, 
        env_name, 
        camera_names, 
        camera_height, 
        camera_width, 
        reward_shaping, 
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. 

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
        """
        is_v1 = (robosuite.__version__.split(".")[0] == "1")
        has_camera = (len(camera_names) > 0)

        new_kwargs = {
            "reward_shaping": reward_shaping,
        }

        if has_camera:
            if is_v1:
                new_kwargs["camera_names"] = list(camera_names)
                new_kwargs["camera_heights"] = camera_height
                new_kwargs["camera_widths"] = camera_width
            else:
                assert len(camera_names) == 1
                if has_camera:
                    new_kwargs["camera_name"] = camera_names[0]
                    new_kwargs["camera_height"] = camera_height
                    new_kwargs["camera_width"] = camera_width

        kwargs.update(new_kwargs)

        # also initialize obs utils so it knows which modalities are image modalities
        image_modalities = list(camera_names)
        if is_v1:
            image_modalities = ["{}_image".format(cn) for cn in camera_names]
        elif has_camera:
            # v0.3 only had support for one image, and it was named "rgb"
            assert len(image_modalities) == 1
            image_modalities = ["rgb"]
        obs_modality_specs = {
            "obs": {
                "low_dim": [], # technically unused, so we don't have to specify all of them
                "rgb": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False, 
            render_offscreen=has_camera, 
            use_image_obs=has_camera, 
            postprocess_visual_obs=False,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return (Exception)

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
