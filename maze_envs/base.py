### Code for Gromov-Wasserstein Imitation Learning, Arnaud Fickinger, 2022
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Convention for Environments
Name: <Task>_<Agent>
Methods:
    - the usual gym methods
    - optional compute_reward for goal envs
    - task_obs(obs): returns agent agnostic componets
    - agent_obs(obs): returns agent specific components
    - skill_obs(obs): returns agent skill space components
    - goal(obs): returns current goal position

The state method should be implemented as follows
    1. Agent specific information
    2. extra agent information (must be relative to the agent skill spc. components.)
    3. agent skill space components
    4. Task Info
The below indiciates how these values should be set. The numbers reference the above.
len(1 to 2) = AGENT_DIM
len(3) = SKILL_DIM
len(2 to 4) = TASK_DIM

Defaults:
agent_obs(obs): return obs[:AGENT_DIM]
skill_obs(obs): return obs[AGENT_DIM:AGENT_DIM+SKILL_DIM]
task_obs(obs): return obs[-TASK_DIM:]

For GoalEnvs, they will handle the defining the goal space as part of the state.
"""

import os
from collections import OrderedDict
import gym
import mujoco_py
import numpy as np
from gym import spaces
from gym.utils import seeding

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

class Env(gym.Env):

    ASSET = None
    AGENT_DIM = None
    TASK_DIM = None
    SKILL_DIM = None
    FRAME_SKIP = None
    NSUBSTEPS = 1
    
    def __init__(self, model_path=None, frame_skip=None):
        if model_path is None:
            model_path = self.ASSET
        if frame_skip is None:
            frame_skip = self.FRAME_SKIP
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=self.NSUBSTEPS)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        # Set the observation space
        self.observation_space = convert_observation_to_space(observation)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override (in addition to those required by gym.Env):
    # ----------------------------

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def agent_obs(self, obs):
        return obs[:self.AGENT_DIM]

    def skill_obs(self, obs):
        return obs[self.AGENT_DIM:self.AGENT_DIM + self.SKILL_DIM]

    def task_obs(self, obs):
        return obs[-self.TASK_DIM:]

    def display_skill(self, skill):
        self.model.body_pos[-1][:self.SKILL_DIM] = skill
    
    # Utils Methods
    # -----------------------------

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self,
               mode='human',
               width=400,
               height=400,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array':
            # import pdb;pdb.set_trace()
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)
            # import pdb; pdb.set_trace()

            # self._get_viewer(mode)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        # import pdb;
        # pdb.set_trace()
        self.viewer = self._viewers.get(mode)

        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            # import pdb; pdb.set_trace()
            self.viewer.cam.distance = 50.
            # self.viewer.cam.azimuth = 132.
            self.viewer.cam.elevation = -90.
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def get_site_com(self, site_name):
        return self.data.get_site_xpos(site_name)

class GoalEnv(gym.GoalEnv, Env):

    def agent_obs(self, obs):
        return obs['observation'][:self.AGENT_DIM]

    def skill_obs(self, obs):
        return obs['observation'][self.AGENT_DIM:self.AGENT_DIM + self.SKILL_DIM]

    def task_obs(self, obs):
        return obs['observation'][-self.TASK_DIM:]