import os
import numpy as np
import json
import collections
from dm_control.utils import rewards

from ... import _SUITE_DIR, _UR5_XML_DIR
from ...robots import UR5WithDex
from ..base import BaseTask
from ...randomize.wrapper import RandPhysics, RandEnvironment


_CONTROL_TIMESTEP = .01  # (Seconds)
_DEFAULT_TIME_LIMIT = 5  # Default duration of an episode, in seconds.

_CONFIG_FILE_NAME = 'ur5e/reach_dex.json'

def ur5_reach_dex(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Create a ur5e env, aiming to reach a specified point.
    """
    config_path = os.path.join(_SUITE_DIR, 'configs', _CONFIG_FILE_NAME)
    with open(config_path, mode='r') as f:
        config = json.load(f)
    robot = UR5WithDex.from_file_path(
        xml_path=config['xml'],
        asset_paths=config['assets'],
        actuator_path=os.path.join(_UR5_XML_DIR, 'actuator_dex.xml'),
        mocap_path=os.path.join(_UR5_XML_DIR, 'mocap_dex.xml'),
        config=config
    )
    physics = Physics.from_rand_mjcf(robot)
    task = Reach()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):
    def end_to_target(self):
        data = self.named.data
        end_to_target = data.site_xpos['object_site'] + np.array([0, 0, 0.04]) - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_target)

class Reach(BaseTask):
    """A simple reach task for UR5.
    """
    def __init__(
        self,
        target_low=(0.6, -0.15, 0.026),
        target_high=(0.8, 0.15, 0.026),
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.target_low = target_low
        self.target_high = target_high

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
        """
        physics.set_freejoint_pos('object_anchor', np.random.uniform(
            low=self.target_low, high=self.target_high), np.zeros(4))
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        return obs

    def get_reward(self, physics):
        new_action = self._rescale_action(physics)
        action_penalty = np.sum(new_action ** 2) / new_action.shape[0]
        # print("action: ", self.current_action)
        # print("new_action: ", new_action)
        # print("penalty: ", action_penalty)
        distance = physics.end_to_target()
        return rewards.tolerance(distance, bounds=(0, 0.01), margin=0.1) - 0.01 * action_penalty