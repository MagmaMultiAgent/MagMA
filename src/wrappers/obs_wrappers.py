"""
Wrapper for Observation Space
"""
from typing import Any, Dict
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from collections import deque
from observation.observation_parser import FeatureParser

MAP_FEATURE_SIZE = 6
GLOBAL_FEATURE_SIZE = 2
FACTORY_FEATURE_SIZE = 6
UNIT_FEATURE_SIZE = 4

class SimpleUnitObservationWrapper(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. \
    If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env: gym.Env) -> None:
        """
        A simple state based observation to work with in pair with the SimpleUnitDiscreteController
        """

        super().__init__(env)
        self.map_space = spaces.Box(low=-999, high=999, shape=(MAP_FEATURE_SIZE, env.env_cfg.map_size, env.env_cfg.map_size), dtype=np.float32)
        self.global_space = spaces.Box(low=-999, high=999, shape=(GLOBAL_FEATURE_SIZE, env.env_cfg.map_size, env.env_cfg.map_size), dtype=np.float32)
        self.factory_space = spaces.Box(low=-999, high=999, shape=(FACTORY_FEATURE_SIZE, env.env_cfg.map_size, env.env_cfg.map_size), dtype=np.float32)
        self.unit_space = spaces.Box(low=-999, high=999, shape=(UNIT_FEATURE_SIZE, env.env_cfg.map_size, env.env_cfg.map_size), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "map": self.map_space,
            "global": self.global_space,
            "factory": self.factory_space,
            "unit": self.unit_space,
        })
        self.observation_parser = FeatureParser()

    def observation(self, obs: Dict[str, Any]) -> Dict[str, npt.NDArray]:
        """
        Takes as input the current "raw observation" and returns converted observation
        """

        converted_obs = SimpleUnitObservationWrapper.convert_obs(obs, self.env.env_cfg, self.observation_parser)
        return converted_obs


    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any, observation_parser: FeatureParser) -> Dict[str, npt.NDArray]:
        """
        Takes as input the current "raw observation" and returns converted observation
        """
        observation = {}
        global_features, map_features, factory_features, unit_features = observation_parser.parse(obs, env_cfg)
        for i, agent in enumerate(obs.keys()):
            observation[agent] = {
                "map": map_features[agent],
                "global": global_features[agent],
                "factory": factory_features[agent],
                "unit": unit_features[agent],
            }
        return observation
