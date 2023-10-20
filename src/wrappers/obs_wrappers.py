"""
Wrapper for Observation Space
"""
import sys
from typing import Any, Dict
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from observation.obs_parser import ObservationParser
from net.test import EncoderDecoderNet
import torch

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
        self.observation_space = spaces.Box(low=-999, high=999, shape=(80, 64, 64))

    def observation(self, obs):
        """
        Takes as input the current "raw observation" and returns
        """

        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        """
        Takes as input the current "raw observation" and returns converted observation
        """
        observation = {}
        obs_pars = ObservationParser()
        map_features, global_features, _ = obs_pars.parse_observation(obs, env_cfg)
        for i,agent in enumerate(obs.keys()):
            observation[agent] = [map_features[i], global_features[i]]
            model = EncoderDecoderNet(observation_space = 30, num_actions=19, num_global_features=44)
            model.eval()
            with torch.no_grad():
                output = model(torch.from_numpy(observation[agent][0]).unsqueeze(0).to(torch.float), torch.from_numpy(observation[agent][1]).unsqueeze(0).to(torch.float))
                print(output.shape, file=sys.stderr)

        return observation