"""
Wrapper for Observation Space
"""
from typing import Any, Dict
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from observation.observation_parser import FeatureParser

class SimpleUnitObservationWrapper(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. \
    If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    """

    def __init__(self, env: gym.Env) -> None:
        """
        A simple state based observation to work with in pair with the SimpleUnitDiscreteController
        """

        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(13,))

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
        return FeatureParser().parse(obs, env_cfg)
