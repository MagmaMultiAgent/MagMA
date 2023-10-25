"""
Wrapper for Observation Space
"""
from typing import Any, Dict
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from observation.obs_parser import ObservationParser
from collections import deque
import random

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
        self.map_space = spaces.Box(low=-999, high=999, shape=(30, 64, 64), dtype=np.float32)
        self.global_space = spaces.Box(low=-999, high=999, shape=(44,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "map": self.map_space,
            "global": self.global_space
        })
        self.observation_parser = ObservationParser()
        self.max_observation_history = 10
        self.observation_queue = deque(maxlen=self.max_observation_history)

    def observation(self, obs):
        """
        Takes as input the current "raw observation" and returns
        """
        converted_obs = SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg, self.observation_parser)
        self.map_observation_queue.append(converted_obs)

        if len(self.observation_queue) >= 5:
            past_3_observations = list(self.observation_queue)[-3:]

            selected_observations = self.select_observations()

            combined_map, combined_global = self.combine_observations(past_3_observations, selected_observations)

            self.observation_queue.append({"map": combined_map, "global": past_3_observations[-1]["global"]})
        return None
    
    def select_observations(self):
        
        num_observations_to_select = 2
        initial_weight = 0.3
        common_ratio = 0.8
        weights = [initial_weight * common_ratio**i for i in range(3, len(self.observation_queue))]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        selected_indices = random.choices(range(3, len(self.observation_queue) - 3), weights=weights, k=num_observations_to_select)
        selected_observations = [self.observation_queue[i] for i in selected_indices]

        return selected_observations
    
    def combine_observations(self, past3_observations, selected_observations):

        combined_map = {}
        combined_global = {}

        for player in ["player_0", "player_1"]:

            concatenated_map_obs = []
            concatenated_global_obs = []

            for obs in past3_observations:
                map_obs = obs[player]["map"]
                global_obs = obs[player]["global"]

                concatenated_map_obs.append(map_obs)
                concatenated_global_obs.append(global_obs)

            for obs in selected_observations:
                map_obs = obs[player]["map"]
                global_obs = obs[player]["global"]

                concatenated_map_obs.append(map_obs)
                concatenated_global_obs.append(global_obs)

            combined_global[player] = np.stack(concatenated_global_obs, axis=0)
            combined_map[player] = np.stack(concatenated_map_obs, axis=0)
    

        return combined_map, combined_global

    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any, obs_parsers: ObservationParser) -> Dict[str, npt.NDArray]:
        """
        Takes as input the current "raw observation" and returns converted observation
        """
        observation = {}
        map_features, global_features, _ = obs_parsers.parse_observation(obs, env_cfg)
        for i, agent in enumerate(obs.keys()):
            observation[agent] = {
                "map": map_features[i],
                "global": global_features[i]
            }
        return observation