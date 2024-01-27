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
import socket
import atexit

MAP_FEATURE_SIZE = 30
GLOBAL_FEATURE_SIZE = 44
FACTORY_FEATURE_SIZE = 1 * 4

def exit_handler(socket_to_close):
    # This function will be called on program exit
    if socket_to_close:
        socket_to_close.close()

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

    def __init__(self, env: gym.Env, ind:int = 0) -> None:
        """
        A simple state based observation to work with in pair with the SimpleUnitDiscreteController
        """

        super().__init__(env)

        # self.map_space = spaces.Box(low=-999, high=999, shape=(MAP_FEATURE_SIZE * 5, env.env_cfg.map_size, env.env_cfg.map_size), dtype=np.float32)
        # self.global_space = spaces.Box(low=-999, high=999, shape=(GLOBAL_FEATURE_SIZE * 5,), dtype=np.float32)
        # self.factory_space = spaces.Box(low=-999, high=999, shape=(FACTORY_FEATURE_SIZE,), dtype=np.float32)
        # self.observation_space = spaces.Dict({
        #     "map": self.map_space,
        #     "global": self.global_space,
        #     "factory": self.factory_space
        # })

        # Change: custom observation space
        self.observation_space = spaces.Dict({
            "_ENTITY_COUNT": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "LOCAL_entity": spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32),
            "GLOBAL_info": spaces.Box(low=-999, high=999, shape=(44,), dtype=np.float32),
            # "GLOBAL_map": spaces.Box(low=-999, high=999, shape=(30, env.env_cfg.map_size, env.env_cfg.map_size), dtype=np.float32)
        })

        self.observation_parser = ObservationParser()

        # self.max_observation_history = 10
        # self.observation_queue = deque(maxlen=self.max_observation_history)
        # for _ in range(self.max_observation_history):
        #     self.observation_queue.append({
        #         "player_0": {
        #             "map": np.zeros((MAP_FEATURE_SIZE, env.env_cfg.map_size, env.env_cfg.map_size)),
        #             "global": np.zeros((GLOBAL_FEATURE_SIZE,)),
        #             "factory": np.zeros((FACTORY_FEATURE_SIZE,))
        #         },
        #         "player_1": {
        #             "map": np.zeros((MAP_FEATURE_SIZE, env.env_cfg.map_size, env.env_cfg.map_size)),
        #             "global": np.zeros((GLOBAL_FEATURE_SIZE,)),
        #             "factory": np.zeros((FACTORY_FEATURE_SIZE,))
        #         }
        #     })

        self.ind = ind

        ADDRESS = "172.19.182.35"
        UDP_PORT = 8001

        addr = socket.getaddrinfo(
            ADDRESS, UDP_PORT,
            socket.AF_INET, socket.SOCK_DGRAM)[0]

        if self.ind == 0:
            self.socket = socket.socket(*addr[:3])
            self.socket.setblocking(False)
            atexit.register(exit_handler, self.socket)
            self.socket.connect(addr[4])
        else:
            self.socket = None


    def observation(self, obs):
        """
        Takes as input the current "raw observation" and returns
        """

        converted_obs, stream_data = SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg, self.observation_parser, ind=self.ind)

        if self.ind == 0:
            self.logger.debug(f"Sending stream data {len(stream_data)}")
            self.socket.send(stream_data)

        # self.observation_queue.append(converted_obs)
        # past_3_observations = list(self.observation_queue)[-3:]
        # selected_observations = self.select_observations()
        # converted_obs: Dict[str, Dict[str, np.ndarray]] = self.combine_observations(past_3_observations, selected_observations)

        # Change: return only entity_obs
        observation = {}
        for _, agent in enumerate(obs.keys()):
            entity = converted_obs[agent]["entity"]
            global_info = converted_obs[agent]["global"]
            global_map = converted_obs[agent]["map"]
            observation[agent] = {
                "_ENTITY_COUNT": entity.shape[0],
                "LOCAL_entity": entity,
                "GLOBAL_info": global_info,
                "GLOBAL_map": global_map
            }
        return observation

    def select_observations(self):
        
        num_observations_to_select = 2
        initial_weight = 0.3
        common_ratio = 0.8
        weights = [initial_weight * common_ratio**i for i in range(3, len(self.observation_queue))]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        selected_indices = random.choices(range(3, len(self.observation_queue)), weights=weights, k=num_observations_to_select)
        selected_observations = [self.observation_queue[i] for i in selected_indices]

        return selected_observations
    
    def combine_observations(self, past3_observations, selected_observations):

        converted_obs = {}
        for player in ["player_0", "player_1"]:

            concatenated_map_obs = []
            concatenated_global_obs = []

            for obs in past3_observations:
                map_obs = obs[player]["map"]
                global_obs = obs[player]["global"]
                factory_obs = obs[player]["factory"]

                concatenated_map_obs.append(map_obs)
                concatenated_global_obs.append(global_obs)

            for obs in selected_observations:
                map_obs = obs[player]["map"]
                global_obs = obs[player]["global"]
                factory_obs = obs[player]["factory"]

                concatenated_map_obs.append(map_obs)
                concatenated_global_obs.append(global_obs)

            combined_global = np.stack(concatenated_global_obs, axis=0)
            combined_global = combined_global.reshape(-1)
            combined_map = np.stack(concatenated_map_obs, axis=0)
            combined_map = combined_map.reshape(-1, combined_map.shape[-2], combined_map.shape[-1])

            converted_obs[player] = {
                "map": combined_map,
                "global": combined_global,
                "factory": factory_obs
            }

        return converted_obs

    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any, obs_parsers: ObservationParser, ind: int = 0) -> Dict[str, npt.NDArray]:
        """
        Takes as input the current "raw observation" and returns converted observation
        """
        observation = {}
        obs_pars = ObservationParser()
        map_features, global_features, factory_features, assembled_features, _, stream_data = obs_pars.parse_observation(obs, env_cfg, ind=ind)
        for i, agent in enumerate(obs.keys()):
            observation[agent] = {
                "map": map_features[i],
                "global": global_features[i],
                "factory": factory_features[i],
                "entity": assembled_features[i]
            }
        return observation, stream_data