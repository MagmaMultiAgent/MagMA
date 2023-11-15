"""
Wrapper for Observation Space
"""
from typing import Any, Dict
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from collections import deque
import random
import kit.kit
from luxai_s2.config import EnvConfig

MAP_FEATURE_SIZE = 30
GLOBAL_FEATURE_SIZE = 44
FACTORY_FEATURE_SIZE = 24 * 4

class ObservationParser():

    def __init__(self):
        self.setup_names()

    def setup_names(self):
        self.global_feature_names = [
            'env_step',
            'cycle',
            'hour',
            'daytime_or_night',
            'num_factory_own',
            'num_factory_enm',
            'total_lichen_own',
            'total_lichen_enm',
        ]
        for own_enemy in ['own', 'enemy']:
            self.global_feature_names += [
                f'factory_total_power_{own_enemy}',
                f'factory_total_ice_{own_enemy}',
                f'factory_total_water_{own_enemy}',
                f'factory_total_ore_{own_enemy}',
                f'factory_total_metal_{own_enemy}',
                f'num_light_{own_enemy}',
                f'num_heavy_{own_enemy}',
                f'robot_total_power_{own_enemy}',
                f'robot_total_ice_{own_enemy}',
                f'robot_total_water_{own_enemy}',
                f'robot_total_ore_{own_enemy}',
                f'robot_total_metal_{own_enemy}',
            ]

        self.map_names = [
            'ice',
            'ore',
            'rubble',
            'lichen',
            'lichen_strains',
            'lichen_strains_own',
            'lichen_strains_enm',
            'valid_region_indicator',
            'factory_name',
            'factory_power',
            'factory_ice',
            'factory_water',
            'factory_ore',
            'factory_metal',
            'factory_own',
            'factory_enm',
            'factory_can_build_light',
            'factory_can_build_heavy',
            'factory_can_grow_lichen',
            'factory_water_cost',
            'unit_id',
            'unit_power',
            'unit_ice',
            'unit_water',
            'unit_ore',
            'unit_metal',
            'unit_own',
            'unit_enm',
            'unit_light',
            'unit_heavy',
        ]

        self.global_information_names = [
            'player_factory_count',
            'player_light_count',
            'player_heavy_count',
            'player_unit_ice',
            'player_unit_ore',
            'player_unit_water',
            'player_unit_metal',
            'player_unit_power',
            'player_factory_ice',
            'player_factory_ore',
            'player_factory_water',
            'player_factory_metal',
            'player_factory_power',
            'player_total_ice',
            'player_total_ore',
            'player_total_water',
            'player_total_metal',
            'player_total_power',
            'player_lichen_count',
        ]

    def parse_observation(self, obs, env_cfg: EnvConfig):
        map_feature_list = []
        global_feature_list = []
        factory_feature_list = []
        for player, player_obs in obs.items():

            env_step = player_obs['real_env_steps'] + player_obs['board']['factories_per_team'] * 2 + 1
            game_state = kit.kit.obs_to_game_state(env_step, env_cfg, player_obs)

            map_features = self.get_map_features(player, game_state, env_cfg)
            global_features = self.get_global_features(game_state, player, env_cfg, map_features)
            factory_features = self.get_factory_features(game_state, player, env_cfg, global_features, map_features)
            
            map_features = np.array(list(map_features.values()))
            global_features = np.array(list(global_features.values()))
            factory_feature_list.append(factory_features)
            map_feature_list.append(map_features)
            global_feature_list.append(global_features)

        global_info = {player: self.get_global_info(player, game_state) for player in ['player_0', 'player_1']}

        return map_feature_list, global_feature_list, factory_feature_list, global_info

    def get_global_info(self, player: str, obs: kit.kit.GameState):

        global_info = {k: None for k in self.global_information_names}

        factories = list(obs.factories[player].values())
        units = list(obs.units[player].values())

        global_info['player_light_count'] = sum(int(unit.unit_type == 'LIGHT') for unit in units)
        global_info['player_heavy_count'] = sum(int(unit.unit_type == 'HEAVY') for unit in units)
        global_info['player_factory_count'] = len(factories)

        global_info['player_unit_ice'] = sum(unit.cargo.ice for unit in units)
        global_info['player_unit_ore'] = sum(unit.cargo.ore for unit in units)
        global_info['player_unit_water'] = sum(unit.cargo.water for unit in units)
        global_info['player_unit_metal'] = sum(unit.cargo.metal for unit in units)
        global_info['player_unit_power'] = sum(unit.power for unit in units)

        global_info['player_factory_ice'] = sum(f.cargo.ice for f in factories)
        global_info['player_factory_ore'] = sum(f.cargo.ore for f in factories)
        global_info['player_factory_water'] = sum(f.cargo.water for f in factories)
        global_info['player_factory_metal'] = sum(f.cargo.metal for f in factories)
        global_info['player_factory_power'] = sum(f.power for f in factories)

        global_info['player_total_ice'] = global_info['player_unit_ice'] + global_info['player_factory_ice']
        global_info['player_total_ore'] = global_info['player_unit_ore'] + global_info['player_factory_ore']
        global_info['player_total_water'] = global_info['player_unit_water'] + global_info['player_factory_water']
        global_info['player_total_metal'] = global_info['player_unit_metal'] + global_info['player_factory_metal']
        global_info['player_total_power'] = global_info['player_unit_power'] + global_info['player_factory_power']

        lichen = obs.board.lichen
        lichen_strains = obs.board.lichen_strains

        if factories:
            lichen_count = sum((np.sum(lichen[lichen_strains == factory.strain_id]) for factory in factories), 0)
            global_info['lichen_count'] = lichen_count
        else:
            global_info['lichen_count'] = 0
        return global_info
    
    def get_map_features(self, player: str, obs: kit.kit.GameState, env_cfg: EnvConfig):

        enemy = 'player_1' if player == 'player_0' else 'player_0'
        map_feature = {name: np.zeros_like(obs.board.ice, dtype=np.float32) for name in self.map_names}
        map_feature['ice'] = obs.board.ice
        map_feature['ore'] = obs.board.ore
        map_feature['rubble'] = obs.board.rubble
        map_feature['lichen'] = obs.board.lichen
        map_feature['lichen_strains'] = obs.board.lichen_strains
        map_feature['lichen_strains_own'] = sum(
            (obs.board.lichen_strains == f.strain_id for f in obs.factories[player].values()) if obs.factories else [],
            np.zeros_like(obs.board.lichen_strains, dtype=np.bool8),
        )
        map_feature['lichen_strains_enm'] = sum(
            (obs.board.lichen_strains == f.strain_id for f in obs.factories[enemy].values()) if obs.factories else [],
            np.zeros_like(obs.board.lichen_strains, dtype=np.bool8),
        )
        map_feature['valid_region_indicator'] = np.ones_like(obs.board.rubble)

        for owner, factories in obs.factories.items():
            for fid, factory in factories.items():
                x, y = factory.pos
                map_feature['factory_name'][x, y] = int(fid[len('factory_'):])
                map_feature['factory_power'][x, y] = factory.power
                map_feature['factory_ice'][x, y] = factory.cargo.ice
                map_feature['factory_water'][x, y] = factory.cargo.water
                map_feature['factory_ore'][x, y] = factory.cargo.ore
                map_feature['factory_metal'][x, y] = factory.cargo.metal
                map_feature['factory_own'][x, y] = owner == player
                map_feature['factory_enm'][x, y] = owner == enemy

                if (factory.cargo.metal >= env_cfg.ROBOTS['LIGHT'].METAL_COST) \
                    and (factory.power >= env_cfg.ROBOTS['LIGHT'].POWER_COST):
                    map_feature['factory_can_build_light'][x, y] = True
                if (factory.cargo.metal >= env_cfg.ROBOTS['HEAVY'].METAL_COST) \
                    and (factory.power >=env_cfg.ROBOTS['HEAVY'].POWER_COST):
                    map_feature['factory_can_build_heavy'][x, y] = True
                water_cost = np.sum(
                    obs.board.lichen_strains == factory.strain_id) // env_cfg.LICHEN_WATERING_COST_FACTOR + 1
                if factory.cargo.water >= water_cost:
                    map_feature['factory_can_grow_lichen'][x, y] = True
                map_feature['factory_water_cost'][x, y] = water_cost

        for owner, units in obs.units.items():
            for uid, unit in units.items():
                x, y = unit.pos
                map_feature['unit_id'][x, y] = int(uid[len('unit_'):])
                map_feature['unit_power'][x, y] = unit.power
                map_feature['unit_ice'][x, y] = unit.cargo.ice
                map_feature['unit_water'][x, y] = unit.cargo.water
                map_feature['unit_ore'][x, y] = unit.cargo.ore
                map_feature['unit_metal'][x, y] = unit.cargo.metal

                map_feature['unit_own'][x, y] = owner == player
                map_feature['unit_enm'][x, y] = owner == enemy

                map_feature['unit_light'][x, y] = unit.unit_type == "LIGHT"
                map_feature['unit_heavy'][x, y] = unit.unit_type == "HEAVY"

        light_cfg = env_cfg.ROBOTS['LIGHT']
        map_feature['rubble'] = map_feature['rubble'] / env_cfg.MAX_RUBBLE
        map_feature['lichen'] = map_feature['lichen'] / env_cfg.MAX_LICHEN_PER_TILE

        map_feature['factory_power'] = map_feature['factory_power'] / light_cfg.BATTERY_CAPACITY
        map_feature['unit_power'] = map_feature['unit_power'] / light_cfg.BATTERY_CAPACITY

        map_feature['factory_ice'] = map_feature['factory_ice'] / light_cfg.CARGO_SPACE
        map_feature['factory_water'] = map_feature['factory_water'] / light_cfg.CARGO_SPACE
        map_feature['factory_ore'] = map_feature['factory_ore'] / light_cfg.CARGO_SPACE
        map_feature['factory_metal'] = map_feature['factory_metal'] / light_cfg.CARGO_SPACE
        map_feature['factory_water_cost'] = map_feature['factory_water_cost'] / light_cfg.CARGO_SPACE
        map_feature['unit_ice'] = map_feature['unit_ice'] / light_cfg.CARGO_SPACE
        map_feature['unit_water'] = map_feature['unit_water'] / light_cfg.CARGO_SPACE
        map_feature['unit_ore'] = map_feature['unit_ore'] / light_cfg.CARGO_SPACE
        map_feature['unit_metal'] = map_feature['unit_metal'] / light_cfg.CARGO_SPACE

        return map_feature

    def get_global_features(self, obs: kit.kit.GameState, player: str, env_cfg: EnvConfig, map_feature: np.ndarray):
   
        enemy = 'player_1' if player == 'player_0' else 'player_0'
        global_feature = {name: 0 for name in self.global_feature_names}
        global_feature['env_step'] = obs.real_env_steps
        global_feature['cycle'] = obs.real_env_steps // env_cfg.CYCLE_LENGTH
        global_feature['hour'] = obs.real_env_steps % env_cfg.CYCLE_LENGTH
        global_feature['daytime_or_night'] = global_feature['hour'] < 30
        global_feature['num_factory_own'] = len(obs.factories[player])
        global_feature['num_factory_enm'] = len(obs.factories[enemy])

        global_feature['total_lichen_own'] = np.sum(obs.board.lichen[map_feature['lichen_strains_own']])
        global_feature['total_lichen_enm'] = np.sum(obs.board.lichen[map_feature['lichen_strains_enm']])

        for own_enm, pid in zip(['own', 'enm'], [player, enemy]):
            global_feature[f'factory_total_power_{own_enm}'] = sum(f.power for f in obs.factories[pid].values())
            global_feature[f'factory_total_ice_{own_enm}'] = sum(f.cargo.ice for f in obs.factories[pid].values())
            global_feature[f'factory_total_water_{own_enm}'] = sum(f.cargo.water for f in obs.factories[pid].values())
            global_feature[f'factory_total_ore_{own_enm}'] = sum(f.cargo.ore for f in obs.factories[pid].values())
            global_feature[f'factory_total_metal_{own_enm}'] = sum(f.cargo.metal for f in obs.factories[pid].values())

            global_feature[f'num_light_{own_enm}'] = sum(u.unit_type == "LIGHT" for u in obs.units[pid].values())
            global_feature[f'num_heavy_{own_enm}'] = sum(u.unit_type == "HEAVY" for u in obs.units[pid].values())
            global_feature[f'robot_total_power_{own_enm}'] = sum(u.power for u in obs.units[pid].values())
            global_feature[f'robot_total_ice_{own_enm}'] = sum(u.cargo.ice for u in obs.units[pid].values())
            global_feature[f'robot_total_water_{own_enm}'] = sum(u.cargo.water for u in obs.units[pid].values())
            global_feature[f'robot_total_ore_{own_enm}'] = sum(u.cargo.ore for u in obs.units[pid].values())
            global_feature[f'robot_total_metal_{own_enm}'] = sum(u.cargo.metal for u in obs.units[pid].values())

        light_cfg = env_cfg.ROBOTS['LIGHT']
        global_feature['total_lichen_own'] = global_feature['total_lichen_own'] / env_cfg.MAX_LICHEN_PER_TILE
        global_feature['total_lichen_enm'] = global_feature['total_lichen_enm'] / env_cfg.MAX_LICHEN_PER_TILE

        for own_enm in ['own', 'enm']:

            global_feature[f'factory_total_power_{own_enm}'] = global_feature[f'factory_total_power_{own_enm}'] / light_cfg.BATTERY_CAPACITY
            global_feature[f'factory_total_ice_{own_enm}'] = global_feature[f'factory_total_ice_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'factory_total_water_{own_enm}'] = global_feature[f'factory_total_water_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'factory_total_ore_{own_enm}'] = global_feature[f'factory_total_ore_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'factory_total_metal_{own_enm}'] = global_feature[f'factory_total_metal_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'robot_total_power_{own_enm}'] = global_feature[f'robot_total_power_{own_enm}'] / light_cfg.BATTERY_CAPACITY
            global_feature[f'robot_total_ice_{own_enm}'] = global_feature[f'robot_total_ice_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'robot_total_water_{own_enm}'] = global_feature[f'robot_total_water_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'robot_total_ore_{own_enm}'] = global_feature[f'robot_total_ore_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'robot_total_metal_{own_enm}'] = global_feature[f'robot_total_metal_{own_enm}'] / light_cfg.CARGO_SPACE

        return global_feature

    def get_factory_features(self, obs: kit.kit.GameState, player: str, env_cfg: EnvConfig, global_feature: np.ndarray, map_feature: np.ndarray):

        light_cfg = env_cfg.ROBOTS['LIGHT']

        factories = obs.factories[player]
        factory_count = len(factories.keys())

        features = np.zeros((4, 24))
        for i, (factory_name, factory) in enumerate(factories.items()):
            factory_id = int(factory_name.split("_")[1])
            cargo = factory.cargo

            power = factory.power / light_cfg.BATTERY_CAPACITY
            ice = cargo.ice / light_cfg.CARGO_SPACE
            ore = cargo.ore / light_cfg.CARGO_SPACE
            water = cargo.water / light_cfg.CARGO_SPACE
            metal = cargo.metal / light_cfg.CARGO_SPACE

            power_ratio = 0 if global_feature['factory_total_power_own'] == 0 else power / global_feature['factory_total_power_own']
            ice_ratio = 0 if global_feature['factory_total_ice_own'] == 0 else ice / global_feature['factory_total_ice_own']
            ore_ratio = 0 if global_feature['factory_total_ore_own'] == 0 else ore / global_feature['factory_total_ore_own']
            water_ratio = 0 if global_feature['factory_total_water_own'] == 0 else water / global_feature['factory_total_water_own']
            metal_ratio = 0 if global_feature['factory_total_metal_own'] == 0 else metal / global_feature['factory_total_metal_own']

            lichen = factory.owned_lichen_tiles(obs)
            lichen_ratio = 0 if global_feature['total_lichen_own'] == 0 else lichen / global_feature['total_lichen_own']

            factory_features = np.array([
                power, ice, ore, water, metal, lichen,
                power_ratio, ice_ratio, ore_ratio, water_ratio, metal_ratio, lichen_ratio
            ], dtype=np.float32)

            global_features = np.array([
                global_feature['env_step'],
                global_feature['cycle'],
                global_feature['hour'],
                global_feature['daytime_or_night'],
                global_feature['num_factory_own'],
                global_feature['num_factory_enm'],
                global_feature['total_lichen_own'],
                global_feature['total_lichen_enm'],
                global_feature['num_light_own'],
                global_feature['num_light_enm'],
                global_feature['num_heavy_own'],
                global_feature['num_heavy_enm']
            ])

            features_tmp = np.concatenate((factory_features, global_features), axis=0)
            features[i] = features_tmp

        features = features.flatten()
        return features

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
        self.map_space = spaces.Box(low=-999, high=999, shape=(MAP_FEATURE_SIZE * 5, env.env_cfg.map_size, env.env_cfg.map_size), dtype=np.float32)
        self.global_space = spaces.Box(low=-999, high=999, shape=(GLOBAL_FEATURE_SIZE * 5,), dtype=np.float32)
        self.factory_space = spaces.Box(low=-999, high=999, shape=(FACTORY_FEATURE_SIZE,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "map": self.map_space,
            "global": self.global_space,
            "factory": self.factory_space
        })
        self.observation_parser = ObservationParser()
        self.max_observation_history = 10
        self.observation_queue = deque(maxlen=self.max_observation_history)
        for _ in range(self.max_observation_history):
            self.observation_queue.append({
                "player_0": {
                    "map": np.zeros((MAP_FEATURE_SIZE, env.env_cfg.map_size, env.env_cfg.map_size)),
                    "global": np.zeros((GLOBAL_FEATURE_SIZE,)),
                    "factory": np.zeros((FACTORY_FEATURE_SIZE,))
                },
                "player_1": {
                    "map": np.zeros((MAP_FEATURE_SIZE, env.env_cfg.map_size, env.env_cfg.map_size)),
                    "global": np.zeros((GLOBAL_FEATURE_SIZE,)),
                    "factory": np.zeros((FACTORY_FEATURE_SIZE,))
                }
            })

    def observation(self, obs):
        """
        Takes as input the current "raw observation" and returns
        """

        converted_obs = SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg, self.observation_parser)

        ## Observation queue keeps 10 most recent observations
        self.observation_queue.append(converted_obs)

        ## Choose 5 observations as input
        recent_past_observations = list(self.observation_queue)[-3:]
        far_past_observations = self.select_observations()
        converted_obs: Dict[str, Dict[str, np.ndarray]] = self.combine_observations(recent_past_observations, far_past_observations)

        return converted_obs
    
    def select_observations(self):
        
        initial_weight = 0.3
        common_ratio = 0.8
        weights = [initial_weight * common_ratio**i for i in range(3, len(self.observation_queue))]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        selected_indices = random.choices(range(3, len(self.observation_queue)), weights=weights, k=2)
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
    def convert_obs(obs: Dict[str, Any], env_cfg: Any, obs_parsers: ObservationParser) -> Dict[str, npt.NDArray]:
        """
        Takes as input the current "raw observation" and returns converted observation
        """
        observation = {}
        obs_pars = ObservationParser()
        map_features, global_features, factory_features, _ = obs_pars.parse_observation(obs, env_cfg)
        for i, agent in enumerate(obs.keys()):
            observation[agent] = {
                "map": map_features[i],
                "global": global_features[i],
                "factory": factory_features[i]
            }
        return observation
