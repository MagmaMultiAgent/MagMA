from typing import Dict, List, Tuple, Union
from rewards.ice_reward_parser import IceRewardParser
import gym
import copy
import numpy as np
import kit.kit


class EarlyRewardParserWrapper(gym.Wrapper):
    """
    Custom wrapper for the LuxAI_S2 environment
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment \
        into a single-agent environment for easy training
        """
        super().__init__(env)
        self.prev_step_metrics = None
        self.reward_parser = IceRewardParser()

        self.global_info_names = [
            'factory_count',
            'unit_count',
            'light_count',
            'heavy_count',
            'unit_ice',
            'unit_ore',
            'unit_water',
            'unit_metal',
            'unit_power',
            'factory_ice',
            'factory_ore',
            'factory_water',
            'factory_metal',
            'factory_power',
            'total_ice',
            'total_ore',
            'total_water',
            'total_metal',
            'total_power',
            'lichen_count',
            'units_on_ice',
            'avg_distance_from_ice',
            'rubble_on_ice',

            'ice_transfered',
            'ore_transfered',
            'ice_mined',
            'ore_mined',
            'lichen_grown',
            'unit_created',
            'light_created',
            'heavy_created',
            'unit_destroyed',
            'light_destroyed',
            'heavy_destroyed',
        ]

    def step(self, action):
        
        agent = "player_0"
        opp_agent = "player_1"

        opp_factories = self.env.state.factories[opp_agent]
        for k in opp_factories.keys():
            factory = opp_factories[k]
            factory.cargo.water = 1000

        action = {agent: action}
        obs, _, termination, truncation, info = self.env.step(action)
        done = dict()
        for k in termination:
            done[k] = termination[k] | truncation[k]

        obs = obs[agent]

        metrics = self._get_info(agent, self.env.state, prev_obs=self.prev_step_metrics)
        reward = self.reward_parser.parse(done, self.env.state, metrics)
        self.prev_step_metrics = copy.deepcopy(metrics)

        info["metrics"] = metrics
        return obs, reward, termination[agent], truncation[agent], info
    
    def _get_info(self, player: str, obs: kit.kit.GameState, prev_obs: kit.kit.GameState = None):

        global_info = {k: 0 for k in self.global_info_names}
        factories = list(obs.factories[player].values())
        units = list(obs.units[player].values())

        global_info['light_count'] = sum(int(u.unit_type == 'LIGHT') for u in units)
        global_info['heavy_count'] = sum(int(u.unit_type == 'HEAVY') for u in units)
        global_info['unit_count'] = global_info['light_count'] + global_info['heavy_count']
        global_info["factory_count"] = len(factories)

        global_info['unit_ice'] = sum(u.cargo.ice for u in units)
        global_info['unit_ore'] = sum(u.cargo.ore for u in units)
        global_info['unit_water'] = sum(u.cargo.water for u in units)
        global_info['unit_metal'] = sum(u.cargo.metal for u in units)
        global_info['unit_power'] = sum(u.power for u in units)

        global_info['factory_ice'] = sum(f.cargo.ice for f in factories)
        global_info['factory_ore'] = sum(f.cargo.ore for f in factories)
        global_info['factory_water'] = sum(f.cargo.water for f in factories)
        global_info['factory_metal'] = sum(f.cargo.metal for f in factories)
        global_info['factory_power'] = sum(f.power for f in factories)

        global_info['total_ice'] = global_info['unit_ice'] + global_info['factory_ice']
        global_info['total_ore'] = global_info['unit_ore'] + global_info['factory_ore']
        global_info['total_water'] = global_info['unit_water'] + global_info['factory_water']
        global_info['total_metal'] = global_info['unit_metal'] + global_info['factory_metal']
        global_info['total_power'] = global_info['unit_power'] + global_info['factory_power']
        global_info['lichen_count'] = np.sum(obs.board.lichen)

        units_positions = [u.pos for u in units]
        unit_board = np.zeros_like(obs.board.ice, dtype=np.bool8)
        for pos in units_positions:
            unit_board[pos.x, pos.x] = True
        ice_board = obs.board.ice
        units_standing_on_ice = [(ice_board[pos.x, pos.y] > 0) for pos in units_positions]
        unit_count_on_ice = sum(units_standing_on_ice)
        global_info['units_on_ice'] = unit_count_on_ice

        if len(units_positions) < 0:
            avg_distance_from_ice = 1.0
        else:
            avg_distance_from_ice = self.get_avg_distance(unit_board, ice_board)
        global_info['avg_distance_from_ice'] = avg_distance_from_ice

        ice_board = obs.board.ice.astype(np.float32)
        rubble_board = obs.board.rubble.astype(np.float32)
        rubble_on_ice = ice_board * rubble_board
        global_info['rubble_on_ice'] = np.sum(rubble_on_ice)

        global_info['ice_transfered'] = 0
        global_info['ore_transfered'] = 0
        global_info['ice_mined'] = 0
        global_info['ore_mined'] = 0
        global_info['lichen_grown'] = 0
        global_info['unit_created'] = 0
        global_info['light_created'] = 0
        global_info['heavy_created'] = 0
        global_info['unit_destroyed'] = 0
        global_info['light_destroyed'] = 0
        global_info['heavy_destroyed'] = 0

        if prev_obs is not None:
            global_info['ice_transfered'] = global_info['unit_ice'] - prev_obs['unit_ice']
            global_info['ore_transfered'] = global_info['unit_ore'] - prev_obs['unit_ore']
            global_info['ice_mined'] = global_info['total_ice'] - prev_obs['total_ice']
            global_info['ore_mined'] = global_info['total_ore'] - prev_obs['total_ore']
            global_info['lichen_grown'] = global_info['lichen_count'] - prev_obs['lichen_count']
            global_info['unit_created'] = global_info['unit_count'] - prev_obs['unit_count']
            global_info['light_created'] = global_info['light_count'] - prev_obs['light_count']
            global_info['heavy_created'] = global_info['heavy_count'] - prev_obs['heavy_count']
            global_info['unit_destroyed'] = prev_obs['unit_count'] - global_info['unit_count']
            global_info['light_destroyed'] = prev_obs['light_count'] - global_info['light_count']
            global_info['heavy_destroyed'] = prev_obs['heavy_count'] - global_info['heavy_count']

        return global_info
    

    @staticmethod
    def get_avg_distance(entities, targets):
        entity_coords = np.argwhere(entities)
        target_coords = np.argwhere(targets)

        if len(entity_coords) == 0 or len(target_coords) == 0:
            return 0

        target_directions = -(entity_coords[:, None] - target_coords)
        closest_target_distance = np.abs(target_directions).sum(axis=-1).min(axis=-1)
        avg_distance = closest_target_distance.mean() / (entities.shape[0] + entities.shape[1])
        
        return avg_distance

    def reset(self, **kwargs):
        """
        Resets the environment
        """
        obs, reset_info = self.env.reset(**kwargs)
        self.prev_step_metrics = None
        return obs["player_0"], reset_info