from typing import Dict, List, Tuple, Union
import gym
import copy
import numpy as np
from luxai_s2.state import StatsStateDict
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

        self.global_info_names = [
            'power_consumed',
            'water_consumed',
            'metal_consumed',
            'ore_transferred',
            'ice_transferred',
            'energy_pickup',
            'rubble_destroyed',
            'lichen_destroyed',
            'ice_dug',
            'ore_dug',
            'metal_produced',
            'water_produced',
            'lichen_produced',
            'light_robots_built',
            'heavy_robots_built',
            'light_power',
            'heavy_power',
            'factory_power',
            'total_power',
            'action_queue_updates_success',
            'action_queue_updates_total',
            'units_on_ice',
            'avg_distance_from_ice',
            'rubble_on_ice',
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

        termination_status = termination[agent] if isinstance(termination, dict) else termination
        truncation_status = truncation[agent] if isinstance(truncation, dict) else truncation

        obs = obs[agent]

        metrics = self._get_info(agent, self.env.state.stats[agent], self.env.state)
        reward = self.parse(self.env.state, metrics)

        self.prev_step_metrics = copy.deepcopy(metrics)

        info["metrics"] = metrics
        return obs, reward, termination_status, truncation_status, info
    
    def _get_info(self, player: str, stats: StatsStateDict , state: kit.kit.GameState):

        metrics = {k: 0 for k in self.global_info_names}

        units = list(state.units[player].values())

        
        metrics["power_consumed"] = stats["consumption"]["power"]["HEAVY"] + stats["consumption"]["power"]["LIGHT"] + stats["consumption"]["power"]["FACTORY"]
        metrics["water_consumed"] = stats["consumption"]["water"]
        metrics["metal_consumed"] = stats["consumption"]["metal"]

        
        metrics["ore_transferred"] = stats["transfer"]["ore"]
        metrics["ice_transferred"] = stats["transfer"]["ice"] ####
        
        metrics["energy_pickup"] = stats["pickup"]["power"]

        metrics["rubble_destroyed"] = stats["destroyed"]["rubble"]["LIGHT"] + stats["destroyed"]["rubble"]["HEAVY"] ##
        metrics["lichen_destroyed"] = stats["destroyed"]["lichen"]["LIGHT"] + stats["destroyed"]["lichen"]["HEAVY"]

        metrics["ice_dug"] = stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"] #### 
        metrics["ore_dug"] = stats["generation"]["ore"]["HEAVY"] + stats["generation"]["ore"]["LIGHT"]
        metrics["metal_produced"] = stats["generation"]["metal"]
        metrics["water_produced"] = stats["generation"]["water"] ####
        metrics["lichen_produced"] = stats["generation"]["lichen"] 
        metrics["light_robots_built"] = stats["generation"]["built"]["LIGHT"]
        metrics["heavy_robots_built"] = stats["generation"]["built"]["HEAVY"]
        metrics["light_power"] = stats["generation"]["power"]["LIGHT"]
        metrics["heavy_power"] = stats["generation"]["power"]["HEAVY"]
        metrics["factory_power"] = stats["generation"]["power"]["FACTORY"]
        
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        units_positions = [u.pos for u in units]
        unit_board = np.zeros_like(state.board.ice, dtype=np.bool8)
        for pos in units_positions:
            unit_board[pos.x, pos.x] = True
        ice_board = state.board.ice
        units_standing_on_ice = [(ice_board[pos.x, pos.y] > 0) for pos in units_positions]
        unit_count_on_ice = sum(units_standing_on_ice)
        metrics['units_on_ice'] = unit_count_on_ice

        if len(units_positions) < 0:
            avg_distance_from_ice = 1.0
        else:
            avg_distance_from_ice = self.get_avg_distance(unit_board, ice_board)
        metrics['avg_distance_from_ice'] = avg_distance_from_ice

        ice_board = state.board.ice.astype(np.float32)
        rubble_board = state.board.rubble.astype(np.float32)
        rubble_on_ice = ice_board * rubble_board
        metrics['rubble_on_ice'] = np.sum(rubble_on_ice)

        return metrics
    
    def parse(self, game_state, metrics):

        final_reward = 0
        reward_scale = 0.01
        ice_norm = 1
        step_weight_early = 1 + ((1000 - game_state.real_env_steps) / 1000) * 0.1


        if self.prev_step_metrics is not None:
            ice_dug_this_step = (metrics['ice_dug'] - self.prev_step_metrics['ice_dug']) / 4 * 0.1
            ice_transfered_this_step = (metrics['ice_transferred'] - self.prev_step_metrics['ice_transferred']) / 4
            water_increment_this_step = (metrics['water_produced'] - self.prev_step_metrics['water_produced'])

            ice_dug_this_step_reward = ice_dug_this_step * reward_scale / ice_norm * step_weight_early
            ice_transfered_this_step_reward = ice_transfered_this_step * reward_scale / ice_norm * step_weight_early
            water_increment_this_step_reward = water_increment_this_step * reward_scale / 4 * step_weight_early

            final_reward += ice_dug_this_step_reward
            final_reward += ice_transfered_this_step_reward
            final_reward += water_increment_this_step_reward

        return final_reward
    

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