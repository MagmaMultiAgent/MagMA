from functools import reduce
from copy import deepcopy
from kit.kit import GameState
from reward.reward_config import EarlyRewardParam, LateRewardParam
from reward.reward_config import stats_reward_params
from reward.reward_config import global_information_names
import numpy as np
import tree

import sys

import logging
logger = logging.getLogger(__name__)

class EarlyRewardParser:

    def __init__(self,):
        logger.debug(f"Creating {self.__class__.__name__}")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")

        self.step_count = 0

    def reset(self, global_info, env_stats):
        self.logger.debug("Resetting environment")
        self.update_last_count(global_info)
        self.update_env_stats(env_stats)

    def parse(self, game_state: GameState, env_stats, own_global_info, enm_global_info = None, done = False, late = False):
        self.logger.debug("Parsing rewards")
        
        self.step_count += 1

        sub_rewards_keys = [
            "reward_light",
            "reward_heavy",
            "reward_ice",
            "reward_ore",
            "reward_water",
            "reward_metal",
            "reward_factory",
            "reward_survival",
            "reward_win_lose",
            "reward_lichen",
        ]

        sub_rewards = {k: 0 for k in sub_rewards_keys}
        

        env_stats_rewards = tree.map_structure(
            lambda cur, last, param: (cur - last) * param,
            env_stats,
            self.last_env_stats,
            stats_reward_params,
        )
        env_stats_rewards = tree.flatten_with_path(env_stats_rewards)
        env_stats_rewards = list(
            map(
                lambda item: {"reward_" + "_".join(item[0]).lower(): item[1]},
                 env_stats_rewards,
            ))
        env_stats_rewards = reduce(lambda cat1, cat2: dict(cat1, **cat2), env_stats_rewards)
        sub_rewards.update(env_stats_rewards)

        last_count = self.last_count
        own_sub_rewards = sub_rewards

        factories_increment = own_global_info["player_factory_count"] - last_count['player_factory_count']
        light_increment = own_global_info["player_light_count"] - last_count['player_light_count']
        heavy_increment = own_global_info["player_heavy_count"] - last_count['player_heavy_count']
        ice_increment = own_global_info["player_total_ice"] - last_count['player_total_ice']
        ore_increment = own_global_info["player_total_ore"] - last_count['player_total_ore']
        water_increment = own_global_info["player_total_water"] - last_count['player_total_water']
        metal_increment = own_global_info["player_total_metal"] - last_count['player_total_metal']
        power_increment = own_global_info["player_total_power"] - last_count['player_total_power']

        all_past_reward = 0
        if late:
            if done:
                if enm_global_info["player_factory_count"] == 0:
                    win = True
                elif own_global_info["player_lichen_count"] > enm_global_info["player_lichen_count"]:
                    win = True
                else:
                    win = False

                if win:
                    own_sub_rewards["reward_win_lose"] = LateRewardParam.win_reward_weight * (own_global_info["player_lichen_count"] - enm_global_info["player_lichen_count"])**0.5
                    own_sub_rewards["reward_win_lose"] += (game_state.env_cfg.max_episode_length - game_state.real_env_steps) * LateRewardParam.survive_reward_weight * 2
            
            light_increment = own_global_info["player_lichen_count"] - last_count["player_lichen_count"]
            own_sub_rewards["reward_lichen"] = light_increment * LateRewardParam.lichen_reward_weight
            all_past_reward += own_global_info["player_lichen_count"] * LateRewardParam.lichen_reward_weight
            own_sub_rewards["reward_light"] = light_increment * LateRewardParam.light_reward_weight
            own_sub_rewards["reward_heavy"] = heavy_increment * LateRewardParam.heavy_reward_weight
            own_sub_rewards["reward_ice"] = max(ice_increment, 0) * LateRewardParam.ice_reward_weight
            own_sub_rewards["reward_ore"] = max(ore_increment, 0) * LateRewardParam.ore_reward_weight
            own_sub_rewards["reward_water"] = water_increment * LateRewardParam.water_reward_weight
            own_sub_rewards["reward_metal"] = metal_increment * LateRewardParam.metal_reward_weight
            own_sub_rewards["reward_power"] = power_increment * LateRewardParam.power_reward_weight
            own_sub_rewards["reward_factory"] = factories_increment * LateRewardParam.factory_penalty_weight
            own_sub_rewards["reward_survival"] = LateRewardParam.survive_reward_weight

            all_past_reward = 0
            all_past_reward += own_global_info["player_light_count"] * LateRewardParam.light_reward_weight
            all_past_reward += own_global_info["player_heavy_count"] * LateRewardParam.heavy_reward_weight
            all_past_reward += own_global_info["player_total_ice"] * LateRewardParam.ice_reward_weight
            all_past_reward += own_global_info["player_total_ore"] * LateRewardParam.ore_reward_weight
            all_past_reward += own_global_info["player_total_water"] * LateRewardParam.water_reward_weight
            all_past_reward += own_global_info["player_total_metal"] * LateRewardParam.metal_reward_weight
            all_past_reward += own_global_info["player_total_power"] * LateRewardParam.power_reward_weight
            all_past_reward += own_global_info["player_factory_count"] * LateRewardParam.factory_penalty_weight
            all_past_reward += game_state.env_steps * LateRewardParam.survive_reward_weight

        else:
            own_sub_rewards["reward_light"] = light_increment * EarlyRewardParam.light_reward_weight
            own_sub_rewards["reward_heavy"] = heavy_increment * EarlyRewardParam.heavy_reward_weight
            own_sub_rewards["reward_ice"] = max(ice_increment, 0) * EarlyRewardParam.ice_reward_weight
            own_sub_rewards["reward_ore"] = max(ore_increment, 0) * EarlyRewardParam.ore_reward_weight
            own_sub_rewards["reward_water"] = water_increment * EarlyRewardParam.water_reward_weight
            own_sub_rewards["reward_metal"] = metal_increment * EarlyRewardParam.metal_reward_weight
            own_sub_rewards["reward_power"] = power_increment * EarlyRewardParam.power_reward_weight
            own_sub_rewards["reward_factory"] = factories_increment * EarlyRewardParam.factory_penalty_weight
            own_sub_rewards["reward_survival"] = EarlyRewardParam.survive_reward_weight

            all_past_reward = 0
            all_past_reward += own_global_info["player_light_count"] * EarlyRewardParam.light_reward_weight
            all_past_reward += own_global_info["player_heavy_count"] * EarlyRewardParam.heavy_reward_weight
            all_past_reward += own_global_info["player_total_ice"] * EarlyRewardParam.ice_reward_weight
            all_past_reward += own_global_info["player_total_ore"] * EarlyRewardParam.ore_reward_weight
            all_past_reward += own_global_info["player_total_water"] * EarlyRewardParam.water_reward_weight
            all_past_reward += own_global_info["player_total_metal"] * EarlyRewardParam.metal_reward_weight
            all_past_reward += own_global_info["player_total_power"] * EarlyRewardParam.power_reward_weight
            all_past_reward += own_global_info["player_factory_count"] * EarlyRewardParam.factory_penalty_weight
            all_past_reward += game_state.env_steps * EarlyRewardParam.survive_reward_weight

        rewards = sum(own_sub_rewards.values())

        self.update_last_count(own_global_info)
        self.update_env_stats(env_stats)

        self.logger.debug(f"Reward: {rewards}, Past reward: {all_past_reward}, Ice increment: {ice_increment}")

        return rewards + all_past_reward
    
    def get_global_info(self, player: str, obs: GameState):

        global_info = {k: None for k in global_information_names}

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

        if (self.step_count % 100) == 0:
            self.logger.debug(f"Step {self.step_count}, Unit ice: {global_info['player_unit_ice']}, Factory ice: {global_info['player_factory_ice']}")

        lichen = obs.board.lichen
        lichen_strains = obs.board.lichen_strains
        if factories:
            if lichen.any():
                factories_with_lichen = [f for f in factories if hasattr(f, "strain_id")]
                lichen_count = sum((np.sum(lichen[lichen_strains == factory.strain_id]) for factory in factories_with_lichen), 0)
                global_info['lichen_count'] = lichen_count
        else:
            global_info['lichen_count'] = 0
        return global_info

    def update_last_count(self, global_info):
        self.last_count = deepcopy(global_info)

    def update_env_stats(self, env_stats):
        self.last_env_stats = deepcopy(env_stats)