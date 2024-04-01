from .dense_reward_parser import DenseRewardParser

import numpy as np

from copy import deepcopy
import sys


class IceRewardParser(DenseRewardParser):
    def parse(self, dones, game_state, env_stats, global_info):
        final_reward = [np.zeros((1000,), dtype=np.float32) for _ in range(2)]

        step_weight = game_state[0].real_env_steps / 1000

        ice_transfer_out_factor = (1/4) * 0.5
        ice_transfer_in_factor = (1/4) * 0.5

        survival_reward = 0.01
        unit_death_reward = -0.01
        factory_death_reward = -0.2

        for team in [0, 1]:
            player = f"player_{team}"
            own_global_info = global_info[player]
            own_unit_info = own_global_info["units"]
            own_factory_info = own_global_info["factories"]

            last_count = self.last_count[player]
            last_count_units = last_count["units"]
            last_count_factories = last_count["factories"]

            unit_count = own_global_info["unit_count"]

            global_reward = 0
            # check for destroyed factories
            for factory_name, factory in own_factory_info.items():
                if factory_name not in last_count_factories:
                    global_reward += factory_death_reward

            own_reward_weight = 1.0
            unit_groups = {}
            for unit_name, unit in own_unit_info.items():
                unit_reward = 0

                if unit_name not in last_count_units:
                    continue

                cargo_ice = unit["cargo_ice"]
                last_cargo_ice = last_count_units[unit_name]['cargo_ice']

                ice_increment = max(cargo_ice - last_cargo_ice, 0)
                ice_decrement = max(last_cargo_ice - cargo_ice, 0)  # transfer to factory

                unit_reward += ice_decrement * ice_transfer_out_factor

                unit_reward += survival_reward
                unit_reward += global_reward

                group_id = unit["group_id"]
                if group_id not in unit_groups:
                    unit_groups[group_id] = 0
                unit_groups[group_id] += unit_reward

            for unit_name, unit in last_count_units.items():
                if unit_name not in own_unit_info:
                    unit_reward = 0
                    unit_reward += unit_death_reward
                    unit_reward += global_reward

                    group_id = unit["group_id"]
                    if group_id not in unit_groups:
                        unit_groups[group_id] = 0
                    unit_groups[group_id] += unit_reward

            for factory_name, factory in own_factory_info.items():
                factory_reward = 0

                if factory_name not in last_count_factories:
                    continue

                # lichen_count = factory["lichen_count"]
                # lichen_reward = lichen_count / 20
                # lichen_reward *= 0.1
                # lichen_reward *= step_weight
                # factory_reward += lichen_reward

                cargo_ice = factory["cargo_ice"]
                last_cargo_ice = last_count_factories[factory_name]['cargo_ice']
                ice_increment = max(cargo_ice - last_cargo_ice, 0)

                factory_reward += ice_increment * ice_transfer_in_factor
                
                factory_reward += survival_reward
                factory_reward += global_reward

                group_id = factory["group_id"]
                if group_id not in unit_groups:
                    unit_groups[group_id] = 0
                unit_groups[group_id] += factory_reward

            for factory_name, factory in last_count_factories.items():
                if factory_name not in own_factory_info:
                    factory_reward = 0
                    factory_reward += global_reward

                    group_id = factory["group_id"]
                    if group_id not in unit_groups:
                        unit_groups[group_id] = 0
                    unit_groups[group_id] += factory_reward

            global_rev = 0
            if len(unit_groups) > 0:
                total_reward = sum(unit_groups.values()) / len(unit_groups)
                global_rev /= len(unit_groups)
            else:
                total_reward = 0

            for group_id, group_reward in unit_groups.items():
                final_reward[team][group_id] += (group_reward * own_reward_weight) + (total_reward * (1 - own_reward_weight)) + global_rev
            
        _, sub_rewards = super(IceRewardParser, self).parse(dones, game_state, env_stats, global_info)

        self.update_last_count(global_info)

        return final_reward, sub_rewards
    
    def reset(self, game_state, global_info, env_stats):
        self.update_last_count(global_info)

    def update_last_count(self, global_info):
        self.last_count = deepcopy(global_info)
