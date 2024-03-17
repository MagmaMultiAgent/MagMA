from .dense_reward_parser import DenseRewardParser

import numpy as np

from copy import deepcopy
import sys


class IceRewardParser(DenseRewardParser):
    def parse(self, dones, game_state, env_stats, global_info):
        global_reward = [0.0, 0.0]

        final_reward = [np.zeros((1000,), dtype=np.float32) for _ in range(2)]

        for team in [0, 1]:
            player = f"player_{team}"
            own_global_info = global_info[player]
            own_unit_info = own_global_info["units"]
            own_factory_info = own_global_info["factories"]

            last_count = self.last_count[player]
            last_count_units = last_count["units"]
            last_count_factories = last_count["factories"]

            unit_count = own_global_info["unit_count"]

            # avg_distance_from_ice = own_global_info["avg_distance_from_ice"]
            # if unit_count > 0:
            #     reward[team] -= avg_distance_from_ice / 1000
            # else:
            #     reward[team] -= 0.001

            units_on_ice = own_global_info["units_on_ice"]
            if unit_count > 0:
                global_reward[team] += units_on_ice / 1000
            else:
                pass  # 0 reward

            # power_increment = (own_global_info["unit_power"] - last_count['unit_power']) / 50 / 200
            # reward[team] += power_increment

            rubble_on_ice_decrease = (last_count['rubble_on_ice'] - own_global_info["rubble_on_ice"]) / 100  # divide by 100 because max 100 rubble on tile
            global_reward[team] += max(rubble_on_ice_decrease, 0)

            ice_increment = own_global_info["total_ice"] - last_count['total_ice']
            global_reward[team] += max(ice_increment, 0)

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

                unit_reward += ice_increment * 0.1
                unit_reward += ice_decrement

                group_id = unit["group_id"]
                if group_id not in unit_groups:
                    unit_groups[group_id] = 0
                unit_groups[group_id] += unit_reward

            if len(unit_groups) > 0:
                total_reward = sum(unit_groups.values()) / len(unit_groups)
            else:
                total_reward = 0

            for group_id, group_reward in unit_groups.items():
                final_reward[team][group_id] += (group_reward * own_reward_weight) + (total_reward * (1 - own_reward_weight))
            
        _, sub_rewards = super(IceRewardParser, self).parse(dones, game_state, env_stats, global_info)

        self.update_last_count(global_info)

        return final_reward, sub_rewards
    
    def reset(self, game_state, global_info, env_stats):
        self.update_last_count(global_info)

    def update_last_count(self, global_info):
        self.last_count = deepcopy(global_info)
