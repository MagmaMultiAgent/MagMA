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

            # for factory_name, factory in own_factory_info.items():
            #     factory_reward = global_reward[team]

            #     factory_id = int(factory_name.split("_")[1])
            #     final_reward[team][factory_id] += factory_reward

            unit_rewards = 0
            for unit_name, unit in own_unit_info.items():
                unit_reward = 0

                cargo_ice = unit["cargo_ice"]
                last_cargo_ice = last_count_units.get(unit_name, {}).get("cargo_ice", 0)
                ice_increment = max(cargo_ice - last_cargo_ice, 0)
                unit_reward += ice_increment

                unit_id = int(unit_name.split("_")[1]) + 10
                unit_rewards += unit_reward

                # if on ice
                if unit["ice_under"]:
                    # unit_reward += 0.01

                    # if the pos as last time
                    if unit["x"] == last_count_units.get(unit_name, {}).get("x") and unit["y"] == last_count_units.get(unit_name, {}).get("y"):
                        # get rubble under
                        rubble_under_decrease = max(last_count_units.get(unit_name, {}).get("rubble_under", 0) - unit["rubble_under"], 0)
                        unit_reward += rubble_under_decrease / 10

                final_reward[team][unit_id] += unit_reward

        _, sub_rewards = super(IceRewardParser, self).parse(dones, game_state, env_stats, global_info)

        self.update_last_count(global_info)

        for i, r in enumerate(global_reward):
            final_reward[i][0] = r

        return final_reward, sub_rewards
    
    def reset(self, game_state, global_info, env_stats):
        self.update_last_count(global_info)

    def update_last_count(self, global_info):
        self.last_count = deepcopy(global_info)
