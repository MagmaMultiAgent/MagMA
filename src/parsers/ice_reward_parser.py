from .dense_reward_parser import DenseRewardParser

from copy import deepcopy
import sys


class IceRewardParser(DenseRewardParser):
    def parse(self, dones, game_state, env_stats, global_info):
        reward = [0.0, 0.0]
        for team in [0, 1]:
            player = f"player_{team}"
            own_global_info = global_info[player]
            last_count = self.last_count[player]

            unit_count = own_global_info["unit_count"]

            # avg_distance_from_ice = own_global_info["avg_distance_from_ice"]
            # if unit_count > 0:
            #      reward[team] -= avg_distance_from_ice * unit_count / 10 / 200
            # else:
            #     reward[team] -= 0.01

            # units_on_ice = own_global_info["units_on_ice"]
            # if unit_count > 0:
            #     reward[team] += units_on_ice / 10 / 200
            # else:
            #     pass  # 0 reward

            # power_increment = (own_global_info["unit_power"] - last_count['unit_power']) / 50 / 200
            # reward[team] += power_increment

            rubble_on_ice_decrease = (last_count['rubble_on_ice'] - own_global_info["rubble_on_ice"]) / 100  # divide by 100 because max 100 rubble on tile
            reward[team] += max(rubble_on_ice_decrease, 0)

            ice_increment = own_global_info["total_ice"] - last_count['total_ice']
            reward[team] += max(ice_increment, 0)

        _, sub_rewards = super(IceRewardParser, self).parse(dones, game_state, env_stats, global_info)

        self.update_last_count(global_info)

        return reward, sub_rewards
    
    def reset(self, game_state, global_info, env_stats):
        self.update_last_count(global_info)

    def update_last_count(self, global_info):
        self.last_count = deepcopy(global_info)
