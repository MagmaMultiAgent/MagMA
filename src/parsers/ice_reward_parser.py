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

            avg_distance_from_ice = own_global_info["avg_distance_from_ice"]
            unit_count = own_global_info["unit_count"]
            if unit_count > 0:
                reward[team] -= avg_distance_from_ice / 10
            else:
                reward[team] -= 1

            ice_increment = own_global_info["total_ice"] - last_count['total_ice']
            reward[team] += max(ice_increment, 0)

        _, sub_rewards = super(IceRewardParser, self).parse(dones, game_state, env_stats, global_info)

        self.update_last_count(global_info)

        return reward, sub_rewards
    
    def reset(self, game_state, global_info, env_stats):
        self.update_last_count(global_info)

    def update_last_count(self, global_info):
        self.last_count = deepcopy(global_info)
