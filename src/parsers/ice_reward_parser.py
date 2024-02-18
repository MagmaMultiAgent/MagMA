from .dense_reward_parser import DenseRewardParser

import sys


class IceRewardParser(DenseRewardParser):
    def parse(self, dones, game_state, env_stats, global_info):
        _, sub_rewards = super(IceRewardParser, self).parse(dones, game_state, env_stats, global_info)

        reward = [0.0, 0.0]
        for team in [0, 1]:
            player = f"player_{team}"
            own_global_info = global_info[player]
            last_count = self.last_count[player]

            ice_increment = own_global_info["total_ice"] - last_count['total_ice']
            if ice_increment > 0:
                print("ice increment is ", ice_increment, file=sys.stderr)
            reward[team] += max(ice_increment, 0)

        return reward, sub_rewards
