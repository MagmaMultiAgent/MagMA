from .dense_reward_parser import DenseRewardParser

import numpy as np

from copy import deepcopy
import sys

class IceRewardParser(DenseRewardParser):
    def __init__(self):
        super(IceRewardParser, self).__init__()
        self.last_count = None

    def parse(self, dones, game_state, env_stats, global_info = None):



        final_reward = 0
        reward_scale = 0.01
        ice_norm = 1
        step_weight_early = 1 + ((1000 - game_state.real_env_steps) / 1000) * 0.1


        if self.last_count is not None and global_info is not None:
            ice_increment = (global_info['total_ice'] - self.last_count['total_ice']) / 4
            ice_decrement = (self.last_count['total_ice'] - global_info['total_ice']) / 4

            ice_mined = (global_info['ice_mined'] - self.last_count['ice_mined']) / 4 # total ice
            ice_transfered = (global_info['ice_transfered'] - self.last_count['ice_transfered']) / 4 # unit ice

            ice_mined_reward = ice_mined * reward_scale / ice_norm * step_weight_early
            ice_transfered_reward = ice_transfered * reward_scale / ice_norm * step_weight_early

            ice_increment_reward = ice_increment * reward_scale / ice_norm * step_weight_early
            ice_decrement_reward = ice_decrement / ice_norm * step_weight_early


            water_increment = global_info['total_water'] - self.last_count['total_water']
            water_increment_reward = water_increment * reward_scale / 4 * step_weight_early

            final_reward = ice_mined_reward + ice_transfered_reward + ice_increment_reward + ice_decrement_reward + water_increment_reward
        
        self.update_last_count(global_info)

        return final_reward
    
    def reset(self, game_state, global_info, env_stats):
        self.update_last_count(global_info)

    def update_last_count(self, global_info):
        self.last_count = deepcopy(global_info)