from kit.kit import obs_to_game_state, GameState, EnvConfig
from kit.utils import direction_to, my_turn_to_place_factory
import numpy as np
from scipy import ndimage
import scipy
import sys


class Player():

    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.env_cfg: EnvConfig = env_cfg

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        k = 200
        ice_nearby_log_weight = 30.0
        ice_log_weight = 0.1
        ore_log_weight = 0.01
        rubble_weight = 0.1
        occupancy_weight = 1.0
        sigma = 3

        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            bid_choice = [-2, 0, 2]
            bid_num = np.random.choice(bid_choice)
            return dict(faction="AlphaStrike", bid=bid_num)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            H, W = obs['board']['ice'].shape
            real_env_steps = game_state.real_env_steps
            # factory placement period
            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            factories_placed = (game_state.board.factory_occupancy_map == int(self.player[len('player_'):])).sum() // 9
            total_factories = factories_to_place + factories_placed

            water_total = (water_left * (total_factories // factories_to_place)) if factories_to_place else 0
            metal_total = (metal_left * (total_factories // factories_to_place)) if factories_to_place else 0
            water_per_Factory = (water_total // factories_to_place) if factories_to_place else 0
            metal_per_Factory = (metal_total // factories_to_place) if factories_to_place else 0

            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
                valid_spawns_mask = game_state.board.valid_spawns_mask
                
                ice_board = np.array(game_state.board.ice, dtype=np.int32)
                nearby_ice_count = np.zeros((H, W), dtype=ice_board.dtype)
                # shift ice board by 2 for each 4 directions (keep shape the same)
                ice_board_up = np.concatenate([np.zeros((2, W), dtype=ice_board.dtype), ice_board[:-2, :]], axis=0)
                ice_board_down = np.concatenate([ice_board[2:, :], np.zeros((2, W), dtype=ice_board.dtype)], axis=0)
                ice_board_left = np.concatenate([np.zeros((H, 2), dtype=ice_board.dtype), ice_board[:, :-2]], axis=1)
                ice_board_right = np.concatenate([ice_board[:, 2:], np.zeros((H, 2), dtype=ice_board.dtype)], axis=1)
                for ice_board_tmp in [ice_board_up, ice_board_down]:
                    # shift left and right
                    ice_board_tmp_left = np.concatenate([np.zeros((H, 1), dtype=ice_board.dtype), ice_board_tmp[:, :-1]], axis=1)
                    ice_board_tmp_right = np.concatenate([ice_board_tmp[:, 1:], np.zeros((H, 1), dtype=ice_board.dtype)], axis=1)
                    nearby_ice_count = nearby_ice_count + ice_board_tmp + ice_board_tmp_left + ice_board_tmp_right
                for ice_board_tmp in [ice_board_left, ice_board_right]:
                    # shift up and down
                    ice_board_tmp_up = np.concatenate([np.zeros((1, W), dtype=ice_board.dtype), ice_board_tmp[:-1, :]], axis=0)
                    ice_board_tmp_down = np.concatenate([ice_board_tmp[1:, :], np.zeros((1, W), dtype=ice_board.dtype)], axis=0)
                    nearby_ice_count = nearby_ice_count + ice_board_tmp + ice_board_tmp_up + ice_board_tmp_down
                
                orig_valid_spawns_mask = valid_spawns_mask
                valid_spawns_mask = valid_spawns_mask & (nearby_ice_count > 0)

                if np.sum(valid_spawns_mask) == 0:
                    print("No valid spawns, using original valid spawns mask.", file=sys.stderr)
                    valid_spawns_mask = orig_valid_spawns_mask

                kernal = np.ones((9, 9))

                rubble_0 = (game_state.board.rubble == 0).astype(np.int32)
                rubble = game_state.board.rubble

                # yapf: disable
                center_weight = ndimage.gaussian_filter(np.array([[1.]], dtype=np.float32), sigma=sigma, mode='constant', cval=0.0)
                ice_sum = ndimage.gaussian_filter(game_state.board.ice.astype(np.float32), sigma=sigma, mode='constant', cval=0.0) / center_weight
                ore_sum = ndimage.gaussian_filter(game_state.board.ore.astype(np.float32), sigma=sigma, mode='constant', cval=0.0) / center_weight
                rubble_sum = ndimage.gaussian_filter(rubble.astype(np.float32), sigma=sigma, mode='constant', cval=0.0) / center_weight
                factory_occupancy_map = game_state.board.factory_occupancy_map
                factory_occupancy_map = factory_occupancy_map == int(self.player[len('player_'):])
                factory_occupancy_map = ndimage.gaussian_filter(factory_occupancy_map.astype(np.float32), sigma=sigma, mode='constant', cval=0.0) / center_weight
                nearby_ice_count = ndimage.gaussian_filter((nearby_ice_count*2 + 1).astype(np.float32), sigma=sigma, mode='constant', cval=0.0) / center_weight
                # yapf: enable

                ice_sum = np.minimum(ice_sum, 3)
                ore_sum = np.minimum(ore_sum, 3)
                factory_occupancy_map = np.minimum(factory_occupancy_map, 3)

                score = sum([
                    np.log(nearby_ice_count) * ice_nearby_log_weight,
                    np.log(ice_sum + 0.2) * ice_log_weight,
                    np.log(ore_sum + 0.2) * ore_log_weight,
                    -np.log(rubble_sum + 1) * rubble_weight,
                    -np.log(factory_occupancy_map + 0.2) * occupancy_weight,
                    np.log(valid_spawns_mask + np.finfo(np.float64).tiny),
                ])
                # get top k scores and coordinates
                topk_idx = np.argsort(score.flat)[-k:]
                topk_score = score.flat[topk_idx]
                pi = scipy.special.softmax(topk_score)
                idx = np.random.choice(topk_idx, p=pi)
                spawn_loc = [idx // W, idx % W]
                while True:
                    i, j = spawn_loc
                    cur_score = score[i, j]
                    max_score = score[i, j]
                    for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                        if 0 <= i + di < H and 0 <= j + dj < W and score[i + di, j + dj] > max_score:
                            max_score = score[i + di, j + dj]
                            spawn_loc = [i + di, j + dj]
                    if max_score == cur_score:
                        break

                if game_state.teams[self.player].place_first and real_env_steps == -2:
                    return dict(spawn=spawn_loc, metal=metal_left, water=water_left)
                elif ~game_state.teams[self.player].place_first and real_env_steps == -1:
                    return dict(spawn=spawn_loc, metal=metal_left, water=water_left)
                else:
                    metal_num = metal_per_Factory
                    water_num = water_per_Factory
                    return dict(spawn=spawn_loc, metal=metal_num, water=water_num)
        return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        return actions