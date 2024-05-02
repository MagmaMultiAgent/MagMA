import numpy as np
from kit.config import EnvConfig
from kit.kit import obs_to_game_state
from luxai_s2.unit import FactoryPlacementActionType, BidActionType
from luxai_s2.state import ObservationStateDict
from kit.utils import my_turn_to_place_factory
from scipy import ndimage

def bid_zero_to_not_waste(player: str, obs: ObservationStateDict) -> BidActionType:
    bid_num = 0
    return dict(bid=bid_num, faction="AlphaStrike")


def gaussian_ice_placement(player: str, step: int, env_cfg: EnvConfig, obs: ObservationStateDict,) -> FactoryPlacementActionType:

    ice_nearby_log_weight = 10.0
    ice_log_weight = 1.0
    ore_log_weight = 0.0001
    rubble_weight = 0.1 / 4
    occupancy_weight = 0.1
    sigma = 3


    game_state = obs_to_game_state(step, env_cfg, obs)
    H, W = obs['board']['ice'].shape
    real_env_steps = game_state.real_env_steps

    water_left = game_state.teams[player].water
    metal_left = game_state.teams[player].metal

    factories_to_place = game_state.teams[player].factories_to_place
    factories_placed = (game_state.board.factory_occupancy_map == int(player[len('player_'):])).sum() // 9
    total_factories = factories_to_place + factories_placed

    water_total = (water_left * (total_factories // factories_to_place)) if factories_to_place else 0
    metal_total = (metal_left * (total_factories // factories_to_place)) if factories_to_place else 0
    water_per_Factory = (water_total // factories_to_place) if factories_to_place else 0
    metal_per_Factory = (metal_total // factories_to_place) if factories_to_place else 0

    if factories_to_place > 0:
        # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
        valid_spawns_mask = game_state.board.valid_spawns_mask
        
        ice_board = np.array(game_state.board.ice, dtype=np.float32)
        nearby_ice_count = np.zeros((H, W), dtype=ice_board.dtype)
        nearby_ice_count2 = np.zeros((H, W), dtype=ice_board.dtype)
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

            ice_board_tmp_left = np.concatenate([np.zeros((H, 2), dtype=ice_board.dtype), ice_board_tmp[:, :-2]], axis=1)
            ice_board_tmp_right = np.concatenate([ice_board_tmp[:, 2:], np.zeros((H, 2), dtype=ice_board.dtype)], axis=1)
            nearby_ice_count2 = nearby_ice_count2 + ice_board_tmp + ice_board_tmp_left + ice_board_tmp_right
            
        for ice_board_tmp in [ice_board_left, ice_board_right]:
            # shift up and down
            ice_board_tmp_up = np.concatenate([np.zeros((1, W), dtype=ice_board.dtype), ice_board_tmp[:-1, :]], axis=0)
            ice_board_tmp_down = np.concatenate([ice_board_tmp[1:, :], np.zeros((1, W), dtype=ice_board.dtype)], axis=0)
            nearby_ice_count = nearby_ice_count + ice_board_tmp + ice_board_tmp_up + ice_board_tmp_down

            ice_board_tmp_up = np.concatenate([np.zeros((2, W), dtype=ice_board.dtype), ice_board_tmp[:-2, :]], axis=0)
            ice_board_tmp_down = np.concatenate([ice_board_tmp[2:, :], np.zeros((2, W), dtype=ice_board.dtype)], axis=0)
            nearby_ice_count2 =  nearby_ice_count2 + ice_board_tmp + ice_board_tmp_up + ice_board_tmp_down

        orig_valid_spawns_mask = valid_spawns_mask
        valid_spawns_mask = valid_spawns_mask & (nearby_ice_count > 0)
        valid_spawns_mask2 = valid_spawns_mask & (nearby_ice_count2 > 0)

        if np.sum(valid_spawns_mask) == 0:
            # print("No valid spawns, using original valid spawns mask.", file=sys.stderr)
            if np.sum(valid_spawns_mask2) == 0:
                valid_spawns_mask = orig_valid_spawns_mask
            else:
                valid_spawns_mask = valid_spawns_mask2

        kernal = np.ones((9, 9))

        rubble_0 = (game_state.board.rubble == 0).astype(np.int32)
        rubble = game_state.board.rubble

        # yapf: disable
        center_weight = ndimage.gaussian_filter(np.array([[1.]], dtype=np.float32), sigma=sigma, mode='constant', cval=0.0)
        ice_sum = ndimage.gaussian_filter((game_state.board.ice.astype(np.float32)), sigma=sigma, mode='constant', cval=0.0) / center_weight
        ore_sum = ndimage.gaussian_filter((game_state.board.ore.astype(np.float32)), sigma=sigma, mode='constant', cval=0.0) / center_weight
        rubble_sum = ndimage.gaussian_filter((rubble.astype(np.float32)), sigma=sigma, mode='constant', cval=0.0) / center_weight
        factory_occupancy_map = game_state.board.factory_occupancy_map
        factory_occupancy_map = factory_occupancy_map == int(player[len('player_'):])
        factory_occupancy_map = ndimage.gaussian_filter(factory_occupancy_map.astype(np.float32), sigma=sigma, mode='constant', cval=0.0) / center_weight
        # yapf: enable

        # supress scientici notation
        # np.set_printoptions(suppress=True)
        # print(ndimage.gaussian_filter(game_state.board.ice.astype(np.float32), sigma=1, mode='constant', cval=0.0) / center_weight, file=sys.stderr)

        ice_sum = np.minimum(ice_sum, 3)
        ore_sum = np.minimum(ore_sum, 3)
        factory_occupancy_map = np.minimum(factory_occupancy_map, 3)

        # print("Ice", np.log(nearby_ice_count + 0.001).mean() * ice_nearby_log_weight, file=sys.stderr)
        # print("ice", np.log(ice_sum + 0.001).mean() * ice_log_weight, file=sys.stderr)
        # print("ore", np.log(ore_sum + 0.001).mean() * ore_log_weight, file=sys.stderr)
        # print("rubble", -np.log(rubble_sum + 0.001).mean() * rubble_weight, file=sys.stderr)
        # print("factory", -np.log(factory_occupancy_map + 0.001).mean() * occupancy_weight, file=sys.stderr)
        # print("valid", np.log(valid_spawns_mask + np.finfo(np.float64).tiny).mean(), file=sys.stderr)

        score = sum([
            np.log(nearby_ice_count + 0.001) * ice_nearby_log_weight,
            np.log(ice_sum + 0.001) * ice_log_weight,
            np.log(ore_sum + 0.001) * ore_log_weight,
            -np.log(rubble_sum + 0.001) * rubble_weight,
            -np.log(factory_occupancy_map + 0.001) * occupancy_weight,
            np.log(valid_spawns_mask + np.finfo(np.float64).tiny) * 1000,
        ])

        spawn_loc = np.unravel_index(np.argmax(score), score.shape)
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

        if game_state.teams[player].place_first and real_env_steps == -2:
            return dict(spawn=spawn_loc, metal=metal_left, water=water_left)
        elif ~game_state.teams[player].place_first and real_env_steps == -1:
            return dict(spawn=spawn_loc, metal=metal_left, water=water_left)
        else:
            metal_num = metal_per_Factory
            water_num = water_per_Factory
            return dict(spawn=spawn_loc, metal=metal_num, water=water_num)
    return dict()
