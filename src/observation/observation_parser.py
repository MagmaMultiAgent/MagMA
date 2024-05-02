import numpy as np
import tree
import kit.kit
from typing import NamedTuple
from luxai_s2.config import EnvConfig
from functools import reduce


class FeatureParser():

    def __init__(self):

        self.last_game_states = {
            "player_0": None,
            "player_1": None
        }

        self.global_feature_names = [
            'env_step',
            # 'cycle',
            # 'hour',
            'daytime_or_night',
            # 'num_factory_own',
            # 'num_factory_enm',
            # 'total_lichen_own',
            # 'total_lichen_enm',
        ]
        # for own_enm in ['own', 'enm']:
        #     self.global_feature_names += [
        #         f'factory_total_power_{own_enm}',
        #         f'factory_total_ice_{own_enm}',
        #         f'factory_total_water_{own_enm}',
        #         f'factory_total_ore_{own_enm}',
        #         f'factory_total_metal_{own_enm}',
        #         f'num_light_{own_enm}',
        #         f'num_heavy_{own_enm}',
        #         f'robot_total_power_{own_enm}',
        #         f'robot_total_ice_{own_enm}',
        #         f'robot_total_water_{own_enm}',
        #         f'robot_total_ore_{own_enm}',
        #         f'robot_total_metal_{own_enm}',
        #     ]

        # self.entity_feature_names = [
        #     'factory',
        #     'unit',
        #     'light',
        #     'heavy',
        #     'ice',
        #     'ore',
        #     'rubble',
        #     'lichen',
        #     'closest_ice_direction.x',
        #     'closest_ice_direction.y',
        #     'closest_ore_direction.x',
        #     'closest_ore_direction.y',
        #     'closest_unit_direction.x',
        #     'closest_unit_direction.y',
        #     'closest_factory_direction.x',
        #     'closest_factory_direction.y',
        #     'cargo_ice',
        #     'cargo_ore',
        #     'cargo_water',
        #     'cargo_metal',
        #     'cargo_power',
        #     'lichen_strain'
        # ]

        self.factory_feature_names = [
            'factory_power',
            'factory_ice',
            'factory_water',
            'factory_ore',
            'factory_metal',
            'factory_water_cost',
        ]

        self.map_featrue_names = [
            'factory',
            'ice',
            'ore',
            'rubble',
            'unit',
            'enemy',
            # 'lichen',
            # 'lichen_strains',
            # 'lichen_strains_own',
            # 'lichen_strains_enm',
            # 'valid_region_indicator',
            # 'factory_id',
            # 'factory_power',
            # 'factory_ice',
            # 'factory_water',
            # 'factory_ore',
            # 'factory_metal',
            # 'factory_own',
            # 'factory_enm',
            # 'factory_can_build_light',
            # 'factory_can_build_heavy',
            # 'factory_can_grow_lichen',
            # 'factory_water_cost',
            # 'unit_id',
            # 'unit_power',
            # 'unit_ice',
            # 'unit_water',
            # 'unit_ore',
            # 'unit_metal',
            # 'unit_own',
            # 'unit_enm',
            # 'unit_light',
            # 'unit_heavy',
        ]

        self.location_feature_names = [
            "factory",
            "unit",
        ]

        self.unit_feature_names = [
            "heavy",
            "power",
            "cargo_ice",
            "cargo_ore",
        ]

        

    def parse(self, obs, env_cfg):
        global_features = {}
        map_features = {}
        factory_features = {}
        unit_features = {}

        for player, player_obs in obs.items():
            env_step = player_obs['real_env_steps'] + player_obs['board']['factories_per_team'] * 2 + 1
            game_state = kit.kit.obs_to_game_state(env_step, env_cfg, player_obs)

            global_feature, map_feature, factory_feature, unit_feature = self._get_feature(obs=game_state, player=player)
            global_features[player] = global_feature
            map_features[player] = map_feature
            factory_features[player] = factory_feature
            unit_features[player] = unit_feature

            self.last_game_states[player] = game_state

        return global_features, map_features, factory_features, unit_features


    def _get_feature(self, obs: kit.kit.GameState, player: str, output_dict=False):
        env_cfg: EnvConfig = obs.env_cfg

        other_player = "player_0" if player == "player_1" else "player_1"

        # normalize
        light_cfg = env_cfg.ROBOTS['LIGHT']
        heavy_cfg = env_cfg.ROBOTS['HEAVY']

        # Global

        global_feature = {name: 0 for name in self.global_feature_names}
        # normalize between -1 and 1
        global_feature['env_step'] = (obs.real_env_steps - 0) / (env_cfg.max_episode_length - 0) * 2 - 1
        # global_feature['cycle'] = obs.real_env_steps // env_cfg.CYCLE_LENGTH
        # global_feature['hour'] = obs.real_env_steps % env_cfg.CYCLE_LENGTH
        hour = obs.real_env_steps % env_cfg.CYCLE_LENGTH
        global_feature['daytime_or_night'] = hour < 30

        # global_feature['num_factory_own'] = (len(obs.factories[player]) - 0) / (env_cfg.MAX_FACTORIES - 0) * 2 - 1
        # global_feature['num_factory_enm'] = (len(obs.factories[other_player]) - 0) / (env_cfg.MAX_FACTORIES - 0) * 2 - 1

        # Map

        map_feature = {name: np.zeros_like(obs.board.ice, dtype=np.float32) for name in self.map_featrue_names}
        map_feature['ice'] = obs.board.ice
        map_feature['ore'] = obs.board.ore
        map_feature['rubble'] = (obs.board.rubble - 0) / (env_cfg.MAX_RUBBLE - 0) * 2 - 1

        # Factory

        factory_feature = {name: np.zeros_like(obs.board.ice, dtype=np.float32) for name in self.factory_feature_names}
        for owner, factories in obs.factories.items():
            for fid, factory in factories.items():
                x, y = factory.pos

                factory_feature['factory_power'][x, y] = (factory.power - 0) / (heavy_cfg.BATTERY_CAPACITY - 0) * 2 - 1
                factory_feature['factory_ice'][x, y] = (factory.cargo.ice - 0) / (heavy_cfg.CARGO_SPACE - 0) * 2 - 1
                factory_feature['factory_water'][x, y] = (factory.cargo.water - 0) / (heavy_cfg.CARGO_SPACE - 0) * 2 - 1
                factory_feature['factory_ore'][x, y] = (factory.cargo.ore - 0) / (heavy_cfg.CARGO_SPACE - 0) * 2 - 1
                factory_feature['factory_metal'][x, y] = (factory.cargo.metal - 0) / (heavy_cfg.CARGO_SPACE - 0) * 2 - 1

                water_cost = np.sum(obs.board.lichen_strains == factory.strain_id) // env_cfg.LICHEN_WATERING_COST_FACTOR + 1
                factory_feature['factory_water_cost'][x, y] = (water_cost - 0) / (heavy_cfg.CARGO_SPACE - 0) * 2 - 1

                if owner == player:
                    for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        dx, dy = offset
                        if 0 <= x + dx < obs.board.ice.shape[0] and 0 <= y + dy < obs.board.ice.shape[1]:
                            map_feature['factory'][x + dx, y + dy] = 1.0
                            pass

        # Unit

        unit_feature = {name: np.zeros_like(obs.board.ice, dtype=np.float32) for name in self.unit_feature_names}
        units = obs.units[player]
        
        for unit in units.values():
            x, y = unit.pos

            unit_type = unit.unit_type
            cargo_space = light_cfg.CARGO_SPACE if unit_type == 'LIGHT' else heavy_cfg.CARGO_SPACE
            battery_capacity = light_cfg.BATTERY_CAPACITY if unit_type == 'LIGHT' else heavy_cfg.BATTERY_CAPACITY
            unit_feature['heavy'][x, y] = unit_type == 'HEAVY'
            unit_feature['power'][x, y] = (unit.power - 0) / (battery_capacity - 0) * 2 - 1
            unit_feature['cargo_ice'][x, y] = (unit.cargo.ice - 0) / (cargo_space - 0) * 2 - 1
            unit_feature['cargo_ore'][x, y] = (unit.cargo.ore - 0) / (cargo_space - 0) * 2 - 1

            map_feature['unit'][x, y] = 1.0

        for enemy_unit in obs.units[other_player].values():
            x, y = enemy_unit.pos
            map_feature['enemy'][x, y] = 1.0
        for enemy_Factory in obs.factories[other_player].values():
            x, y = enemy_Factory.pos
            deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in deltas:
                map_feature['enemy'][x + dx, y + dy] = 1.0

        # Assemble return

        global_feature = np.array(list(global_feature.values()))
        map_feature = np.array(list(map_feature.values()))
        factory_feature = np.array(list(factory_feature.values()))
        unit_feature = np.array(list(unit_feature.values()))

        if output_dict:
            return {'global_feature': global_feature, 'map_feature': map_feature, 'factory_feature': factory_feature, 'unit_feature': unit_feature}
        
        feature_map_size = (48, 48)
        upscaled_features_global_features = np.tile(global_feature[:, np.newaxis, np.newaxis], (1, *feature_map_size))

        return upscaled_features_global_features, map_feature, factory_feature, unit_feature

    @staticmethod
    def get_unit_id(unit, factories):
        unit_id = int(unit.unit_id.split('_')[1])
        return unit_id + 10
    
    @staticmethod
    def get_factory_id(factory):
        factory_id = int(factory.unit_id.split('_')[1])
        return factory_id

    @staticmethod
    def cluster_board(board):
        board_int = board.astype(np.int32)
        board_up = np.roll(board_int, -1, axis=0)
        board_up[-1, :] = 0
        board_down = np.roll(board_int, 1, axis=0)
        board_down[0, :] = 0
        board_left = np.roll(board_int, -1, axis=1)
        board_left[:, -1] = 0
        board_right = np.roll(board_int, 1, axis=1)
        board_right[:, 0] = 0

        board_int_sum = board_int + board_up + board_down + board_left + board_right

        board_int_sum_up = np.roll(board_int_sum, -1, axis=0)
        board_int_sum_up[-1, :] = 0
        board_int_sum_down = np.roll(board_int_sum, 1, axis=0)
        board_int_sum_down[0, :] = 0
        board_int_sum_left = np.roll(board_int_sum, -1, axis=1)
        board_int_sum_left[:, -1] = 0
        board_int_sum_right = np.roll(board_int_sum, 1, axis=1)
        board_int_sum_right[:, 0] = 0

        cluster_center = board & (board_int_sum >= board_int_sum_up) & (board_int_sum >= board_int_sum_down) & (board_int_sum >= board_int_sum_left) & (board_int_sum >= board_int_sum_right)

        return cluster_center


    @staticmethod
    def get_best_directions(entities, targets, can_match=True):
        base_up = np.zeros(entities.shape, dtype=np.float32)
        base_down = np.zeros(entities.shape, dtype=np.float32)
        base_left = np.zeros(entities.shape, dtype=np.float32)
        base_right = np.zeros(entities.shape, dtype=np.float32)

        max_distance = entities.shape[0] + entities.shape[1]

        if not can_match:
            targets = targets & ~entities

        entity_coords = np.argwhere(entities)

        target_coords = np.argwhere(targets)

        if len(entity_coords) == 0 or len(target_coords) == 0:
            return base_up, base_down, base_left, base_right

        target_directions = -(entity_coords[:, None] - target_coords * 1.0)

        closest_target = np.abs(target_directions).sum(axis=-1).argmin(axis=-1)
        closest_target_direction = target_directions[np.arange(len(entity_coords)), closest_target]

        up_target_directions = target_directions.copy()
        up_target_directions[~(target_directions[..., 1] < 0)] = max_distance
        up_target_min_distance = np.abs(up_target_directions).sum(axis=-1).min(axis=-1)
        up_target_min_distance[up_target_min_distance == (max_distance * 2)] = 0
        base_up[entity_coords[:, 0], entity_coords[:, 1]] = up_target_min_distance

        down_target_directions = target_directions.copy()
        down_target_directions[~(target_directions[..., 1] > 0)] = max_distance
        down_target_min_distance = np.abs(down_target_directions).sum(axis=-1).min(axis=-1)
        down_target_min_distance[down_target_min_distance == (max_distance * 2)] = 0
        base_down[entity_coords[:, 0], entity_coords[:, 1]] = down_target_min_distance

        left_target_directions = target_directions.copy()
        left_target_directions[~(target_directions[..., 0] < 0)] = max_distance
        left_target_min_distance = np.abs(left_target_directions).sum(axis=-1).min(axis=-1)
        left_target_min_distance[left_target_min_distance == (max_distance * 2)] = 0
        base_left[entity_coords[:, 0], entity_coords[:, 1]] = left_target_min_distance

        right_target_directions = target_directions.copy()
        right_target_directions[~(target_directions[..., 0] > 0)] = max_distance
        right_target_min_distance = np.abs(right_target_directions).sum(axis=-1).min(axis=-1)
        right_target_min_distance[right_target_min_distance == (max_distance * 2)] = 0
        base_right[entity_coords[:, 0], entity_coords[:, 1]] = right_target_min_distance

        return base_up, base_down, base_left, base_right

    @staticmethod
    def get_closest_coords(entities, targets, can_match=True):
        base = np.zeros(entities.shape + (2,), dtype=np.float32)

        if not can_match:
            targets = targets & ~entities

        entity_coords = np.argwhere(entities)
        target_coords = np.argwhere(targets)

        if len(entity_coords) == 0 or len(target_coords) == 0:
            return base

        target_directions = -(entity_coords[:, None] - target_coords)
        closest_target = np.abs(target_directions).sum(axis=-1).argmin(axis=-1)
        closest_target_direction = target_directions[np.arange(len(entity_coords)), closest_target]

        base[entity_coords[:, 0], entity_coords[:, 1]] = closest_target_direction
        return base

    @staticmethod
    def get_distance(entities, targets):
        base = np.zeros(entities.shape, dtype=np.float32)

        entity_coords = np.argwhere(entities)
        target_coords = np.argwhere(targets)

        if len(entity_coords) == 0 or len(target_coords) == 0:
            return base

        target_directions = -(entity_coords[:, None] - target_coords)
        closest_target_distance = np.abs(target_directions).sum(axis=-1).min(axis=-1) / (entities.shape[0] + entities.shape[1])

        base[entity_coords[:, 0], entity_coords[:, 1]] = closest_target_distance
        return base
    
    @staticmethod
    def get_avg_distance(entities, targets):
        entity_coords = np.argwhere(entities)
        target_coords = np.argwhere(targets)

        if len(entity_coords) == 0 or len(target_coords) == 0:
            return 0

        target_directions = -(entity_coords[:, None] - target_coords)
        closest_target_distance = np.abs(target_directions).sum(axis=-1).min(axis=-1)
        avg_distance = closest_target_distance.mean() / (entities.shape[0] + entities.shape[1])
        
        return avg_distance

    @staticmethod
    def log_env_stats(env_stats):
        for team in [0, 1]:
            player = f"player_{team}"
            stat = tree.flatten_with_path(env_stats[player])
            stat = list(map(
                lambda item: {"_".join(item[0]).lower(): item[1]},
                stat,
            ))
            stat = reduce(lambda cat1, cat2: dict(cat1, **cat2), stat)
            env_stats_logs = stat
        return env_stats_logs