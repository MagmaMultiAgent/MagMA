import numpy as np
import tree

from impl_config import EnvParam
import kit.kit
from typing import NamedTuple
from luxai_s2.config import EnvConfig
from functools import reduce

import sys


class LuxFeature(NamedTuple):
    global_feature: np.ndarray
    map_feature: np.ndarray
    factory_feature: np.ndarray
    unit_feature: np.ndarray
    location_feature: np.ndarray


class FeatureParser():

    def __init__(self):

        self.global_feature_names = [
            'env_step',
            'cycle',
            'hour',
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
            'factory_water_cost'
        ]

        self.map_featrue_names = [
            'factory',
            'ice',
            # 'ore',
            'rubble',
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
            "unit"
        ]

        self.unit_feature_names = [
            "heavy",
            "power",
            "cargo_ice"
        ]

        self.global_info_names = [
            'factory_count',
            'unit_count',
            'light_count',
            'heavy_count',
            'unit_ice',
            'unit_ore',
            'unit_water',
            'unit_metal',
            'unit_power',
            'factory_ice',
            'factory_ore',
            'factory_water',
            'factory_metal',
            'factory_power',
            'total_ice',
            'total_ore',
            'total_water',
            'total_metal',
            'total_power',
            'lichen_count',
            'units_on_ice',
            'avg_distance_from_ice',
            'rubble_on_ice'
        ]
        self.global_info_dicts = {
            'units': [
                'heavy',
                'power',
                'cargo_ice',
                'cargo_ore',
                'cargo_water',
                'cargo_metal',
                'rubble_under',
                'ice_under',
                'x',
                'y',
                'group_id'
            ],
            'factories': [
                'power',
                'cargo_ice',
                'cargo_ore',
                'cargo_water',
                'cargo_metal',
                'water_cost'
            ]
        }

    def parse(self, obs, env_cfg):
        all_feature = {}
        for player, player_obs in obs.items():
            env_step = player_obs['real_env_steps'] + player_obs['board']['factories_per_team'] * 2 + 1
            game_state = kit.kit.obs_to_game_state(env_step, env_cfg, player_obs)
            parsed_feature = self._get_feature(obs=game_state, player=player)
            # all_feature.append(parsed_feature)
            all_feature[player] = parsed_feature
        global_info = {player: self._get_info(player, game_state) for player in ['player_0', 'player_1']}
        return all_feature, global_info
    
    def json_parser(self, obs, env_cfg):
        all_feature = {}
        for player, player_obs in obs.items():
            env_step = player_obs['real_env_steps'] + player_obs['board']['factories_per_team'] * 2 + 1
            game_state = kit.kit.json_obs_to_game_state(env_step, env_cfg, player_obs)
            parsed_feature = self._get_feature(obs=game_state, player=player)
            # all_feature.append(parsed_feature)
            all_feature[player] = parsed_feature
        return all_feature

    def parse2(self, game_state, player):
        return self._get_feature(obs=game_state, player=player)

    def _get_info(self, player: str, obs: kit.kit.GameState):
        global_info = {k: 0 for k in self.global_info_names}
        factories = list(obs.factories[player].values())
        units = list(obs.units[player].values())

        unit_info = {}
        factory_info = {}

        for unit in units:
            unit_info[unit.unit_id] = {
                'heavy': int(unit.unit_type == 'HEAVY'),
                'power': unit.power,
                'cargo_ice': unit.cargo.ice,
                'cargo_ore': unit.cargo.ore,
                'cargo_water': unit.cargo.water,
                'cargo_metal': unit.cargo.metal,
                'rubble_under': obs.board.rubble[unit.pos[0], unit.pos[1]],
                'ice_under': obs.board.ice[unit.pos[0], unit.pos[1]],
                'x': unit.pos[0],
                'y': unit.pos[1],
                'group_id': self.get_unit_id(unit)
            }
        for factory in factories:
            factory_info[factory.unit_id] = {
                'power': factory.power,
                'cargo_ice': factory.cargo.ice,
                'cargo_ore': factory.cargo.ore,
                'cargo_water': factory.cargo.water,
                'cargo_metal': factory.cargo.metal,
                'water_cost': np.sum(obs.board.lichen_strains == factory.strain_id) // obs.env_cfg.LICHEN_WATERING_COST_FACTOR + 1
            }

        global_info['light_count'] = sum(int(u.unit_type == 'LIGHT') for u in units)
        global_info['heavy_count'] = sum(int(u.unit_type == 'HEAVY') for u in units)
        global_info['unit_count'] = global_info['light_count'] + global_info['heavy_count']
        global_info["factory_count"] = len(factories)

        global_info['unit_ice'] = sum(u.cargo.ice for u in units)
        global_info['unit_ore'] = sum(u.cargo.ore for u in units)
        global_info['unit_water'] = sum(u.cargo.water for u in units)
        global_info['unit_metal'] = sum(u.cargo.metal for u in units)
        global_info['unit_power'] = sum(u.power for u in units)

        global_info['factory_ice'] = sum(f.cargo.ice for f in factories)
        global_info['factory_ore'] = sum(f.cargo.ore for f in factories)
        global_info['factory_water'] = sum(f.cargo.water for f in factories)
        global_info['factory_metal'] = sum(f.cargo.metal for f in factories)
        global_info['factory_power'] = sum(f.power for f in factories)

        global_info['total_ice'] = global_info['unit_ice'] + global_info['factory_ice']
        global_info['total_ore'] = global_info['unit_ore'] + global_info['factory_ore']
        global_info['total_water'] = global_info['unit_water'] + global_info['factory_water']
        global_info['total_metal'] = global_info['unit_metal'] + global_info['factory_metal']
        global_info['total_power'] = global_info['unit_power'] + global_info['factory_power']

        units_positions = [u.pos for u in units]
        unit_board = np.zeros_like(obs.board.ice, dtype=np.bool8)
        for pos in units_positions:
            unit_board[pos[0], pos[1]] = True
        ice_board = obs.board.ice
        units_standing_on_ice = [(ice_board[pos[0], pos[1]] > 0) for pos in units_positions]
        unit_count_on_ice = sum(units_standing_on_ice)
        global_info['units_on_ice'] = unit_count_on_ice

        # get avg distance from ice
        if len(units_positions) < 0:
            avg_distance_from_ice = 1.0
        else:
            avg_distance_from_ice = self.get_avg_distance(unit_board, ice_board)
        global_info['avg_distance_from_ice'] = avg_distance_from_ice

        lichen = obs.board.lichen
        lichen_strains = obs.board.lichen_strains
        if factories:
            lichen_count = sum((np.sum(lichen[lichen_strains == f.strain_id]) for f in factories), 0)
            global_info['lichen_count'] = lichen_count
        else:
            global_info['lichen_count'] = 0

        # rubble on ice
        ice_board = obs.board.ice.astype(np.float32)
        rubble_board = obs.board.rubble.astype(np.float32)
        rubble_on_ice = ice_board * rubble_board
        global_info['rubble_on_ice'] = np.sum(rubble_on_ice)

        # Add unit and factory info

        global_info['units'] = unit_info
        global_info['factories'] = factory_info

        return global_info

    def _get_feature(self, obs: kit.kit.GameState, player: str, output_dict=True):
        env_cfg: EnvConfig = obs.env_cfg

        # normalize
        light_cfg = env_cfg.ROBOTS['LIGHT']
        heavy_cfg = env_cfg.ROBOTS['HEAVY']

        # Location

        location_feature = {name: np.zeros_like(obs.board.ice, dtype=np.int32) for name in self.location_feature_names}
        location_feature['factory'][:] = -1
        location_feature['unit'][:] = -1

        # Global

        global_feature = {name: 0 for name in self.global_feature_names}
        global_feature['env_step'] = obs.real_env_steps
        global_feature['cycle'] = obs.real_env_steps // env_cfg.CYCLE_LENGTH
        global_feature['hour'] = obs.real_env_steps % env_cfg.CYCLE_LENGTH
        global_feature['daytime_or_night'] = global_feature['hour'] < 30

        # Map

        map_feature = {name: np.zeros_like(obs.board.ice, dtype=np.float32) for name in self.map_featrue_names}
        map_feature['ice'] = obs.board.ice
        map_feature['rubble'] = obs.board.rubble

        # Factory

        factory_feature = {name: np.zeros_like(obs.board.ice, dtype=np.float32) for name in self.factory_feature_names}
        for owner, factories in obs.factories.items():
            for fid, factory in factories.items():
                x, y = factory.pos
                location_feature['factory'][x, y] = int(factory.unit_id.split('_')[1])

                factory_feature['factory_power'][x, y] = factory.power
                factory_feature['factory_ice'][x, y] = factory.cargo.ice
                factory_feature['factory_water'][x, y] = factory.cargo.water
                factory_feature['factory_ore'][x, y] = factory.cargo.ore
                factory_feature['factory_metal'][x, y] = factory.cargo.metal

                water_cost = np.sum(
                    obs.board.lichen_strains == factory.strain_id) // env_cfg.LICHEN_WATERING_COST_FACTOR + 1
                factory_feature['factory_water_cost'][x, y] = water_cost

                if owner == player:
                    for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        dx, dy = offset
                        if 0 <= x + dx < obs.board.ice.shape[0] and 0 <= y + dy < obs.board.ice.shape[1]:
                            map_feature['factory'][x + dx, y + dy] = 1.0
                            pass

        factory_feature['factory_power'] = factory_feature['factory_power'] / heavy_cfg.BATTERY_CAPACITY
        factory_feature['factory_ice'] = factory_feature['factory_ice'] / heavy_cfg.CARGO_SPACE
        factory_feature['factory_water'] = factory_feature['factory_water'] / heavy_cfg.CARGO_SPACE
        factory_feature['factory_ore'] = factory_feature['factory_ore'] / heavy_cfg.CARGO_SPACE
        factory_feature['factory_metal'] = factory_feature['factory_metal'] / heavy_cfg.CARGO_SPACE
        factory_feature['factory_water_cost'] = factory_feature['factory_water_cost'] / heavy_cfg.CARGO_SPACE

        # Unit

        unit_feature = {name: np.zeros_like(obs.board.ice, dtype=np.float32) for name in self.unit_feature_names}
        units = obs.units[player]
        
        for unit in units.values():
            x, y = unit.pos

            unit_group_id = self.get_unit_id(unit)
            location_feature['unit'][x, y] = unit_group_id

            unit_type = unit.unit_type
            cargo_space = light_cfg.CARGO_SPACE if unit_type == 'LIGHT' else heavy_cfg.CARGO_SPACE
            battery_capacity = light_cfg.BATTERY_CAPACITY if unit_type == 'LIGHT' else heavy_cfg.BATTERY_CAPACITY
            unit_feature['heavy'][x, y] = unit_type == 'HEAVY'
            unit_feature['power'][x, y] = unit.power / battery_capacity
            unit_feature['cargo_ice'][x, y] = unit.cargo.ice / cargo_space

        # Assemble return

        global_feature = np.array(list(global_feature.values()))
        map_feature = np.array(list(map_feature.values()))
        factory_feature = np.array(list(factory_feature.values()))
        unit_feature = np.array(list(unit_feature.values()))
        location_feature = np.array(list(location_feature.values()))

        if output_dict:
            return {'global_feature': global_feature, 'map_feature': map_feature, 'factory_feature': factory_feature, 'unit_feature': unit_feature, 'location_feature': location_feature}
        
        return LuxFeature(global_feature, map_feature, factory_feature, unit_feature, location_feature)

    @staticmethod
    def get_unit_id(unit):
        unit_id = int(unit.unit_id.split('_')[1])
        x = unit.pos[0]
        y = unit.pos[1]
        # 4 groups
        group_id = (x % 2) + (y % 2) * 2
        return group_id

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