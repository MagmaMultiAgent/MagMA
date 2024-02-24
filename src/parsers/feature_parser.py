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
    factory_feature: np.ndarray
    unit_feature: np.ndarray


class FeatureParser():

    def __init__(self):

        self.global_feature_names = [
            'env_step',
            'cycle',
            'hour',
            'daytime_or_night',
            'num_factory_own',
            'num_factory_enm',
            'total_lichen_own',
            'total_lichen_enm',
        ]
        for own_enm in ['own', 'enm']:
            self.global_feature_names += [
                f'factory_total_power_{own_enm}',
                f'factory_total_ice_{own_enm}',
                f'factory_total_water_{own_enm}',
                f'factory_total_ore_{own_enm}',
                f'factory_total_metal_{own_enm}',
                f'num_light_{own_enm}',
                f'num_heavy_{own_enm}',
                f'robot_total_power_{own_enm}',
                f'robot_total_ice_{own_enm}',
                f'robot_total_water_{own_enm}',
                f'robot_total_ore_{own_enm}',
                f'robot_total_metal_{own_enm}',
            ]

        self.map_featrue_names = [
            'ice',
            'ore',
            'rubble',
            'lichen',
            'lichen_strains',
            'lichen_strains_own',
            'lichen_strains_enm',
            'valid_region_indicator',
            'factory_id',
            'factory_power',
            'factory_ice',
            'factory_water',
            'factory_ore',
            'factory_metal',
            'factory_own',
            'factory_enm',
            'factory_can_build_light',
            'factory_can_build_heavy',
            'factory_can_grow_lichen',
            'factory_water_cost',
            'unit_id',
            'unit_power',
            'unit_ice',
            'unit_water',
            'unit_ore',
            'unit_metal',
            'unit_own',
            'unit_enm',
            'unit_light',
            'unit_heavy',
        ]

        self.entity_feature_names = [
            'factory',
            'unit',
            'light',
            'heavy',
            'ice',
            'ore',
            'rubble',
            'lichen',
            'closest_ice_direction.x',
            'closest_ice_direction.y',
            'closest_ore_direction.x',
            'closest_ore_direction.y',
            'closest_unit_direction.x',
            'closest_unit_direction.y',
            'closest_factory_direction.x',
            'closest_factory_direction.y',
            'cargo_ice',
            'cargo_ore',
            'cargo_water',
            'cargo_metal',
            'cargo_power',
            'lichen_strain'
        ]

        self.factory_feature_names = [
            'factory_power',
            'factory_ice',
            'factory_water',
            'factory_ore',
            'factory_metal',
            'factory_water_cost'
        ]

        self.unit_feature_names = [
            'factory',
            'light',
            'heavy',
            'ice',
            'power',
            'cargo_ice',
            'distance_from_ice',
            'cloest_ice_up',
            'cloest_ice_down',
            'cloest_ice_left',
            'cloest_ice_right',
            'ice_up',
            'ice_down',
            'ice_left',
            'ice_right'
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
            'avg_distance_from_ice'
        ]

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
        return global_info

    def _get_feature(self, obs: kit.kit.GameState, player: str, output_dict=True):
        env_cfg: EnvConfig = obs.env_cfg
        enemy = 'player_1' if player == 'player_0' else 'player_0'

        map_feature = {name: np.zeros_like(obs.board.ice, dtype=np.float32) for name in self.map_featrue_names}
        map_feature['ice'] = obs.board.ice
        map_feature['ore'] = obs.board.ore
        map_feature['rubble'] = obs.board.rubble
        map_feature['lichen'] = obs.board.lichen
        map_feature['lichen_strains'] = obs.board.lichen_strains
        map_feature['lichen_strains_own'] = sum(
            (obs.board.lichen_strains == f.strain_id for f in obs.factories[player].values()) if obs.factories else [],
            np.zeros_like(obs.board.lichen_strains, dtype=np.bool8),
        )
        map_feature['lichen_strains_enm'] = sum(
            (obs.board.lichen_strains == f.strain_id for f in obs.factories[enemy].values()) if obs.factories else [],
            np.zeros_like(obs.board.lichen_strains, dtype=np.bool8),
        )
        map_feature['valid_region_indicator'] = np.ones_like(obs.board.rubble)

        global_feature = {name: 0 for name in self.global_feature_names}
        global_feature['env_step'] = obs.real_env_steps
        global_feature['cycle'] = obs.real_env_steps // env_cfg.CYCLE_LENGTH
        global_feature['hour'] = obs.real_env_steps % env_cfg.CYCLE_LENGTH
        global_feature['daytime_or_night'] = global_feature['hour'] < 30
        global_feature['num_factory_own'] = len(obs.factories[player])
        global_feature['num_factory_enm'] = len(obs.factories[enemy])
        global_feature['total_lichen_own'] = np.sum(obs.board.lichen[map_feature['lichen_strains_own']])
        global_feature['total_lichen_enm'] = np.sum(obs.board.lichen[map_feature['lichen_strains_enm']])

        for own_enm, pid in zip(['own', 'enm'], [player, enemy]):
            global_feature[f'factory_total_power_{own_enm}'] = sum(f.power for f in obs.factories[pid].values())
            global_feature[f'factory_total_ice_{own_enm}'] = sum(f.cargo.ice for f in obs.factories[pid].values())
            global_feature[f'factory_total_water_{own_enm}'] = sum(f.cargo.water for f in obs.factories[pid].values())
            global_feature[f'factory_total_ore_{own_enm}'] = sum(f.cargo.ore for f in obs.factories[pid].values())
            global_feature[f'factory_total_metal_{own_enm}'] = sum(f.cargo.metal for f in obs.factories[pid].values())

            global_feature[f'num_light_{own_enm}'] = sum(u.unit_type == "LIGHT" for u in obs.units[pid].values())
            global_feature[f'num_heavy_{own_enm}'] = sum(u.unit_type == "HEAVY" for u in obs.units[pid].values())
            assert global_feature[f'num_light_{own_enm}'] + global_feature[f'num_heavy_{own_enm}'] == len(
                obs.units[pid])
            global_feature[f'robot_total_power_{own_enm}'] = sum(u.power for u in obs.units[pid].values())
            global_feature[f'robot_total_ice_{own_enm}'] = sum(u.cargo.ice for u in obs.units[pid].values())
            global_feature[f'robot_total_water_{own_enm}'] = sum(u.cargo.water for u in obs.units[pid].values())
            global_feature[f'robot_total_ore_{own_enm}'] = sum(u.cargo.ore for u in obs.units[pid].values())
            global_feature[f'robot_total_metal_{own_enm}'] = sum(u.cargo.metal for u in obs.units[pid].values())

        for owner, factories in obs.factories.items():
            for fid, factory in factories.items():
                x, y = factory.pos
                map_feature['factory_id'][x, y] = int(fid[len('factory_'):])
                map_feature['factory_power'][x, y] = factory.power
                map_feature['factory_ice'][x, y] = factory.cargo.ice
                map_feature['factory_water'][x, y] = factory.cargo.water
                map_feature['factory_ore'][x, y] = factory.cargo.ore
                map_feature['factory_metal'][x, y] = factory.cargo.metal
                map_feature['factory_own'][x, y] = owner == player
                map_feature['factory_enm'][x, y] = owner == enemy

                if (factory.cargo.metal >= env_cfg.ROBOTS['LIGHT'].METAL_COST) \
                    and (factory.power >= env_cfg.ROBOTS['LIGHT'].POWER_COST):
                    map_feature['factory_can_build_light'][x, y] = True
                if (factory.cargo.metal >= env_cfg.ROBOTS['HEAVY'].METAL_COST) \
                    and (factory.power >=env_cfg.ROBOTS['HEAVY'].POWER_COST):
                    map_feature['factory_can_build_heavy'][x, y] = True
                water_cost = np.sum(
                    obs.board.lichen_strains == factory.strain_id) // env_cfg.LICHEN_WATERING_COST_FACTOR + 1
                if factory.cargo.water >= water_cost:
                    map_feature['factory_can_grow_lichen'][x, y] = True
                map_feature['factory_water_cost'][x, y] = water_cost

        for owner, units in obs.units.items():
            for uid, unit in units.items():
                x, y = unit.pos
                map_feature['unit_id'][x, y] = int(uid[len('unit_'):])
                map_feature['unit_power'][x, y] = unit.power
                map_feature['unit_ice'][x, y] = unit.cargo.ice
                map_feature['unit_water'][x, y] = unit.cargo.water
                map_feature['unit_ore'][x, y] = unit.cargo.ore
                map_feature['unit_metal'][x, y] = unit.cargo.metal

                map_feature['unit_own'][x, y] = owner == player
                map_feature['unit_enm'][x, y] = owner == enemy

                map_feature['unit_light'][x, y] = unit.unit_type == "LIGHT"
                map_feature['unit_heavy'][x, y] = unit.unit_type == "HEAVY"
                assert unit.unit_type in ["LIGHT", "HEAVY"]

        # action queue
        action_feature = dict(
            unit_indicator=np.zeros((env_cfg.map_size, env_cfg.map_size), dtype=np.int16),
            type=np.zeros((env_cfg.map_size, env_cfg.map_size, env_cfg.UNIT_ACTION_QUEUE_SIZE), dtype=np.int16),
            direction=np.zeros((env_cfg.map_size, env_cfg.map_size, env_cfg.UNIT_ACTION_QUEUE_SIZE), dtype=np.int16),
            resource=np.zeros((env_cfg.map_size, env_cfg.map_size, env_cfg.UNIT_ACTION_QUEUE_SIZE), dtype=np.int16),
            amount=np.zeros((env_cfg.map_size, env_cfg.map_size, env_cfg.UNIT_ACTION_QUEUE_SIZE), dtype=np.int16),
            repeat=np.zeros((env_cfg.map_size, env_cfg.map_size, env_cfg.UNIT_ACTION_QUEUE_SIZE), dtype=np.int16),
            n=np.zeros((env_cfg.map_size, env_cfg.map_size, env_cfg.UNIT_ACTION_QUEUE_SIZE), dtype=np.int16),
        )
        empty_action = [0] * 6
        for units in obs.units.values():
            for unit in units.values():
                padding = [empty_action] * (env_cfg.UNIT_ACTION_QUEUE_SIZE - len(unit.action_queue))
                actions = np.array(list(unit.action_queue) + padding)

                x, y = unit.pos
                action_feature['unit_indicator'][x, y] = True
                action_feature['type'][x, y, :] = actions[:, 0]
                action_feature['direction'][x, y, :] = actions[:, 1]
                action_feature['resource'][x, y, :] = actions[:, 2]
                action_feature['amount'][x, y, :] = actions[:, 3]
                action_feature['repeat'][x, y, :] = actions[:, 4]
                action_feature['n'][x, y, :] = actions[:, 5]

        # normalize
        light_cfg = env_cfg.ROBOTS['LIGHT']
        map_feature

        map_feature['rubble'] = map_feature['rubble'] / env_cfg.MAX_RUBBLE
        map_feature['lichen'] = map_feature['lichen'] / env_cfg.MAX_LICHEN_PER_TILE

        map_feature['factory_power'] = map_feature['factory_power'] / light_cfg.BATTERY_CAPACITY
        map_feature['unit_power'] = map_feature['unit_power'] / light_cfg.BATTERY_CAPACITY

        map_feature['factory_ice'] = map_feature['factory_ice'] / light_cfg.CARGO_SPACE
        map_feature['factory_water'] = map_feature['factory_water'] / light_cfg.CARGO_SPACE
        map_feature['factory_ore'] = map_feature['factory_ore'] / light_cfg.CARGO_SPACE
        map_feature['factory_metal'] = map_feature['factory_metal'] / light_cfg.CARGO_SPACE
        map_feature['factory_water_cost'] = map_feature['factory_water_cost'] / light_cfg.CARGO_SPACE
        map_feature['unit_ice'] = map_feature['unit_ice'] / light_cfg.CARGO_SPACE
        map_feature['unit_water'] = map_feature['unit_water'] / light_cfg.CARGO_SPACE
        map_feature['unit_ore'] = map_feature['unit_ore'] / light_cfg.CARGO_SPACE
        map_feature['unit_metal'] = map_feature['unit_metal'] / light_cfg.CARGO_SPACE

        global_feature['total_lichen_own'] = global_feature['total_lichen_own'] / env_cfg.MAX_LICHEN_PER_TILE
        global_feature['total_lichen_enm'] = global_feature['total_lichen_enm'] / env_cfg.MAX_LICHEN_PER_TILE

        for own_enm in ['own', 'enm']:
            # yapf: disable
            global_feature[f'factory_total_power_{own_enm}'] = global_feature[f'factory_total_power_{own_enm}'] / light_cfg.BATTERY_CAPACITY
            global_feature[f'factory_total_ice_{own_enm}'] = global_feature[f'factory_total_ice_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'factory_total_water_{own_enm}'] = global_feature[f'factory_total_water_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'factory_total_ore_{own_enm}'] = global_feature[f'factory_total_ore_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'factory_total_metal_{own_enm}'] = global_feature[f'factory_total_metal_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'robot_total_power_{own_enm}'] = global_feature[f'robot_total_power_{own_enm}'] / light_cfg.BATTERY_CAPACITY
            global_feature[f'robot_total_ice_{own_enm}'] = global_feature[f'robot_total_ice_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'robot_total_water_{own_enm}'] = global_feature[f'robot_total_water_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'robot_total_ore_{own_enm}'] = global_feature[f'robot_total_ore_{own_enm}'] / light_cfg.CARGO_SPACE
            global_feature[f'robot_total_metal_{own_enm}'] = global_feature[f'robot_total_metal_{own_enm}'] / light_cfg.CARGO_SPACE
            # yapf: enable



        # Unit positions
        units = obs.units[player]
        unit_positions = [unit.pos for unit in units.values()]
        unit_pos = np.array(unit_positions)
        units_on_board = np.zeros((env_cfg.map_size, env_cfg.map_size), dtype=np.bool8)
        if len(unit_pos) > 0:
            units_on_board[unit_pos[:, 0], unit_pos[:, 1]] = 1
        light_units = [unit for unit in units.values() if unit.unit_type == 'LIGHT']
        heavy_units = [unit for unit in units.values() if unit.unit_type == 'HEAVY']
        light_pos = np.array([unit.pos for unit in light_units])
        heavy_pos = np.array([unit.pos for unit in heavy_units])
        light_on_board = np.zeros((env_cfg.map_size, env_cfg.map_size), dtype=np.bool8)
        if len(light_pos) > 0:
            light_on_board[light_pos[:, 0], light_pos[:, 1]] = 1
        heavy_on_board = np.zeros((env_cfg.map_size, env_cfg.map_size), dtype=np.bool8)
        if len(heavy_pos) > 0:
            heavy_on_board[heavy_pos[:, 0], heavy_pos[:, 1]] = 1

        # Factory positions
        factories = obs.factories[player]
        factory_pos = np.array([factory.pos for factory in factories.values()])
        factories_on_board = np.zeros((env_cfg.map_size, env_cfg.map_size), dtype=np.bool8)
        if len(factory_pos) > 0:
            factories_on_board[factory_pos[:, 0], factory_pos[:, 1]] = 1
        not_factory_tile = ~factories_on_board

        entities_on_board = np.clip(units_on_board + factories_on_board, None, 1)

        # Resources on map
        ice = (np.array(obs.board.ice) > 0)
        ore = (np.array(obs.board.ore) > 0)

        ice_left = np.roll(ice, -1, axis=1)
        ice_left[:, -1] = False
        ice_right = np.roll(ice, 1, axis=1)
        ice_right[:, 0] = False
        ice_up = np.roll(ice, -1, axis=0)
        ice_up[-1, :] = False
        ice_down = np.roll(ice, 1, axis=0)
        ice_down[0, :] = False

        # Cargo
        cargo_ice = np.zeros((env_cfg.map_size, env_cfg.map_size), dtype=np.float32)
        if len(factory_pos) > 0:
            cargo_ice[factory_pos[:, 0], factory_pos[:, 1]] = [factory.cargo.ice for factory in factories.values()]
        if len(unit_pos) > 0:
            cargo_ice[unit_pos[:, 0], unit_pos[:, 1]] = [unit.cargo.ice for unit in units.values()]
        cargo_ice /= env_cfg.ROBOTS['HEAVY'].CARGO_SPACE

        cargo_ore = np.zeros((env_cfg.map_size, env_cfg.map_size), dtype=np.float32)
        if len(factory_pos) > 0:
            cargo_ore[factory_pos[:, 0], factory_pos[:, 1]] = [factory.cargo.ore for factory in factories.values()]
        if len(unit_pos) > 0:
            cargo_ore[unit_pos[:, 0], unit_pos[:, 1]] = [unit.cargo.ore for unit in units.values()]
        cargo_ore /= env_cfg.ROBOTS['HEAVY'].CARGO_SPACE

        cargo_water = np.zeros((env_cfg.map_size, env_cfg.map_size), dtype=np.float32)
        if len(factory_pos) > 0:
            cargo_water[factory_pos[:, 0], factory_pos[:, 1]] = [factory.cargo.water for factory in factories.values()]
        if len(unit_pos) > 0:
            cargo_water[unit_pos[:, 0], unit_pos[:, 1]] = [unit.cargo.water for unit in units.values()]
        cargo_water /= env_cfg.ROBOTS['HEAVY'].CARGO_SPACE

        cargo_metal = np.zeros((env_cfg.map_size, env_cfg.map_size), dtype=np.float32)
        if len(factory_pos) > 0:
            cargo_metal[factory_pos[:, 0], factory_pos[:, 1]] = [factory.cargo.metal for factory in factories.values()]
        if len(unit_pos) > 0:
            cargo_metal[unit_pos[:, 0], unit_pos[:, 1]] = [unit.cargo.metal for unit in units.values()]
        cargo_metal /= env_cfg.ROBOTS['HEAVY'].CARGO_SPACE

        cargo_power = np.zeros((env_cfg.map_size, env_cfg.map_size), dtype=np.float32)
        if len(factory_pos) > 0:
            cargo_power[factory_pos[:, 0], factory_pos[:, 1]] = [factory.power for factory in factories.values()]
        if len(unit_pos) > 0:
            cargo_power[unit_pos[:, 0], unit_pos[:, 1]] = [unit.power for unit in units.values()]
        cargo_power /= env_cfg.ROBOTS['HEAVY'].CARGO_SPACE

        factory_feature = {}

        factory_feature['factory_power'] = map_feature['factory_power']
        factory_feature['factory_ice'] = map_feature['factory_ice']
        factory_feature['factory_water'] = map_feature['factory_water']
        factory_feature['factory_ore'] = map_feature['factory_ore']
        factory_feature['factory_metal'] = map_feature['factory_metal']
        factory_feature['factory_water_cost'] = map_feature['factory_water_cost']

        unit_feature = {}

        unit_feature['factory'] = factories_on_board.astype(np.float32)
        unit_feature['light'] = light_on_board.astype(np.float32)
        unit_feature['heavy'] = heavy_on_board.astype(np.float32)
        unit_feature['ice'] = ice.astype(np.float32)
        unit_feature['power'] = cargo_power.astype(np.float32)
        unit_feature['cargo_ice'] = cargo_ice.astype(np.float32)

        if len(units) < 0:
            distance_from_ice = 1.0
        else:
            distance_from_ice = self.get_distance(units_on_board, ice)
        unit_feature['distance_from_ice'] = distance_from_ice

        ice_clusters = self.cluster_board(ice)
        closest_ice_cluster = self.get_closest_coords(units_on_board, ice_clusters) / (env_cfg.map_size * 2)
        closest_ice_cluster_x = closest_ice_cluster[..., 0]
        closest_ice_cluster_y = closest_ice_cluster[..., 1]
        closest_ice_cluster_x_pos = (closest_ice_cluster_x > 0 & ~ice).astype(np.float32) * closest_ice_cluster_x
        closest_ice_cluster_y_pos = (closest_ice_cluster_y > 0 & ~ice).astype(np.float32) * closest_ice_cluster_y
        closest_ice_cluster_x_neg = (closest_ice_cluster_x < 0 & ~ice).astype(np.float32) * closest_ice_cluster_x
        closest_ice_cluster_y_neg = (closest_ice_cluster_y < 0 & ~ice).astype(np.float32) * closest_ice_cluster_y

        unit_feature['cloest_ice_up'] = closest_ice_cluster_y_neg
        unit_feature['cloest_ice_down'] = closest_ice_cluster_y_pos
        unit_feature['cloest_ice_left'] = closest_ice_cluster_x_neg
        unit_feature['cloest_ice_right'] = closest_ice_cluster_x_pos

        unit_feature['ice_up'] = ice_up.astype(np.float32)
        unit_feature['ice_down'] = ice_down.astype(np.float32)
        unit_feature['ice_left'] = ice_left.astype(np.float32)
        unit_feature['ice_right'] = ice_right.astype(np.float32)

        factory_feature = np.array(list(factory_feature.values()))
        unit_feature = np.array(list(unit_feature.values()))

        global_feature = np.array(list(global_feature.values()))
        map_feature = np.array(list(map_feature.values()))

        if output_dict:
            return {'global_feature': global_feature, 'factory_feature': factory_feature, 'unit_feature': unit_feature}
        
        return LuxFeature(global_feature, factory_feature, unit_feature)

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