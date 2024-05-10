"""
Controller Wrapper for the Competition
"""

from typing import Any, Dict
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
import sys

move_deltas = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
transfer_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
factory_adjacent_delta_xy = np.array([
    [-2, -1],
    [-2, +0],
    [-2, +1],
])
factory_adjacent_delta_xy = np.concatenate([factory_adjacent_delta_xy, -factory_adjacent_delta_xy])
factory_adjacent_delta_xy = np.concatenate([factory_adjacent_delta_xy, factory_adjacent_delta_xy[:, ::-1]])

class Controller:
    """
    A controller is a class that takes in an action space and converts it into a lux action
    """
    def __init__(self, action_space: spaces.Space) -> None:
        self.action_space = action_space

    def unit_action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()
    
    def factory_action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Generates a boolean action mask indicating in each \
        discrete dimension whether it would be valid or not
        """
        self.logger.debug("Generating action masks")
        raise NotImplementedError()


class MultiUnitController(Controller):
    """
    A simple controller that controls only the robot \
    that will get spawned.
    """

    def __init__(self, env_cfg) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one heavy robot if there are none regardless of action given

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - transfer action just for transferring in 4 cardinal directions or center (5 * 3)
        - pickup action for power (1 dims)
        - dig action (1 dim)

        It does not include
        - self destruct action
        - planning (via actions executing multiple times or repeating actions)
        - transferring power or resources other than ice

        To help understand how to this controller works to map one \
        action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """

        self.env_cfg = env_cfg
        self.move_act_dims = 4
        self.one_transfer_dim = 5
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.recharge_act_dims = 1
        self.light_unit_build = 1
        self.heavy_unit_build = 1
        self.water_lichen = 1
        self.do_nothing = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.one_transfer_dim
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.recharge_dim_high = self.dig_dim_high + self.recharge_act_dims
        self.light_unit_build_dim_high = self.recharge_dim_high + self.light_unit_build
        self.heavy_unit_build_dim_high = self.light_unit_build_dim_high + self.heavy_unit_build
        self.water_lichen_dim_high = self.heavy_unit_build_dim_high + self.water_lichen
        self.do_nothing_dim_high = self.water_lichen_dim_high + self.do_nothing

        self.total_act_dims = self.do_nothing_dim_high
        action_space = spaces.Discrete(self.total_act_dims)
        super().__init__(action_space)

    def _is_move_action(self, id):
        """
        Checks if the action id corresponds to a move action
        """
        return id < self.move_dim_high

    def _get_move_action(self, id):
        """
        Converts the action id to a move action
        """
        return np.array([0, id + 1, 0, 0, 0, 1])

    def _is_transfer_action(self, id):
        """
        Checks if the action id corresponds to a transfer action
        """
        return id < self.transfer_dim_high

    def _get_transfer_action(self, id, unit_type):
        """
        Converts the action id to a transfer action
        """
        id += 1
        transfer_dir = id % self.one_transfer_dim
        if unit_type == "HEAVY":
            return np.array([1, transfer_dir, 0, self.env_cfg.ROBOTS["HEAVY"].CARGO_SPACE, 0, 1], dtype=int)
        else:
            return np.array([1, transfer_dir, 0, self.env_cfg.ROBOTS["LIGHT"].CARGO_SPACE, 0, 1], dtype=int)

    def _is_pickup_action(self, id):
        """
        Checks if the action id corresponds to a pickup action
        """
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id, unit_type):
        """
        Converts the action id to a pickup action
        """
        if unit_type == "HEAVY":
            return np.array([2, 0, 4, self.env_cfg.ROBOTS["HEAVY"].BATTERY_CAPACITY, 0, 1])
        else:
            return np.array([2, 0, 4, self.env_cfg.ROBOTS["LIGHT"].BATTERY_CAPACITY, 0, 1])

    def _is_dig_action(self, id):
        """
        Checks if the action id corresponds to a dig action
        """
        return id < self.dig_dim_high

    def _get_dig_action(self, id):
        """
        Converts the action id to a dig action
        """
        return np.array([3, 0, 0, 0, 0, 1])
    
    def _is_recharge_action(self, id):
        """
        Checks if the action id corresponds to a recharge action
        """
        return id < self.recharge_dim_high
    
    def _get_recharge_action(self, id, unit_type):
        """
        Converts the action id to a recharge action
        """
        if unit_type == "HEAVY":
            return np.array([5, 0, 0, self.env_cfg.ROBOTS["HEAVY"].BATTERY_CAPACITY, 0, 1])
        else:
            return np.array([5, 0, 0, self.env_cfg.ROBOTS["LIGHT"].BATTERY_CAPACITY, 0, 1])
    
    def _is_light_unit_build_action(self, id):
        """
        Checks if the action id corresponds to a light unit build action
        """
        return id < self.light_unit_build_dim_high
    
    def _get_light_unit_build_action(self, id):
        """
        Converts the action id to a light unit build action
        """
        return 0
    
    def _is_heavy_unit_build_action(self, id):
        """
        Checks if the action id corresponds to a heavy unit build action
        """
        return id < self.heavy_unit_build_dim_high
    
    def _get_heavy_unit_build_action(self, id):
        """
        Converts the action id to a heavy unit build action
        """
        return 1
    
    def _is_water_lichen_action(self, id):
        """
        Checks if the action id corresponds to a water lichen action
        """
        return id < self.water_lichen_dim_high
    
    def _get_water_lichen_action(self, id):
        """
        Converts the action id to a water lichen action
        """
        return 2
    
    def _is_fact_do_nothing_action(self, id):
        """
        Checks if the action id corresponds to a factory do nothing action
        """
        return id < self.do_nothing_dim_high

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Converts the action to a lux action
        """

        shared_obs = obs["player_0"]
        lux_action = {}

        units = shared_obs["units"][agent]
        for unit_id, unit in units.items():
            pos = tuple(unit['pos'])
            filtered_action = action[pos]
            choice = filtered_action
            unit_type = unit["unit_type"]
            action_queue = []
            no_op = False
            if self._is_move_action(choice):
                action_queue = [self._get_move_action(choice)]
            elif self._is_transfer_action(choice):
                action_queue = [self._get_transfer_action(choice, unit_type)]
            elif self._is_pickup_action(choice):
                action_queue = [self._get_pickup_action(choice, unit_type)]
            elif self._is_dig_action(choice):
                action_queue = [self._get_dig_action(choice)]
            elif self._is_recharge_action(choice):
                action_queue = [self._get_recharge_action(choice, unit_type)]
            else:
                no_op = True

            if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
                same_actions = (unit["action_queue"][0] == action_queue[0]).all()
                if same_actions:
                    no_op = True
            if not no_op:
                lux_action[unit_id] = action_queue

        factories = shared_obs["factories"][agent]
        for factory_id, factory in factories.items():
            pos = tuple(factory['pos'])
            filtered_action = action[pos]
            choice = filtered_action
            if self._is_light_unit_build_action(choice):
                lux_action[factory_id] = self._get_light_unit_build_action(choice)
            elif self._is_heavy_unit_build_action(choice):
                lux_action[factory_id] = self._get_heavy_unit_build_action(choice)
            elif self._is_water_lichen_action(choice):
                lux_action[factory_id] = self._get_water_lichen_action(choice)
            elif self._is_fact_do_nothing_action(choice):
                continue
            else:
                continue

        return lux_action
    

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Defines a simplified action mask for this controller's action space
        Doesn't account for whether robot has enough power
        """

        # Get the own player and enemy
        own = "player_0" if agent == "player_0" else "player_1"
        enemy = "player_1" if agent == "player_0" else "player_0"

        # Get the shared observation
        shared_obs = obs[agent]

        # Get the positions of the units
        enemy_units = shared_obs["units"][enemy]
        enemy_units_pos = [tuple(unit["pos"]) for unit in enemy_units.values()]
        own_units = shared_obs["units"][own]
        own_units_pos = [tuple(unit["pos"]) for unit in own_units.values()]
        own_factories = shared_obs["factories"][own]
        enemy_factories = shared_obs["factories"][enemy]
        

        # Create occupancy maps
        factory_occupancy_map = (
            np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )
        player_occupancy_map = (
            np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )

        
        for player in shared_obs["units"]:
            for unit_id in shared_obs["units"][player]:
                unit = shared_obs["units"][player][unit_id]
                pos = np.array(unit["pos"])
                player_occupancy_map[pos[0], pos[1]] = int(unit["unit_id"].split("_")[1])
                
        factories = {}
        for player in shared_obs["factories"]:
            factories[player] = {}
            for f_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][f_id]
                f_pos = f_data["pos"]
                # store in a 3x3 space around the factory position it's strain id.
                factory_occupancy_map[
                    f_pos[0] - 1 : f_pos[0] + 2, f_pos[1] - 1 : f_pos[1] + 2
                ] = f_data["strain_id"]


        # Everything is invalid by default
        action_mask = np.zeros((self.total_act_dims, self.env_cfg.map_size, self.env_cfg.map_size), dtype=bool)

        sorted_units = sorted(shared_obs["units"][agent].items(), reverse=True, key=(lambda x: (x[1]["unit_type"] == "HEAVY", x[1]["cargo"]["ice"], x[1]["power"])))
        for unit_id, unit in sorted_units:
            
            
            x, y = np.array(unit["pos"])
            action_mask[0:4, x, y] = True
            # Go trough move directions
            for direction in range(len(move_deltas)):
                # Target position
                target_pos = (x + move_deltas[direction][0], y + move_deltas[direction][1])

                # Check if the target is out of bounds
                if (target_pos[0] < 0 or target_pos[1] < 0 or target_pos[0] >= self.env_cfg.map_size or target_pos[1] >= self.env_cfg.map_size):
                    action_mask[0 + direction, x, y] = False
                    continue

                # Check if there is a factory at target
                factory_at_pos = next(
                    (factory_id for factory_id, details in own_factories.items() if np.array_equal(details['pos'], target_pos)),
                    False
                )
                # Check if enemy factory is at target
                enemy_factory_at_pos = next(
                    (factory_id for factory_id, details in enemy_factories.items() if np.array_equal(details['pos'], target_pos)),
                    False
                )

                # Check if there is a unit at target
                unit_there = player_occupancy_map[target_pos[0], target_pos[1]]

                enemy_unit_there = target_pos in enemy_units_pos

                # If there is factory there, we can't move there
                if factory_at_pos or enemy_factory_at_pos:
                    action_mask[0 + direction, x, y] = False
                    continue
                
                # If there is a unit there, we can't move there
                if unit_there != -1:
                    action_mask[0 + direction, x, y] = False
                    continue
                
                # if the unit is not heavy, it can't move to a tile with an enemy unit
                if unit["unit_type"] != "HEAVY" and enemy_unit_there:
                    action_mask[0 + direction, x, y] = False
                    continue

            # Go trough transfer directions
            for trans_direction in range(len(transfer_deltas)):
                
                # target position
                transfer_pos = (x + transfer_deltas[trans_direction][0], y + transfer_deltas[trans_direction][1])

                # Check if the target is out of bounds
                if (transfer_pos[0] < 0 or transfer_pos[1] < 0 or transfer_pos[0] >= self.env_cfg.map_size or transfer_pos[1] >= self.env_cfg.map_size):
                    continue

                # Check if there is a factory at target
                factory_there = factory_occupancy_map[transfer_pos[0], transfer_pos[1]]

                # Check if there is a unit at target
                unit_there = player_occupancy_map[transfer_pos[0], transfer_pos[1]]
                enemy_unit_there = transfer_pos in enemy_units_pos
                
                # If unit has ice cargo
                if unit["cargo"]["ice"] > 0:
                    # If there is a factory there, we can transfer to it
                    if factory_there in shared_obs["teams"][agent]["factory_strains"]:
                        action_mask[4 + trans_direction, x, y] = True

            factory_there = factory_occupancy_map[x, y]
            on_top_of_factory = (
                factory_there in shared_obs["teams"][agent]["factory_strains"]
            )
            
            # Dig action is valid if there is ice, ore, rubble or lichen
            board_sum = (
                shared_obs["board"]["ice"][x, y]
                + shared_obs["board"]["ore"][x, y]
                + shared_obs["board"]["rubble"][x, y]
                + shared_obs["board"]["lichen"][x, y]
            )

            if board_sum > 0:
                action_mask[10, x , y] = True

            # Pickup action is valid if there is power
            if on_top_of_factory:
                action_mask[
                    9, x, y
                ] = True


            # Recharge action is valid if the unit is not on top of a factory and the real_env_steps is greater than 30
            if shared_obs["real_env_steps"] % 40 > 30 and not on_top_of_factory:
                action_mask[
                    11, x , y
                ] = True

            if unit["power"] <= 0:
                action_mask[
                    11, x, y
                ] = True
                action_mask[:11, x, y] = False # Can't do anything else if no power

            # do nothing is always valid for units too
        action_mask[15, :, :] = True

        indices = []
        for _, factory in own_factories.items():
            x, y = factory["pos"]
            indices.append((x, y))
            unit_on_factory = (x, y) in own_units_pos

            # Light unit build is valid only if there is no unit on the factory
            if factory['cargo']['metal'] >= self.env_cfg.ROBOTS['LIGHT'].METAL_COST \
                and factory['power'] >= self.env_cfg.ROBOTS['LIGHT'].POWER_COST \
                and not unit_on_factory:
                action_mask[12, x, y] = True

            # Heavy unit build is valid only if there is no unit on the factory
            if factory['cargo']['metal'] >= self.env_cfg.ROBOTS['HEAVY'].METAL_COST \
                and factory['power'] >= self.env_cfg.ROBOTS['HEAVY'].POWER_COST \
                and not unit_on_factory:
                action_mask[13, x, y] = True

            # Water lichen is valid only if there is enough water and there is a free adjacent tile
            lichen_strains_size = np.sum(shared_obs["board"]["lichen"] == factory["strain_id"])
            if factory["cargo"]["water"] >= (lichen_strains_size + 1) // self.env_cfg.LICHEN_WATERING_COST_FACTOR:
                adj_xy = factory['pos'] + factory_adjacent_delta_xy
                adj_xy = adj_xy[(adj_xy >= 0).all(axis=1) & (adj_xy < self.env_cfg.map_size).all(axis=1)]
                adj_x, adj_y = adj_xy[:, 0], adj_xy[:, 1]
                no_rubble = (shared_obs["board"]["rubble"][adj_x, adj_y] == 0)
                no_ice = (shared_obs["board"]["ice"][adj_x, adj_y] == 0)
                no_ore = (shared_obs["board"]["ore"][adj_x, adj_y] == 0)
                if (no_rubble & no_ice & no_ore).any():
                    action_mask[14, x, y] = False

        return action_mask
