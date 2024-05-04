"""
Controller Wrapper for the Competition
"""

from typing import Any, Dict
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
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
        self.num_of_move = 0
        self.num_of_transfer = 0
        self.num_of_pickup = 0
        self.num_of_dig = 0
        self.num_of_recharge = 0
        self.num_of_light_unit_build = 0
        self.num_of_heavy_unit_build = 0
        self.num_of_water_lichen = 0


        self.env_cfg = env_cfg
        self.move_act_dims = 4
        self.one_transfer_dim = 5
        self.transfer_act_dims = self.one_transfer_dim * 1
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.recharge_act_dims = 1
        self.light_unit_build = 1
        self.heavy_unit_build = 1
        self.water_lichen = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.recharge_dim_high = self.dig_dim_high + self.recharge_act_dims
        self.light_unit_build_dim_high = self.recharge_dim_high + self.light_unit_build
        self.heavy_unit_build_dim_high = self.light_unit_build_dim_high + self.heavy_unit_build
        self.water_lichen_dim_high = self.heavy_unit_build_dim_high + self.water_lichen

        self.total_act_dims = self.water_lichen_dim_high
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

    def _get_transfer_action(self, id):
        
        id = id - (self.transfer_act_dims/1)
        id += 1
        transfer_dir = id % (self.transfer_act_dims/1)
        transfer_type_mapping = [0]  # Mapping index to desired value
        transfer_type_index = id // (self.transfer_act_dims/1)  # 0 for ice, 1 for ore
        transfer_type = transfer_type_mapping[transfer_type_index.astype(int).item()]
        return np.array([1, transfer_dir, transfer_type, self.env_cfg.max_transfer_amount, 0, 1], dtype=int)

    def _is_pickup_action(self, id):
        """
        Checks if the action id corresponds to a pickup action
        """
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        """
        Converts the action id to a pickup action
        """
        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])

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
    
    def _get_recharge_action(self, id):
        """
        Converts the action id to a recharge action
        """
        return np.array([5, 0, 0, 0, 0, 1])
    
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
            action_queue = []
            no_op = False
            if self._is_move_action(choice):
                action_queue = [self._get_move_action(choice)]
            elif self._is_transfer_action(choice):
                action_queue = [self._get_transfer_action(choice)]
            elif self._is_pickup_action(choice):
                action_queue = [self._get_pickup_action(choice)]
            elif self._is_dig_action(choice):
                action_queue = [self._get_dig_action(choice)]
            elif self._is_recharge_action(choice):
                action_queue = [self._get_recharge_action(choice)]
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
            else:
                continue
        
        if lux_action:
            for k, v in lux_action.items():
                if isinstance(v, list):
                    if v[0][0] == 0:
                        self.num_of_move += 1
                    elif v[0][0] == 1:
                        self.num_of_transfer += 1
                    elif v[0][0] == 2:
                        self.num_of_pickup += 1
                    elif v[0][0] == 3:
                        self.num_of_dig += 1
                    elif v[0][0] == 5:
                        self.num_of_recharge += 1
                else:
                    if v == 0:
                        self.num_of_light_unit_build += 1
                    elif v == 1:
                        self.num_of_heavy_unit_build += 1
                    elif v == 2:
                        self.num_of_water_lichen += 1

        #print("Number of move actions: ", self.num_of_move)
        #print("Number of transfer actions: ", self.num_of_transfer)
        #print("Number of pickup actions: ", self.num_of_pickup)
        #print("Number of dig actions: ",self. num_of_dig)
        #print("Number of recharge actions: ", self.num_of_recharge)
        #print("Number of light unit build actions: ", self.num_of_light_unit_build)
        #print("Number of heavy unit build actions: ", self.num_of_heavy_unit_build)
        #print("Number of water lichen actions: ",self. num_of_water_lichen)

        return lux_action

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Defines a simplified action mask for this controller's action space
        Doesn't account for whether robot has enough power
        """
        # Get the own player and enemy
        own = "player_0" if agent == "player_0" else "player_1"
        enemy = "player_1" if agent == "player_0" else "player_0"

        def factory_under_unit(unit_pos, factories):
            for _, factory in factories.items():
                factory_pos = factory["pos"]
                if abs(unit_pos[0] - factory_pos[0]) <= 1 and abs(unit_pos[1] - factory_pos[1]) <= 1:
                    return factory
            return None
        
        # Get the shared observation
        shared_obs = obs[agent]

        # Create occupancy maps
        factory_occupancy_map = (np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1)
        player_occupancy_map = (np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1)

        # Get the positions of the units
        enemy_units = shared_obs["units"][enemy]
        enemy_units_pos = [tuple(unit["pos"]) for unit in enemy_units.values()]
        own_units = shared_obs["units"][own]
        own_units_pos = [tuple(unit["pos"]) for unit in own_units.values()]
        own_factories = shared_obs["factories"][own]     

        # Fill the unit occupancy maps
        for player in shared_obs["units"]:
            for unit_id in shared_obs["units"][player]:
                unit = shared_obs["units"][player][unit_id]
                pos = np.array(unit["pos"])
                player_occupancy_map[pos[0], pos[1]] = int(unit["unit_id"].split("_")[1])

        # Fill the factory occupancy map
        for player in shared_obs["factories"]:
            for f_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][f_id]
                f_pos = f_data["pos"]
                # store in a 3x3 space around the factory position it's strain id.
                factory_occupancy_map[f_pos[0] - 1 : f_pos[0] + 2, f_pos[1] - 1 : f_pos[1] + 2] = f_data["strain_id"]
        
        # Everything is invalid by default
        action_mask = np.zeros((self.total_act_dims, self.env_cfg.map_size, self.env_cfg.map_size), dtype=bool)

        for factory_id, factory in own_factories.items():
            x, y = factory["pos"]

            unit_on_factory = (x, y) in own_units_pos

            # Light unit build is valid only if there is no unit on the factory
            if factory['cargo']['metal'] >= self.env_cfg.ROBOTS['LIGHT'].METAL_COST \
                and factory['power'] >= self.env_cfg.ROBOTS['LIGHT'].POWER_COST \
                and not unit_on_factory:
                action_mask[self.light_unit_build_dim_high-self.light_unit_build_dim_high, x, y] = True

            # Heavy unit build is valid only if there is no unit on the factory
            if factory['cargo']['metal'] >= self.env_cfg.ROBOTS['HEAVY'].METAL_COST \
                and factory['power'] >= self.env_cfg.ROBOTS['HEAVY'].POWER_COST \
                and not unit_on_factory:
                action_mask[self.heavy_unit_build_dim_high-self.heavy_unit_build_dim_high, x, y] = True

            lichen_strains_size = np.sum(shared_obs["board"]["lichen"] == factory["strain_id"])
            if factory["cargo"]["water"] >= (lichen_strains_size + 1) // self.env_cfg.LICHEN_WATERING_COST_FACTOR:
                adj_xy = factory['pos'] + factory_adjacent_delta_xy
                adj_xy = adj_xy[(adj_xy >= 0).all(axis=1) & (adj_xy < self.env_cfg.map_size).all(axis=1)]
                adj_x, adj_y = adj_xy[:, 0], adj_xy[:, 1]
                no_rubble = (shared_obs["board"]["rubble"][adj_x, adj_y] == 0)
                no_ice = (shared_obs["board"]["ice"][adj_x, adj_y] == 0)
                no_ore = (shared_obs["board"]["ore"][adj_x, adj_y] == 0)
                if (no_rubble & no_ice & no_ore).any():
                    action_mask[self.water_lichen_dim_high - self.water_lichen, x, y] = True

        unit_move_targets = set()
        sorted_units = sorted(shared_obs["units"][agent].items(), reverse=True, key=(lambda x: (x[1]["unit_type"] == "HEAVY", x[1]["cargo"]["ice"], x[1]["power"])))
        for unit_id, unit in sorted_units:
            
            x, y = np.array(unit["pos"])
            if self.env_cfg.ROBOTS[unit["unit_type"]].ACTION_QUEUE_POWER_COST <= unit["power"]:
                    for direction in range(len(move_deltas)):
                        target_pos = (x + move_deltas[direction][0], y + move_deltas[direction][1])

                        if (
                            target_pos[0] < 0
                            or target_pos[1] < 0
                            or target_pos[0] >= self.env_cfg.map_size
                            or target_pos[1] >= self.env_cfg.map_size
                        ):
                            continue

                        if factory_under_unit(target_pos, shared_obs["factories"][agent]) is not None:
                            continue

                        unit_at_target = player_occupancy_map[target_pos[0], target_pos[1]]
                        if unit_at_target != -1:
                            continue

                        if tuple(target_pos) in factory_occupancy_map:
                            continue

                        if not unit["unit_type"] != "HEAVY" and tuple(target_pos) in enemy_units_pos:
                            continue
                        
                        if tuple(target_pos) not in unit_move_targets:
                            action_mask[0:4, :, :] = True
                            unit_move_targets.add(tuple(target_pos))


                        if unit["cargo"]["ice"] > 0:
                            unit_has_ice = True
                        else:
                            unit_has_ice = False

                        for direction in range(1, len(move_deltas)):
                            transfer_pos = (x + move_deltas[direction][0], y + move_deltas[direction][1])

                            if (
                            transfer_pos[0] < 0
                            or transfer_pos[1] < 0
                            or transfer_pos[0] >= len(factory_occupancy_map)
                            or transfer_pos[1] >= len(factory_occupancy_map[0])
                            ):
                                continue

                            if factory_under_unit(transfer_pos, shared_obs["factories"][agent]) is not None and unit_has_ice:
                                action_mask[self.move_dim_high + direction, x, y] = True
                            

                            unit_there = player_occupancy_map[transfer_pos[0], transfer_pos[1]]

                            if unit_there != -1:
                                if unit_has_ice:
                                    action_mask[self.move_dim_high + direction, x, y] = True


                        board_sum = (
                            shared_obs["board"]["ice"][pos[0], pos[1]]
                            + shared_obs["board"]["ore"][pos[0], pos[1]]
                            + shared_obs["board"]["rubble"][pos[0], pos[1]]
                            + shared_obs["board"]["lichen"][pos[0], pos[1]]
                        )
                        if board_sum > 0 and factory_under_unit((x, y), shared_obs["factories"][agent]) is None:
                            action_mask[
                                self.dig_dim_high - self.dig_act_dims : self.dig_dim_high, pos[0], pos[1]
                            ] = True


                        if factory_under_unit((x, y), shared_obs["factories"][agent]) is not None:
                            action_mask[
                                self.pickup_dim_high - self.pickup_act_dims : self.pickup_dim_high, pos[0], pos[1]
                            ] = True

            
            if shared_obs["real_env_steps"] % 40 > 30:
                action_mask[
                    self.recharge_dim_high - self.recharge_act_dims : self.recharge_dim_high, pos[0], pos[1]
                ] = True

        action_mask = np.ones((self.total_act_dims, self.env_cfg.map_size, self.env_cfg.map_size), dtype=bool)
        action_mask[14, :, :] = False
        return action_mask
