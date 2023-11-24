"""
Controller Wrapper for the Competition
"""

from typing import Any, Dict
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from observation.obs_parser import ObservationParser

import logging
logger = logging.getLogger(__name__)


class Controller:
    """
    A controller is a class that takes in an action space and converts it into a lux action
    """
    def __init__(self, action_space: spaces.Space) -> None:
        logger.debug(f"Creating {self.__class__.__name__}")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")
        self.action_space = action_space

    def entity_actions_to_lux_action(
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
        self.logger.debug("Creating lux action")
        raise NotImplementedError()

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Generates a boolean action mask indicating in each \
        discrete dimension whether it would be valid or not
        """
        self.logger.debug("Generating action masks")
        raise NotImplementedError()


class SimpleUnitDiscreteController(Controller):
    """
    A simple controller that controls only the robot \
    that will get spawned.
    """

    DISABLED_ACTIONS = [
        "transfer_ore_center",
        "transfer_ore_up",
        "transfer_ore_right",
        "transfer_ore_down",
        "transfer_ore_left",
        "transfer_power_center",
        "transfer_power_up",
        "transfer_power_right",
        "transfer_power_down",
        "transfer_power_left",
        "pickup"
    ]

    ID_TO_ACTION_NAME = {
        0: "move_up",
        1: "move_right",
        2: "move_down",
        3: "move_left",
        4: "transfer_ice_center",
        5: "transfer_ice_up",
        6: "transfer_ice_right",
        7: "transfer_ice_down",
        8: "transfer_ice_left",
        9: "transfer_ore_center",
        10: "transfer_ore_up",
        11: "transfer_ore_right",
        12: "transfer_ore_down",
        13: "transfer_ore_left",
        14: "transfer_power_center",
        15: "transfer_power_up",
        16: "transfer_power_right",
        17: "transfer_power_down",
        18: "transfer_power_left",
        19: "pickup",
        20: "dig",
        21: "recharge",
        22: "build_light",
        23: "build_heavy",
        24: "water_lichen",
        25: "skip"
    }
    UNIT_ACTIONS = [a for a in ID_TO_ACTION_NAME.keys() if 0 <= a < 22]
    FACTORY_ACTIONS = [a for a in ID_TO_ACTION_NAME.keys() if 22 <= a < 25]
    ACTION_NAME_TO_ID = {v:k for k,v in ID_TO_ACTION_NAME.items()}

    move_act_dims = 4
    one_transfer_dim = 5
    transfer_act_dims = one_transfer_dim * 3
    pickup_act_dims = 1
    dig_act_dims = 1
    recharge_act_dims = 1
    light_unit_build = 1
    heavy_unit_build = 1
    water_lichen = 1
    skip = 1

    move_dim_low = 0
    move_dim_high = move_dim_low + move_act_dims

    transfer_dim_low = move_dim_high
    transfer_dim_high = transfer_dim_low + transfer_act_dims

    pickup_dim_low = transfer_dim_high
    pickup_dim_high = pickup_dim_low + pickup_act_dims

    dig_dim_low = pickup_dim_high
    dig_dim_high = dig_dim_low + dig_act_dims

    recharge_dim_low = dig_dim_high
    recharge_dim_high = recharge_dim_low + recharge_act_dims

    light_unit_build_dim_low = recharge_dim_high
    light_unit_build_dim_high = light_unit_build_dim_low + light_unit_build

    heavy_unit_build_dim_low = light_unit_build_dim_high
    heavy_unit_build_dim_high = heavy_unit_build_dim_low + heavy_unit_build

    water_lichen_dim_low = heavy_unit_build_dim_high
    water_lichen_dim_high = water_lichen_dim_low + water_lichen

    skip_dim_low = water_lichen_dim_high
    skip_dim_high = skip_dim_low + skip

    total_act_dims = skip_dim_high
    action_space = spaces.Discrete(total_act_dims)

    power_costs = {
        "LIGHT": {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            20: 5
        },
        "HEAVY": {
            0: 20,
            1: 20,
            2: 20,
            3: 20,
            20: 60
        }
    }

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
        logger.debug(f"Creating SimpleUnitDiscreteController")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")

        self.env_cfg = env_cfg

        super().__init__(self.action_space)

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
        relative_id = id - self.transfer_dim_low
        transfer_dir = relative_id % self.one_transfer_dim
        transfer_type = relative_id // self.one_transfer_dim
        transfer_type_mapping = [0, 1, 4]  # Mapping index to desired value, skip water and metal, 0 for ice, 1 for ore, 2 for power
        transfer_type = transfer_type_mapping[transfer_type.astype(int).item()]

        return np.array([1, transfer_dir, transfer_type, self.env_cfg.max_transfer_amount, 0, 1], dtype=int)

    def _is_pickup_action(self, id):
        """
        Checks if the action id corresponds to a pickup action
        """
        return self.pickup_dim_low <= id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        """
        Converts the action id to a pickup action
        """
        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_dig_action(self, id):
        """
        Checks if the action id corresponds to a dig action
        """
        return self.dig_dim_low <= id < self.dig_dim_high

    def _get_dig_action(self, id):
        """
        Converts the action id to a dig action
        """
        return np.array([3, 0, 0, 0, 0, 1])
    
    def _is_recharge_action(self, id):
        """
        Checks if the action id corresponds to a recharge action
        """
        return self.recharge_dim_low <= id < self.recharge_dim_high
    
    def _get_recharge_action(self, id):
        """
        Converts the action id to a recharge action
        """
        return np.array([4, 0, 0, 0, 0, 1])
    
    def _is_light_unit_build_action(self, id):
        """
        Checks if the action id corresponds to a light unit build action
        """
        return self.light_unit_build_dim_low <= id < self.light_unit_build_dim_high
    
    def _get_light_unit_build_action(self, id):
        """
        Converts the action id to a light unit build action
        """
        return 0
    
    def _is_heavy_unit_build_action(self, id):
        """
        Checks if the action id corresponds to a heavy unit build action
        """
        return self.heavy_unit_build_dim_low <= id < self.heavy_unit_build_dim_high
    
    def _get_heavy_unit_build_action(self, id):
        """
        Converts the action id to a heavy unit build action
        """
        return 1
    
    def _is_water_lichen_action(self, id):
        """
        Checks if the action id corresponds to a water lichen action
        """
        return self.water_lichen_dim_low <= id < self.water_lichen_dim_high
    
    def _get_water_lichen_action(self, id):
        """
        Converts the action id to a water lichen action
        """
        return 2

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        entity_actions = self.entity_actions_to_lux_action(agent, obs, action)
        return entity_actions

    def entity_actions_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Converts the action to a lux action
        """

        shared_obs = obs["player_0"]
        lux_action = {}

        factories = shared_obs["factories"][agent]
        factory_id = list(shared_obs["factories"][agent].keys())
        factory_count = len(factory_id)
        units = shared_obs["units"][agent]
        unit_id = list(units.keys())
        unit_count = len(unit_id)

        assert action.shape[0] >= factory_count + unit_count

        self.logger.debug(f"units are: {unit_id}")
        self.logger.debug(f"factories are: {factory_id}")
        self.logger.debug(f"env steps: {shared_obs['real_env_steps']}")

        factory_actions = action[:factory_count]
        unit_actions = action[factory_count:]

        for i, (factory_id, _) in enumerate(factories.items()):
            filtered_action = factory_actions[i]
            choice = filtered_action
            self.logger.debug(f"Factory action: {factory_id} {choice} -> {self.ID_TO_ACTION_NAME[choice]}")
            if self._is_light_unit_build_action(choice):
                lux_action[factory_id] = self._get_light_unit_build_action(choice)
            elif self._is_heavy_unit_build_action(choice):
                lux_action[factory_id] = self._get_heavy_unit_build_action(choice)
            elif self._is_water_lichen_action(choice):
                lux_action[factory_id] = self._get_water_lichen_action(choice)
            else:
                continue

        for i, (unit_id, unit) in enumerate(units.items()):
            filtered_action = unit_actions[i]
            choice = filtered_action

            pos = unit["pos"]
            has_ice = obs["player_0"]["board"]["ice"][pos[0], pos[1]]

            action_queue = []
            no_op = False
            if self._is_move_action(choice):
                action_queue = [self._get_move_action(choice)]
            elif self._is_transfer_action(choice):
                action_queue = [self._get_transfer_action(choice)]
            elif self._is_pickup_action(choice):
                action_queue = [self._get_pickup_action(choice)]
            elif self._is_dig_action(choice):
                if has_ice:
                    self.logger.debug(f"Unit action: {unit_id} {choice} ice={has_ice} -> {self.ID_TO_ACTION_NAME[choice]}")

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
        
        self.logger.debug(f"Created lux action\n{lux_action}")
        return lux_action

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Defines a simplified action mask for this controller's action space

        Doesn't account for whether robot has enough power
        """

        ## Should return a 19x64x64 mask. 19 actions, 64x64 board
        ## It should return all false where units are not there

        self.logger.debug(f"Creating simplified action mask for {agent}")

        # Change: return simple action mask
        # TODO: implement
        map_size = self.env_cfg.map_size

        obs = obs[agent]
        
        factories = obs["factories"][agent]
        units = obs["units"][agent]

        self.logger.debug(f"Count: {len(factories) + len(units)}, {list(factories.keys()) + list(units.keys())}")

        unit_positions = [tuple(u["pos"]) for u in units.values()]
        entities = list(factories.items()) + list(units.items())
        entity_count = len(entities)
        board = obs["board"]

        action_mask = np.ones((entity_count, self.total_act_dims))
        self.logger.debug(f"Action mask in controller: {action_mask.shape}")

        collision_avoider = {}

        for i, (entity_name, entity) in enumerate(entities):
            entity_id = int(entity_name.split("_")[1])
            pos = entity["pos"]
            is_factory = "factory" in entity_name
            is_unit = "unit" in entity_name
            assert is_factory or is_unit, "Entity should be either a factory or a unit"
            power = entity["power"]
            ice = entity["cargo"]["ice"]
            ore = entity["cargo"]["ore"]
            water = entity["cargo"]["water"]
            metal = entity["cargo"]["metal"]
            strain_id = None
            if is_factory:
                strain_id = entity["strain_id"]
            is_light = None
            is_heavy = None
            power_costs = None
            if is_unit:
                is_light = entity["unit_type"] == "LIGHT"
                is_heavy = entity["unit_type"] == "HEAVY"
                assert is_light or is_heavy, "Unit should be either light or heavy"
                power_costs = self.power_costs[entity["unit_type"]]

            up, right, down, left, up_left, up_right, down_right, down_left = ObservationParser.get_neighbours(pos, map_size)

            # if factory, don't do unit actions
            if is_factory:
                action_mask[i, self.UNIT_ACTIONS] = 0
            # if unit, don't do factory actions
            if is_unit:
                action_mask[i, self.FACTORY_ACTIONS] = 0

            if is_unit:
                has_ice = board["ice"][pos[0], pos[1]] > 0
                has_ore = board["ore"][pos[0], pos[1]] > 0
                has_rubble = board["rubble"][pos[0], pos[1]] > 0

                # if there is no resource or rubble, don't dig
                if not has_ice and not has_ore and not has_rubble:
                    action_mask[i, self.ACTION_NAME_TO_ID["dig"]] = 0
                
                # if action is out of the map, don't move
                # if agent is standing in the way, don't move
                for new_pos, action_name in zip([up, right, down, left], ["move_up", "move_right", "move_down", "move_left"]):
                    if new_pos is None:
                        action_mask[i, self.ACTION_NAME_TO_ID[action_name]] = 0
                        continue

                    new_pos_tuple = tuple(new_pos)
                    has_unit = new_pos_tuple in unit_positions
                    if has_unit:
                        action_mask[i, self.ACTION_NAME_TO_ID[action_name]] = 0
                    else:
                        # add unit to collision avoider
                        tup = ((i, entity_name, entity, action_name), (ice, ore, power))
                        if new_pos_tuple not in collision_avoider:
                            collision_avoider[new_pos_tuple] = []
                        collision_avoider[new_pos_tuple].append(tup)
                
                transfer_actions = [a for a in self.UNIT_ACTIONS if "transfer" in self.ID_TO_ACTION_NAME[a]]
                transfer_center_actions = [a for a in transfer_actions if "center" in self.ID_TO_ACTION_NAME[a]]
                transfer_up_actions = [a for a in transfer_actions if "up" in self.ID_TO_ACTION_NAME[a]]
                transfer_right_actions = [a for a in transfer_actions if "right" in self.ID_TO_ACTION_NAME[a]]
                transfer_down_actions = [a for a in transfer_actions if "down" in self.ID_TO_ACTION_NAME[a]]
                transfer_left_actions = [a for a in transfer_actions if "left" in self.ID_TO_ACTION_NAME[a]]
                transfer_ice_actions = [a for a in transfer_actions if "ice" in self.ID_TO_ACTION_NAME[a]]
                transfer_ore_actions = [a for a in transfer_actions if "ore" in self.ID_TO_ACTION_NAME[a]]
                transfer_power_actions = [a for a in transfer_actions if "power" in self.ID_TO_ACTION_NAME[a]]

                # if no resource, don't transfer
                if not ice:
                    action_mask[i, transfer_ice_actions] = 0
                if not ore:
                    action_mask[i, transfer_ore_actions] = 0
                if not power:
                    action_mask[i, transfer_power_actions] = 0

                all_factory_positions = (np.array([[f["pos"]] + list(ObservationParser.get_neighbours(f["pos"], map_size)) for f in factories.values()]).reshape(-1, 2))
                # if not standing on factory, don't transfer to center
                if not (all_factory_positions == pos).all(axis=1).any():
                    action_mask[i, transfer_center_actions] = 0
                # if factory not on up, don't transfer to center
                if not (all_factory_positions == up).all(axis=1).any():
                    action_mask[i, transfer_up_actions] = 0
                # if factory not on right, don't transfer to center
                if not (all_factory_positions == right).all(axis=1).any():
                    action_mask[i, transfer_right_actions] = 0
                # if factory not on down, don't transfer to center
                if not (all_factory_positions == down).all(axis=1).any():
                    action_mask[i, transfer_down_actions] = 0
                # if factory not on left, don't transfer to center
                if not (all_factory_positions == left).all(axis=1).any():
                    action_mask[i, transfer_left_actions] = 0

                # if not enough power, don't do action
                for a in self.UNIT_ACTIONS:
                    if a in power_costs:
                        power_cost = power_costs[a]
                        if power_cost > power:
                            action_mask[i, a] = 0

                # if action is diabled, don't do action
                disabled_actions = [self.ACTION_NAME_TO_ID[a] for a in self.DISABLED_ACTIONS]
                action_mask[i, disabled_actions] = 0

            if is_factory:
                factory_positions = [p for p in [pos, up_left, up, up_right, right, down_right, down, down_left, left] if p is not None]
                assert len(factory_positions) == 9

                can_build_unit = True
                # if there is already a unit on top of the factory, don't build
                for p in factory_positions:
                    has_unit = tuple(p) in unit_positions
                    if has_unit:
                        can_build_unit = False
                        break
                
                if not can_build_unit:
                    action_mask[i, self.ACTION_NAME_TO_ID["build_light"]] = 0
                    action_mask[i, self.ACTION_NAME_TO_ID["build_heavy"]] = 0

            valid_actions = np.where(action_mask[i] == 1)[0]
            valid_actions = [self.ID_TO_ACTION_NAME[a] for a in valid_actions]
            if len([v for v in valid_actions if "transfer" in v and "ice" in v]) > 0:
                self.logger.debug(f"{entity_name} {pos} {valid_actions}")
            
            assert (action_mask[i] > 0).any(), "Action mask must contain at least 1 non-zero element"
        
        if collision_avoider:
            for pos, lst in collision_avoider.items():
                lst = sorted(lst, key=lambda x: x[1], reverse=True)
                (_, entity_name, entity, action_name), _ = lst[0]
                if len(lst) > 1:
                    #print(f"Blocking at {pos}, choosen: {entity_name} {entity['pos']} {action_name}")
                    for u in lst[1:]:
                        (i, entity_name, entity, action_name), _ = u
                        #print(f"\tBlocked: {entity_name} {entity['pos']} {action_name}")
                        action_mask[i, self.ACTION_NAME_TO_ID[action_name]] = 0
        
        # for i in range(entity_count):
        #     # if valid action without skip, don't skip
        #     skip_action_id = self.ACTION_NAME_TO_ID["skip"]
        #     if action_mask[i, [a for a in self.ID_TO_ACTION_NAME.keys() if a != skip_action_id]].sum() > 0:
        #         action_mask[i, skip_action_id] = 0

        self.logger.debug(f"Created action mask: {action_mask.shape}")

        return action_mask

