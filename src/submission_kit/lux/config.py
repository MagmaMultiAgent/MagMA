"""
Module responsible for storing configurations of the game
"""
import dataclasses
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict

def convert_dict_to_ns(dict_x):
    """
    Function converting dictionary into namespace
    """
    if isinstance(dict_x, dict):
        for k in dict_x:
            dict_x[k] = convert_dict_to_ns(dict_x)
        return Namespace(dict_x)
    return Namespace()

@dataclass
class UnitConfig:
    """Dataclass storing unit information"""
    metal_cost: int = 100
    power_cost: int = 500
    cargo_space: int = 1000
    battery_capacity: int = 1500
    charge: int = 1
    init_power: int = 50
    move_cost: int = 1
    rubble_movement_cost: float = 1
    dig_cost: int = 5
    dig_rubble_removed: int = 1
    dig_resource_gain: int = 2
    dig_lichen_removed: int = 10
    self_destruct_cost: int = 10
    rubble_after_destruction: int = 1
    action_queue_power_cost: int = 1

@dataclass
class EnvConfig:
    """Dataclass containing environment configurations"""
    ## various options that can be configured if needed

    ### Variable parameters that don't affect game logic much ###
    max_episode_length: int = 1000
    map_size: int = 48
    verbose: int = 1

    # this can be disabled to improve env FPS but assume your actions are well formatted
    # During online competition this is set to True
    validate_action_space: bool = True

    ### Constants ###
    # you can only ever transfer in/out 1000 as this is the max cargo space.
    max_transfer_amount: int = 10000
    min_factories: int = 2
    max_factories: int = 5
    cycle_length: int = 50
    day_length: int = 30
    unit_action_queue_size: int = 20  # when set to 1, then no action queue is used

    max_rubble: int = 100
    factory_rubble_after_destruction: int = 50
    init_water_metal_per_factory: int = (
        150  # amount of water and metal units given to each factory
    )
    init_power_per_factory: int = 1000

    #### LICHEN ####
    min_lichen_to_spread: int = 20
    lichen_lost_without_water: int = 1
    lichen_gained_with_water: int = 1
    max_lichen_per_tile: int = 100
    power_per_connected_lichen_tile: int = 1

    # cost of watering with a factory is `ceil(# of connected lichen tiles) / (this factor) + 1`
    lichen_watering_cost_factor: int = 10

    #### Bidding System ####
    bidding_system: bool = True

    #### Factories ####
    factory_processing_rate_water: int = 100
    ice_water_ratio: int = 4
    factory_processing_rate_metal: int = 50
    ore_metal_ratio: int = 5
    # game design note: Factories close to resource cluster = more resources are refined per turn
    # Then the high ice:water and ore:metal ratios encourages transfer of refined resources between
    # factories dedicated to mining particular clusters which is more possible as it is more compact

    factory_charge: int = 50
    factory_water_consumption: int = 1
    # with a positive water consumption, the game becomes quite hard for new competitors.
    # so we set it to 0

    #### Collision Mechanics ####
    power_loss_factor: float = 0.5

    #### Units ####
    robots: Dict[str, UnitConfig] = dataclasses.field(
        default_factory=lambda: 
            {"light": UnitConfig(metal_cost=10,
                                 power_cost=50,
                                 init_power=50,
                                 cargo_space=100,
                                 battery_capacity=150,
                                 charge=1,
                                 move_cost=1,
                                 rubble_movement_cost=0.05,
                                 dig_cost=5, 
                                 self_destruct_cost=5, 
                                 dig_rubble_removed=2, 
                                 dig_resource_gain=2, 
                                 dig_lichen_removed=10, 
                                 rubble_after_destruction=1, 
                                 action_queue_power_cost=1), 
            "heavy": UnitConfig( metal_cost=100,
                                 power_cost=500,
                                 init_power=500,
                                 cargo_space=1000,
                                 battery_capacity=3000,
                                 charge=10,
                                 move_cost=20,
                                 rubble_movement_cost=1,
                                 dig_cost=60,
                                 self_destruct_cost=100,
                                 dig_rubble_removed=20,
                                 dig_resource_gain=20,
                                 dig_lichen_removed=100,
                                 rubble_after_destruction=10,
                                 action_queue_power_cost=10,), 
            }
    )

    @classmethod
    def from_dict(cls, data):
        """Class method"""
        data["robots"]["light"] = UnitConfig(**data["robots"]["light"])
        data["robots"]["heavy"] = UnitConfig(**data["robots"]["heavy"])
        return cls(**data)
