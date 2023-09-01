"""
This file contains the configuration for the game. \
It is used to set the constants for the game.
"""

import dataclasses
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List
# pylint: disable=C0103

def convert_dict_to_ns(x_keys):
    """
    Converts a dictionary to a namespace
    """

    if isinstance(x_keys, dict):
        for key in x_keys:
            x_keys[key] = convert_dict_to_ns(x_keys)
        return Namespace(x_keys)
    return Namespace()


@dataclass
class UnitConfig:
    """
    Configuration for a unit
    """

    METAL_COST: int = 100
    POWER_COST: int = 500
    CARGO_SPACE: int = 1000
    BATTERY_CAPACITY: int = 1500
    CHARGE: int = 1
    INIT_POWER: int = 50
    MOVE_COST: int = 1
    RUBBLE_MOVEMENT_COST: float = 1
    DIG_COST: int = 5
    DIG_RUBBLE_REMOVED: int = 1
    DIG_RESOURCE_GAIN: int = 2
    DIG_LICHEN_REMOVED: int = 10
    SELF_DESTRUCT_COST: int = 10
    RUBBLE_AFTER_DESTRUCTION: int = 1
    ACTION_QUEUE_POWER_COST: int = 1


@dataclass
class EnvConfig:
    """
    Configuration for the environment
    """

    max_episode_length: int = 1000
    map_size: int = 48
    verbose: int = 1

    validate_action_space: bool = True

    max_transfer_amount: int = 10000
    MIN_FACTORIES: int = 2
    MAX_FACTORIES: int = 5
    CYCLE_LENGTH: int = 50
    DAY_LENGTH: int = 30
    UNIT_ACTION_QUEUE_SIZE: int = 20

    MAX_RUBBLE: int = 100
    FACTORY_RUBBLE_AFTER_DESTRUCTION: int = 50
    INIT_WATER_METAL_PER_FACTORY: int = 150
    INIT_POWER_PER_FACTORY: int = 1000

    #### LICHEN ####
    MIN_LICHEN_TO_SPREAD: int = 20
    LICHEN_LOST_WITHOUT_WATER: int = 1
    LICHEN_GAINED_WITH_WATER: int = 1
    MAX_LICHEN_PER_TILE: int = 100
    POWER_PER_CONNECTED_LICHEN_TILE: int = 1

    LICHEN_WATERING_COST_FACTOR: int = 10

    BIDDING_SYSTEM: bool = True

    FACTORY_PROCESSING_RATE_WATER: int = 100
    ICE_WATER_RATIO: int = 4
    FACTORY_PROCESSING_RATE_METAL: int = 50
    ORE_METAL_RATIO: int = 5

    FACTORY_CHARGE: int = 50
    FACTORY_WATER_CONSUMPTION: int = 1

    POWER_LOSS_FACTOR: float = 0.5

    #### Units ####
    ROBOTS: Dict[str, UnitConfig] = dataclasses.field(
        default_factory=lambda: dict(
            LIGHT=UnitConfig(
                METAL_COST=10,
                POWER_COST=50,
                INIT_POWER=50,
                CARGO_SPACE=100,
                BATTERY_CAPACITY=150,
                CHARGE=1,
                MOVE_COST=1,
                RUBBLE_MOVEMENT_COST=0.05,
                DIG_COST=5,
                SELF_DESTRUCT_COST=5,
                DIG_RUBBLE_REMOVED=2,
                DIG_RESOURCE_GAIN=2,
                DIG_LICHEN_REMOVED=10,
                RUBBLE_AFTER_DESTRUCTION=1,
                ACTION_QUEUE_POWER_COST=1,
            ),
            HEAVY=UnitConfig(
                METAL_COST=100,
                POWER_COST=500,
                INIT_POWER=500,
                CARGO_SPACE=1000,
                BATTERY_CAPACITY=3000,
                CHARGE=10,
                MOVE_COST=20,
                RUBBLE_MOVEMENT_COST=1,
                DIG_COST=60,
                SELF_DESTRUCT_COST=100,
                DIG_RUBBLE_REMOVED=20,
                DIG_RESOURCE_GAIN=20,
                DIG_LICHEN_REMOVED=100,
                RUBBLE_AFTER_DESTRUCTION=10,
                ACTION_QUEUE_POWER_COST=10,
            ),
        )
    )

    @classmethod
    def from_dict(cls, data):
        """
        Converts a dictionary to a namespace
        """

        data["ROBOTS"]["LIGHT"] = UnitConfig(**data["ROBOTS"]["LIGHT"])
        data["ROBOTS"]["HEAVY"] = UnitConfig(**data["ROBOTS"]["HEAVY"])
        return cls(**data)
