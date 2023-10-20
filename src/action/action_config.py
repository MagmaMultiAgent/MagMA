from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Dict
from typing_extensions import Literal


@dataclass
class ActDims:
    factory_act: int = 4
    robot_act: int = 7
    direction: int = 5
    resource: int = 5
    amount: int = 10
    repeat: int = 2
    bid: int = 11


class UnitActChannel(IntEnum):
    TYPE = 0
    DIRECTION = 1
    RESOURCE = 2
    AMOUNT = 3
    REPEAT = 4
    N = 5



@dataclass
class ModelParam:
    action_emb_dim: int = 6
    action_queue_size: int = 20
    global_emb_dim: int = 10
    global_feature_dims: int = 32
    map_channel: int = 30
    n_res_blocks: int = 4
    all_channel: int = 64
    amount_distribution: str = "categorical"  # "beta" or "categorical"
    spawn_distribution: str = "beta"  # "beta" or "categorical" or "normal" or "script"

    actor_head: str = "full"  # "full" or "simple"


class UnitActType(IntEnum):
    MOVE = 0
    TRANSFER = 1
    PICKUP = 2
    DIG = 3
    SELF_DESTRUCT = 4
    RECHARGE = 5
    DO_NOTHING = 6

    @classmethod
    def get_value(cls, s: str):
        return cls.__members__[s.upper()]


class FactoryActType(IntEnum):
    BUILD_LIGHT = 0
    BUILD_HEAVY = 1
    WATER = 2
    DO_NOTHING = 3


class ResourceType(IntEnum):
    ICE = 0
    ORE = 1
    WATER = 2
    METAL = 3
    POWER = 4


@dataclass
class UnitAct:
    act_type: int = ActDims.robot_act
    # action type 0
    move: Dict[str, int] = field(default_factory=lambda: {
        "direction": ActDims.direction,
        "repeat": ActDims.repeat,
    })

    # action type 1
    transfer: Dict[str, int] = field(default_factory=lambda: {
        "direction": ActDims.direction,
        "resource": ActDims.resource,
        "amount": ActDims.amount,
        "repeat": ActDims.repeat,
    })

    # action type 2
    pickup: Dict[str, int] = field(default_factory=lambda: {
        "resource": ActDims.resource,
        "amount": ActDims.amount,
        "repeat": ActDims.repeat,
    })

    # action type 3
    dig: Dict[str, int] = field(default_factory=lambda: {
        "repeat": ActDims.repeat,
    })

    # action type 4
    self_destruct: Dict[str, int] = field(default_factory=lambda: {
        "repeat": ActDims.repeat,
    })

    # action type 5
    recharge: Dict[str, int] = field(default_factory=lambda: {
        "amount": ActDims.amount,
        "repeat": ActDims.repeat,
    })

    # action type 6
    do_nothing: Dict = field(default_factory=lambda: {})


@dataclass
class FullAct:
    factory_act: Dict[str, int] = ActDims.factory_act
    unit_act: UnitAct = UnitAct()
    bid: int = ActDims.bid
    factory_spawn: int = 1
