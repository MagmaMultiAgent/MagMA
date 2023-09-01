"""
Unit class
"""

# pylint: disable=E0401
import math
from dataclasses import dataclass
from typing import List
import numpy as np
from lux.cargo import UnitCargo
from lux.config import EnvConfig, UnitConfig

move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])


@dataclass
class Unit:
    """
    Unit class
    """

    team_id: int
    unit_id: str
    unit_type: str
    pos: np.ndarray
    power: int
    cargo: UnitCargo
    env_cfg: EnvConfig
    unit_cfg: UnitConfig
    action_queue: List

    @property
    def agent_id(self):
        """
        Returns the agent id
        """

        if self.team_id == 0:
            return "player_0"
        return "player_1"

    def action_queue_cost(self, game_state):
        """
        Returns the cost of the action queue
        """

        cost = self.env_cfg.ROBOTS[self.unit_type].ACTION_QUEUE_POWER_COST
        return cost

    def move_cost(self, game_state, direction):
        """
        Returns the cost of moving in a direction
        """

        board = game_state.board
        target_pos = self.pos + move_deltas[direction]
        if (
            target_pos[0] < 0
            or target_pos[1] < 0
            or target_pos[1] >= len(board.rubble)
            or target_pos[0] >= len(board.rubble[0])
        ):
            return None
        factory_there = board.factory_occupancy_map[target_pos[0], target_pos[1]]
        if (
            factory_there not in game_state.teams[self.agent_id].factory_strains
            and factory_there != -1
        ):
            return None
        rubble_at_target = board.rubble[target_pos[0]][target_pos[1]]

        return math.floor(
            self.unit_cfg.MOVE_COST
            + self.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
        )

    def move(self, direction, repeat=0, num=1):
        """
        Moves in a direction
        """

        if isinstance(direction, int):
            return np.array([0, direction, 0, 0, repeat, num])
        return np.array([0, 0, 0, 0, 0, 0])

    def transfer(
        self, transfer_direction, transfer_resource, transfer_amount, repeat=0, num=1
    ):
        """
        Transfer resources
        """

        assert 0 <= transfer_resource < 5
        assert 0 <= transfer_direction < 5
        return np.array(
            [1, transfer_direction, transfer_resource, transfer_amount, repeat, num]
        )

    def pickup(self, pickup_resource, pickup_amount, repeat=0, num=1):
        """
        Pickup resources
        """

        assert 0 <= pickup_resource < 5
        return np.array([2, 0, pickup_resource, pickup_amount, repeat, num])

    def dig_cost(self, game_state):
        """
        Returns the cost of digging
        """

        return self.unit_cfg.DIG_COST

    def dig(self, repeat=0, num=1):
        """
        Digs
        """

        return np.array([3, 0, 0, 0, repeat, num])

    def self_destruct_cost(self, game_state):
        """
        Returns the cost of self destructing
        """

        return self.unit_cfg.SELF_DESTRUCT_COST

    def self_destruct(self, repeat=0, num=1):
        """
        Self destructs
        """

        return np.array([4, 0, 0, 0, repeat, num])

    def recharge(self, x_val, repeat=0, num=1):
        """
        Recharges
        """

        return np.array([5, 0, 0, x_val, repeat, num])

    def __str__(self) -> str:
        """
        Returns the string representation of the unit
        """

        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.pos}"
        return out
