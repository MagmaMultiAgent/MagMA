"""Module for representing a factory"""
from dataclasses import dataclass

import numpy as np

from lux.cargo import UnitCargo
from lux.config import EnvConfig


@dataclass
class Factory:
    """Dataclass containing factory config"""

    team_id: int
    unit_id: str
    strain_id: int
    power: int
    cargo: UnitCargo
    pos: np.ndarray
    # lichen tiles connected to this factory
    # lichen_tiles: np.ndarray
    env_cfg: EnvConfig

    def build_heavy_metal_cost(self):
        """Function returning metal cost of building a heavy"""
        unit_cfg = self.env_cfg.robots["heavy"]
        return unit_cfg.metal_cost

    def build_heavy_power_cost(self):
        """Function returning power cost of building a heavy"""
        unit_cfg = self.env_cfg.robots["heavy"]
        return unit_cfg.power_cost

    def can_build_heavy(self):
        """Function returning if metal and power are enough for heavy"""
        return self.power >= self.build_heavy_power_cost() \
            and self.cargo.metal >= self.build_heavy_metal_cost()

    def build_heavy(self):
        """Function returning 1"""
        return 1

    def build_light_metal_cost(self):
        """Function returning metal cost of building a light"""
        unit_cfg = self.env_cfg.robots["light"]
        return unit_cfg.metal_cost

    def build_light_power_cost(self):
        """Function returning power cost of building a light"""
        unit_cfg = self.env_cfg.robots["light"]
        return unit_cfg.power_cost

    def can_build_light(self):
        """Function returning if metal and power are enough for light"""
        return self.power >= self.build_light_power_cost() \
        and self.cargo.metal >= self.build_light_metal_cost()

    def build_light(self):
        """Function returning 0"""
        return 0

    def water_cost(self, game_state):
        """
        Water required to perform water action
        """
        owned_lichen_tiles = (game_state.board.lichen_strains == self.strain_id).sum()
        return np.ceil(owned_lichen_tiles / self.env_cfg.lichen_watering_cost_factor)

    def can_water(self, game_state):
        """Function returning whether factory has enough water to water"""
        return self.cargo.water >= self.water_cost(game_state)

    def water(self):
        """Function returning 2"""
        return 2

    @property
    def pos_slice(self):
        """Property containing sliced position"""
        return slice(self.pos[0] - 1, self.pos[0] + 2), slice(
            self.pos[1] - 1, self.pos[1] + 2
        )
