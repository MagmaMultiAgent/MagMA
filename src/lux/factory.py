"""
Factory class
"""

# pylint: disable=E0401
from dataclasses import dataclass
import numpy as np
from lux.cargo import UnitCargo
from lux.config import EnvConfig

@dataclass
class Factory:
    """
    Factory class
    """

    team_id: int
    unit_id: str
    strain_id: int
    power: int
    cargo: UnitCargo
    pos: np.ndarray
    lichen_tiles: np.ndarray
    env_cfg: EnvConfig

    def build_heavy_metal_cost(self, game_state):
        """
        Metal required to build heavy
        """

        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.METAL_COST

    def build_heavy_power_cost(self, game_state):
        """
        Power required to build heavy
        """

        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.POWER_COST

    def can_build_heavy(self, game_state):
        """
        Checks if heavy can be built
        """

        return self.power >= self.build_heavy_power_cost(
            game_state
        ) and self.cargo.metal >= self.build_heavy_metal_cost(game_state)

    def build_heavy(self):
        """
        Builds heavy
        """

        return 1

    def build_light_metal_cost(self, game_state):
        """
        Metal required to build light
        """

        unit_cfg = self.env_cfg.ROBOTS["LIGHT"]
        return unit_cfg.METAL_COST

    def build_light_power_cost(self, game_state):
        """
        Power required to build light
        """

        unit_cfg = self.env_cfg.ROBOTS["LIGHT"]
        return unit_cfg.POWER_COST

    def can_build_light(self, game_state):
        """
        Checks if light can be built
        """

        return self.power >= self.build_light_power_cost(
            game_state
        ) and self.cargo.metal >= self.build_light_metal_cost(game_state)

    def build_light(self):
        """
        Builds light
        """

        return 0

    def water_cost(self, game_state):
        """
        Water required to perform water action
        """

        owned_lichen_tiles = (game_state.board.lichen_strains == self.strain_id).sum()
        return np.ceil(owned_lichen_tiles / self.env_cfg.LICHEN_WATERING_COST_FACTOR)

    def can_water(self, game_state):
        """
        Checks if water can be performed
        """

        return self.cargo.water >= self.water_cost(game_state)

    def water(self):
        """
        Waters lichen
        """

        return 2

    @property
    def pos_slice(self):
        """
        Returns the slice of the map that the factory occupies
        """

        return slice(self.pos[0] - 1, self.pos[0] + 2), slice(
            self.pos[1] - 1, self.pos[1] + 2
        )
