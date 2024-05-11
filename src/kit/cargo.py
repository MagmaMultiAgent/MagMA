"""
Cargo class for Lux AI
"""

from dataclasses import dataclass

@dataclass
class UnitCargo:
    """
    Cargo class for Lux AI
    """

    ice: int = 0
    ore: int = 0
    water: int = 0
    metal: int = 0
