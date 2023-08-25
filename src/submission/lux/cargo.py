"""
Module containing cargo dataclass
"""
from dataclasses import dataclass


@dataclass
class UnitCargo:
    """
    Dataclass containing all possible cargo
    """
    ice: int = 0
    ore: int = 0
    water: int = 0
    metal: int = 0
