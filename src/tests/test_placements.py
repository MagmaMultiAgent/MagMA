import pytest
import numpy as np

from src.utils import place_near_random_ice

@pytest.mark.parametrize("obs", [
    {
    "board": {
        "ice": np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]),
        "valid_spawns_mask": np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]),
    },
    "teams": {
        "player_1": {"metal": 0, "water": 0},
        "player_2": {"metal": 0, "water": 0},
    },},]
)
def test_place_near_random_ice2(obs):
    assert place_near_random_ice("player_1", obs) == {}
    assert place_near_random_ice("player_2", obs) == {}

@pytest.mark.parametrize("obs", [
    {
    "board": {
        "ice": np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]),
        "valid_spawns_mask": np.array([
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]),
    },
    "teams": {
        "player_1": {"metal": 1, "water": 1},
        "player_2": {"metal": 2, "water": 3},
    },},]
)
def test_place_near_random_ice(obs):
    assert place_near_random_ice("player_1", obs) == {'spawn': [2, 1], 'metal': 1, 'water': 1}
    assert place_near_random_ice("player_2", obs) == {'spawn': [2, 1], 'metal': 2, 'water': 2}

