from src.wrappers.utils import place_near_random_ice
import pytest

obs = {
    "board": {
        "ice": [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "valid_spawns_mask": [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    },
    "teams": {
        "player_1": {"metal": 0, "water": 0},
        "player_2": {"metal": 0, "water": 0},
    },
}

obs2 = {
    "board": {
        "ice": [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "valid_spawns_mask": [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    },
    "teams": {
        "player_1": {"metal": 1, "water": 1},
        "player_2": {"metal": 1, "water": 1},
    },
}

def test_place_near_random_ice2(obs2):
    assert place_near_random_ice("player_1", obs2) == {"spawn": (2, 2), "metal": 1, "water": 1}
    assert place_near_random_ice("player_2", obs2) == {"spawn": (2, 2), "metal": 1, "water": 1}

def test_place_near_random_ice(obs):
    assert place_near_random_ice("player_1", obs) == {}
    assert place_near_random_ice("player_2", obs) == {}

