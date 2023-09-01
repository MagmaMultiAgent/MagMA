'''
Tests for the placement policies
'''
import pytest
import numpy as np
from src.agent import Agent

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
def test_factory_placement_policy2(obs):
    '''
    Test that the factory placement policy returns a valid placement.
    '''
    mock_agent = Agent(None, None)
    mock_agent.player = "player_1"
    assert mock_agent.factory_placement_policy(0, obs, 60) == {}
    mock_agent.player = "player_2"
    assert mock_agent.factory_placement_policy(0, obs, 60) == {}

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
def test_factory_placement_policy(obs):
    '''
    Test that the factory placement policy returns a valid placement.
    '''
    mock_agent = Agent(None, None)
    mock_agent.player = "player_1"
    assert mock_agent.factory_placement_policy(0, obs, 60) == {'spawn': [2, 1], 'metal': 1, 'water': 1}
    mock_agent.player = "player_2"
    assert mock_agent.factory_placement_policy(0, obs, 60) == {'spawn': [2, 1], 'metal': 2, 'water': 2}

