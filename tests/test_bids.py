from ..src.wrappers.utils import zero_bid
import pytest


def test_zero_bid():
    assert zero_bid("player_1") == {"bid": 0, "faction": "MotherMars"}
    assert zero_bid("player_2") == {"bid": 0, "faction": "AlphaStrike"}

@pytest.mark.parametrize("player", ["player_1", "player_2"])
def test_zero_bid_parametrized(player):
    
    faction = "AlphaStrike"
    if player == "player_1":
        faction = "MotherMars"
    return {"bid": 0, "faction": faction}