from src.utils import zero_bid


def test_zero_bid():
    assert zero_bid("player_1") == {"bid": 0, "faction": "MotherMars"}
    assert zero_bid("player_2") == {"bid": 0, "faction": "AlphaStrike"}