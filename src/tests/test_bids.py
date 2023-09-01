'''
Test the bid policy.
'''
from src.agent import Agent

def test_bid_policy():
    '''
    Test that the bid policy returns a valid bid.
    '''
    mock_agent = Agent(None, None)
    assert mock_agent.bid_policy(0, None, 60) == {"faction": "AlphaStrike", "bid": 0}
