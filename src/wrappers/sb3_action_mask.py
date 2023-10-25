"""
This file contains a wrapper that adds action masks to the environment \
for use with stable-baselines3
"""
import copy
from typing import Dict, Callable
import gymnasium as gym
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict, StatsStateDict
from luxai_s2.wrappers import SB3Wrapper
from luxai_s2.unit import BidActionType, FactoryPlacementActionType
from action.controllers import SimpleUnitDiscreteController
from reward.early_reward_parser import EarlyRewardParser

class SB3InvalidActionWrapper(SB3Wrapper):
    """
    This wrapper adds action masks to the environment for use with stable-baselines3
    """

    def __init__(
        self,
        env: LuxAI_S2,
        bid_policy: Callable[
            [str, ObservationStateDict], Dict[str, BidActionType]
        ] = None,
        factory_placement_policy: Callable[
            [str, ObservationStateDict], Dict[str, FactoryPlacementActionType]
        ] = None,
        controller: SimpleUnitDiscreteController = None,
    ) -> None:
        """
        This wrapper adds action masks to the environment for use with stable-baselines3
        """

        super().__init__(env, bid_policy, factory_placement_policy, controller)

    def action_masks(self):
        """
        Generates a boolean action mask indicating in each \
        discrete dimension whether it would be valid or not
        """
        return self.controller.action_masks('player_0', self.prev_obs)
