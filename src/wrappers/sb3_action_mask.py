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
from luxai_s2.wrappers.controllers import Controller

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
        controller: Controller = None,
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

class CustomEnvWrapper(gym.Wrapper):
    """
    Adds a custom reward and turns the LuxAI_S2 environment into a single-agent \
    environment for easy training
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent \
        environment for easy training
        """

        super().__init__(env)
        self.prev_step_metrics = None

    def step(self, action):
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent \
        environment for easy training
        """
        agent = "player_0"
        opp_agent = "player_1"

        opp_factories = self.env.state.factories[opp_agent]
        for k in opp_factories.keys():
            factory = opp_factories[k]
            factory.cargo.water = 1000

        action = {agent: action}
        obs, _, done, info = self.env.step(action)
        obs = obs[agent]
        done = done[agent]

        stats: StatsStateDict = self.env.state.stats[agent]

        info = {}
        metrics = {}
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["water_produced"] = stats["generation"]["water"]
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]
        info["metrics"] = metrics

        reward = 0
        if self.prev_step_metrics is not None:
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = (
                metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            )
            reward = ice_dug_this_step / 100 + water_produced_this_step

        self.prev_step_metrics = copy.deepcopy(metrics)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        return obs
