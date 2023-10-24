"""
Module responsible for creating a wrapper for stable baselines"
"""

from typing import Callable, Dict
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import BidActionType, FactoryPlacementActionType
from luxai_s2.utils import my_turn_to_place_factory
from action.controllers import Controller

import logging
logger = logging.getLogger(__name__)

class SB3Wrapper(gym.Wrapper):
    """
    A environment wrapper for Stable Baselines 3. It reduces the LuxAI_S2 env
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
        A environment wrapper for Stable Baselines 3. It reduces the LuxAI_S2 env
        into a single phase game and places the first two phases
        (bidding and factory placement) into the env.reset function so that
        interacting agents directly start generating actions to play the
        third phase of the game.

        It also accepts a Controller that translates action's in one action
        space to a Lux S2 compatible action

        Parameters
        ----------
        bid_policy: Function
            A function accepting player: str and obs: ObservationStateDict
            as input that returns a bid action such as
            dict(bid=10, faction="AlphaStrike"). By default will bid 0
        factory_placement_policy: Function
            A function accepting player: str and obs: ObservationStateDict
            as input that returns a factory placement action such as
            dict(spawn=np.array([2, 4]), metal=150, water=150).
            By default will spawn in a random valid location with metal=150, water=150
        controller : Controller
            A controller that parameterizes the action space into something
            more usable and converts parameterized actions to lux actions.
            See luxai_s2/wrappers/controllers.py for available controllers
            and how to make your own
        """
        logger.info(f"Creating {self.__class__.__name__}")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")

        gym.Wrapper.__init__(self, env)
        self.env = env

        assert controller is not None

        self.controller = controller
        self.action_space = controller.action_space

        if factory_placement_policy is None:
            def factory_placement_policy(player, obs: ObservationStateDict):
                potential_spawns = np.array(
                    list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
                )
                spawn_loc = potential_spawns[
                    np.random.randint(0, len(potential_spawns))
                ]
                return {"spawn": spawn_loc, "metal": 150, "water": 150}

        self.factory_placement_policy = factory_placement_policy
        if bid_policy is None:
            def bid_policy(player, obs: ObservationStateDict):
                faction = "AlphaStrike"
                if player == "player_1":
                    faction = "MotherMars"
                return {"bid": 0, "faction": faction}

        self.bid_policy = bid_policy
        self.prev_obs = None

    def step(self, action: Dict[str, npt.NDArray]):
        """
        Takes as input a dictionary mapping agent name to an action and returns
        """
        self.logger.debug(f"Stepping environment with action\n{action}")

        lux_action = {}
        for agent in self.env.agents:
            if agent in action:
                lux_action[agent] = self.controller.unit_action_to_lux_action(
                    agent=agent, obs=self.prev_obs, action=action[agent]
                )
            else:
                lux_action[agent] = {}

        obs, reward, done, info = self.env.step(lux_action)
        self.prev_obs = obs
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Resets the environment and returns the initial observation
        """
        self.logger.info("Resetting")

        obs = self.env.reset(**kwargs)

        action = {}
        for agent in self.env.agents:
            action[agent] = self.bid_policy(agent, obs[agent])
        obs, _, _, _ = self.env.step(action)

        while self.env.state.real_env_steps < 0:
            action = {}
            for agent in self.env.agents:
                if my_turn_to_place_factory(
                    obs["player_0"]["teams"][agent]["place_first"],
                    self.env.state.env_steps,
                ):
                    action[agent] = self.factory_placement_policy(agent, obs[agent])
                else:
                    action[agent] = {}
            obs, _, _, _ = self.env.step(action)
        self.prev_obs = obs

        return obs
