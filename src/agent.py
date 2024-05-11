"""
This file is where your agent's logic is kept. Define a bidding policy, \
factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and \
use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard \
error e.g. print("message", file=sys.stderr)
"""

# pylint: disable=E0401
import os.path as osp
import numpy as np
import torch as th
import sys
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.ppo import PPO
from kit.config import EnvConfig
from controller.controller import SimpleUnitDiscreteController
from wrappers.obs_wrappers import SimpleUnitObservationWrapper
from wrappers.utils import gaussian_ice_placement, bid_zero_to_not_waste

MODEL_WEIGHTS_RELATIVE_PATH = "logs/models/best_model"

class Agent:
    """
    Agent for the RL tutorial. This agent uses a trained PPO agent to play the game.
    """

    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        """
        Initialize the agent
        """
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        self.policy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))

        self.controller = SimpleUnitDiscreteController(self.env_cfg)

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        """
        The bid policy is a function that takes in the current step, observation, \
        and remaining overage time and returns a dictionary of bids for each resource
        """

        return bid_zero_to_not_waste(self.player, obs)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        """
        The factory placement policy is a function that takes in the current step, observation, \
        and remaining overage time and returns a dictionary of factory placement actions
        """

        return gaussian_ice_placement(self.player, step, self.env_cfg, obs)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players
        raw_obs = dict(player_0=obs, player_1=obs)
        obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        obs = th.from_numpy(obs).float()
        with th.no_grad():

            # to improve performance, we have a rule based action mask generator for the controller used
            # which will force the agent to generate actions that are valid only.
            action_mask = (
                th.from_numpy(self.controller.action_masks(self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )
            
            # SB3 doesn't support invalid action masking. So we do it ourselves here
            features = self.policy.policy.features_extractor(obs.unsqueeze(0))
            x = self.policy.policy.mlp_extractor.policy_net(features)
            logits = self.policy.policy.action_net(x) # shape (1, N) where N=12 for the default controller
            logits[~action_mask] = -1e8 # mask out invalid actions
            dist = th.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy() # shape (1, 1)

        # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        lux_action = self.controller.action_to_lux_action(
            self.player, raw_obs, actions[0]
        )
        return lux_action
