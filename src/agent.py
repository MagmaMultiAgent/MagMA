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
import seeding
import numpy as np
import torch as th
import sys
from sb3_contrib.ppo_mask import MaskablePPO
from kit.config import EnvConfig
from controller.controller import MultiUnitController
from wrappers.obs_wrappers import SimpleUnitObservationWrapper
from wrappers.utils import gaussian_ice_placement, bid_zero_to_not_waste
from observation.observation_parser import FeatureParser

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
        seeding.set_seed(42)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        self.policy = MaskablePPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))
        self.controller = MultiUnitController(self.env_cfg)
        self.observation_parser = FeatureParser()

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
        """
        The act policy is a function that takes in the current step, observation, \
        and remaining overage time and returns a dictionary of actions for each unit
        """

        raw_obs = {"player_0": obs, "player_1": obs}
        obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg, observation_parser=self.observation_parser)
        obs = obs[self.player]

        tensors = {}
        for key, value in obs.items():
            val = th.from_numpy(value).unsqueeze(0).float()
            val = val.to('cuda')
            tensors[key] = val
        
        with th.no_grad():

            action_mask = (
                th.from_numpy(self.controller.action_masks(self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )

            features = self.policy.policy.features_extractor(tensors)
            x_step = self.policy.policy.mlp_extractor.policy_net(features)
            
            batch_size, feature_dim, height, width = x_step.shape
    
            
            x_step = x_step.permute(0, 2, 3, 1)
            x_step = x_step.reshape(-1, feature_dim)

            logits = self.policy.policy.action_net(x_step)

            action_mask = action_mask.permute(0, 2, 3, 1)
            action_mask = action_mask.reshape(-1, feature_dim)

            logits[~action_mask] = -1e8
            dist = th.distributions.Categorical(logits=logits)


            actions = dist.sample()
            actions = actions.reshape(batch_size, height, width)
            actions = actions.permute(0, 1, 2).cpu().numpy()

        lux_action = self.controller.action_to_lux_action(
            self.player, raw_obs, actions[0]
        )

        return lux_action
