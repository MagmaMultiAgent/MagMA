"""
This file is where your agent's logic is kept. Define a bidding policy, \
factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and \
use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard \
error e.g. print("message", file=sys.stderr)
"""

import os.path as osp
import numpy as np
import torch as th
from sb3_contrib.ppo_mask import MaskablePPO
from lux.config import EnvConfig
from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper

MODEL_WEIGHTS_RELATIVE_PATH = "best_model"

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
        self.policy = MaskablePPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))

        self.controller = SimpleUnitDiscreteController(self.env_cfg)

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        """
        The bid policy is a function that takes in the current step, observation, \
        and remaining overage time and returns a dictionary of bids for each resource
        """

        return {"faction": 'AlphaStrike', "bid": 0}

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        """
        The factory placement policy is a function that takes in the current step, observation, \
        and remaining overage time and returns a dictionary of factory placement actions
        """

        if obs["teams"][self.player]["metal"] == 0:
            return {}
        potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
        potential_spawns_set = set(potential_spawns)
        done_search = False

        ice_diff = np.diff(obs["board"]["ice"])
        pot_ice_spots = np.argwhere(ice_diff == 1)
        if len(pot_ice_spots) == 0:
            pot_ice_spots = potential_spawns
        trials = 5
        while trials > 0:
            pos_idx = np.random.randint(0, len(pot_ice_spots))
            pos = pot_ice_spots[pos_idx]

            area = 3
            for x_coord in range(area):
                for y_coord in range(area):
                    check_pos = [pos[0] + x_coord - area // 2, pos[1] + y_coord - area // 2]
                    if tuple(check_pos) in potential_spawns_set:
                        done_search = True
                        pos = check_pos
                        break
                if done_search:
                    break
            if done_search:
                break
            trials -= 1
        spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        if not done_search:
            pos = spawn_loc

        metal = obs["teams"][self.player]["metal"]
        return {"spawn": pos, "metal": metal, "water": metal}

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        The act policy is a function that takes in the current step, observation, \
        and remaining overage time and returns a dictionary of actions for each unit
        """

        raw_obs = {"player_0": obs, "player_1": obs}
        obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        obs = th.from_numpy(obs).float()
        with th.no_grad():

            action_mask = (
                th.from_numpy(self.controller.action_masks(self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )

            features = self.policy.policy.features_extractor(obs.unsqueeze(0))
            x = self.policy.policy.mlp_extractor.shared_net(features)
            logits = self.policy.policy.action_net(x)

            logits[~action_mask] = -1e8
            dist = th.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy()

        lux_action = self.controller.action_to_lux_action(
            self.player, raw_obs, actions[0]
        )

        # commented code below adds watering lichen which can easily improve your agent
        # shared_obs = raw_obs[self.player]
        # factories = shared_obs["factories"][self.player]
        # for unit_id in factories.keys():
        #     factory = factories[unit_id]
        #     if 1000 - step < 50 and factory["cargo"]["water"] > 100:
        #         lux_action[unit_id] = 2 # water and grow lichen at the very end of the game

        return lux_action
