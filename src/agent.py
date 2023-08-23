"""Module containing the running agent"""
import os.path as osp
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO
from objects.config import EnvConfig
from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary.
# Make sure to exclude the .zip extension here.
MODEL_WEIGHTS_RELATIVE_PATH = "./best_model"

class Agent:
    """Class representing an agent"""

    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        self.policy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))

        self.controller = SimpleUnitDiscreteController(self.env_cfg)

    def act(self, step: int, obs):
        """Function used for action selection by the agent"""
        
        raw_obs = {"player_0": obs, "player_1": obs}
        obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        # pylint: disable=E1101
        obs = th.from_numpy(obs).float()
        # pylint: enable=E1101
        with th.no_grad():

            action_mask = (
                # pylint: disable=E1101
                th.from_numpy(self.controller.action_masks(self.player, raw_obs))
                # pylint: enable=E1101
                .unsqueeze(0)
                .bool()
            )
            features = self.policy.policy.features_extractor(obs.unsqueeze(0))
            x_coord = self.policy.policy.mlp_extractor.shared_net(features)
            logits = self.policy.policy.action_net(x_coord)

            logits[~action_mask] = -1e8 # mask out invalid actions
            dist = th.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy() # shape (1, 1)

        lux_action = self.controller.action_to_lux_action(
            self.player, raw_obs, actions[0]
        )

        # commented code below adds watering lichen which can easily improve your agent
        shared_obs = raw_obs[self.player]
        factories = shared_obs["factories"][self.player]
        for unit_id in factories.keys():
            factory = factories[unit_id]
            if 1000 - step < 50 and factory["cargo"]["water"] > 100:
                lux_action[unit_id] = 2 # water and grow lichen at the very end of the game

        return lux_action
