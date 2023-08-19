"""Module containing the running agent"""
import os.path as osp
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO
from submission_kit.lux.config import EnvConfig
from submission_kit.wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary.
# Make sure to exclude the .zip extension here.
MODEL_WEIGHTS_RELATIVE_PATH = "./logs/exp_2/models/best_model"

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

    def bid_policy(self):
        """Function used for factory bidding"""

        return {"faction": 'AlphaStrike', "bid": 0}

    def factory_placement_policy(self, obs):
        """Function used for placing the actual factory after bidding"""

        if obs["teams"][self.player]["metal"] == 0:
            return {}

        potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
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
                    if tuple(check_pos) in set(potential_spawns):
                        done_search = True
                        pos = check_pos
                        break
                if done_search:
                    break
            if done_search:
                break
            trials -= 1
        spawn_loc = potential_spawns[int(np.random.randint(0, len(potential_spawns)))]
        if not done_search:
            pos = spawn_loc

        metal = obs["teams"][self.player]["metal"]
        return {"spawn": pos, "metal": metal, "water": metal}

    def act(self, obs):
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
        # shared_obs = raw_obs[self.player]
        # factories = shared_obs["factories"][self.player]
        # for unit_id in factories.keys():
        #     factory = factories[unit_id]
        #     if 1000 - step < 50 and factory["cargo"]["water"] > 100:
        #         lux_action[unit_id] = 2 # water and grow lichen at the very end of the game

        return lux_action
