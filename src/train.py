"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 \
are packages not available during the competition running (ATM)
"""

# pylint: disable=E0401
import copy
import argparse
import os.path as osp
import gymnasium as gym
import torch as th
from gymnasium.wrappers import TimeLimit
from luxai_s2.state import StatsStateDict
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from stable_baselines3.common.callbacks import (
    BaseCallback
)
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder, CustomDummyVecEnv
from stable_baselines3.ppo import PPO
from sb3_contrib.ppo_mask import MaskablePPO
from action.controllers import SimpleUnitDiscreteController
from wrappers.obs_wrappers import SimpleUnitObservationWrapper
from wrappers.sb3_action_mask import SB3InvalidActionWrapper
from net.net import SimpleEntityNet
from reward.early_reward_parser import EarlyRewardParser

import random

import sys
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)
logger.debug('Creating logger')

logging.setLoggerClass


class EarlyRewardParserWrapper(gym.Wrapper):
    """
    Custom wrapper for the LuxAI_S2 environment
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment \
        into a single-agent environment for easy training
        """
        super().__init__(env)
        self.prev_step_metrics = None
        self.reward_parser = EarlyRewardParser()

        # controller to get action mask
        self.controller = SimpleUnitDiscreteController(self.env.env_cfg)

    def step(self, action):

        agent = "player_0"
        opp_agent = "player_1"

        factories = self.env.state.factories[opp_agent]
        for k in factories.keys():
            factory = factories[k]
            factory.cargo.water = 1000
        
        factories = self.env.state.factories[agent]
        units = self.env.state.units[agent]

        assert self.prev_obs
        prev_obs = self.prev_obs
        prev_mask = self.controller.action_masks(agent, prev_obs)

        action = {agent: action}
        obs, _, termination, truncation, info = self.env.step(action)
        done = dict()
        for k in termination:
            done[k] = termination[k] | truncation[k]
        obs = obs[agent]

        stats: StatsStateDict = self.env.state.stats[agent]

        global_info_own = self.reward_parser.get_global_info(agent, self.env.state)
        self.reward_parser.reset(global_info_own, stats)
        reward = self.reward_parser.parse(self.env.state, stats, global_info_own)

        stats: StatsStateDict = self.env.state.stats[agent]
        info = dict()
        metrics = dict()
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["ore_dug"] = (
            stats["generation"]["ore"]["HEAVY"] + stats["generation"]["ore"]["LIGHT"]
        )
        metrics["power_consumed"] = (
            stats["consumption"]["power"]["HEAVY"] + stats["consumption"]["power"]["LIGHT"] + stats["consumption"]["power"]["FACTORY"]
        )
        metrics["water_consumed"] = stats["consumption"]["water"]
        metrics["metal_consumed"] = stats["consumption"]["metal"]
        metrics["rubble_destroyed"] = (
            stats["destroyed"]["rubble"]["LIGHT"] + stats["destroyed"]["rubble"]["HEAVY"]
        )
        metrics["ice_transferred"] = stats["transfer"]["ice"]
        metrics["ore_transferred"] = stats["transfer"]["ore"]
        metrics["water_transferred"] = stats["transfer"]["water"]
        metrics["energy_transferred"] = stats["transfer"]["power"]
        metrics["energy_pickup"] = stats["pickup"]["power"]
        metrics["light_robots_built"] = stats["generation"]["built"]["LIGHT"]
        metrics["heavy_robots_built"] = stats["generation"]["built"]["HEAVY"]
        metrics["light_power"] = stats["generation"]["power"]["LIGHT"]
        metrics["heavy_power"] = stats["generation"]["power"]["HEAVY"]
        metrics["factory_power"] = stats["generation"]["power"]["FACTORY"]
        metrics["metal_produced"] = stats["generation"]["metal"]
        metrics["water_produced"] = stats["generation"]["water"]
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

    

        # Give rewards
        reward = 0
        if self.prev_step_metrics is not None:
            # we check how much ice and water is produced and reward the agent for generating both
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = (
                metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            )
            # we reward water production more as it is the most important resource for survival
            reward = ice_dug_this_step / 100 + water_produced_this_step

            ice_transferred_this_step = metrics["ice_transferred"] - self.prev_step_metrics["ice_transferred"]

            if ice_transferred_this_step or water_produced_this_step:
                print(f"Transferred ice: {ice_transferred_this_step}, Water produced: {water_produced_this_step}, Reward: {reward}", file=sys.stderr)
        
        self.prev_step_metrics = copy.deepcopy(metrics)
        

        # Record actions
        action = action[agent]
        factories = prev_obs[agent]["factories"][agent]
        units = prev_obs[agent]["units"][agent]
        mask_size = prev_mask.shape[0]
        action_size = action.shape[0]
        entity_size = len(factories) + len(units)
        
        assert action_size >= mask_size
        assert mask_size == entity_size

        action = action[:entity_size]

        actions = {}
        for action_name, action_id in self.controller.ACTION_NAME_TO_ID.items():
            if action_name in self.controller.DISABLED_ACTIONS:
                continue

            # Add actions counts
            prev_value = 0
            if self.prev_actions and action_name in self.prev_actions:
                prev_value = self.prev_actions[action_name]
            actions[action_name] = prev_value
            actions[action_name] += (action == action_id).sum()

            # Add action availability counts (not masked actions)
            action_availability_name = f"{action_name}_availability"
            prev_value = 0
            if self.prev_actions and action_availability_name in self.prev_actions:
                prev_value = self.prev_actions[action_availability_name]
            actions[action_availability_name] = prev_value
            actions[action_availability_name] += prev_mask[:, action_id].sum()
        
        self.prev_actions = copy.deepcopy(actions)

        info["metrics"] = metrics
        info["actions"] = actions
        return obs, reward, termination[agent], truncation[agent], info

    def reset(self, **kwargs):
        """
        Resets the environment
        """
        obs, reset_info = self.env.reset(**kwargs)
        self.prev_step_metrics = None
        self.prev_actions = None
        return obs["player_0"], reset_info


def parse_args():
    """
    Parses the arguments
    """

    parser = argparse.ArgumentParser(
        description="Simple script that simplifies Lux AI Season 2 as a single-agent \
        environment with a reduced observation and action space. It trains a policy \
        that can succesfully control a heavy unit to dig ice and transfer it back to \
        a factory to keep it alive"
    )
    parser.add_argument("-s", "--seed", type=int, default=666, help="seed for training")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=2,
        help="Number of parallel envs to run. Note that the rollout \
        size is configured separately and invariant to this value",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1000,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5_000_000,
        help="Total timesteps for training",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, will only evaluate a given policy. \
            Otherwise enters training mode",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to SB3 model \
            weights to use for evaluation"
    )
    parser.add_argument(
        "-l",
        "--log-path",
        type=str,
        default="logs",
        help="Logging path",
    )
    args = parser.parse_args()
    return args


def make_env(env_id: str, rank: int, seed: int = 0, max_episode_steps=100):
    """
    Creates the environment
    """

    def _init() -> gym.Env:
        """
        Initializes the environment
        """

        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=4, disable_env_checker=True)

        env = SB3InvalidActionWrapper(
            env,
            factory_placement_policy=place_near_random_ice,
            controller=SimpleUnitDiscreteController(env.env_cfg),
        )

        env = SimpleUnitObservationWrapper(
            env,
            ind=rank
        )
        env = EarlyRewardParserWrapper(env)
        env = TimeLimit(
            env, max_episode_steps=max_episode_steps
        )
        env = Monitor(env)        
        env.reset(seed=seed + rank)
        set_random_seed(seed)

        return env

    return _init


class TensorboardCallback(BaseCallback):
    """
    Callback for logging metrics to tensorboard
    """

    def __init__(self, tag: str, info_name: str, verbose=0):
        """
        Initializes the callback
        """

        super().__init__(verbose)
        self.tag = tag
        self.info_name = info_name

    def _on_step(self) -> bool:
        """
        Called on every step
        """
        count = 0

        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                count += 1
                for k in info[self.info_name]:
                    stat = info[self.info_name][k]
                    self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True

def save_model_state_dict(save_path, model):
    """
    Saves the model state dict
    """

    state_dict = model.policy.to("cpu").state_dict()
    th.save(state_dict, save_path)


def evaluate(args, env_id, model):
    """
    Evaluates the model
    """

    model = model.load(args.model_path)
    video_length = 1000
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, seed=random.random(), max_episode_steps=1000) for i in range(args.n_envs)]
    )
    eval_env = VecVideoRecorder(
        eval_env,
        osp.join(args.log_path, "eval_videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="evaluation_video",
    )
    eval_env.reset()
    out = evaluate_policy(model, eval_env, render=False, deterministic=False)
    print(out)


def train(args, env_id, model: PPO, invalid_action_masking):
    """
    Trains the model
    """

    eval_environments = [make_env(env_id, i, max_episode_steps=1000) for i in range(4)]
    eval_env = CustomDummyVecEnv(eval_environments)
    eval_env.reset()
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models"),
        log_path=osp.join(args.log_path, "eval_logs"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
        use_masking=True
    )

    model.learn(
        args.total_timesteps,
        callback=[TensorboardCallback(tag="train_metrics", info_name="metrics"),
                  TensorboardCallback(tag="train_actions", info_name="actions"),
                  eval_callback],
    )
    model.save(osp.join(args.log_path, "models/latest_model"))


def main(args):
    """
    Main function
    """

    if args.seed is not None:
        set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"
    invalid_action_masking = True

    environments = [make_env(env_id, i, max_episode_steps=args.max_episode_steps) \
                    for i in range(args.n_envs)]

    # Change: needed to make a custom environment to handle the changing observation space 🙃
    env = CustomDummyVecEnv(environments)
    env.reset()

    policy_kwargs = {
        "features_extractor_class": SimpleEntityNet,
        "features_extractor_kwargs": {
                "action_dim": SimpleUnitDiscreteController.total_act_dims
            }
        }
    rollout_steps = 512
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        n_steps=rollout_steps // args.n_envs,
        batch_size=512,
        learning_rate=0.001,
        policy_kwargs=policy_kwargs,
        verbose=1,
        target_kl=None,
        gamma=0.99,
        tensorboard_log=osp.join(args.log_path),
        n_epochs=2,
        ent_coef=0.001,
    )
    if args.eval:
        evaluate(args, env_id, model)
    else:
        train(args, env_id, model, invalid_action_masking)


if __name__ == "__main__":
    main(parse_args())
