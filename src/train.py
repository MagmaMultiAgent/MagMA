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
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from wrappers.reward_wrapper import EarlyRewardParserWrapper
from wrappers.monitor_wrapper import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.ppo import PPO
from sb3_contrib.ppo_mask import MaskablePPO
from controller.controller import MultiUnitController
from wrappers.obs_wrappers import SimpleUnitObservationWrapper
from wrappers.sb3_iam_wrapper import SB3InvalidActionWrapper
from wrappers.utils import gaussian_ice_placement, bid_zero_to_not_waste
from net.mixed_net import UNetWithResnet50Encoder

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
    parser.add_argument("-s", "--seed", type=int, default=42, help="seed for training")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel envs to run. Note that the rollout \
        size is configured separately and invariant to this value",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=4096,
        help="Number of timesteps between evaluations",
    )
    parser.add_argument(
        "--eval-num",
        type=int,
        default=12,
        help="Number of episodes to evaluate the policy on",
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
        default=500000,
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





def make_env(env_id: str, rank: int, max_episode_steps: int = 1024, seed: int = 42):
    """
    Creates the environment
    """

    def _init() -> gym.Env:
        """
        Initializes the environment
        """

        env = gym.make(env_id, verbose=1, collect_stats=True, disable_env_checker=True, max_episode_steps=max_episode_steps)

        env = SB3InvalidActionWrapper(
            env,
            factory_placement_policy=gaussian_ice_placement,
            bid_policy=bid_zero_to_not_waste,
            controller=MultiUnitController(env.env_cfg),
        )
        env = SimpleUnitObservationWrapper(
            env
        )
        env = EarlyRewardParserWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        set_random_seed(seed)
        return env

    return _init


class TensorboardCallback(BaseCallback):
    """
    Callback for logging metrics to tensorboard
    """

    def __init__(self, tag: str, verbose=0):
        """
        Initializes the callback
        """

        super().__init__(verbose)
        self.tag = tag

    def _on_step(self) -> bool:
        """
        Called on every step
        """
        count = 0
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                count += 1
                for k in info["metrics"]:
                    stat = info["metrics"][k]
                    self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True


def evaluate(args, env_id, model):
    """
    Evaluates the model
    """

    model = model.load(args.model_path)
    video_length = 1024
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, max_episode_steps=1024) for i in range(args.eval_num)]
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


def train(args, env_id, model: PPO):
    """
    Trains the model
    """

    eval_environments = [make_env(env_id, i, max_episode_steps=1024) for i in range(args.eval_num)]
    eval_env = SubprocVecEnv(eval_environments)
    eval_env.reset()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models"),
        log_path=osp.join(args.log_path, "eval_logs"),
        eval_freq=args.eval_interval,
        deterministic=False,
        render=False,
        n_eval_episodes=args.eval_num,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=4096,
        save_path=osp.join(args.log_path, "models"),
        name_prefix="model",
    )

    model.learn(
        args.total_timesteps,
        callback=[TensorboardCallback(tag="train_metrics"), eval_callback, checkpoint_callback],
    )
    model.save(osp.join(args.log_path, "models/latest_model"))


def main(args):
    """
    Main function
    """

    if args.seed is not None:
        set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"

    environments = [make_env(env_id, i, max_episode_steps=args.max_episode_steps) \
                    for i in range(args.n_envs)]

    env = SubprocVecEnv(environments)
    env.reset()

    policy_kwargs_unit = {
        "features_extractor_class": UNetWithResnet50Encoder,
        "features_extractor_kwargs": {
            "output_channels": 20,
            }
        }
    rollout_steps = 4096
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        n_steps=rollout_steps // args.n_envs,
        batch_size=512,
        learning_rate=2e-4,
        policy_kwargs=policy_kwargs_unit,
        verbose=1,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=osp.join(args.log_path),
    )
    if args.eval:
        evaluate(args, env_id, model)
    else:
        train(args, env_id, model)


if __name__ == "__main__":
    main(parse_args())
