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
import seeding
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
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
from net.bottleneck_net import BottleNet
from net.dash_net import DashNet
import torch as th
th.autograd.set_detect_anomaly(True)

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
        default=16,
        help="Number of parallel envs to run. Note that the rollout \
        size is configured separately and invariant to this value",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=8192,
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
        default=256,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="Seed for evaluation",
    )
    parser.add_argument(
        "--eval-max-episode-steps",
        type=int,
        default=1024,
        help="Seed for evaluation",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=8192*25,
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
    parser.add_argument(
        "--net",
        type=str,
        default="bottle",
        help="Type of neural network to use",
    )
    parser.add_argument(
        "--bottleneck-channels",
        type=int,
        default=256,
        help="Number of bottleneck channels",
    )
    parser.add_argument(
        "--bottleneck-type",
        type=str,
        default="regular",
        help="Type of bottleneck",
    )
    args = parser.parse_args()
    return args





def make_env(env_id: str, rank: int, max_episode_steps: int = 256, seed: int = 42, eval_seed = 0):
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


def train(args, env_id, model: PPO):
    """
    Trains the model
    """
    seeding.set_seed(args.seed)
    eval_environments = [make_env(env_id, i, max_episode_steps=args.eval_max_episode_steps, seed=args.eval_seed) for i in range(args.eval_num)]
    eval_env = SubprocVecEnv(eval_environments)
    eval_env.reset()
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models"),
        log_path=osp.join(args.log_path, "eval_logs"),
        eval_freq=args.eval_interval,
        deterministic=False,
        render=False,
        n_eval_episodes=args.eval_num,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=8192,
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

    seeding.set_seed(args.seed)
    environments = [make_env(env_id, i, max_episode_steps=args.max_episode_steps, seed=args.seed) for i in range(args.n_envs)]

    env = SubprocVecEnv(environments)
    env.reset()

    if args.net == "bottle":
        policy_kwargs_unit = {
            "features_extractor_class": BottleNet,
            "features_extractor_kwargs": {
                "output_channels": 16,
                "num_bottleneck_channels": args.bottleneck_channels,
                "bottleneck_type": args.bottleneck_type,
                "seed": args.seed,
                }
            }
    else:
        policy_kwargs_unit = {
            "features_extractor_class": DashNet,
            "features_extractor_kwargs": {
                "output_channels": 16,
                "seed": args.seed,
                }
            }

    rollout_steps = 8192
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
        ent_coef=0.001,
        max_grad_norm=0.5,
        tensorboard_log=osp.join(args.log_path),
    )

    train(args, env_id, model)


if __name__ == "__main__":
    main(parse_args())
