"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 are packages not available during the competition running (ATM)
"""


import copy
import os.path as osp

import gym
import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces
from gym.wrappers import TimeLimit
from luxai_s2.state import ObservationStateDict, StatsStateDict
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from wrappers.sb3 import SB3Wrapper
from wrappers.sb3_action_mask import SB3InvalidActionWrapper
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
)
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.ppo import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
        )
        
        # Calculate the size of the flattened output after CNN layers
        with th.no_grad():
            n_flatten = self._get_flattened_size(observation_space)
        
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
        )

    def _get_flattened_size(self, observation_space: gym.Space) -> int:
        dummy_input = th.zeros(1, 1, observation_space.shape[0])  # Batch size of 1, 1 channel
        cnn_output = self.cnn(dummy_input)
        return cnn_output.view(1, -1).shape[1]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Reshape observations to (batch_size, channels, sequence_length)
        observations = observations.unsqueeze(1)
        cnn_output = self.cnn(observations)
        flattened = cnn_output.view(cnn_output.size(0), -1)
        return self.fc(flattened)


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training
        """
        super().__init__(env)
        self.prev_step_metrics = None

    def step(self, action):
        agent = "player_0"
        opp_agent = "player_1"

        opp_factories = self.env.state.factories[opp_agent]
        for k in opp_factories.keys():
            factory = opp_factories[k]
            # set enemy factories to have 1000 water to keep them alive the whole around and treat the game as single-agent
            factory.cargo.water = 1000

        # submit actions for just one agent to make it single-agent
        # and save single-agent versions of the data below
        action = {agent: action}
        obs, _, done, info = self.env.step(action)
        obs = obs[agent]
        done = done[agent]

        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions
        stats: StatsStateDict = self.env.state.stats[agent]

        info = dict()
        metrics = dict()
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["water_produced"] = stats["generation"]["water"]

        # we save these two to see often the agent updates robot action queues and how often enough
        # power to do so and succeed (less frequent updates = more power is saved)
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
        info["metrics"] = metrics

        reward = 0
        if self.prev_step_metrics is not None:
            # we check how much ice and water is produced and reward the agent for generating both
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = (
                metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            )
            # we reward water production more as it is the most important resource for survival
            reward = ice_dug_this_step / 100 + water_produced_this_step

        self.prev_step_metrics = copy.deepcopy(metrics)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        return obs


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple script that simplifies Lux AI Season 2 as a single-agent environment with a reduced observation and action space. It trains a policy that can succesfully control a heavy unit to dig ice and transfer it back to a factory to keep it alive"
    )
    parser.add_argument("-s", "--seed", type=int, default=12, help="seed for training")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel envs to run. Note that the rollout size is configured separately and invariant to this value",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=200,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=3_000_000,
        help="Total timesteps for training",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, will only evaluate a given policy. Otherwise enters training mode",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to SB3 model weights to use for evaluation"
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
    def _init() -> gym.Env:
        # verbose = 0
        # collect stats so we can create reward functions
        # max factories set to 2 for simplification and keeping returns consistent as we survive longer if there are more initial resources
        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)

        # Add a SB3 wrapper to make it work with SB3 and simplify the action space with the controller
        # this will remove the bidding phase and factory placement phase. For factory placement we use
        # the provided place_near_random_ice function which will randomly select an ice tile and place a factory near it.

        # env = SB3Wrapper(
        #     env,
        #     factory_placement_policy=place_near_random_ice,
        #     controller=SimpleUnitDiscreteController(env.env_cfg),
        # )

        env = SB3InvalidActionWrapper(
            env,
            factory_placement_policy=place_near_random_ice,
            controller=SimpleUnitDiscreteController(env.env_cfg),
        )

        env = SimpleUnitObservationWrapper(
            env
        )  # changes observation to include a few simple features
        env = CustomEnvWrapper(env)  # convert to single agent, add our reward
        env = TimeLimit(
            env, max_episode_steps=max_episode_steps
        )  # set horizon to 100 to make training faster. Default is 1000
        env = Monitor(env)  # for SB3 to allow it to record metrics
        env.reset(seed=seed + rank)
        set_random_seed(seed)
        return env

    return _init


class TensorboardCallback(BaseCallback):
    def __init__(self, tag: str, verbose=0):
        super().__init__(verbose)
        self.tag = tag

    def _on_step(self) -> bool:
        c = 0

        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                c += 1
                for k in info["metrics"]:
                    stat = info["metrics"][k]
                    self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True


def save_model_state_dict(save_path, model):
    # save the policy state dict for kaggle competition submission
    state_dict = model.policy.to("cpu").state_dict()
    th.save(state_dict, save_path)


def evaluate(args, env_id, model):
    model = model.load(args.model_path)
    video_length = 1000  # default horizon
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, max_episode_steps=1000) for i in range(args.n_envs)]
    )
    eval_env = VecVideoRecorder(
        eval_env,
        osp.join(args.log_path, "eval_videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"evaluation_video",
    )
    eval_env.reset()
    out = evaluate_policy(model, eval_env, render=False, deterministic=False)
    print(out)


def train(args, env_id, model: PPO, invalid_action_masking):
    # eval_env = SubprocVecEnv(
    #     [make_env(env_id, i, max_episode_steps=1000) for i in range(4)]
    # )
    eval_environments = [make_env(env_id, i, max_episode_steps=1000) for i in range(4)]
    eval_env = DummyVecEnv(eval_environments) if invalid_action_masking else SubprocVecEnv(eval_environments)
    eval_env.reset()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models"),
        log_path=osp.join(args.log_path, "eval_logs"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )

    model.learn(
        args.total_timesteps,
        callback=[TensorboardCallback(tag="train_metrics"), eval_callback],
    )
    model.save(osp.join(args.log_path, "models/latest_model"))


def main(args):
    print("Training with args", args)
    if args.seed is not None:
        set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"
    invalid_action_masking=True
    # env = SubprocVecEnv(
    #     [
    #         make_env(env_id, i, max_episode_steps=args.max_episode_steps)
    #         for i in range(args.n_envs)
    #     ]
    # )
    environments = [make_env(env_id, i, max_episode_steps=args.max_episode_steps) for i in range(args.n_envs)]
    env = DummyVecEnv(environments) if invalid_action_masking else SubprocVecEnv(environments)
    env.reset()

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    rollout_steps = 4000
    #policy_kwargs = dict(net_arch=(128, 128))
    model = MaskablePPO(
        "CnnPolicy",
        env,
        n_steps=rollout_steps // args.n_envs,
        batch_size=800,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        verbose=1,
        target_kl=0.05,
        gamma=0.99,
        tensorboard_log=osp.join(args.log_path),
    )
    if args.eval:
        evaluate(args, env_id, model)
    else:
        train(args, env_id, model, invalid_action_masking)


if __name__ == "__main__":
    # python ../examples/sb3.py -l logs/exp_1 -s 42 -n 1
    main(parse_args())
