"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 \
are packages not available during the competition running (ATM)
"""

# pylint: disable=E0401
import copy
import argparse
import os.path as osp
import gym
import torch as th
from torch import nn
from gym.wrappers import TimeLimit
from luxai_s2.state import StatsStateDict
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.ppo import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.ppo_mask import MaskablePPO
from wrappers.controllers import SimpleUnitDiscreteController
from wrappers.obs_wrappers import SimpleUnitObservationWrapper
from wrappers.sb3_action_mask import SB3InvalidActionWrapper

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for extracting features from the observation space
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
        """
        Initializes the CNN
        """

        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
        )

        with th.no_grad():
            n_flatten = self._get_flattened_size(observation_space)

        self.fully_conv = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
        )

    def _get_flattened_size(self, observation_space: gym.Space) -> int:
        """
        Returns the size of the flattened output of the CNN
        """

        dummy_input = th.zeros(1, 1, observation_space.shape[0])
        cnn_output = self.cnn(dummy_input)
        return cnn_output.view(1, -1).shape[1]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        '''
        Forward pass of the CNN
        '''
        observations = observations.unsqueeze(1)
        cnn_output = self.cnn(observations)
        flattened = cnn_output.view(cnn_output.size(0), -1)
        return self.fully_conv(flattened)


class CustomEnvWrapper(gym.Wrapper):
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

    def step(self, action):
        """
        Steps the environment
        """

        agent = "player_0"
        opp_agent = "player_1"

        opp_factories = self.env.state.factories[opp_agent]
        for k in opp_factories.keys():
            factory = opp_factories[k]
            factory.cargo.water = 1000

        action = {agent: action}
        obs, _, done, info = self.env.step(action)
        obs = obs[agent]
        done = done[agent]

        stats: StatsStateDict = self.env.state.stats[agent]

        info = {}
        metrics = {}
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["water_produced"] = stats["generation"]["water"]
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        info["metrics"] = metrics

        reward = 0
        if self.prev_step_metrics is not None:

            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = (
                metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            )

            reward = ice_dug_this_step / 100 + water_produced_this_step

        self.prev_step_metrics = copy.deepcopy(metrics)
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Resets the environment
        """

        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        return obs

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
    parser.add_argument("-s", "--seed", type=int, default=12, help="seed for training")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel envs to run. Note that the rollout \
        size is configured separately and invariant to this value",
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

        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)

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
        )
        env = CustomEnvWrapper(env)
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
        [make_env(env_id, i, max_episode_steps=1000) for i in range(args.n_envs)]
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

    # eval_env = SubprocVecEnv(
    #     [make_env(env_id, i, max_episode_steps=1000) for i in range(4)]
    # )
    eval_environments = [make_env(env_id, i, max_episode_steps=1000) for i in range(4)]
    eval_env = DummyVecEnv(eval_environments) if invalid_action_masking \
        else SubprocVecEnv(eval_environments)
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
    """
    Main function
    """

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
    environments = [make_env(env_id, i, max_episode_steps=args.max_episode_steps) \
                    for i in range(args.n_envs)]
    env = DummyVecEnv(environments) if invalid_action_masking \
        else SubprocVecEnv(environments)
    env.reset()

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {
            "features_dim": 128
            }
        }
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
    main(parse_args())
