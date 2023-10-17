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
from torch import nn
from gymnasium.wrappers import TimeLimit
from luxai_s2.state import StatsStateDict
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.ppo import PPO
from sb3_contrib.ppo_mask import MaskablePPO
from wrappers.controllers import SimpleUnitDiscreteController
from wrappers.obs_wrappers import SimpleUnitObservationWrapper
from wrappers.sb3_action_mask import SB3InvalidActionWrapper
from net.net import CustomResNet
from parsers.reward_parser import DenseRewardParser
import lux.kit
import numpy as np

global_information_names = [
            'player_factory_count',
            'player_light_count',
            'player_heavy_count',
            'player_unit_ice',
            'player_unit_ore',
            'player_unit_water',
            'player_unit_metal',
            'player_unit_power',
            'player_factory_ice',
            'player_factory_ore',
            'player_factory_water',
            'player_factory_metal',
            'player_factory_power',
            'player_total_ice',
            'player_total_ore',
            'player_total_water',
            'player_total_metal',
            'player_total_power',
            'player_lichen_count',
        ]

def get_global_info(player: str, obs: lux.kit.GameState):

    global_info = {k: None for k in global_information_names}

    factories = list(obs.factories[player].values())
    units = list(obs.units[player].values())

    global_info['player_light_count'] = sum(int(unit.unit_type == 'LIGHT') for unit in units)
    global_info['player_heavy_count'] = sum(int(unit.unit_type == 'HEAVY') for unit in units)
    global_info['player_factory_count'] = len(factories)

    global_info['player_unit_ice'] = sum(unit.cargo.ice for unit in units)
    global_info['player_unit_ore'] = sum(unit.cargo.ore for unit in units)
    global_info['player_unit_water'] = sum(unit.cargo.water for unit in units)
    global_info['player_unit_metal'] = sum(unit.cargo.metal for unit in units)
    global_info['player_unit_power'] = sum(unit.power for unit in units)

    global_info['player_factory_ice'] = sum(f.cargo.ice for f in factories)
    global_info['player_factory_ore'] = sum(f.cargo.ore for f in factories)
    global_info['player_factory_water'] = sum(f.cargo.water for f in factories)
    global_info['player_factory_metal'] = sum(f.cargo.metal for f in factories)
    global_info['player_factory_power'] = sum(f.power for f in factories)

    global_info['player_total_ice'] = global_info['player_unit_ice'] + global_info['player_factory_ice']
    global_info['player_total_ore'] = global_info['player_unit_ore'] + global_info['player_factory_ore']
    global_info['player_total_water'] = global_info['player_unit_water'] + global_info['player_factory_water']
    global_info['player_total_metal'] = global_info['player_unit_metal'] + global_info['player_factory_metal']
    global_info['player_total_power'] = global_info['player_unit_power'] + global_info['player_factory_power']

    lichen = obs.board.lichen
    lichen_strains = obs.board.lichen_strains

    #if factories:
        #lichen_count = sum((np.sum(lichen[lichen_strains == factory.strain_id]) for factory in factories), 0)
        #global_info['lichen_count'] = lichen_count
    #else:
    global_info['lichen_count'] = 0
    return global_info

class SurvivalRewardParser(gym.Wrapper):
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
        self.reward_parser = DenseRewardParser()

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
        obs, _, termination, truncation, info = self.env.step(action)
        done = dict()
        for k in termination:
            done[k] = termination[k] | truncation[k]
        #obs = obs[agent]
        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions

        #stats: StatsStateDict = self.env.state.stats[agent]
        stats: StatsStateDict = self.env.state.stats
        global_info_own = get_global_info(agent, self.env.state)
        global_info_opp = get_global_info(opp_agent, self.env.state)

        global_info = {"player_0": global_info_own, "player_1": global_info_opp}

        self.reward_parser.reset(global_info, stats)
        rev1, rev2 = self.reward_parser.parse(done, self.env.state, stats, global_info)
        # Below is where you get to have some fun with reward design! we provide a simple reward function that rewards digging ice and producing water
        
        stats: StatsStateDict = self.env.state.stats[agent]
        info = dict()
        metrics = dict()
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["water_produced"] = stats["generation"]["water"]

        # we save these two to see how often the agent updates robot action queues and how often the robot has enough
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
        reward = rev1[0]
        obs = obs[agent]
        return obs, reward, termination[agent], truncation[agent], info

    def reset(self, **kwargs):
        """
        Resets the environment
        """
        obs, reset_info = self.env.reset(**kwargs)
        self.prev_step_metrics = None
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
            env
        )
        env = SurvivalRewardParser(env)
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
    invalid_action_masking = True

    environments = [make_env(env_id, i, max_episode_steps=args.max_episode_steps) \
                    for i in range(args.n_envs)]
    env = DummyVecEnv(environments) if invalid_action_masking \
        else SubprocVecEnv(environments)
    env.reset()

    policy_kwargs = {
        "features_extractor_class": CustomResNet,
        "features_extractor_kwargs": {
            "features_dim": 256,
            }
        }
    rollout_steps = 4000
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
