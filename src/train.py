#%%writefile src/train.py

import argparse
import os
import random
import time
from distutils.util import strtobool
from pprint import pprint
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from policy.net import Net
from policy.simple_net import SimpleNet
from luxenv import LuxSyncVectorEnv,LuxEnv
import tree
import json
import gzip
from kit.load_from_replay import replay_to_state_action, get_obs_action_from_json
from utils import save_args, save_model, load_model, eval_model, _process_eval_resluts, cal_mean_return, make_env

import cProfile

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                    handlers=[logging.StreamHandler()])
stream_handler = [h for h in logging.root.handlers if isinstance(h , logging.StreamHandler)][0]
stream_handler.setLevel(logging.INFO)
stream_handler.setStream(sys.stderr)
logger = logging.getLogger("train")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# Types
TensorPerKey = dict[str, torch.Tensor]
TensorPerPlayer = dict[str, dict[str, torch.Tensor]]

LOG = True
log_from_global_info = [
    'factory_count',
    'unit_count',
    'light_count',
    'heavy_count',
    'unit_ice',
    'unit_ore',
    'unit_water',
    'unit_metal',
    'unit_power',
    'factory_ice',
    'factory_ore',
    'factory_water',
    'factory_metal',
    'factory_power',
    'total_ice',
    'total_ore',
    'total_water',
    'total_metal',
    'total_power',
    'lichen_count',
    'units_on_ice',
    'avg_distance_from_ice',
    'rubble_on_ice',

    'ice_transfered',
    'ore_transfered',
    'ice_mined',
    'ore_mined',
    'lichen_grown',
    'unit_created',
    'light_created',
    'heavy_created',
    'unit_destroyed',
    'light_destroyed',
    'heavy_destroyed',
]

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="LuxAI_S2-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1024,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--train-num-collect", type=int, default=4096,
        help="the number of data collections in training process")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=2,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--save-interval", type=int, default=8192, 
        help="global step interval to save model")
    parser.add_argument("--load-model-path", type=str, default=None,
        help="path for pretrained model loading")
    parser.add_argument("--evaluate-interval", type=int, default=4096,
        help="evaluation steps")
    parser.add_argument("--evaluate-num", type=int, default=1,
        help="evaluation numbers")
    parser.add_argument("--replay-dir", type=str, default=None,
        help="replay dirs to reset state")
    parser.add_argument("--eval", type=bool, default=False,
        help="is eval model")
    
    args = parser.parse_args()

    # Test arguments
    args.num_steps = 200
    args.train_num_collect = args.num_envs*args.num_steps
    args.evaluate_interval = args.train_num_collect

    # Reward per entity
    args.max_entity_number = 1000

    # size of a batch
    args.batch_size = int(args.num_envs * args.num_steps)
    # number of steps to train on from all envs
    args.train_num_collect = args.minibatch_size if args.train_num_collect is None else args.train_num_collect
    # size of a minibatch
    args.minibatch_size = int(args.train_num_collect // args.num_minibatches)
    # how many steps to stop at when collecting data
    args.max_train_step = int(args.train_num_collect // args.num_envs)

    logger.info(args)
    return args


def layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def put_into_store(data: dict, ind: int, store: dict, max_train_step: int, num_envs: int, device: torch.device):
    for key, value in data.items():
        if isinstance(value, dict):
            if key not in store:
                store[key] = {}
            put_into_store(value, ind, store[key], max_train_step, num_envs, device)
        else:
            if key not in store:
                store[key] = torch.zeros((max_train_step, num_envs) + value.shape[1:], dtype=value.dtype).to(device)
            store[key][ind] = value


def reset_store(store: dict):
    for key, value in store.items():
        if isinstance(value, dict):
            reset_store(store[key])
        else:
            if store[key].dtype in {torch.float32, torch.float64, torch.int32, torch.int64}:
                store[key][:] = 0
            elif store[key].dtype in {torch.bool}: 
                store[key][:] = False
            else:
                raise NotImplementedError(f"store[key].dtype={store[key].dtype}")


def create_model(device: torch.device, eval: bool, load_model_path: Union[str, None], evaluate_num: int, learning_rate: float, writer: Union[None, SummaryWriter] = None):
    """
    Create the model
    """
    agent = SimpleNet().to(device)
    # agent = Net().to(device)
    if load_model_path is not None:
        agent.load_state_dict(torch.load(load_model_path))
        print('load successfully')
        if eval:
            import sys
            for i in range(10):
                eval_results = []
                for _ in range(evaluate_num):
                    eval_results.append(eval_model(agent))
                eval_results = _process_eval_resluts(eval_results)
                if LOG:
                    for key, value in eval_results.items():
                        if writer:
                            writer.add_scalar(f"eval/{key}", value, i)
                pprint(eval_results)
            sys.exit()
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    return agent, optimizer


def sample_action_for_player(agent: Net, obs: TensorPerKey, valid_action: TensorPerKey, forced_action: Union[TensorPerKey, None] = None):
    """
    Sample action and value from the agent
    """
    logprob, value, action, entropy = agent(
        obs['global_feature'],
        obs['map_feature'],
        obs['factory_feature'],
        obs['unit_feature'],
        obs['location_feature'],
        valid_action,
        forced_action
    )

    return logprob, value, action, entropy


def sample_actions_for_players(envs: LuxSyncVectorEnv,
                               agent: Net,
                               next_obs: TensorPerPlayer
                               ) -> tuple[TensorPerPlayer, TensorPerPlayer, TensorPerPlayer, TensorPerPlayer]:
    """
    Sample action and value for both players
    """
    action = dict()
    valid_action = dict()
    logprob = dict()
    value = dict()

    for player_id, player in enumerate(['player_0', 'player_1']):
        with torch.no_grad():
            _valid_action = envs.get_valid_actions(player_id)
            _valid_action = tree.map_structure(lambda x: np2torch(x, torch.bool), _valid_action)
            
            _logprob, _value, _action, _ = sample_action_for_player(agent, next_obs[player], _valid_action, None)

            action[player] = _action
            valid_action[player] = _valid_action
            logprob[player] = _logprob
            value[player] = _value

    return action, valid_action, logprob, value


def calculate_returns(envs: LuxSyncVectorEnv,
                      agent: Net,
                      next_obs: TensorPerKey,
                      next_done: torch.Tensor,
                      dones: torch.Tensor,
                      rewards: TensorPerKey,
                      values: TensorPerKey,
                      device: torch.device,
                      max_train_step: int,
                      num_envs: int,
                      max_entity_number: int,
                      gamma: float,
                      gae_lambda: float
                      ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Calculate GAE returns
    """
    returns = dict(player_0=torch.zeros((max_train_step, num_envs, max_entity_number)).to(device),player_1=torch.zeros((max_train_step, num_envs, max_entity_number)).to(device))
    advantages = dict(player_0=torch.zeros((max_train_step, num_envs, max_entity_number)).to(device),player_1=torch.zeros((max_train_step, num_envs, max_entity_number)).to(device))
    with torch.no_grad():
        _, _, _, value = sample_actions_for_players(envs, agent, next_obs)

        for player in ['player_0', 'player_1']:
            next_value = value[player]
            next_value = next_value.reshape(1,-1)
            lastgaelam = 0
            for t in reversed(range(max_train_step-1)):
                if t == max_train_step - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[player][t + 1]
                    nextvalues = values[player][t + 1]
                delta = rewards[player][t] + gamma * nextvalues * nextnonterminal - values[player][t]
                advantages[player][t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns[player] = advantages[player] + values[player]
    
    return returns, advantages


def calculate_loss(advantages: torch.Tensor,
                   returns: torch.Tensor,
                   values: torch.Tensor,
                   newvalue: torch.Tensor,
                   entropy: torch.Tensor,
                   ratio: torch.Tensor,
                   max_entity_number: int,
                   clip_vloss: bool,
                   clip_coef: float,
                   ent_coef: float,
                   vf_coef: float
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the loss
    """
    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1, 1000)
    if clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = values + torch.clamp(
            newvalue - values,
            -clip_coef,
            clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    # Entropy loss
    entropy_loss = entropy.mean()

    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

    return loss, pg_loss, entropy_loss, v_loss


def optimize_for_player(player: str,
                        agent: Net,
                        envs: LuxSyncVectorEnv,
                        optimizer: optim.Optimizer,
                        b_inds: torch.Tensor,
                        b_obs: dict[str, list[torch.Tensor]],
                        b_va: dict[str, list[torch.Tensor]],
                        b_actions: dict[str, list[torch.Tensor]],
                        b_logprobs: dict[str, list[torch.Tensor]],
                        b_advantages: dict[str, list[torch.Tensor]],
                        b_returns: dict[str, list[torch.Tensor]],
                        b_values: dict[str, list[torch.Tensor]],
                        max_entity_number: int,
                        train_num_collect: int,
                        minibatch_size: int,
                        clip_vloss: bool,
                        clip_coef: float,
                        norm_adv: bool,
                        ent_coef: float,
                        vf_coef: float,
                        max_grad_norm: float
                        ) -> tuple[float, float, float, float, float, list[float]]:
    """
    Update weights for a player with PPO
    """
    clipfracs = []
    for start in range(0, train_num_collect, minibatch_size):
        end = start + minibatch_size
        mb_inds = b_inds[start:end]

        mb_obs = tree.map_structure(lambda x: x.view(-1, *x.shape[2:])[mb_inds], b_obs[player])
        mb_va = tree.map_structure(lambda x: x.view(-1, *x.shape[2:])[mb_inds], b_va[player])
        mb_actions = tree.map_structure(lambda x: x.view(-1, *x.shape[2:])[mb_inds], b_actions[player])

        newlogprob, newvalue, _, entropy = sample_action_for_player(agent, mb_obs, mb_va, mb_actions)

        logratio = newlogprob - b_logprobs[player][mb_inds]
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

        mb_advantages = b_advantages[player][mb_inds]
        if norm_adv:
            if len(mb_inds)==1:
                mb_advantages = mb_advantages
            else:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        mb_returns = b_returns[player][mb_inds]
        mb_values = b_values[player][mb_inds]

        # get top 10 lrgest logprob and use it to index every other tensor
        mb_advantages_abs = torch.abs(mb_advantages)
        _, indices = torch.topk(mb_advantages_abs, 1000, dim=1)
        mb_advantages = torch.gather(mb_advantages, 1, indices)
        mb_returns = torch.gather(mb_returns, 1, indices)
        mb_values = torch.gather(mb_values, 1, indices)
        newvalue = torch.gather(newvalue, 1, indices)
        entropy = torch.gather(entropy, 1, indices)
        ratio = torch.gather(ratio, 1, indices)

        loss, pg_loss, entropy_loss, v_loss = calculate_loss(mb_advantages, mb_returns, mb_values, newvalue, entropy, ratio, max_entity_number, clip_vloss, clip_coef, ent_coef, vf_coef)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        return v_loss, pg_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs


def main(args, device):
    player_id = 0
    enemy_id = 1 - player_id
    player = f'player_{player_id}'
    enemy = f'player_{enemy_id}'
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    save_path = f'../results/{run_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if LOG:
        writer = SummaryWriter(f"../results/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None

    save_args(args, save_path+'args.json')

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env setup
    envs = LuxSyncVectorEnv(
        [make_env(i, args.seed + i, args.replay_dir, device=device) for i in range(args.num_envs)],
        device=device
    )
    
    # Create model
    agent, optimizer = create_model(device, args.eval, args.load_model_path, args.evaluate_num, args.learning_rate, writer)

    # Start the game
    global_step = 0
    last_save_model_step = 0
    last_eval_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size

    # Init value stores for PPO
    obs = {}
    actions = {}
    valid_actions = {}
    logprobs = dict(player_0=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number)).to(device), player_1=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number)).to(device))
    rewards = dict(player_0=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number)).to(device),player_1=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number)).to(device))
    dones = dict(player_0=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number)).to(device),player_1=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number)).to(device))
    values = dict(player_0=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number)).to(device),player_1=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number)).to(device))
    
    logger.info("Starting train")
    for update in range(1, num_updates + 1):

        logger.info(f"Update {update} / {num_updates}")

        # Reset envs, get obs
        next_obs, _ = envs.reset()
        next_obs = tree.map_structure(lambda x: np2torch(x, torch.float32), next_obs)
        next_done = torch.zeros(args.num_envs, 2, args.max_entity_number).to(device)
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Init stats
        total_return = 0.0
        episode_return = np.zeros(args.num_envs)
        episode_return_list = []
        step_counts = np.zeros(args.num_envs)
        episode_lengths = []
        episode_sub_return = {}
        episode_sub_return_list = []
        train_step = -1
        global_info_save = {}
        
        for step in range(0, args.num_steps):

            if (step+1) % (args.num_steps / 8) == 0:
                logger.info(f"Step {step + 1} / {args.num_steps}")

            train_step += 1 
            global_step += 1 * args.num_envs

            # Save obervations for PPO
            for player_id, player in enumerate(['player_0', 'player_1']):
                for env_id in range(0, args.num_envs):
                    # insert tensor [env, player, entity] into [player, step, env, entity]
                    dones[player][train_step, env_id] = next_done[env_id, player_id]
            put_into_store(next_obs, train_step, obs, args.max_train_step, args.num_envs, device)

            # Sample actions
            action, valid_action, logprob, value = sample_actions_for_players(envs, agent, next_obs)

            # Save actions for PPO
            put_into_store(action, train_step, actions, args.max_train_step, args.num_envs, device)
            put_into_store(valid_action, train_step, valid_actions, args.max_train_step, args.num_envs, device)
            for player in ['player_0', 'player_1']:
                logprobs[player][train_step] = logprob[player]
                values[player][train_step] = value[player]

            # Step environment
            _action = {}
            for player_id, player in enumerate(['player_0', 'player_1']):
                _action[player_id] = action[player]
            action = tree.map_structure(lambda x: torch2np(x, np.int32), _action)
            del _action
            next_obs, reward, terminated, truncation, info = envs.step(action)
            next_obs = tree.map_structure(lambda x: np2torch(x, torch.float32), next_obs)

            # reward is shape (env, player, group)
            episode_return += np.mean(np.sum(reward, axis=-1), axis=-1)

            step_counts += 1

            reward = np2torch(reward, torch.float32)

            done = terminated | truncation
            # all entities done for a player, at least one player is done
            _done = done.all(axis=-1).any(-1)
            next_done = np2torch(done, torch.bool)

            # Save rewards for PPO
            for player_id, player in enumerate(['player_0', 'player_1']):
                rewards[player][train_step] = reward[:, player_id]

            # Save stats
            if _done.any():
                episode_return_list.append((episode_return[np.where(_done==True)]).mean())
                episode_return[np.where(_done==True)] = 0
                episode_lengths.append(step_counts[np.where(_done==True)].mean())
                step_counts[np.where(_done==True)] = 0
                tmp_sub_return_dict = {}
                for key in episode_sub_return:
                    tmp_sub_return_dict.update({key: np.mean(episode_sub_return[key][np.where(_done==True)])})
                    episode_sub_return[key][np.where(_done==True)] = 0
                episode_sub_return_list.append(tmp_sub_return_dict)

            total_return += cal_mean_return(info['agents'], player_id=0)
            total_return += cal_mean_return(info['agents'], player_id=1)

            if (step == args.num_steps-1):
                logger.info(f"global_step={global_step}, total_return={np.mean(episode_return_list)}, episode_length={np.mean(episode_lengths)}")
                if LOG:
                    writer.add_scalar("charts/episodic_total_return", np.mean(episode_return_list), global_step)
                    writer.add_scalar("charts/episodic_length", np.mean(episode_lengths), global_step)
                    mean_episode_sub_return = {}
                    for key in episode_sub_return.keys():
                        mean_episode_sub_return[key] = np.mean(list(map(lambda sub: sub[key], episode_sub_return_list)))
                        writer.add_scalar(f"sub_reward/{key}", mean_episode_sub_return[key], global_step)
                    global_info_log = {
                        "total": {},
                        "player_0": {},
                        "player_1": {}
                    }
                    for key in log_from_global_info:
                        for env_id in range(args.num_envs):
                            for player in ["player_0", "player_1"]:
                                if key not in global_info_log[player]:
                                    global_info_log[player][key] = []
                                if key not in global_info_log["total"]:
                                    global_info_log["total"][key] = []
                                global_info_log[player][key].append(info[player][env_id][key])
                                global_info_log["total"][key].append(info[player][env_id][key])
                    for groupname, group in global_info_log.items():
                        for key, value in group.items():
                            writer.add_scalar(f"global_info/{groupname}_{key}", sum(value)/len(value), global_step)

                    for groupname, group in global_info_save.items():
                        for key, value in group.items():
                            multiplier = (1 / args.num_envs) if groupname != "total" else (1 / (args.num_envs * 2))
                            writer.add_scalar(f"global_info/sum_{groupname}_{key}", value * multiplier, global_step)
                            global_info_save[groupname][key] = 0
                        
            else:
                for key in log_from_global_info:
                    for env_id in range(args.num_envs):
                        for player in ["player_0", "player_1"]:
                            if player not in global_info_save:
                                global_info_save[player] = {}
                            if "total" not in global_info_save:
                                global_info_save["total"] = {}
                            if key not in global_info_save[player]:
                                global_info_save[player][key] = 0
                            if key not in global_info_save["total"]:
                                global_info_save["total"][key] = 0
                            
                            global_info_save[player][key] += info[player][env_id][key]
                            global_info_save["total"][key] += info[player][env_id][key]
                


            # Train with PPO
            if train_step >= args.max_train_step-1 or step == args.num_steps-1:  
                logger.info("Training with PPO")
                returns, advantages = calculate_returns(envs, agent, next_obs, next_done, dones, rewards, values, device, args.max_train_step, args.num_envs, args.max_entity_number, args.gamma, args.gae_lambda)

                # flatten the batch
                b_obs = obs
                b_actions = actions
                b_va = valid_actions
                
                b_logprobs = tree.map_structure(lambda x: x.view(-1, args.max_entity_number), logprobs)
                b_advantages = tree.map_structure(lambda x: x.view(-1, args.max_entity_number), advantages)
                b_returns = tree.map_structure(lambda x: x.view(-1, args.max_entity_number), returns)
                b_values = tree.map_structure(lambda x: x.view(-1, args.max_entity_number), values)

                # Optimizing the policy and value network
                b_inds = np.arange(args.train_num_collect)
                clipfracs = []
                for epoch in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    _b_inds = np2torch(b_inds, torch.long)
                    for player_id, player in enumerate(['player_0', 'player_1']):
                        v_loss, pg_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs = optimize_for_player(player, agent, envs, optimizer, _b_inds, b_obs, b_va, b_actions, b_logprobs, b_advantages, b_returns, b_values, args.max_entity_number, args.train_num_collect, args.minibatch_size, args.clip_vloss, args.clip_coef, args.norm_adv, args.ent_coef, args.vf_coef, args.max_grad_norm)
                        clipfracs += clipfracs

                        if args.target_kl is not None:
                            if approx_kl > args.target_kl:
                                break
                        
                        y_pred, y_true = b_values[player].cpu().numpy(), b_returns[player].cpu().numpy()
                        var_y = np.var(y_true)
                        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                        # TRY NOT TO MODIFY: record rewards for plotting purposes
                        if LOG:
                            writer.add_scalar(f"losses/value_loss_{player_id}", v_loss.item(), global_step)
                            writer.add_scalar(f"losses/policy_loss_{player_id}", pg_loss.item(), global_step)
                            writer.add_scalar(f"losses/entropy_{player_id}", entropy_loss.item(), global_step)
                            writer.add_scalar(f"losses/old_approx_kl_{player_id}", old_approx_kl.item(), global_step)
                            writer.add_scalar(f"losses/approx_kl_{player_id}", approx_kl.item(), global_step)
                            writer.add_scalar(f"losses/clipfrac_{player_id}", np.mean(clipfracs), global_step)
                            writer.add_scalar(f"losses/explained_variance_{player_id}", explained_var, global_step)
                
                if LOG:
                    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("charts/SPS", round(global_step / (time.time() - start_time), 2), global_step)
                
                logger.info(f"SPS: {round(global_step / (time.time() - start_time), 2)}")
                logger.info(f"global step: {global_step}")

                reset_store(obs)
                reset_store(actions)
                reset_store(valid_actions)

                for player_id, player in enumerate(['player_0', 'player_1']):
                    logprobs[player][:] = 0
                    rewards[player][:] = 0
                    dones[player][:] = 0
                    values[player][:] = 0

                train_step = -1
            
            # Evaluate
            if (global_step - last_eval_step) >= args.evaluate_interval:
                eval_results = []
                for _ in range(args.evaluate_num):
                    eval_results.append(eval_model(agent))
                eval_results = _process_eval_resluts(eval_results)
                if LOG:
                    for key, value in eval_results.items():
                        writer.add_scalar(f"eval/{key}", value, global_step)
                pprint(eval_results)
                last_eval_step = global_step
            
            # Save model
            if (global_step - last_save_model_step) >= args.save_interval:
                save_model(agent, save_path+f'model_{global_step}.pth')
                last_save_model_step = global_step
    envs.close()
    if LOG:
        writer.close()


if __name__ == "__main__":
    args = parse_args()

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Device: {device}")

    np2torch = lambda x, dtype: torch.tensor(x).type(dtype).to(device)
    torch2np = lambda x, dtype: x.cpu().numpy().astype(dtype)
 
    main(args, device)
