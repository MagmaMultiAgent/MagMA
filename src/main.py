"""Module containing the main loop"""
import json
from argparse import Namespace
from submission_kit.lux.config import EnvConfig
from submission_kit.lux.kit import (
    process_action,
    process_obs,
)
from agent import Agent


### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = (
    {}
)  # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = {}


def agent_fn(observ, config):
    """
    agent definition for kaggle submission.
    """
    step = observ.step

    player = observ.player
    remaining_overage_time = observ.remaining_overage_time
    if step == 0:
        env_cfg = EnvConfig.from_dict(config["env_cfg"])
        agent_dict[player] = Agent(player, env_cfg)
        agent_prev_obs[player] = {}
        agent = agent_dict[player]
    agent = agent_dict[player]
    obs_tmp = process_obs(agent_prev_obs[player], step, json.loads(observ.obs))
    agent_prev_obs[player] = obs_tmp
    agent.step = step
    if step == 0:
        act = agent.bid_policy(step, obs_tmp, remaining_overage_time)
    elif obs_tmp["real_env_steps"] < 0:
        act = agent.factory_placement_policy(step, obs_tmp, remaining_overage_time)
    else:
        act = agent.act(step, obs_tmp, remaining_overage_time)

    return process_action(act)


if __name__ == "__main__":

    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof) from eof

    i = 0
    while True:
        inputs = read_input()
        obs = json.loads(inputs)

        observation = Namespace(
            {"step": obs['step'], "obs": json.dumps(obs['obs']),
             "remaining_overage_time": obs['remaining_overage_time'],
             "player": obs['player'], "info":obs['info']}
        )
        if i == 0:
            configurations = obs["info"]["env_cfg"]
        i += 1
        actions = agent_fn(observation, {"env_cfg": configurations})
        # send actions to engine
        print(json.dumps(actions))
