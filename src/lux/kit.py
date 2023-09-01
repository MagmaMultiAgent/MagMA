"""
This file contains the GameState class, which is the main object \
that contains all the information about the game state.
"""

# pylint: disable=E0401
from dataclasses import dataclass, field
from typing import Dict
import numpy as np
from lux.cargo import UnitCargo
from lux.config import EnvConfig
from lux.factory import Factory
from lux.team import FactionTypes, Team
from lux.unit import Unit

def process_action(action):
    """
    Converts action to json
    """

    return to_json(action)


def to_json(obj):
    """
    Converts object to json
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [to_json(s) for s in obj]
    if isinstance(obj, dict):
        out = {}
        for key in obj:
            out[key] = to_json(obj[key])
        return out
    return obj

def from_json(state):
    """
    Converts json to object
    """

    if isinstance(state, list):
        return np.array(state)
    if isinstance(state, dict):
        out = {}
        for key in state:
            out[key] = from_json(state[key])
        return out
    return state

def process_obs(player, game_state, step, obs):
    """
    Converts obs to game state
    """

    if step == 0:
        game_state = from_json(obs)
    else:
        obs = from_json(obs)
        for key in obs:
            if key != "board":
                game_state[key] = obs[key]
            else:
                if "valid_spawns_mask" in obs[key]:
                    game_state["board"]["valid_spawns_mask"] = obs[key][
                        "valid_spawns_mask"
                    ]
        for item in ["rubble", "lichen", "lichen_strains"]:
            for key, value in obs["board"][item].items():
                key = key.split(",")
                x_coor, y_coord = int(key[0]), int(key[1])
                game_state["board"][item][x_coor, y_coord] = value
    return game_state

def obs_to_game_state(step, env_cfg: EnvConfig, obs):
    """
    Converts obs to game state
    """

    units = {}
    for agent in obs["units"]:
        units[agent] = {}
        for unit_id in obs["units"][agent]:
            unit_data = obs["units"][agent][unit_id]
            cargo = UnitCargo(**unit_data["cargo"])
            unit = Unit(
                **unit_data,
                unit_cfg=env_cfg.ROBOTS[unit_data["unit_type"]],
                env_cfg=env_cfg
            )
            unit.cargo = cargo
            units[agent][unit_id] = unit

    factory_occupancy_map = np.ones_like(obs["board"]["rubble"], dtype=int) * -1
    factories = {}
    for agent in obs["factories"]:
        factories[agent] = {}
        for unit_id in obs["factories"][agent]:
            f_data = obs["factories"][agent][unit_id]
            cargo = UnitCargo(**f_data["cargo"])
            factory = Factory(**f_data, env_cfg=env_cfg)
            factory.cargo = cargo
            factories[agent][unit_id] = factory
            factory_occupancy_map[factory.pos_slice] = factory.strain_id
    teams = {}
    for agent in obs["teams"]:
        team_data = obs["teams"][agent]
        faction = FactionTypes[team_data["faction"]]
        teams[agent] = Team(**team_data, agent=agent)

    return GameState(
        env_cfg=env_cfg,
        env_steps=step,
        board=Board(
            rubble=obs["board"]["rubble"],
            ice=obs["board"]["ice"],
            ore=obs["board"]["ore"],
            lichen=obs["board"]["lichen"],
            lichen_strains=obs["board"]["lichen_strains"],
            factory_occupancy_map=factory_occupancy_map,
            factories_per_team=obs["board"]["factories_per_team"],
            valid_spawns_mask=obs["board"]["valid_spawns_mask"],
        ),
        units=units,
        factories=factories,
        teams=teams,
    )

@dataclass
class Board:
    """
    Board class
    """

    rubble: np.ndarray
    ice: np.ndarray
    ore: np.ndarray
    lichen: np.ndarray
    lichen_strains: np.ndarray
    factory_occupancy_map: np.ndarray
    factories_per_team: int
    valid_spawns_mask: np.ndarray

@dataclass
class GameState:
    """
    A GameState object at step env_steps. Copied from luxai_s2/state/state.py
    """

    env_steps: int
    env_cfg: EnvConfig
    board: Board
    units: Dict[str, Dict[str, Unit]] = field(default_factory=dict)
    factories: Dict[str, Dict[str, Factory]] = field(default_factory=dict)
    teams: Dict[str, Team] = field(default_factory=dict)

    @property
    def real_env_steps(self):
        """
        the actual env step in the environment, which subtracts the time \
        spent bidding and placing factories
        """
        if self.env_cfg.BIDDING_SYSTEM:
            return self.env_steps - (self.board.factories_per_team * 2 + 1)
        return self.env_steps

    def is_day(self):
        """
        Returns true if it is day
        """

        return self.real_env_steps % self.env_cfg.CYCLE_LENGTH < self.env_cfg.DAY_LENGTH
