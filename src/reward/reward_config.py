from dataclasses import dataclass

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

@dataclass
class EarlyRewardParam:
    global_reward_weight = 1
    light_reward_weight: float = 0.4 * global_reward_weight
    heavy_reward_weight: float = 4 * global_reward_weight
    ice_reward_weight: float = 0.1 * global_reward_weight
    ore_reward_weight: float = 0.5 * global_reward_weight
    water_reward_weight: float = 0.2 * global_reward_weight
    metal_reward_weight: float = 0.7 * global_reward_weight
    power_reward_weight: float = 0.01 * global_reward_weight
    factory_penalty_weight: float = 2 * global_reward_weight
    survive_reward_weight: float = 0.1

@dataclass
class LateRewardParam:
    use_gamma_coe: bool = False
    zero_sum: bool = True
    global_reward_weight = 1
    win_reward_weight: float = 10 * global_reward_weight
    light_reward_weight: float = 0.1 * global_reward_weight
    heavy_reward_weight: float = 1 * global_reward_weight
    ice_reward_weight: float = 0.5 * global_reward_weight
    ore_reward_weight: float = 0.1 * global_reward_weight
    water_reward_weight: float = 0.7 * global_reward_weight
    metal_reward_weight: float = 0.2 * global_reward_weight
    power_reward_weight: float = 0.01 * global_reward_weight
    lichen_reward_weight: float = 0.02 * global_reward_weight
    factory_penalty_weight: float = 2 * global_reward_weight
    lose_penalty_coe: float = 0.
    survive_reward_weight: float = 0.2

stats_reward_params = dict(
    action_queue_updates_total=1,
    action_queue_updates_success=1,
    consumption={
        "power": {
            "LIGHT": 1,
            "HEAVY": 1,
            "FACTORY": 1,
        },
        "water": 1,
        "metal": 1,
        "ore": {
            "LIGHT": 1,
            "HEAVY": 1,
        },
        "ice": {
            "LIGHT": 1,
            "HEAVY": 1,
        },
    },
    destroyed={
        'FACTORY': 1,
        'LIGHT': 1,
        'HEAVY': 1,
        'rubble': {
            'LIGHT': 1,
            'HEAVY': 1,
        },
        'lichen': {
            'LIGHT': 1,
            'HEAVY': 1,
        },
    },
    generation={
        'power': {
            'LIGHT': 1,
            'HEAVY': 1,
            'FACTORY': 1,
        },
        'water': 1,
        'metal': 1,
        'ore': {
            'LIGHT': 1,
            'HEAVY': 1,
        },
        'ice': {
            'LIGHT': 1,
            'HEAVY': 1,
        },
        'lichen': 1,
        'built': {
            'LIGHT': 1,
            'HEAVY': 1,
        },
    },
    pickup={
        'power': 1,
        'water': 1,
        'metal': 1,
        'ice': 1,
        'ore': 1,
    },
    transfer={
        'power': 1,
        'water': 1,
        'metal': 1,
        'ice': 1,
        'ore': 1,
    },
)