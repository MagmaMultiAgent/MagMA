import copy
import gymnasium as gym
from luxai_s2.state import StatsStateDict

class RewardWrapper(gym.Wrapper):
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
            factory.cargo.water = 10000

        action = {agent: action}
        obs, _, termination, truncation, info = self.env.step(action)

        termination_status = termination[agent] if isinstance(termination, dict) else termination
        truncation_status = truncation[agent] if isinstance(truncation, dict) else truncation

        obs = obs[agent]

        stats: StatsStateDict = self.env.state.stats[agent]

        info = {}
        metrics = self._get_info(stats)
        reward = self.parse(metrics)
        self.prev_step_metrics = copy.deepcopy(metrics)
        info["metrics"] = metrics
        return obs, reward, termination_status, truncation_status, info
    
    def _get_info(self, stats: StatsStateDict):
        """
        Parses the stats
        """
        metrics = {}
        metrics["ice_transferred"] = stats["transfer"]["ice"]
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["water_produced"] = stats["generation"]["water"]
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]
        return metrics
    
    def parse(self, metrics):
        """
        Parses the metrics
        """
        reward = 0
        if self.prev_step_metrics is not None:

            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = (
                metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            )

            reward = ice_dug_this_step / 100 + water_produced_this_step

        return reward

    def reset(self, **kwargs):
        """
        Resets the environment
        """
        obs, reset_info = self.env.reset(**kwargs)
        self.prev_step_metrics = None
        return obs["player_0"], reset_info