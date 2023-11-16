from stable_baselines3.common.vec_env import CustomDummyVecEnv
import numpy as np
from collections import OrderedDict

obs = OrderedDict({
    "entity_count": [4, 5, 4],
    "entity_obs": [
        np.ones((4, 10)),
        np.ones((5, 10)),
        np.ones((4, 10))
    ]
})
copied = CustomDummyVecEnv.copy_obs_dict(obs)
print(copied)