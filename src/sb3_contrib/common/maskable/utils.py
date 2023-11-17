import numpy as np
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv

EXPECTED_METHOD_NAME = "action_masks"


def get_action_masks(env: GymEnv) -> np.ndarray:
    """
    Checks whether gym env exposes a method returning invalid action masks

    :param env: the Gym environment to get masks from
    :return: A numpy array of the masks
    """

    if isinstance(env, VecEnv):
        action_masks = env.env_method(EXPECTED_METHOD_NAME)

        max_size = max([a.shape[0] for a in action_masks])
        shape = list(action_masks[0].shape)
        shape[0] = max_size
        shape = [len(action_masks)] + shape
        action_masks2 = np.zeros(shape)
        for i, a in enumerate(action_masks):
            size = a.shape[0]
            assert size <= max_size
            action_masks2[i, 0:size, :] = a
        action_masks = action_masks2

        return np.stack(action_masks)
    else:
        return getattr(env, EXPECTED_METHOD_NAME)()


def is_masking_supported(env: GymEnv) -> bool:
    """
    Checks whether gym env exposes a method returning invalid action masks

    :param env: the Gym environment to check
    :return: True if the method is found, False otherwise
    """

    if isinstance(env, VecEnv):
        try:
            # TODO: add VecEnv.has_attr()
            env.get_attr(EXPECTED_METHOD_NAME)
            return True
        except AttributeError:
            return False
    else:
        return hasattr(env, EXPECTED_METHOD_NAME)
