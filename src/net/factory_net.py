import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from net import *


class FactoryNet(BaseFeaturesExtractor):
    def __init__(self, observation_space, num_actions):
        super(FactoryNet, self).__init__(observation_space, num_actions)
        self.linear = nn.Linear(observation_space, num_actions)

    def forward(self, x):
        x = self.linear(x)
        return x


