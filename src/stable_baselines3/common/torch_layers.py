from typing import Dict, List, Tuple, Type, Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device

import torch.nn as nn
import sys
import time
import hashlib

used_names = set()
def myHash(text: str) -> int:
    text_bytes = text.encode('utf-8')
    hash_str = hashlib.md5(text_bytes).hexdigest()
    hash_int = int(hash_str, 16) % (10 ** 8)
    return hash_int

def seed_init(seed: int, name: str, salt: str = ""):
    name = name + salt
    if name in used_names:
        raise ValueError(f"Name {name} already used")
    else:
        used_names.add(name)
    
    seed = myHash(f"seed{seed}_{name}")
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

def get_activation(activation):
    if activation == "relu":
        activation = nn.ReLU()
    elif activation == "leaky_relu":
        activation = nn.LeakyReLU()
    elif activation == "sigmoid":
        activation = nn.Sigmoid()
    elif activation == "tanh":
        activation = nn.Tanh()
    else:
        activation = None
    return activation

def init_orthogonal(module, weight_init, bias_init, gain=1, scaling=1.0):
    """Helper to initialize a layer weight and bias."""
    weight_init(module.weight.data, gain=gain)
    module.weight.data *= scaling
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

weight_scale = 0.01
init_leaky_relu_ = lambda m: init_orthogonal(m, nn.init.orthogonal_, nn.init.zeros_, nn.init.calculate_gain('leaky_relu'), weight_scale)
init_relu_ = lambda m: init_orthogonal(m, nn.init.orthogonal_, nn.init.zeros_, nn.init.calculate_gain('relu'), weight_scale)
init_sigmoid_ = lambda m: init_orthogonal(m, nn.init.orthogonal_, nn.init.zeros_, nn.init.calculate_gain('sigmoid'), weight_scale)

USE_BATCH_NORM = True
USE_LAYER_NORM = True
USE_SPECTRAL_NORM = True


def MyConv2d(name, in_channels, out_channels, kernel_size=1, stride=1, padding="same", bias=True, dilation=1, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, layer_norm=USE_LAYER_NORM, activation="leaky_relu", init_fn=None, seed=None):
    if seed is None:
        print(f"No seed provided for {name}")
        seed = int(time.time())

    name = name + "_conv2d"

    activation = get_activation(activation)

    seed_init(seed, name)
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)

    if spectral_norm:
        layer = nn.utils.spectral_norm(layer)

    if init_fn is not None:
        seed_init(seed, name, "_init")
        layer = init_fn(layer)
    
    if batch_norm:
        seed_init(seed, name, "_batch_norm")
        layer = nn.Sequential(layer, nn.BatchNorm2d(out_channels))
    
    if layer_norm:
        seed_init(seed, name, "_layer_norm")
        layer = nn.Sequential(layer, nn.GroupNorm(1, out_channels))

    if activation is not None:
        layer = nn.Sequential(layer, activation)

    return layer

def MyLinear(name, in_features, out_features, bias=True, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, layer_norm=USE_LAYER_NORM, activation="leaky_relu", init_fn=None, seed=None):
    if seed is None:
        print(f"No seed provided for {name}", file=sys.stderr)
        seed = int(time.time())

    name = name + "_linear"

    activation = get_activation(activation)

    seed_init(seed, name)
    layer = nn.Linear(in_features, out_features, bias=bias)

    if spectral_norm:
        layer = nn.utils.spectral_norm(layer)

    if init_fn is not None:
        seed_init(seed, name, "_init")
        layer = init_fn(layer)
    
    if batch_norm:
        seed_init(seed, name, "_batch_norm")
        layer = nn.Sequential(layer, nn.BatchNorm1d(out_features))

    if layer_norm:
        seed_init(seed, name, "_layer_norm")
        layer = nn.Sequential(layer, nn.GroupNorm(1, out_features))

    if activation is not None:
        layer = nn.Sequential(layer, activation)

    return layer

def Conv1x1(name, in_channels, out_channels, bias=True, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, layer_norm=USE_LAYER_NORM, activation="leaky_relu", init_fn=None, seed=None):
    return MyConv2d(name, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, spectral_norm=spectral_norm, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation, init_fn=init_fn, seed=seed)

def Conv3x3(name, in_channels, out_channels, bias=True, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, layer_norm=USE_LAYER_NORM, activation="leaky_relu", init_fn=None, seed=None):
    return MyConv2d(name, in_channels, out_channels, kernel_size=3, stride=1, padding="same", bias=bias, spectral_norm=spectral_norm, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation, init_fn=init_fn, seed=seed)

def Conv3x3_2(name, in_channels, out_channels, bias=True, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, layer_norm=USE_LAYER_NORM, activation="leaky_relu", init_fn=None, seed=None):
    return MyConv2d(name, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias, spectral_norm=spectral_norm, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation, init_fn=init_fn, seed=seed)

def Conv3x3_d(name, in_channels, out_channels, dilation = 2, padding = 2, bias=True, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, layer_norm=USE_LAYER_NORM, activation="leaky_relu", init_fn=None, seed=None):
    return MyConv2d(name, in_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=bias, dilation=dilation, spectral_norm=spectral_norm, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation, init_fn=init_fn, seed=seed)

def Conv5x5(name, in_channels, out_channels, bias=True, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, layer_norm=USE_LAYER_NORM, activation="leaky_relu", init_fn=None, seed=None):
    return MyConv2d(name, in_channels, out_channels, kernel_size=5, stride=1, padding="same", bias=bias, spectral_norm=spectral_norm, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation, init_fn=init_fn, seed=seed)

class SqueezeExcitation(nn.Module):
    def __init__(self, name, channel, reduction=16, seed=None):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            MyLinear(name + "_se_fc", channel, channel // reduction, bias=False, activation="relu", init_fn=init_relu_, seed=seed),
            MyLinear(name + "_se_fc2", channel // reduction, channel, bias=False, activation="sigmoid", init_fn=init_sigmoid_, seed=seed)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, name, in_channels, out_channels, use_se=True, seed=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv3x3(name + "_conv1", in_channels, out_channels, bias=(not USE_BATCH_NORM), spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)
        self.conv2 = Conv3x3(name + "_conv2", out_channels, out_channels, bias=(not USE_BATCH_NORM), spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(name + "_se", out_channels, reduction=16, seed=seed)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        if in_channels != out_channels:
            self.skip = Conv1x1(name + "_skip", in_channels, out_channels, bias=(not USE_BATCH_NORM), spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation=None, init_fn=init_leaky_relu_, seed=seed)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_se:
            out = self.se(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.leaky_relu(out)
        return out

class DashNet(nn.Module):
    def __init__(self, seed=42):

        super(DashNet, self).__init__()

        self.init_conv = MyConv2d("init", 16, 64, kernel_size=3, stride=1, padding="same", batch_norm=USE_BATCH_NORM, spectral_norm=USE_SPECTRAL_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)

        self.res_block1 = ResidualBlock("res_block1", 64, 64, use_se=True, seed=seed)
        #self.res_block2 = ResidualBlock("res_block2", 64, 64, use_se=True, seed=seed)
        #self.res_block3 = ResidualBlock("res_block3", 64, 64, use_se=True, seed=seed)
        #self.res_block4 = ResidualBlock("res_block4", 64, 64, use_se=True, seed=seed)

        self.final_conv = MyConv2d("final", 64, 16, kernel_size=3, stride=1, padding="same", batch_norm=USE_BATCH_NORM, spectral_norm=USE_SPECTRAL_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)

    def forward(self, x):
        x = self.init_conv(x)

        x = self.res_block1(x)
        #x = self.res_block2(x)
        #x = self.res_block3(x)
        #x = self.res_block4(x)

        x = self.final_conv(x)

        return x



class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class CustomExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        obs_shape: int = 0,
    ) -> None:
        super().__init__()
        device = get_device(device)
        #policy_net: List[nn.Module] = [nn.Identity()]
        policy_net: List[nn.Module] = [DashNet()]
        value_net: List[nn.Module] = [
            #nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(feature_dim),
            #activation_fn(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        ]

        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

        self.latent_dim_pi = feature_dim
        self.latent_dim_vf = feature_dim

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], List[int]]:
    """
    Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.

    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.

    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).

    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.

    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).

    :param net_arch: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return: The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch
