import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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

class DashNet(BaseFeaturesExtractor):
    def __init__(self, observation, output_channels, seed=42):
        input_channels = observation['map'].shape[0] + observation['global'].shape[0] + observation['factory'].shape[0] + observation['unit'].shape[0]

        super(DashNet, self).__init__(input_channels, output_channels)

        self.init_conv = MyConv2d("init", input_channels, 64, kernel_size=3, stride=1, padding="same", batch_norm=USE_BATCH_NORM, spectral_norm=USE_SPECTRAL_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)

        self.res_block1 = ResidualBlock("res_block1", 64, 64, reduction=input_channels, seed=seed)
        self.res_block2 = ResidualBlock("res_block2", 64, 64, reduction=input_channels, seed=seed)
        self.res_block3 = ResidualBlock("res_block3", 64, 64, reduction=input_channels, seed=seed)
        self.res_block4 = ResidualBlock("res_block4", 64, 64, reduction=input_channels, seed=seed)

        self.final_conv = MyConv2d("final", 64, output_channels, kernel_size=3, stride=1, padding="same", batch_norm=USE_BATCH_NORM, spectral_norm=USE_SPECTRAL_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)

    def forward(self, x):
        cnn_features = x['map']
        global_features = x['global']
        factory_features = x['factory']
        unit_features = x['unit']
        x = torch.cat((cnn_features, global_features, factory_features, unit_features), dim=1)

        x = self.init_conv(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        x = self.final_conv(x)

        return x
