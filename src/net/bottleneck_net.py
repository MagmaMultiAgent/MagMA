import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import time
import hashlib
import sys

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
init_value_ = lambda m: init_orthogonal(m, nn.init.orthogonal_, nn.init.zeros_, 0.01, weight_scale)
init_actor_ = lambda m: init_orthogonal(m, nn.init.orthogonal_, nn.init.zeros_, 0.01, weight_scale)

USE_BATCH_NORM = True
USE_LAYER_NORM = False
USE_SPECTRAL_NORM = True

def MyConv2d(name, in_channels, out_channels, kernel_size=1, stride=1, padding="same", bias=True, dilation=1, spectral_norm=False, batch_norm=False, layer_norm=False, activation="leaky_relu", init_fn=None, seed=None):
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

def MyLinear(name, in_features, out_features, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation="leaky_relu", init_fn=None, seed=None):
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

def Conv1x1(name, in_channels, out_channels, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation="leaky_relu", init_fn=None, seed=None):
    return MyConv2d(name, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, spectral_norm=spectral_norm, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation, init_fn=init_fn, seed=seed)

def Conv3x3(name, in_channels, out_channels, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation="leaky_relu", init_fn=None, seed=None):
    return MyConv2d(name, in_channels, out_channels, kernel_size=3, stride=1, padding="same", bias=bias, spectral_norm=spectral_norm, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation, init_fn=init_fn, seed=seed)

def Conv3x3_2(name, in_channels, out_channels, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation="leaky_relu", init_fn=None, seed=None):
    return MyConv2d(name, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias, spectral_norm=spectral_norm, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation, init_fn=init_fn, seed=seed)

def Conv3x3_d(name, in_channels, out_channels, dilation = 2, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation="leaky_relu", init_fn=None, seed=None):
    return MyConv2d(name, in_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=bias, dilation=dilation, spectral_norm=spectral_norm, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation, init_fn=init_fn, seed=seed)

def Conv5x5(name, in_channels, out_channels, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation="leaky_relu", init_fn=None, seed=None):
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

class DownsampleBlock(nn.Module):
    def __init__(self, name, in_channels, out_channels, use_se=True, seed=None):
        super(DownsampleBlock, self).__init__()
        self.block = nn.Sequential(
            ResidualBlock(name + "_res", in_channels, out_channels, use_se=True, seed=seed),
            Conv3x3_2(name + "_down", out_channels, out_channels, bias=(not USE_BATCH_NORM), spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)
        )

    def forward(self, x):
        return self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, name, in_channels, out_channels, use_se=True, seed=None):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            ResidualBlock(name + "_res", out_channels, out_channels, use_se=True, seed=seed)
        )

    def forward(self, x):
        return self.block(x)


class SimpleBottleneck(nn.Module):
    def __init__(self, name, in_channels, seed=None):
        super(SimpleBottleneck, self).__init__()
        self.block = nn.Sequential(
            ResidualBlock(name + "_res1", in_channels, in_channels, seed=seed),
            ResidualBlock(name + "_res2", in_channels, in_channels, seed=seed)
        )

    def forward(self, x):
        return self.block(x)

class DilatedBottleneck(nn.Module):
    def __init__(self, name, in_channels, seed=None):
        super(DilatedBottleneck, self).__init__()
        self.block = nn.Sequential(
            Conv3x3_d(name + "_conv1", in_channels, in_channels, dilation=2, padding=2, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed),
            Conv3x3_d(name + "_conv2", in_channels, in_channels, dilation=4, padding=4, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed),
            ResidualBlock(name + "_res", in_channels, in_channels, seed=seed)
        )

    def forward(self, x):
        return self.block(x)

class MultiScaleBottleneck(nn.Module):
    def __init__(self, name, in_channels, seed=None):
        super(MultiScaleBottleneck, self).__init__()
        self.conv1x1 = Conv1x1(name + "_1x1", in_channels, in_channels // 4, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)
        self.conv3x3 = Conv3x3(name + "_3x3", in_channels, in_channels // 4, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)
        self.conv5x5 = Conv5x5(name + "_5x5", in_channels, in_channels // 4, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)
        self.conv7x7 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(in_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3(x)
        out3 = self.conv5x5(x)
        out4 = self.conv7x7(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.bn(out)
        out = self.leaky_relu(out)
        return out

class InceptionBottleneck(nn.Module):
    def __init__(self, name, in_channels, seed=None):
        super(InceptionBottleneck, self).__init__()
        self.branch1x1 = nn.Sequential(
            Conv1x1(name + "_branch1x1", in_channels, in_channels // 4, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed),
        )
        self.branch3x3 = nn.Sequential(
            Conv1x1(name + "_branch3x3_1x1", in_channels, in_channels // 4, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed),
            Conv3x3(name + "_branch3x3", in_channels // 4, in_channels // 4, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)
        )
        self.branch5x5 = nn.Sequential(
            Conv1x1(name + "_branch5x5_1x1", in_channels, in_channels // 4, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed),
            Conv5x5(name + "_branch5x5", in_channels // 4, in_channels // 4, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)
        )
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv1x1(name + "_branch_pool", in_channels, in_channels // 4, spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x).expand_as(branch1x1)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)
        out = self.bn(out)
        out = self.leaky_relu(out)
        return out

class BottleNet(BaseFeaturesExtractor):
    def __init__(self, observation, output_channels, num_bottleneck_channels, bottleneck_type, seed=42):
        input_channels = observation['map'].shape[0] + observation['global'].shape[0] + observation['factory'].shape[0] + observation['unit'].shape[0]

        super(BottleNet, self).__init__(input_channels, output_channels)

        self.down1 = DownsampleBlock("down1", input_channels, 32, use_se=True, seed=seed)
        self.down2 = DownsampleBlock("down2", 32, 64, use_se=True, seed=seed)
        self.down3 = DownsampleBlock("down3", 64, 128, use_se=True, seed=seed)
        self.down4 = DownsampleBlock("down4", 128, num_bottleneck_channels, use_se=True, seed=seed)

        if bottleneck_type == "dilated":
            self.bottleneck = DilatedBottleneck("dilated_bottleneck", num_bottleneck_channels, seed=seed)
        elif bottleneck_type == "multiscale":
            self.bottleneck = MultiScaleBottleneck("multiscale_bottleneck", num_bottleneck_channels, seed=seed)
        elif bottleneck_type == "inception":
            self.bottleneck = InceptionBottleneck("inception_bottleneck", num_bottleneck_channels, seed=seed)
        else:
            self.bottleneck = SimpleBottleneck("simple_bottleneck", num_bottleneck_channels, seed=seed)

        self.up4 = UpsampleBlock("up4", num_bottleneck_channels, 128, use_se=True, seed=seed)
        self.up3 = UpsampleBlock("up3", 128, 64, use_se=True, seed=seed)
        self.up2 = UpsampleBlock("up2", 64, 32, use_se=True, seed=seed)
        self.up1 = UpsampleBlock("up1", 32, output_channels, use_se=True, seed=seed)

    def forward(self, x):

        cnn_features = x['map']
        global_features = x['global']
        factory_features = x['factory']
        unit_features = x['unit']
        x = torch.cat((cnn_features, global_features, factory_features, unit_features), dim=1)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bottleneck = self.bottleneck(d4)

        u4 = self.up4(bottleneck)
        u3 = self.up3(u4 + d3)
        u2 = self.up2(u3 + d2)
        u1 = self.up1(u2 + d1)

        return u1