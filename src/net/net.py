import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torchvision

import logging
logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        logger.info(f"Creating {self.__class__.__name__}")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        self.logger.debug(f"foward with shape {x.shape}")
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which consists of linear layers.
    """

    def __init__(self, in_channels, out_channels, hidden_size):
        logger.info(f"Creating {self.__class__.__name__}")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")

        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_channels)

    def forward(self, x):
        self.logger.debug(f"foward with shape {x.shape}")
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        logger.info(f"Creating {self.__class__.__name__}")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")

        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        Forward pass
        """
        self.logger.debug(f"foward with shape {up_x.shape} {down_x.shape}")
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(BaseFeaturesExtractor):
    DEPTH = 6
    def __init__(self, observation, output_channels):
        logger.info(f"Creating {self.__class__.__name__}")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")

        num_cnn_channels = observation['map'].shape[0]
        num_global_features = observation['global'].shape[0]
        super(UNetWithResnet50Encoder, self).__init__(num_cnn_channels, output_channels)
        
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []

        num_output_channels = resnet.conv1.out_channels

        new_conv1 = nn.Conv2d(num_cnn_channels, num_output_channels, kernel_size=resnet.conv1.kernel_size,
                      stride=resnet.conv1.stride, padding=resnet.conv1.padding, bias=resnet.conv1.bias)

        resnet.conv1 = new_conv1

        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]

        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.global_fc_1 = nn.Linear(num_global_features, 512)
        self.global_fc_2 = nn.Linear(512, 1024)

        self.bridge = Bridge(9216, 8192, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 30, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, output_channels, kernel_size=1, stride=1)
        
    def forward(self, x):        
        cnn_features = x['map']
        global_features = x['global']

        self.logger.debug(f"foward with shape {cnn_features.shape} {global_features.shape}")

        x = cnn_features

        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        self.logger.debug(x.shape)
        x = x.view(8, -1)
        self.logger.debug(x.shape)

        global_features = self.global_fc_1(global_features)
        global_features = self.global_fc_2(global_features)
        self.logger.debug(f"{x.shape} {global_features.shape}")
        x = torch.cat((x, global_features), dim=1)
        x = self.bridge(x)

        x = x.view(8, 2048, 2, 2)


        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])

        output_feature_map = x
        x = self.out(x)
        del pre_pools
        return x
    


