import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import sys


class EncoderDecoderNet(BaseFeaturesExtractor):
    def __init__(self, observation_space, num_actions, num_global_features):
        super(EncoderDecoderNet, self).__init__(observation_space, num_actions)
        
        # Pretrained ResNet as the encoder
        self.encoder = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.encoder.conv1 = nn.Conv2d(observation_space, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Bottleneck layers
        self.global_fc_1 = nn.Linear(num_global_features, 512)
        self.global_fc_2 = nn.Linear(512, 2048)
        
        # Decoder layers
        self.decoder_fc = nn.Linear(10240, 19 * 64 * 64)
        self.decoder_conv = nn.Conv2d(19, num_actions, kernel_size=1)
        self.decoder_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x, global_features):

        print(x.shape, file=sys.stderr)
        # Encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = torch.flatten(x, 1)
        
        # Bottleneck
        global_features = self.global_fc_1(global_features)
        global_features = self.global_fc_2(global_features)
        
        x = torch.cat((x, global_features), dim=1)
        x = self.decoder_fc(x)
        x = x.view(x.size(0), 19, 64, 64)
        
        # Decoder
        x = self.decoder_conv(x)
        return x
    


    # num_channels = 24
    # num_actions = 19
    # num_global_features = 50

    # batch_size = 1
    # grid_size = 64

    # input_channels = num_channels
    # input_tensor = torch.randn(batch_size, input_channels, grid_size, grid_size)

    # global_features = torch.randn(batch_size, num_global_features)

    # net = EncoderDecoderNet(num_channels, num_actions, num_global_features)

    # output = net(input_tensor, global_features)

    # print("Output shape:", output.shape)





    