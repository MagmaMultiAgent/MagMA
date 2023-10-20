import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import sys


class EncoderDecoderNet(BaseFeaturesExtractor):
    def __init__(self, observation, output_channels):
        num_cnn_channels = observation['map'].shape[0]
        num_global_features = observation['global'].shape[0]
        super(EncoderDecoderNet, self).__init__(num_cnn_channels, output_channels)
        
        # Pretrained ResNet as the encoder
        self.encoder = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.encoder.conv1 = nn.Conv2d(num_cnn_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Bottleneck layers
        self.global_fc_1 = nn.Linear(num_global_features, 512)
        self.global_fc_2 = nn.Linear(512, 2048)
        
        # Decoder layers
        self.decoder_fc = nn.Linear(1024 * 2 * 2, 128 * 4 * 4)
        self.decoder_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.decoder_deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.decoder_deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.decoder_deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.decoder_deconv5 = nn.ConvTranspose2d(128, output_channels, kernel_size=3, stride=1, padding=1, output_padding=0)

    def forward(self, x):
        cnn_features = x['map']
        global_features = x['global']

        # Encoder
        cnn_features = self.encoder.conv1(cnn_features)
        cnn_features = self.encoder.bn1(cnn_features)
        cnn_features = self.encoder.relu(cnn_features)
        
        cnn_features = self.encoder.maxpool(cnn_features)
        cnn_features = self.encoder.layer1(cnn_features)
        cnn_features = self.encoder.layer2(cnn_features)
        cnn_features = self.encoder.layer3(cnn_features)
        cnn_features = self.encoder.layer4(cnn_features)

        cnn_features = torch.flatten(cnn_features, 1)
        
        # Bottleneck
        global_features = self.global_fc_1(global_features)
        global_features = self.global_fc_2(global_features)
        
        cnn_features = torch.cat((cnn_features, global_features), dim=1)

        # Decoder
        cnn_features = self.decoder_fc(cnn_features)
        cnn_features = cnn_features.view(cnn_features.size(0), 128, 4, 4)
        cnn_features = self.decoder_upsample(cnn_features)
        cnn_features = self.decoder_deconv1(cnn_features)
        cnn_features = self.decoder_upsample(cnn_features)
        cnn_features = self.decoder_deconv2(cnn_features)
        cnn_features = self.decoder_upsample(cnn_features)
        cnn_features = self.decoder_deconv3(cnn_features)
        cnn_features = self.decoder_upsample(cnn_features)
        cnn_features = self.decoder_deconv4(cnn_features)
        cnn_features = self.decoder_upsample(cnn_features)
        cnn_features = self.decoder_deconv5(cnn_features)
        return cnn_features