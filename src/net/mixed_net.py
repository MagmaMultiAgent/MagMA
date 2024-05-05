import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.leaky_relu(out)
        return out


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super(DownsampleBlock, self).__init__()
        self.block = nn.Sequential(
            ResidualBlock(in_channels, out_channels, use_se=use_se),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            ResidualBlock(out_channels, out_channels, use_se=use_se)
        )

    def forward(self, x):
        return self.block(x)
    
class SimpleBottleneck(nn.Module):
    def __init__(self, in_channels):
        super(SimpleBottleneck, self).__init__()
        self.block = nn.Sequential(
            ResidualBlock(in_channels, in_channels),
            ResidualBlock(in_channels, in_channels)
        )

    def forward(self, x):
        return self.block(x)

class DilatedBottleneck(nn.Module):
    def __init__(self, in_channels):
        super(DilatedBottleneck, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            ResidualBlock(in_channels, in_channels)
        )

    def forward(self, x):
        return self.block(x)

class MultiScaleBottleneck(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleBottleneck, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=5, padding=2)
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
    def __init__(self, in_channels):
        super(InceptionBottleneck, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.LeakyReLU(inplace=True)
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.LeakyReLU(inplace=True)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.LeakyReLU(inplace=True)
        )
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.LeakyReLU(inplace=True)
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


class MonoNet(BaseFeaturesExtractor):
    def __init__(self, observation, output_channels, num_bottleneck_channels, bottleneck_type, use_se):

        input_channels = observation['map'].shape[0] + observation['global'].shape[0] + observation['factory'].shape[0] + observation['unit'].shape[0]

        super(MonoNet, self).__init__(input_channels, output_channels)

        self.down1 = DownsampleBlock(input_channels, 32, use_se=use_se)
        self.down2 = DownsampleBlock(32, 64, use_se=use_se)
        self.down3 = DownsampleBlock(64, 128, use_se=use_se)
        self.down4 = DownsampleBlock(128, num_bottleneck_channels, use_se=use_se)

        if bottleneck_type == "dilated":
            self.bottleneck = DilatedBottleneck(num_bottleneck_channels)
        elif bottleneck_type == "multiscale":
            self.bottleneck = MultiScaleBottleneck(num_bottleneck_channels)
        elif bottleneck_type == "inception":
            self.bottleneck = InceptionBottleneck(num_bottleneck_channels)
        else:
            self.bottleneck = SimpleBottleneck(num_bottleneck_channels)

        self.up4 = UpsampleBlock(num_bottleneck_channels, 128, use_se=use_se)
        self.up3 = UpsampleBlock(128, 64, use_se=use_se)
        self.up2 = UpsampleBlock(64, 32, use_se=use_se)
        self.up1 = UpsampleBlock(32, output_channels, use_se=use_se)

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
    


