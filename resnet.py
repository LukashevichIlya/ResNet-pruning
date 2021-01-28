import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class ShortcutHelper(nn.Module):
    def __init__(self, expr):
        super(ShortcutHelper, self).__init__()
        self.expr = expr

    def forward(self, src):
        return self.expr(src)


class ConvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=channels)

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

        self.res_connection = nn.Sequential()
        if stride != 1 or in_channels != channels:
            # All the shortcuts are identity shortcuts in the original paper for CIFAR-10 dataset
            self.res_connection = ShortcutHelper(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, channels // 4, channels // 4), 'constant', 0))

    def forward(self, src):
        out = F.relu(self.bn1(self.conv1(src)))
        out = self.bn2(self.conv2(out))
        out += self.res_connection(src)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):

    def __init__(self, block, num_classes=10):
        super(ResNet20, self).__init__()

        self.in_channels = 16

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.in_channels)

        self.block_layer1 = self._make_layer(block, channels=16, num_blocks=3, stride=1)
        self.block_layer2 = self._make_layer(block, channels=32, num_blocks=3, stride=2)
        self.block_layer3 = self._make_layer(block, channels=64, num_blocks=3, stride=2)

        self.linear = nn.Linear(in_features=64, out_features=num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.apply(init_weights)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, src):
        out = F.relu(self.bn1(self.conv1(src)))
        out = self.block_layer1(out)
        out = self.block_layer2(out)
        out = self.block_layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def create_resnet20():
    return ResNet20(ConvBlock)
