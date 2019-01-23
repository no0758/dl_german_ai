

from torch import nn
import math
import torch
import torch.nn.functional as F



class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class ResNet_glr(nn.Module):

    def __init__(self, block, layers, num_classes=1000,channels=18):
        super(ResNet_glr, self).__init__()
        self.conv1 = mfm(channels, 48, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc = mfm(6 * 6 * 128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)

        x = self.pool1(x)
        res = x
        out1 = self.block1(x)
        out1 = res + out1
        out1 = self.group1(out1)
        out1 = self.pool2(out1)
        res = out1
        out2 = self.block2(out1)
        out2 = res + out2
        out2 = self.group2(out2)
        out2 = self.pool3(out2)
        res = out2
        out3 = self.block3(out2)
        out3 = res + out3
        out3 = self.group3(out3)
        res = out3
        out4 = self.block4(out3)
        out4 = res + out4
        out4 = self.group4(out4)
        out4 = self.pool4(out4)
        out4 = out4.view(x.size(0), -1)
        fc = self.fc(out4)
        fc = F.dropout(fc, training=self.training)
        out = self.fc2(fc)

        return out


def resnet_glr(**kwargs):
    model = ResNet_glr(resblock, [1, 2, 3, 4], **kwargs)
    return model

