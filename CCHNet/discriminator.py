import torch.nn.functional as F
import torch.nn.init as nninit
from torch import nn
import numpy as np
import random
import torch

class AP():

    def __init__(self, tensor):
        self._x = tensor

    def avg_pool(self):
        return (self._x[:, :, ::2, ::2] + self._x[:, :, 1::2, ::2] + self._x[:, :, ::2, 1::2]
                + self._x[:, :, 1::2, 1::2]) / 4

class DiscriminatorBlock(nn.Module):

    def __init__(self, in_chans, out_chans, downsample=False):

        super(DiscriminatorBlock, self).__init__()
        self.inchannels = in_chans
        self.outchannels = out_chans
        self.downsample = downsample
        self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        self.conv1 = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = inputs[0]
        if self.downsample:
            shortcut = self.shortcut_conv(x)
            shortcut = AP(shortcut).avg_pool()
        else:
            shortcut = x
        x = F.relu(x, inplace=False)
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        if self.downsample:
            x = AP(x).avg_pool()
        return x + shortcut

class OptDiscriminatorBlock(nn.Module):

    def __init__(self, in_chans, out_chans):

        super(OptDiscriminatorBlock, self).__init__()
        self.inchannels = in_chans
        self.outchannels = out_chans
        self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):

        x = inputs[0]
        shortcut = AP(x).avg_pool()
        shortcut = self.shortcut_conv(shortcut)
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        x = AP(x).avg_pool()
        return x + shortcut

class CTDiscriminator(nn.Module):

    # Initialization(start)
    def __init__(self, dp1, dp2, dp3):

        super(CTDiscriminator, self).__init__()
        feats = 128
        self.block1 = OptDiscriminatorBlock(3, feats)
        self.block2 = DiscriminatorBlock(feats, feats, downsample=True)
        self.dp1 = nn.Dropout2d(p=dp1)
        self.block3 = DiscriminatorBlock(feats, feats, downsample=False)
        self.dp2 = nn.Dropout2d(p=dp2)
        self.block4 = DiscriminatorBlock(feats, feats, downsample=False)
        self.dp3 = nn.Dropout2d(p=dp3)
        self.output_linear = nn.Linear(feats, 1)
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != nn.Conv2d else 1.0
                nninit.xavier_uniform(module.weight.data, gain=gain)
                module.bias.data.zero_()

    def forward(self, *inputs):

        x = inputs[0]
        x = x.view(-1, 3, 16, 16)
        x = self.block1(x)
        x = self.block2(x)
        x = self.dp1(x)
        x = self.block3(x)
        x = self.dp2(x)
        x = self.block4(x)
        x = self.dp3(x)
        x = F.relu(x)
        d2 = x.mean(-1, keepdim=False).mean(-1, keepdim=False)
        x = d2.view(-1, 128)
        d1 = self.output_linear(x)
        return d1, d2, None