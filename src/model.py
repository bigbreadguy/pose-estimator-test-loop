import os
import numpy as np

import torch
import torch.nn as nn

from src.layer import *

# Deep Residual Learning for Image Recognition
# https://arxiv.org/abs/1512.03385
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, learning_type="plain", norm="bnorm", nblk=16):
        super(ResNet, self).__init__()

        self.learning_type = learning_type

        self.enc = CBR2d(in_channels, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=None, relu=0.0)

        res = []
        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)]
        self.res = nn.Sequential(*res)

        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)

        self.fc = CBR2d(nker, out_channels, kernel_size=1, stride=1, padding=0, bias=True, norm=None, relu=None)

    def forward(self, x):
        x0 = x

        x = self.enc(x)
        x = self.res(x)
        x = self.dec(x)

        if self.learning_type == "plain":
            x = self.fc(x)
        elif self.learning_type == "residual":
            x = x0 + self.fc(x)

        return x

# Simple Baselines for Human Pose Estimation and Tracking
# https://arxiv.org/abs/1804.06208v2
class PoseResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm", num_layers=50):
        super(PoseResNet, self).__init__()
        
        arch_settings = {
                18: (True, (2, 2, 2, 2), 3),
                34: (True, (3, 4, 6, 3), 3),
                50: (False, (3, 4, 6, 3), 3),
                101: (False, (3, 4, 23, 3), 3),
                152: (False, (3, 8, 36, 3), 3)
            }

        is_basic, spec, num_dec = arch_settings[num_layers]
            
        self.enc = CBR2d(in_channels, nker, kernel_size=3, stride=2, padding=1, bias=True, norm=None, relu=0.0)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        res = []

        for i, nblk in enumerate(spec):
            if i==0:
                stride=1
            else:
                stride=2
            for j in range(nblk):
                base_nker = nker*2**i
                if j > 0:
                    mult=4
                elif i == 0 :
                    mult=1
                else:
                    mult=2
                res += [ResBlock(base_nker=base_nker, mult=mult, kernel_size=3, stride=stride,
                                 padding=1, bias=True, norm=norm, relu=0.0, basic=is_basic)]        
        
        self.res = nn.Sequential(*res)

        dec = []
        dec_out_channels = 4*nker
        for i in range(num_dec):
            if i == 0:
                if is_basic:
                    dec_in_channels = base_nker
                else:
                    dec_in_channels = 4*base_nker
            else:
                dec_in_channels = 4*nker

            dec += [DECBR2d(dec_in_channels, dec_out_channels, kernel_size=3, stride=2, padding=1, bias=True, norm=norm, relu=0.0)]
        
        self.dec = nn.Sequential(*dec)

        self.fc = CBR2d(dec_out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True, norm=None, relu=None)

    def forward(self, x):
        x = self.enc(x)
        x = self.maxpool(x)
        x = self.res(x)
        x = self.dec(x)
        x = self.fc(x)

        return x

# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# https://arxiv.org/abs/1609.04802
class SRResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, learning_type="plain", norm="bnorm", nblk=16):
        super(SRResNet, self).__init__()

        self.learning_type = learning_type

        self.enc = CBR2d(in_channels, nker, kernel_size=9, stride=1, padding=4, bias=True, norm=None, relu=0.0)

        res = []
        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)]
        self.res = nn.Sequential(*res)
        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=None)

        # ps1 = []
        # ps1 += [nn.Conv2d(in_channels=nker, out_channels=nker, kernel_size=3, stride=1, padding=1)]
        # ps1 += [nn.ReLU()]
        # self.ps1 = nn.Sequential(*ps1)
        #
        # ps2 = []
        # ps2 += [nn.Conv2d(in_channels=nker, out_channels=nker, kernel_size=3, stride=1, padding=1)]
        # ps2 += [nn.ReLU()]
        # self.ps2 = nn.Sequential(*ps2)

        ps1 = []
        ps1 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1, padding=1)]
        ps1 += [PixelShuffle(ry=2, rx=2)]
        ps1 += [nn.ReLU()]
        self.ps1 = nn.Sequential(*ps1)

        ps2 = []
        ps2 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1, padding=1)]
        ps2 += [PixelShuffle(ry=2, rx=2)]
        ps2 += [nn.ReLU()]
        self.ps2 = nn.Sequential(*ps2)

        self.fc = CBR2d(nker, out_channels, kernel_size=9, stride=1, padding=4, bias=True, norm=None, relu=None)

    def forward(self, x):
        x = self.enc(x)
        x0 = x

        x = self.res(x)

        x = self.dec(x)
        x = x + x0

        x = self.ps1(x)
        x = self.ps2(x)

        x = self.fc(x)

        return x

