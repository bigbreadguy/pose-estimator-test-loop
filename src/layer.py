import os
import numpy as np

import torch
import torch.nn as nn

class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)

class DECBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=0, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []
        # layers += [nn.ReflectionPad2d(padding=padding)]
        layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                                      bias=bias)]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)

class ResBlock(nn.Module):
    def __init__(self, base_nker, mult, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0, basic=True):
        super().__init__()

        layers = []

        if basic:
            # 1st conv
            layers += [CBR2d(in_channels=base_nker, out_channels=base_nker,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            bias=bias, norm=norm, relu=relu)]

            # 2nd conv
            layers += [CBR2d(in_channels=base_nker, out_channels=base_nker,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            bias=bias, norm=norm, relu=None)]
        else:
            # 1st conv
            layers += [CBR2d(in_channels=base_nker*mult, out_channels=base_nker,
                            kernel_size=1, stride=stride, padding=0,
                            bias=bias, norm=norm, relu=relu)]

            # 2nd conv
            layers += [CBR2d(in_channels=base_nker, out_channels=base_nker,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            bias=bias, norm=norm, relu=None)]
            
            # 3rd conv
            layers += [CBR2d(in_channels=base_nker, out_channels=4*base_nker,
                            kernel_size=1, stride=stride, padding=0,
                            bias=bias, norm=norm, relu=None)]
        
        self.shortcut = nn.Conv2d(in_channels=base_nker*mult, out_channels=4*base_nker,
                                    kernel_size=1, stride=1, padding=0)
        self.resblk = nn.Sequential(*layers)

    def forward(self, x):
        return self.shortcut(x) + self.resblk(x)


class PixelUnshuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C, H // ry, ry, W // rx, rx)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * (ry * rx), H // ry, W // rx)

        return x


class PixelShuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C // (ry * rx), ry, rx, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // (ry * rx), H * ry, W * rx)

        return x