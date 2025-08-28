import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, last_layer=False, groups=32):
        super().__init__()
        self.last_layer = last_layer

        self.conv1gn = nn.GroupNorm(groups, in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv2gn = nn.GroupNorm(groups, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes * self.expansion:
            self.cut = nn.Sequential(
                nn.GroupNorm(groups, in_planes),
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.cut = nn.Identity()

    def forward(self, x):
        out = self.conv1(F.silu(self.conv1gn(x)))
        out = self.conv2(F.silu(self.conv2gn(out)))
        out = out + self.cut(x)
        return out


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, last_layer=False, groups=32):
        super().__init__()
        self.last_layer = last_layer

        # Pre-activation bottleneck: GN -> SiLU -> Conv
        self.conv1gn = nn.GroupNorm(groups, in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.conv2gn = nn.GroupNorm(groups, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv3gn = nn.GroupNorm(groups, planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)

        if stride != 1 or in_planes != planes * self.expansion:
            self.cut = nn.Sequential(
                nn.GroupNorm(groups, in_planes),
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.cut = nn.Identity()

    def forward(self, x):
        out = self.conv1(F.silu(self.conv1gn(x)))
        out = self.conv2(F.silu(self.conv2gn(out)))
        out = self.conv3(F.silu(self.conv3gn(out)))
        out = out + self.cut(x)
        return out


class VAEEncoder(nn.Module):
    def __init__(self, in_channels, C, r, num_blocks=(2, 2, 2, 2)):
        super().__init__()
        self.C = C
        self.r = r
        self.num_blocks = num_blocks

        self.stem = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)

        # track channels explicitly per stage
        self.in_planes = 64
        self.layer1 = self._make_layer(ResBlock, self.num_blocks[0], planes=64, stride=1)

        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.in_planes = 128
        self.layer2 = self._make_layer(ResBlock, self.num_blocks[1], planes=128, stride=1)

        self.down2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.in_planes = 256
        self.bottleneck = self._make_layer(ResBottleneck, self.num_blocks[2], planes=256, stride=1)

        self.mu = nn.Conv2d(256 * ResBottleneck.expansion, C, kernel_size=1, bias=True)
        self.logvar = nn.Conv2d(256 * ResBottleneck.expansion, C, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.down1(out)
        out = self.layer2(out)
        out = self.down2(out)
        out = self.bottleneck(out)
        mu = self.mu(out)
        logvar = self.logvar(out)

        return mu, logvar

    def _make_layer(self, block, num_blocks, planes, stride=1):
        layers = []
        layers.append(block(self.in_planes, planes, stride=stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
