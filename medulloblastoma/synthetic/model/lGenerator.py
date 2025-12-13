import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm
import torch.nn.functional as F

from ..globals import *

_inputDim = GLOBAL_LATENCY_NUM_INPUT_RANDOM_GENERATOR
_outputDim = GLOBAL_LATENCY_INPUT_DIM
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.block(x) + x * 0.8)


class lGenerator(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        # Grow
        self.fc1 = nn.Linear(_inputDim, 64)
        self.res1 = ResidualBlock(64, dropout)

        self.fc2 = nn.Linear(64, 128)
        self.res2 = ResidualBlock(128, dropout)

        self.fc3 = nn.Linear(128, 256)
        self.res3 = ResidualBlock(256, dropout)

        self.fc4 = nn.Linear(256, 512)
        self.res4 = ResidualBlock(512, dropout)

        # Contract
        self.fc5 = nn.Linear(512 + 256, 256)
        self.res5 = ResidualBlock(256, dropout)

        self.fc6 = nn.Linear(256 + 128, 128)
        self.res6 = ResidualBlock(128, dropout)

        self.fc7 = nn.Linear(128 + 64, 64)
        self.res7 = ResidualBlock(64, dropout)

        # Output
        self.fc_out = nn.Linear(64 + 64, _outputDim)

    def forward(self, z):
        x1 = F.leaky_relu(self.fc1(z), 0.2)
        x1 = self.res1(x1)

        x2 = F.leaky_relu(self.fc2(x1), 0.2)
        x2 = self.res2(x2)

        x3 = F.leaky_relu(self.fc3(x2), 0.2)
        x3 = self.res3(x3)

        x4 = F.leaky_relu(self.fc4(x3), 0.2)
        x4 = self.res4(x4)

        # Contract
        x = torch.cat([x4, x3], dim=1)  # 512 + 256 = 768
        x = F.leaky_relu(self.fc5(x), 0.2)
        x = self.res5(x)

        x = torch.cat([x, x2], dim=1)  # 256 + 128 = 384
        x = F.leaky_relu(self.fc6(x), 0.2)
        x = self.res6(x)

        x = torch.cat([x, x1], dim=1)  # 128 + 64 = 192
        x = F.leaky_relu(self.fc7(x), 0.2)
        x = self.res7(x)

        x = torch.cat([x, x1], dim=1)  # 64 + 64 = 128 for output
        return self.fc_out(x)
