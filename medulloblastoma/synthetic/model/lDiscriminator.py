import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm

from ..globals import *

_inputDim = GLOBAL_LATENCY_INPUT_DIM

class ResidualDisBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Linear(dim, dim)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(dim, dim))
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.activation(self.block(x) + x))

class lDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial layers
        self.fc1 = spectral_norm(nn.Linear(_inputDim, 64))
        self.res1 = ResidualDisBlock(64)

        self.fc2 = spectral_norm(nn.Linear(64, 512))
        self.res2 = ResidualDisBlock(512)

        self.fc3 = spectral_norm(nn.Linear(512, 512))
        self.res3 = ResidualDisBlock(512)

        self.fc4 = spectral_norm(nn.Linear(512, 2048))
        self.res4 = ResidualDisBlock(2048)

        self.fc5 = spectral_norm(nn.Linear(2048, 1024))
        self.res5 = ResidualDisBlock(1024)

        self.fc6 = spectral_norm(nn.Linear(1024, 128))
        self.res6 = ResidualDisBlock(128)
        self.res7 = ResidualDisBlock(128)  # second residual block

        self.fc_out = spectral_norm(nn.Linear(128 + 64, 1))  # forward from fc1

        self.dropout = nn.Dropout(0.3)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Layer 1
        x1 = self.act(self.fc1(x))
        x1 = self.res1(x1)
        x1 = self.dropout(x1)

        # Layer 2
        x2 = self.act(self.fc2(x1))
        x2 = self.res2(x2)
        x2 = self.dropout(x2)

        # Layer 3
        x3 = self.act(self.fc3(x2))
        x3 = self.res3(x3)
        x3 = self.dropout(x3)

        # Layer 4
        x4 = self.act(self.fc4(x3))
        x4 = self.res4(x4)
        x4 = self.dropout(x4)

        # Layer 5
        x5 = self.act(self.fc5(x4))
        x5 = self.res5(x5)
        x5 = self.dropout(x5)

        # Layer 6
        x6 = self.act(self.fc6(x5))
        x6 = self.res6(x6)
        x6 = self.res7(x6)
        x6 = self.dropout(x6)

        # Forward from early layer (x1) to output
        out_input = torch.cat([x6, x1], dim=1)
        out = self.fc_out(out_input)

        return self.sigmoid(out)
