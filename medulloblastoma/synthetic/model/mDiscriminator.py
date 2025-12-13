import torch.nn as nn

from ..globals import GLOBAL_NUM_OUTPUT_DIM

_inputDim = GLOBAL_NUM_OUTPUT_DIM


class mDiscriminatorV1(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(_inputDim, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class ResidualDisBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(dim, dim))
        self.fc2 = spectral_norm(nn.Linear(dim, dim))
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        out = out + residual
        out = self.activation(out)
        out = self.dropout(out)
        return out


class mDiscriminatorV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = spectral_norm(nn.Linear(_inputDim, 2048))
        self.fc2 = spectral_norm(nn.Linear(2048, 1024))

        self.res1 = ResidualDisBlock(1024)
        self.res2 = ResidualDisBlock(1024)

        self.fc3 = spectral_norm(nn.Linear(1024, 512))
        self.fc4 = spectral_norm(nn.Linear(512, 256))
        self.out = spectral_norm(nn.Linear(256, 1))

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)

        x = self.res1(x)
        x = self.res2(x)

        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.activation(self.fc4(x))
        x = self.out(x)
        return self.sigmoid(x)


import torch.nn as nn
from torch.nn.utils import spectral_norm


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


class mDiscriminatorV3(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            spectral_norm(nn.Linear(_inputDim, 2048)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            spectral_norm(nn.Linear(2048, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            ResidualDisBlock(1024),
            ResidualDisBlock(1024),

            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Linear(256, 1))
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.net(x))
