import torch.nn as nn

from ..globals import GLOBAL_NUM_INPUT_RANDOM_GENERATOR, GLOBAL_NUM_OUTPUT_DIM

_inputRandomValues = GLOBAL_NUM_INPUT_RANDOM_GENERATOR
_outputDim = GLOBAL_NUM_OUTPUT_DIM


class mGeneratorV1_2886(nn.Module):
    def __init__(self):
        super().__init__()
        if _inputRandomValues == 2886:
            print("ERROR ON MGENERATORV1 THE INPUT RANDOM MUST BE 2886")

        _tNLoops = 2
        self.linears = nn.ModuleList([nn.Linear(_inputRandomValues, _inputRandomValues) for _ in range(_tNLoops)])
        self.activations = nn.ModuleList([nn.LeakyReLU(0.2) for _ in range(_tNLoops)])
        self.output = nn.Linear(_inputRandomValues, _inputRandomValues)

    def forward(self, r):
        x = r
        for linear, act in zip(self.linears, self.activations):
            x = act(linear(x))
        x = self.output(x)
        return x


class mGeneratorV1_64(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_inputRandomValues, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, _outputDim)
        )

    def forward(self, r):
        return self.net(r)


class mGeneratorV2_64(nn.Module):
    def __init__(self):
        super().__init__()

        self.skip = nn.Linear(256, 512)

        self.fc1 = nn.Linear(_inputRandomValues, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc4 = nn.Linear(1024, _outputDim)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, r):
        x1 = self.activation(self.bn1(self.fc1(r)))

        x2 = self.activation(self.bn2(self.fc2(x1)))
        x2 = x2 + self.skip(x1)

        x3 = self.activation(self.bn3(self.fc3(x2)))

        out = self.fc4(x3)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual  # residual connection
        out = self.activation(out)
        return out


class mGeneratorV3_64_Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_inputRandomValues, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.skip = nn.Linear(256, 512)

        self.res_block = ResidualBlock(512)

        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc4 = nn.Linear(1024, _outputDim)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, r):
        x1 = self.activation(self.bn1(self.fc1(r)))

        x2 = self.activation(self.bn2(self.fc2(x1)))
        x2 = x2 + self.skip(x1)
        x2 = self.res_block(x2)
        x3 = self.activation(self.bn3(self.fc3(x2)))
        out = self.fc4(x3)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.block(x) + x)


class mGeneratorV4_Residual(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_inputRandomValues, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            ResidualBlock(512),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            ResidualBlock(1024),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(2048, 2048 * 2),
            nn.Linear(2048 * 2, _outputDim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)
