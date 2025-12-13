import pandas as pd
import torch
import random
from typing import Optional

from ..globals import GLOBAL_NUM_INPUT_RANDOM_GENERATOR, GLOBAL_LATENCY_INPUT_DIM
from ..model.lGenerator import lGenerator
from ..model.mGenerator import mGeneratorV4_Residual
from ..synthetic import getSyntheticData

class dMixer:
    def __init__(self, file_csv: str, p: float):
        # Load data and metadata
        data = pd.read_csv(file_csv, index_col=0).astype(float)
        metadata = pd.read_csv('data/cavalli_subgroups.csv', index_col=0).squeeze()
        metadata = metadata.map({'Group3': 'G3', 'Group4': 'G4'})
        metadata_g3g4 = metadata[metadata.isin(['G3', 'G4'])]

        # Filter data
        data_g3g4 = data.loc[metadata_g3g4.index]
        values = torch.tensor(data_g3g4.values, dtype=torch.float32)

        # Shuffle
        idx = torch.randperm(values.size(0))
        values = values[idx]

        # Split and generate synthetic
        self._splitA = values
        self._splitB = getSyntheticData(
            mGeneratorV4_Residual(),
            "data/models/mGeneratorV4_Residual_1756-Statistical.pt",
            int(values.size(0) * p),
            GLOBAL_NUM_INPUT_RANDOM_GENERATOR
        )

        # Combine
        self._comb = torch.cat([self._splitA, self._splitB], dim=0)
        self._columns = data_g3g4.columns
        self._index = range(self._comb.size(0))

    def combination(self):
        # Just return combined tensor as DataFrame
        return pd.DataFrame(self._comb.numpy(), columns=self._columns, index=self._index)

    def save_combination(self, save_csv: str):
        df_comb = self.combination()
        df_comb.to_csv(save_csv)

class lMixer:
    def __init__(self, file_csv: str, p: float):
        # Load data and metadata
        data = pd.read_csv(file_csv, index_col=0).astype(float)
        values = torch.tensor(data.values, dtype=torch.float32)

        # Shuffle
        idx = torch.randperm(values.size(0))
        values = values[idx]

        # Split and generate synthetic
        self._splitA = values
        self._splitB = getSyntheticData(
            lGenerator(),
            "data/models/lGenerator_2035.pt",
            int(values.size(0) * p),
            GLOBAL_LATENCY_INPUT_DIM
        )

        # Combine
        self._comb = torch.cat([self._splitA, self._splitB], dim=0)
        self._columns = data.columns
        self._index = range(self._comb.size(0))

    def combination(self):
        # Just return combined tensor as DataFrame
        return pd.DataFrame(self._comb.numpy(), columns=self._columns, index=self._index)

    def save_combination(self, save_csv: str):
        df_comb = self.combination()
        df_comb.to_csv(save_csv)


class dParser:
    def __init__(self, file_csv: str, p: float, seed: Optional[int] = None):
        assert 0 < p < 1, "p must be in (0,1)"
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        data = pd.read_csv(file_csv, index_col=0).astype(float)

        metadata = pd.read_csv('data/cavalli_subgroups.csv', index_col=0).squeeze()
        metadata = metadata.map({'Group3': 'G3', 'Group4': 'G4'})
        metadata_g3g4 = metadata[metadata.isin(['G3', 'G4'])]

        data_g3g4 = data.loc[metadata_g3g4.index]
        values = torch.tensor(data_g3g4.values, dtype=torch.float32)

        idx = torch.randperm(values.size(0))
        values = values[idx]
        split = int(values.size(0) * p)
        self._all = values
        self._train = values[:split]
        self._test = values[split:]

    def all(self):
        return self._all

    def train(self) -> torch.Tensor:
        return self._train

    def test(self) -> torch.Tensor:
        return self._test
