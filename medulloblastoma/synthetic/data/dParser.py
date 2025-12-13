import pandas as pd
import torch
import random
from typing import Optional

class dParser:
    def __init__(self, file_csv: str, p: float, seed: Optional[int] = None):
        assert 0 < p < 1, "p must be in (0,1)"
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Load data
        data = pd.read_csv(file_csv, index_col=0).astype(float)

        # Load metadata and filter only G3 and G4
        metadata = pd.read_csv('data/cavalli_subgroups.csv', index_col=0).squeeze()
        metadata = metadata.map({'Group3': 'G3', 'Group4': 'G4'})
        metadata_g3g4 = metadata[metadata.isin(['G3', 'G4'])]

        # Select only rows present in metadata_g3g4
        data_g3g4 = data.loc[metadata_g3g4.index]

        # Convert to torch tensor
        values = torch.tensor(data_g3g4.values, dtype=torch.float32)

        # Shuffle rows
        idx = torch.randperm(values.size(0))
        values = values[idx]

        # Split into train/test
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
