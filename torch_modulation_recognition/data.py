
import os
import gc
import pickle
import torch
import numpy as np
from typing import List, Optional

MODULATIONS = {
    "8PSK": 0,
    "AM-DSB": 1,
    "AM-SSB": 2,
    "BPSK": 3,
    "CPFSK": 4,
    "GFSK": 5,
    "PAM4": 6,
    "QAM16": 7,
    "QAM64": 8,
    "QPSK": 9,
    "WBFM": 10,
}

SNRS = [
    -20, -18, -16, -14, -12, -10, -8, -6, -4, -2,
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18
]


class RadioML2016(torch.utils.data.Dataset):
    modulations = MODULATIONS
    snrs = SNRS

    def __init__(
        self,
        data_dir="./data",
        file_name="RML2016.10a_dict.pkl"
    ):
        self.file_name = file_name
        self.data_dir = data_dir
        self.n_classes = len(self.modulations)
        self.X, self.y = self.load_data()
        gc.collect()

    def load_data(self):
        file_path = os.path.join(self.data_dir, self.file_name)

        print("Loading dataset from:", file_path)

        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        X = []
        y = []

        for mod, snr in data.keys():
            X.append(data[(mod, snr)])
            y.extend([(mod, snr)] * data[(mod, snr)].shape[0])

        X = np.vstack(X)

        print("Dataset loaded successfully.")
        print("Total samples:", len(X))
        print("Number of classes:", self.n_classes)

        return X, y

    def __getitem__(self, idx):
        x, (mod, snr) = self.X[idx], self.y[idx]

        label = self.modulations[mod]

        x = torch.from_numpy(x).to(torch.float32)
        x = x * 100
        x = x.unsqueeze(0)

        y = torch.tensor(label, dtype=torch.long)

        return x, y

    def __len__(self):
        return self.X.shape[0]

    def get_signals(self, mod=None, snr=None):
        indices = []

        for i, (sample_mod, sample_snr) in enumerate(self.y):
            if mod is not None and sample_mod != mod:
                continue

            if snr is not None and sample_snr != snr:
                continue

            indices.append(i)

        if not indices:
            return np.empty((0, 2, 128)), []

        return self.X[indices], [self.y[i] for i in indices]
