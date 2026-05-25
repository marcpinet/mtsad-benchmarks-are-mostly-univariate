import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def train_val_split(data, ratio, seed=None):
    n = int(len(data) * ratio)
    if isinstance(data, pd.DataFrame):
        return data.iloc[:n], data.iloc[n:]
    return data[:n], data[n:]


class _WindowDataset(Dataset):
    def __init__(self, data, win_size, step):
        if isinstance(data, pd.DataFrame):
            data = data.values
        self.data = np.asarray(data, dtype=np.float32)
        self.win_size = win_size
        self.step = step
        self.n = max(0, (len(self.data) - win_size) // step + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.step
        x = self.data[s:s + self.win_size]
        y = np.zeros(self.win_size, dtype=np.float32)
        return x, y


def anomaly_detection_data_provider(data, batch_size, win_size, step, mode):
    shuffle = (mode == "train")
    actual_step = win_size if mode == "thre" else step
    ds = _WindowDataset(data, win_size, actual_step)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
