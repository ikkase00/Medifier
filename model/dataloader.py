import torch
from torch.utils.data import Dataset, DataLoader


class Spectro(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        if self.transform:
            return self.transform(path)
        else:
            return path