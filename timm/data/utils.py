"""
Added by FS.
"""
import torch
from torch.utils.data import Dataset

class DataClass:
    def __init__(self, dataset: Dataset, split: str):
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, targets = self.dataset[index]

        return {'data': data, 
                'targets': targets, 
                'ind': index}

