import torch_geometric
import torch
from torch.utils.data import Dataset
import numpy as np

class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)



class custom_const_target(Dataset):#torch_geometric.data.Dataset):
    """
    Arguments:
        dataset (Dataset): The whole Dataset
                
        const_label (float) : targets 
    """
    def __init__(self, dataset, const_label):
        self.dataset = dataset
        self.targets = torch.ones(len(dataset), dtype=torch.long)*const_label
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):

            x = self.dataset[idx].x
            edge_index = self.dataset[idx].edge_index
            y = self.targets[idx]
            return torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)

        else: raise NotImplementedError

    def __len__(self):
        return len(self.targets)



        

