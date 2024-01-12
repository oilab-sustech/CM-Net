import torch
from  torch.utils.data import Dataset
from os.path import join

class PointsFolder(Dataset):
    
    def __init__(self, data, label, transform=None) -> None:
        super(PointsFolder, self).__init__()

        self.data = data
        self.label = label

        self.transform = transform


    def __getitem__(self, index):

        if self.data is not None:
            sample = self.data[index]
            target = self.label[index]
        else:
            raise ValueError(
            "The data cannot be None."
            )

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target
        
    def __len__(self):
        return self.data.shape[0]