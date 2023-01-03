import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Dataset,Subset
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

ROOT_DIR = "./data/"


class CelebaDataset(Dataset):

    def __init__(self,data_dir = ROOT_DIR,eps = 1e-4):
        self.data_dir = data_dir

        self.means = torch.tensor([0.5,0.5,0.5]).view(3,1,1)
        self.stds = torch.tensor([0.5,0.5,0.5]).view(3,1,1)
        self.data = ImageFolder(
            root=data_dir,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(self.means,self.stds)]
            )
        )
        self.data.class_to_idx = {}
        self.N = len(self.data)


    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or len(self)
            return Subset(self, range(start,stop))
        else:
            img, _ = self.data[index]
            return img

    def __len__(self):
        return self.N

    def un_normalize(self,img):
        device = img.device
        return img*self.stds.to(device) + self.means.to(device)



class Augmentator(nn.Sequential):

    def __init__(self,transforms):
        super(Augmentator, self).__init__()
        if transforms is not None:
            for trans in transforms:
                self.add_module(trans)
        else:
            pass

        
