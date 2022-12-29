import torch
import torchvision.datasets.celeba as celeba
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

ROOT_DIR = "./img_align_celeba"


class CelebaDataset(Dataset):

    def __init__(self,data_dir = ROOT_DIR,eps = 1e-4):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.data = []

        for file in self.files:
            im = Image.open(os.path.join(data_dir,file))
            im = np.array(im)
            self.data.append(
                torch.tensor(im,dtype = torch.float32,requires_grad=False).permute(2,0,1)
            )
        self.data = torch.tensor(self.data,dtype=torch.float32)
        mu = self.data.mean(dim=(0,2,3))
        std = self.data.std(dim=(0,2,3)) + eps
        self.data = (self.data-mu)/std
        
        self.N = self.data.shape[0]


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.N