import torch
import torchvision.transforms as T
import torch.nn as nn



class Augmentatior():

    def __init__(self,prob = 0.65,transform_list = None) -> None:
        if transform_list is not None:
            self.transform_list = transform_list
        else:
            self.transform_list = [
                T.

            ]
        