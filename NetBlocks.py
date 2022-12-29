import torch
import torch.nn as nn



class SynthBlock(nn.Sequential):
    
    def __init__(self,in_channels,out_channels,img_size,latent_dim = 128):
        self.upsample = nn.Upsample(img_size)
        self.affine = nn.Linear(latent_dim,2*img_size*img_size,bias = True)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,out_channels=out_channels,
            kernel_size=3,padding = 1)

    def AdaIN(self,x,y,eps = 1e-4,):
        mu = x.mean(dim = 0)
        std = x.std(dim = 0) + eps
        ys,yb = torch.tensor_split(y,indices_or_sections=2,dim = 1)
        return ys*(x - mu)/std + yb

    def forward(self,img,w):
        self.AdaIN(img,self.affine(w))


class convBlock(nn.Sequential):
    def __init__(self,in_chs,out_chs,kernel_size = 4,
                 stride = 1,padding= 1,activation = nn.LeakyReLU(0.2,inplace = True)):
        super().__init__()
        self.add_module(
           "conv",
           nn.Conv2d(
            in_channels = in_chs,
           out_channels = out_chs,
           kernel_size=kernel_size,
           padding=padding,stride = stride,bias = False)
        )
        if out_chs >1:
            self.add_module("bnorm",nn.BatchNorm2d(out_chs))
        if activation is not None:
            self.add_module("activation",activation)

        
class deConvBlock(nn.Sequential):
    def __init__(self,in_chs,out_chs,kernel_size = 4,
                 stride = 1,padding= 1,activation = nn.LeakyReLU(0.2,inplace = True)):
        super().__init__()
        self.add_module("deConv",
        nn.ConvTranspose2d(
                in_channels=in_chs,
                out_channels=out_chs,
                kernel_size=kernel_size,
                padding=padding,bias = False,stride=stride,
        ))
        if out_chs >1:
            self.add_module("bnorm",nn.BatchNorm2d(out_chs))
        if activation is not None:
            self.add_module("activation", activation)
