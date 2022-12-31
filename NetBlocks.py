import torch
import torch.nn as nn


###################################
# Adaptive Instance Normalization #
###################################
class AdaIN(nn.Module):
    def __init__(self,image_size,latent_dim,channels):
        super().__init__()
        self.A = nn.Linear(latent_dim,2*channels,bias=True)
        self.channels = channels
        self.image_size = image_size

    def forward(self,img,w,eps = 1e-4):
        mu = img.mean(dim =(1,2),keepdim = True)
        std = img.std(dim =(1,2),keepdim = True) + eps
        ys,yb = self.A(w).view(-1,2*self.channels,1,1).chunk(2,dim = 1)
        return ys*(img-mu)/std + yb#y[:,0,...]*(img-mu)/std + y[:,1,...]

################################
# Base Block in the Generator #
################################
class BaseBlock(nn.Module):
    """
    First block of the synthesis networks
    """
    def __init__(self,latent_dim = 512) -> None:
        super().__init__()
        self.base = nn.parameter.Parameter(torch.randn(latent_dim,4,4,requires_grad = True))
        self.ada_in1 = AdaIN(4,latent_dim,latent_dim)
        self.conv = nn.Conv2d(in_channels=latent_dim,out_channels=256,kernel_size=3,padding=1)
        self.ada_in2 = AdaIN(4,latent_dim,256)
    
    def forward(self,w):
        out = self.base #+ noise
        out = self.ada_in1(out,w)
        out = self.conv(out) #+noise
        return self.ada_in2(out,w)

#######################
# Generator Block     #
#######################
class SynthBlock(nn.Module):
    """Synthesys block, after upsampling"""
    def __init__(self,in_channels,out_channels,img_size,latent_dim = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.out_channels = out_channels

        #self.B_noise = nn.Linear(latent_dim,out_channels)
        #self.conv1 = nn.Conv2d(
        #    in_channels=in_channels,out_channels=out_channels,
        #    kernel_size=3,padding = 1)
        
        self.conv1 = convBlock(in_chs=in_channels,out_chs=out_channels,
           kernel_size=3,padding = 1
        )
        
        self.conv2 = convBlock(
            in_chs=out_channels,out_chs=out_channels,
            kernel_size=3,padding = 1)

        #self.conv2 = nn.Conv2d(
        #    in_channels=out_channels,out_channels=out_channels,
        #    kernel_size=3,padding = 1)
        
        self.ada_in1 = AdaIN(img_size,latent_dim,out_channels)
        self.ada_in2 = AdaIN(img_size,latent_dim,out_channels)
        
    #def AdaIN(self,x,y,eps = 1e-4,):
    #    mu = x.mean(dim = 0)
    #    std = x.std(dim = 0) + eps
    #    ys,yb = torch.tensor_split(y,indices_or_sections=2,dim = 1)
    #    return ys*(x - mu)/std + yb

    def forward(self,img,w):
        out = self.conv1(img)
        #out: out_channels X img_size X img_size
        noise = torch.randn(out.shape).to(img.device)
        #noise = self.B_noise(noise).view(-1,self.out_channels,1,1)
        out += noise
        out = self.ada_in1(out,w)
        out = self.conv2(out)
        noise = torch.randn(out.shape).to(img.device)
        out += noise
        out = self.ada_in2(out,w)
        return out

#####################################
#   Conv - BatchNorm - Activation   #
#####################################
class convBlock(nn.Sequential):
    def __init__(self,in_chs,out_chs,kernel_size = 4,
                 stride = 1,padding= 1,activation = nn.LeakyReLU(0.2,inplace = False)):
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


##########################################
# ConvTranspose - BatchNorm - Activation #
##########################################        
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
