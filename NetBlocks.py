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
        #mu = img.mean(dim =(1,2),keepdim = True)
        #std = img.std(dim =(1,2),keepdim = True) + eps
        mu = img.mean(dim =(2,3),keepdim = True)
        std = img.std(dim =(2,3),keepdim = True) + eps
        
        ys,yb = self.A(w).view(-1,2*self.channels,1,1).chunk(2,dim = 1)
        #return ys*(img-mu)/std + yb
        return (ys+1)*(img-mu)/std + yb

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    

################################
# Base Block in the Generator  #
################################
class BaseBlock(nn.Module):
    """
    First block of the synthesis networks
    """
    def __init__(self,latent_dim = 512) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.randn(1,latent_dim,4,4,requires_grad = True))
        self.ada_in1 = AdaIN(4,latent_dim,latent_dim)
        self.conv = nn.Conv2d(in_channels=latent_dim,out_channels=256,kernel_size=3,padding=1)
        self.ada_in2 = AdaIN(4,latent_dim,256)
        self.B1_noise = nn.Parameter(torch.randn(latent_dim)).view(1,-1,1,1)
        self.B2_noise = nn.Parameter(torch.randn(256)).view(1,-1,1,1)    
    
    def forward(self,w):
        out = self.base 
        noise = torch.randn(out.size(0),1,out.size(2),out.size(3)).to(w.device)
        out = out + self.B1_noise.to(noise.device)*noise
        out = self.ada_in1(out,w)
        out = self.conv(out) 
        noise = torch.randn(out.size(0),1,out.size(2),out.size(3)).to(w.device)
        out =out + self.B2_noise.to(noise.device)*noise
        return self.ada_in2(out,w)

    def initialize_weights(self):
        nn.init.kaiming_normal(self.base, mode='fan_out', nonlinearity='relu')
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif (not isinstance(module, BaseBlock)) and hasattr(module,"initialize_weights"):
                module.initialize_weights()

#######################
# Generator Block     #
#######################
class SynthBlock(nn.Module):
    """Synthesys block, after upsampling"""
    def __init__(self,in_channels,out_channels,img_size,latent_dim = 512,use_batchNorm = True,activation = nn.LeakyReLU(0.2)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.out_channels = out_channels
        self.B_noise = nn.Parameter(1e-1*torch.randn(out_channels)).view(1,-1,1,1)
        
       
        self.conv1 = convBlock(
            in_chs=in_channels,out_chs=out_channels,
            kernel_size=3,padding = 1,use_batchNorm=use_batchNorm,activation = activation
        )
        
        self.conv2 = convBlock(
            in_chs=out_channels,out_chs=out_channels,
            kernel_size=3,padding = 1,use_batchNorm=use_batchNorm,activation = activation)

        
        self.ada_in1 = AdaIN(img_size,latent_dim,out_channels)
        self.ada_in2 = AdaIN(img_size,latent_dim,out_channels)
        


    def forward(self,img,w):
        out = self.conv1(img)
        #out: out_channels X img_size X img_size
        noise = torch.randn(out.size(0),1,out.size(2),out.size(3)).to(img.device)
        out =out + self.B_noise.to(noise.device)*noise
        out = self.ada_in1(out,w)
        out = self.conv2(out)
        noise = torch.randn(out.size(0),1,out.size(2),out.size(3)).to(img.device)
        out =out+ self.B_noise.to(noise.device)*noise
        out = self.ada_in2(out,w)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif not isinstance(m,SynthBlock) and hasattr(m, 'initialize_weights'):
                m.initialize_weights()
                
#####################################
#   Conv - BatchNorm - Activation   #
#####################################
class convBlock(nn.Sequential):
    def __init__(self,in_chs,out_chs,kernel_size = 4,
                 stride = 1,padding= 1,
                 activation = nn.LeakyReLU(0.2,inplace = False),
                 use_batchNorm = True):
        super().__init__()
        self.add_module(
           "conv",
           nn.Conv2d(
            in_channels = in_chs,
           out_channels = out_chs,
           kernel_size=kernel_size,
           padding=padding,stride = stride,bias = False)
        )
        if out_chs > 1 and use_batchNorm:
            self.add_module("bnorm",nn.BatchNorm2d(out_chs))
        if activation is not None:
            self.add_module("activation",activation)
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

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
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)


class DiscBlock(nn.Module):

    def __init__(self,
    in_channels,out_channels,
    activation = nn.LeakyReLU(0.2),
    use_batchNorm = True):
    
        super(DiscBlock, self).__init__()

        self.activation = activation
        self.conv1 = nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size=3,padding = 1,
        bias = not use_batchNorm,
         )
        self.conv2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size=3,padding = 1,
            bias = not use_batchNorm
        )
        if use_batchNorm:
            self.net = nn.Sequential(
                self.conv1,
                nn.BatchNorm2d(out_channels),
                self.activation,
                self.conv2,
                nn.BatchNorm2d(out_channels),
                self.activation,
            )
        else:
            self.net = nn.Sequential(
                self.conv1,
                self.activation,
                self.conv2,
                self.activation,
            )

    def forward(self, x):
        return self.net(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                