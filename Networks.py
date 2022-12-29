import torch
import torch.nn as nn

from NetBlocks import convBlock, deConvBlock,SynthBlock



class MappingNetwork(nn.Module):
    """
    Mapping network for style generation
    """
    def __init__(self,latent_dim = 128,n_layers = 8,activation = nn.LeakyReLU(inplace=True)) -> None:
        super().__init__()
        self.activation = activation
        self.network = nn.Sequential()
        for _ in range(n_layers):
            self.network.add_module(nn.Linear(latent_dim,latent_dim))
            self.network.add_module(self.activation)

    def forward(self,z):
        return self.network(z)



class SythesisNetwork(nn.Module):
    """
    styleGAN generator
    """
    def __init__(self):
        super().__init__()

        self.mapping_network = MappingNetwork()

    def forward(self,z):
        W = self.mapping_network(z)

        




class Discriminator(nn.Module):
    """
    DCGAN discriminator
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        self.layer1 = convBlock(
        in_chs = 3, out_chs =64 , kernel_size =4 , stride =2,  
        activation=self.lrelu
        )
        
        self.layer2 = convBlock(
        in_chs =64 , out_chs =128 , kernel_size = 4, stride =2,  
        activation=self.lrelu
        )
        
        self.layer3 = convBlock(
        in_chs =128 , out_chs =256 , kernel_size = 4, stride =2,  
        activation=self.lrelu
        )
        
        self.layer4 = convBlock(
        in_chs =256 , out_chs =512 , kernel_size = 4, stride =2,
        activation=self.lrelu
        )
        
        self.layer5 = convBlock(
        in_chs =512 , out_chs = 1, kernel_size = 4, stride =1, padding= 0,
        activation=None)
        

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out.squeeze()





class DCGenerator(nn.Module):
    """
    DCGAN generator
    """
    def __init__(self, latent_size=100,use_dropout=False):
        super().__init__()
        self.latent_size = latent_size

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()
        
        self.layer1 = deConvBlock(
            in_chs =self.latent_size ,out_chs=512, kernel_size =4 ,stride = 1, padding=0,
                        activation=self.lrelu)

        self.layer2 = deConvBlock(
            in_chs =512 ,out_chs=256, kernel_size =4 ,stride = 2, padding=1,
                       activation=self.lrelu )
        self.layer3 = deConvBlock(
            in_chs =256 ,out_chs=128, kernel_size =4 ,stride = 2, padding=1,
                        activation=self.lrelu)
        self.layer4 = deConvBlock(
            in_chs =128 ,out_chs=64, kernel_size =4 ,stride = 2, padding=1,
                        activation=self.lrelu)
        self.layer5 = deConvBlock(
            in_chs =64 ,out_chs=3, kernel_size =4 ,stride = 2, padding=1,
            activation = self.tanh  
                        )

        if use_dropout:
            self.dp1 = nn.Dropout2d(p = 0.25)
            self.dp2 = nn.Dropout2d(p = 0.25)
            self.dp3 = nn.Dropout2d(p = 0.25)
            self.dp4 = nn.Dropout2d(p = 0.25)
            self.net = nn.Sequential(
                self.layer1,
                self.dp1,
                self.layer2,
                self.dp2,
                self.layer3,
                self.dp3,
                self.layer4,
                self.dp4,
                self.layer5
            )
        else:
            self.net = nn.Sequential(
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
                self.layer5
            )

    def forward(self, x):
        out = self.net(x)
        return out