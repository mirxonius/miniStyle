import torch
import torch.nn as nn
from NetBlocks import convBlock, deConvBlock, SynthBlock, BaseBlock,AdaIN, DiscBlock



class MappingNetwork(nn.Module):
    """
    Mapping network for style generation
    """
    def __init__(self,latent_dim = 512,n_layers = 8,activation = nn.LeakyReLU(inplace=True)):
        super().__init__()
        self.activation = activation
        modules = list()
        for _ in range(n_layers):
            modules.append(nn.Linear(latent_dim,latent_dim))
            modules.append(self.activation)
        self.network = nn.Sequential(*modules)
        
    def forward(self,z):
        return self.network(z)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data,mode='fan_out', nonlinearity='relu')
                



class SythesisNetwork(nn.Module):
    """
    styleGAN generator
    """
    def __init__(self,latent_dim = 512,):
        super().__init__()

        self.mapping_network = MappingNetwork(latent_dim=latent_dim)
        self.base = BaseBlock(latent_dim=latent_dim)
        self.up1 = nn.Upsample(size = 8,mode = "bilinear")
        self.block1 = SynthBlock(
            in_channels=256,out_channels=128,img_size=8,latent_dim=latent_dim
        )
        self.up2 = nn.Upsample(size = 16,mode = "bilinear")
        self.block2 =SynthBlock(
            in_channels=128,out_channels=128,img_size=16,latent_dim=latent_dim
        )
        self.up3 = nn.Upsample(size = 32,mode = "bilinear")
        self.block3 = SynthBlock(
            in_channels=128,out_channels=64,img_size=32,latent_dim=latent_dim
        )
        self.up4 = nn.Upsample(size = 64,mode = "bilinear")
        self.block4 = SynthBlock(
            in_channels=64,out_channels=3,img_size=64,latent_dim=latent_dim
            )
        self.tanh = nn.Tanh()
        self.initialize_weights()

    def forward(self,z):
        W = self.mapping_network(z)
        out = self.base(W)
        
        out = self.up1(out)
        out = self.block1(out,W)
        
        out = self.up2(out)
        out = self.block2(out,W)
        
        out = self.up3(out)
        out = self.block3(out,W)

        out = self.up4(out)
        out = self.block4(out,W)
        return self.tanh(out)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif (not isinstance(m,SythesisNetwork)) and hasattr(m,"initialize_weights"):
                m.initialize_weights()


class Discriminator(nn.Module):
    """
    DCGAN discriminator
    """

    def __init__(self):
        super().__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        self.layer1 = convBlock(
        in_chs = 3, out_chs = 64 , kernel_size =4 , stride =2,  
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
        
        self.initialize_weights()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out.squeeze()

    def initialize_weights(self):
        for m in self.modules():
            if hasattr(m, 'initialize_weights') and not isinstance(m,Discriminator):
                m.initialize_weights()



class DCStyleGenerator(nn.Module):
    def __init__(self,latent_size = 100,) -> None:
        super().__init__()
        """
        DC GAN generator that uses styes with ADAIN
        """
        self.latent_size = latent_size
        self.mapping_network = MappingNetwork(latent_dim=latent_size)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()
        
        self.layer1 = deConvBlock(
            in_chs =self.latent_size ,out_chs=512, kernel_size = 4 ,stride = 1, padding=0,
                        activation=self.lrelu)
        self.ada_in1 = AdaIN(4,latent_size,512)

        self.layer2 = deConvBlock(
            in_chs =512 ,out_chs=256, kernel_size = 4 ,stride = 2, padding=1,
                       activation=self.lrelu )
        self.ada_in2 = AdaIN(8,latent_size,256)

        self.layer3 = deConvBlock(
            in_chs =256 ,out_chs=256, kernel_size = 4 ,stride = 2, padding=1,
                        activation=self.lrelu)
        self.ada_in3 = AdaIN(16,latent_size,256)
        self.layer4 = deConvBlock(
            in_chs =256 ,out_chs=128, kernel_size = 4 ,stride = 2, padding=1,
                        activation=self.lrelu)
        self.ada_in4 = AdaIN(32,latent_size,128)
        self.layer5 = deConvBlock(
            in_chs =128 ,out_chs=3, kernel_size = 4 ,stride = 2, padding=1,
            activation = self.tanh  
                        )
        #self.ada_in5 = AdaIN(32,latent_size,3)

    def forward(self,z):
        w = self.mapping_network(z.view(-1,self.latent_size))
        
        out = self.layer1(z)
        out = self.ada_in1(out,w)
        out = self.layer2(out)
        out = self.ada_in2(out,w)
        out = self.layer3(out)
        out = self.ada_in3(out,w)
        out = self.layer4(out)
        out = self.ada_in4(out,w)
        out = self.layer5(out)
        #out = self.ada_in5(out,w)
        return self.tanh(out)

    def inialize_weights(self):
        for m in self.modules():
            if hasattr(m, 'initialize_weights') and not isinstance(m,DCStyleGenerator):
                m.initialize_weights()
            



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
            in_chs =256 ,out_chs=256, kernel_size =4 ,stride = 2, padding=1,
                        activation=self.lrelu)
        self.layer4 = deConvBlock(
            in_chs =256 ,out_chs=128, kernel_size =4 ,stride = 2, padding=1,
                        activation=self.lrelu)
        self.layer5 = deConvBlock(
            in_chs =128 ,out_chs=3, kernel_size =4 ,stride = 2, padding=1,
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



class StyleDiscrimnator(nn.Module):
    """
    Disciminator similar to the one described in: https://arxiv.org/pdf/1710.10196.pdf
    It uses convolutional blocks and downsampling
    """
    def __init__(self,img_resolution = 64):
        super().__init__()
        
        self.img_resolution = img_resolution
        n_blocks = int(torch.log2(64))
        blocks = []
        res = self.img_resolution
        for _ in range(n_blocks):
            res /= 2
            blocks.append(
                DiscBlock(in_channels=512,out_channels=512)
                )
            #Upsample can be used for downsampling as well    
            blocks.append(nn.Upsample(
                size =int(res),mode = "bilinear")
                )

        blocks.append(nn.Flatten())
        blocks.append(nn.Linear(512,1))
        self.net = nn.Sequential(*blocks)


    def forward(self, x):
        return self.net(x)