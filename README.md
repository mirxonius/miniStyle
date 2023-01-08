# miniStyleGAN
We trained 3 separate models on the CelebA dataset, with a reduced resolution of 64x64.
## The models

#styleGAN
The first model considered was a standard styleGAN implemented in https://arxiv.org/abs/1812.04948, but reduced to produces images od smaller resolution.
#DCGAN
The baseline model used was a DCGAN
#Hybrid model
The last model considered was esentially a styleGAN but with deconvolutional upsampling.
