import torch
import torch.nn as nn

# Generator Block
def g_block(in_channels, out_channels, kernel_size, stride, padding):
    block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
    )
    return block

# Discriminator Block
def d_block(in_channels, out_channels, kernel_size, stride, padding):
    block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()    
    )
    return block

# Generator
class Generator(nn.Module):
    def __init__(self, d=64):
        super(Generator, self).__init__()
        self.block1 = g_block(in_channels=100, out_channels=d*4, kernel_size=4, stride=1, padding=0)
        self.block2 = g_block(in_channels=d*4, out_channels=d*2, kernel_size=3, stride=2, padding=1)
        self.block3 = g_block(in_channels=d*2, out_channels=d, kernel_size=4, stride=2, padding=1)
        self.output = nn.Sequential(
            nn.ConvTranspose2d(in_channels=d, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        ) 
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output(x)
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        self.input  = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ) 
        self.block1 = d_block(in_channels=d, out_channels=d*2, kernel_size=4, stride=2, padding=1)
        self.block2 = d_block(in_channels=d*2, out_channels=d*4, kernel_size=3, stride=2, padding=1)
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=d*4, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):        
        x = self.input(x)
        x = self.block1(x)
        x = self.block2(x)
        feature = x
        x = self.output(x)
        return x, feature

def init_weights(layer):
    layername = layer.__class__.__name__
    if layername.find('Conv') != -1:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif layername.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)