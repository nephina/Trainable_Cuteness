import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor

#Define the net characteristics

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,block_resolution_drop):
        super(ConvBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels,out_channels,3,1,1)
        self.conv_2 = nn.Conv2d(out_channels,out_channels,3,1,1)
        self.conv_3 = nn.Conv2d(out_channels,out_channels,3,1,1)
        self.skip_conv = nn.Conv2d(in_channels,out_channels,1,1,1)
        self.act = nn.ReLU()
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=block_resolution_drop)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()
        
    def init_weights(self,val=1./3):
        for m in self.modules():
            if type(m) == nn.Conv2d:
                nn.init.orthogonal_(m.weight,val)
                torch.nn.utils.weight_norm(m)
        
    def forward(self, x):
        skip = self.skip_conv(x)
        x = self.conv_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv_3(x)
        x = self.act(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.pool(x)
        skip = self.pool(skip)
        x = x + skip
        return x

class CNNSingleValueRanker(nn.Module):
    def __init__(self,image_size = 64):
        super().__init__()
        self.n_initial_channels = 6
        self.n_conv_blocks = 3
        self.block_resolution_drop = 4
        self.image_size = image_size
        self.conv_blocks = nn.ModuleList()

        channel_expansion = True
        for level in range(self.n_conv_blocks):
            if level == 0:
                self.conv_blocks.append(ConvBlock(3,
                                                  self.n_initial_channels,
                                                  self.block_resolution_drop))
            else:
                if channel_expansion:
                    block = ConvBlock(self.n_initial_channels*(2**(level-1)),
                                      self.n_initial_channels*(2**(level)),
                                      self.block_resolution_drop)
                    self.conv_blocks.append(block)
                else:
                    block = ConvBlock(self.n_initial_channels,
                                      self.n_initial_channels,
                                      self.block_resolution_drop)
                    self.conv_blocks.append(block)
        
        final_resolution_drop = (self.block_resolution_drop**self.n_conv_blocks)
        out_conv_size = self.image_size/final_resolution_drop
        out_conv_pixels = out_conv_size**2
        if channel_expansion:
            n_final_channels = (2**(self.n_conv_blocks-1))
            out_conv_channels = self.n_initial_channels*n_final_channels
            out_conv_params = out_conv_pixels*out_conv_channels
        else:
            out_conv_params = out_conv_pixels*self.n_initial_channels
        out_conv_params = int(out_conv_params)
        self.linear1 = nn.Linear(out_conv_params,
                                 int(0.3*out_conv_params))
        nn.init.orthogonal_(self.linear1.weight,1./3)
        self.linear2 = nn.Linear(int(0.3*out_conv_params),1)
        nn.init.orthogonal_(self.linear2.weight,1./3)

        self.act = nn.ReLU()

    def forward(self,x):
        for level in range(self.n_conv_blocks):
            x = self.conv_blocks[level](x)
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x