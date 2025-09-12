import torch
import torch.nn as nn
import torch.nn.functional as F

# Gated conv for inpainting
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.ELU(inplace=True)):
        super(GatedConv2d, self).__init__()
        self.activation = activation
        
        # Standard convolution for feature extraction
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Gating convolution learns the mask
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.activation(self.conv2d(x))
        gate = self.sigmoid(self.mask_conv2d(x))
        return features * gate

class GatedConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=2, padding=1, activation='relu', bn=True):
        super().__init__()
        
        if activation == 'relu':
            act = nn.ReLU(True)
        elif activation == 'leaky':
            act = nn.LeakyReLU(0.2, True)
        else:
            act = nn.ELU(True)

        self.gated_conv = GatedConv2d(in_c, out_c, kernel_size, stride, padding, activation=act)
        self.bn = nn.BatchNorm2d(out_c) if bn else nn.Identity()
            
    def forward(self, x):
        x = self.gated_conv(x)
        x = self.bn(x)
        return x

class GatedConvUNet(nn.Module):
    def __init__(self, input_channels=4): # Input is Image(3) + Mask(1) = 4 channels
        super().__init__()
        # encoder
        self.enc_1 = GatedConvBlock(input_channels, 64, kernel_size=7, stride=2, padding=3, bn=False, activation='leaky')
        self.enc_2 = GatedConvBlock(64, 128, kernel_size=5, stride=2, padding=2, activation='leaky')
        self.enc_3 = GatedConvBlock(128, 256, kernel_size=5, stride=2, padding=2, activation='leaky')
        self.enc_4 = GatedConvBlock(256, 512, kernel_size=3, stride=2, padding=1, activation='leaky')
        
        # Bottleneck
        self.bottleneck = GatedConvBlock(512, 512, kernel_size=3, stride=2, padding=1, activation='leaky')

        # decoder
        self.dec_4 = GatedConvBlock(512 + 512, 512, stride=1, activation='relu')
        self.dec_3 = GatedConvBlock(512 + 256, 256, stride=1, activation='relu')
        self.dec_2 = GatedConvBlock(256 + 128, 128, stride=1, activation='relu')
        self.dec_1 = GatedConvBlock(128 + 64, 64, stride=1, activation='relu')
        
        # final layers
        self.final_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.final_conv = nn.Conv2d(64, 3, 3, 1, 1)
        
    def forward(self, x, mask):
        x_with_mask = torch.cat([x, mask], dim=1)

        # encoder
        x1 = self.enc_1(x_with_mask)
        x2 = self.enc_2(x1)
        x3 = self.enc_3(x2)
        x4 = self.enc_4(x3)
        
        xb = self.bottleneck(x4)

        # decoder
        dx = F.interpolate(xb, scale_factor=2, mode='nearest')
        dx = torch.cat([dx, x4], 1)
        dx = self.dec_4(dx)
        
        dx = F.interpolate(dx, scale_factor=2, mode='nearest')
        dx = torch.cat([dx, x3], 1)
        dx = self.dec_3(dx)

        dx = F.interpolate(dx, scale_factor=2, mode='nearest')
        dx = torch.cat([dx, x2], 1)
        dx = self.dec_2(dx)

        dx = F.interpolate(dx, scale_factor=2, mode='nearest')
        dx = torch.cat([dx, x1], 1)
        dx = self.dec_1(dx)
        
        dx = self.final_upsample(dx)
        out = self.final_conv(dx)
        
        return torch.sigmoid(out)
