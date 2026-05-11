import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np


class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.fpre = nn.Conv2d(nc//2, nc//2, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc//2, nc//2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc//2, nc//2, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc//2, nc//2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc//2, nc//2, 1, 1, 0))
        self.process3 = nn.Sequential(
            nn.Conv2d(nc//2, nc//2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc//2, nc//2, 1, 1, 0))
        self.process4 = nn.Sequential(
            nn.Conv2d(nc//2, nc//2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc//2, nc//2, 1, 1, 0))

        self.conv1 = nn.Conv2d(2*nc,nc,3,1,1)

    def forward(self, x):
        _, _, H, W = x.shape
        c = x.size(1) // 2
        x_1, x_2 = torch.split(x, c ,dim=1)
        x_freq1 = torch.fft.rfft2(self.fpre(x_1), norm='backward')
        mag1 = torch.abs(x_freq1)
        pha1 = torch.angle(x_freq1)
        mag1 = self.process1(mag1)
        pha1 = self.process2(pha1)
        x_freq2 = torch.fft.rfft2(self.fpre(x_2), norm='backward')
        mag2 = torch.abs(x_freq2)
        pha2 = torch.angle(x_freq2)
        mag2 = self.process3(mag2)
        pha2 = self.process4(pha2)

        real1 = mag1 * torch.cos(pha1)
        imag1 = mag1 * torch.sin(pha1)
        x_out1 = torch.complex(real1, imag1)
        x_out1 = torch.fft.irfft2(x_out1, s=(H, W), norm='backward')
        real2 = mag1 * torch.cos(pha2)
        imag2 = mag1 * torch.sin(pha2)
        x_out2 = torch.complex(real2, imag2)
        x_out2 = torch.fft.irfft2(x_out2, s=(H, W), norm='backward')
        real3 = mag2 * torch.cos(pha1)
        imag3 = mag2 * torch.sin(pha1)
        x_out3 = torch.complex(real3, imag3)
        x_out3 = torch.fft.irfft2(x_out3, s=(H, W), norm='backward')
        real4 = mag2 * torch.cos(pha2)
        imag4 = mag2 * torch.sin(pha2)
        x_out4 = torch.complex(real4, imag4)
        x_out4 = torch.fft.irfft2(x_out4, s=(H, W), norm='backward')
        x_out = self.conv1(torch.cat((x_out1,x_out2,x_out3,x_out4),dim=1))
        return x_out+x




class AmplitudeNet_skip(nn.Module):
    def __init__(self, nc):
        super(AmplitudeNet_skip,self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, nc, 1, 1, 0),
            FreBlock(nc),
        )
        self.conv1 = FreBlock(nc)
        self.conv2 = FreBlock(nc)
        self.conv3 = FreBlock(nc)
        self.conv4 = nn.Sequential(
            FreBlock(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.conv5 = nn.Sequential(
            FreBlock(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.convout1 = nn.Sequential(
            FreBlock(nc * 2),
            nn.Conv2d(nc * 2, 3, 1, 1, 0),
        )
        self.sigmoid = nn.Sigmoid()
        self.convout2 = nn.Sequential(
            FreBlock(nc * 2),
            nn.Conv2d(nc * 2, 24, 1, 1, 0),
        )

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(torch.cat((x2, x3), dim=1))
        x5 = self.conv5(torch.cat((x1, x4), dim=1))
        xout = self.convout1(torch.cat((x, x5), dim=1))
        xout1 = self.sigmoid(xout)
        xout2 = self.convout2(torch.cat((x, x5), dim=1))

        return xout1, xout2



class DCENet(nn.Module):
    def __init__(self):
        super(DCENet, self).__init__()

        self.AmpNet = AmplitudeNet_skip(8)




    def forward(self,x):

        curve_amps, x_r = self.AmpNet(x)

        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)

        _, _, H, W = enhance_image.shape
        image_fft = torch.fft.fft2(enhance_image, norm='backward')
        mag_image = torch.abs(image_fft)
        pha_image = torch.angle(image_fft)




        mag_image = mag_image / (curve_amps + 0.00000001)  # * d4
        real_image_enhanced = mag_image * torch.cos(pha_image)
        imag_image_enhanced = mag_image * torch.sin(pha_image)
        img_amp_enhanced = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                           norm='backward').real

        x_center = img_amp_enhanced


        return x_center






