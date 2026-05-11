import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np
import DCE
from torch.nn import Softmax

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
            )
    def forward(self, x):
        x1 = self.conv(x)
        return x1

class conv_block3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block3, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 3, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(in_ch , out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(in_ch, 3, kernel_size=3, stride=1, padding=1, bias=True)
            )
    def forward(self, x):
        x1 = self.conv(x)
        return x1

class conv_block1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch , out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            )
    def forward(self, x):
        x1 = self.conv(x)
        return x1

class down_conv_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_conv_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
            nn.PReLU(),
            )

    def forward(self, x):
        x1 = self.conv(x)
        return x1

class up_conv_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv_2, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.up(x)
        return x

class down_conv_4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_conv_4, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=8, stride=4, padding=2, bias=True),
            nn.PReLU(),
            )
    def forward(self, x):
        x1 = self.conv(x)
        return x1

class up_conv_4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv_4, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=8, stride=4, padding=2, bias=True),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Inter(nn.Module):
    def __init__(self,channel):
        super(Inter, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            )
        self.softmax = Softmax(dim=-1)
        self.spatial1 = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.spatial2 = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )


    def forward(self, x, y):
        z1 = x - y
        x1 = (self.conv1_1(x) - self.conv3_1(x))
        B1,C1,H1,W1 =z1.size()
        z1_1 = z1.view(B1, -1, H1*W1)#C*HW
        x1_1 = x1.view(B1, -1, H1*W1).permute(0, 2, 1) #HW*C
        z1_2 = (torch.bmm(z1_1,x1_1)) #C*C
        min1 = torch.min(z1_2)
        max1 = torch.max(z1_2)
        norm1 = (z1_2 - min1) / (max1 - min1)
        z1_3 = self.softmax(norm1)
        x1_2 = x.view(B1, -1, H1*W1)#C*HW
        x1_3 = (torch.bmm(z1_3, x1_2)) #C*HW
        x1_4 = x1_3.view(B1, -1, H1, W1)
        x1_5 = self.spatial1(x) * x1_4 + x1_4
        x1_6 = x1_5 +x

        z2 = y - x
        y1 = (self.conv1_2(y) - self.conv3_2(y))
        B2,C2,H2,W2 =z2.size()
        z2_1 = z2.view(B2, -1, H2*W2)#C*HW
        y1_1 = y1.view(B2, -1, H2*W2).permute(0, 2, 1) # HW*C
        z2_2 = (torch.bmm(z2_1,y1_1)) #C*C
        min2 = torch.min(z2_2)
        max2 = torch.max(z2_2)
        norm2 = (z2_2 - min2) / (max2 - min2)
        z2_3 = self.softmax(norm2)
        y1_2 = y.view(B2, -1, H2*W2)# C*HW
        y1_3 = (torch.bmm(z2_3, y1_2)) #C*HW
        y1_4 = y1_3.view(B2, -1, H2, W2)
        y1_5 = self.spatial2(y) * y1_4 +y1_4
        y1_6 = y1_5 +y

        return  x1_6, y1_6




class fusion(nn.Module):
    def __init__(self,channel):
        super(fusion, self).__init__()

        self.Up2_1 = up_conv_2(channel, channel)
        self.Up2_2 = up_conv_2(channel, channel)
        # self.Up2_3 = up_conv_2(channel, channel)
        # self.Up2_4 = up_conv_2(channel, channel)

        self.Down2_1 = down_conv_2(channel, channel)
        self.Down2_2 = down_conv_2(channel, channel)
        self.Down2_3 = down_conv_2(channel, channel)

        self.conv1 = nn.Conv2d(2 * channel, channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(3 * channel, channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(2 * channel, channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * channel, channel, 3, 1, 1)
        # self.conv5 = nn.Conv2d(channel, channel, 3, 1, 1)
        # self.conv6 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.Conv1 = conv_block1(channel, channel)
        self.Conv2 = conv_block1(channel, channel)
        self.Conv3 = conv_block1(channel, channel)




    def forward(self, x, y, z):
        z1 = self.Down2_1(z)
        z2 = self.Down2_2(z1)
        y1 = self.Down2_3(y)
        y_cat = self.conv1(torch.cat((z1, y), dim=1))
        x_cat = self.conv2(torch.cat((z2, y1, x), dim=1))
        x_d = self.Conv1(x_cat)
        x_d1 = self.Up2_1(x_d)
        y_d = self.Conv2(self.conv3(torch.cat((x_d1, y_cat), dim=1)))
        y_d1 = self.Up2_2(y_d)
        U = self.Conv3(self.conv4(torch.cat((y_d1, z), dim=1)))

        return U


class Net(nn.Module):
    def __init__(self, channel):
        super(Net, self).__init__()
        self.DCE = DCE.DCENet()
        self.Conv1 = nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.Conv2 = nn.Conv2d(1, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.Conv3 = conv_block3(channel,channel)
        # self.Conv4 = conv_block1(channel,channel)
        # self.Conv3 = fusion(channel)
        self.Conv4 = fusion(channel)

        self.Conv7 = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.Conv8 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.Down2 = down_conv_2(channel,channel)
        self.Up2_1 = up_conv_2(channel,channel)
        self.Up2_2 = up_conv_2(channel, channel)
        self.Up2_3 = up_conv_2(channel, channel)
        self.Up2_4 = up_conv_2(channel, channel)
        self.Up2_5 = up_conv_2(channel, channel)
        self.Up2_6 = up_conv_2(channel, channel)


        self.Down4 = down_conv_4(channel,channel)
        self.Up4_1   = up_conv_4(channel,channel)
        self.Up4_2 = up_conv_4(channel, channel)

        self.Inter1 = Inter(channel)
        self.Inter2 = Inter(channel)
        self.Inter3 = Inter(channel)
        self.conv1 = conv_block(channel,channel)
        self.conv2 = conv_block(channel,channel)
        self.conv3 = conv_block(channel,channel)
        self.conv4 = conv_block(channel,channel)



    def forward(self, rgb, depth):
        # D1 = F.interpolate(depth, scale_factor=2, mode='bicubic', align_corners=True)
        # D = F.interpolate(depth, scale_factor=4, mode='bicubic', align_corners=True)
        rgb1 = self.DCE(rgb)
        r = self.Conv1(rgb1)
        r1 = self.Down4(r)
        depth1 = self.Conv2(depth)
        R1, D1 = self.Inter1(r1, depth1)

        r2_1 = self.Up2_1(R1)
        r2_2 = self.Down2(r)
        r2 = self.conv1(torch.cat((r2_1,r2_2),dim=1))

        d2_1 = self.Up2_2(depth1)
        d2_2 = self.Up2_3(D1)
        d2 = self.conv2(torch.cat((d2_1, d2_2),dim=1))

        R2, D2 = self.Inter2(r2, d2)

        r3_1 = self.Up2_4(R2)
        r3 = self.conv3(torch.cat((r,r3_1), dim=1))

        d3_1 = self.Up4_1(depth1)
        d3_2 = self.Up2_5(D2)
        d3 = self.conv4(torch.cat((d3_1,d3_2), dim=1))

        R3, D3 = self.Inter3(r3, d3)

        r4_1 = self.Up4_2(R1)
        r4_2 = self.Up2_6(R2)
        R4 = self.Conv3(torch.cat((r4_1, r4_2, R3),dim=1))

        D4 = self.Conv8(self.Conv4(D1, D2, D3))
        # d4 = d4 + D

        return rgb1, R4, D4









