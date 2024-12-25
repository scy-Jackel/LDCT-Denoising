import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import time


class Conv_2d(nn.Module):
    """
    input:N*C*D*H*W
    """
    def __init__(self, in_ch, out_ch, use_relu="use_relu"):
        super().__init__()
        if use_relu is "use_relu":
            self.conv2d = nn.Sequential(
                # Conv3d input:N*C*D*H*W
                # Conv3d output:N*C*D*H*W
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2d = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        out = self.conv2d(x)
        return out


## Out Conv
##***********************************************************************************************************
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SACNN(nn.Module):
    def __init__(self, args, in_channel, output_channel, N=4):
        super(SACNN, self).__init__()
        self.input_channels = in_channel
        self.output_channels = output_channel
        self.N = N
        self.lay1 = Conv_2d(in_ch=self.input_channels, out_ch=64, use_relu="use_relu")
        self.lay2 = Conv_2d(in_ch=64, out_ch=32, use_relu="use_relu")
        # self.lay3 = SA(in_ch=32, out_ch=32, N=self.N)
        self.lay3 = Conv_2d(in_ch=32, out_ch=32, use_relu="use_relu")
        self.lay4 = Conv_2d(in_ch=32, out_ch=16, use_relu="use_relu")
        # self.lay5 = SA(in_ch=16, out_ch=16, N=self.N)
        self.lay5 = Conv_2d(in_ch=16, out_ch=16, use_relu="use_relu")
        self.lay6 = Conv_2d(in_ch=16, out_ch=32, use_relu="use_relu")
        # self.lay7 = SA(in_ch=32, out_ch=32, N=self.N)
        self.lay7 = Conv_2d(in_ch=32, out_ch=32, use_relu="use_relu")
        self.lay8 = Conv_2d(in_ch=32, out_ch=64, use_relu="use_relu")
        self.lay9 = OutConv(in_ch=64, out_ch=self.output_channels)

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        x = self.lay6(x)
        x = self.lay7(x)
        x = self.lay8(x)
        x = self.lay9(x)
        return x


class SACNN_sigmoid(nn.Module):
    def __init__(self, args, in_channel, output_channel, N=4):
        super(SACNN_sigmoid, self).__init__()
        self.input_channels = in_channel
        self.output_channels = output_channel
        self.N = N
        self.lay1 = Conv_2d(in_ch=self.input_channels, out_ch=64, use_relu="use_relu")
        self.lay2 = Conv_2d(in_ch=64, out_ch=32, use_relu="use_relu")
        # self.lay3 = SA(in_ch=32, out_ch=32, N=self.N)
        self.lay3 = Conv_2d(in_ch=32, out_ch=32, use_relu="use_relu")
        self.lay4 = Conv_2d(in_ch=32, out_ch=16, use_relu="use_relu")
        # self.lay5 = SA(in_ch=16, out_ch=16, N=self.N)
        self.lay5 = Conv_2d(in_ch=16, out_ch=16, use_relu="use_relu")
        self.lay6 = Conv_2d(in_ch=16, out_ch=32, use_relu="use_relu")
        # self.lay7 = SA(in_ch=32, out_ch=32, N=self.N)
        self.lay7 = Conv_2d(in_ch=32, out_ch=32, use_relu="use_relu")
        self.lay8 = Conv_2d(in_ch=32, out_ch=64, use_relu="use_relu")
        self.lay9 = OutConv(in_ch=64, out_ch=self.output_channels)
        self.final = nn.Sigmoid()

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        x = self.lay6(x)
        x = self.lay7(x)
        x = self.lay8(x)
        x = self.lay9(x)
        x = self.final(x)
        return x



class RED_CNN(nn.Module):
    def __init__(self, args, inchannel, outchannel, inter_ch=64):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, inter_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(inter_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out


class RED_CNN_sigmoid(nn.Module):
    def __init__(self, args, inchannel, outchannel, inter_ch=64):
        super(RED_CNN_sigmoid, self).__init__()
        self.conv1 = nn.Conv2d(1, inter_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(inter_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.final = nn.Sigmoid()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        out = self.final(out)
        return out