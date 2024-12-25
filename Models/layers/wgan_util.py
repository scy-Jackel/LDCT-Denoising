import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

## 2D Convolutional
##***********************************************************************************************************
class Conv_2d(nn.Module):
    """
    input:N*C*H*W
    """

    def __init__(self, in_ch, out_ch, use_bn="use_bn"):
        super().__init__()
        if use_bn is "use_bn":
            self.conv2d = nn.Sequential(
                # Conv2d input:N*C*H*W
                # Conv2d output:N*C*H*W
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2d = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(negative_slope=0.2),
            )

    def forward(self, x):
        out = self.conv2d(x)
        return out


## 3D Convolutional
##***********************************************************************************************************
class Conv_3d(nn.Module):
    """
    input:N*C*D*H*W
    """

    def __init__(self, in_ch, out_ch, use_bn="use_bn"):
        super().__init__()
        if use_bn is "use_bn":
            self.conv3d = nn.Sequential(
                # Conv3d input:N*C*D*H*W
                # Conv3d output:N*C*D*H*W
                nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv3d = nn.Sequential(
                nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        out = self.conv3d(x)
        return out


## Out Conv
##***********************************************************************************************************
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OutConv_2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


## AE_Conv
##***********************************************************************************************************
class AE_Down(nn.Module):
    """
    input:N*C*D*H*W
    batch_number*channel(1)*depth(3)*height*width
    """

    def __init__(self, in_channels, out_channels):
        super(AE_Down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AE_Up(nn.Module):
    """
    input:N*C*D*H*W
    """

    def __init__(self, in_channels, out_channels):
        super(AE_Up, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class AE_Down_2d(nn.Module):
    """
    input:N*C*H*W
    batch_number*channel(1)*height*width
    """

    def __init__(self, in_channels, out_channels):
        super(AE_Down_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AE_Up_2d(nn.Module):
    """
    input:N*C*H*W
    """

    def __init__(self, in_channels, out_channels):
        super(AE_Up_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


## Updata ae
##***********************************************************************************************************
def update_ae(model, ae_path):
     if os.path.isfile(ae_path):
          checkpoint = torch.load(ae_path)
          model.load_state_dict(checkpoint["model"].state_dict())
          del checkpoint
          torch.cuda.empty_cache()
          if torch.cuda.is_available():
               model = model.cuda()
          print("Ae Reload!\n")
     else:
          print("Can not reload Ae....\n")
          time.sleep(10)
          sys.exit(0)
     return model
