#!/usr/bin/python3

import argparse
import math
import os
import time
import warnings
from distutils.version import LooseVersion

import cv2
import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from Datasets.aapm import BuildDataSet_dicom
from Datasets.aapmSequece import DicomData
from Models.networks.network import Comprehensive_Atten_Unet
from Models.simpleNet import SACNN
from metric.fsim import FSIM
from utils.dice_loss import SoftDiceLoss
from utils.evaluation import AverageMeter
from utils.save_img import save_fig, save_fig4, save_fig6
from utils.splitCombine import split_tensor, combine_np
from utils.train_logger import write_log
from utils.dicom_trans import get_data

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

Test_Model = {'Comp_Atten_Unet': Comprehensive_Atten_Unet,
              'simpleNet': SACNN}

# numpy version
def getMask(x, y):
    resi = y - x
    print('resi shape:', resi.shape)  # (16,1,256,256)
    resi_cp = resi.copy()
    resi[resi <= 0] = 0
    resi[resi > 0] = 1
    resi_cp[resi >= 0] = 0
    resi_cp[resi < 0] = 1
    # ret = np.concatenate((resi, resi_cp), axis=1)
    # ret = torch.cat([resi, resi_cp], dim=1)  # first is >0, second is <0.
    ret = resi
    print('ret shape:', ret.shape)  # (16,2,256,256)
    return ret

def main():
    data_path = '/home/cuiyang/data/zhilin/aapm_data_all/L506'
    save_path = './result/aapm_binary/whole'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    low_path = os.path.join(data_path, 'quarter_3mm')
    full_path = os.path.join(data_path, 'full_3mm')
    low_image = get_data(low_path)
    full_image = get_data(full_path)
    c,w,h = low_image.shape
    index = 20
    low_slice = low_image[index]
    full_slice = full_image[index]
    mask = getMask(low_slice, full_slice)
    mask= mask*255
    # plt.imshow(low_slice, cmap='gray')
    # plt.savefig(save_path+'/L506_20_255.png')
    cv2.imwrite(save_path+'/L506_20_low.png', low_slice*255)
    cv2.imwrite(save_path+'/L506_20_full.png', full_slice*255)
    cv2.imwrite(save_path+'/L506_20_mask.png', mask)
    pass


if __name__ == '__main__':
    main()