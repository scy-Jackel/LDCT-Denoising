#!/usr/bin/python3

import argparse
import math
import os
import sys
import time
import warnings
from distutils.version import LooseVersion

import torch
import torch.nn.init as init
import torch.utils.data as Data
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from Datasets.aapm import BuildDataSet_dicom
from Models.networks.network import WGAN_CA_AE, update_model
from utils.save_img import save_fig

warnings.filterwarnings("ignore")

def main(args):
    patch_size = 256
    args.out_size = (patch_size, patch_size)
    model = WGAN_CA_AE(args,1,1)
    optimizer_g = optim.RMSprop(model.generator.parameters(), lr=args.lr_rate, alpha=0.9)
    optimizer_d = optim.RMSprop(model.discriminator.parameters(), lr=args.lr_rate, alpha=0.9)

    model, optimizer_g, optimizer_d = update_model(model, optimizer_g, optimizer_d, args.ckpt, args.start_epoch)

    print("--------------------------------- list var name -------------------------------")
    print(optimizer_g.state_dict()['param_groups'][0]['lr'])
    print(optimizer_d.state_dict()['param_groups'][0]['lr'])


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='WGAN_CA_AE',
                        help='a name for identitying the model. Choose from the following options: Unet')

    # Path related arguments
    parser.add_argument('--data_root_path', default='/home/cuiyang/data/zhilin/aapm_data_all',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output checkpoints')
    parser.add_argument('--save_img_path', default='./result',
                        help='folder to output checkpoints')

    # optimization related arguments
    parser.add_argument('--epoch_num', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=20, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 12)')
    parser.add_argument('--lr_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='weights regularizer')
    parser.add_argument('--particular_epoch', default=30, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=10, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')

    # other arguments
    parser.add_argument('--data', default='aapm', help='choose the dataset')
    parser.add_argument('--n_d_train', default=4, help='choose the dataset')
    parser.add_argument('--out_size', default=(256, 256), help='the output image size')
    parser.add_argument('--val_folder', default='folder0', type=str,
                        help='which cross validation folder')

    args = parser.parse_args()
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    args.ckpt = os.path.join(args.ckpt, 'wgan')
    print('Models are saved at %s' % (args.ckpt))
    args.save_img_path = os.path.join(args.save_img_path, 'wgan')
    print('images are saved at %s' % (args.save_img_path))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt + '/model/')
        os.makedirs(args.ckpt + '/optimizerG/')
        os.makedirs(args.ckpt + '/optimizerD/')
    if not os.path.isdir(args.save_img_path):
        os.makedirs(args.save_img_path)

    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str(args.start_epoch) + '_' + args.data + '_checkpoint.pth.tar'

    main(args)