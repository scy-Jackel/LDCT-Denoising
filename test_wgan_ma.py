#!/usr/bin/python3

import argparse
import math
import os
import sys
import time
import warnings
from distutils.version import LooseVersion

import numpy as np
import torch
import torch.nn.init as init
import torch.utils.data as Data
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from Datasets.aapm import BuildDataSet_dicom
from Datasets.aapmSequece import DicomData
from Models.networks.maNet import WGAN_MA_AE, update_model
from metric.psnr import compute_PSNR, compute_RMSE
from metric.ssim import compute_ssim
from utils.dicom_trans import denormAndTrunc_cuda, denormAndTrunc_np
from utils.save_img import save_fig, save_fig4
from utils.splitCombine import split_tensor, combine_np

warnings.filterwarnings("ignore")

Test_Model = {'WGAN_MA_AE': WGAN_MA_AE}

def compute_measure(quarter, pred, full):
    psnr_quarter = compute_PSNR(quarter, full, data_range=255)
    psnr_pred = compute_PSNR(pred, full,data_range=255)
    rmse_quarter = compute_RMSE(quarter, full)
    rmse_pred = compute_RMSE(pred, full)
    ssim_quarter, ssim_pred = compute_ssim(quarter, pred, full)
    return (psnr_quarter, ssim_quarter, rmse_quarter), (psnr_pred, ssim_pred, rmse_pred)


def test_aapm(model, test_loader, args):
    re_load = False
    if args.start_epoch!=0:
        re_load = True
    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr_rate, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr_rate, betas=(0.5, 0.9))

    # optimizer_g = optim.RMSprop(model.generator.parameters(), lr=args.lr_rate, alpha=0.9)
    # optimizer_d = optim.RMSprop(model.discriminator.parameters(), lr=args.lr_rate, alpha=0.9)

    if re_load is False:
        print('please set the test epoch.')
    else:
        print("Re_load is True !")
        model, optimizer_g, optimizer_d = update_model(model, optimizer_g, optimizer_d, args.ckpt, args.start_epoch)

    model.eval()
    for i, batch in enumerate(test_loader):
        full_image = batch["full_image"]
        quarter_image = batch["quarter_image"]
        full_image = Variable(full_image).cuda()
        quarter_image = Variable(quarter_image).cuda()
        with torch.no_grad():
            M = model.attentionBlock(quarter_image)
            image_pred = model.generator(quarter_image, M)
        full_image_np = denormAndTrunc_cuda(full_image)
        quarter_image_np = denormAndTrunc_cuda(quarter_image)
        image_pred_np = denormAndTrunc_cuda(image_pred)
        resi = np.abs(image_pred_np-full_image_np)

        orig_metric, pred_metric = compute_measure(quarter_image_np, image_pred_np, full_image_np)

        # save_fig(quarter_image_np, full_image_np, image_pred_np, i, orig_metric, pred_metric, args.save_img_path)
        save_fig4(quarter_image_np, full_image_np, image_pred_np,resi, i, orig_metric, pred_metric, args.save_img_path)
        print(i, 'saved')
    print('test done.')



def test_whole(test_loader, model, save_path):
    re_load = False
    if args.start_epoch!=0:
        re_load = True
    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr_rate, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr_rate, betas=(0.5, 0.9))

    # optimizer_g = optim.RMSprop(model.generator.parameters(), lr=args.lr_rate, alpha=0.9)
    # optimizer_d = optim.RMSprop(model.discriminator.parameters(), lr=args.lr_rate, alpha=0.9)

    if re_load is False:
        print('please set the test epoch.')
    else:
        print("Re_load is True !")
        model, optimizer_g, optimizer_d = update_model(model, optimizer_g, optimizer_d, args.ckpt, args.start_epoch)
    save_path = os.path.join(save_path, 'whole/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.eval()
    for step, batch_data in tqdm(enumerate(test_loader), total=len(test_loader)):
        x = batch_data['quarter_image']
        y = batch_data['full_image']
        image = x.float().cuda()
        target = y.float().cuda()
        # target = target - image

        img_arr = split_tensor(image, patch_size=256)  # torch.tensor
        target_arr = split_tensor(target, patch_size=256)
        output_arr = []  # np
        for i in range(len(img_arr)):
            # print('input shape:', img_arr[i].shape)
            with torch.no_grad():
                M = model.attentionBlock(img_arr[i])
                image_pred = model.generator(img_arr[i], M)
            output_arr.append(image_pred.cpu().detach().numpy())

        whole_output = combine_np(output_arr, patch_size=256)
        x = image.cpu().detach().numpy().squeeze()
        y = target.cpu().detach().numpy().squeeze()
        whole_output = whole_output.squeeze()
        # resi = np.abs(y-whole_output)

        full_image_np = denormAndTrunc_np(y)
        quarter_image_np = denormAndTrunc_np(x)
        image_pred_np = denormAndTrunc_np(whole_output)
        resi = np.abs(image_pred_np-full_image_np)

        orig_metric, pred_metric = compute_measure(quarter_image_np, image_pred_np, full_image_np)

        save_fig4(quarter_image_np, full_image_np, image_pred_np, resi, step, orig_metric, pred_metric, save_path)
    print('test done.')



def main(args):

    test_folder = ["L067"]
    data_length = {'train':5000, 'val':2000, 'test': 20}
    patch_size = 256
    args.num_input = 1
    args.num_classes = 1
    args.out_size = (patch_size, patch_size)
    # test_datasets = BuildDataSet_dicom(args.data_root_path, test_folder, None, data_length['test'], 'test', patch_size, in_channel=args.num_input, out_channel=args.num_classes)
    # testloader = Data.DataLoader(dataset=test_datasets, batch_size=1, shuffle=False, pin_memory=True)
    test_Datasets = DicomData(args.data_root_path, test_folder, depth=1, size=512)
    testloader = Data.DataLoader(dataset=test_Datasets, batch_size=1, shuffle=False)
    print('Loading is done\n')

    model = Test_Model[args.id](args, args.num_input, args.num_classes)
    # if args.resume:
    #     model = model_updata(model, model_old_name=args.model_name + "{}".format(model_index),
    #                          model_old_path=args.model_path)
    model = model.cuda()
    # test_aapm(model, testloader, args)
    test_whole(testloader, model, args.save_img_path)

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='WGAN_MA_AE',
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
    parser.add_argument('--start_epoch', default=80, type=int,
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

    args.ckpt = os.path.join(args.ckpt, 'wgan_ma_051312')
    print('Models are saved at %s' % (args.ckpt))
    args.save_img_path = os.path.join(args.save_img_path, 'wgan_ma_051312')
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