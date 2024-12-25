#!/usr/bin/python3

import argparse
import math
import os
import time
import warnings
from distutils.version import LooseVersion

import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from Datasets.aapm import BuildDataSet_dicom
from Datasets.aapmSequece import DicomData
from Models.networks.network import Comprehensive_Atten_Unet
from Models.simpleNet import SACNN,SACNN_sigmoid, RED_CNN, RED_CNN_sigmoid
from metric.fsim import FSIM
from utils.dice_loss import SoftDiceLoss
from utils.evaluation import AverageMeter
from utils.save_img import save_fig, save_fig4, save_fig6, save_fig4_squre, save_fig_cv
from utils.splitCombine import split_tensor, combine_np
from utils.train_logger import write_log

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

Test_Model = {'Comp_Atten_Unet': Comprehensive_Atten_Unet,
              'simpleNet': SACNN,
              'simpleNetsigmoid': SACNN_sigmoid,
              'redcnn': RED_CNN,
              'redcnnsigmoid': RED_CNN_sigmoid}


def getMask(x, y):
    resi = y - x
    # print('resi shape:', resi.shape)  # (16,1,256,256)
    resi_cp = resi.clone()
    resi[resi <= 0] = 0
    resi[resi > 0] = 1
    resi_cp[resi >= 0] = 0
    resi_cp[resi < 0] = 1
    ret = resi
    # ret = torch.cat([resi, resi_cp], dim=1)  # first is >0, second is <0.
    # print('ret shape:', ret.shape)  # (16,2,256,256)
    return ret


def train(train_loader, model, criterion, optimizer, args, epoch):
    losses = AverageMeter()

    model.train()
    for step, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):
        x = batch_data['quarter_image']
        y = batch_data['full_image']
        mask = getMask(x,y)
        image = x.float().cuda()
        target = mask.float().cuda()

        output = model(image)                                      # model output
        # print('check shape:',output.shape, target.shape)
        loss = criterion(output, target)

        if torch.isnan(loss):
            x = image.cpu().detach().numpy().squeeze()
            y = target.cpu().detach().numpy().squeeze()
            pred = output.cpu().detach().numpy().squeeze()
            orig_metric = [0, 0, 0]
            pred_metric = [0, 0, 0]
            for i in range(args.batch_size):
                save_fig(x[i], y[i], pred[i], 'nan_{}'.format(i), orig_metric, pred_metric, './result/')
            time.sleep(100)
            exit(-1)

        losses.update(loss.data, image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if step % (math.ceil(float(len(train_loader.dataset))/args.batch_size)) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {losses.avg:.6f}'.format(
                epoch, step * len(image), len(train_loader.dataset),
                100. * step / len(train_loader), losses=losses))

    print('The average loss:{losses.avg:.4f}'.format(losses=losses))
    return losses.avg


def test_aapm(test_loader, model, save_path):

    # modelname = r'/home/cuiyang/bishe/LDCTDenoising_MA/saved_models/aapm/mse/300_aapm_checkpoint.pth.tar'
    # modelname = r'/home/cuiyang/bishe/LDCTDenoising_MA/saved_models/aapm_binary/l1loss/40_aapm_binary_checkpoint.pth.tar'
    # modelname = r'/home/cuiyang/bishe/LDCTDenoising_MA/saved_models/aapm_binary/dice/60_aapm_binary_checkpoint.pth.tar'
    # modelname = r'/home/cuiyang/bishe/LDCTDenoising_MA/saved_models/aapm_binary/redcnnsigmoid/2_aapm_binary_checkpoint.pth.tar'
    modelname = r'/home/cuiyang/bishe/LDCTDenoising_MA/saved_models/aapm_binary/simpleNet_sigmoid/90_aapm_binary_checkpoint.pth.tar'
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        # start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['opt_dict'])
        print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    model.eval()
    for step, batch_data in tqdm(enumerate(test_loader), total=len(test_loader)):
        x = batch_data['quarter_image']
        y = batch_data['full_image']
        # image = x.float().cuda()
        # target = y.float().cuda()
        # target = target - image

        mask = getMask(x,y)
        image = x.float().cuda()
        target = mask.float().cuda()

        output = model(image)                                   # model output
        # x = image.cpu().detach().numpy().squeeze()
        # y = target.cpu().detach().numpy().squeeze()
        # pred = output.cpu().detach().numpy().squeeze()
        # resi = np.abs(y-pred)
        # orig_metric = [0,0,0]
        # pred_metric = [0,0,0]
        #
        # save_fig4(x, y, pred, resi, step, orig_metric, pred_metric, save_path)

        x = x.squeeze()
        y = y.squeeze()
        pred = output.cpu().detach().numpy().squeeze()
        mask = mask.cpu().detach().numpy().squeeze()

        print('mask shape:', mask.shape)
        print('pred shape:', pred.shape)
        # mask_posi = mask[0][0].squeeze()
        # mask_neg = mask[0][1].squeeze()
        # output_posi = pred[0].squeeze()
        # output_neg = pred[1].squeeze()
        #
        # save_fig6(x, y, mask_posi, mask_neg, output_posi, output_neg, save_path, step)
        save_fig4_squre(x,y, mask, pred, save_path, step)

    print('test done.')


def test_whole(test_loader, model, save_path):
    # modelname = r'/home/cuiyang/bishe/LDCTDenoising_MA/saved_models/aapm/l1loss/300_aapm_checkpoint.pth.tar'
    modelname = r'/home/cuiyang/bishe/LDCTDenoising_MA/saved_models/aapm_binary/l1loss/100_aapm_binary_checkpoint.pth.tar'
    # modelname = r'/home/cuiyang/bishe/LDCTDenoising_MA/saved_models/aapm_binary/simpleNet_sigmoid/3_aapm_binary_checkpoint.pth.tar'
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        # start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['opt_dict'])
        print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))
    total_resi=0.0
    CRIT = nn.L1Loss()
    model.eval()
    for step, batch_data in tqdm(enumerate(test_loader), total=len(test_loader)):
        x = batch_data['quarter_image']
        y = batch_data['full_image']
        # image = x.float().cuda()
        # target = y.float().cuda()
        # target = target - image

        mask = getMask(x,y)
        image = x.float().cuda()
        target = mask.float().cuda()

        img_arr = split_tensor(image, patch_size=256)  # torch.tensor
        target_arr = split_tensor(target, patch_size=256)
        output_arr = []  # np
        for i in range(len(img_arr)):
            # print('input shape:', img_arr[i].shape)
            output_i = model(img_arr[i])
            output_i = torch.index_select(output_i, 1, torch.tensor([0]).cuda())
            # print(output_i.shape)
            output_arr.append(output_i.cpu().detach().numpy())

        whole_output = combine_np(output_arr, patch_size=256)
        x = image.cpu().detach().numpy().squeeze()
        y = target.cpu().detach().numpy().squeeze()
        whole_output = whole_output.squeeze()
        resi = np.abs(y-whole_output)

        total_resi += CRIT(torch.from_numpy(y), torch.from_numpy(whole_output)).data
        orig_metric = [0,0,0]
        pred_metric = [0,0,0]
        # save_fig_cv(x, y, whole_output, resi, step, save_path)
        # save_fig4(x, y, whole_output, resi, step, orig_metric, pred_metric, save_path)
    print('avg resi:', total_resi/len(test_loader))

    print('test done.')


def main(args):
    minloss = [1.0]
    start_epoch = args.start_epoch
    # Define model
    # loading the dataset
    train_folder = ["L192", "L286", "L310", "L333", "L506", "L143"]
    test_folder = ["L506"]
    val_folder = ["L067", "L096", "L109"]
    data_length = {'train':5000, 'val':2000, 'test':40}
    patch_size = 256
    args.num_input = 1
    args.num_classes = 1
    args.out_size = (patch_size, patch_size)
    # train_datasets = BuildDataSet_dicom(args.data_root_path, train_folder, None, data_length['train'], 'train', patch_size, in_channel=args.num_input, out_channel=args.num_classes)
    # trainloader = Data.DataLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_Datasets = DicomData(args.data_root_path, test_folder, depth=1, size=512)
    testloader = Data.DataLoader(dataset=test_Datasets, batch_size=1, shuffle=False)
    print('Loading is done\n')



    model = Test_Model[args.id](args, args.num_input, args.num_classes)

    if torch.cuda.is_available():
        print('We can use', torch.cuda.device_count(), 'GPUs to train the network')
        model = model.cuda()

    # Define optimizers and loss function
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr_rate,
                                 weight_decay=args.weight_decay)    # optimize all model parameters
    # criterion = FSIM(channels=1)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    # criterion = SoftDiceLoss()

    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))


    # print("Start training ...")
    # for epoch in range(start_epoch + 1, args.epochs + 1):
    #
    #     train_avg_loss = train(trainloader, model, criterion, optimizer, args, epoch)
    #     write_log("epoch {}:train_avg_loss {}".format(epoch, train_avg_loss))
    #     scheduler.step()
    #
    #     # save models
    #     # if epoch > args.particular_epoch:
    #     #     if epoch % args.save_epochs_steps == 0:
    #     if epoch > 0:
    #         if epoch % 1 == 0:
    #             filename = args.ckpt + '/' + str(epoch) + '_' + args.data + '_checkpoint.pth.tar'
    #             print('the model will be saved at {}'.format(filename))
    #             state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
    #             torch.save(state, filename)

    print("Start testing ...")
    # test_aapm(testloader, model, args.save_img_path)
    test_whole(testloader, model, args.save_img_path)


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='Comp_Atten_Unet',
                        help='a name for identitying the model. Choose from the following options: Unet')

    # Path related arguments
    parser.add_argument('--data_root_path', default='/home/cuiyang/data/zhilin/aapm_data_all',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output checkpoints')
    parser.add_argument('--save_img_path', default='./result',
                        help='folder to output checkpoints')

    # optimization related arguments
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 12)')
    parser.add_argument('--lr_rate', type=float, default=2e-4, metavar='LR',
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
    parser.add_argument('--out_size', default=(256, 256), help='the output image size')
    parser.add_argument('--val_folder', default='folder0', type=str,
                        help='which cross validation folder')


    args = parser.parse_args()
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))



    args.data = 'aapm_binary'
    # metric_name = 'mseloss'
    # metric_name = 'mse_whole'
    # metric_name = 'simpleNet_l1'
    # metric_name = 'simpleNet_sigmoid'
    metric_name = 'whole'
    # metric_name = 'redcnnsigmoid'

    args.ckpt = os.path.join(args.ckpt, args.data, metric_name)
    print('Models are saved at %s' % (args.ckpt))
    args.save_img_path = os.path.join(args.save_img_path, args.data, metric_name)
    print('images are saved at %s' % (args.save_img_path))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)
    if not os.path.isdir(args.save_img_path):
        os.makedirs(args.save_img_path)


    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str(args.start_epoch) + '_' + args.data + '_checkpoint.pth.tar'

    main(args)
