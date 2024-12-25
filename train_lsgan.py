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
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from Datasets.aapm import BuildDataSet_dicom
from Models.networks.maNet import WGAN_MA_AE, update_model

from utils.save_img import save_fig
from utils.train_logger import write_log

warnings.filterwarnings("ignore")

Test_Model = {'WGAN_MA_AE': WGAN_MA_AE}

## train function
##***********************************************************************************************************
def train(model, epoch, optimizer_g, optimizer_d, dataloaders, args):
    data_length = 5000
    g_loss_all = 0
    p_loss_all = 0
    d_loss_all = 0
    dr_loss_all=0
    df_loss_all=0

    generator = model.generator
    attblock = model.attentionBlock
    disc = model.discriminator

    for p in attblock.parameters():
        p.requires_grad = False
    for i,batch in enumerate(dataloaders):
        time_batch_start = time.time()
        full_image = batch["full_image"]
        quarter_image = batch["quarter_image"]

        full_image = Variable(full_image).cuda()
        quarter_image = Variable(quarter_image).cuda()

        mask = attblock(quarter_image)
        mask = torch.index_select(mask,1,torch.tensor([0]).cuda())

        f_imgs = generator(quarter_image,mask)
        # train D
        r_logit = disc(full_image)
        f_logit = disc(f_imgs.detach())
        drloss = torch.mean((r_logit-1)**2)
        dfloss = torch.mean(f_logit**2)
        d_loss = drloss+dfloss
        disc.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        d_loss_all += (d_loss.data.cpu().numpy()*quarter_image.size(0))
        dr_loss_all += (drloss.data.cpu().numpy()*quarter_image.size(0))
        df_loss_all += (dfloss.data.cpu().numpy()*quarter_image.size(0))

        #train G
        f_logit = disc(f_imgs)
        g_loss = torch.mean((f_logit-1)**2)
        disc.zero_grad()
        generator.zero_grad()
        g_loss.backward()
        optimizer_g.step()
        g_loss_all += g_loss.data.cpu().numpy() * quarter_image.size(0)

        if i>0 and math.fmod(i, 50) == 0:
            print("Epoch {} Batch {}-{} {}, Time:{:.4f}s".format(epoch+1,
                                                                 i-50, i, len(dataloaders), (time.time()-time_batch_start)*50))
    g_loss = g_loss_all/data_length
    p_loss = p_loss_all/data_length
    d_loss = d_loss_all/data_length
    print("g_loss:{} d_loss:{} p_loss:{}".format(g_loss, d_loss, p_loss))
    write_log(
        'EPOCH {} ,g_loss:{} d_loss:{} p_loss:{}'.format(epoch, g_loss, d_loss, p_loss))
    return g_loss, p_loss, d_loss








## Train
##***********************************************************************************************************
def train_model(model,dataloaders,args):
    re_load = False
    if args.start_epoch!=0:
        re_load = True
    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr_rate, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr_rate, betas=(0.5, 0.9))

    # optimizer_g = optim.RMSprop(model.generator.parameters(), lr=args.lr_rate, alpha=0.9)
    # optimizer_d = optim.RMSprop(model.discriminator.parameters(), lr=args.lr_rate, alpha=0.9)

    # scheduler_g = StepLR(optimizer_g, step_size=1, gamma=0.3)
    # scheduler_d = StepLR(optimizer_d, step_size=1, gamma=0.3)
    # stepArr = np.zeros(args.epoch_num, dtype=np.int32)
    # stepArr[50] = 1
    # stepArr[70] = 1


    if re_load is False:
        print("\nInit Start**")
        model.generator.apply(weights_init)  # ADD 20210511
        model.discriminator.apply(weights_init)
        print("******Init End******\n")
    else:
        print("Re_load is True !")
        model, optimizer_g, optimizer_d = update_model(model, optimizer_g, optimizer_d, args.ckpt, args.start_epoch)
        # optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr_rate, betas=(0.5, 0.9))
        # optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr_rate, betas=(0.5, 0.9))

    losses = torch.zeros(args.epoch_num, 4)
    temp = 0
    ##********************************************************************************************************************
    time_all_start = time.time()
    # for epoch in range(args.old_index if args.re_load else 0, args.epoch_num):
    for epoch in range(args.start_epoch, args.epoch_num):
        time_epoch_start = time.time()
        print("-" * 60)
        print(".........Training and Val epoch {}, all {} epochs..........".format(epoch+1, args.epoch_num))
        print('g lr:',optimizer_g.state_dict()['param_groups'][0]['lr'])
        print('d lr:',optimizer_d.state_dict()['param_groups'][0]['lr'])
        print("-" * 60)

        model.train()
        g_loss,p_loss,d_loss = train(model, epoch, optimizer_g, optimizer_d, dataloaders, args)
        losses[epoch] = torch.tensor([g_loss,p_loss,d_loss])
        # if stepArr[epoch] == 1:
        #     scheduler_g.step()
        #     scheduler_d.step()
        if math.fmod(epoch, 5) == 0:
            torch.save(model.state_dict(), args.ckpt + "/model/" + "lsgan_{}.pkl".format(epoch))
            torch.save(optimizer_g.state_dict(), args.ckpt + "/optimizerG/" + "optG_{}.pkl".format(epoch))
            torch.save(optimizer_d.state_dict(), args.ckpt + "/optimizerD/" + "optD_{}.pkl".format(epoch))

        print("Time for epoch {} : {:.4f}min".format(epoch+1, (time.time()-time_epoch_start)/60))
        print("Time for ALL : {:.4f}h\n".format((time.time()-time_all_start)/3600))
    print("\nTrain Completed!! Time for ALL : {:.4f}h".format((time.time()-time_all_start)/3600))


## Init the model
##***********************************************************************************************************
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
        print("Init {} Parameters.................".format(classname))
    if classname.find("Linear") != -1:
        init.xavier_normal(m.weight)
        print("Init {} Parameters.................".format(classname))
    else:
        print("{} Parameters Do Not Need Init !!".format(classname))


def main(args):
    # loading the dataset
    train_folder = ["L192", "L286", "L310", "L333", "L506", "L143"]
    test_folder = ["L067"]
    val_folder = ["L067", "L096", "L109"]
    data_length = {'train':5000, 'val':2000, 'test':40}
    patch_size = 256
    args.num_input = 1
    args.num_classes = 1
    args.out_size = (patch_size, patch_size)
    train_datasets = BuildDataSet_dicom(args.data_root_path, train_folder, None, data_length['train'], 'train', patch_size, in_channel=args.num_input, out_channel=args.num_classes)
    trainloader = Data.DataLoader(dataset=train_datasets, batch_size=args.batch_size, num_workers=6, shuffle=True, pin_memory=True)
    print('Loading is done\n')

    model = Test_Model[args.id](args, args.num_input, args.num_classes)
    # if args.resume:
    #     model = model_updata(model, model_old_name=args.model_name + "{}".format(model_index),
    #                          model_old_path=args.model_path)
    model = model.cuda()
    write_log('start train.')
    train_model(model, trainloader, args)





if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='WGAN_MA_AE',
                        help='a name for identitying the model. Choose from the following options: Unet')

    # Path related arguments
    parser.add_argument('--data_root_path', default='../aapm_data',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output checkpoints')
    parser.add_argument('--save_img_path', default='./result',
                        help='folder to output checkpoints')

    # optimization related arguments
    parser.add_argument('--epoch_num', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 12)')
    parser.add_argument('--lr_rate', type=float, default=1e-5, metavar='LR',
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


    args.ckpt = os.path.join(args.ckpt, 'lsgan')
    print('Models are saved at %s' % (args.ckpt))
    args.save_img_path = os.path.join(args.save_img_path, 'lsgan')
    print('images are saved at %s' % (args.save_img_path))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt+'/model/')
        os.makedirs(args.ckpt+'/optimizerG/')
        os.makedirs(args.ckpt+'/optimizerD/')
    if not os.path.isdir(args.save_img_path):
        os.makedirs(args.save_img_path)


    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str(args.start_epoch) + '_' + args.data + '_checkpoint.pth.tar'

    main(args)

    #pcl: cd /userhome/cuiyang/LDCTDenoising/LDCTDenoising_MA/;python train_wgan_ma.py --batch_size 16



