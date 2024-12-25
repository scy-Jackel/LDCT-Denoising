import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from Models.layers.modules import conv_block, UpCat, UpCatconv, UnetDsv3, UnetGridGatingSignal3
from Models.layers.grid_attention_layer import GridAttentionBlock2D, MultiAttentionBlock
from Models.layers.channel_attention_layer import SE_Conv_Block
from Models.layers.scale_attention_layer import scale_atten_convblock
from Models.layers.nonlocal_layer import NONLocalBlock2D
from Models.layers.wgan_util import Conv_2d, AE_Down_2d, AE_Up_2d, OutConv_2d, update_ae
from Models.networks.network import Comprehensive_Atten_Unet
from utils.train_logger import write_log



## USE WGAN-GP

"""
Discriminator
"""


##******************************************************************************************************************************
class DISC(nn.Module):
    def __init__(self, in_size):
        super(DISC, self).__init__()
        self.in_h, self.in_w = in_size
        self.lay1 = Conv_2d(in_ch=1, out_ch=16, use_bn="no")
        self.lay2 = Conv_2d(in_ch=16, out_ch=32, use_bn="no")
        self.lay3 = Conv_2d(in_ch=32, out_ch=64, use_bn="no")
        self.lay4 = Conv_2d(in_ch=64, out_ch=64, use_bn="no")
        self.lay5 = Conv_2d(in_ch=64, out_ch=32, use_bn="no")
        self.lay6 = Conv_2d(in_ch=32, out_ch=16, use_bn="no")
        self.lay7 = Conv_2d(in_ch=16, out_ch=1, use_bn="no")

        self.fc1 = nn.Linear(self.in_h * self.in_w, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        x = self.lay6(x)
        x = self.lay7(x)

        x = self.fc1(x.view(-1, self.in_h * self.in_w))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


"""
a normal Generator
"""
##******************************************************************************************************************************
class normalCNN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(normalCNN, self).__init__()
        self.input_channels = in_ch
        self.output_channels = out_ch
        # kernel_size = [64, 32, 32, 16, 16, 32, 32, 64]
        # kernel_size = [64, 64, 64, 64, 64, 64, 64, 64]
        kernel_size = [32, 32, 32, 32, 32, 32, 32, 32]
        self.lay1 = Conv_2d(in_ch=self.input_channels, out_ch=kernel_size[0], use_bn="no")
        self.lay2 = Conv_2d(in_ch=kernel_size[0], out_ch=kernel_size[1], use_bn="no")
        self.lay3 = Conv_2d(in_ch=kernel_size[1], out_ch=kernel_size[2], use_bn="no")
        self.lay4 = Conv_2d(in_ch=kernel_size[2], out_ch=kernel_size[3], use_bn="no")
        self.lay5 = Conv_2d(in_ch=kernel_size[3], out_ch=kernel_size[4], use_bn="no")
        self.lay6 = Conv_2d(in_ch=kernel_size[4], out_ch=kernel_size[5], use_bn="no")
        self.lay7 = Conv_2d(in_ch=kernel_size[5], out_ch=kernel_size[6], use_bn="no")
        self.lay8 = Conv_2d(in_ch=kernel_size[6], out_ch=kernel_size[7], use_bn="no")
        self.lay9 = nn.Conv2d(in_channels=kernel_size[7], out_channels=self.output_channels, kernel_size=(1, 1))

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


"""
Perceptual loss
"""


##******************************************************************************************************************************
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.lay1 = AE_Down_2d(in_channels=1, out_channels=64)
        self.lay2 = AE_Down_2d(in_channels=64, out_channels=128)
        self.lay3 = AE_Down_2d(in_channels=128, out_channels=256)
        self.lay4 = AE_Down_2d(in_channels=256, out_channels=256)

        self.lay5 = AE_Up_2d(in_channels=256, out_channels=256)
        self.lay6 = AE_Up_2d(in_channels=256, out_channels=128)
        self.lay7 = AE_Up_2d(in_channels=128, out_channels=64)
        self.lay8 = AE_Up_2d(in_channels=64, out_channels=32)
        self.lay9 = OutConv_2d(in_ch=32, out_ch=1)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        x = self.lay1(x)
        x = self.maxpool(x)
        x = self.lay2(x)
        x = self.maxpool(x)
        x = self.lay3(x)
        y = self.lay4(x)

        x = self.lay5(y)
        x = self.lay6(x)
        x = self.deconv1(x)
        x = self.lay7(x)
        x = self.deconv2(x)
        x = self.lay8(x)
        out = self.lay9(x)
        return out, y


"""
Whole Network
"""


##******************************************************************************************************************************
class WGAN_PURE(nn.Module):
    def __init__(self, args, in_ch=1, out_ch=1, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(WGAN_PURE, self).__init__()
        self.root_path = r'/home/cuiyang/bishe/LDCTDenoising_MA'

        self.generator = normalCNN(in_ch, out_ch).cuda()
        self.discriminator = DISC(args.out_size).cuda()
        self.p_criterion = nn.MSELoss().cuda()

        ae_path = self.root_path + "/AE/checkpoint/mse/model_epoch_20.pth"  ## The network has been trained to compute perceputal loss
        Ae = AE()
        self.ae = update_ae(Ae, ae_path)
        # self.lambda_gp = 240+160


    def feature_extractor(self, image, model):
        model.eval()
        pred, y = model(image)
        return y

    def d_loss(self, x, y, gp=True, return_gp=False):
        """
        discriminator loss
        """
        fake = self.generator(x)
        d_real = self.discriminator(y)
        d_fake = self.discriminator(fake)

        d_loss = torch.mean(d_fake) - torch.mean(d_real)

        if d_loss.detach().cpu().numpy() == 0:
            write_log('d_loss=0 here.', place='all')
            # print('d_loss=0 here.')

        if gp:
            gp_loss = self.gp(y, fake)
            # gp_loss = self.gp(y.detach(), fake.detach())
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss

    def g_loss(self, x, y, perceptual=True, return_p=False):
        """
        generator loss
        """
        fake = self.generator(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        # mse_loss = self.p_criterion(x, y)
        # g_loss = g_loss + mse_loss * 100
        if perceptual:
            p_loss = self.p_loss(x, y)
            # p_loss = p_loss * 1e16
            loss = g_loss + (0.1 * p_loss)
        else:
            # p_loss = None
            p_loss = 0.0
            loss = g_loss
        return (loss, p_loss) if return_p else loss

    def p_loss(self, x, y):
        """
        percetual loss
        """
        fake = self.generator(x)
        real = y
        fake_feature = self.feature_extractor(fake, self.ae)
        real_feature = self.feature_extractor(real, self.ae)
        # print("fake_feature shape:", fake_feature.shape)  # (4, 256, 64, 64)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10):
        """
        gradient penalty
        """
        assert y.size() == fake.size()

        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        if torch.cuda.is_available():
            a = a.cuda()

        interp = (a * y + ((1 - a) * fake)).requires_grad_(True)
        d_interp = self.discriminator(interp)

        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
        return gradient_penalty




def model_update(model, model_reload_path):
    if os.path.isfile(model_reload_path):
        print("Path is right, Loading...")
        checkpoint = torch.load(model_reload_path)
        model_dict = model.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        del checkpoint
        torch.cuda.empty_cache()

        print("{} Load Done!\n".format(model_reload_path))
        return model
    else:
        print("\nLoading Fail!\n")
        sys.exit(0)


def update_model(model, optimizer_g, optimizer_d, model_path, epoch):
    time.sleep(3)
    model_reload_path = os.path.join(model_path, 'model', 'wgan_pure_{}.pkl'.format(epoch))
    optimizer_g_reload_path = os.path.join(model_path, 'optimizerG', 'optG_{}.pkl'.format(epoch))
    optimizer_d_reload_path = os.path.join(model_path, 'optimizerD', 'optD_{}.pkl'.format(epoch))

    if os.path.isfile(model_reload_path):
        print("Loading previously trained network...")
        print("Load model:{}".format(model_reload_path))
        checkpoint = torch.load(model_reload_path)
        model_dict = model.state_dict()
        checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        del checkpoint
        torch.cuda.empty_cache()

        # model = model.cuda()
        print("Done Reload!")
    else:
        print("Can not reload model....\n")
        time.sleep(10)
        sys.exit(0)

    if os.path.isfile(optimizer_g_reload_path):
        print("Loading previous optimizer...")
        print("Load optimizer:{}".format(optimizer_g_reload_path))
        checkpoint = torch.load(optimizer_g_reload_path)
        optimizer_g.load_state_dict(checkpoint)
        del checkpoint
        torch.cuda.empty_cache()
        print("Done Reload!")
    else:
        print("Can not reload optimizer_g....\n")
        time.sleep(10)
        sys.exit(0)

    if os.path.isfile(optimizer_d_reload_path):
        print("Loading previous optimizer...")
        print("Load optimizer:{}".format(optimizer_d_reload_path))
        checkpoint = torch.load(optimizer_d_reload_path)
        optimizer_d.load_state_dict(checkpoint)
        del checkpoint
        torch.cuda.empty_cache()
        print("Done Reload!")
    else:
        print("Can not reload optimizer_d....\n")
        time.sleep(10)
        sys.exit(0)

    return model, optimizer_g, optimizer_d
