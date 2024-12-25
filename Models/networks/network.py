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
from utils.train_logger import write_log


class Comprehensive_Atten_Unet(nn.Module):
    def __init__(self, args, in_ch=3, out_ch=2, feature_scale=4, is_deconv=True, is_batchnorm=False,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(Comprehensive_Atten_Unet, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = out_ch
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = args.out_size

        self.use_scaleAtt = True

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=False)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=False)

        # attention blocks
        # self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
        #                                             inter_channels=filters[0])
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)

        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCat(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up4 = SE_Conv_Block(filters[4], filters[3], drop_out=False)
        self.up3 = SE_Conv_Block(filters[3], filters[2])
        self.up2 = SE_Conv_Block(filters[2], filters[1])
        self.up1 = SE_Conv_Block(filters[1], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=4, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=4, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        # final conv (without any concat)
        if self.use_scaleAtt:
            # self.final = nn.Sequential(nn.Conv2d(4, out_ch, kernel_size=1))
            self.final = nn.Sequential(nn.Conv2d(4, 2, kernel_size=1), nn.Sigmoid())  # use for seg
        else:
            self.final = nn.Sequential(nn.Conv2d(16, out_ch, kernel_size=1))


    def forward(self, inputs):
        # Feature Extraction
        # print('NETWORK::input size:', inputs.size())
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        up4 = self.up_concat4(conv4, center)
        g_conv4 = self.nonlocal4_2(up4)
        # print('NETWORK::g_conv4 shape:', g_conv4.shape)  # [16, 256, 32, 32]

        up4, att_weight4 = self.up4(g_conv4)
        g_conv3, att3 = self.attentionblock3(conv3, up4)

        # atten3_map = att3.cpu().detach().numpy().astype(np.float)
        # atten3_map = ndimage.interpolation.zoom(atten3_map, [1.0, 1.0, 224 / atten3_map.shape[2],
        #                                                      300 / atten3_map.shape[3]], order=0)

        up3 = self.up_concat3(g_conv3, up4)
        up3, att_weight3 = self.up3(up3)
        g_conv2, att2 = self.attentionblock2(conv2, up3)

        # atten2_map = att2.cpu().detach().numpy().astype(np.float)
        # atten2_map = ndimage.interpolation.zoom(atten2_map, [1.0, 1.0, 224 / atten2_map.shape[2],
        #                                                      300 / atten2_map.shape[3]], order=0)

        up2 = self.up_concat2(g_conv2, up3)
        up2, att_weight2 = self.up2(up2)
        # g_conv1, att1 = self.attentionblock1(conv1, up2)

        # atten1_map = att1.cpu().detach().numpy().astype(np.float)
        # atten1_map = ndimage.interpolation.zoom(atten1_map, [1.0, 1.0, 224 / atten1_map.shape[2],
        #                                                      300 / atten1_map.shape[3]], order=0)
        up1 = self.up_concat1(conv1, up2)
        up1, att_weight1 = self.up1(up1)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        if self.use_scaleAtt:
            out = self.scale_att(dsv_cat)
        else:
            out = dsv_cat
        out = self.final(out)
        # print('NETWORK::out size:', out.size())

        return out


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
        self.lay1 = Conv_2d(in_ch=self.input_channels, out_ch=64, use_bn="no")
        self.lay2 = Conv_2d(in_ch=64, out_ch=32, use_bn="no")
        self.lay3 = Conv_2d(in_ch=32, out_ch=32, use_bn="no")
        self.lay4 = Conv_2d(in_ch=32, out_ch=16, use_bn="no")
        self.lay5 = Conv_2d(in_ch=16, out_ch=16, use_bn="no")
        self.lay6 = Conv_2d(in_ch=16, out_ch=32, use_bn="no")
        self.lay7 = Conv_2d(in_ch=32, out_ch=32, use_bn="no")
        self.lay8 = Conv_2d(in_ch=32, out_ch=64, use_bn="no")
        self.lay9 = nn.Conv2d(in_channels=64, out_channels=self.output_channels, kernel_size=(1, 1))

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
class WGAN_CA_AE(nn.Module):
    def __init__(self, args, in_ch=1, out_ch=1, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(WGAN_CA_AE, self).__init__()
        self.root_path = r'/home/cuiyang/bishe/LDCTDenoising_CA'
        # self.generator = normalCNN(in_ch, out_ch).cuda()
        self.generator = Comprehensive_Atten_Unet(args, in_ch=in_ch, out_ch=out_ch, feature_scale=feature_scale,
                                                  is_deconv=is_deconv, is_batchnorm=is_batchnorm,
                                                  nonlocal_mode=nonlocal_mode, attention_dsample=attention_dsample).cuda()
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
        mse_loss = self.p_criterion(x, y)
        g_loss = g_loss + mse_loss * 100
        if perceptual:
            p_loss = self.p_loss(x, y)
            p_loss *= 1e16
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
    print("Please set the path of expected model!")
    time.sleep(3)
    model_reload_path = os.path.join(model_path, 'model', 'wgan_ca_ae_{}.pkl'.format(epoch))
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
