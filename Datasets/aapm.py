# 重新写一个dicom版的dataset, dataloader. 用于训练测试
import torch
# import astra
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from utils.dicom_trans import RandomCrop, ToTensor, Normalize, Scale2Gen
from utils.dicom_trans import get_data


## Basic datasets
##***********************************************************************************************************
class BasicData_dicom(Dataset):
    def __init__(self, data_root_path, folder, data_length, Dataset_name, in_channel=1, out_channel=1):
        self.folder = folder
        self.Dataset_name = Dataset_name
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.data_length = data_length  ## The data size of each epoch
        self.Full_Image = {x: get_data(data_root_path + "/{}/full_3mm".format(x)) for x in self.folder}  ## High-dose images of all patients
        self.Quarter_Image = {x: get_data(data_root_path + "/{}/quarter_3mm".format(x)) for x in self.folder}  ## Low-dose images of all patients

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        # 初始化中Full_image是一个病人的整张CT图像，在这里随机选取一个image_index作为z轴方向坐标，然后选择三张图像作为返回的full_image 和 quarter_image
        patient_index = np.random.randint(len(self.folder))  ## One image of the patient was chosen at a time
        full_image_set = self.Full_Image[self.folder[patient_index]]
        quarter_image_set = self.Quarter_Image[self.folder[patient_index]]

        image_index = np.random.randint(1, full_image_set.shape[0] - 1)  ## Three consecutive images were randomly selected from people with the disease

        if self.in_channel == 3:
            quarter_image = quarter_image_set[image_index - 1:image_index + 2]
        else:
            quarter_image = quarter_image_set[image_index]
            quarter_image = np.expand_dims(quarter_image, 0)

        if self.out_channel == 3:
            full_image = quarter_image_set[image_index - 1:image_index + 2]
        else:
            full_image = full_image_set[image_index]
            full_image = np.expand_dims(full_image, 0)

        # print('aapm::shape:', full_image.shape, quarter_image.shape)  # (3, 512, 512) (1, 512, 512)
        return full_image, quarter_image


# data_length = {"train":5000, "val":500, "test":200}
# pre_trans_img = [Transpose(), TensorFlip(0), TensorFlip(1)]
#
class BuildDataSet_dicom(Dataset):
    def __init__(self, data_root_path, folder, pre_trans_img=None, data_length=None, Dataset_name="train",
                 patch_size=64, in_channel=1, out_channel=1):
        self.Dataset_name = Dataset_name
        self.pre_trans_img = pre_trans_img

        self.imgset = BasicData_dicom(data_root_path, folder, data_length, Dataset_name=self.Dataset_name, in_channel=in_channel, out_channel=out_channel)
        self.patch_size = patch_size
        self.data_dim = 4
        if in_channel==1 and out_channel==1:
            self.data_dim = 3


    def __len__(self):
        return len(self.imgset)

    @classmethod
    def Cal_transform(cls, Dataset_name, pre_trans_img, fix_list):
        random_list = []
        if Dataset_name is "train":
            if pre_trans_img is not None:
                keys = np.random.randint(2, size=len(pre_trans_img))
                for i, key in enumerate(keys):
                    random_list.append(pre_trans_img[i]) if key == 1 else None
        transform = transforms.Compose(fix_list + random_list)
        return transform

    @classmethod
    def preProcess(cls, image, Transf, patch_size):
        new_image = torch.zeros((image.shape[0], patch_size, patch_size))
        for i in range(image.shape[0]):
            new_image[i] = Transf(image[i])
        return new_image

    def __getitem__(self, idx):
        full_image, quarter_image = self.imgset[idx]

        if self.patch_size == 512:
            crop_point = [0, 0]
        else:
            crop_point = np.random.randint(192, size=2)
            crop_point += 64


        # fix_list = [Scale2Gen(scale_type="image"), Normalize(normalize_type="image"),RandomCrop(self.patch_size, crop_point), ToTensor()]
        if self.Dataset_name == 'test':
            fix_list = [RandomCrop(self.patch_size, crop_point), ToTensor()]
        else:
            fix_list = [RandomCrop(self.patch_size, crop_point), ToTensor()]

        transf = self.Cal_transform(self.Dataset_name, self.pre_trans_img, fix_list)

        full_image = self.preProcess(full_image, transf, patch_size=self.patch_size)
        quarter_image = self.preProcess(quarter_image, transf, patch_size=self.patch_size)

        if self.data_dim==4:
            sample = {"full_image": full_image.unsqueeze_(0),
                      "quarter_image": quarter_image.unsqueeze_(0)}
        else:
            sample = {"full_image": full_image,
                      "quarter_image": quarter_image}
        # print('dataloader data shape:', full_image.shape, quarter_image.shape)  # [1, 3, 64, 64]
        return sample

