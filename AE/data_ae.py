import numpy as np
import glob
import pydicom
import os
import torch.utils.data as data
import torch

from torch.utils.data import DataLoader
import pickle


def read_dicom_scan(scan_path):
    dicom_files = glob.glob(os.path.join(scan_path, '*.IMA'))
    print('len of dicom files:', len(dicom_files))
    slices = [pydicom.read_file(each_dicom_path) for each_dicom_path in dicom_files]

    slices.sort(key=lambda x: float(x.InstanceNumber))
    if len(slices) == 0:
        print('Scan reading error, please check the scan path')

    return (slices)


def get_pixels_HU(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def get_data(scan_path):
    test_slices = read_dicom_scan(scan_path)
    image = get_pixels_HU(test_slices)

    img_max = np.max(image)
    img_min = np.min(image)
    print('img_max, img_min:', img_max, img_min)

    crop_max = 240.0
    crop_min = -160.0

    # crop_max = 3072.0
    # crop_min = -1024.0

    image = np.clip(image, crop_min, crop_max)
    nor_image = image.astype('float32')
    nor_image = (nor_image - crop_min) / (crop_max - crop_min)
    return nor_image


class DatasetFromFolder(data.Dataset):
    def __init__(self, input_image_array, target_image_array):
        super(DatasetFromFolder, self).__init__()

        self.input_image = input_image_array
        self.target_image = target_image_array

    def __getitem__(self, index):
        input = torch.from_numpy(self.input_image[index])
        target = torch.from_numpy(self.target_image[index])
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

        return input, target

    def __len__(self):
        return len(self.input_image)


def get_training_set(crop_size=64):
    work_dir = "/home/cuiyang/data/zhilin/aapm_data_all/"
    train_dir = [work_dir + "L067/",
                 work_dir + "L096/",
                 work_dir + "L109/",
                 work_dir + "L143/",
                 work_dir + "L192/",
                 work_dir + "L286/"]

    input_image_array = []
    target_image_array = []
    for index in range(0, len(train_dir)):
        ld_files = train_dir[index] + "quarter_3mm/"
        nd_files = train_dir[index] + "full_3mm/"
        ld_image = get_data(ld_files)
        nd_image = get_data(nd_files)

        c, h, w = ld_image.shape
        # print('c,h,w:', c, h, w)
        print('loading data {}'.format(index))
        # depth:3

        # if crop:
        #     for i in range(0, c // 3):
        #         for j in range(0, h // 64):
        #             for k in range(0, w // 64):
        #                 input_image_array.append(ld_image[i*3:i*3+3, j * 64:j * 64 + 64, k * 64:k * 64 + 64])
        #                 target_image_array.append(ld_image[i*3:i*3+3, j * 64:j * 64 + 64, k * 64:k * 64 + 64])
        #                 input_image_array.append(nd_image[i*3:i*3+3, j * 64:j * 64 + 64, k * 64:k * 64 + 64])
        #                 target_image_array.append(nd_image[i*3:i*3+3, j * 64:j * 64 + 64, k * 64:k * 64 + 64])
        # else:
        #     for i in range(0, c//3):
        #         input_image_array.append(ld_image[i * 3:i * 3 + 3])
        #         input_image_array.append(nd_image[i * 3:i * 3 + 3])
        #         target_image_array.append(np.array([],dtype=np.float32))
        #         target_image_array.append(np.array([],dtype=np.float32))

        if crop_size==-1:
            for i in range(0, c):
                input_image_array.append(ld_image[i])
                input_image_array.append(nd_image[i])
                target_image_array.append(np.array([],dtype=np.float32))
                target_image_array.append(np.array([],dtype=np.float32))
        else:
            for i in range(0, c):
                for j in range(0, h // crop_size):
                    for k in range(0, w // crop_size):
                        input_image_array.append(ld_image[i, j * crop_size:j * crop_size + crop_size, k * crop_size:k * crop_size + crop_size])
                        target_image_array.append(ld_image[i, j * crop_size:j * crop_size + crop_size, k * crop_size:k * crop_size + crop_size])
                        input_image_array.append(nd_image[i, j * crop_size:j * crop_size + crop_size, k * crop_size:k * crop_size + crop_size])
                        target_image_array.append(nd_image[i, j * crop_size:j * crop_size + crop_size, k * crop_size:k * crop_size + crop_size])

    print("data process complete. data shape:", len(input_image_array), input_image_array[0].shape)


    return DatasetFromFolder(input_image_array, target_image_array)



def get_validation_set():
    work_dir = "/home/cuiyang/data/aapm/"
    train_dir = [work_dir + "L310/",
                 work_dir + "L333/"]

    input_image_array = []
    target_image_array = []
    for index in range(0, len(train_dir)):
        ld_files = train_dir[index] + "quarter_1mm/"
        nd_files = train_dir[index] + "full_1mm/"
        ld_image = get_data(ld_files)
        nd_image = get_data(nd_files)

        c, h, w = ld_image.shape
        for i in range(0, c // 3):
            for j in range(0, h // 64):
                for k in range(0, w // 64):
                    input_image_array.append(ld_image[i*3:i*3+3, j * 64:j * 64 + 64, k * 64:k * 64 + 64])
                    target_image_array.append(ld_image[i*3:i*3+3, j * 64:j * 64 + 64, k * 64:k * 64 + 64])
                    input_image_array.append(nd_image[i*3:i*3+3, j * 64:j * 64 + 64, k * 64:k * 64 + 64])
                    target_image_array.append(nd_image[i*3:i*3+3, j * 64:j * 64 + 64, k * 64:k * 64 + 64])

    return DatasetFromFolder(input_image_array, target_image_array)


def get_test_set():
    work_dir = "/home/cuiyang/data/aapm/"
    train_dir = [work_dir + "L506/"]

    input_image_array = []
    target_image_array = []
    for index in range(0, len(train_dir)):
        ld_files = train_dir[index] + "quarter_1mm/"
        nd_files = train_dir[index] + "full_1mm/"
        ld_image = get_data(ld_files)
        nd_image = get_data(nd_files)

        c, h, w = ld_image.shape

        for i in range(0, c // 3):
            for j in range(0, h // 64):
                for k in range(0, w // 64):
                    input_image_array.append(ld_image[i*3:i*3+3, j * 64:j * 64 + 64, k * 64:k * 64 + 64])
                    target_image_array.append(ld_image[i*3:i*3+3, j * 64:j * 64 + 64, k * 64:k * 64 + 64])
                    input_image_array.append(nd_image[i*3:i*3+3, j * 64:j * 64 + 64, k * 64:k * 64 + 64])
                    target_image_array.append(nd_image[i*3:i*3+3, j * 64:j * 64 + 64, k * 64:k * 64 + 64])

    return DatasetFromFolder(input_image_array, target_image_array)


def get_test_full_set():
    work_dir = "/home/cuiyang/data/aapm/"
    test_dir = [work_dir + "L506/"]

    input_image_array = []
    target_image_array = []
    for index in range(0, len(test_dir)):
        ld_files = test_dir[index] + "quarter_1mm/"
        nd_files = test_dir[index] + "full_1mm/"
        ld_image = get_data(ld_files)
        nd_image = get_data(nd_files)
        c, h, w = ld_image.shape

        for i in range(0, c//3):
            input_image_array.append(ld_image[i * 3:i * 3 + 3])
            input_image_array.append(nd_image[i * 3:i * 3 + 3])
            target_image_array.append(np.array([],dtype=np.float32))
            target_image_array.append(np.array([],dtype=np.float32))

    return DatasetFromFolder(input_image_array, target_image_array)

if __name__ == '__main__':
    print("===> Loading datasets")

    # train_set = get_training_set()
    # validation_set = get_validation_set()
    # test_set = get_test_set()
    test_full_set = get_test_full_set()
    # training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=8, shuffle=True)
    # validation_data_loader = DataLoader(dataset=validation_set, num_workers=1, batch_size=8, shuffle=True)
    # test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=test_full_set, num_workers=1, batch_size=1, shuffle=False)

    print('===> Saving datasets')
    # pickle.dump(training_data_loader, open("training_data.p", "wb"))
    # pickle.dump(validation_data_loader, open("validation_data.p", "wb"))
    # pickle.dump(test_loader, open("test_data.p", "wb"))
    pickle.dump(test_loader, open("test_data_full.p", "wb"))
    print('save done.')