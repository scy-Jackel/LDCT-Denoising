import os

import torch
from torch.utils.data import Dataset

from utils.dicom_trans import get_data


class DicomData(Dataset):
    def __init__(self, data_root_path, folder, depth=3, size=512):
        self.folder_path = []
        self.FullImage = []
        self.QuarterImage = []
        for f in folder:
            self.folder_path.append(os.path.join(data_root_path, f))
        for f in self.folder_path:
            self.FullImage.append(get_data("{}/full_3mm".format(f)))
            self.QuarterImage.append(get_data("{}/quarter_3mm".format(f)))
        self.data_arr = []
        for p in range(len(self.folder_path)):
            fullImg = self.FullImage[p]
            quaterImg = self.QuarterImage[p]
            d, w, h = fullImg.shape
            print("CT IMAGE SHAPE:", d, w, h)

            assert d * w * h > 0, 'img shape error. {}'.format((d, w, h))
            for i in range(0, d - depth + 1):
                for j in range(0, h // size):
                    for k in range(0, w // size):
                        full_patch = fullImg[i:i + depth, j * size:j * size + size, k * size:k * size + size]
                        quater_patch = quaterImg[i:i + depth, j * size:j * size + size, k * size:k * size + size]
                        full_patch = torch.from_numpy(full_patch).type(torch.FloatTensor)
                        quater_patch = torch.from_numpy(quater_patch).type(torch.FloatTensor)

                        sample = {"full_image": full_patch,
                                  "quarter_image": quater_patch}

                        # sample = {"full_image": full_patch.unsqueeze_(0),
                        #           "quarter_image": quater_patch.unsqueeze_(0)}
                        self.data_arr.append(sample)

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, idx):
        return self.data_arr[idx]


class DicomData_test(Dataset):
    def __init__(self, data_root_path, folder, patch_size=64):
        self.folder_path = []
        self.FullImage = []
        self.QuarterImage = []
        for f in folder:
            self.folder_path.append(os.path.join(data_root_path, f))
        for f in self.folder_path:
            self.FullImage.append(get_data("{}/full_3mm".format(f)))
            self.QuarterImage.append(get_data("{}/quarter_3mm".format(f)))
        self.data_arr = []
        for p in range(len(self.folder_path)):
            fullImg = self.FullImage[p]
            quaterImg = self.QuarterImage[p]
            d, w, h = fullImg.shape
            print("CT IMAGE SHAPE:", d, w, h)

            assert d * w * h > 0, 'img shape error. {}'.format((d, w, h))
            for i in range(0, d - 2):
                for j in range(0, 15):
                    for k in range(0, 15):
                        full_patch = fullImg[i:i + 3, j * 32:j * 32 + 64, k * 32:k * 32 + 64]
                        quater_patch = quaterImg[i:i + 3, j * 32:j * 32 + 64, k * 32:k * 32 + 64]
                        full_patch = torch.from_numpy(full_patch).type(torch.FloatTensor)
                        quater_patch = torch.from_numpy(quater_patch).type(torch.FloatTensor)
                        sample = {"full_image": full_patch,
                                  "quarter_image": quater_patch}
                        self.data_arr.append(sample)

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, idx):
        return self.data_arr[idx]
