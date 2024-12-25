import numpy as np
import torch

whole_img_size = 512


def combine_256(res_arr, whole):
    assert len(res_arr) == 9
    whole[:, :, 0:192, 0:192] = res_arr[0][:, :, 0:192, 0:192]
    whole[:, :, 0:192, 192:320] = res_arr[1][:, :, 0:192, 64:192]
    whole[:, :, 0:192, 320:512] = res_arr[2][:, :, 0:192, 64:256]
    whole[:, :, 192:320, 0:192] = res_arr[3][:, :, 64:192, 0:192]
    whole[:, :, 192:320, 192:320] = res_arr[4][:, :, 64:192, 64:192]
    whole[:, :, 192:320, 320:512] = res_arr[5][:, :, 64:192, 64:256]
    whole[:, :, 320:512, 0:192] = res_arr[6][:, :, 64:256, 0:192]
    whole[:, :, 320:512, 192:320] = res_arr[7][:, :, 64:256, 64:192]
    whole[:, :, 320:512, 320:512] = res_arr[8][:, :, 64:256, 64:256]
    return whole


def combine_np(res_arr, patch_size=256):
    whole = np.zeros((1, 1, whole_img_size, whole_img_size), dtype=np.float32)
    if patch_size == 256:
        whole = combine_256(res_arr, whole)
    return whole


def split_np(whole, patch_size=256):
    res_arr = []
    if patch_size == 256:
        crop_point = [(0, 0), (0, 128), (0, 256), (128, 0), (128, 128), (128, 256), (256, 0), (256, 128), (256, 256)]
        for p in crop_point:
            x, y = p
            res_arr.append(whole[:, :, x:x + patch_size, y:y + patch_size])
    return res_arr


def combine_tensor(res_arr, patch_size=256):
    whole = torch.zeros((1,1,whole_img_size, whole_img_size), dtype=torch.float32)
    if patch_size == 256:
        whole = combine_256(res_arr, whole)
    return whole


def split_tensor(whole, patch_size=256):
    res_arr = []
    if patch_size == 256:
        crop_point = [(0, 0), (0, 128), (0, 256), (128, 0), (128, 128), (128, 256), (256, 0), (256, 128), (256, 256)]
        for p in crop_point:
            x, y = p
            res_arr.append(whole[:, :, x:x + patch_size, y:y + patch_size])
    # print('split:shape:', len(res_arr))
    # for i in res_arr:
    #     print(i.shape)
    return res_arr


if __name__ == '__main__':
    whole = torch.randn((1, 1, 512, 512))
    res = split_tensor(whole, 256)

    whole_out = combine_tensor(res, 256)

    print(torch.equal(whole, whole_out))
