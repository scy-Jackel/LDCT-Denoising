import os
import numpy as np
import matplotlib.pyplot as plt


def denormAndTrunc(image, norm_min=-1024.0, norm_max=3072, trunc_min=-240.0, trunc_max=160.0):
    image = image * (norm_max - norm_min) + norm_min
    image[image <= trunc_min] = trunc_min
    image[image >= trunc_max] = trunc_max
    image = (image - trunc_min) / (trunc_max - trunc_min)
    image = image * 255
    return image


def save_fig(x, y, pred, fig_name, original_result, pred_result, save_path):
    # x, y, pred = x.numpy(), y.numpy(), pred.numpy()
    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(x, cmap=plt.cm.gray)
    ax[0].set_title('Quarter-dose', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                       original_result[1],
                                                                       original_result[2]), fontsize=20)
    ax[1].imshow(pred, cmap=plt.cm.gray)
    ax[1].set_title('Result', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                       pred_result[1],
                                                                       pred_result[2]), fontsize=20)
    ax[2].imshow(y, cmap=plt.cm.gray)
    ax[2].set_title('Full-dose', fontsize=30)

    f.savefig(os.path.join(save_path,'result_{}.png'.format(fig_name)))
    plt.close()


def save_fig4(x, y, pred, resi,  fig_name, original_result, pred_result, save_path):
    # x, y, pred = x.numpy(), y.numpy(), pred.numpy()
    f, ax = plt.subplots(1, 4, figsize=(40, 10))
    ax[0].imshow(x, cmap=plt.cm.gray)
    ax[0].set_title('Quarter-dose', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                       original_result[1],
                                                                       original_result[2]), fontsize=20)
    ax[1].imshow(pred, cmap=plt.cm.gray)
    ax[1].set_title('Result', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                       pred_result[1],
                                                                       pred_result[2]), fontsize=20)
    ax[2].imshow(y, cmap=plt.cm.gray)
    ax[2].set_title('Full-dose', fontsize=30)

    ax[3].imshow(resi, cmap=plt.cm.gray)
    ax[3].set_title('Residual', fontsize=30)
    ax[3].set_xlabel("MEAN: {}".format(np.mean(resi)), fontsize=20)
    f.savefig(os.path.join(save_path,'result_{}.png'.format(fig_name)))
    plt.close()

#
# def save_fig(x, y, fig_name, loss, save_path):
#     import matplotlib.pyplot as plt
#     x, y = x.numpy(), y.numpy()
#     # print("ipnut shape:", x.shape, "result shape:", y.shape)
#     x=np.squeeze(x)
#     y=np.squeeze(y)
#     # print("ipnut shape:", x.shape, "result shape:", y.shape)
#
#     x = trunc(Denorm(x))
#     y = trunc(Denorm(y))
#     f, ax = plt.subplots(1, 2, figsize=(30, 10))
#     ax[0].imshow(x, cmap=plt.cm.gray, vmin=-160.0, vmax=240.0)
#     ax[0].set_title('input', fontsize=30)
#
#     ax[1].imshow(y, cmap=plt.cm.gray, vmin=-160.0, vmax=240.0)
#     ax[1].set_title('output', fontsize=30)
#     ax[1].set_xlabel("mse loss: {:.8f}\n".format(loss), fontsize=20)
#
#     f.savefig(os.path.join(save_path, 'output','result_{}.png'.format(fig_name)))
#     plt.close()