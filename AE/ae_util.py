import os
import numpy as np


def Denorm(x, norm_min=-1024.0, norm_max=3072.0):
    x = x * (norm_max - norm_min) + norm_min
    return x


def trunc(x, trunc_min=-160.0, trunc_max=240.0):
    x[x <= trunc_min] = trunc_min
    x[x >= trunc_max] = trunc_max
    return x


def save_fig(x, y, fig_name, loss, save_path):
    import matplotlib.pyplot as plt
    x, y = x.numpy(), y.numpy()
    # print("ipnut shape:", x.shape, "result shape:", y.shape)
    x=np.squeeze(x)
    y=np.squeeze(y)
    # print("ipnut shape:", x.shape, "result shape:", y.shape)

    x = trunc(Denorm(x))
    y = trunc(Denorm(y))
    f, ax = plt.subplots(1, 2, figsize=(30, 10))
    ax[0].imshow(x, cmap=plt.cm.gray, vmin=-160.0, vmax=240.0)
    ax[0].set_title('input', fontsize=30)

    ax[1].imshow(y, cmap=plt.cm.gray, vmin=-160.0, vmax=240.0)
    ax[1].set_title('output', fontsize=30)
    ax[1].set_xlabel("mse loss: {:.4f}\n".format(loss), fontsize=20)

    f.savefig(os.path.join(save_path, 'output','result_{}.png'.format(fig_name)))
    plt.close()
