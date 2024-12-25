import sys

sys.path.append('../')

import argparse, os
import torch
import random
import pickle
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import AE
# from dataset import DatasetFromHdf5
# from readdata import get_training_set
# from readdata import get_validation_set
# from readdata import get_test_set
from data_ae import *  # pickle.load()

# Training settings
parser = argparse.ArgumentParser(description="sacnn")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=10, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.00001, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=1, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", type=bool, default=True, help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = True
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # opt.seed = random.randint(1, 10000)
    # print("Random Seed: ", opt.seed)
    # torch.manual_seed(opt.seed)
    # if cuda:
    #     torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    # training_data_loader = pickle.load(open("training_data.p", "rb"))
    # validation_data_loader = pickle.load(open("validation_data.p","rb"))
    # test_data_loader = pickle.load(open("test_data_full.p", "rb"))

    # train_set = get_training_set()
    # validation_set = get_validation_set()
    test_set = get_test_full_set()

    # training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    # validation_data_loader = DataLoader(dataset=validation_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    test_data_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    # print('===> Saving datasets')
    # pickle.dump(training_data_loader,open("/data/CT/training_data.p", "wb"))
    # pickle.dump(validation_data_loader,open("/data/CT/validation_data.p", "wb"))

    print("===> Building model")
    model = AE()

    criterion = nn.MSELoss()

  

    # optionally resume from a checkpoint
    # model_path = './checkpoint/model_epoch_1.pth'  # average loss 0.0008576809
    model_path = './checkpoint/model_epoch_2.pth'  # average loss 0.0001001800

    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    # opt.start_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model"].state_dict())

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    print("===> Testing")

    # test(test_data_loader,criterion, model)
    test_and_savefigs(test_data_loader, criterion, model)


def test(test_data_loader, criterion, model):
    overall_loss = 0.0
    model.eval()

    for iteration, batch in enumerate(test_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        input = input.cuda()
        target = target.cuda()

        result, inter = model(input)
        # print('result shape:', result.shape, 'target shape:', target.shape)

        loss = criterion(result, target)

        print("===> {}/{} Loss: {:.10f}".format(iteration, len(test_data_loader), loss.item()))
        overall_loss += loss.item()
    print("Overall average loss: {:.10f}".format(overall_loss / len(test_data_loader)))


def test_and_savefigs(test_data_loader, criterion, model):
    overall_loss = 0.0
    model.eval()
    save_path = './figs'
    with torch.no_grad():
        for iteration, batch in enumerate(test_data_loader, 1):
            input, _ = Variable(batch[0]), Variable(batch[1], requires_grad=False)
            input = input.cuda()
            target = input.cuda()

            result, inter = model(input)

            # print('result shape:', result.shape, 'target shape:', target.shape)

            loss = criterion(result, target)
            result = result.detach()
            print("===> {}/{} Loss: {:.10f}".format(iteration, len(test_data_loader), loss.item()))
            overall_loss += loss.item()
            # print('input:', input.shape)
            # print('target:', target.shape)
            input = input.cpu().squeeze()
            result = result.cpu().squeeze()
            save_fig(input[1], result[1], iteration, loss.item(), save_path)
            del input
            del result
            del target
    print("Overall average loss: {:.10f}".format(overall_loss / len(test_data_loader)))


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
    ax[1].set_xlabel("mse loss: {:.8f}\n".format(loss), fontsize=20)

    f.savefig(os.path.join(save_path, 'output','result_{}.png'.format(fig_name)))
    plt.close()


if __name__ == "__main__":
    main()
