import  torch
from    matplotlib import pyplot as plt
from constants import *

def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue', linewidth=0.3)
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.savefig(root_dir + "train_loss")



def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(root_dir + name)


def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

def one_hot_num(label, depth=10):
    out = torch.zeros(1, depth)
    out[0, label] = 1
    return out