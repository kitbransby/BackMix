import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def plot(train_loss_all, val_loss_all, train_acc_all, val_acc_all, folder):
    epochs = len(train_loss_all)
    f, axes = plt.subplots(1,2, figsize=(20,10))
    axes[0].plot(list(range(epochs)), train_loss_all, label='train_loss')
    axes[0].plot(list(range(epochs)), val_loss_all, label='val_loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_ylim(0, 1)
    axes[1].plot(list(range(epochs)), train_acc_all, label='train_acc')
    axes[1].plot(list(range(epochs)), val_acc_all, label='val_acc')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.5, 1)
    plt.savefig(folder + '/progress.png', dpi=200)
    plt.close('all')