import gym
import numpy as np
import random
import torch
from sklearn.utils import shuffle
from PIL import Image
import torch.nn as nn
import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt


def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


def plot_smoothed_return(n_workers=12):
    for i in range(n_workers):
        filename = "worker_{}-rewards.npy".format(i+1)
        reward_log = np.load(filename)
        smoothed = smooth(reward_log)
        plt.plot(smoothed, label="worker-{}".format(i+1))

    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_smoothed_return()