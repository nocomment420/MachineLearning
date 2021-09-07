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
import pathlib


def process_image(state):
    # state = 210 x 160 x 3

    # 164 x 160 x 3
    state = state[31:195]

    # 164 x 160
    state = np.mean(state, axis=2)

    # 80 x 80
    state = Image.fromarray(state).resize(size=(84, 84), resample=Image.NEAREST)

    state = np.array(state)
    state = state / 255.0

    return state


def repeat_frame(frame):
    return np.stack([frame] * 4, axis=0)


def frame_to_state(state, next_frame):
    return np.append(state[1:, :, :], np.expand_dims(next_frame, 0), axis=0)

def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


def plot_smoothed_return(n_workers=12):
    for i in range(4):
        if i == 0:
            extention = ""
            name = "1st"
        elif i == 1:
            extention = "-2"
            name = "2nd"
        elif i == 2:
            extention = "-3"
            name = "3rd"
        else:
            extention = "-final"
            name = "final"
        smoothed = None
        for i in range(n_workers):
            filename = "worker_{}-rewards{}.npy".format(i+1, extention)
            reward_log = np.load(filename)
            current = smooth(reward_log)
            if smoothed is None or len(current) > len(smoothed):
                smoothed = current

        plt.plot(smoothed, label=name)

    plt.legend()
    plt.show()
def plot_run(filename):
    reward_log = np.load(filename)
    smoothed = smooth(reward_log)
    plt.plot(smoothed, label="reward")
    plt.legend()
    plt.show()

def run_from_file(episodes=20,agent_name="a3c.pt"):
    env_name = "Breakout-v0"
    env = gym.make(env_name)

    network = torch.load(agent_name)

    for i in range(episodes):

        curr_state = repeat_frame(process_image(env.reset()))
        done = False

        while not done:
            env.render()


            a = network.chose_action([curr_state])


            s_, r, done, info = env.step(a)

            curr_state = frame_to_state(curr_state, process_image(s_))


if __name__ == "__main__":
    # plot_smoothed_return()
    # run_from_file(agent_name="a3c-21000.pt")
    base_path = pathlib.Path(__file__).parent.absolute()
    path = "{}\\run-06-06-21\\a3c-rewards.npy".format(base_path)
    plot_run(path)