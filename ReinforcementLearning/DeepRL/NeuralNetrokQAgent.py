from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import random
import matplotlib
import gym
from gym import wrappers
import os
import numpy as np
import torch
import datetime


class DeepNerualNetwork():
    def __init__(self, dims, activations, lr, dropouts=None):
        self.dims = dims

        self.activations = activations
        self.layers = []
        self.params = []
        self.dropouts = []

        if dropouts is None:
            dropouts = [None for _ in range(len(activations))]
        else:
            dropouts.append(None)   # no dropout on final layer
        self.dropouts = dropouts

        for (dim_in, dim_out) in zip(dims[:-1],dims[1:]):
            layer = torch.nn.Linear(dim_in, dim_out)
            self.params += layer.parameters()
            self.layers.append(layer)
        assert len(self.activations) == len(self.layers) == len(self.dropouts)

        self.criterion = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.params, lr=lr)

    def forwards(self, X, train=False):
        h = torch.tensor(X, dtype=torch.float32)
        for (layer, activation, d_p) in zip(self.layers, self.activations, self.dropouts):
            h = activation(layer(h))
            if train and d_p is not None:
                dropout = torch.nn.Dropout(d_p)
                h = dropout(h)
            elif not train and d_p is not None:
                h = h * d_p
        return h

    def predict(self, X):
        return self.forwards(X).detach().numpy()

    def partial_fit(self, X, Y):
        self.optim.zero_grad()

        prediction = self.forwards(X, train=True)
        loss = torch.sum((torch.tensor(Y, dtype=torch.float32) - prediction) ** 2)

        loss.backward()

        self.optim.step()


class NeuralNetworkNetworkAgent:
    def __init__(self, env, actions, models, epsilon_update, gamma=0.99, e=0.0):
        self.env = env
        self.epsilon_update = epsilon_update
        self.actions = actions
        self.gamma = gamma
        self.e = e
        self.models = models

    def max_Q(self, s, return_index=False):
        predictions = [self.models[a].predict(s) for a in self.actions]
        if return_index:
            return np.argmax(predictions)
        else:
            return np.max(predictions)

    def update_Q(self, s, a, r, s_):
        actual = r + self.gamma * self.max_Q([s_])
        self.models[a].partial_fit(s, actual)

    def chose_action(self, s):
        if random.random() < self.e:
            return self.env.action_space.sample()
        else:
            return self.max_Q(s, return_index=True)

    def train(self, episodes=10000, print_freq=1000, max_iter=10000):

        cumulative_rewards = 0
        for i_episode in range(episodes):
            s = self.env.reset()
            episode_reward = 0
            self.e = self.epsilon_update(i_episode, self.e)

            for t in range(max_iter):

                a = self.chose_action([s])

                s_, r, done, info = self.env.step(a)
                episode_reward += r

                # if done:
                #     r = -300

                self.update_Q(s, a, r, s_)

                s = s_

                if done:
                    cumulative_rewards += episode_reward
                    break

            if i_episode % print_freq == 0:
                print("Episode {} - av reward: {} | epsilon: {}".format(i_episode, cumulative_rewards / print_freq,
                                                                        self.e))
                cumulative_rewards = 0


def run_cart_pole():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    actions = [0, 1]

    X = env.observation_space.shape[0]
    H = 500
    K = len(actions)
    lr = 0.01

    softmax_activation = lambda h: h
    models = [DeepNerualNetwork(dims=[X, H, 1],
                              activations=[torch.tanh, softmax_activation],
                              #dropouts=[0.3 for _ in range(1)],
                              lr=lr) for _ in range(len(actions))]

    epsilon_update = lambda i, e: 1 / np.sqrt((i/10) + 1)

    agent = NeuralNetworkNetworkAgent(env, actions, models, epsilon_update, e=1)

    agent.train(episodes=10000, print_freq=100, max_iter=2000)


if __name__ == "__main__":
    run_cart_pole()
