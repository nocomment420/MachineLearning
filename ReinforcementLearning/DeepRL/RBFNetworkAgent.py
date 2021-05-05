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
from DeepRL.NeuralNetrokQAgent import DeepNerualNetwork


class MountainCarTransformer:

    def __init__(self, env, num_rbfs=4, H=100, n_samples=10000):
        self.env = env
        self.prepare_feature_vector(n=num_rbfs, H=H, n_samples=n_samples)

    def get_features(self, s):
        return self.feature_generator.transform(s)

    def prepare_feature_vector(self, n, H, n_samples):
        rbfs = []
        samples = [self.env.observation_space.sample() for _ in range(n_samples)]
        scaler = StandardScaler()
        samples = scaler.fit_transform(samples)
        for i in range(n):
            r = RBFSampler(n_components=H, gamma=0.8 * (1 + n))
            r.fit(samples)
            rbf = Pipeline(steps=[["scale", scaler], ["rbf", r]])
            rbfs.append(rbf)

        self.feature_generator = FeatureUnion([["rbf-{}".format(i), rbf] for (i, rbf) in enumerate(rbfs)])


class PoleCartTransformer:

    def __init__(self, env, num_rbfs=4, H=100, n_samples=10000):
        self.env = env
        self.prepare_feature_vector(n=num_rbfs, H=H, n_samples=n_samples)

    def get_features(self, s):
        return self.feature_generator.transform(s)

    def prepare_feature_vector(self, n, H, n_samples):
        rbfs = []
        samples = np.random.random((n_samples, 4)) * 2 - 2

        scaler = StandardScaler()
        samples = scaler.fit_transform(samples)
        for i in range(n):
            r = RBFSampler(n_components=H, gamma=0.8 * (1 + n))
            r.fit(samples)
            rbf = Pipeline(steps=[["scale", scaler], ["rbf", r]])
            rbfs.append(rbf)

        self.feature_generator = FeatureUnion([["rbf-{}".format(i), rbf] for (i, rbf) in enumerate(rbfs)])


class LinearModel:
    def __init__(self, H, lr=0.01):
        W_init = np.random.random((H, 1)) / np.sqrt(H)
        self.W = torch.tensor(W_init, requires_grad=True)

        b_init = np.zeros(1)
        self.b = torch.tensor(b_init, requires_grad=True)

        self.lr = lr

    def forwards(self, X):
        x = torch.from_numpy(X)
        return torch.matmul(x, self.W) + self.b

    def predict(self, X):
        return self.forwards(X).detach().numpy()

    def criterion(self, prediction, target):
        return torch.sum((torch.tensor(target) - prediction) ** 2)

    def zero_grads(self):
        if self.W.grad is not None:
            self.W.grad.data.zero()
        if self.b.grad is not None:
            self.b.grad.data.zero()

    def partial_fit(self, X, Y):
        self.zero_grads()

        prediction = self.forwards(X)
        mse = torch.mean((prediction - torch.tensor(Y)) ** 2)

        mse.backward()

        with torch.no_grad():
            self.W = self.W - self.lr * self.W.grad
            self.b = self.b - self.lr * self.b.grad
        self.W.requires_grad = True
        self.b.requires_grad = True


class LinearModelElegibility:
    def __init__(self, H, lr=0.01):
        W_init = np.random.random((H, 1)) / np.sqrt(H)
        self.W = torch.tensor(W_init)  # , requires_grad=True)

        b_init = np.zeros(1)
        self.b = torch.tensor(b_init)  # , requires_grad=True)

        self.lr = lr

    def forwards(self, X):
        x = torch.from_numpy(X)
        return torch.matmul(x, self.W)  # + self.b

    def predict(self, X):
        return self.forwards(X).detach().numpy()

    def criterion(self, prediction, target):
        return torch.sum((torch.tensor(target) - prediction) ** 2)

    def zero_grads(self):
        if self.W.grad is not None:
            self.W.grad.data.zero()
        if self.b.grad is not None:
            self.b.grad.data.zero()

    def partial_fit(self, X, Y, elege):
        self.zero_grads()

        prediction = self.forwards(X)
        self.W += torch.transpose(torch.mul(self.lr * (prediction - torch.tensor(Y)), torch.tensor(elege)), 0, 1)
        # mse = torch.mean(((prediction - torch.tensor(Y)) ** 2) * torch.tensor(elege))
        #
        # mse.backward()
        #
        # with torch.no_grad():
        #     self.W = self.W - self.lr * self.W.grad
        #     self.b = self.b - self.lr * self.b.grad
        # self.W.requires_grad = True
        # self.b.requires_grad = True


class RBFNetworkAgent:
    def __init__(self, env, actions, models, transformer, epsilon_update, gamma=0.99, e=0.0):
        self.env = env
        self.epsilon_update = epsilon_update
        self.transformer = transformer
        self.actions = actions
        self.gamma = gamma
        self.e = e
        self.models = models

    def get_features(self, s):
        return self.transformer.feature_generator.transform(s)

    def max_Q(self, s, return_index=False):
        if return_index:
            return np.argmax(np.array([self.Q(s, a) for a in self.actions]))
        else:
            return max([self.Q(s, a) for a in self.actions])

    def Q(self, s, a):
        features = self.get_features(s)
        return self.models[a].predict(features)[0]

    def update_Q(self, s, a, r, s_):
        actual = r + self.gamma * self.max_Q([s_])

        pred_features = self.get_features([s])
        self.models[a].partial_fit(pred_features, [actual])

    def prepare_feature_vector(self, n, H, n_samples):
        rbfs = []
        samples = [self.env.observation_space.sample() for _ in range(n_samples)]
        scaler = StandardScaler()
        samples = scaler.fit_transform(samples)
        for i in range(n):
            r = RBFSampler(n_components=H, gamma=0.8 * (1 + n))
            r.fit(samples)
            rbf = Pipeline(steps=[["scale", scaler], ["rbf", r]])
            rbfs.append(rbf)

        self.feature_generator = FeatureUnion([["rbf-{}".format(i), rbf] for (i, rbf) in enumerate(rbfs)])

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

                self.update_Q(s, a, r, s_)

                s = s_

                if done:
                    cumulative_rewards += episode_reward
                    break

            if i_episode % print_freq == 0:
                print("Episode {} - av reward: {} | epsilon: {}".format(i_episode, cumulative_rewards / print_freq,
                                                                        self.e))
                cumulative_rewards = 0

    def display_episode(self, n=20):

        for i in range(n):
            self.env.render()
            s = self.env.reset()
            for t in range(1000):

                a = self.chose_action([s])

                s, r, done, info = self.env.step(a)

                if done:
                    break

    def predict(self, s):
        X = self.get_features([s])
        return np.stack([m.predict(X) for m in self.models]).T

    def plot_cost_to_go(self, dims, N=20):
        X = np.linspace(dims[0][0], dims[0][1], N)
        Y = np.linspace(dims[0][0], dims[0][1], N)
        X, Y = np.meshgrid(X, Y)
        Z = np.apply_along_axis(lambda _: -np.max(self.predict(_)), 2, np.dstack([X, Y]))
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, vmin=-1.0, vmax=1.0)
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_zlabel("Value")
        ax.set_title("Cost-To-Go")
        fig.colorbar(surf)
        plt.show()


class RBFNetworkAgentLambda:
    def __init__(self, env, actions, models, transformer, epsilon_update, gamma=0.99, e=0.0, lambda_=0.5):
        self.env = env
        self.epsilon_update = epsilon_update
        self.transformer = transformer
        self.actions = actions
        self.gamma = gamma
        self.e = e
        self.models = models
        self.lambda_ = lambda_
        self.elegibility = np.zeros((3, 400))

    def get_features(self, s):
        return self.transformer.feature_generator.transform(s)

    def max_Q(self, s, return_index=False):
        if return_index:
            return np.argmax(np.array([self.Q(s, a) for a in self.actions]))
        else:
            return max([self.Q(s, a) for a in self.actions])

    def Q(self, s, a):
        features = self.get_features(s)
        return self.models[a].predict(features)[0][0]

    def update_Q(self, s, a, r, s_):
        actual = r + self.gamma * self.max_Q([s_])

        pred_features = self.get_features([s])

        self.elegibility *= self.lambda_ * self.gamma
        self.elegibility[a] += pred_features[0]

        self.models[a].partial_fit(pred_features, [actual], self.elegibility[a])

    def prepare_feature_vector(self, n, H, n_samples):
        rbfs = []
        samples = [self.env.observation_space.sample() for _ in range(n_samples)]
        scaler = StandardScaler()
        samples = scaler.fit_transform(samples)
        for i in range(n):
            r = RBFSampler(n_components=H, gamma=0.8 * (1 + n))
            r.fit(samples)
            rbf = Pipeline(steps=[["scale", scaler], ["rbf", r]])
            rbfs.append(rbf)

        self.feature_generator = FeatureUnion([["rbf-{}".format(i), rbf] for (i, rbf) in enumerate(rbfs)])

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

                self.update_Q(s, a, r, s_)

                s = s_

                if done:
                    cumulative_rewards += episode_reward
                    break

            if i_episode % print_freq == 0:
                print("Episode {} - av reward: {} | epsilon: {}".format(i_episode, cumulative_rewards / print_freq,
                                                                        self.e))
                cumulative_rewards = 0

    def display_episode(self, n=20):

        for i in range(n):
            self.env.render()
            s = self.env.reset()
            for t in range(1000):

                a = self.chose_action([s])

                s, r, done, info = self.env.step(a)

                if done:
                    break

    def predict(self, s):
        X = self.get_features([s])
        return np.stack([m.predict(X) for m in self.models]).T

    def plot_cost_to_go(self, dims, N=20):
        X = np.linspace(dims[0][0], dims[0][1], N)
        Y = np.linspace(dims[0][0], dims[0][1], N)
        X, Y = np.meshgrid(X, Y)
        Z = np.apply_along_axis(lambda _: -np.max(self.predict(_)), 2, np.dstack([X, Y]))
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, vmin=-1.0, vmax=1.0)
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_zlabel("Value")
        ax.set_title("Cost-To-Go")
        fig.colorbar(surf)
        plt.show()


def run_cart_pole():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    actions = [0, 1]

    transformer = PoleCartTransformer(env)

    # models = [LinearModel(400, lr=0.01) for _ in range(len(actions))]
    # models = [LinearModelElegibility(400, lr=0.01) for _ in range(len(actions))]
    no_activation = lambda x: x
    models = [DeepNerualNetwork(dims=[400, 1], activations=[no_activation], lr=0.01) for _ in range(len(actions))]

    epsilon_update = lambda i, e: 1 / np.sqrt(i + 1)
    agent = RBFNetworkAgent(env, actions, models, transformer, epsilon_update, e=1)
    # agent = RBFNetworkAgentLambda(env, actions, models, transformer, epsilon_update, e=1)

    agent.train(episodes=5000, print_freq=100, max_iter=2000)


def run_mountain_car():
    env_name = "MountainCar-v0"
    env = gym.make(env_name)
    actions = [0, 1, 2]

    transformer = MountainCarTransformer(env)

    models = [LinearModel(400) for _ in range(len(actions))]
    # models = [SGDRegressor(learning_rate="constant") for _ in range(len(actions))]
    # features = transformer.get_features([env.reset()])
    # [models[a].partial_fit(features, np.zeros(1)) for a in actions]
    epsilon_update = lambda i, e: 0.1 * 0.97 ** i

    agent = RBFNetworkAgent(env, actions, models, transformer, epsilon_update, e=0.1)

    agent.train(episodes=500, print_freq=10)
    # agent.display_episode()
    agent.plot_cost_to_go([[-1.2, 0.6], [-0.07, 0.07]])


if __name__ == '__main__':
    run_mountain_car()
    # run_cart_pole()
