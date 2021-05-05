from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import random
import gym
import numpy as np
import torch


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
        self.W = torch.tensor(W_init, requires_grad=True, device=torch.device(device))

        b_init = np.zeros(1)
        self.b = torch.tensor(b_init, requires_grad=True, device=torch.device(device))

        self.optim = torch.optim.Adam([self.b, self.W], lr=lr)
        self.lr = lr

    def forwards(self, X):
        x = torch.tensor(X).to(torch.device(device))
        return torch.matmul(x, self.W) + self.b

    def predict(self, X):
        return self.forwards(X).cpu().detach().numpy()

    def criterion(self, prediction, target):
        return torch.sum((torch.tensor(target).to(torch.device(device)) - prediction) ** 2)

    def partial_fit(self, X, Y):
        self.optim.zero_grad()

        prediction = self.forwards(X)
        target = torch.tensor(Y).to(torch.device(device))
        mse = torch.mean(torch.square(prediction - target))

        mse.backward()
        self.optim.step()


class LinearModelElegibility:
    def __init__(self, H, lr=0.01):
        W_init = np.random.random((H, 1)) / np.sqrt(H)
        self.W = torch.tensor(W_init)

        b_init = np.zeros(1)
        self.b = torch.tensor(b_init)

        self.lr = lr

    def forwards(self, X):
        x = torch.from_numpy(X)
        return torch.matmul(x, self.W)

    def predict(self, X):
        return self.forwards(X).detach().numpy()

    def partial_fit(self, X, Y, elege):
        prediction = self.forwards(X)
        self.W += torch.transpose(torch.mul(self.lr * (prediction - torch.tensor(Y)), torch.tensor(elege)), 0, 1)


class RBFNetworkAgent:
    def __init__(self, env, actions, models, transformer, epsilon_update, gamma=0.99, e=0.0, lambda_=None, H=None):
        self.env = env
        self.epsilon_update = epsilon_update
        self.transformer = transformer
        self.actions = actions
        self.gamma = gamma
        self.e = e
        self.models = models
        self.lambda_ = lambda_
        if lambda_ is not None:
            self.elegibility = np.zeros((len(actions), H))

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

        if self.lambda_ is None:
            self.models[a].partial_fit(pred_features, [actual])
        else:
            self.elegibility *= self.lambda_ * self.gamma
            self.elegibility[a] += pred_features[0]

            self.models[a].partial_fit(pred_features, [actual], self.elegibility[a])

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


def run_cart_pole(lambda_=None):
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    actions = [0, 1]
    H = 400
    lr = 0.01

    transformer = PoleCartTransformer(env)
    epsilon_update = lambda i, e: 1 / np.sqrt(i + 1)

    if lambda_ is None:
        models = [LinearModel(H, lr=lr) for _ in range(len(actions))]
    else:
        models = [LinearModelElegibility(H, lr=lr) for _ in range(len(actions))]

    agent = RBFNetworkAgent(env, actions, models, transformer, epsilon_update, e=1, lambda_=lambda_, H=H)
    agent.train(episodes=5000, print_freq=100, max_iter=2000)


def run_mountain_car(lambda_=None):
    env_name = "MountainCar-v0"
    env = gym.make(env_name)
    actions = [0, 1, 2]
    H = 400
    lr = 0.01

    transformer = MountainCarTransformer(env)

    # epsilon_update = lambda i, e: 1 * 0.97 ** i
    epsilon_update = lambda i, e: 1 / np.sqrt((i/) + 1)

    if lambda_ is None:
        models = [LinearModel(H, lr=lr) for _ in range(len(actions))]

    else:
        models = [LinearModelElegibility(H, lr=lr) for _ in range(len(actions))]

    agent = RBFNetworkAgent(env, actions, models, transformer, epsilon_update, e=1, lambda_=lambda_, H=H)
    agent.train(episodes=500, print_freq=10)
    agent.plot_cost_to_go([[-1.2, 0.6], [-0.07, 0.07]])
    agent.display_episode(10)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    run_mountain_car()
    # run_cart_pole()
