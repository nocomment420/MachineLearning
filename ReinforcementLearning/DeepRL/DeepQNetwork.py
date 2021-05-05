import gym
import numpy as np
import random
import torch
from sklearn.utils import shuffle


class ValueModel:
    def __init__(self, layers, lr):
        self.activations = []
        self.layers = []
        self.params = []
        self.lr = lr

        for (dim_in, dim_out, activation, bias) in layers:
            layer = torch.nn.Linear(dim_in, dim_out, bias=bias).to(torch.device(device))
            self.params += layer.parameters()
            self.layers.append(layer)
            self.activations.append(activation)

        assert len(self.activations) == len(self.layers)

        self.criterion = torch.nn.MSELoss().to(torch.device(device))
        self.optim = torch.optim.Adam(self.params, lr=lr)

    def forwards(self, X):
        h = torch.tensor(X, dtype=torch.float32).to(torch.device(device))
        for (layer, activation) in zip(self.layers, self.activations):
            h = layer(h)
            if activation is not None:
                h = activation(h)

        return h

    def predict(self, X):
        return self.forwards(X).cpu().detach().numpy()

    def partial_fit(self, X, Y, i):
        self.optim.zero_grad()

        prediction = self.forwards(X)
        prediction = prediction[torch.arange(len(prediction)), i]
        target = torch.tensor(Y, dtype=torch.float32).to(torch.device(device))
        # loss = self.criterion(prediction, target)
        loss = torch.sum(torch.square(target - prediction))
        loss.backward()

        self.optim.step()

    def copy(self, copy):
        for (i, (me_layer, copy_layer)) in enumerate(zip(self.layers, copy.layers)):
            for (j, (me_param, copy_param)) in enumerate(zip(me_layer.parameters(), copy_layer.parameters())):
                me_param.data = copy_param.data.clone()


class Agent:
    def __init__(self, env, value, gamma=0.99, max_buffer_size=10000, min_buffer_size=100, batch_size=32):
        self.env = env
        self.gamma = gamma
        self.V = value
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.n_state_buffer = []
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size

    def chose_action(self, s, e):
        if random.random() < e:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.V.predict([s]))

    def add_to_buffer(self, s, r, a, s_, done):
        if len(self.state_buffer) > self.max_buffer_size:
            self.state_buffer.pop(0)
            self.action_buffer.pop(0)
            self.reward_buffer.pop(0)
            self.n_state_buffer.pop(0)
            self.done_buffer.pop(0)

        self.state_buffer.append(s)
        self.action_buffer.append(a)
        self.reward_buffer.append(r)
        self.n_state_buffer.append(s_)
        self.done_buffer.append(done)

    def train(self, target_network):
        if len(self.state_buffer) < self.min_buffer_size:
            return

        indexes = np.random.choice(len(self.state_buffer), size=self.batch_size, replace=False)

        states = [self.state_buffer[i] for i in indexes]
        actions = [self.action_buffer[i] for i in indexes]
        rewards = [self.reward_buffer[i] for i in indexes]
        n_states = [self.n_state_buffer[i] for i in indexes]
        dones = [self.done_buffer[i] for i in indexes]
        next_Q = np.max(target_network.predict(n_states), axis=1)
        targets = [r + self.gamma * next_q if not done else r for (r, next_q, done) in zip(rewards, next_Q, dones)]

        self.V.partial_fit(states, targets, actions)

    def play_one(self, e, target_network, max_iter=2000, update_freq=50, train=True,done_mod=True):
        episode_reward = 0

        s = self.env.reset()

        for t in range(max_iter):
            a = self.chose_action(s, e)

            s_, r, done, info = self.env.step(a)
            episode_reward += r

            if done and done_mod:
                r = -200

            self.add_to_buffer(s, r, a, s_, done)

            if train:
                self.train(target_network)
                if t % update_freq == 0:
                    target_network.copy(self.V)
            s = s_

            if done and t > max_iter - 2:
                return episode_reward


def train(agent, target_network, episodes=1000, max_iter=1000, pritn_freq=1,done_mod=True):
    av_reward = 0

    for i in range(episodes):
        eps = 1.0 / np.sqrt(1 + (i))  # /10))

        episode_reward = agent.play_one(target_network=target_network, e=eps, max_iter=max_iter, train=i % 4 == 0, done_mod=done_mod)

        av_reward += episode_reward

        # display message
        if i % pritn_freq == 0:
            print("Episode {} - Av reward : {} | e:{}".format(i + 1, av_reward / pritn_freq, eps))
            av_reward = 0


def run_discrete_cartpole():
    print('Cartpole Discerete Action space')

    env_name = "CartPole-v0"
    env = gym.make(env_name)

    actions = [0, 1]
    X = env.observation_space.shape[0]
    H = 200
    K = len(actions)
    lr = 0.01

    v_layers = [[X, H, torch.tanh, True],
                [H, H, torch.tanh, True],
                [H, K, None, True]]
    V = ValueModel(v_layers, lr=lr)
    target_net = ValueModel(v_layers, lr=lr)
    agent = Agent(env, V)

    train(agent, target_net, episodes=100000, pritn_freq=4, max_iter=2000)

def run_discrete_mountain_climb():
    print('Mountain Car Discerete Action space')

    env_name = "MountainCar-v0"
    env = gym.make(env_name)

    actions = [0, 1, 2]
    X = env.observation_space.shape[0]
    H = 500
    K = len(actions)
    lr = 0.01

    v_layers = [[X, H, torch.tanh, True],
                [H, H, torch.tanh, True],
                [H, K, None, True]]
    V = ValueModel(v_layers, lr=lr)
    target_net = ValueModel(v_layers, lr=lr)
    agent = Agent(env, V)

    train(agent, target_net, episodes=100000, pritn_freq=4, max_iter=10000, done_mod=False)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # run_discrete_cartpole()
    run_discrete_mountain_climb()